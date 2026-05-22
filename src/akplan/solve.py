"""MILP construction, solving, result extraction, and the ``akplan-solve`` CLI.

This module is the **core engine** of the package.  It has four public
responsibilities:

1. ``create_lp`` — translates a ``SchedulingInput`` into a fully-specified
   linopy ``Model`` (binary variables, all constraints, objective function).

2. ``solve_scheduling`` — wraps ``create_lp`` and hands the model to an ILP
   solver (HiGHS or Gurobi), returning the solution arrays or ``None`` on
   infeasibility.

3. ``export_scheduling_result`` / ``process_solved_lp`` — read the solved
   variable arrays and produce a ``dict[AkId, ScheduleAtom]`` that maps every
   AK to its assigned room, timeslots, and participants.

4. ``main`` — ``argparse``-based CLI entry point registered as ``akplan-solve``
   in ``pyproject.toml``.

MILP overview
-------------
Three families of binary decision variables are created:

- ``Room[ak, room]`` — 1 if AK is in that room.
- ``Time[ak, timeslot]`` — 1 if AK uses that timeslot.
- ``Part[ak, person]`` — 1 if person attends that AK.

Plus two auxiliary variables:

- ``Block[ak, block]`` — 1 if AK is placed in that day-block.
- ``Working[person, timeslot]`` — 1 if person is occupied in that slot.

The objective maximises the sum of preference-weighted ``Part`` values,
normalised per person so that people with many preferences don't dominate.

For the full mathematical specification see:
https://github.com/Die-KoMa/ak-plan-optimierung/wiki/LP-formulation

Relationship to other modules
------------------------------
- Imports all dataclasses and preprocessing helpers from ``util``.
- Imports type aliases and ``ExportTuple`` from ``types``.
- Called by the test suite via ``solve_scheduling`` and ``process_solved_lp``.
"""

import argparse
import json
import logging
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, TypeVar, cast, get_args, overload

import linopy
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from akplan import types
from akplan.util import (
    ProblemIds,
    ProblemProperties,
    ScheduleAtom,
    SchedulingInput,
    SolverConfig,
    _construct_constraint_name,
    default_num_threads,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


def create_lp(
    input_data: SchedulingInput, solver_dir: str | None = None
) -> linopy.Model:
    """Construct the MILP model for a conference scheduling problem.

    Translates a fully-parsed ``SchedulingInput`` into a linopy ``Model``
    with all variables, constraints, and the objective function.  The returned
    model is ready to be handed to a solver via ``model.solve()``.

    The construction proceeds in five phases:
    1. Compute ``ProblemIds`` and ``ProblemProperties`` (all preprocessing).
    2. Determine variable bounds (tighten bounds to prune the search space).
    3. Add variables to the linopy model.
    4. Set the objective function.
    5. Add all hard constraints (feasibility, conflicts, dependencies, …).

    For a specification of the input JSON format, see
    https://github.com/Die-KoMa/ak-plan-optimierung/wiki/Input-&-output-format

    For a specification of the MILP, see
    https://github.com/Die-KoMa/ak-plan-optimierung/wiki/LP-formulation

    The MILP models each person to have three kinds of preferences for an AK:
    0 (no preference), 1 (weak preference) and ``mu`` (strong preference).
    The choice of ``mu`` is a hyperparameter of the MILP that weights the
    balance between weak and strong preferences.

    Args:
        input_data: The parsed and validated scheduling input.
        solver_dir: Directory where linopy stores temporary LP and solution
            files.  ``None`` (default) uses the system temp directory and
            cleans up automatically after solving.

    Returns:
        A linopy ``Model`` ready for solving.
    """
    time_lp_construction_start = perf_counter()

    logger.debug("Start construction of the LP")

    ids = ProblemIds.init_from_problem(input_data)
    props = ProblemProperties.init_from_problem(input_data, ids=ids)

    logger.debug("IDs and Properties initialized")

    # ``force_dim_names=True`` makes linopy raise an error if a variable or
    # constraint is created without named coordinate dimensions, preventing
    # hard-to-debug broadcasting mistakes.
    # TODO: Consider chunking
    m = linopy.Model(force_dim_names=True, solver_dir=solver_dir)

    # -------------------------------------------------------------------------
    # Helper: default lower=0 / upper=1 bounds for a binary variable grid.
    # -------------------------------------------------------------------------
    def _init_lower_upper(coords: list[pd.Index]) -> tuple[xr.DataArray, xr.DataArray]:
        return xr.DataArray(0, coords=coords), xr.DataArray(1, coords=coords)

    # -------------------------------------------------------------------------
    # Variable bounds — tighten before declaring variables so the solver does
    # not waste time exploring provably infeasible assignments.
    # -------------------------------------------------------------------------

    # Part[ak, person]:
    #   lower=1 for (ak, person) pairs where participation is required.
    #   upper=0 for (ak, person) pairs where the person has no preference at all.
    person_lower, person_upper = _init_lower_upper([ids.ak, ids.person])
    # required aks have P_{P,A} = 1 implicitly
    person_lower = person_lower.where(~props.required_persons, 1)
    # aks without preferences have P_{P,A} = 0 implicitly
    person_upper = person_upper.where(
        (props.preferences != 0) | props.required_persons, 0
    )

    # Working[person, timeslot]: upper=0 where the timeslot violates a
    # participant's time constraint (they can never be active then).
    time_impossible_for_person_mask = (
        props.participant_time_constraints & (~props.fulfilled_time_constraints)
    ).any("time_constraint")
    person_time_lower, person_time_upper = _init_lower_upper([ids.person, ids.timeslot])
    person_time_upper = person_time_upper.where(~time_impossible_for_person_mask, 0)

    # Time[ak, timeslot]: upper=0 where the timeslot violates an AK's
    # time constraint (the AK can never be scheduled at that slot).
    time_impossible_for_ak_mask = (
        props.ak_time_constraints & (~props.fulfilled_time_constraints)
    ).any("time_constraint")
    time_lower, time_upper = _init_lower_upper([ids.ak, ids.timeslot])
    time_upper = time_upper.where(~time_impossible_for_ak_mask, 0)

    # Room[ak, room]: upper=0 where the room fails an AK's room constraint.
    room_impossible_for_ak_mask = (
        props.ak_room_constraints & (~props.fulfilled_room_constraints)
    ).any("room_constraint")
    room_lower, room_upper = _init_lower_upper([ids.ak, ids.room])
    room_upper = room_upper.where(~room_impossible_for_ak_mask, 0)

    # -------------------------------------------------------------------------
    # Fix values for pre-scheduled AKs (lower = upper = 1 for their assignments).
    # -------------------------------------------------------------------------
    # Pre-fixed AKs are represented as lower=upper=1 on the relevant variable
    # cells.  Using integer variables (rather than binary) is required here
    # because linopy only supports lower/upper bound fixing on integer variables.
    for scheduled_ak in input_data.scheduled_aks:
        if (
            scheduled_ak.room_id is not None
            and not input_data.config.allow_changing_rooms
        ):
            room_lower.loc[scheduled_ak.ak_id, scheduled_ak.room_id] = 1

        person_lower.loc[scheduled_ak.ak_id, scheduled_ak.participant_ids] = 1
        time_lower.loc[scheduled_ak.ak_id, scheduled_ak.timeslot_ids] = 1

    # -------------------------------------------------------------------------
    # Declare variables
    # -------------------------------------------------------------------------
    # All variables are semantically binary (0/1).  We declare them as
    # ``integer`` rather than ``binary`` because linopy's binary variables do
    # not support custom lower/upper bounds, which we need for fixing
    # pre-scheduled AKs and for pruning impossible assignments above.

    room = m.add_variables(
        name="Room",
        integer=True,
        coords=[ids.ak, ids.room],
        lower=room_lower,
        upper=room_upper,
    )
    time = m.add_variables(
        name="Time",
        integer=True,
        coords=[ids.ak, ids.timeslot],
        lower=time_lower,
        upper=time_upper,
    )
    block = m.add_variables(name="Block", integer=True, coords=[ids.ak, ids.block])
    person = m.add_variables(
        name="Part",
        integer=True,
        coords=[ids.ak, ids.person],
        lower=person_lower,
        upper=person_upper,
    )
    person_time = m.add_variables(
        name="Working",
        integer=True,
        coords=[ids.person, ids.timeslot],
        lower=person_time_lower,
        upper=person_time_upper,
    )
    logger.debug("Variables added")

    # -------------------------------------------------------------------------
    # Objective function
    # -------------------------------------------------------------------------
    # Maximise:  Σ_{P,A}  (pref[P,A] / num_prefs[P]) * Part[P,A]
    #
    # Dividing by ``num_prefs_per_person`` normalises each person's
    # contribution so that a person with 20 preferences does not dominate over
    # a person with 3 preferences.  Required AKs (pref weight = 0) are
    # excluded from the count — they don't contribute to the objective.
    # TODO: Do we want to include 'required' AKs?
    num_prefs_per_person = (props.preferences != 0).sum(
        "ak"
    ) + props.required_persons.sum("ak")
    weighted_prefs = (props.preferences / num_prefs_per_person).where(
        num_prefs_per_person != 0
    )
    m.add_objective((weighted_prefs * person).sum(), sense="max")
    logger.debug("Objective added")

    # -------------------------------------------------------------------------
    # Shared upper-triangular mask for all pair-wise AK constraints.
    # ak_pair_mask[ak1, ak2] = True iff ak1 < ak2 (canonical order: no
    # self-pairs, no duplicate reversed pairs).  Reused for both
    # MaxOneAKPerPersonAndTime and MaxOneAKPerRoomAndTime below.
    # -------------------------------------------------------------------------
    ak_pair_mask = xr.DataArray(
        np.triu(np.ones((len(ids.ak), len(ids.ak)), dtype=bool), k=1),
        coords=[ids.ak.rename("ak1"), ids.ak.rename("ak2")],
    )

    # -------------------------------------------------------------------------
    # Constraint: MaxOneAKPerPersonAndTime
    # For every pair of distinct AKs (a1, a2) and every (person, timeslot):
    #   Time[a1, t] + Part[a1, p] + Time[a2, t] + Part[a2, p] <= 3
    # Equivalent to: a person cannot attend two AKs at the same time.
    # (If both Time and Part are 1 for both AKs, the sum would be 4 > 3.)
    #
    # Vectorised: rename 'ak' → 'ak1'/'ak2' on two copies of the expression
    # so xarray broadcasts to shape (ak1 × ak2 × timeslot × person); the
    # upper-triangular mask restricts to canonical (ak1 < ak2) pairs only.
    # -------------------------------------------------------------------------
    c = time + person  # shape: (ak, timeslot, person)
    m.add_constraints(
        c.rename({"ak": "ak1"}) + c.rename({"ak": "ak2"}) <= 3,
        mask=ak_pair_mask,
        name="MaxOneAKPerPersonAndTime",
    )
    logger.debug("Constraints MaxOneAKPerPersonAndTime added")

    # -------------------------------------------------------------------------
    # Constraint: MaxOneAKPerRoomAndTime
    # For every pair of distinct AKs (a1, a2) and every (room, timeslot):
    #   Time[a1, t] + Room[a1, r] + Time[a2, t] + Room[a2, r] <= 3
    # Equivalent to: two AKs cannot use the same room at the same time.
    # -------------------------------------------------------------------------
    c = time + room  # shape: (ak, timeslot, room)
    m.add_constraints(
        c.rename({"ak": "ak1"}) + c.rename({"ak": "ak2"}) <= 3,
        mask=ak_pair_mask,
        name="MaxOneAKPerRoomAndTime",
    )
    logger.debug("Constraints MaxOneAKPerRoomAndTime added")

    # -------------------------------------------------------------------------
    # Constraint: AKDuration
    #   Σ_t Time[a, t] >= duration[a]   for all a
    # Each AK must be assigned at least its required number of timeslots.
    # (Combined with AKContiguous, this becomes an equality in practice.)
    # -------------------------------------------------------------------------
    m.add_constraints((time.sum("timeslot") >= props.ak_durations), name="AKDuration")
    logger.debug("Constraints AKDuration added")

    # -------------------------------------------------------------------------
    # Constraint: AKSingleBlock
    #   Σ_b Block[a, b] <= 1   for all a
    # An AK may span at most one day-block.
    # -------------------------------------------------------------------------
    m.add_constraints((block.sum("block") <= 1), name="AKSingleBlock")
    logger.debug("Constraints AKSingleBlock added")

    # -------------------------------------------------------------------------
    # Constraint: AKBlockAssign
    #   Time[a, t] <= Block[a, b]   for all (a, b, t) where t in block b
    # If an AK uses timeslot t, it must be assigned to the block that contains t.
    # The ``.where(props.block_mask)`` filters to only the (b, t) pairs where
    # t actually belongs to b — the mask has shape (block × timeslot).
    # -------------------------------------------------------------------------
    m.add_constraints((time - block).where(props.block_mask) <= 0, name="AKBlockAssign")
    logger.debug("Constraints AKBlockAssign added")

    # -------------------------------------------------------------------------
    # Constraint: Roomsize
    #   Part[a, :].sum() + num_interested[a] * Room[a, r]
    #       <= num_interested[a] + capacity[r]
    # Applied only where num_interested[a] > capacity[r] (i.e. the room could
    # actually be over-full).  Rearranged: attendees <= capacity when Room[a,r]=1.
    # -------------------------------------------------------------------------
    m.add_constraints(
        lhs=person.sum("person") + props.ak_num_interested * room,
        sign="<=",
        rhs=props.ak_num_interested + props.room_capacities,
        mask=props.ak_num_interested > props.room_capacities,
        name="Roomsize",
    )
    logger.debug("Constraints Roomsize added")

    # -------------------------------------------------------------------------
    # Constraints: AtMostOneRoomPerAK / AtLeastOneRoomPerAK / RoomForAK
    #   Σ_r Room[a, r] == 1   for all a
    # Every AK gets exactly one room.
    # -------------------------------------------------------------------------
    m.add_constraints(room.sum("room") <= 1, name="AtMostOneRoomPerAK")
    logger.debug("Constraints AtMostOneRoomPerAK added")
    m.add_constraints(room.sum("room") >= 1, name="AtLeastOneRoomPerAK")
    logger.debug("Constraints AtLeastOneRoomPerAK added")

    # -------------------------------------------------------------------------
    # Constraint: NotMorePeopleThanInterested
    #   Part[a, :].sum() <= num_interested[a]   for all a
    # The number of assigned attendees cannot exceed the number of people who
    # expressed any interest (this also indirectly caps the objective).
    # -------------------------------------------------------------------------
    m.add_constraints(
        person.sum("person") <= props.ak_num_interested,
        name="NotMorePeopleThanInterested",
    )
    logger.debug("Constraints NotMorePeopleThanInterested added")

    # -------------------------------------------------------------------------
    # Constraint: TimePersonVar  (linking constraint)
    #   Time[a, t] + Part[p, a] - Working[p, t] <= 1   for all (a, p, t)
    # Forces Working[p, t] = 1 whenever person p attends AK a AND AK a is
    # in timeslot t.  (Working is then used by MaxOneAKPerPersonAndTime and
    # the BreakForPerson constraint.)
    # -------------------------------------------------------------------------
    m.add_constraints(time + person - person_time <= 1, name="TimePersonVar")
    logger.debug("Constraints TimePersonVar added")

    # -------------------------------------------------------------------------
    # Constraint: RoomForAK (duplicate of AtLeastOneRoomPerAK — kept for clarity)
    # -------------------------------------------------------------------------
    m.add_constraints(room.sum("room") >= 1, name="RoomForAK")
    logger.debug("Constraints RoomForAK added")

    # -------------------------------------------------------------------------
    # Constraint: RoomImpossibleForPerson
    #   Room[a, r] + Part[p, a] <= 1
    # Applied only where room r does NOT fulfill one of person p's room
    # constraints.  Prevents assigning a person to an AK in an inaccessible room.
    # -------------------------------------------------------------------------
    room_impossible_for_person_mask = (
        props.participant_room_constraints & (~props.fulfilled_room_constraints)
    ).any("room_constraint")
    m.add_constraints(
        room + person <= 1,
        name="RoomImpossibleForPerson",
        mask=room_impossible_for_person_mask,
    )
    logger.debug("Constraints RoomImpossibleForPerson added")

    # -------------------------------------------------------------------------
    # Constraint: TimeImpossibleForRoom
    #   Room[a, r] + Time[a, t] <= 1
    # Applied only where room r is NOT available during timeslot t.
    # -------------------------------------------------------------------------
    time_impossible_for_room_mask = (
        props.room_time_constraints & (~props.fulfilled_time_constraints)
    ).any("time_constraint")
    m.add_constraints(
        room + time <= 1,
        name="TimeImpossibleForRoom",
        mask=time_impossible_for_room_mask,
    )
    logger.debug("Constraints TimeImpossibleForRoom added")

    # -------------------------------------------------------------------------
    # Constraint: AKConflict
    #   Time[a1, t] + Time[a2, t] <= 1   for all t, for all (a1, a2) in conflicts
    # Conflicting AK pairs must not overlap in time.  This also covers
    # dependency pairs (a dependency implies no overlap).
    # -------------------------------------------------------------------------
    for ak_a, ak_b in props.conflict_pairs:
        m.add_constraints(
            time.loc[ak_a] + time.loc[ak_b] <= 1,
            name=_construct_constraint_name("AKConflict", ak_a, ak_b),
        )
    logger.debug("Constraints AKConflict added")

    # -------------------------------------------------------------------------
    # Constraint: AKContiguous (vectorised)
    # Within a block, an AK of duration d must use d *consecutive* timeslots.
    # For each AK and each pair of timeslots (ta, tb) where ta and tb are in
    # the same block and tb is at least duration[ak] positions ahead of ta:
    #   Time[ak, ta] + Time[ak, tb] <= 1
    # This forbids any two timeslot assignments that are too far apart to form
    # a contiguous run of the required length.
    #
    # Build a boolean (ak × timeslot_a × timeslot_b) mask in numpy, then call
    # add_constraints once instead of looping over all (ak, block, ta, tb)
    # triples.
    # -------------------------------------------------------------------------
    ts_list = list(ids.timeslot)
    n_ts = len(ts_list)
    ts_idx_map = {int(ts_id): i for i, ts_id in enumerate(ts_list)}

    # For every timeslot: record its block index and its position within that block.
    ts_block_arr = np.empty(n_ts, dtype=int)
    ts_pos_arr = np.empty(n_ts, dtype=int)
    for block_id, block_lst in ids.block_dict.items():
        for pos, ts_id in enumerate(block_lst):
            i = ts_idx_map[int(ts_id)]
            ts_block_arr[i] = int(block_id)
            ts_pos_arr[i] = pos

    # same_block[i, j] = True iff timeslots i and j belong to the same block.
    same_block = ts_block_arr[:, None] == ts_block_arr[None, :]  # (n_ts, n_ts)

    # pos_diff[i, j] = position(j) − position(i) within their block.
    pos_diff = ts_pos_arr[None, :] - ts_pos_arr[:, None]  # (n_ts, n_ts)

    # ak_dur[k] = required duration for AK at index k.
    ak_dur = np.array([props.ak_durations.loc[ak_id].item() for ak_id in ids.ak])

    # ak_contiguous_mask[ak, ta, tb]:
    #   True iff ta and tb are in the same block AND tb ≥ ta + duration[ak].
    ak_contiguous_mask = xr.DataArray(
        same_block[None, :, :] & (pos_diff[None, :, :] >= ak_dur[:, None, None]),
        coords=[
            ids.ak,
            pd.Index(ts_list, name="timeslot_a"),
            pd.Index(ts_list, name="timeslot_b"),
        ],
    )

    m.add_constraints(
        time.rename({"timeslot": "timeslot_a"}) + time.rename({"timeslot": "timeslot_b"})
        <= 1,
        mask=ak_contiguous_mask,
        name="AKContiguous",
    )
    logger.debug("Constraints AKContiguous added")

    # -------------------------------------------------------------------------
    # Constraint: PersonNeedsBreak  (optional, disabled when limit == 0)
    # Within each block, no person may be scheduled for more than
    # `max_num_timeslots_before_break` consecutive timeslots.
    # For each window of (limit + 1) consecutive slots:
    #   Working[p, window].sum() <= limit
    # TODO: vectorize
    # -------------------------------------------------------------------------
    if input_data.config.max_num_timeslots_before_break > 0:
        for block_entry in ids.block_dict.values():
            for idx in range(
                len(block_entry) - input_data.config.max_num_timeslots_before_break - 1
            ):
                # sliding window of width (limit + 1) over the block
                block_subset = block_entry[
                    idx : idx + input_data.config.max_num_timeslots_before_break + 1
                ]
                m.add_constraints(
                    lhs=person_time.loc[:, block_subset].sum("timeslot"),
                    sign="<=",
                    rhs=input_data.config.max_num_timeslots_before_break,
                    name=_construct_constraint_name("BreakForPerson", block_entry[idx]),
                )
    logger.debug("Constraints BreakForPerson added")

    # -------------------------------------------------------------------------
    # Constraint: AKDependenciesDoneBeforeAK
    # If AK `a` depends on AK `b`, then every timeslot used by `a` must come
    # after every timeslot used by `b`.
    #
    # For each timeslot t and each (a, b) dependency pair:
    #   Time[a, t:].sum() - Time[b, t] >= 0
    # Meaning: if b uses slot t, then a must use some slot at or after t.
    # Combined with the AKConflict constraint (which bans simultaneous slots),
    # this forces `b` to finish strictly before `a` starts.
    # TODO: vectorize
    # -------------------------------------------------------------------------
    for ak_id in ids.ak:
        if ak_id not in props.dependencies:
            continue
        other_ak_ids = props.dependencies[ak_id]
        for idx, timeslot_id in enumerate(ids.timeslot):
            m.add_constraints(
                lhs=time.loc[ak_id, ids.timeslot[idx:]].sum("timeslot")
                - time.loc[other_ak_ids, timeslot_id],
                sign=">=",
                rhs=0,
                name=_construct_constraint_name(
                    "AKDependenciesDoneBeforeAK", ak_id, timeslot_id
                ),
            )
    logger.debug("Constraints AKDependenciesDoneBeforeAK added")

    time_lp_construction_end = perf_counter()
    logger.info(
        "LP constructed. Time elapsed: %.1fs",
        time_lp_construction_end - time_lp_construction_start,
    )
    return m


def export_scheduling_result(
    input_data: SchedulingInput,
    solution: types.ExportTuple,
    allow_unscheduled_aks: bool = False,
) -> dict[types.AkId, ScheduleAtom]:
    """Extract a human-readable schedule from the solved MILP variable arrays.

    Reads the three solution DataArrays (Room, Time, Part) and for each AK
    finds which room, timeslots, and participants were assigned (value == 1).

    For a specification of the output format, see
    https://github.com/Die-KoMa/ak-plan-optimierung/wiki/Input-&-output-format

    Args:
        input_data: The scheduling input (used to get the list of AK IDs).
        solution: Named tuple with the rounded binary solution arrays.
        allow_unscheduled_aks: If ``True``, AKs with no assigned room or
            timeslots are represented as ``ScheduleAtom(room_id=None, …)``
            rather than raising an error.  Controlled by
            ``ConfigData.allow_unscheduled_aks``.

    Returns:
        Dict mapping each AK ID to its ``ScheduleAtom`` assignment.

    Raises:
        ValueError: If an AK is assigned to multiple rooms (should never
            happen in a valid solution) or has no room when
            ``allow_unscheduled_aks=False``.
    """
    ids = ProblemIds.init_from_problem(input_data)

    @overload
    def _get_id(
        ak_id: types.AkId,
        var_key: str,
        allow_multiple: Literal[True],
        allow_none: bool,
        coord: str | None = None,
    ) -> npt.NDArray[np.int64]: ...

    @overload
    def _get_id(
        ak_id: types.AkId,
        var_key: str,
        allow_multiple: Literal[False],
        allow_none: bool,
        coord: str | None = None,
    ) -> types.Id | None: ...

    def _get_id(
        ak_id: types.AkId,
        var_key: str,
        allow_multiple: bool,
        allow_none: bool,
        coord: str | None = None,
    ) -> Any:
        """Extract the IDs where a solution variable equals 1 for a given AK.

        Args:
            ak_id: The AK to inspect.
            var_key: One of ``"room"``, ``"time"``, or ``"person"`` —
                selects which ``ExportTuple`` field to read.
            allow_multiple: If ``True`` return all matching IDs as an array;
                if ``False`` expect exactly 0 or 1 match.
            allow_none: If ``True`` a zero-match result is allowed (returns
                ``None`` or empty array); if ``False`` it raises ``ValueError``.
            coord: Coordinate name to extract IDs from.  Defaults to
                ``var_key`` (works for ``"room"`` and ``"person"``).
                Pass ``"timeslot"`` for the ``"time"`` variable because the
                coordinate name in linopy is ``"timeslot"`` not ``"time"``.
        """
        if coord is None:
            coord = var_key
        # Slice the solution DataArray to the row for this AK, then keep only
        # entries where the value is positive (i.e. == 1 after rounding).
        ak_row = getattr(solution, var_key).loc[ak_id]
        matched_ids = ak_row.where(ak_row > 0, drop=True).coords[coord]
        if not allow_multiple and matched_ids.size > 1:
            raise ValueError(f"AK {ak_id} is assigned multiple {var_key}")
        elif matched_ids.size == 0 and not allow_none:
            raise ValueError(f"no {var_key} assigned to ak {ak_id}")
        else:
            if allow_multiple:
                return matched_ids.data.tolist()
            else:
                return matched_ids.item() if matched_ids.size > 0 else None

    scheduled_aks: dict[types.AkId, ScheduleAtom] = {
        ak_id: ScheduleAtom(
            ak_id=ak_id,
            room_id=_get_id(
                ak_id=ak_id,
                var_key="room",
                allow_multiple=False,
                allow_none=allow_unscheduled_aks,
            ),
            timeslot_ids=_get_id(
                ak_id=ak_id,
                var_key="time",
                coord="timeslot",  # linopy coordinate name differs from var_key
                allow_multiple=True,
                allow_none=allow_unscheduled_aks,
            ),
            participant_ids=_get_id(
                ak_id=ak_id, var_key="person", allow_multiple=True, allow_none=True
            ),
        )
        for ak_id in ids.ak
    }

    return scheduled_aks


def solve_scheduling(
    input_data: SchedulingInput,
    solver_config: SolverConfig,
    solver_name: str | None = None,
) -> tuple[linopy.Model, types.ExportTuple] | None:
    """Build and solve the MILP scheduling problem.

    Orchestrates ``create_lp`` followed by ``model.solve``.  Solver selection
    falls back gracefully: prefers Gurobi, then HiGHS, then any available
    linopy solver.

    For a specification of the input format, see
    https://github.com/Die-KoMa/ak-plan-optimierung/wiki/Input-&-output-format

    For a specification of the ILP used, see
    https://github.com/Die-KoMa/ak-plan-optimierung/wiki/New-LP-formulation

    The ILP models each person to have three kinds of preferences for an AK:
    0 (no preference), 1 (weak preference) and ``mu`` (strong preference).
    The choice of ``mu`` is a hyperparameter of the ILP that weights the
    balance between weak and strong preferences.

    Args:
        input_data: The parsed scheduling input.
        solver_config: Runtime settings (time limit, threads, gap tolerance…).
        solver_name: The linopy solver name to use.  ``None`` (default) selects
            automatically from installed solvers in preference order.

    Returns:
        ``(model, solution)`` tuple on success, where ``solution`` is an
        ``ExportTuple`` of rounded binary DataArrays.
        ``None`` if the model is infeasible.

    Raises:
        ValueError: If no linopy-compatible solver is installed at all.
    """
    if not linopy.available_solvers:
        raise ValueError(
            "No linopy solvers available! "
            "Consider installing any solver of "
            f"{get_args(types.SupportedSolver)}."
        )

    if solver_name is None:
        # Walk the preferred solver list and take the first installed one.
        for solver_candidate in get_args(types.SupportedSolver):
            if solver_candidate in linopy.available_solvers:
                solver_name = cast(str, solver_candidate)
                break
        else:
            # Fall back to whatever linopy found, but warn that performance
            # tuning arguments may not be forwarded correctly.
            solver_name = linopy.available_solvers[0]
            logger.warning(
                "No supported solver available. "
                "Solver %s will be used with default config values.",
                solver_name,
            )

    model = create_lp(input_data, solver_dir=solver_config.solver_dir)

    status, term_cond = model.solve(
        keep_files=solver_config.solver_dir is not None,
        solver_name=solver_name,
        **solver_config.generate_kwargs(solver_name),
    )

    logger.info("Termination Condition: %s", term_cond)
    logger.info("Solution status: %s", status)

    if term_cond == "infeasible":
        # Gurobi and Xpress can compute the IIS (Irreducible Infeasible Subsystem)
        # to pinpoint the minimal set of conflicting constraints.
        # HiGHS does not support IIS computation (linopy limitation).
        if model.solver_name in ("gurobi", "xpress"):
            logger.warning(
                "Infeasible model. Conflicting constraints (IIS):\n%s",
                model.format_infeasibilities(),
            )
        else:
            logger.warning(
                "Infeasible model. To compute the IIS and identify conflicting "
                "constraints, re-run with '--solver gurobi' or '--solver xpress'."
            )
        return None

    # Round solution values to {0, 1} to clean up floating-point noise from
    # the solver (values like 0.9999… or 0.0001… should be exact integers).
    solution = types.ExportTuple(
        room=model.variables["Room"].solution.round(),
        time=model.variables["Time"].solution.round(),
        person=model.variables["Part"].solution.round(),
    )
    return (model, solution)


def process_solved_lp(
    model: linopy.Model,
    solution: types.ExportTuple,
    input_data: SchedulingInput,
) -> dict[types.AkId, ScheduleAtom] | None:
    """Convert a solved linopy model into a ``dict[AkId, ScheduleAtom]``.

    Thin wrapper around ``export_scheduling_result`` that first checks the
    model status and respects the ``allow_unscheduled_aks`` config flag.

    Args:
        model: The linopy model after ``model.solve()`` has been called.
        solution: The rounded binary solution arrays from ``solve_scheduling``.
        input_data: The original scheduling input.

    Returns:
        A dict mapping every AK ID to its ``ScheduleAtom``, or ``None`` if
        the model's status is not ``"ok"`` (e.g. time-limit with no feasible
        solution found).
    """
    if model.status != "ok":
        return None

    # TODO: Test if a check for partial solutions is necessary

    return export_scheduling_result(
        input_data,
        solution,
        allow_unscheduled_aks=input_data.config.allow_unscheduled_aks,
    )


def calc_changed_fixed_schedule_atoms(
    input_atoms: Iterable[ScheduleAtom],
    schedule_atoms: Iterable[ScheduleAtom],
    ignore_room_change: bool = False,
    ignore_timeslots_change: bool = False,
    ignore_participants_change: bool = True,
) -> list[ScheduleAtom]:
    """Find pre-fixed AK assignments that were not preserved in the solver output.

    After solving, we verify that every AK in ``input_data.scheduled_aks``
    (the pre-fixed set) still appears unchanged in the output schedule.  If
    ``allow_changing_rooms`` is ``True`` we ignore room differences; participant
    changes are always ignored (the solver may optimise attendance even for
    fixed AKs).

    Args:
        input_atoms: The pre-fixed ``ScheduleAtom`` entries from the input.
        schedule_atoms: The ``ScheduleAtom`` entries produced by the solver.
        ignore_room_change: If ``True``, two atoms that differ only in
            ``room_id`` are considered equal.
        ignore_timeslots_change: If ``True``, timeslot differences are ignored.
        ignore_participants_change: If ``True`` (default), participant-list
            differences are ignored.

    Returns:
        Sorted list of input atoms that are NOT present in the output
        (after applying the ignore flags).  An empty list means all fixings
        were respected.
    """

    def _stripped_atom_set(atom_it: Iterable[ScheduleAtom]) -> set[ScheduleAtom]:
        """Convert atoms to a set, clearing the ignored fields first."""
        return {
            atom.stripped_copy(
                strip_room=ignore_room_change,
                strip_participants=ignore_participants_change,
                strip_timeslots=ignore_timeslots_change,
            )
            for atom in atom_it
        }

    input_data_atom_set = _stripped_atom_set(input_atoms)
    schedule_atom_set = _stripped_atom_set(schedule_atoms)
    # Set difference: atoms that were in the input but not in the output.
    changed_schedule_set = input_data_atom_set - schedule_atom_set

    return sorted(changed_schedule_set)


def main() -> None:
    """CLI entry point for ``akplan-solve``.

    Parses command-line arguments, reads a JSON input file, runs the solver,
    checks that pre-fixed AKs were respected, and writes the result to a JSON
    output file.

    The output file is named ``out-<input-filename>`` by default and is placed
    in the current working directory.  Use ``--output`` to override.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver",
        type=str,
        default=None,
        help=(
            "The solver to use. We currently only support passing CLI args to solvers "
            f"in {get_args(types.SupportedSolver)}. If None, chooses a default "
            "from installed solvers. Defaults to None."
        ),
    )
    parser.add_argument(
        "--solver-dir",
        type=str,
        default=None,
        help=(
            "Path where linopy's temporary files like the lp file "
            "or the intermediate solution file should be stored. "
            "The default None results in taking the default temporary directory "
            " and an automatic removal after the solving is done."
        ),
    )
    parser.add_argument(
        "--solver-io-api",
        type=str,
        choices=["direct", "lp", "mps"],
        default="direct",
        help=(
            "API to use for communicating with the solver, must be one of "
            "{'lp', 'mps', 'direct'}. If set to 'lp'/'mps' the problem is written to "
            "an LP/MPS file which is then read by the solver. If set to "
            "'direct' the problem is communicated to the solver via the solver "
            "specific API, e.g. gurobipy. This may lead to faster run times. "
            "Defaults to 'direct'."
        ),
    )
    parser.add_argument(
        "--solver-warmstart-fn",
        type=str,
        default=None,
        help=(
            "Optional path of the basis file which should be used to "
            "warmstart the solving."
        ),
    )
    parser.add_argument(
        "--timelimit",
        type=float,
        default=None,
        help="Timelimit as stopping criterion (in seconds)",
    )
    parser.add_argument(
        "--gap-rel", type=float, default=None, help="Relative gap as stopping criterion"
    )
    parser.add_argument(
        "--gap-abs", type=float, default=None, help="Absolute gap as stopping criterion"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of threads to use. Defaults to #CPUs minus 1.",
    )
    parser.add_argument(
        "--loglevel",
        type=str.lower,
        choices=["error", "warning", "info", "debug"],
        default="info",
        help="Select logging level. Defaults to 'info'.",
    )
    parser.add_argument(
        "path", type=str, help="Path of the JSON input file to the solver."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Json File to output the calculated schedule to. If not specified, "
        "the prefix 'out-' is added to the input file name and it is stored in the "
        "current working directory.",
    )
    parser.add_argument(
        "--override-output",
        action="store_true",
        help="If set, overrides the output file if it exists.",
    )
    args = parser.parse_args()

    # Configure the root logger before any other logging calls.
    numeric_loglevel = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_loglevel, int):
        raise ValueError(f"Invalid log level: {args.loglevel}")
    logging.basicConfig(
        level=numeric_loglevel,
        format="[%(levelname)s] %(asctime)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.threads is None:
        # default threads to number of available CPUs minus 1
        args.threads = default_num_threads()

    # Gurobi's Python bindings create their own logger that duplicates every
    # message already emitted by linopy.  Detach it from the root logger to
    # avoid doubled output.
    # See: https://linopy.readthedocs.io/en/latest/gurobi-double-logging.html
    gurobi_logger = logging.getLogger("gurobipy")
    gurobi_logger.propagate = False

    solver_config = SolverConfig(
        solver_dir=args.solver_dir,
        solver_io_api=args.solver_io_api,
        warmstart_fn=args.solver_warmstart_fn,
        time_limit=args.timelimit,
        gap_rel=args.gap_rel,
        gap_abs=args.gap_abs,
        threads=args.threads,
    )

    json_file = Path(args.path)
    assert json_file.suffix == ".json"
    with json_file.open("r") as f:
        input_dict = json.load(f)

    # Default output path: prepend "out-" to the input filename, place in CWD.
    if args.output is None:
        args.output = Path.cwd() / f"out-{json_file.name}"

    if args.output.exists() and not args.override_output:
        raise ValueError(
            f"Output file {args.output} already exists. We do not simply override it."
        )

    # Ensure the output directory exists (supports nested paths via --output).
    args.output.parent.mkdir(exist_ok=True, parents=True)

    scheduling_input = SchedulingInput.from_dict(input_dict)

    solution_tuple = solve_scheduling(
        scheduling_input,
        solver_config,
        args.solver,
    )

    if solution_tuple is None:
        # Infeasible — solver already printed diagnostics; nothing to write.
        return

    schedule = process_solved_lp(*solution_tuple, input_data=scheduling_input)

    if schedule is None:
        # Model status was not "ok" (e.g. no feasible solution within time limit).
        return

    # -------------------------------------------------------------------------
    # Integrity check: verify that every pre-fixed AK assignment was honoured.
    # Emit a warning (not an error) so the user can investigate without losing
    # the (possibly useful) partial result.
    # -------------------------------------------------------------------------
    changed_fixed_schedule_atoms = calc_changed_fixed_schedule_atoms(
        scheduling_input.scheduled_aks,
        schedule.values(),
        ignore_room_change=scheduling_input.config.allow_changing_rooms,
    )

    if changed_fixed_schedule_atoms:
        string_repr = [
            (
                f"\t(AK {atom.ak_id}, "
                f"Room {atom.room_id}, "
                f"Timeslots {sorted(atom.timeslot_ids)})"
            )
            for atom in changed_fixed_schedule_atoms
        ]

        logger.warning(
            "Some fixed scheduling was NOT respected in the output! "
            "The following entries of the input are affected:\n%s",
            "\n".join(string_repr),
        )

    # Write output: the schedule plus an echo of the input for traceability.
    out_dict = {
        "scheduled_aks": list(map(asdict, schedule.values())),
        "input": scheduling_input.to_dict(),
    }
    with args.output.open("w") as ff:
        json.dump(out_dict, ff)
    logger.info("Stored result at %s", args.output)


if __name__ == "__main__":
    main()
