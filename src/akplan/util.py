"""Dataclasses, input parsing, and preprocessing helpers for the scheduling problem.

This module is the **data layer** of the package.  It has three responsibilities:

1. **Schema mirror** — frozen dataclasses (``AKData``, ``ParticipantData``,
   ``RoomData``, ``TimeSlotData``, ``ScheduleAtom``, ``ConfigData``,
   ``SchedulingInput``) that map 1-to-1 onto the JSON input/output format
   documented in the project wiki.

2. **Input parsing** — ``SchedulingInput.from_dict()`` deserialises a raw
   ``dict`` (from ``json.load``) into the dataclass hierarchy using
   ``dacite.from_dict``.

3. **MILP preprocessing** — ``ProblemIds`` and ``ProblemProperties`` translate
   the human-readable input into the numerical arrays (``pandas.Index``,
   ``xarray.DataArray``) that ``solve.create_lp`` needs to build constraints
   efficiently.  All expensive preprocessing is done here once, before the
   solver loop.

Relationship to other modules
------------------------------
- ``solve`` imports ``ProblemIds``, ``ProblemProperties``, ``ScheduleAtom``,
  ``SchedulingInput``, ``SolverConfig`` and the helper functions from here.
- ``tests`` imports the individual dataclasses to build lightweight fixtures.
- ``types`` is imported for the ID aliases and ``ExportTuple``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Literal, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from dacite import from_dict

from akplan import types

logger = logging.getLogger(__name__)


# Solvers that support the "direct" IO API, i.e. they can receive the model
# object in memory without writing an LP/MPS file to disk first.
# Other solvers fall back to the "lp" file-based interface.
solvers_supporting_direct_api = ["gurobi", "highs", "mosek"]


# ---------------------------------------------------------------------------
# Input dataclasses  (mirror the JSON schema)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class AKData:
    """All input data for a single AK (Arbeitskreis — working-group session).

    AKs are the items being scheduled.  Each AK needs a contiguous block of
    ``duration`` timeslots, exactly one room, and optionally a set of
    participants to attend.

    For a specification of the input format, see
    https://github.com/Die-KoMa/ak-plan-optimierung/wiki/Input-&-output-format

    Args:
        id: Unique integer identifier for this AK.
        duration: Number of *consecutive* timeslots the AK requires.
            Must be ≥ 1.  A duration of 2 means the AK occupies two
            back-to-back slots within the same day-block.
        properties: Flexible dict for relational constraints.  The solver
            reads two keys:
            - ``"conflicts"``  — list of AK IDs that must NOT overlap in time.
            - ``"dependencies"`` — list of AK IDs that must *finish* before
              this AK starts (implies no overlap as well).
        room_constraints: Labels (strings) that the assigned room must fulfill.
            E.g. ``["Beamer"]`` means the AK needs a projector.
        time_constraints: Labels that every timeslot of the AK must fulfill.
            E.g. ``["ResoAK"]`` restricts the AK to a specific day.
        info: Free-form metadata (name, organiser, description, …).
            Not used by the optimiser.
    """

    id: types.AkId
    duration: int
    properties: dict[str, Any]
    room_constraints: list[str]
    time_constraints: list[str]
    info: dict[str, Any]


@dataclass(frozen=True, order=True)
class PreferenceData:
    """A single participant's preference for one AK.

    Preferences drive the objective function.  The solver tries to maximise
    the total weighted preference score across all (participant, AK) pairs.

    For a specification of the input format, see
    https://github.com/Die-KoMa/ak-plan-optimierung/wiki/Input-&-output-format

    Args:
        ak_id: The AK this preference refers to.
        preference_score: Raw input score:
            - ``0`` — not interested (excluded from objective).
            - ``1`` — weak interest (weight 1 in objective).
            - ``2`` — strong interest (weight μ in objective, default μ=2).
            - ``-1`` — required (hard constraint, weight 0 in objective).
              Use together with ``required=True``.
        required: If ``True`` this participant *must* attend the AK regardless
            of room or time.  Hard constraint; takes priority over the
            preference score.
    """

    ak_id: types.AkId
    preference_score: int
    required: bool


@dataclass(frozen=True, order=True)
class ParticipantData:
    """All input data for a single conference participant.

    Participants express preferences over AKs.  The schedule is optimised to
    maximise satisfaction of those preferences while honouring hard constraints
    (availability, accessibility).

    For a specification of the input format, see
    https://github.com/Die-KoMa/ak-plan-optimierung/wiki/Input-&-output-format

    Args:
        id: Unique integer identifier.
        preferences: List of ``PreferenceData`` entries.  AKs not mentioned
            here are treated as having zero preference (the participant will
            not be assigned to them).
        room_constraints: Accessibility requirements; the participant may only
            attend AKs in rooms that fulfill all listed labels.
        time_constraints: Availability constraints; the participant may only
            be scheduled in timeslots that fulfill all listed labels.
        info: Free-form metadata (e.g. name).  Not used by the optimiser.
    """

    id: types.PersonId
    preferences: list[PreferenceData]
    room_constraints: list[str]
    time_constraints: list[str]
    info: dict[str, Any]


@dataclass(frozen=True, order=True)
class RoomData:
    """All input data for a physical room.

    Rooms provide features (``fulfilled_room_constraints``) and have an
    availability window (``time_constraints``).  The solver matches AK
    requirements against room features.

    For a specification of the input format, see
    https://github.com/Die-KoMa/ak-plan-optimierung/wiki/Input-&-output-format

    Args:
        id: Unique integer identifier.
        capacity: Maximum number of attendees.  Use ``-1`` for unlimited
            capacity; ``process_room_cap`` normalises this to the total
            participant count so the MILP constraint stays finite.
        fulfilled_room_constraints: Labels that this room provides.
            E.g. ``["Beamer", "Whiteboard"]``.
        time_constraints: Labels that every timeslot must fulfill for this
            room to be usable (e.g. a room might only be available on certain
            days).
        info: Free-form metadata (e.g. name).  Not used by the optimiser.
    """

    id: types.RoomId
    capacity: int
    fulfilled_room_constraints: list[str]
    time_constraints: list[str]
    info: dict[str, Any]


@dataclass(frozen=True, order=True)
class TimeSlotData:
    """All input data for a single timeslot.

    Timeslots are grouped into *blocks* (one block = one day).  An AK must
    occupy consecutive slots within a single block.

    For a specification of the input format, see
    https://github.com/Die-KoMa/ak-plan-optimierung/wiki/Input-&-output-format

    Args:
        id: Unique integer identifier.  IDs are globally unique across all
            blocks, and ordered chronologically.
        fulfilled_time_constraints: Labels that this slot provides.
            E.g. ``["Dienstag", "ResoAK"]`` on day-1 slots.  AKs and
            participants requiring a label will only be placed in slots that
            fulfill it.
        info: Free-form metadata (e.g. human-readable start time).
            Not used by the optimiser.
    """

    id: types.TimeslotId
    fulfilled_time_constraints: list[str]
    info: dict[str, Any]


@dataclass(frozen=False)
class ScheduleAtom:
    """One AK's complete scheduling assignment.

    Used in two roles:
    - **Input** (``SchedulingInput.scheduled_aks``): pre-fixed assignments that
      the solver must respect.
    - **Output** (returned by ``export_scheduling_result``): the solver's
      assignment for every AK after optimisation.

    Args:
        ak_id: The AK being described.
        room_id: The assigned room.  ``None`` when a pre-fixed assignment has
            no room yet (room will be chosen by the solver) or when
            ``allow_unscheduled_aks=True`` and the AK was not placed.
        timeslot_ids: NumPy array of timeslot IDs used by this AK.
            Length must equal ``AKData.duration``; slots are consecutive and
            within the same block.
        participant_ids: NumPy array of participant IDs attending this AK.
    """

    ak_id: types.AkId
    room_id: types.RoomId | None
    timeslot_ids: npt.NDArray[np.int64]
    participant_ids: npt.NDArray[np.int64]

    @property
    def _comparison_tuple(self) -> types.ScheduleAtomComparisonTuple:
        """Canonical sortable / hashable form of this atom.

        Sorts both arrays in-place before packing into a tuple so that two
        atoms with the same content but different array orderings compare equal.
        """
        self.timeslot_ids.sort()
        self.participant_ids.sort()
        return (
            self.ak_id,
            self.room_id,
            tuple(self.timeslot_ids),
            tuple(self.participant_ids),
        )

    def __lt__(self, other: object) -> bool:
        """Determine lesser than by comparing the ``_comparison_tuple``."""
        if not isinstance(other, ScheduleAtom):
            return NotImplemented
        return self._comparison_tuple < other._comparison_tuple

    def __eq__(self, other: object) -> bool:
        """Determine equality by comparing the ``_comparison_tuple``."""
        if not isinstance(other, ScheduleAtom):
            return NotImplemented
        return self._comparison_tuple == other._comparison_tuple

    def __hash__(self) -> int:
        """Calculate hash by hashing the ``_comparison_tuple``."""
        return hash(self._comparison_tuple)

    def stripped_copy(
        self,
        strip_room: bool = False,
        strip_timeslots: bool = False,
        strip_participants: bool = False,
    ) -> ScheduleAtom:
        """Return a copy with selected fields zeroed out.

        Used by ``calc_changed_fixed_schedule_atoms`` to perform
        field-selective comparisons.  For example, when
        ``allow_changing_rooms=True`` the room field is stripped before
        comparing input fixings to solver output, so a room change alone does
        not count as a violated fixing.

        Args:
            strip_room: Replace ``room_id`` with ``None``.
            strip_timeslots: Replace ``timeslot_ids`` with an empty array.
            strip_participants: Replace ``participant_ids`` with an empty array.

        Returns:
            A new ``ScheduleAtom`` with the requested fields cleared.
        """
        # sort before copy to avoid double effort
        self.timeslot_ids.sort()
        self.participant_ids.sort()
        return ScheduleAtom(
            self.ak_id,
            None if strip_room else self.room_id,
            (
                np.array([], dtype=np.int64)
                if strip_timeslots
                else self.timeslot_ids.copy()
            ),
            (
                np.array([], dtype=np.int64)
                if strip_participants
                else self.participant_ids.copy()
            ),
        )


@dataclass(frozen=False)
class ConfigData:
    """Hyperparameters and behavioural flags for the MILP solver.

    All fields have sensible defaults so the ``"config"`` key is optional in
    the input JSON.

    Args:
        mu: Weight assigned to a *strong* preference (``preference_score=2``).
            A weak preference (score=1) always has weight 1.  Increasing μ
            shifts the optimiser towards satisfying fewer but strongly-wanted
            sessions over many weakly-wanted ones.  Default: ``2``.
        max_num_timeslots_before_break: Maximum number of consecutive timeslots
            any participant may be scheduled without a gap in a single block.
            ``0`` disables the break constraint entirely (default).
        allow_unscheduled_aks: If ``True``, the solver may leave AKs without a
            room/time assignment when the schedule is too crowded.  If
            ``False``, the model is infeasible when not all AKs fit.
            Default: ``True``.
        allow_changing_rooms: If ``True``, the solver is allowed to reassign
            the room of a pre-fixed AK (``scheduled_aks``).  Default: ``False``.
    """

    mu: float = 2
    max_num_timeslots_before_break: int = 0
    allow_unscheduled_aks: bool = True
    allow_changing_rooms: bool = False


@dataclass(frozen=True)
class SchedulingInput:
    """Complete input to the scheduling optimisation problem.

    This is the top-level container created by ``from_dict`` and consumed by
    ``solve.create_lp``.  It is intentionally **frozen** (immutable) so that
    neither the solver nor the tests can accidentally mutate it — the one
    exception is ``config.mu``, which tests vary across parameterised runs.

    For a specification of the input format, see
    https://github.com/Die-KoMa/ak-plan-optimierung/wiki/Input-&-output-format

    Args:
        aks: The AKs to schedule, sorted by id.
        participants: The participants, sorted by id.
        rooms: The available rooms, sorted by id.
        timeslot_info: Free-form metadata about the timeslot grid (e.g. slot
            duration in hours).  Not used by the optimiser.
        timeslot_blocks: Outer list = blocks (days); inner list = timeslots
            within that block, in chronological order.
        scheduled_aks: Pre-fixed AK assignments the solver must respect.
            May be empty.
        config: Hyperparameters and flags.  Defaults to ``ConfigData()`` if
            the ``"config"`` key is absent in the JSON.
        info: Free-form metadata about the conference.  Not used by the
            optimiser.
    """

    aks: list[AKData]
    participants: list[ParticipantData]
    rooms: list[RoomData]
    timeslot_info: dict[str, str]
    timeslot_blocks: list[list[TimeSlotData]]
    scheduled_aks: list[ScheduleAtom]
    config: ConfigData
    info: dict[str, str]

    @classmethod
    def from_dict(cls, input_dict: dict[str, Any]) -> SchedulingInput:
        """Deserialise a raw JSON dict into a ``SchedulingInput``.

        Uses ``dacite.from_dict`` for each nested object so that type
        annotations are enforced automatically.  All lists are sorted by ``id``
        so that the order of entries in the JSON file does not affect the model.

        Args:
            input_dict: The top-level dict from ``json.load``.

        Returns:
            A fully populated, frozen ``SchedulingInput``.
        """
        aks = sorted(from_dict(data_class=AKData, data=ak) for ak in input_dict["aks"])
        rooms = sorted(
            from_dict(data_class=RoomData, data=room) for room in input_dict["rooms"]
        )
        participants = sorted(
            from_dict(data_class=ParticipantData, data=participant)
            for participant in input_dict["participants"]
        )
        # Each block is sorted independently — block order is chronological,
        # slot order within a block is also chronological.
        timeslot_blocks = [
            sorted(
                from_dict(data_class=TimeSlotData, data=timeslot) for timeslot in block
            )
            for block in input_dict["timeslots"]["blocks"]
        ]
        # Pre-fixed AKs are optional in the JSON.
        scheduled_aks = (
            sorted(
                from_dict(data_class=ScheduleAtom, data=scheduled_ak)
                for scheduled_ak in input_dict["scheduled_aks"]
            )
            if "scheduled_aks" in input_dict
            else []
        )
        # Config is optional; fall back to all-default values.
        config = (
            from_dict(data_class=ConfigData, data=input_dict["config"])
            if "config" in input_dict
            else ConfigData()
        )

        return cls(
            aks=aks,
            participants=participants,
            rooms=rooms,
            timeslot_blocks=timeslot_blocks,
            timeslot_info=input_dict["timeslots"]["info"],
            scheduled_aks=scheduled_aks,
            config=config,
            info=input_dict["info"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise back to a plain dict suitable for ``json.dump``.

        Note: ``scheduled_aks`` and ``config`` are intentionally omitted from
        the output — the output JSON only echoes the *problem definition* (not
        the solution or solver settings).  The solution is stored separately
        under the ``"scheduled_aks"`` key at the top level of the output file.

        Returns:
            A JSON-serialisable dict.
        """
        return_dict = {
            "aks": [asdict(ak) for ak in self.aks],
            "rooms": [asdict(room) for room in self.rooms],
            "participants": [asdict(participant) for participant in self.participants],
            "info": self.info,
        }
        blocks = [
            [asdict(timeslot) for timeslot in block] for block in self.timeslot_blocks
        ]
        return_dict["timeslots"] = {"info": self.timeslot_info, "blocks": blocks}
        return return_dict


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def get_ak_name(input_data: SchedulingInput, ak_id: types.AkId) -> str:
    """Look up the human-readable name of an AK for use in log messages.

    Args:
        input_data: The full scheduling input.
        ak_id: The AK whose name is requested.

    Returns:
        A comma-joined string of all ``info["name"]`` values for AKs with the
        given id (normally exactly one).  Returns an empty string if no match
        or if the ``"name"`` key is absent.
    """
    ak_names = [
        ak.info["name"]
        for ak in input_data.aks
        if ak.id == ak_id and "name" in ak.info.keys()
    ]
    return ", ".join(ak_names)


def process_pref_score(preference_score: int, required: bool, mu: float) -> float:
    """Convert a raw preference score into its MILP objective weight.

    The three-tier weight system (0 / 1 / μ) is central to the objective
    function.  Required participants are treated as hard constraints and
    therefore contribute 0 to the objective (their attendance is guaranteed
    by a lower-bound fix on the ``Part`` variable, not by the objective).

    Args:
        preference_score: Raw score from the JSON:
            0 = not interested, 1 = weak, 2 = strong, −1 = required.
        required: ``True`` if the participant is required for the AK.
            When ``True`` the return value is always 0 regardless of score.
        mu: Weight for strong preferences (score == 2).

    Returns:
        0.0 for required or uninterested participants,
        1.0 for weak preferences,
        ``mu`` for strong preferences.

    Raises:
        ValueError: If ``preference_score`` is not in {−1, 0, 1, 2}.
    """
    if required or preference_score == -1:
        return 0
    elif preference_score in [0, 1]:
        return preference_score
    elif preference_score == 2:
        return mu
    else:
        raise ValueError(preference_score)


def process_room_cap(room_capacity: int, num_participants: int) -> int:
    """Normalise a room capacity for use as a finite MILP upper bound.

    The JSON allows ``-1`` as a sentinel for "unlimited" capacity.  The MILP
    needs a concrete finite number, so unlimited rooms are capped at the total
    participant count (no room can ever be over-occupied by more than the total
    number of attendees).

    Args:
        room_capacity: Raw capacity from the JSON.  ``-1`` means unlimited;
            any non-negative integer is a hard cap.
        num_participants: Total number of participants in the problem instance.

    Returns:
        ``num_participants`` if the room is unlimited or larger than the
        participant pool; otherwise the original ``room_capacity``.

    Raises:
        ValueError: If ``room_capacity < -1``.
    """
    if room_capacity == -1:
        return num_participants
    if room_capacity >= num_participants:
        return num_participants
    if room_capacity < 0:
        raise ValueError(
            f"Invalid room capacity {room_capacity}. "
            "Room capacity must be non-negative or -1."
        )
    return room_capacity


# ---------------------------------------------------------------------------
# MILP preprocessing  (ProblemIds + ProblemProperties)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProblemIds:
    """Pandas Index objects for every entity dimension in the MILP.

    linopy uses ``pd.Index`` objects as coordinate axes for its variables and
    constraints.  Gathering all IDs here once avoids repeated list
    comprehensions inside ``create_lp``.

    Attributes:
        ak: Index of all AK IDs.
        room: Index of all room IDs.
        timeslot: Index of all timeslot IDs (across all blocks, in order).
        person: Index of all participant IDs.
        block: Index of block IDs (0, 1, 2, …).
        block_dict: Maps each block ID to the ``pd.Index`` of timeslot IDs
            that belong to that block.  Used for the contiguity and break
            constraints which operate within a single block.
    """

    ak: pd.Index
    room: pd.Index
    timeslot: pd.Index
    person: pd.Index
    block: pd.Index
    block_dict: dict[types.BlockId, types.Block]

    @staticmethod
    def get_ids(
        input_data: SchedulingInput,
    ) -> tuple[
        list[types.AkId],
        list[types.PersonId],
        list[types.RoomId],
        list[types.TimeslotId],
    ]:
        """Extract flat ID lists from a ``SchedulingInput``.

        Timeslot IDs are flattened across all blocks in block order.

        Args:
            input_data: The parsed scheduling input.

        Returns:
            Four lists: (ak_ids, participant_ids, room_ids, timeslot_ids).
        """

        def _retrieve_ids(
            input_iterable: Iterable[
                AKData | ParticipantData | RoomData | TimeSlotData
            ],
        ) -> list[types.Id]:
            return [obj.id for obj in input_iterable]

        ak_ids = _retrieve_ids(input_data.aks)
        participant_ids = _retrieve_ids(input_data.participants)
        room_ids = _retrieve_ids(input_data.rooms)
        # chain.from_iterable flattens the list-of-lists into a single sequence
        timeslot_ids = _retrieve_ids(chain.from_iterable(input_data.timeslot_blocks))
        return ak_ids, participant_ids, room_ids, timeslot_ids

    @classmethod
    def init_from_problem(
        cls: type[ProblemIds],
        input_data: SchedulingInput,
    ) -> ProblemIds:
        """Build a ``ProblemIds`` from a ``SchedulingInput``.

        Args:
            input_data: The parsed scheduling input.

        Returns:
            A fully populated ``ProblemIds``.
        """
        ak_ids, person_ids, room_ids, timeslot_ids = cls.get_ids(input_data)

        # Build per-block timeslot indices.  Each block gets a named index
        # so that linopy can display meaningful coordinate names in debug output.
        block_dict = {
            block_idx: pd.Index(
                [timeslot.id for timeslot in block], name=f"block-{block_idx}"
            )
            for block_idx, block in enumerate(input_data.timeslot_blocks)
        }

        return cls(
            ak=pd.Index(ak_ids, name="ak"),
            room=pd.Index(room_ids, name="room"),
            timeslot=pd.Index(timeslot_ids, name="timeslot"),
            person=pd.Index(person_ids, name="person"),
            block=pd.Index(block_dict.keys(), name="block"),
            block_dict=block_dict,
        )


@dataclass(frozen=True)
class ProblemProperties:
    """Precomputed numerical arrays derived from a ``SchedulingInput``.

    Building xarray DataArrays from the raw input is relatively expensive and
    is needed multiple times when constructing the many constraint groups.
    ``ProblemProperties`` does this work once and stores the results as named
    fields so ``create_lp`` can reference them by name without recomputing.

    All DataArrays use the same coordinate names as the linopy variables
    (``"ak"``, ``"room"``, ``"timeslot"``, ``"person"``, ``"time_constraint"``,
    ``"room_constraint"``) so they can be broadcast against variables directly.

    Attributes:
        conflict_pairs: Set of ``(ak_a, ak_b)`` pairs (with ``ak_a < ak_b``)
            that must not overlap in time.  Includes both explicit AK conflicts
            *and* dependency pairs (a dependency also implies no overlap).
        dependencies: Maps each AK to the list of AK IDs that must finish
            before it starts.
        time_constraint: Sorted ``pd.Index`` of all time-constraint label
            strings appearing anywhere in the input.
        room_constraint: Sorted ``pd.Index`` of all room-constraint label
            strings appearing anywhere in the input.
        room_capacities: Shape ``(room,)``.  Processed capacity value for each
            room (``-1`` replaced by ``num_participants``).
        ak_durations: Shape ``(ak,)``.  Required duration in timeslots for
            each AK.
        preferences: Shape ``(ak × person)``.  MILP objective weight for each
            (AK, participant) pair: 0, 1, or μ.
        required_persons: Shape ``(ak × person)``.  Boolean — ``True`` where
            a participant is required to attend an AK.
        ak_num_interested: Shape ``(ak,)``.  Count of participants with any
            non-zero preference (weak, strong, or required) for each AK.
            Used for the room-size constraint and participant-count bound.
        block_mask: Shape ``(block × timeslot)``.  ``True`` where timeslot
            ``t`` belongs to block ``b``.  Used in ``AKBlockAssign``.
        participant_time_constraints: Shape ``(person × time_constraint)``.
            ``True`` where a participant requires a given time label.
        participant_room_constraints: Shape ``(person × room_constraint)``.
            ``True`` where a participant requires a given room label.
        ak_time_constraints: Shape ``(ak × time_constraint)``.
        ak_room_constraints: Shape ``(ak × room_constraint)``.
        room_time_constraints: Shape ``(room × time_constraint)``.
            ``True`` where a room is only available during slots with this label.
        fulfilled_time_constraints: Shape ``(timeslot × time_constraint)``.
            ``True`` where a timeslot satisfies a given label.
        fulfilled_room_constraints: Shape ``(room × room_constraint)``.
            ``True`` where a room satisfies a given label.
    """

    conflict_pairs: set[tuple[types.AkId, types.AkId]]
    dependencies: dict[types.AkId, list[types.AkId]]
    time_constraint: pd.Index
    room_constraint: pd.Index
    room_capacities: xr.DataArray
    ak_durations: xr.DataArray
    preferences: xr.DataArray
    required_persons: xr.DataArray
    ak_num_interested: xr.DataArray
    block_mask: xr.DataArray
    participant_time_constraints: xr.DataArray
    participant_room_constraints: xr.DataArray
    ak_time_constraints: xr.DataArray
    ak_room_constraints: xr.DataArray
    room_time_constraints: xr.DataArray
    fulfilled_time_constraints: xr.DataArray
    fulfilled_room_constraints: xr.DataArray

    @classmethod
    def init_from_problem(
        cls: type[ProblemProperties],
        input_data: SchedulingInput,
        ids: ProblemIds | None = None,
    ) -> ProblemProperties:
        """Build a ``ProblemProperties`` from a ``SchedulingInput``.

        This method performs all the O(n) setup work that ``create_lp`` would
        otherwise repeat in multiple places: iterating over participants to
        build preference matrices, iterating over constraint labels, etc.

        Args:
            input_data: The parsed scheduling input.
            ids: Pre-built ``ProblemIds``.  If ``None``, it is constructed
                internally.  Pass an existing one to avoid duplicate work.

        Returns:
            A fully populated, frozen ``ProblemProperties``.
        """
        if ids is None:
            ids = ProblemIds.init_from_problem(input_data)

        # --- Conflict and dependency pairs -----------------------------------
        # Walk every AK's ``properties`` dict once to collect both
        # conflicts and dependencies into a single set of (a, b) pairs.
        # The canonical form always has the smaller ID first to avoid
        # duplicate pairs like (3,7) and (7,3).
        conflict_pairs: set[tuple[types.AkId, types.AkId]] = set()
        dependencies: dict[types.AkId, list[types.AkId]] = {}
        for ak in input_data.aks:
            conflicting_aks: list[types.AkId] = ak.properties.get("conflicts", [])
            depending_aks: list[types.AkId] = ak.properties.get("dependencies", [])
            if depending_aks:
                dependencies[ak.id] = depending_aks
            conflict_pairs.update(
                [
                    (
                        (ak.id, other_ak_id)
                        if ak.id < other_ak_id
                        else (other_ak_id, ak.id)
                    )
                    # dependency pairs also imply no time overlap
                    for other_ak_id in conflicting_aks + depending_aks
                ]
            )

        # --- Scalar properties per room / AK ---------------------------------
        room_capacities = xr.DataArray(
            data=[
                process_room_cap(room.capacity, len(ids.person))
                for room in input_data.rooms
            ],
            coords=[ids.room],
        )
        ak_durations = xr.DataArray(
            data=[ak.duration for ak in input_data.aks],
            coords=[ids.ak],
        )

        # --- Preference matrices ---------------------------------------------
        # Start with zeros; fill in non-zero entries from each participant's
        # preference list.  AKs not mentioned in a participant's list keep 0.
        preferences = xr.DataArray(0.0, coords=[ids.ak, ids.person])
        required_persons = xr.DataArray(False, coords=[ids.ak, ids.person])
        for person in input_data.participants:
            for pref in person.preferences:
                preferences.loc[pref.ak_id, person.id] = process_pref_score(
                    pref.preference_score,
                    pref.required,
                    mu=input_data.config.mu,
                )
                if pref.required:
                    required_persons.loc[pref.ak_id, person.id] = True

        # Warn about AKs with no required person — they lack an "owner" and
        # may be impossible to schedule correctly in practice.
        num_required_per_ak = required_persons.sum("person")
        if (num_required_per_ak == 0).any():
            for ak_id in num_required_per_ak.where(
                num_required_per_ak == 0, drop=True
            ).coords["ak"]:
                logger.warning(
                    "AK %s with id %d has no required persons. Who owns this?",
                    get_ak_name(input_data, ak_id),
                    ak_id.item(),
                )

        # ak_num_interested = required + anyone with a non-zero preference.
        # This is the maximum number of people who could possibly attend an AK
        # and is used as the upper bound on the participant count in Roomsize.
        ak_num_interested = num_required_per_ak + (preferences != 0).sum("person")

        # --- Block membership mask -------------------------------------------
        # block_mask[b, t] = True iff timeslot t belongs to block b.
        # Used in AKBlockAssign: Time[a, t] <= Block[a, b] for all t in b.
        block_mask = xr.DataArray(data=False, coords=[ids.block, ids.timeslot])
        for block_id, block_lst in ids.block_dict.items():
            block_mask.loc[block_id, block_lst] = True

        # --- Constraint label universe ---------------------------------------
        # Collect every constraint string that appears anywhere in the input.
        # This becomes the shared coordinate axis for all the boolean constraint
        # membership arrays below.
        all_time_constraints: set[str] = set()
        all_room_constraints: set[str] = set()
        for participant in input_data.participants:
            all_time_constraints.update(participant.time_constraints)
            all_room_constraints.update(participant.room_constraints)
        for ak in input_data.aks:
            all_time_constraints.update(ak.time_constraints)
            all_room_constraints.update(ak.room_constraints)
        for room in input_data.rooms:
            all_time_constraints.update(room.time_constraints)
            all_room_constraints.update(room.fulfilled_room_constraints)
        for timeslot in chain.from_iterable(input_data.timeslot_blocks):
            all_time_constraints.update(timeslot.fulfilled_time_constraints)

        time_constraint = pd.Index(sorted(all_time_constraints), name="time_constraint")
        room_constraint = pd.Index(sorted(all_room_constraints), name="room_constraint")

        # --- Constraint membership arrays ------------------------------------
        # Each array is a boolean matrix: rows = entity, columns = constraint label.
        # "requires" arrays are True where an entity NEEDS a label.
        # "fulfilled" arrays are True where an entity PROVIDES a label.
        # The solver rule is: requires[e, c] => fulfilled[assigned_entity, c].

        participant_time_constraints = xr.DataArray(
            False, coords=[ids.person, time_constraint]
        )
        participant_room_constraints = xr.DataArray(
            False, coords=[ids.person, room_constraint]
        )
        for person in input_data.participants:
            participant_time_constraints.loc[person.id, person.time_constraints] = True
            participant_room_constraints.loc[person.id, person.room_constraints] = True

        ak_time_constraints = xr.DataArray(False, coords=[ids.ak, time_constraint])
        ak_room_constraints = xr.DataArray(False, coords=[ids.ak, room_constraint])
        for ak in input_data.aks:
            ak_time_constraints.loc[ak.id, ak.time_constraints] = True
            ak_room_constraints.loc[ak.id, ak.room_constraints] = True

        room_time_constraints = xr.DataArray(False, coords=[ids.room, time_constraint])
        fulfilled_room_constraints = xr.DataArray(
            False, coords=[ids.room, room_constraint]
        )
        for room in input_data.rooms:
            room_time_constraints.loc[room.id, room.time_constraints] = True
            fulfilled_room_constraints.loc[room.id, room.fulfilled_room_constraints] = (
                True
            )

        fulfilled_time_constraints = xr.DataArray(
            False, coords=[ids.timeslot, time_constraint]
        )
        for timeslot in chain.from_iterable(input_data.timeslot_blocks):
            fulfilled_time_constraints.loc[
                timeslot.id, timeslot.fulfilled_time_constraints
            ] = True

        return cls(
            conflict_pairs=conflict_pairs,
            dependencies=dependencies,
            room_capacities=room_capacities,
            ak_durations=ak_durations,
            preferences=preferences,
            required_persons=required_persons,
            ak_num_interested=ak_num_interested,
            block_mask=block_mask,
            room_constraint=room_constraint,
            time_constraint=time_constraint,
            participant_time_constraints=participant_time_constraints,
            participant_room_constraints=participant_room_constraints,
            ak_time_constraints=ak_time_constraints,
            ak_room_constraints=ak_room_constraints,
            room_time_constraints=room_time_constraints,
            fulfilled_time_constraints=fulfilled_time_constraints,
            fulfilled_room_constraints=fulfilled_room_constraints,
        )


def _construct_constraint_name(name: str, *args: Any) -> str:
    """Build a unique linopy constraint name from a base string and entity IDs.

    linopy requires each constraint (or constraint group) to have a unique name.
    For pair-wise constraints (e.g. "no two AKs share a room at time t") we
    append the entity IDs to the base name.

    Example: ``_construct_constraint_name("AKConflict", 3, 7)`` → ``"AKConflict_3_7"``

    Args:
        name: Descriptive base name for the constraint group.
        *args: Additional identifiers (typically entity IDs) appended with
            underscores.

    Returns:
        A string of the form ``"<name>_<arg0>_<arg1>_…"``.
    """
    return name + "_" + "_".join(map(str, args))


# ---------------------------------------------------------------------------
# Solver configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SolverConfig:
    """Runtime options passed to the ILP solver.

    Decoupled from ``ConfigData`` (which holds problem hyperparameters) because
    solver settings are infrastructure concerns, not part of the mathematical
    problem definition.

    ``generate_kwargs`` translates the generic fields into the solver-specific
    dict that linopy's ``model.solve()`` expects.

    Attributes:
        solver_io_api: How the model is handed to the solver.
            - ``"direct"`` — in-memory via the solver's Python API (fastest,
              supported by Gurobi, HiGHS, MOSEK).
            - ``"lp"`` / ``"mps"`` — write a file, let the solver read it.
        solver_dir: Directory for linopy's temporary LP/solution files.
            ``None`` uses the system temp directory and auto-deletes after
            solving.
        time_limit: Stop the solver after this many seconds.  Returns the
            best feasible solution found so far.
        gap_abs: Stop when the absolute MIP gap is ≤ this value.
        gap_rel: Stop when the relative MIP gap is ≤ this fraction.
        threads: Parallel solver threads.  Defaults to ``#CPUs − 1`` via
            ``default_num_threads()``.
        warmstart_fn: Path to a basis file for warm-starting (e.g. a ``.bas``
            file from a previous solve of a similar problem).
    """

    solver_io_api: Literal["direct", "lp", "mps"] = "direct"
    solver_dir: str | None = None
    time_limit: float | None = None
    gap_abs: float | None = None
    gap_rel: float | None = None
    threads: int | None = None
    warmstart_fn: str | Path | None = None

    @property
    def _has_non_none_attr(self) -> bool:
        """Return ``True`` if any performance-tuning attribute is set.

        Used to emit a warning when the chosen solver does not support
        translating these settings.
        """
        if (
            self.time_limit is not None
            or self.gap_abs is not None
            or self.gap_rel is not None
            or self.threads is not None
        ):
            return True
        return False

    @overload
    def generate_kwargs(self, solver_name: None) -> types.SolverKwargs: ...

    @overload
    def generate_kwargs(
        self, solver_name: Literal["gurobi"]
    ) -> types.HighsSolverKwargs: ...

    @overload
    def generate_kwargs(
        self, solver_name: Literal["highs"]
    ) -> types.HighsSolverKwargs: ...

    @overload
    def generate_kwargs(self, solver_name: str) -> types.SolverKwargs: ...

    def generate_kwargs(
        self, solver_name: str | None
    ) -> types.GurobiSolverKwargs | types.HighsSolverKwargs | types.SolverKwargs:
        """Translate generic solver settings to solver-specific keyword arguments.

        Different solvers use different parameter names for the same concept
        (e.g. Gurobi uses ``"TimeLimit"`` while HiGHS uses ``"time_limit"``).
        This method produces the correct dict for the given solver so that
        ``model.solve(**generate_kwargs(solver_name))`` works transparently.

        For unsupported solvers a warning is emitted and only ``io_api`` /
        ``warmstart_fn`` are forwarded (performance params are dropped).

        Args:
            solver_name: The linopy solver name string, or ``None``.

        Returns:
            A ``TypedDict`` of keyword arguments ready to be unpacked into
            ``linopy.Model.solve``.
        """
        io_api = self.solver_io_api
        # Fall back to the file-based LP interface for solvers that don't
        # support the in-memory direct API.
        if solver_name not in solvers_supporting_direct_api:
            logger.warning(
                "Using 'direct' IO-API for solver %s is not supported. "
                "Changing IO-API to 'lp' instead.",
                solver_name,
            )
            io_api = "lp"

        if solver_name == "highs":
            highs_solver_kwargs: types.HighsSolverKwargs = {}
            if self.solver_io_api is not None:
                highs_solver_kwargs["io_api"] = io_api
            if self.time_limit is not None:
                highs_solver_kwargs["time_limit"] = self.time_limit
            if self.gap_abs is not None:
                highs_solver_kwargs["mip_abs_gap"] = self.gap_abs
            if self.gap_rel is not None:
                highs_solver_kwargs["mip_rel_gap"] = self.gap_rel
            if self.threads is not None:
                highs_solver_kwargs["threads"] = self.threads
            if self.warmstart_fn is not None:
                highs_solver_kwargs["warmstart_fn"] = self.warmstart_fn
            return highs_solver_kwargs
        elif solver_name == "gurobi":
            gurobi_solver_kwargs: types.GurobiSolverKwargs = {}
            if self.solver_io_api is not None:
                gurobi_solver_kwargs["io_api"] = io_api
            if self.time_limit is not None:
                gurobi_solver_kwargs["TimeLimit"] = self.time_limit
            if self.gap_abs is not None:
                gurobi_solver_kwargs["MIPGapAbs"] = self.gap_abs
            if self.gap_rel is not None:
                gurobi_solver_kwargs["MIPGap"] = self.gap_rel
            if self.threads is not None:
                gurobi_solver_kwargs["Threads"] = self.threads
            if self.warmstart_fn is not None:
                gurobi_solver_kwargs["warmstart_fn"] = self.warmstart_fn
            return gurobi_solver_kwargs
        else:
            if self._has_non_none_attr:
                logger.warning(
                    "Exporting CLI args to solver '%s' is not supported. "
                    "The solver is run with its default parameters.",
                    solver_name,
                )
            solver_kwargs: types.SolverKwargs = {}
            if self.solver_io_api is not None:
                solver_kwargs["io_api"] = io_api
            if self.warmstart_fn is not None:
                solver_kwargs["warmstart_fn"] = self.warmstart_fn
            return solver_kwargs


def default_num_threads() -> int:
    """Return the default number of solver threads: ``max(#available_CPUs − 1, 1)``.

    Leaving one CPU free keeps the machine responsive during a long solve.
    Uses ``os.sched_getaffinity`` on Linux (respects cgroup/container CPU
    quotas) and falls back to ``os.cpu_count`` on macOS and Windows where
    ``sched_getaffinity`` is not available.

    Returns:
        Integer number of threads, always ≥ 1.
    """
    try:
        # gives the number of CPUs that the current process is allowed to use
        # might be unequal to `os.cpu_count` which gives the # of logical CPUs
        n_available_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        # on windows / mac, os.sched_getaffinity is not available.
        n_available_cpus = os.cpu_count() or 0
    return max(n_available_cpus - 1, 1)
