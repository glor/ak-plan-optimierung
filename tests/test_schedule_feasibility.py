"""Property-based feasibility tests for the akplan scheduling solver.

Strategy
--------
For every example JSON file in ``examples/``, we:

1. Load the file into a ``SchedulingInput`` (``scheduling_input`` fixture).
2. Solve the MILP with a combination of ``mu`` value and solver name
   (``solved_lp_fixture``).
3. Extract the schedule (``scheduled_aks`` fixture).
4. Run a suite of assertion functions, each checking one scheduling constraint.

Because solving can be expensive (several seconds per instance), all fixtures
are scoped to ``"module"`` so that a single solve is shared across every test
function for the same ``(input_file, mu, solver)`` combination.

Pytest marks
------------
- ``slow``     — large/hard instances; skipped in the ``fast-test`` nox session.
- ``extensive`` — even larger instances; only run in ``extensive-test``.
- ``licensed`` — requires a Gurobi licence; skipped in CI.

These marks are declared in ``pytest.ini`` and applied to individual
parameter sets in the ``scheduling_params`` list below.

Relationship to other modules
------------------------------
Imports ``solve_scheduling`` and ``process_solved_lp`` from ``solve``, and
all input dataclasses from ``util``.
"""

import json
from itertools import product
from pathlib import Path
from typing import TypeVar, cast

import linopy
import linopy.solvers
import numpy as np
import numpy.typing as npt
import pytest
from _pytest.mark import ParameterSet

from akplan import types
from akplan.solve import process_solved_lp, solve_scheduling
from akplan.util import (
    AKData,
    ParticipantData,
    RoomData,
    ScheduleAtom,
    SchedulingInput,
    SolverConfig,
    TimeSlotData,
    default_num_threads,
)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _test_uniqueness(
    lst: list[T],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.intp], bool]:
    """Check that every element in ``lst`` appears exactly once.

    Used to verify that (room, timeslot) pairs and (person, timeslot) pairs
    are unique across the whole schedule.

    Args:
        lst: A list of hashable items (typically tuples of integer IDs).

    Returns:
        A 3-tuple ``(unique_values, counts, all_unique)`` where ``all_unique``
        is ``True`` iff every element appeared exactly once.
    """
    arr = np.asarray(lst, dtype=np.int64)
    unique_vals, cnts = np.unique(arr, axis=0, return_counts=True)
    return unique_vals, cnts, not bool(np.abs(cnts - 1).sum())


# ---------------------------------------------------------------------------
# Fixtures: input loading
# ---------------------------------------------------------------------------


@pytest.fixture(
    scope="module",
    params=[
        # Fast instances — always included in the default test run.
        "examples/test_20a_20p_3r_5rc_0.25rc-lam_0.json",
        "examples/test_20a_40p_4r_5rc_0.25rc-lam_0.json",
        "examples/test_20a_100p_4r_5rc_0.25rc-lam_0.json",
        "examples/test_10a_15p_4r_5rc_0.25rc-lam_0.json",
        "examples/test_20a_20p_5r_5rc_0.25rc-lam_3confl_3dep_0.json",
        "examples/test_20a_20p_5r_5rc_0.25rc-lam_5confl_5dep_0.json",
        "examples/test_20a_20p_5r_5rc_0.25rc-lam_10confl_0.json",
        "examples/test_20a_20p_5r_5rc_0.25rc-lam_10dep_0.json",
        "examples/test1.json",
        # Slow instances — skipped in fast-test, run in test and extensive-test.
        pytest.param(
            "examples/test_30a_20p_3r_5rc_0.25rc-lam_0.json", marks=pytest.mark.slow
        ),
        pytest.param(
            "examples/test_40a_10p_4r_5rc_0.25rc-lam_0.json", marks=pytest.mark.slow
        ),
        pytest.param(
            "examples/test_40a_70p_4r_10rc_1.00rc-lam_0.json", marks=pytest.mark.slow
        ),
        pytest.param("examples/test2.json", marks=pytest.mark.slow),
    ],
)
def scheduling_input(request: pytest.FixtureRequest) -> SchedulingInput:
    """Load a ``SchedulingInput`` from a JSON example file.

    Parameterised over all example files.  ``scope="module"`` ensures each
    file is parsed only once per test session.
    """
    json_file = Path(request.param)
    assert json_file.suffix == ".json"
    with json_file.open("r") as f:
        input_dict = json.load(f)

    return SchedulingInput.from_dict(input_dict)


# ---------------------------------------------------------------------------
# Fixtures: solve
# ---------------------------------------------------------------------------

# mu values to test: default (2), low (1), high (5).
mus: list[float] = [2, 1, 5]
fast_mu_values = mus[:1]  # only mu=2 in the fast test run

# All solvers available in the current environment plus a sentinel ``None``
# that makes ``solve_scheduling`` choose automatically.
available_solvers = linopy.solvers.available_solvers + [None]
core_solver_set = {"highs", "gurobi"}
licensed_solvers = {"gurobi"}

# Build the cross-product of (mu, solver) parameter combinations and assign
# marks based on how expensive / how licensed each combination is.
scheduling_params: list[ParameterSet] = []
for mu, solver_name in product(mus, available_solvers):
    marks = []
    if solver_name not in core_solver_set:
        # Non-core solvers are both slow and only run in the extensive suite.
        marks.extend([pytest.mark.slow, pytest.mark.extensive])
    elif mu not in fast_mu_values:
        # Non-default mu values are slower (model is rebuilt with a new mu).
        marks.append(pytest.mark.slow)

    if solver_name in licensed_solvers:
        marks.append(pytest.mark.licensed)

    scheduling_params.append(pytest.param((mu, solver_name), marks=marks))


scheduling_param_ids: list[str] = []
for param in scheduling_params:
    mu, solver_name = cast(tuple[float, str], param.values[0])
    scheduling_param_ids.append(f"mu={mu}-{solver_name}")


@pytest.fixture(
    scope="module",
    ids=scheduling_param_ids,
    params=scheduling_params,
)
def solved_lp_fixture(
    request: pytest.FixtureRequest, scheduling_input: SchedulingInput
) -> tuple[linopy.Model, types.ExportTuple, SchedulingInput]:
    """Solve the scheduling MILP and return the model, solution, and input.

    Parameterised over ``(mu, solver_name)`` combinations.  ``scope="module"``
    means each combination is solved only once and the result is shared among
    all assertion test functions.

    The 60-second time limit is intentionally generous — it ensures even
    harder instances finish in CI while still catching pathological cases.

    Args:
        request: Pytest fixture request; ``request.param`` is ``(mu, solver_name)``.
        scheduling_input: The loaded ``SchedulingInput`` from the outer fixture.

    Returns:
        ``(model, solution, scheduling_input)`` tuple.

    Raises:
        AssertionError: If the solver returns infeasible (the test should not
            fail on a valid input).
    """
    mu, solver_name = request.param
    solver_config = SolverConfig(
        threads=default_num_threads(),
        time_limit=60,
    )
    # Mutate mu on the shared config object.  ``ConfigData`` is not frozen so
    # this is intentional; the fixture scope ensures no race condition.
    scheduling_input.config.mu = mu

    solution_tuple = solve_scheduling(
        scheduling_input,
        solver_config=solver_config,
        solver_name=solver_name,
    )
    assert solution_tuple is not None, "Model is infeasible!"

    return (*solution_tuple, scheduling_input)


@pytest.fixture(scope="module")
def scheduled_aks(
    solved_lp_fixture: tuple[linopy.Model, types.ExportTuple, SchedulingInput],
) -> dict[types.AkId, ScheduleAtom]:
    """Extract a ``dict[AkId, ScheduleAtom]`` from the solved LP fixture.

    Skips the test (rather than failing) if ``process_solved_lp`` returns
    ``None`` (e.g. when a time limit was hit before any feasible solution was
    found).
    """
    solved_lp_problem, solution, scheduling_input = solved_lp_fixture

    schedule = process_solved_lp(
        solved_lp_problem, solution, input_data=scheduling_input
    )

    if schedule is None:
        pytest.skip("No LP solution found")

    return schedule


# ---------------------------------------------------------------------------
# Convenience fixtures: lookup dicts
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ak_dict(scheduling_input: SchedulingInput) -> dict[types.AkId, AKData]:
    """Dict mapping AK ID → ``AKData`` for fast lookup in tests."""
    return {ak.id: ak for ak in scheduling_input.aks}


@pytest.fixture(scope="module")
def participant_dict(
    scheduling_input: SchedulingInput,
) -> dict[types.PersonId, ParticipantData]:
    """Dict mapping participant ID → ``ParticipantData`` for fast lookup."""
    return {
        participant.id: participant for participant in scheduling_input.participants
    }


@pytest.fixture(scope="module")
def room_dict(scheduling_input: SchedulingInput) -> dict[types.RoomId, RoomData]:
    """Dict mapping room ID → ``RoomData`` for fast lookup in tests."""
    return {room.id: room for room in scheduling_input.rooms}


@pytest.fixture(scope="module")
def timeslot_dict(
    scheduling_input: SchedulingInput,
) -> dict[types.TimeslotId, TimeSlotData]:
    """Dict mapping timeslot ID → ``TimeSlotData`` for fast lookup in tests."""
    return {
        timeslot.id: timeslot
        for block in scheduling_input.timeslot_blocks
        for timeslot in block
    }


@pytest.fixture(scope="module")
def timeslot_blocks(scheduling_input: SchedulingInput) -> list[list[TimeSlotData]]:
    """The raw timeslot block structure from the input."""
    return scheduling_input.timeslot_blocks


# ---------------------------------------------------------------------------
# Assertion tests
# Each test function receives the ``scheduled_aks`` fixture (and optionally
# lookup-dict fixtures) and asserts one scheduling constraint.
# ---------------------------------------------------------------------------


def test_rooms_not_overbooked(scheduled_aks: dict[types.AkId, ScheduleAtom]) -> None:
    """No (room, timeslot) pair is used by more than one AK.

    Verifies the ``MaxOneAKPerRoomAndTime`` constraint: two AKs cannot share
    a room at the same time.
    """
    assert _test_uniqueness(
        [
            (ak.room_id, timeslot_id)
            for ak in scheduled_aks.values()
            for timeslot_id in ak.timeslot_ids
        ]
    )[-1]


def test_participant_no_overlapping_timeslot(
    scheduled_aks: dict[types.AkId, ScheduleAtom],
) -> None:
    """No participant is scheduled to attend two AKs at the same time.

    Verifies the ``MaxOneAKPerPersonAndTime`` constraint.
    """
    assert _test_uniqueness(
        [
            (participant_id, timeslot_id)
            for ak in scheduled_aks.values()
            for timeslot_id in ak.timeslot_ids
            for participant_id in ak.participant_ids
        ]
    )[-1]


def test_ak_lengths(
    scheduled_aks: dict[types.AkId, ScheduleAtom], ak_dict: dict[types.AkId, AKData]
) -> None:
    """Each AK is assigned exactly the number of timeslots its ``duration`` requires.

    Also checks that all assigned timeslot IDs are distinct (no duplicates).
    Verifies the ``AKDuration`` constraint.
    """
    for ak in scheduled_aks.values():
        timeslots = set(ak.timeslot_ids)
        assert len(ak.timeslot_ids) == len(timeslots)
        assert len(timeslots) == ak_dict[ak.ak_id].duration


def test_room_capacities(
    scheduled_aks: dict[types.AkId, ScheduleAtom],
    room_dict: dict[types.RoomId, RoomData],
) -> None:
    """The number of attendees for each AK does not exceed its room's capacity.

    Also checks that participant IDs are distinct within each AK.
    Verifies the ``Roomsize`` constraint.
    """
    for ak in scheduled_aks.values():
        participants = set(ak.participant_ids)
        assert len(ak.participant_ids) == len(participants)
        assert ak.room_id is not None
        assert len(participants) <= room_dict[ak.room_id].capacity


def test_timeslots_consecutive(
    scheduled_aks: dict[types.AkId, ScheduleAtom],
    timeslot_blocks: list[list[TimeSlotData]],
) -> None:
    """Each AK's timeslots are consecutive and lie within a single block.

    Verifies ``AKContiguous`` and ``AKSingleBlock`` / ``AKBlockAssign``.
    The test converts global timeslot IDs to ``(block_idx, slot_idx)`` pairs
    and checks that consecutive pairs differ by exactly one position within
    the same block.
    """
    for ak in scheduled_aks.values():
        # Map each timeslot ID to its (block_idx, position_within_block).
        timeslots = [
            (block_idx, timeslot_idx)
            for block_idx, block in enumerate(timeslot_blocks)
            for timeslot_idx, timeslot in enumerate(block)
            if timeslot.id in ak.timeslot_ids
        ]
        timeslots.sort()

        # Walk consecutive pairs and assert they are adjacent within the same block.
        for (prev_block_idx, prev_timeslot_idx), (
            next_block_idx,
            next_timeslot_idx,
        ) in zip(timeslots, timeslots[1:], strict=False):
            assert prev_timeslot_idx + 1 == next_timeslot_idx
            assert prev_block_idx == next_block_idx


def test_room_constraints(
    scheduled_aks: dict[types.AkId, ScheduleAtom],
    ak_dict: dict[types.AkId, AKData],
    participant_dict: dict[types.PersonId, ParticipantData],
    room_dict: dict[types.RoomId, RoomData],
) -> None:
    """Every AK's assigned room fulfills all required room constraints.

    Checks both the AK's own room constraints and the union of all attending
    participants' room constraints.
    Verifies the ``RoomImpossibleForPerson`` and variable-bound pruning logic.
    """
    for ak in scheduled_aks.values():
        assert ak.room_id is not None
        fulfilled_room_constraints = set(
            room_dict[ak.room_id].fulfilled_room_constraints
        )
        room_constraints_ak = set(ak_dict[ak.ak_id].room_constraints)
        if ak.participant_ids:
            room_constraints_participants = set.union(
                *(
                    set(participant_dict[participant_id].room_constraints)
                    for participant_id in ak.participant_ids
                )
            )
        else:
            room_constraints_participants = set()
        # Verify: required labels ⊆ fulfilled labels (no unsatisfied requirement).
        assert not room_constraints_ak.difference(fulfilled_room_constraints)
        assert not room_constraints_participants.difference(fulfilled_room_constraints)


def test_time_constraints(
    scheduled_aks: dict[types.AkId, ScheduleAtom],
    ak_dict: dict[types.AkId, AKData],
    participant_dict: dict[types.PersonId, ParticipantData],
    room_dict: dict[types.RoomId, RoomData],
    timeslot_dict: dict[types.TimeslotId, TimeSlotData],
) -> None:
    """Every AK's timeslots fulfill all required time constraints.

    Checks the AK's constraints, the assigned room's constraints, and all
    attending participants' constraints.  For multi-slot AKs, the
    *intersection* of fulfilled constraints across all slots is used
    (a constraint must be fulfilled by every slot, not just one).
    Verifies the ``TimeImpossibleForRoom`` and variable-bound pruning logic.
    """
    for ak in scheduled_aks.values():
        assert ak.room_id is not None
        time_constraints_room = set(room_dict[ak.room_id].time_constraints)
        time_constraints_ak = set(ak_dict[ak.ak_id].time_constraints)

        # Compute the intersection of fulfilled labels across all AK timeslots.
        # A label counts as fulfilled only if ALL slots of this AK satisfy it.
        fullfilled_time_constraints = set(
            timeslot_dict[ak.timeslot_ids[0]].fulfilled_time_constraints
        )
        for timeslot_id in ak.timeslot_ids[1:]:
            fullfilled_time_constraints = fullfilled_time_constraints.intersection(
                set(timeslot_dict[timeslot_id].fulfilled_time_constraints)
            )
        if ak.participant_ids:
            time_constraints_participants = set.union(
                *(
                    set(participant_dict[participant_id].time_constraints)
                    for participant_id in ak.participant_ids
                )
            )
        else:
            time_constraints_participants = set()
        assert not time_constraints_room.difference(fullfilled_time_constraints)
        assert not time_constraints_ak.difference(fullfilled_time_constraints)
        assert not time_constraints_participants.difference(fullfilled_time_constraints)


def test_required(
    scheduled_aks: dict[types.AkId, ScheduleAtom],
    participant_dict: dict[types.PersonId, ParticipantData],
) -> None:
    """Every participant who is ``required`` for an AK is present in its attendees.

    Verifies the hard-constraint encoding: ``required=True`` preferences are
    implemented as lower-bound fixes on the ``Part`` variable (not via the
    objective), so they must always be satisfied.
    """
    for participant_id, participant in participant_dict.items():
        for pref in participant.preferences:
            pref_fulfilled = participant_id in scheduled_aks[pref.ak_id].participant_ids
            # Logical implication: required => pref_fulfilled
            # Equivalent to: (not required) or pref_fulfilled
            assert not pref.required or pref_fulfilled


def test_conflicts(
    scheduled_aks: dict[types.AkId, ScheduleAtom], ak_dict: dict[types.AkId, AKData]
) -> None:
    """AKs declared as conflicts do not share any timeslots.

    Verifies the ``AKConflict`` constraint for explicit conflict pairs
    (not dependency pairs, which are checked separately).
    """
    for ak_id, ak in ak_dict.items():
        for conflicting_ak in ak.properties.get("conflicts", []):
            ak_timeslots = scheduled_aks[ak_id].timeslot_ids
            conflicting_ak_timeslots = scheduled_aks[conflicting_ak].timeslot_ids
            assert not set(ak_timeslots).intersection(set(conflicting_ak_timeslots))


def test_dependencies(
    scheduled_aks: dict[types.AkId, ScheduleAtom], ak_dict: dict[types.AkId, AKData]
) -> None:
    """AKs with dependencies are scheduled strictly after all their dependencies.

    Verifies the ``AKDependenciesDoneBeforeAK`` constraint: the *maximum*
    timeslot ID of the dependency must be strictly less than the *minimum*
    timeslot ID of the dependent AK.  This relies on timeslot IDs being
    globally ordered (which the generator and real inputs guarantee).
    """
    for ak_id, ak in ak_dict.items():
        for dependent_ak in ak.properties.get("dependencies", []):
            ak_timeslots = scheduled_aks[ak_id].timeslot_ids
            dependent_ak_timeslots = scheduled_aks[dependent_ak].timeslot_ids
            assert max(map(int, dependent_ak_timeslots)) < min(map(int, ak_timeslots))
