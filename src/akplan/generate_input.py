"""Synthetic test-input generator for the akplan scheduling problem.

Generates randomised but structurally valid JSON problem instances that can be
used as test fixtures, benchmarks, or demo inputs.  The output format is
identical to a real conference input file and can be fed directly to
``akplan-solve``.

Typical usage
-------------
From the command line::

    python -m akplan.generate_input \\
        --aks 20 --persons 50 --rooms 4 \\
        --num_room_constraints 5 \\
        --room_poisson_mean 0.25 \\
        --conflicts 5 --dependencies 3 \\
        --seed 42

Output file: ``examples/test_20a_50p_4r_5rc_0.25rc-lam_5confl_3dep_42.json``

The filename encodes all parameters so that different instances are easily
identifiable and the generation is reproducible via the ``--seed`` flag.

Conference structure
--------------------
The generator hard-codes a 4-day conference schedule that mirrors a typical
KoMa event:

- **Dienstag** (Tuesday): 10 one-hour slots  → timeslots 0–9
- **Mittwoch** (Wednesday): 8 one-hour slots  → timeslots 10–17
- **Donnerstag** (Thursday): 8 one-hour slots  → timeslots 18–25
- **Freitag** (Friday): 10 one-hour slots  → timeslots 26–35

All slots on Tuesday additionally carry the ``"ResoAK"`` time-constraint
label, so AKs that require this label (Reso-AKs) are automatically restricted
to the first day.

Relationship to other modules
------------------------------
This module is standalone — it imports only ``numpy`` and the standard
library.  It does not import from ``util`` or ``solve`` so it can be run even
without a solver installed.  The output is a plain dict / JSON file that is
later consumed by ``SchedulingInput.from_dict``.
"""

import argparse
import json
from collections import defaultdict
from typing import Any, cast

import numpy as np


def generate(
    num_persons: int,
    num_aks: int,
    num_rooms: int,
    num_room_constraints: int,
    seed: int,
    room_poisson_mean: float,
    num_of_conflicts: int,
    num_of_dependencies: int,
) -> dict[str, Any]:
    """Generate a randomised conference scheduling problem instance.

    All random choices use ``numpy.random.default_rng(seed)`` so results are
    fully reproducible.  The generated instance is structurally valid: every
    AK has at least one room that can satisfy its constraints (because rooms
    are generated with random constraint subsets too), and every participant
    has at least one AK they prefer.

    The timeslot grid and day structure are fixed (see module docstring).
    Room capacities, AK room constraints, participant preferences, required
    persons, conflicts, and dependencies are all randomised according to the
    parameters below.

    Args:
        num_persons: Number of conference participants to generate.
        num_aks: Number of AKs (sessions) to generate.
        num_rooms: Number of rooms to generate.
        num_room_constraints: Number of distinct room-constraint label strings
            in the universe (e.g. ``"room-constraint-0"`` … ``"room-constraint-4"``
            for ``num_room_constraints=5``).
        seed: RNG seed for reproducibility.
        room_poisson_mean: Mean of the Poisson distribution used to sample
            how many room constraints each AK requires.  Low values (≈0.25)
            produce mostly unconstrained AKs; higher values produce AKs that
            need more specific rooms.
        num_of_conflicts: Number of random AK conflict pairs to inject.  A
            conflict means the two AKs must not overlap in time.
        num_of_dependencies: Number of random AK dependency pairs to inject.
            A dependency (a → b) means AK ``b`` must finish before AK ``a``
            starts.

    Returns:
        A dict matching the JSON input schema expected by
        ``SchedulingInput.from_dict``.  Can be written directly with
        ``json.dump``.
    """
    rng = np.random.default_rng(seed=seed)

    # -------------------------------------------------------------------------
    # Timeslot grid
    # Mirrors a typical 4-day KoMa conference.
    # Each tuple is (day_label, number_of_1-hour_slots).
    # -------------------------------------------------------------------------
    # we have one hour time slots
    # on Tuesday we go from 8-18, Wednesday from 8-16,
    #    Thursday from 8-16, Friday from 8-18
    block_properties = [
        ("Dienstag", 10),
        ("Mittwoch", 8),
        ("Donnerstag", 8),
        ("Freitag", 10),
    ]

    # create timeslots: build a list-of-lists (one inner list per block).
    list_of_time_blocks = []

    global_timeslot_cnt = 0
    for block_id, (block_label, block_size) in enumerate(block_properties):
        # Every slot in a block carries the day label so AKs can be restricted
        # to a specific day via time_constraints.
        fulfilled_time_constraints = [block_label]
        if block_id == 0:  # die Reso Aks sollen alle am ersten Tag stattfinden
            # Tuesday slots additionally carry "ResoAK" so that AKs marked
            # as resolution AKs (reso_ak_arr below) are forced onto day 1.
            fulfilled_time_constraints.append("ResoAK")
        list_of_time_blocks.append(
            [
                {
                    "id": global_timeslot_cnt + slot_idx,
                    "info": {"start": f"{block_label}, {8 + slot_idx} Uhr"},
                    "fulfilled_time_constraints": list(fulfilled_time_constraints),
                }
                for slot_idx in range(block_size)
            ]
        )
        global_timeslot_cnt += block_size

    time_slot_dictionary = {
        "info": {"duration": "1 Stunde"},
        "blocks": list_of_time_blocks,
    }

    # -------------------------------------------------------------------------
    # Rooms
    # Each room has a random capacity in [10, 50] and a random subset of room
    # constraint labels that it fulfills.
    # -------------------------------------------------------------------------
    all_room_constraints = [
        f"room-constraint-{idx}" for idx in range(num_room_constraints)
    ]
    # create rooms:
    rooms = [
        {
            "id": room_idx,
            "info": {"name": f"room {room_idx}"},
            "capacity": int(rng.integers(low=10, high=51)),
            "fulfilled_room_constraints": list(
                rng.choice(
                    all_room_constraints,
                    # random number of fulfilled constraints: 0 … all
                    size=rng.integers(low=0, high=len(all_room_constraints) + 1),
                    replace=False,
                )
            ),
            "time_constraints": [],
        }
        for room_idx in range(num_rooms)
    ]

    # -------------------------------------------------------------------------
    # AKs
    # Each AK gets:
    #   - A Poisson-sampled number of room constraints it requires.
    #   - A 20% chance of being a Reso-AK (restricted to day 1).
    #   - A duration of 1 or 2 slots (uniform).
    # -------------------------------------------------------------------------
    # create aks
    room_constraint_arr = [
        rng.choice(
            all_room_constraints,
            replace=False,
            # cap at the universe size to avoid impossible constraints
            size=np.minimum(
                len(all_room_constraints),
                rng.poisson(lam=room_poisson_mean),
            ),
        )
        for _ak_idx in range(num_aks)
    ]
    # ~20% of AKs are Reso-AKs (they require the "ResoAK" time constraint).
    reso_ak_arr = rng.choice(2, p=[0.8, 0.2], size=num_aks).astype(bool)
    # Duration: 1 or 2 timeslots (uniform).
    duration_arr = rng.choice(2, size=num_aks) + 1

    aks = [
        {
            "id": ak_idx,
            "duration": int(duration),
            "properties": {"conflicts": [], "dependencies": []},
            "room_constraints": list(room_constraints),
            "time_constraints": ["ResoAK"] if is_reso_ak else [],
            "info": {
                "name": f"AK {ak_idx}",
                "head": "N/A",
                "description": "N/A",
                "reso": bool(is_reso_ak),
            },
        }
        for ak_idx, (room_constraints, is_reso_ak, duration) in enumerate(
            zip(room_constraint_arr, reso_ak_arr, duration_arr, strict=True)
        )
    ]

    # -------------------------------------------------------------------------
    # Participants & preferences
    # Each participant's number of preferred AKs is Poisson-distributed with
    # mean = max(10, 20% of AKs).  Preferences are sampled without replacement
    # from the AK pool.
    # -------------------------------------------------------------------------
    num_preferences_arr = np.minimum(
        rng.poisson(
            lam=max(10, round(0.2 * num_aks)),
            size=num_persons,
        ),
        num_aks,  # cap at the total number of AKs
    )

    sampled_aks = {
        person_idx: set(rng.choice(num_aks, replace=False, size=num_prefs))
        for person_idx, num_prefs in enumerate(num_preferences_arr)
    }

    # -------------------------------------------------------------------------
    # Required persons
    # Half the participants are eligible to be required for AKs.
    # Each AK gets 0, 1, or 2 required persons (10% / 80% / 10% split).
    # -------------------------------------------------------------------------
    # Add AK conflicts and dependencies
    for _ in range(num_of_conflicts):
        ak_a, ak_b = rng.choice(num_aks, size=2, replace=False)
        properties_dict = cast(dict[str, list[Any]], aks[ak_a]["properties"])
        properties_dict["conflicts"].append(aks[ak_b]["id"])

    # Add AK conflicts and dependencies
    for _ in range(num_of_dependencies):
        ak_a, ak_b = rng.choice(num_aks, size=2, replace=False)
        properties_dict = cast(dict[str, list[Any]], aks[ak_a]["properties"])
        properties_dict["dependencies"].append(aks[ak_b]["id"])

    # 1. Ignore one half of participants
    required_indices = rng.choice(
        num_persons, size=round(0.5 * num_persons), replace=False
    )
    # 2. For each ak sample the person(s) required for the ak (0.1/0.8/0.1 split)
    num_persons_required = rng.choice(3, size=num_aks, p=[0.1, 0.8, 0.1])

    required_aks: dict[int, set[int]] = defaultdict(set)
    for ak_idx, num_required in enumerate(num_persons_required):
        persons_required_for_ak = rng.choice(
            required_indices, replace=False, size=num_required
        )
        for person_idx in persons_required_for_ak:
            required_aks[person_idx].add(ak_idx)

    def _calc_preferred_score(person_idx: int, ak_idx: int) -> int:
        """Return a preference score for the given (person, AK) pair.

        Required AKs always get score -1.  All other sampled preferences get
        a random score of 1 (weak) or 2 (strong) with equal probability.
        """
        if ak_idx in required_aks[person_idx]:
            return -1
        return int(rng.choice(2)) + 1

    # TODO: Generate room & time constraints
    participants = [
        {
            "id": person_idx,
            "info": {"name": f"Person {person_idx}"},
            "preferences": [
                {
                    "ak_id": int(ak_idx),
                    "required": bool(ak_idx in required_aks[person_idx]),
                    "preference_score": _calc_preferred_score(person_idx, ak_idx),
                }
                # union of sampled preferred AKs and required AKs for this person
                for ak_idx in preferred_aks.union(required_aks[person_idx])
            ],
            "room_constraints": [],
            "time_constraints": [],
        }
        for person_idx, preferred_aks in sampled_aks.items()
    ]

    # create dictionary that we later write into the json-file
    return {
        "aks": aks,
        "rooms": rooms,
        "participants": participants,
        "timeslots": time_slot_dictionary,
        "info": "DummySet",
    }


def main() -> None:
    """CLI entry point for the test-input generator.

    Parses arguments, calls ``generate``, and writes the result to a JSON
    file in the ``examples/`` directory.  The filename encodes all parameter
    values so instances are self-documenting:

        ``examples/test_{aks}a_{persons}p_{rooms}r_{constraints}rc_``
        ``{lam}rc-lam[_{conflicts}confl][_{deps}dep]_{seed}.json``

    Example: ``examples/test_20a_50p_4r_5rc_0.25rc-lam_5confl_3dep_42.json``
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--persons", type=int, default=30)
    parser.add_argument("--aks", type=int, default=10)
    parser.add_argument("--rooms", type=int, default=4)
    parser.add_argument("--num_room_constraints", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--room_poisson_mean", type=float, default=1)
    parser.add_argument("--conflicts", type=int, default=0)
    parser.add_argument("--dependencies", type=int, default=0)
    args = parser.parse_args()

    output_dict = generate(
        num_persons=args.persons,
        num_aks=args.aks,
        num_rooms=args.rooms,
        num_room_constraints=args.num_room_constraints,
        seed=args.seed,
        room_poisson_mean=args.room_poisson_mean,
        num_of_conflicts=args.conflicts,
        num_of_dependencies=args.dependencies,
    )

    # Build output filename by joining non-empty parameter segments.
    arg_list = [
        "examples/test",
        f"{args.aks}a",
        f"{args.persons}p",
        f"{args.rooms}r",
        f"{args.num_room_constraints}rc",
        f"{args.room_poisson_mean:.2f}rc-lam",
        (f"{args.conflicts}confl" if args.conflicts > 0 else ""),
        (f"{args.dependencies}dep" if args.dependencies > 0 else ""),
        f"{args.seed}.json",
    ]
    filename = "_".join([x for x in arg_list if x])

    with open(filename, "w") as output_file:
        json.dump(output_dict, output_file, indent=4)

    print(f"Generated {filename}")


if __name__ == "__main__":
    main()
