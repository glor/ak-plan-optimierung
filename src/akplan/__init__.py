"""Conference scheduling using MILPs.

This is the top-level package for ``akplan``.

The package solves the problem of scheduling a set of AKs (Arbeitskreise —
working-group sessions) at a conference into rooms and timeslots while
maximising how many participants can attend the sessions they prefer.

Sub-modules
-----------
types
    Lightweight type aliases, TypedDicts and NamedTuples shared across the
    package.  No business logic lives here.
util
    All dataclasses that mirror the JSON input/output schema
    (``AKData``, ``ParticipantData``, ``RoomData``, ``TimeSlotData``,
    ``SchedulingInput``, …) together with preprocessing helpers that turn raw
    input data into the numerical arrays the MILP builder needs.
solve
    The MILP construction (``create_lp``), the solve orchestration
    (``solve_scheduling``), result extraction (``export_scheduling_result``)
    and the ``akplan-solve`` CLI entry-point (``main``).
generate_input
    A standalone script / CLI tool that generates synthetic JSON problem
    instances for testing and benchmarking.
"""
