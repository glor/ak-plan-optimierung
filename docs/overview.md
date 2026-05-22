# Overview

> **TL;DR:** `akplan` is a Python backend that reads a conference schedule problem from a JSON file, solves it as a Mixed-Integer Linear Program (MILP), and writes the optimal schedule back as JSON. It is designed as a pure backend — all user interaction happens via JSON import/export with an external frontend.

## What is this?

`akplan` optimizes the schedule of a conference where participants self-organise into topic-based working groups called **AKs** (*Arbeitskreise*, German: working groups). This is the scheduling backend used for conferences run by the [KoMa](https://die-koma.org) (Konferenz der Mathematikfachschaften — Conference of German Math Student Councils).

The conference has:
- **AKs** — sessions/workshops, each with a required duration and optional constraints (needs a beamer, must be on day 1, etc.)
- **Participants** — attendees with preferences over which AKs they want to join and hard availability constraints
- **Rooms** — physical spaces with a capacity and a set of features they fulfill
- **Timeslots** — chronological slots grouped into **blocks** (typically one day per block)

The goal is to assign every AK a room and contiguous timeslots within a single block, while maximising how many participants can attend their preferred sessions.

## High-level workflow

```
JSON input file
      │
      ▼
┌───────────────────┐
│  Parse & validate │  SchedulingInput.from_dict()
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Build MILP model │  create_lp()  ─── linopy Model
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Solve with ILP   │  HiGHS / Gurobi
│  solver           │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Extract schedule │  export_scheduling_result()
└────────┬──────────┘
         │
         ▼
JSON output file
```

Steps:
1. **Read** — Load the JSON input (AKs, participants, rooms, timeslots, constraints).
2. **Construct** — Translate the problem into a Mixed-Integer Linear Program using [linopy](https://github.com/PyPSA/linopy).
3. **Solve** — Hand the model to HiGHS or Gurobi.
4. **Export** — Write the assignment (AK → room, timeslots, participants) to a JSON output file.

## Package structure

```
src/akplan/
├── __init__.py         # package entry point ("Conference scheduling using MILPs")
├── types.py            # Type aliases and TypedDicts (IDs, solver kwargs, ExportTuple)
├── util.py             # Data-classes for input (AKData, ParticipantData, RoomData,
│                       #   TimeSlotData, SchedulingInput, …) and helper logic
├── solve.py            # MILP construction (create_lp), solving (solve_scheduling),
│                       #   result export, and the CLI entry-point (main)
├── generate_input.py   # Test-input generator (CLI: generate random problem instances)
└── py.typed            # PEP 561 marker – package ships type stubs
```

## CLI entry points

| Command | Description |
|---|---|
| `akplan-solve <input.json>` | Solve a scheduling problem and write `out-<input>.json` |

See [Development Guide](development.md) for the test-input generator.

## Key dependencies

| Package | Role |
|---|---|
| [linopy](https://github.com/PyPSA/linopy) | Builds and interfaces the MILP model |
| HiGHS (`highspy`) | Open-source MIP solver (default) |
| Gurobi (`gurobipy`) | Commercial MIP solver (optional, requires licence) |
| numpy / pandas / xarray | Numerical data structures for constraint matrices |
| dacite | Deserialises JSON dicts into typed dataclasses |
| tqdm | Progress bars |
