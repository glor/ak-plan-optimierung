# Development Guide

> **TL;DR:** Install with `pip install -e .`, run tests with `nox -s test`, lint with `nox -s lint`. Use `generate_input.py` to create synthetic test problems.

---

## Setup

### User / production install

```sh
pip install git+https://github.com/Die-KoMa/ak-plan-optimierung.git
akplan-solve path/to/input.json
```

### Development install

```sh
git clone https://github.com/Die-KoMa/ak-plan-optimierung.git
cd ak-plan-optimierung
pip install -e .          # editable install
pip install nox           # task runner
```

---

## Running the solver (CLI)

```sh
akplan-solve INPUT.json [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--solver` | auto | `gurobi` or `highs` |
| `--solver-dir PATH` | temp dir | Where linopy writes LP/solution files |
| `--solver-io-api` | `direct` | `direct` \| `lp` \| `mps` |
| `--solver-warmstart-fn PATH` | — | Basis file for warm-starting |
| `--timelimit SECS` | — | Stop after N seconds |
| `--gap-rel FLOAT` | — | Stop when relative MIP gap ≤ this |
| `--gap-abs FLOAT` | — | Stop when absolute MIP gap ≤ this |
| `--threads N` | `#CPUs − 1` | Parallel threads |
| `--loglevel` | `info` | `debug` \| `info` \| `warning` \| `error` |
| `--output PATH` | `out-INPUT.json` | Output file |
| `--override-output` | off | Overwrite existing output file |

The output is written to `./out-<input-filename>.json` by default. The solver exits cleanly (no output file) if the problem is infeasible.

---

## Using the Python API

```python
import json
from akplan.util import SchedulingInput, SolverConfig
from akplan.solve import solve_scheduling, process_solved_lp

with open("input.json") as f:
    data = json.load(f)

scheduling_input = SchedulingInput.from_dict(data)
solver_config = SolverConfig(time_limit=60, threads=4)

result = solve_scheduling(scheduling_input, solver_config, solver_name="highs")
if result is None:
    print("Infeasible!")
else:
    model, solution = result
    schedule = process_solved_lp(model, solution, scheduling_input)
    # schedule: dict[AkId, ScheduleAtom]
```

---

## Generating test inputs

`generate_input.py` produces synthetic JSON inputs with randomised participants, rooms, and preferences:

```sh
python -m akplan.generate_input \
  --aks 20 --persons 50 --rooms 4 \
  --num_room_constraints 5 \
  --room_poisson_mean 0.25 \
  --conflicts 5 --dependencies 3 \
  --seed 42
```

Output is written to `examples/test_20a_50p_4r_5rc_0.25rc-lam_5confl_3dep_42.json`.

The filename encodes all parameters for easy identification. Example files already committed under `examples/` use this naming convention.

---

## Nox sessions

Run any session with `nox -s <name>`:

| Session | Command run | Notes |
|---|---|---|
| `test` | `pytest -m "not extensive"` | Default test run |
| `fast-test` | `pytest -m "not slow and not extensive"` | Skips long-running cases |
| `fast-unlicensed-test` | `pytest -m "not slow and not extensive and not licensed"` | CI-safe (no Gurobi licence needed) |
| `extensive-test` | `pytest` | All tests |
| `lint` | `ruff check` | Style and convention checks |
| `typing` | `mypy` | Static type checking (strict mode) |
| `format` | `ruff format` | Auto-fix formatting |
| `coverage` | `coverage run -m pytest` + HTML report | Coverage report in `htmlcov/` |
| `coverage-clean` | `rm -r htmlcov` | Remove HTML report |

---

## Test suite (`tests/test_schedule_feasibility.py`)

Tests are **property-based**: the solver is run on each example JSON, and the resulting schedule is verified against the problem constraints.

### Test fixtures

| Fixture | What it provides |
|---|---|
| `scheduling_input` | A `SchedulingInput` loaded from each example JSON |
| `solved_lp_fixture` | A solved `(model, solution, input)` for a `(μ, solver)` combo |
| `scheduled_aks` | A `dict[AkId, ScheduleAtom]` produced by `process_solved_lp` |

### Test functions

| Test | What it checks |
|---|---|
| `test_rooms_not_overbooked` | No `(room, timeslot)` pair is used by two AKs |
| `test_participant_no_overlapping_timeslot` | No participant is double-booked |
| `test_ak_lengths` | Each AK has exactly `duration` distinct timeslots |
| `test_room_capacities` | Attendee count ≤ room capacity |
| `test_timeslots_consecutive` | AK timeslots are consecutive within a single block |
| `test_room_constraints` | Room satisfies all room constraints of the AK and its participants |
| `test_time_constraints` | Timeslots satisfy all time constraints of AK, room, and participants |
| `test_required` | Every `required=true` preference is fulfilled |
| `test_conflicts` | Conflicting AK pairs have no overlapping timeslots |
| `test_dependencies` | Dependent AKs finish before the AKs that depend on them start |

### Pytest marks

| Mark | Meaning |
|---|---|
| `slow` | Larger/harder instances, skipped in `fast-test` |
| `extensive` | Skipped in `test` and `fast-test`, only in `extensive-test` |
| `licensed` | Requires a Gurobi licence |

---

## Code style

- **Python ≥ 3.11** required.
- **Ruff** for linting and formatting (configured in `pyproject.toml`).
- **mypy** in strict mode.
- All relative imports are banned (`flake8-tidy-imports`).
- Docstrings follow Google style (`pydocstyle`).

---

## Repository layout

```
ak-plan-optimierung/
├── src/akplan/           # source package
│   ├── __init__.py
│   ├── types.py
│   ├── util.py
│   ├── solve.py
│   ├── generate_input.py
│   └── py.typed
├── tests/
│   └── test_schedule_feasibility.py
├── examples/             # example + test JSON inputs
├── docs/                 # this documentation
├── .github/              # CI workflows (lint, typing, tests, release)
├── noxfile.py
├── pyproject.toml
└── README.md
```
