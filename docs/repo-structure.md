# Repository Structure

> **TL;DR:** A standard Python `src`-layout package. Source lives in `src/akplan/`, tests in `tests/`, real and synthetic example inputs in `examples/`, CI in `.github/workflows/`, and project config in `pyproject.toml`.

---

## Annotated tree

```
ak-plan-optimierung/
│
├── src/                          # ── Source root (PEP 517 src-layout)
│   └── akplan/                   #    Installable package "akplan"
│       ├── __init__.py           #    Package marker + module docstring
│       ├── py.typed              #    PEP 561: signals that the package ships type info
│       ├── types.py              #    Type aliases, TypedDicts, NamedTuples
│       ├── util.py               #    Dataclasses for all input/output entities +
│       │                         #      helpers (preference processing, room capacity,
│       │                         #      ProblemIds, ProblemProperties, SolverConfig)
│       ├── solve.py              #    MILP construction, solving, result export,
│       │                         #      and the CLI entry-point `akplan-solve`
│       └── generate_input.py     #    Synthetic test-input generator (CLI tool)
│
├── tests/
│   ├── __init__.py               #    Makes `tests` a package (needed for mypy)
│   └── test_schedule_feasibility.py
│                                 #    Property-based tests: solve every example,
│                                 #      then assert all scheduling constraints hold
│
├── examples/                     # ── Input JSON files used as test fixtures and demos
│   ├── koma89.json               #    Real schedule data from KoMa 89
│   ├── koma90.json               #    Real schedule data from KoMa 90
│   ├── koma91.json               #    Real schedule data from KoMa 91
│   ├── koma92.json               #    Real schedule data from KoMa 92
│   ├── test1.json                #    Small hand-crafted test case
│   ├── test2.json                #    Larger hand-crafted test case
│   └── test_<params>_<seed>.json #    Synthetically generated cases (see below)
│
├── docs/                         # ── Developer documentation (this folder)
│   ├── index.md                  #    Navigation index
│   ├── overview.md               #    Project purpose, workflow, dependency table
│   ├── data-model.md             #    JSON schema + Python dataclass hierarchy
│   ├── algorithm.md              #    MILP variables, constraints, objective
│   ├── development.md            #    Setup, CLI, API, nox sessions, test guide
│   └── repo-structure.md         #    This file
│
├── .github/
│   └── workflows/
│       ├── push-tests.yml        #    On every push: lint (ruff) + type-check (mypy)
│       ├── pr-tests.yml          #    On PRs and main: build package + fast tests
│       │                         #      on all supported Python versions (3.11–3.13)
│       └── release.yml           #    On `v*.*.*` tags: build + create GitHub Release
│
├── pyproject.toml                #    Single source of truth for build, deps, tools
├── noxfile.py                    #    Task runner sessions (test, lint, typing, coverage)
├── pytest.ini                   #    Pytest config: timeout=180s, custom markers
├── setup.py                     #    Minimal shim — delegates everything to pyproject.toml
├── .gitignore                    #    Standard Python ignores
└── README.md                     #    User-facing quickstart
```

---

## File-by-file reference

### `src/akplan/types.py`

Thin module of type-level definitions. Nothing executable lives here.

| Symbol | Kind | Purpose |
|---|---|---|
| `Id`, `RoomId`, `PersonId`, `AkId`, `TimeslotId`, `BlockId` | `int` type aliases | Semantic IDs (all resolve to `int` at runtime) |
| `Block` | `pd.Index` alias | A single timeslot block |
| `ScheduleAtomComparisonTuple` | `tuple` alias | Canonical form for hashing/sorting a `ScheduleAtom` |
| `ExportTuple` | `NamedTuple` | Holds the three solution arrays (`room`, `time`, `person`) returned after solving |
| `SupportedSolver` | `Literal["gurobi", "highs"]` | Solvers with full CLI-arg support |
| `SolverKwargs` / `GurobiSolverKwargs` / `HighsSolverKwargs` | `TypedDict` | Per-solver keyword argument shapes |

---

### `src/akplan/util.py`

The data layer. Contains all dataclasses that mirror the JSON schema and the preprocessing logic the MILP builder needs.

| Class / function | Purpose |
|---|---|
| `AKData` | Frozen dataclass for one AK (id, duration, constraints, properties, info) |
| `PreferenceData` | Frozen dataclass for one person↔AK preference entry |
| `ParticipantData` | Frozen dataclass for one participant (id, preferences, constraints, info) |
| `RoomData` | Frozen dataclass for one room (id, capacity, fulfilled constraints, info) |
| `TimeSlotData` | Frozen dataclass for one timeslot (id, fulfilled constraints, info) |
| `ScheduleAtom` | Mutable dataclass for one scheduled AK (ak_id, room_id, timeslot_ids, participant_ids); used for both pre-fixed input AKs and solver output |
| `ConfigData` | Mutable dataclass for solver hyperparameters (μ, break limit, flags) |
| `SchedulingInput` | Top-level frozen dataclass; owns `from_dict()` / `to_dict()` for JSON round-tripping |
| `ProblemIds` | Frozen dataclass of `pd.Index` collections extracted from a `SchedulingInput` (one per entity type) |
| `ProblemProperties` | Frozen dataclass of precomputed `xr.DataArray` matrices (preferences, constraints, capacities, etc.) |
| `SolverConfig` | Frozen dataclass for solver runtime options; `generate_kwargs()` translates to solver-specific dicts |
| `process_pref_score()` | Maps raw preference score + `required` flag to a float MILP weight |
| `process_room_cap()` | Normalises room capacity (−1 → number of participants) |
| `get_ak_name()` | Looks up a human-readable AK name for log messages |
| `default_num_threads()` | Returns `#CPUs − 1`, cross-platform |
| `_construct_constraint_name()` | Builds a unique constraint name string for linopy |

---

### `src/akplan/solve.py`

The core engine and CLI entry point.

| Function | Purpose |
|---|---|
| `create_lp(input_data, solver_dir)` | Builds the complete linopy `Model`: adds all variables, sets bounds, adds all constraints and the objective |
| `export_scheduling_result(input_data, solution, ...)` | Reads the solved variable arrays and constructs a `dict[AkId, ScheduleAtom]` |
| `solve_scheduling(input_data, solver_config, solver_name)` | Orchestrates model build + solve; returns `(model, ExportTuple)` or `None` if infeasible |
| `process_solved_lp(model, solution, input_data)` | Thin wrapper: checks `model.status` then calls `export_scheduling_result` |
| `calc_changed_fixed_schedule_atoms(...)` | After solving, compares pre-fixed input AKs against output to surface any violated fixings |
| `main()` | `argparse`-based CLI; reads JSON → solves → writes JSON; registered as `akplan-solve` entry point |

---

### `src/akplan/generate_input.py`

Standalone script that generates randomised problem instances for testing and benchmarking.

**Output filename convention:**  
`examples/test_{#aks}a_{#persons}p_{#rooms}r_{#room_constraints}rc_{lam}rc-lam[_{#conflicts}confl][_{#deps}dep]_{seed}.json`

Example: `test_20a_20p_5r_5rc_0.25rc-lam_3confl_3dep_0.json`  
→ 20 AKs, 20 persons, 5 rooms, 5 room constraint types, Poisson λ=0.25, 3 conflicts, 3 dependencies, seed 0.

The generated schedule mimics a 4-day conference (Tue 10 h, Wed 8 h, Thu 8 h, Fri 10 h = 36 timeslots total). 20 % of AKs are randomly flagged as "ResoAK" (resolution AKs, constrained to day 1).

---

### `tests/test_schedule_feasibility.py`

All tests follow the same pattern: **solve → assert constraints hold**.  
See [Development Guide — Test suite](development.md#test-suite) for the full test list.

Key design choices:
- `scope="module"` fixtures — the solve is performed once per `(input_file, μ, solver)` combination and reused across all assertion tests, keeping CI fast.
- Tests are parameterised over all example JSON files × `μ ∈ {1, 2, 5}` × available solvers. The matrix is pruned with `pytest.mark.slow` / `extensive` / `licensed` marks.
- A 180-second per-test timeout is enforced via `pytest.ini`.

---

### `examples/`

| File pattern | Origin | Used in tests? |
|---|---|---|
| `koma89.json` … `koma92.json` | Real KoMa conference data | `test2.json` is `slow`; koma files not directly in test suite |
| `test1.json` | Hand-crafted small instance | Yes — default (fast) suite |
| `test2.json` | Hand-crafted larger instance | Yes — `slow` |
| `test_<params>_0.json` | Generated by `generate_input.py` | Yes — various marks |

---

### `pyproject.toml`

Single configuration file for the entire project:

| Section | Controls |
|---|---|
| `[build-system]` | `setuptools` + `setuptools-scm` (version from git tags) |
| `[project]` | Package name, description, Python ≥ 3.11, MIT licence, runtime deps |
| `[project.optional-dependencies]` | `test`, `typing`, `lint`, `format`, `coverage` extras |
| `[project.scripts]` | Registers `akplan-solve = akplan.solve:main` |
| `[tool.setuptools_scm]` | Writes `src/akplan/version.py` from the latest `v*.*.*` git tag |
| `[tool.mypy]` | Strict mode, targets Python 3.11, covers both `akplan` and `tests` packages |
| `[tool.ruff]` | Enables pyflakes, pycodestyle, isort, pydocstyle, pyupgrade, numpy, bugbear, and naming rules; bans relative imports |

---

### `.github/workflows/`

Three workflows, each depending on a shared **build** job that uses [`hynek/build-and-inspect-python-package`](https://github.com/hynek/build-and-inspect-python-package) to produce and validate the wheel:

| Workflow file | Trigger | Jobs |
|---|---|---|
| `push-tests.yml` | Every push | `build` → `check-types` (mypy) + `lint` (ruff) |
| `pr-tests.yml` | PRs, pushes to `main`, weekly Tuesday cron | `build` → `test` (matrix: Python 3.11–3.13, runs `fast-unlicensed-test`) |
| `release.yml` | Push of `v*.*.*` tag | `build` → `release` (creates GitHub Release with auto-generated notes) |

**Versioning:** `setuptools-scm` derives the version from the most recent `v*.*.*` git tag. The wheel produced in CI embeds this version; no manual version bumping needed.
