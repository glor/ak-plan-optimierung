# MILP Formulation

> **TL;DR:** The schedule is found by solving a Mixed-Integer Linear Program with three binary decision variables (Room, Time, Person). Hard constraints enforce feasibility; the objective maximises a weighted sum of preference scores.

The full mathematical specification lives in the [project wiki](https://github.com/Die-KoMa/ak-plan-optimierung/wiki/LP-formulation). This page explains the structure in plain English with implementation pointers.

---

## Decision variables

All variables are **binary** (0 or 1). Linopy declares them as `integer` so that lower/upper bounds can be fixed to lock pre-scheduled AKs.

| Variable | Dimensions | Meaning |
|---|---|---|
| `Room[a, r]` | AK × Room | 1 if AK `a` is assigned to room `r` |
| `Time[a, t]` | AK × Timeslot | 1 if AK `a` uses timeslot `t` |
| `Block[a, b]` | AK × Block | 1 if AK `a` is placed in day-block `b` |
| `Part[p, a]` | AK × Person | 1 if person `p` attends AK `a` |
| `Working[p, t]` | Person × Timeslot | 1 if person `p` is occupied in timeslot `t` |

Variable bounds are tightened *before* the solver runs to prune the search space:
- `Time[a, t] = 0` if timeslot `t` violates a time-constraint of AK `a`.
- `Room[a, r] = 0` if room `r` fails a room-constraint of AK `a`.
- `Part[p, a] = 0` if person `p` has zero preference for AK `a` (and is not required).
- `Part[p, a] = 1` if person `p` is *required* for AK `a`.
- Pre-fixed AKs have their corresponding `Room`, `Time`, and `Part` entries locked to 1.

---

## Objective function

Maximise the total *normalised preference* of the schedule:

```
max  Σ_{p,a}  (pref[p,a] / num_prefs[p]) * Part[p,a]
```

where:
- `pref[p,a]` is the processed preference weight: 0 (none), 1 (weak), or `μ` (strong).
- `num_prefs[p]` normalises by each person's total number of preferences so that no one person dominates the objective.
- `μ` (default 2) is the hyperparameter that controls the trade-off between satisfying many weak preferences vs. fewer strong preferences.

Required participants (`required=true`) contribute 0 to the objective — their attendance is a **hard constraint**, not a preference.

---

## Hard constraints

### Scheduling feasibility

| Constraint name | Description |
|---|---|
| `AKDuration` | Each AK must use exactly `duration` timeslots |
| `AKSingleBlock` | Each AK is placed in at most one day-block |
| `AKBlockAssign` | The timeslots used by AK `a` must all belong to the block assigned to `a` |
| `AKContiguous` | The timeslots must be *consecutive* within their block (no gaps) |
| `AtMostOneRoomPerAK` | Each AK is assigned to at most one room |
| `AtLeastOneRoomPerAK` / `RoomForAK` | Each AK must have a room |

### Conflict-free scheduling

| Constraint name | Description |
|---|---|
| `MaxOneAKPerRoomAndTime` | No two AKs share a room at the same timeslot |
| `MaxOneAKPerPersonAndTime` | No person is scheduled to attend two AKs simultaneously |
| `AKConflict` | Explicitly conflicting AK pairs must not overlap in time |

### Dependencies

| Constraint name | Description |
|---|---|
| `AKDependenciesDoneBeforeAK` | If AK `a` depends on AK `b`, all timeslots of `b` must finish before the earliest timeslot of `a` starts |

### Room and time constraints

| Constraint name | Description |
|---|---|
| `RoomImpossibleForPerson` | Person `p` cannot be placed in room `r` if `r` doesn't fulfill `p`'s room constraints |
| `TimeImpossibleForRoom` | Room `r` cannot be used at timeslot `t` if `t` doesn't satisfy `r`'s time constraints |
| `Roomsize` | Attendee count must not exceed room capacity |

### Break constraints (optional)

| Constraint name | Description |
|---|---|
| `BreakForPerson` | No person is scheduled for more than `max_num_timeslots_before_break` consecutive slots in a block (disabled by default, `= 0`) |

### Auxiliary linking constraint

`TimePersonVar`:  `Time[a,t] + Part[p,a] - Working[p,t] ≤ 1`  
Links the `Working` variable so it is 1 whenever person `p` is active in timeslot `t`.

---

## Preprocessing in `ProblemProperties`

Before the MILP is built, `ProblemProperties.init_from_problem()` precomputes:

| Property | What it is |
|---|---|
| `conflict_pairs` | Set of `(ak_a, ak_b)` pairs that must not overlap (conflicts **and** dependencies both contribute) |
| `dependencies` | Dict `ak → [predecessor AKs]` |
| `preferences` | xarray `(ak × person)` with processed pref weights |
| `required_persons` | Boolean xarray `(ak × person)` |
| `ak_num_interested` | Number of interested + required persons per AK |
| `room_capacities` | Room capacity vector (−1 is replaced by total participant count) |
| `block_mask` | Boolean xarray `(block × timeslot)` for block membership |
| `{participant,ak,room}_time_constraints` | Boolean constraint membership matrices |
| `fulfilled_{time,room}_constraints` | Boolean fulfillment matrices for timeslots / rooms |

---

## Solver interface

`solve_scheduling()` in `solve.py`:

1. Calls `create_lp()` to build the linopy model.
2. Selects a solver: prefers `gurobi` → `highs` → any available linopy solver.
3. Calls `model.solve(...)` with solver-specific kwargs (`SolverConfig.generate_kwargs()`).
4. Returns the solved model and an `ExportTuple(room, time, person)` of solution arrays, or `None` if infeasible.

**Infeasibility:** If the model is infeasible, Gurobi can print the IIS (Irreducible Infeasible Subsystem); HiGHS cannot.

---

## Complexity notes

The number of constraints grows quadratically with the number of AKs because the pair-wise `MaxOneAKPerPersonAndTime` and `MaxOneAKPerRoomAndTime` loops iterate over all `O(|AK|²)` pairs. The `AKContiguous` and `AKDependencies` constraints also scale with AK count and block length.

For large conferences (≫40 AKs), solver time can be significant. Use `--timelimit` to get a good-enough schedule quickly:

```sh
akplan-solve input.json --timelimit 120 --gap-rel 0.05
```
