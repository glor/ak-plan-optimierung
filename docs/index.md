# akplan — Developer Documentation

> **TL;DR:** Conference schedule optimizer. Takes a JSON of AKs, participants, rooms, and timeslots → solves a MILP → outputs an optimal schedule as JSON.

## Contents

| Document | What you'll find |
|---|---|
| [Overview](overview.md) | What the project is, the high-level workflow, package structure, and key dependencies |
| [Data Model & JSON Format](data-model.md) | Input/output JSON schema, all fields explained, Python dataclass hierarchy |
| [MILP Formulation](algorithm.md) | Decision variables, objective function, every hard constraint, solver interface |
| [Development Guide](development.md) | Setup, CLI usage, Python API, test-input generator, nox sessions, test suite |

## Quick start

```sh
# Install
pip install git+https://github.com/Die-KoMa/ak-plan-optimierung.git

# Solve
akplan-solve examples/koma92.json

# Output written to ./out-koma92.json
```

## Concepts at a glance

- **AK** (*Arbeitskreis*) — a conference session/workshop with a required duration and optional room/time constraints.
- **Timeslot block** — a group of consecutive timeslots (typically one day). AKs must be scheduled entirely within one block.
- **Constraint tag** — a string label. AKs/participants *require* certain tags; rooms/timeslots *fulfill* them. The solver matches requirements to fulfillments.
- **μ (mu)** — the preference weight hyperparameter. Strong preferences (`score=2`) count as μ times a weak preference (`score=1`). Default: 2.
- **Required participant** — `preference_score=-1, required=true`. Hard constraint: this person *must* attend the AK.
