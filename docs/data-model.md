# Data Model & JSON Format

> **TL;DR:** The solver reads and writes plain JSON. The input describes AKs, rooms, participants, and timeslots with constraints. The output maps each AK to a room, timeslots, and attending participants.

The authoritative format specification lives in the [project wiki](https://github.com/Die-KoMa/ak-plan-optimierung/wiki/Input-&-output-format). This page is a developer-oriented summary.

---

## Input format

The top-level input JSON object has these keys:

```jsonc
{
  "aks":          [...],   // list of AK objects
  "rooms":        [...],   // list of Room objects
  "participants": [...],   // list of Participant objects
  "timeslots":    {...},   // timeslot info + blocks
  "scheduled_aks":[...],  // (optional) pre-fixed AK assignments
  "config":       {...},   // (optional) solver hyperparameters
  "info":         {...}    // (optional) free-form metadata
}
```

### AK object (`AKData`)

```jsonc
{
  "id": 3,                         // unique integer ID
  "duration": 2,                   // number of consecutive timeslots needed
  "room_constraints": ["Beamer"],  // room features this AK requires
  "time_constraints": ["ResoAK"],  // time-window labels this AK must be in
  "properties": {
    "conflicts":    [7, 12],       // AK IDs that must NOT overlap with this one
    "dependencies": [1]            // AK IDs that must finish BEFORE this one starts
  },
  "info": {                        // free-form, not used by solver
    "name": "AK Finanzen",
    "head": "Alice",
    "description": "…",
    "reso": false
  }
}
```

### Room object (`RoomData`)

```jsonc
{
  "id": 0,
  "capacity": 30,                         // max attendees; -1 = unlimited
  "fulfilled_room_constraints": ["Beamer", "Whiteboard"],
  "time_constraints": ["room0.003"],      // when this room is available
  "info": { "name": "Raum 0.003" }
}
```

### Participant object (`ParticipantData`)

```jsonc
{
  "id": 5,
  "room_constraints": [],          // accessibility / special needs
  "time_constraints": [],          // when this person is unavailable
  "preferences": [
    {
      "ak_id": 3,
      "preference_score": 2,       // 0=no pref, 1=weak, 2=strong, -1=required
      "required": false
    },
    {
      "ak_id": 7,
      "preference_score": -1,
      "required": true             // must attend – hard constraint
    }
  ],
  "info": { "name": "Bob" }
}
```

**Preference scores:**

| `preference_score` | `required` | Meaning | MILP weight |
|---|---|---|---|
| `0` | `false` | Not interested | 0 |
| `1` | `false` | Weakly interested | 1 |
| `2` | `false` | Strongly interested | μ (default 2) |
| `-1` | `true` | Required to attend | 0 (hard constraint) |

### Timeslots object

```jsonc
{
  "info": { "duration": "1 Stunde" },   // free-form metadata
  "blocks": [                            // one sub-list per day / block
    [
      {
        "id": 0,
        "fulfilled_time_constraints": ["Dienstag", "ResoAK"],
        "info": { "start": "Dienstag, 8 Uhr" }
      },
      { "id": 1, "fulfilled_time_constraints": ["Dienstag"], … },
      …
    ],
    [ … ]  // next day
  ]
}
```

AKs must be scheduled in **one block only** and in **consecutive slots** within that block. Cross-block scheduling is not allowed.

### Pre-fixed AKs (`scheduled_aks`)

An optional list of assignments that the solver must honour (room changes allowed via `config.allow_changing_rooms`):

```jsonc
[
  {
    "ak_id": 3,
    "room_id": 1,
    "timeslot_ids": [4, 5],
    "participant_ids": [0, 7, 12]
  }
]
```

### Config object (`ConfigData`)

```jsonc
{
  "mu": 2,                             // weight for strong preferences (default 2)
  "max_num_timeslots_before_break": 0, // 0 = no break constraint (default)
  "allow_unscheduled_aks": true,       // allow solver to skip AKs (default true)
  "allow_changing_rooms": false        // may solver move pre-fixed AKs? (default false)
}
```

---

## Output format

```jsonc
{
  "scheduled_aks": [
    {
      "ak_id": 3,
      "room_id": 1,
      "timeslot_ids": [4, 5],
      "participant_ids": [0, 7, 12]
    },
    …
  ],
  "input": { … }   // echo of the full input (for traceability)
}
```

---

## Python data-classes (`util.py`)

The JSON is deserialised into a hierarchy of frozen dataclasses using `dacite.from_dict`:

```
SchedulingInput
├── aks:            list[AKData]
├── participants:   list[ParticipantData]
│   └── preferences: list[PreferenceData]
├── rooms:          list[RoomData]
├── timeslot_blocks: list[list[TimeSlotData]]
├── scheduled_aks:  list[ScheduleAtom]
├── config:         ConfigData
└── info:           dict[str, str]
```

`ScheduleAtom` — used both for pre-fixed input AKs and the solver's output — stores `ak_id`, `room_id`, `timeslot_ids` (numpy int64 array), and `participant_ids` (numpy int64 array).

All entity IDs (`AkId`, `RoomId`, `PersonId`, `TimeslotId`) are plain `int` type aliases defined in `types.py`.

---

## Constraint resolution

Constraints are **tag-based**: AKs and participants declare which tags they *require*; rooms and timeslots declare which tags they *fulfill*. The solver ensures every requirement is covered:

| Requires tags | Fulfilled by |
|---|---|
| `AKData.room_constraints` | `RoomData.fulfilled_room_constraints` |
| `ParticipantData.room_constraints` | `RoomData.fulfilled_room_constraints` |
| `AKData.time_constraints` | `TimeSlotData.fulfilled_time_constraints` |
| `ParticipantData.time_constraints` | `TimeSlotData.fulfilled_time_constraints` |
| `RoomData.time_constraints` | `TimeSlotData.fulfilled_time_constraints` |
