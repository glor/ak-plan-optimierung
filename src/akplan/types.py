"""Type aliases, TypedDicts and NamedTuples shared across the ``akplan`` package.

This module contains **only type-level definitions** — no executable business
logic.  Keeping them in a dedicated module avoids circular imports between
``util`` and ``solve``, and makes it easy to find the canonical meaning of any
ID or data-structure used elsewhere.

Relationship to other modules
------------------------------
- ``util`` imports every ID alias and ``ExportTuple`` from here.
- ``solve`` imports ``ExportTuple``, ``SupportedSolver`` and the solver-kwargs
  TypedDicts from here.
- ``types`` itself imports nothing from the rest of the package.
"""

from pathlib import Path
from typing import Literal, NamedTuple, NotRequired, TypedDict, TypeVar

import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# Integer ID aliases
# ---------------------------------------------------------------------------
# All entity IDs in the JSON input are plain integers.  These aliases add
# semantic meaning so that type-checkers and readers can distinguish, for
# example, an AkId argument from a RoomId argument even though both are `int`.

Id = int
T = TypeVar("T")
IdType = TypeVar("IdType", bound=Id)
IdType2 = TypeVar("IdType2", bound=Id)

RoomId = Id
PersonId = Id
AkId = Id
TimeslotId = Id
BlockId = Id

# A Block is a pandas Index of TimeslotIds representing one contiguous day-block
# (e.g. all slots on Tuesday).  AKs must be scheduled entirely within one block.
Block = pd.Index

# ---------------------------------------------------------------------------
# ScheduleAtomComparisonTuple
# ---------------------------------------------------------------------------
# A hashable, sortable representation of a scheduled AK used for equality
# checks and set operations in ``calc_changed_fixed_schedule_atoms``.
# Fields: (ak_id, room_id | None, sorted timeslot IDs, sorted participant IDs)
ScheduleAtomComparisonTuple = tuple[
    AkId,
    RoomId | None,
    tuple[TimeslotId, ...],
    tuple[PersonId, ...],
]


# ---------------------------------------------------------------------------
# ExportTuple
# ---------------------------------------------------------------------------
class ExportTuple(NamedTuple):
    """Named tuple holding the three solution arrays after the MILP is solved.

    Each field is an ``xr.DataArray`` of rounded binary values (0.0 or 1.0)
    taken directly from the linopy solution.

    Attributes:
        room: Shape ``(ak × room)``.  ``room[a, r] == 1`` means AK ``a``
            is assigned to room ``r``.
        time: Shape ``(ak × timeslot)``.  ``time[a, t] == 1`` means AK ``a``
            occupies timeslot ``t``.
        person: Shape ``(ak × person)``.  ``person[a, p] == 1`` means
            person ``p`` attends AK ``a``.
    """

    room: xr.DataArray
    time: xr.DataArray
    person: xr.DataArray


# ---------------------------------------------------------------------------
# Solver support
# ---------------------------------------------------------------------------

# Solvers for which ``SolverConfig.generate_kwargs`` knows how to translate
# the generic CLI arguments (time limit, gap, threads) into solver-specific
# keyword names.  Any other linopy-supported solver can still be used, but
# will run with its own defaults.
SupportedSolver = Literal["gurobi", "highs"]


class SolverKwargs(TypedDict, total=False):
    """Base keyword arguments accepted by any linopy solver.

    These two keys are common across all solvers.  Solver-specific subclasses
    add their own performance-tuning keys.
    """

    warmstart_fn: NotRequired[str | Path | None]
    io_api: Literal["direct", "lp", "mps"]


class GurobiSolverKwargs(SolverKwargs):
    """Keyword arguments for the Gurobi linopy solver.

    Key names follow the Gurobi parameter convention (CamelCase).
    See https://www.gurobi.com/documentation/current/refman/parameters.html
    """

    TimeLimit: NotRequired[float]  # noqa: N815
    MIPGap: NotRequired[float]  # noqa: N815
    MIPGapAbs: NotRequired[float]  # noqa: N815
    Threads: NotRequired[int]  # noqa: N815


class HighsSolverKwargs(SolverKwargs):
    """Keyword arguments for the HiGHS linopy solver.

    Key names follow the HiGHS option convention (snake_case).
    See https://ergo-code.github.io/HiGHS/dev/options/definitions/
    """

    time_limit: NotRequired[float]
    mip_rel_gap: NotRequired[float]
    mip_abs_gap: NotRequired[float]
    threads: NotRequired[int]
