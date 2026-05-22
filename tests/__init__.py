"""Test package for ``akplan``.

This ``__init__.py`` is intentionally minimal — its sole purpose is to make
``tests/`` a proper Python package so that ``mypy`` (configured with
``packages = ["akplan", "tests"]`` in ``pyproject.toml``) can type-check the
test files using the same strict settings as the source code.

All test logic lives in ``test_schedule_feasibility.py``.
"""
