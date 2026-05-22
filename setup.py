"""Minimal setup shim required for editable installs (``pip install -e .``).

All real package metadata, dependencies, and build configuration live in
``pyproject.toml``.  This file exists only because some tooling (notably older
versions of pip) still expects a ``setup.py`` when performing an editable
install.  It delegates everything to ``setuptools.setup()`` with no arguments,
which reads ``pyproject.toml`` automatically.
"""

import setuptools

if __name__ == "__main__":
    setuptools.setup()
