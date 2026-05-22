"""Nox task runner configuration for the akplan project.

Nox is a Python automation tool (similar to tox) that creates isolated virtual
environments for each session.  Run ``nox --list`` to see all sessions, and
``nox -s <session>`` to execute one.

Available sessions
------------------
test                 Run the default test suite (excludes the extensive set).
fast-test            Run only the fast test cases (excludes slow + extensive).
fast-unlicensed-test Run fast tests without any solver licence (used in CI).
extensive-test       Run the complete test suite including slow cases.
lint                 Check code style and conventions with ruff.
typing               Check type annotations with mypy.
format               Auto-fix formatting issues with ruff.
coverage             Measure test coverage and generate an HTML report.
coverage-clean       Delete the HTML coverage report.

Session design
--------------
Every test session installs the package with the ``[test]`` extra
(``pip install .[test]``) into a fresh virtual environment, then prints the
list of available solvers so that CI logs make it obvious which solver was
used.  This install step runs on every nox invocation — nox caches venvs
between runs so it is fast after the first time.
"""

import nox


def _setup_test_session(session):
    """Install the package with test dependencies and print available solvers.

    Shared by all test sessions to avoid duplicating the install logic.

    Args:
        session: The active nox ``Session`` object.

    Returns:
        The same ``session`` object (for chaining).
    """
    session.install(".[test]")
    # Print available solvers so CI logs are informative about which solver
    # will actually be used when ``solver_name=None`` is passed.
    session.run(
        "python",
        "-c",
        "import linopy; print('Available solvers:', linopy.solvers.available_solvers)",
    )
    return session


@nox.session(name="test")
def run_test(session):
    """Run pytest on all test cases besides the extensive suite."""
    session = _setup_test_session(session)
    session.run("pytest", "-m", "not extensive", *session.posargs)


@nox.session(name="fast-test")
def run_test_fast(session):
    """Run pytest on fast test cases."""
    session = _setup_test_session(session)
    session.run("pytest", "-m", "not slow and not extensive", *session.posargs)


@nox.session(name="fast-unlicensed-test")
def run_test_fast_unlicensed(session):
    """Run pytest on fast test cases without any license.

    This is the session used in CI (``pr-tests.yml``) where a Gurobi licence
    is not available.  The ``not licensed`` marker excludes all Gurobi-specific
    parameter combinations from the test matrix.
    """
    session = _setup_test_session(session)
    session.run(
        "pytest", "-m", "not slow and not extensive and not licensed", *session.posargs
    )


@nox.session(name="extensive-test")
def run_test_extensive(session):
    """Run pytest on all test cases."""
    session = _setup_test_session(session)
    session.run("pytest", *session.posargs)


@nox.session(name="lint")
def lint(session):
    """Check code conventions with ruff.

    Ruff rules are configured in ``pyproject.toml`` under ``[tool.ruff]``.
    This session only *checks* — use ``format`` to auto-fix issues.
    """
    session.install(".[lint]")
    session.run("ruff", "check", *session.posargs)


@nox.session(name="typing")
def mypy(session):
    """Check type annotations with mypy in strict mode.

    mypy is configured in ``pyproject.toml`` under ``[tool.mypy]``.
    Both the ``akplan`` package and the ``tests`` package are checked.
    """
    session.install(".[typing]")
    session.run("mypy", *session.posargs)


@nox.session(name="format")
def format(session):  # noqa: A001
    """Auto-fix formatting issues with ruff.

    Unlike ``lint``, this session modifies files in place.  Run it locally
    before committing to keep diffs clean.
    """
    session.install(".[format]")
    session.run("ruff", "format", *session.posargs)


@nox.session(name="coverage")
def check_coverage(session):
    """Measure test coverage and generate an HTML report in ``htmlcov/``.

    Runs the full test suite under ``coverage run`` and then generates an
    HTML report.  The report is always generated even if some tests fail
    (``try/finally``), so partial results are still visible.
    """
    session.install(".[coverage,test]")
    try:
        session.run("coverage", "run", "-m", "pytest", *session.posargs)
    finally:
        session.run("coverage", "html")


@nox.session(name="coverage-clean")
def clean_coverage(session):
    """Remove the HTML coverage report directory (``htmlcov/``)."""
    session.run("rm", "-r", "htmlcov", external=True)
