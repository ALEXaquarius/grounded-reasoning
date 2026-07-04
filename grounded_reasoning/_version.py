"""
Single source of truth for the package version.

Kept free of any other import so build tooling can read `__version__` via
static analysis (pyproject.toml's `[tool.setuptools.dynamic]` attr directive)
without executing the rest of the package or requiring its dependencies to be
installed yet.
"""
__version__ = "0.1.3"
