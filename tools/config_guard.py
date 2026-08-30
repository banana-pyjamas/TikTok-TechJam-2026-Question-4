"""Configuration guard shared by every measurement tool.

A measured number is only worth as much as the configuration it was measured
under. This module makes that configuration explicit and FAILS LOUDLY when it
is not, so a tool cannot quietly report a number produced by a setting the
tool never pinned.

It exists because the first version of the guard lived inside
``tools/phase7_ablation.py`` and scanned ``starter.agent`` alone (D-N2). A new
``USE_`` flag in ``agent.py`` tripped it; the same flag in ``ranking.py`` did
not, and ``retrieval.DEFAULT_ROUTES`` -- which decides every row of the union
pool -- was not pinned at all. Changing it swings the measured score by up to
0.019 while the ablation's Run-0 baseline check still passes. That is the
exact failure mode the guard was added to prevent, one module over.

Three things are guarded:

the module list       ``STARTER_MODULES`` is checked against the package
                      directory. A hand-maintained list is the same failure
                      mode one level up: a new ``starter/reranker.py`` with a
                      ``USE_SEMANTIC_RERANK`` flag would be silent, because
                      nothing would look in it (D-P1). The roadmap adds a
                      semantic reranker at Phase 13/14, so this is a live risk,
                      not a hypothetical one.
``USE_`` flags        discovered across the WHOLE ``starter`` package. Every
                      one must appear in the caller's pinned set; an
                      unregistered flag raises.
committed constants   values that are not flags but still determine what is
                      measured. Checked against ``COMMITTED_CONSTANTS`` before
                      any run. This module never MUTATES a constant, so there
                      is nothing for it to restore -- a tool that changes one
                      (``tools/phase9_retrieval_evidence.py`` swaps the route
                      table) restores it itself (D-P4).

Usage::

    from tools import config_guard

    config_guard.assert_committed_constants()
    config_guard.assert_all_flags_pinned(MY_FLAGS)
    ...
    config_guard.restore_committed_flags()
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Iterable

import starter

# Every module of the shipped package. Listed rather than auto-walked so that
# adding a module is a deliberate act that shows up in review -- but the list
# is CHECKED against the directory (``assert_module_list_is_complete``), so
# "deliberate" cannot degrade into "forgotten".
STARTER_MODULES = (
    "agent",
    "catalog_meta",
    "contracts",
    "profile",
    "ranking",
    "retrieval",
    "state",
    "strategy",
    "text",
)

FLAG_PREFIX = "USE_"

# The committed values of every ablation flag, wherever it lives. A run may
# deviate from these deliberately; ``restore_committed_flags`` puts them back.
COMMITTED_FLAGS: dict[tuple[str, str], Any] = {
    ("agent", "USE_STATE"): True,
    ("agent", "USE_MULTI_ROUTE"): True,
    ("agent", "USE_CONSTRAINT_RANKING"): True,
    ("ranking", "USE_PROFILE"): False,
}

# Not flags, but they decide what gets measured just as hard. ``DEFAULT_ROUTES``
# is the whole point: it determines every row of the union pool.
COMMITTED_CONSTANTS: dict[tuple[str, str], Any] = {
    ("retrieval", "DEFAULT_ROUTES"): ("bm25", "category"),
    ("retrieval", "POOL_LIMIT"): 300,
    ("ranking", "W_MATCH"): 0.10,
    ("ranking", "W_PENALTY"): 0.02,
}


def _module(name: str):
    return importlib.import_module(f"starter.{name}")


def assert_module_list_is_complete() -> None:
    """Raise if ``STARTER_MODULES`` has drifted from the package directory.

    This is the guard on the guard. Scoping it to ``starter.agent`` was the
    first version of this failure (D-N2); a hand-maintained module list is the
    second (D-P1). Reading the directory removes the third, because a module
    that exists but is unlisted can no longer hide a flag.
    """
    package = Path(starter.__file__).resolve().parent
    on_disk = {
        path.stem for path in package.glob("*.py")
        if path.stem != "__init__"
    }
    listed = set(STARTER_MODULES)
    problems = []
    if on_disk - listed:
        problems.append(
            "not listed in STARTER_MODULES, so any USE_ flag in them is "
            "invisible to this guard: " + ", ".join(sorted(on_disk - listed))
        )
    if listed - on_disk:
        problems.append(
            "listed but no longer on disk: " + ", ".join(sorted(listed - on_disk))
        )
    if problems:
        raise SystemExit("config_guard: " + "; ".join(problems))


def discover_flags() -> set[tuple[str, str]]:
    """Every module-level ``USE_`` flag in the whole ``starter`` package.

    Checks the module list first, so a flag cannot hide in a module nobody
    remembered to add.
    """
    assert_module_list_is_complete()
    found: set[tuple[str, str]] = set()
    for module_name in STARTER_MODULES:
        module = _module(module_name)
        for attribute in dir(module):
            if attribute.startswith(FLAG_PREFIX):
                found.add((module_name, attribute))
    return found


def assert_all_flags_pinned(pinned: Iterable[tuple[str, str]]) -> None:
    """Raise unless the caller pins every ``USE_`` flag that exists.

    Package-wide on purpose: a flag added to ``ranking.py`` or ``state.py``
    must break the tool exactly as loudly as one added to ``agent.py``.
    """
    pinned_set = set(pinned)
    unpinned = sorted(discover_flags() - pinned_set)
    if unpinned:
        raise SystemExit(
            "config_guard: these USE_ flags are not pinned by this tool: "
            + ", ".join(f"{module}.{name}" for module, name in unpinned)
            + ". Add each one to the tool's flag set AND to every run's "
            "configuration before trusting any number it prints."
        )
    stale = sorted(pinned_set - discover_flags())
    if stale:
        raise SystemExit(
            "config_guard: these pinned flags no longer exist: "
            + ", ".join(f"{module}.{name}" for module, name in stale)
        )


def assert_committed_constants() -> None:
    """Raise if a non-flag constant has drifted from its committed value.

    Prevents the reported numbers from silently describing a different
    configuration than the one the comments in ``starter/`` justify.
    """
    drifted = []
    for (module_name, name), expected in sorted(COMMITTED_CONSTANTS.items()):
        actual = getattr(_module(module_name), name)
        if actual != expected:
            drifted.append(f"{module_name}.{name}: {actual!r} != {expected!r}")
    if drifted:
        raise SystemExit(
            "config_guard: committed constants have drifted -- "
            + "; ".join(drifted)
            + ". Update COMMITTED_CONSTANTS in the same change that moves "
            "them, and re-measure every number that depends on them."
        )


def set_flag(module_name: str, name: str, value: Any) -> None:
    setattr(_module(module_name), name, value)


def restore_committed_flags() -> None:
    """Put every FLAG back to its committed value.

    Flags only. Constants are never mutated by this module, so there is
    nothing here to restore; a tool that swaps one restores it itself.
    """
    for (module_name, name), value in COMMITTED_FLAGS.items():
        set_flag(module_name, name, value)


def describe() -> str:
    """One line naming the configuration the numbers were produced under."""
    parts = [
        f"{module}.{name}={getattr(_module(module), name)!r}"
        for module, name in sorted(COMMITTED_FLAGS)
    ]
    parts += [
        f"{module}.{name}={getattr(_module(module), name)!r}"
        for module, name in sorted(COMMITTED_CONSTANTS)
    ]
    return ", ".join(parts)
