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

WHAT THIS GUARD DOES NOT COVER, STATED SO THE NEXT PHASE PLANS FOR IT

Prose. Every check here is an ASSERTION about a value in memory or in an AST,
and the numbers that go stale in this repository are mostly in comments. Phase
14 moved the popularity ablation (``ranking.py``: OFF 0.134566 -> 0.187711,
McNemar 10/0 -> 13/0, the whole W_POPULARITY sweep), the cap-loss pair in
``retrieval.py``, and the conversion ratio in ``phase13_dense_gate``'s
docstring -- and every tool still exited 0, because not one of those numbers
is asserted anywhere (D Phase 14 review, Finding 3). The amended "re-run every
tool" method catches a stale assertion and is blind to a stale sentence.

There is no mechanical fix here that is not a registry of literals, which is
the maintained-expectation shape this module exists to avoid. So the
convention instead: any comment quoting a number that MOVES WITH THE PIPELINE
names the tool that regenerates it, and a checkpoint's method includes reading
those blocks. The three sites above now say so in as many words.

Usage::

    from tools import config_guard

    config_guard.assert_committed_constants()
    config_guard.assert_all_flags_pinned(MY_FLAGS)
    ...
    config_guard.restore_committed_flags()
"""

from __future__ import annotations

import ast
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
    "clarify",
    "contracts",
    "popularity",
    "profile",
    "ranking",
    "reliability",
    "reranker",
    "retrieval",
    "state",
    "strategy",
    "text",
    "vocabulary",
)

FLAG_PREFIX = "USE_"

# The committed values of every ablation flag, wherever it lives. A run may
# deviate from these deliberately; ``restore_committed_flags`` puts them back.
COMMITTED_FLAGS: dict[tuple[str, str], Any] = {
    ("agent", "USE_STATE"): True,
    ("agent", "USE_MULTI_ROUTE"): True,
    ("agent", "USE_CONSTRAINT_RANKING"): True,
    ("ranking", "USE_PROFILE"): False,
    ("ranking", "USE_CONFIDENCE_WEIGHTING"): False,
    ("ranking", "USE_POPULARITY"): True,
    # Phase 14. Note the module: `agent.respond` reads this through
    # `starter.reranker`, not through a name imported into `agent`, so that
    # `set_flag` actually reaches the branch it governs.
    ("reranker", "USE_SEMANTIC_RERANK"): True,
    # Phase 15. Same note as the reranker above: `agent.respond` reads this
    # through `starter.clarify`, so `set_flag` reaches the branch it governs.
    ("clarify", "USE_CLARIFICATION"): True,
}

# Not flags, but they decide what gets measured just as hard. ``DEFAULT_ROUTES``
# is the whole point: it determines every row of the union pool.
COMMITTED_CONSTANTS: dict[tuple[str, str], Any] = {
    ("retrieval", "DEFAULT_ROUTES"): ("bm25", "category"),
    ("retrieval", "POOL_LIMIT"): 300,
    ("ranking", "W_MATCH"): 0.10,
    ("ranking", "W_PENALTY"): 0.02,
    # Not flagged by any reviewer -- found by the completeness check below,
    # which is the point of having one.
    ("ranking", "W_PROFILE"): 0.008,
    # Phase 10. Registered with the phase, not after it (D-V2): the first two
    # of these are values a measurement REJECTED -- a 0.5 ratio made
    # most_discriminative degenerate and a 400 cap silently discarded 31% of
    # the vocabulary -- so leaving them unpinned meant the tree could return
    # to them with every check still passing.
    ("vocabulary", "MAX_DOCUMENT_RATIO"): 0.8,
    ("vocabulary", "VOCABULARY_LIMIT"): 1500,
    ("vocabulary", "MIN_MAPPED_SUPPORT"): 0.05,
    ("vocabulary", "MIN_DOCUMENT_FREQUENCY"): 2,
    ("vocabulary", "NOISE_FLOOR_POOL"): 8,
    ("vocabulary", "INDEX_TERM_LIMIT"): 40,
    # Phase 11.
    ("reliability", "DEFAULT_RELIABILITY"): 1.0,
    ("reliability", "MIN_RELIABILITY"): 0.1,
    ("state", "EC_REQUIREMENT"): 1.0,
    ("state", "EC_CORRECTION"): 0.9,
    ("state", "EC_STATED"): 0.7,
    ("state", "EC_HEDGED"): 0.4,
    # Phase 12.
    ("popularity", "W_POPULARITY"): 0.008,
    ("popularity", "DEFAULT_SCALE"): 13.0,
    ("popularity", "DEFAULT_MISSING"): 0.5,
    # Phase 14. The reranker's window and its latency budget both decide what
    # gets measured: the window bounds which candidates the stage can even
    # reach, and the budget decides whether a slow scorer's answer counts.
    ("reranker", "RERANK_TOP_N"): 50,
    ("reranker", "RERANK_BUDGET_MS"): 150.0,
    # Phase 15. The window a question's value is judged over, the quantile
    # grid for the one continuous attribute, and the bar a specific question
    # must clear to be preferred to the open one. All three decide WHICH
    # question gets asked, and the answer is what the rest of the session is
    # built on -- so they move the measured score harder than any weight in
    # this registry.
    ("clarify", "ASK_POOL_DEPTH"): 50,
    ("clarify", "PRICE_BUCKETS"): 4,
    ("clarify", "ASK_VALUE_FLOOR"): 0.10,
}


# The public-set TechnicalScore the committed configuration produces. Not a
# constant OF the starter package -- an outcome of it -- but it belongs in this
# registry all the same, because several tools open by asserting "my capture
# reproduced the shipped run" and they need one place to check against.
#
# It is here because three tools each pinned their own copy of 0.134566 and two
# of them went stale the moment Phase 12 moved the score to 0.182258:
# ``phase10_vocabulary`` and ``phase11_confidence`` have been exiting with
# "the vocabulary layer changed the score" / "OFF no longer reproduces the
# committed score" ever since -- a guard failing for the one reason a guard
# must never fail, its own staleness. (``phase12_popularity`` still pins
# 0.134566 on purpose: that is the score of its USE_POPULARITY=False arm, a
# historical reference rather than the committed one.)
#
# Moving the committed score means updating this line in the same change.
COMMITTED_TECHNICAL_SCORE = 0.726726


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


def discover_numeric_constants() -> set[tuple[str, str]]:
    """Every module-level numeric tunable defined in the ``starter`` package.

    Read from the SOURCE rather than the imported module, so a re-export
    (``agent`` imports ``POOL_LIMIT`` from ``retrieval``) is attributed to the
    module that actually defines it and is not demanded twice.

    Numeric on purpose. A float or int assigned at module level is a knob --
    a threshold, a weight, a limit -- and is the class of thing that changes
    measured numbers when it moves. The string vocabularies (``SOFT_CUES``,
    ``GROUNDING``, ``BOILERPLATE``) are content, not configuration; pinning
    them here would mean re-approving the registry on every word added, which
    is how a guard becomes noise and then gets ignored.

    Only literal assignments are found. A computed constant would be missed;
    none exist today, and one would be worth a comment saying why.
    """
    found: set[tuple[str, str]] = set()
    for module_name in STARTER_MODULES:
        source = Path(_module(module_name).__file__).read_text(encoding="utf-8")
        for node in ast.parse(source).body:
            if not isinstance(node, ast.Assign):
                continue
            value = node.value
            if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub):
                value = value.operand
            if not (isinstance(value, ast.Constant)
                    and isinstance(value.value, (int, float))
                    and not isinstance(value.value, bool)):
                continue
            for target in node.targets:
                if (isinstance(target, ast.Name)
                        and target.id.isupper()
                        and not target.id.startswith("_")):
                    found.add((module_name, target.id))
    return found


def source_flags() -> dict[tuple[str, str], Any]:
    """Every ``USE_`` flag's value as WRITTEN IN THE SOURCE.

    Not as currently held in memory. The two differ whenever anything has
    assigned to a flag at runtime -- a measurement tool mid-sweep, or a test
    that called ``restore_committed_flags``.

    That difference is why this exists (D Phase 12 review, Q2). A test that
    tried to pin a committed flag value by reading the live module could be
    silently defeated by any earlier test in the same process writing the
    committed value back over a source edit, which is exactly what happened:
    the Phase 11/12 interaction guard passed in the full suite and failed
    only when run alone. Reading the file cannot be undone by another test.
    """
    found: dict[tuple[str, str], Any] = {}
    for module_name in STARTER_MODULES:
        source = Path(_module(module_name).__file__).read_text(encoding="utf-8")
        for node in ast.parse(source).body:
            if not isinstance(node, ast.Assign):
                continue
            if not (isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, bool)):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.startswith(FLAG_PREFIX):
                    found[(module_name, target.id)] = node.value.value
    return found


def assert_committed_flags_match_source() -> None:
    """Raise if ``COMMITTED_FLAGS`` disagrees with what the modules declare.

    The registry says what the committed configuration IS; the source is that
    configuration. Letting them drift would make every "restored to committed
    values" claim in this package unverifiable.
    """
    declared = source_flags()
    drifted = [
        f"{module}.{name}: source {declared[(module, name)]!r} != "
        f"COMMITTED_FLAGS {value!r}"
        for (module, name), value in sorted(COMMITTED_FLAGS.items())
        if (module, name) in declared and declared[(module, name)] != value
    ]
    if drifted:
        raise SystemExit("config_guard: " + "; ".join(drifted))


def assert_all_constants_pinned() -> None:
    """Raise if a numeric tunable exists that ``COMMITTED_CONSTANTS`` omits.

    D-V2: Phase 10 shipped six new thresholds and none were registered --
    including the two a measurement had just REJECTED, so the tree could have
    returned to a 0.5 document ratio and a 400-term cap with every check still
    green. That is the fourth appearance of one shape (agent-only scan, then
    the module list, then the module list's completeness, now the constant
    registry's), so it is fixed the same way: by deriving the expectation
    instead of maintaining it.
    """
    unpinned = sorted(discover_numeric_constants() - set(COMMITTED_CONSTANTS))
    if unpinned:
        raise SystemExit(
            "config_guard: these numeric constants are not pinned by "
            "COMMITTED_CONSTANTS: "
            + ", ".join(f"{module}.{name}" for module, name in unpinned)
            + ". Add each with its committed value, so that moving it later "
            "fails loudly instead of silently changing every measured number."
        )


def assert_committed_constants() -> None:
    """Raise if a non-flag constant has drifted from its committed value.

    Prevents the reported numbers from silently describing a different
    configuration than the one the comments in ``starter/`` justify.
    """
    assert_all_constants_pinned()
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
