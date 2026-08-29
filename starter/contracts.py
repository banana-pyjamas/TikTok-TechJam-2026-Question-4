"""Shared type contracts for the Shopping Copilot pipeline.

FROZEN as of CP 0.2 (Phase 0 - Contracts).

The five dataclasses below are the contract surface between Person A
(production implementation) and reviewers B / C / D. Their field sets must
not be added to, removed from, renamed, or retyped without an approved
INTERFACE CHANGE REQUEST -- see ``docs/interface_mutation_rule.md``.
``tests/test_contracts.py`` pins the exact field set of every type; an
unapproved change breaks that test on purpose.

Principles enforced by these shapes:

* ``SessionState`` is the single authoritative, deterministic state object.
  LLM / local-parser output (a delta) is never stored as state directly;
  deterministic Python applies it and mutates ``SessionState``.
* ``SessionState`` keeps evidence, not raw conversation history: structured
  ``slots``, persistent free-text ``evidence``, and ``provenance`` for every
  change. Evidence Confidence / Match Reliability are not frozen as
  top-level fields here; they live inside slot entries from Phase 11.
* ``Candidate`` keeps each route's score separate so the retrieval UNION can
  deduplicate without collapsing provenance. Candidate-scoped SCORE
  normalization is forbidden; these fields carry raw route scores.
* Every type is constructible with no arguments and is safe to use in that
  empty form (CP 0.4).

This module defines data shapes only. It contains no retrieval, ranking, or
state-mutation behavior and is imported by no production code yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "SessionState",
    "Context",
    "Strategy",
    "Candidate",
    "RankingResult",
]


@dataclass
class SessionState:
    """Authoritative per-session conversational state.

    Mutated only by the deterministic state manager (Phase 2+).

    Fields
    ------
    session_id:
        Opaque id supplied by the evaluator at ``reset``.
    user_profile:
        Anonymized profile dict from ``reset``. A prior only; it must never
        override an explicit current request.
    turn:
        1-based index of the most recently processed turn (0 before any).
    slots:
        Structured active constraints keyed by slot name
        (``category``, ``color``, ``material``, ``size``, ``brand``,
        ``budget``, ``style``, ``use_case``, ``feature``, ...). Value shape
        is defined per-slot by later checkpoints.
    evidence:
        Persistent free-text evidence items not (yet) promoted to a slot,
        e.g. ``"gift for my dad"``. Retained across turns.
    provenance:
        Append-only log of state changes; each entry records at least the
        turn, slot, operation, and source that produced the change. Used to
        prevent stale-evidence resurrection.
    """

    session_id: str = ""
    user_profile: dict[str, Any] = field(default_factory=dict)
    turn: int = 0
    slots: dict[str, Any] = field(default_factory=dict)
    evidence: list[Any] = field(default_factory=list)
    provenance: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Context:
    """Per-turn snapshot handed to retrieval and ranking.

    Built fresh each turn from the current user message plus the current
    ``SessionState``. Downstream layers read ``Context``; they do not reach
    through it to mutate ``SessionState``.
    """

    session_id: str = ""
    turn: int = 0
    user_message: str = ""
    state: "SessionState" = field(default_factory=lambda: SessionState())


@dataclass
class Strategy:
    """Adaptive strategy decision for the current turn (Phase 9).

    ``mode`` is ``"buying"``, ``"browsing"``, or ``"unknown"``. ``routes``
    is the ordered list of retrieval route names to run for the UNION. The
    strategy layer never ranks individual products.
    """

    mode: str = "unknown"
    routes: list[str] = field(default_factory=list)


@dataclass
class Candidate:
    """One retrieved product with per-route diagnostic scores.

    A product retrieved by several routes is a single ``Candidate`` whose
    ``route_sources`` lists every contributing route. Route scores stay
    separate; they are raw (never candidate-pool min-max normalized).

    ``metadata`` carries catalog fields / matched evidence needed for
    ranking and diagnostics. Missing catalog metadata is UNKNOWN, not a
    violation.
    """

    parent_asin: str = ""
    route_sources: list[str] = field(default_factory=list)
    bm25_score: float = 0.0
    category_score: float = 0.0
    attribute_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RankingResult:
    """Output of the ranking stage.

    ``ranked`` is the candidate list ordered best-to-worst. ``diagnostics``
    is keyed by ``parent_asin`` and, from Phase 6, exposes base score,
    attribute contribution, violation penalty, popularity prior, final
    score, and rank. Clarification wording is NOT owned here; the
    clarification layer (Phase 15) adds it.
    """

    ranked: list[Candidate] = field(default_factory=list)
    diagnostics: dict[str, dict[str, Any]] = field(default_factory=dict)
