"""Shared type contracts for the Shopping Copilot pipeline.

FROZEN as of CP 0.2 (Phase 0 - Contracts).

The five dataclasses below are the contract surface between Person A
(production implementation) and reviewers B / C / D. Their field sets AND
field types must not be added to, removed from, renamed, or retyped without
an approved INTERFACE CHANGE REQUEST -- see
``docs/interface_mutation_rule.md``. ``tests/test_contracts.py`` pins both
the field names/order and the field type strings of every type; an
unapproved change breaks that test on purpose.

Principles enforced by these shapes:

* ``SessionState`` is the single authoritative, deterministic state object.
  LLM / local-parser output (a delta) is never stored as state directly;
  deterministic Python applies it and mutates ``SessionState``.
* ``SessionState`` keeps evidence, not raw conversation history: structured
  ``slots``, persistent free-text ``evidence``, and ``provenance`` for every
  change. Evidence Confidence / Match Reliability are not frozen as
  top-level fields here; they live inside slot entries from Phase 11.
* Route-specific and strategy-specific knobs live in **generic containers**
  (``Candidate.route_scores``, ``Strategy.route_weights`` / ``Strategy.params``)
  so new retrieval routes and strategy parameters can be added in later
  phases without mutating this frozen contract. Candidate-scoped SCORE
  normalization is still forbidden; ``route_scores`` carries raw route scores.
* Every type is constructible with no arguments and is safe to use in that
  empty form. An explicit ``None`` passed for ANY field at construction is
  normalized to that field's declared default (CP 0.4 frozen None rule):
  ``""`` / ``0`` / ``"unknown"`` for scalars, a fresh empty container for
  dict/list fields, ``SessionState()`` for ``Context.state``. Construction
  never raises on ``None``; the resulting object always satisfies its own
  type contract.

This module defines data shapes only. It contains no retrieval, ranking, or
state-mutation behavior and is imported by no production code yet.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
from typing import Any

__all__ = [
    "SessionState",
    "Context",
    "Strategy",
    "Candidate",
    "RankingResult",
]


def _normalize_none_fields(instance: Any) -> None:
    """CP 0.4 frozen None rule.

    Any field left as ``None`` after construction is replaced by that
    field's declared default -- ``field.default`` for scalars (``""``,
    ``0``, ``"unknown"``) or ``field.default_factory()`` for containers and
    ``Context.state`` (a fresh empty ``dict`` / ``list`` / ``SessionState``).
    Passing ``None`` is a caller error; the contract neither raises nor
    invents an out-of-contract value -- it falls back to the same default
    the no-argument constructor would use. Runs at construction only;
    assigning ``None`` afterwards is out of scope.
    """
    for spec in fields(instance):
        if getattr(instance, spec.name) is None:
            if spec.default_factory is not MISSING:  # type: ignore[misc]
                setattr(instance, spec.name, spec.default_factory())
            elif spec.default is not MISSING:
                setattr(instance, spec.name, spec.default)


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

    def __post_init__(self) -> None:
        _normalize_none_fields(self)


@dataclass
class Context:
    """Per-turn snapshot handed to retrieval and ranking.

    Built fresh each turn from the current user message plus the current
    ``SessionState``. Downstream layers read ``Context``; they do not reach
    through it to mutate ``SessionState``. ``derived`` is a generic bag for
    per-turn computed inputs (accumulated query text, term lists, ...) whose
    exact keys later checkpoints define.
    """

    session_id: str = ""
    turn: int = 0
    user_message: str = ""
    state: SessionState = field(default_factory=SessionState)
    derived: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _normalize_none_fields(self)


@dataclass
class Strategy:
    """Adaptive strategy decision for the current turn (Phase 9).

    ``mode`` is ``"buying"``, ``"browsing"``, or ``"unknown"``. ``routes``
    is the ordered list of retrieval route names to run for the UNION.
    ``route_weights`` maps a route name to its relative weight for the
    union / ranking blend. ``params`` is a generic bag for any other
    strategy knob (thresholds, ask-decision inputs, ...) so new parameters
    do not require a contract mutation. The strategy layer never ranks
    individual products.
    """

    mode: str = "unknown"
    routes: list[str] = field(default_factory=list)
    route_weights: dict[str, float] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _normalize_none_fields(self)


@dataclass
class Candidate:
    """One retrieved product with per-route diagnostic scores.

    A product retrieved by several routes is a single ``Candidate`` whose
    ``route_scores`` holds one raw score per contributing route, keyed by
    route name (``"bm25"``, ``"category"``, ``"attribute"``, ``"dense"``,
    ...). New routes add a key; they never add a field. Route scores are
    raw -- never candidate-pool min-max normalized.

    ``metadata`` carries catalog fields / matched evidence needed for
    ranking and diagnostics. Missing catalog metadata is UNKNOWN, not a
    violation.

    ``route_sources`` is a derived read-only view (not a frozen field).
    """

    parent_asin: str = ""
    route_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _normalize_none_fields(self)

    @property
    def route_sources(self) -> tuple[str, ...]:
        """Routes that contributed this candidate, in insertion order.

        Derived from ``route_scores`` so it can never desync. Not part of
        the frozen field set.
        """
        return tuple(self.route_scores)


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

    def __post_init__(self) -> None:
        _normalize_none_fields(self)
