"""Multi-route retrieval over the FTS index.

Three deterministic routes, combined by a UNION rather than an intersection:

  bm25       full-text match on the raw user message
  category   BM25 on the accumulated query, restricted to the category slot
  attribute  products mentioning any color / material / brand / size slot value

Pool order is Reciprocal Rank Fusion: each route votes with ``1 / (K + rank)``,
so a candidate near the head of any route outranks one buried deep in another,
and a candidate found by several routes rises above one found by a single
route. Fusion is rank-based, never score-based -- no candidate-pool score
normalization. Ties break on best per-route rank, then route priority, then
``parent_asin``.

Missing catalog metadata reduces how many routes surface a product but never
eliminates it; no route hard-filters on a field being present. Route-scoped
filtering is allowed -- an out-of-category product still reaches the pool
through the BM25 route.

Retrieval never mutates ``SessionState``; it reads ``context.state`` only.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from typing import Iterable

from starter.contracts import Candidate, Context
from starter.text import terms

# Column weights: parent_asin, title, categories, features, details, store,
# description.
_BM25_RANK = "bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)"
_ATTRIBUTE_SLOTS = ("color", "material", "brand", "size")
# Deterministic tie-break order when two candidates fuse to the same score.
_ROUTE_ORDER = ("bm25", "category", "attribute")
# RRF damping. Rank 1 of any route (1/61) outranks rank 61+ of another, which
# is what gives the auxiliary routes real access to the pool.
_RRF_K = 60

POOL_LIMIT = 300

# The routes the agent runs, chosen on candidate recall of the hidden target.
# The category route gains recall at every depth tested and loses at none. The
# attribute route gains at none -- it re-ranks on terms BM25 already covers,
# and its unique candidates arrive too far down to help -- so it is excluded.
# `python3 -m tools.phase9_retrieval_evidence` regenerates the comparison.
DEFAULT_ROUTES = ("bm25", "category")

# A dense route is not implemented, for two separate reasons.
#
# Measured: the strongest vector retriever available under the submission
# constraints is lexical TF-IDF cosine, and fusing it in makes the pool WORSE
# -- established negative at recall@50. It adds unique candidates but displaces
# better ones at the cap. `python3 -m tools.phase13_dense_gate` reproduces this
# and should be re-run after any change to ranking or to the dialogue.
#
# Packaging: nothing is vendored here, the organizer may disable network access
# for scoring, the submission is certified on a Python version current torch
# builds do not support, and this package is standard-library only by design.
# None of that is evidence that a trained encoder would not help -- it is a
# statement about what can ship. That question belongs in the reranker's scorer
# interface, which is where an encoder would land.


def _fusion_priority(names: Iterable[str]) -> list[str]:
    """Deterministic fusion order over whatever routes were actually run.

    ``_ROUTE_ORDER`` first, then any other route name alphabetically. The tail
    matters: iterating ``_ROUTE_ORDER`` alone would silently discard results
    from a route not named in it, returning a pool that looks correct and is
    missing a route's candidates.
    """
    known = [name for name in _ROUTE_ORDER if name in names]
    return known + sorted(set(names) - set(_ROUTE_ORDER))


def _fts_or(tokens: list[str]) -> str:
    unique = list(dict.fromkeys(token for token in tokens if len(token) > 1))
    return " OR ".join(f'"{token}"' for token in unique[:40])


def _stem(token: str) -> str:
    return token[:-1] if len(token) > 3 and token.endswith("s") else token


def _run(connection: sqlite3.Connection, expression: str, limit: int) -> list[tuple[str, float]]:
    if not expression:
        return []
    rows = connection.execute(
        f"SELECT parent_asin, {_BM25_RANK} FROM products WHERE products MATCH ? "
        f"ORDER BY {_BM25_RANK} LIMIT ?",
        (expression, limit),
    ).fetchall()
    return [(str(parent_asin), -float(score)) for parent_asin, score in rows]


def _slot_values(context: Context, name: str) -> list[str]:
    slot = context.state.slots.get(name)
    if not isinstance(slot, dict):
        return []
    return [str(value) for value in slot.get("values", ()) if str(value)]


def bm25_route(connection: sqlite3.Connection, context: Context, limit: int) -> list[tuple[str, float]]:
    return _run(connection, _fts_or(terms(context.user_message)), limit)


def category_route(connection: sqlite3.Connection, context: Context, limit: int) -> list[tuple[str, float]]:
    """BM25 on the accumulated query, restricted to the category slot.

    Membership is decided by the ``categories:`` filter alone; the message and
    attribute terms only influence BM25 order. The category prefix is also
    OR-ed into the ranked query, so a product in the category that shares no
    other term still matches. The filter is route-scoped, not a catalog filter.
    """
    stems = sorted({
        _stem(token)
        for value in _slot_values(context, "category")
        for token in terms(value)
        if token
    })
    if not stems:
        return []
    attribute_tokens = [
        token
        for name in _ATTRIBUTE_SLOTS
        for value in _slot_values(context, name)
        for token in terms(value)
    ]
    # ``stems`` come from ``terms()``: lowercase [a-z0-9]+ only, safe to inline.
    prefixes = " OR ".join(f"{stem}*" for stem in stems)
    ranked = _fts_or(terms(context.user_message) + attribute_tokens)
    query = f"{ranked} OR {prefixes}" if ranked else prefixes
    category_filter = " OR ".join(f"categories:{stem}*" for stem in stems)
    return _run(connection, f"({query}) AND ({category_filter})", limit)


def attribute_route(connection: sqlite3.Connection, context: Context, limit: int) -> list[tuple[str, float]]:
    tokens = [
        token
        for name in _ATTRIBUTE_SLOTS
        for value in _slot_values(context, name)
        for token in terms(value)
    ]
    return _run(connection, _fts_or(tokens), limit)


ROUTES: dict[str, object] = {
    "bm25": bm25_route,
    "category": category_route,
    "attribute": attribute_route,
}


def run_routes(
    connection: sqlite3.Connection,
    context: Context,
    limit: int = POOL_LIMIT,
    routes: Iterable[str] | None = None,
) -> dict[str, list[tuple[str, float]]]:
    """Execute the selected routes and return their raw result lists.

    ``routes`` selects which route functions actually RUN; ``None`` runs every
    route. An unselected route is never called, so it issues no query --
    selection happens before execution, not by discarding results afterwards.
    Unknown names are ignored rather than raising, so a caller naming a route
    that does not exist degrades to fewer routes instead of breaking the turn.
    """
    selected = ROUTES if routes is None else {
        name: ROUTES[name] for name in routes if name in ROUTES
    }
    return {
        name: route(connection, context, limit)  # type: ignore[operator]
        for name, route in selected.items()
    }


def fuse(
    per_route: dict[str, list[tuple[str, float]]], limit: int = POOL_LIMIT
) -> list[Candidate]:
    """Reciprocal Rank Fusion over per-route results -> the candidate pool.

    Every route contributes ``1 / (_RRF_K + rank)`` per candidate, so a
    candidate ranked highly by any route can enter the final pool even when
    another route alone would fill the whole budget. Multi-route candidates
    accumulate votes and rise.

    Deterministic, deduplicated by construction, and each ``Candidate`` keeps
    its raw per-route scores. Every route in ``per_route`` is fused, not only
    those named in ``_ROUTE_ORDER``.
    """
    fused: dict[str, float] = defaultdict(float)
    route_scores: dict[str, dict[str, float]] = defaultdict(dict)
    best_rank: dict[str, int] = {}
    best_route: dict[str, int] = {}

    for priority, name in enumerate(_fusion_priority(per_route)):
        for rank, (parent_asin, score) in enumerate(per_route[name]):
            fused[parent_asin] += 1.0 / (_RRF_K + rank + 1)
            route_scores[parent_asin][name] = score
            best_rank[parent_asin] = min(best_rank.get(parent_asin, rank), rank)
            best_route.setdefault(parent_asin, priority)

    ordered = sorted(
        fused,
        key=lambda asin: (-fused[asin], best_rank[asin], best_route[asin], asin),
    )[:limit]
    return [
        Candidate(
            parent_asin=asin,
            route_scores=dict(route_scores[asin]),
            # The rank-derived, pool-independent base that ranking builds on.
            metadata={"fusion_score": fused[asin]},
        )
        for asin in ordered
    ]


def retrieve(
    connection: sqlite3.Connection,
    context: Context,
    limit: int = POOL_LIMIT,
    routes: list[str] | None = None,
) -> list[Candidate]:
    """UNION of the selected routes, Reciprocal-Rank-Fusion ordered.

    ``routes`` selects which routes execute; ``None`` runs them all. Returns up
    to ``limit`` candidates, each carrying its raw per-route scores so
    ``route_sources`` names every route that surfaced it.
    """
    return fuse(run_routes(connection, context, limit, routes), limit)
