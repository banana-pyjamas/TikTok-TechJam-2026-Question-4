"""Multi-route retrieval (Phase 5).

Three deterministic routes over the FTS index, combined by a UNION (never an
intersection -- principle F):

  bm25       full-text match on the raw user message (the Phase 1 baseline)
  category   BM25 on the accumulated query, restricted to the category slot
  attribute  products mentioning any color / material / brand / size slot value

Pool order: Reciprocal Rank Fusion over the routes. Each route votes with
``1 / (RRF_K + rank)``, so a candidate near the head of ANY route outranks
one buried deep in another, and a candidate found by several routes rises
above one found by a single route. This is what lets the category and
attribute routes contribute unique candidates to the FINAL pool rather than
being truncated behind a full BM25 head.

Fusion is rank-based, never score-based: no candidate-pool min-max
normalization (principle G). Ordering is fully deterministic -- ties break
on best per-route rank, then ``parent_asin``.

Missing catalog metadata reduces how many routes surface a product but never
eliminates it -- one route is enough, and no route hard-filters on a field
being present (CP 5.8 / principle D). Route-scoped filtering (e.g. the
category route dropping out-of-category rows) is allowed; the product can
still enter the pool through another route.

Retrieval never mutates ``SessionState`` -- it reads ``context.state`` only.
Scores on each ``Candidate`` are raw per-route BM25 values (negated so higher
= better); no candidate-pool score normalization (principle G).
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from typing import Iterable

from starter.contracts import Candidate, Context
from starter.text import terms

# Column weights: parent_asin, title, categories, features, details, store,
# description. Same as the Phase 1 baseline.
_BM25_RANK = "bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)"
_ATTRIBUTE_SLOTS = ("color", "material", "brand", "size")
# Deterministic tie-break order when two candidates fuse to the same score.
_ROUTE_ORDER = ("bm25", "category", "attribute")
# Standard RRF damping. Rank 1 of any route (1/61) outranks rank 61+ of
# another, which is what gives the auxiliary routes real access to the pool.
_RRF_K = 60

POOL_LIMIT = 300

# The routes the agent actually runs. Chosen on CANDIDATE RECALL, which is
# retrieval's own metric and where the difference is established; NOT on the
# technical score, where it is not. Both halves matter, so both are stated.
#
# Recall of the hidden target over the live dialogue, session-level, paired
# McNemar vs bm25 alone (``python3 -m tools.phase9_retrieval_evidence``):
#
#                                 @50      @100      @300   vs bm25 @300
#   bm25 only                  0.3800    0.5300    0.7700   --
#   bm25 + category            0.4150    0.5650    0.8100   +8, 9/1, p = 0.0215
#   bm25 + cat + attribute     0.4000    0.5450    0.7950   +5, 8/3, p = 0.2266
#
# Read that as ONE comparison observed at three thresholds, not three
# independent results (D-P3). The three K are nested views of the same paired
# data, so counting them as three findings would overstate it: at a Bonferroni
# alpha/6 none clear, at alpha/2 only @100 and @300 do. What actually carries
# the conclusion is that the category route gains at every threshold and loses
# at none -- +7/+7/+8 with 1, 0 and 1 sessions moving the other way.
#
# The attribute route gains at no threshold, and costs 0.019 TS -- it re-ranks
# on terms the BM25 route already covers, and its unique candidates arrive too
# far down to help. So it is excluded.
#
# What this comment does NOT claim: that the route set is worth a known amount
# of score. End-to-end, with ranking ON -- the arm that ships -- adding the
# category route is +0.003372 TS, 3 sessions gained and 2 lost, McNemar
# p = 1.0000. NO VERDICT. The "+0.0127" quoted for this change in 66d127c is
# the ranking-OFF arm (Run 2), which is not a configuration that ships, and it
# should not have been banked (D-N1). Run ``python3 -m tools.phase7_ablation``
# for both arms side by side.
#
# This is a retrieval constant, not a per-turn decision: selecting per mode was
# measured worth +0.000298 and is not worth a knob.
DEFAULT_ROUTES = ("bm25", "category")


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

    Membership is decided by the ``categories:`` filter alone; the message
    and attribute terms only influence BM25 ORDER. The category prefix is
    also OR-ed into the ranked query, so a product in the category that
    shares no other term still matches -- nothing is dropped for lacking an
    attribute or a description (CP 5.8).

    The filter is route-scoped, not a catalog filter: an out-of-category
    product still reaches the pool through the BM25 route (principle E).
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

    ``routes`` selects which route functions actually RUN. ``None`` runs
    every route, preserving the Phase 5 reviewer diagnostics. An unselected
    route is never called, so it issues no query -- selection happens before
    execution, not by discarding results afterwards.

    Unknown names are ignored rather than raising: a strategy naming a route
    that does not exist should degrade to fewer routes, not break the turn.
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
    candidate ranked highly by ANY route can enter the final pool even when
    another route alone would fill the whole budget. Multi-route candidates
    accumulate votes and rise.

    Deterministic: ties break on best per-route rank, then route priority,
    then ``parent_asin``. Deduplicated by construction (fusion is keyed by
    ``parent_asin``). Each ``Candidate`` keeps its RAW per-route scores.
    """
    fused: dict[str, float] = defaultdict(float)
    route_scores: dict[str, dict[str, float]] = defaultdict(dict)
    best_rank: dict[str, int] = {}
    best_route: dict[str, int] = {}

    for priority, name in enumerate(_ROUTE_ORDER):
        for rank, (parent_asin, score) in enumerate(per_route.get(name, [])):
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
            # The fused score is the rank-derived, pool-independent base that
            # constraint ranking (Phase 6) builds on.
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

    ``routes`` selects which routes EXECUTE -- the unselected ones are never
    called and issue no query. ``None`` runs them all. Returns up to
    ``limit`` ``Candidate`` objects; each carries its raw per-route scores in
    ``route_scores``, so ``route_sources`` names every route that surfaced it.
    """
    return fuse(run_routes(connection, context, limit, routes), limit)
