"""Multi-route retrieval (Phase 5).

Three deterministic routes over the FTS index, combined by a UNION (never an
intersection -- principle F):

  bm25       full-text match on the raw user message (the Phase 1 baseline)
  category   BM25 on the accumulated query, restricted to the category slot
  attribute  products mentioning any color / material / brand / size slot value

Pool order: BM25 results in BM25 order, then the candidates that only the
category route found, then the attribute-only ones. BM25 stays authoritative
for the head of the pool, so pool Recall@K can never drop below BM25's for
K within the BM25 result size -- the extra routes only widen reach in the
tail. Turning the extra routes into rank movement in the top-K needs
constraint-aware scoring, which is Phase 6.

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

from starter.contracts import Candidate, Context
from starter.text import terms

# Column weights: parent_asin, title, categories, features, details, store,
# description. Same as the Phase 1 baseline.
_BM25_RANK = "bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)"
_ATTRIBUTE_SLOTS = ("color", "material", "brand", "size")
_ROUTE_ORDER = ("bm25", "category", "attribute")

POOL_LIMIT = 300


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

    The category-column filter is a route-scoped restriction, not a catalog
    filter: an out-of-category product can still reach the pool via the
    BM25 route. The category words are part of the ranked query, so every
    in-category product matches (nothing is dropped for lacking an
    attribute term -- CP 5.8).
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
    query = _fts_or(terms(context.user_message) + attribute_tokens + stems)
    category_filter = " OR ".join(f"categories:{stem}*" for stem in stems)
    expression = f"({query}) AND ({category_filter})" if query else f"({category_filter})"
    return _run(connection, expression, limit)


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


def retrieve(
    connection: sqlite3.Connection, context: Context, limit: int = POOL_LIMIT
) -> list[Candidate]:
    """UNION of every route.

    Returns up to ``limit`` ``Candidate`` objects in pool order (BM25 head,
    then category-only, then attribute-only). Each carries its raw per-route
    scores in ``route_scores`` (so ``route_sources`` names every route that
    surfaced it).
    """
    per_route: dict[str, list[tuple[str, float]]] = {
        name: route(connection, context, limit) for name, route in ROUTES.items()  # type: ignore[operator]
    }

    route_scores: dict[str, dict[str, float]] = defaultdict(dict)
    order: list[str] = []
    seen: set[str] = set()
    for name in _ROUTE_ORDER:
        for parent_asin, score in per_route.get(name, []):
            route_scores[parent_asin][name] = score
            if parent_asin not in seen:
                seen.add(parent_asin)
                order.append(parent_asin)

    return [
        Candidate(parent_asin=parent_asin, route_scores=dict(route_scores[parent_asin]))
        for parent_asin in order[:limit]
    ]
