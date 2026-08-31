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

# --------------------------------------------------------------------------
# Phase 13 (dense retrieval) -- NOT IMPLEMENTED. There are two reasons and
# they are NOT equally strong: one is measured, one is a packaging constraint.
# Keeping them apart is the whole point of this comment, because the first
# version of it merged them and overstated both.
#
# `python3 -m tools.phase13_dense_gate` reproduces every number below.
#
# 1  WHAT WAS MEASURED: A LEXICAL VECTOR ROUTE, AND IT MAKES THE POOL WORSE
#
# The strongest vector-space retriever that can actually be built under the
# submission constraints is lexical: TF-IDF cosine over the same catalog text
# BM25 scores, with BM25's own column weights. Built and run rather than
# argued from theory -- this repo has been burned once for predicting a
# mechanism's behaviour instead of measuring it (the Phase 11 "cannot
# reorder" claim). Session-level candidate recall over the live dialogue, all
# four arms fused through the RRF below with route provenance intact:
#
#                                   @50      @100      @300   vs committed @300
#   bm25                         0.3800    0.5300    0.7700   +1 / -9
#   bm25 + category (committed)  0.4150    0.5650    0.8100   --
#   tfidf                        0.2600    0.4150    0.6950   +4 / -27
#   bm25 + category + tfidf      0.3600    0.5100    0.8050   +1 / -2
#
# Standalone recall was never the question -- a weaker retriever can still
# contribute unique candidates through a union. The UNION is the question,
# and it is worse rather than neutral: -11 sessions at @50 (2/13 discordant,
# p = 0.0074) and -11 at @100 (1/12, p = 0.0034), no verdict at @300 (1/2).
# The lexical route adds 114.9 unique candidates per turn and pushes cap loss
# from 130.3 to 245.2 discarded candidates per turn; what it displaces is
# worth more than what it adds. OFF on measurement, not on principle.
#
# The recall and McNemar rows above are pure retrieval and do not move when
# anything downstream does. The two cap-loss figures DO -- they are counted
# over the live dialogue, and the dialogue changes when the ranker changes.
# They read 146 and 257 until Phase 14 moved them and nothing noticed, which
# is the whole of D's Phase 14 Finding 3: this repo's guards check assertions
# and are blind to prose. Phase 15 moved them again, which is what "moves
# with the pipeline" means -- the dialogue is shorter now, so fewer turns
# accumulate a wide pool. `python3 -m tools.phase13_dense_gate` regenerates
# both, and moving the pipeline means re-running it.
#
# 2  WHAT WAS NOT MEASURED: A TRAINED SEMANTIC ENCODER
#
# TF-IDF is a LEXICAL vector space. Two products with disjoint wording are
# orthogonal in it however related they are, which is the one thing a trained
# encoder does differently. It is therefore not a proxy for a dense retriever
# and NOT an upper bound on one, and nothing in this repo measures what a
# dense encoder would be worth here.
#
# It is absent for FEASIBILITY reasons, stated precisely because the previous
# version of this comment said the flat and false "none is installed":
# nothing is vendored in this repository; docs/submission_rules.md notes the
# organizer "may disable network access" for final scoring, so nothing can be
# fetched at scoring time; the submission is certified on the bare system
# python3.9, which current torch builds do not support; and this package is
# standard-library only by design. Those are packaging facts. Another machine
# having torch changes none of them -- and none of them is evidence that
# semantic retrieval would not help.
#
# 3  THE SIZE OF THE PRIZE, BRACKETED BY MEASUREMENT
#
# Scope first, because 7c52e87 got it wrong (C) and both of its terms moved
# the same way. A turn only counts if the evaluator would SCORE it: its hit
# test is `if override_applied and target in ranked`, and for intent_override
# sessions that flag is False until the override turn, so a target in the
# pool before then is a pool hit that could never have become a score.
# Turn-level and override-aware:
#
#   target in pool on ANY turn                   162/200
#   target in pool on a SCORING-ELIGIBLE turn    149/200   <- the honest one
#   never in pool on an eligible turn             51/200
#      of which never in the pool at all          38   <- what 7c52e87 counted
#      of which in the pool ONLY pre-override     13
#
# So retrieval loses 51 sessions, not 38, and its own metric says the headroom
# is +0.2550 candidate recall over eligible turns. That is the only figure
# here that is not an inference. Downstream it depends on WHERE a route puts
# the target, measured by re-running the evaluator with the answer injected
# into those 51 sessions and no others:
#
#   injected at the pool FLOOR (fusion_score 0)   TS +0.0047    1/51 convert
#   injected at the pool HEAD  (best fusion)      TS +0.2225   51/51 convert
#
# So retrieval's downstream value is somewhere in [+0.0047, +0.2225] TS, and
# RANK matters far more than presence: getting the target into the pool at the
# bottom recovers one session in fifty-one. The "+0.025 TS ceiling" quoted for
# this phase in 7c52e87 is neither bound -- it was an extrapolation, and it
# moved the HitRate term only, while TS = 0.5*HR + 0.3*MRR + 0.2*eff and a
# recovered hit moves all three. Applied consistently and at the corrected
# scope it is +0.0766. Withdrawn in favour of the bracket.
#
# Meanwhile 96 targets reach the pool on a turn that COULD have scored and
# still lose: ~1.9x the session count of the whole retrieval surface, all of
# it downstream of this file. (7c52e87 published 120 and 3.2x; both terms of
# that ratio were session-level and override-blind.)
#
# WHICH NUMBERS HERE MOVE WHEN THE RANKER MOVES. The retrieval facts do not:
# 162/149/51/38/13 and +0.2550 recall are properties of the pool and were
# identical before and after Phase 14 shipped. The CONVERSION facts do, and
# these are stated against the committed ranker WITH the Phase 14 reranker ON.
# Phase 14 moved in-pool conversion from 42/149 to 53/149, which is why the
# in-pool losses fell from 107 to 96 and the ratio from 2.1x to 1.9x. Re-run
# the gate after any ranking change; it reads the committed configuration and
# will disagree with this comment rather than quietly agree with it.
#
# 4  RETRACTION: THE VOCABULARY ARGUMENT MEASURED THE SIMULATOR
#
# 7c52e87 claimed "of the 38 missed targets, the number sharing NO vocabulary
# with what the shopper said is ZERO", and read that as "there is nothing for
# dense retrieval to bridge". WITHDRAWN. evaluator/local_evaluator.py:154
# builds the shopper's opening out of coarse_category(target.categories), so
# counting the target's categories field among its terms makes overlap >= 1
# an identity: 200/200 openings contain that generated string and 35 of the
# 38 misses have their entire overlap inside it. The sessions retrieval FINDS
# show the same distribution, so the metric separates nothing. With the
# copied taxonomy field removed, 5 of 38 misses share no vocabulary (13.2%,
# against 8.6% of found sessions); title only, 14 of 38 (36.8% vs 31.5%).
#
# What survives is narrow: pure vocabulary-mismatch cases are rare in this
# set. That is not the claim "semantic retrieval has no value", and this
# comment does not make it.
#
# 5  WHAT IS AND IS NOT DELIVERED
#
# CP 13.2 ("dense OFF preserves previous behaviour") holds trivially. CP 13.1
# / 13.3 / 13.4 / 13.5 are not implemented. No USE_DENSE flag is added: a flag
# whose ON position has no implementation is a knob that changes nothing,
# which this repo has deleted twice already.
#
# WHAT WOULD REOPEN THIS: a vendored encoder that runs offline on the
# certified interpreter. The bracket in (3) says what to measure about it
# FIRST -- the rank it assigns the target, not the recall it achieves, since
# recall alone bought +0.0047.
#
# That advice has since been taken. Phase 14 (starter/reranker.py) attacked
# the in-pool losses rather than the retrieval surface and measured +0.041 TS
# with a purely LEXICAL scorer, which is the strongest evidence available that
# this file was the wrong place to spend the effort. The encoder question is
# still open there, and the reranker's scorer interface is where it lands.
# --------------------------------------------------------------------------


def _fusion_priority(names: Iterable[str]) -> list[str]:
    """Deterministic fusion order over whatever routes were actually run.

    ``_ROUTE_ORDER`` first, in its declared order, then any other route name
    alphabetically. The tail matters: ``fuse`` used to iterate ``_ROUTE_ORDER``
    alone, so a per-route dict carrying a route not in that tuple had its
    results SILENTLY DISCARDED -- the pool came back looking correct and simply
    missing a route's candidates. A measurement tool fusing an experimental
    route (``tools/phase13_dense_gate.py``) would have measured a union that
    never happened.

    Behaviour for the committed route set is unchanged: with ``bm25`` and
    ``category`` present and nothing else, this returns exactly what iterating
    ``_ROUTE_ORDER`` returned.
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

    Every route in ``per_route`` is fused, not only the ones named in
    ``_ROUTE_ORDER`` -- see ``_fusion_priority`` for why that distinction was
    load-bearing.
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
