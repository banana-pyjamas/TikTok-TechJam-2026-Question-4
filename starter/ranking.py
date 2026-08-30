"""Constraint-aware ranking (Phase 6).

Reorders the retrieval pool by how well each candidate satisfies the
constraints accumulated in ``SessionState``:

    final = base + W_MATCH * match_ratio - W_PENALTY * violation_ratio

``base`` is the retrieval fusion score, which is rank-derived and therefore
already on a globally comparable scale -- no candidate-pool min-max
normalization anywhere (principle G).

Each active slot classifies a candidate as MATCH / VIOLATION / UNKNOWN:

* MATCH     the catalog says the product has the requested value;
* VIOLATION the catalog asserts a DIFFERENT value for that slot;
* UNKNOWN   the catalog says nothing about that slot.

UNKNOWN is never a violation (CP 6.4, principle D) -- it contributes to
neither ratio. This matters: on the frozen catalog only 47% of products
carry a recognisable colour, 63% a material and 21% a price, so treating
absent metadata as a mismatch would bury most of the catalog.

The violation penalty is bounded (``W_PENALTY``): a violating candidate is
pushed down, never eliminated and never sent to -infinity (CP 6.3,
principle E). A parser mistake is therefore always recoverable.

Ranking is read-only with respect to ``SessionState`` and does no I/O beyond
one indexed metadata lookup per turn.
"""

from __future__ import annotations

from typing import Any

from starter.contracts import Candidate, Context, RankingResult

# Weights, calibrated against the fusion base, which spans roughly
# [0.003, 0.049] (see retrieval).
#
# W_MATCH sits well above that spread so constraint evidence, not raw text
# overlap, decides the ordering. Measured on the public set the result
# saturates above ~0.02 and is then flat, so 0.10 is inside a stable plateau
# rather than balanced on a cliff.
#
# W_PENALTY is deliberately kept BELOW the base spread: a fully violating
# candidate loses less than a strong retrieval score can supply, so one
# mistaken extracted constraint demotes a candidate without burying it
# beneath the entire pool (principle E -- a parser mistake must stay
# recoverable). A violation therefore also always costs less than a match
# gains.
W_MATCH = 0.10
W_PENALTY = 0.02

# Slots whose catalog evidence is reliable enough to call a mismatch a
# violation. `size` is deliberately excluded: size metadata is sparse and
# inconsistent, so a size mismatch is treated as UNKNOWN rather than
# penalising the candidate.
VIOLATION_SLOTS = frozenset({"category", "color", "material", "brand", "budget"})
SCORED_SLOTS = ("category", "color", "material", "brand", "size", "budget")

MATCH, VIOLATION, UNKNOWN = "match", "violation", "unknown"

# CP 6.7 -- the diagnostics contract. Every ranked candidate exposes exactly
# these keys; ``tests/test_ranking.py`` freeze-guards the set.
DIAGNOSTIC_KEYS = frozenset({
    "base_score",
    "attribute_score",
    "violation_penalty",
    "final_score",
    "rank",
    "matched",
    "violated",
    "route_sources",
})


def active_constraints(
    context: Context,
) -> tuple[dict[str, list[str]], dict[str, Any] | None]:
    """The scored slots that currently hold values, plus any budget bounds.

    Reads only the ACTIVE slot values, so a constraint superseded by an
    override (Phase 3) can never influence ranking.
    """
    constraints: dict[str, list[str]] = {}
    bounds: dict[str, Any] | None = None
    for name in SCORED_SLOTS:
        slot = context.state.slots.get(name)
        if not isinstance(slot, dict):
            continue
        values = [str(v).lower() for v in slot.get("values", ()) if str(v)]
        if not values:
            continue
        constraints[name] = values
        if name == "budget" and isinstance(slot.get("bounds"), dict):
            bounds = slot["bounds"]
    return constraints, bounds


def _stem(token: str) -> str:
    return token[:-1] if len(token) > 3 and token.endswith("s") else token


def classify(slot: str, values: list[str], meta: dict[str, Any],
             bounds: dict[str, Any] | None = None) -> str:
    """MATCH / VIOLATION / UNKNOWN for one slot against one product."""
    if slot == "budget":
        price = meta.get("price")
        if price is None or not bounds:
            return UNKNOWN
        low, high = bounds.get("min"), bounds.get("max")
        if low is not None and price < low:
            return VIOLATION
        if high is not None and price > high:
            return VIOLATION
        return MATCH

    if slot == "category":
        tokens = meta.get("cats") or set()
        if not tokens:
            return UNKNOWN
        stems = {_stem(v) for value in values for v in value.split()}
        if any(token.startswith(stem) for token in tokens for stem in stems if stem):
            return MATCH
        return VIOLATION

    if slot == "brand":
        store = meta.get("store") or ""
        if not store:
            return UNKNOWN
        return MATCH if any(value in store for value in values) else VIOLATION

    if slot == "size":
        sizes = meta.get("sizes") or set()
        if not sizes:
            return UNKNOWN
        # Sparse, inconsistent metadata: a hit is informative, a miss is not.
        return MATCH if any(value in sizes for value in values) else UNKNOWN

    present = meta.get(slot) or set()   # color / material
    if not present:
        return UNKNOWN
    return MATCH if any(value in present for value in values) else VIOLATION


def score_candidate(
    candidate: Candidate,
    constraints: dict[str, list[str]],
    meta: dict[str, Any],
    bounds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Constraint verdicts plus the score components for one candidate."""
    matched: list[str] = []
    violated: list[str] = []
    considered = 0
    for slot, values in constraints.items():
        considered += 1
        verdict = classify(slot, values, meta, bounds)
        if verdict == MATCH:
            matched.append(slot)
        elif verdict == VIOLATION and slot in VIOLATION_SLOTS:
            violated.append(slot)

    base = float(candidate.metadata.get("fusion_score", 0.0))
    if considered:
        attribute_score = W_MATCH * (len(matched) / considered)
        violation_penalty = W_PENALTY * (len(violated) / considered)
    else:
        attribute_score = 0.0
        violation_penalty = 0.0
    return {
        "base_score": base,
        "attribute_score": attribute_score,
        "violation_penalty": violation_penalty,
        "final_score": base + attribute_score - violation_penalty,
        "matched": matched,
        "violated": violated,
        "route_sources": list(candidate.route_sources),
    }


def rank(
    candidates: list[Candidate],
    context: Context,
    metadata: dict[str, dict[str, Any]],
    top_k: int,
) -> RankingResult:
    """Score, order, and truncate the pool.

    Deterministic (CP 6.5): ties break on base score then ``parent_asin``.
    Deduplicated (CP 6.6): the pool is keyed by ``parent_asin`` upstream and
    the ranked list preserves that. Every entry gets full diagnostics
    (CP 6.7).
    """
    constraints, bounds = active_constraints(context)
    scored: list[tuple[Candidate, dict[str, Any]]] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.parent_asin in seen:
            continue
        seen.add(candidate.parent_asin)
        detail = score_candidate(
            candidate, constraints, metadata.get(candidate.parent_asin, {}), bounds
        )
        scored.append((candidate, detail))

    scored.sort(
        key=lambda pair: (
            -pair[1]["final_score"],
            -pair[1]["base_score"],
            pair[0].parent_asin,
        )
    )
    kept = scored[: max(0, top_k)]

    diagnostics: dict[str, dict[str, Any]] = {}
    for index, (candidate, detail) in enumerate(kept):
        diagnostics[candidate.parent_asin] = {**detail, "rank": index + 1}
    return RankingResult(ranked=[c for c, _ in kept], diagnostics=diagnostics)
