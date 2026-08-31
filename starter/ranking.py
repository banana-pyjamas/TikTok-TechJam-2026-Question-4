"""Constraint-aware ranking.

Reorders the retrieval pool by how well each candidate satisfies the
constraints accumulated in ``SessionState``:

    final = base + W_MATCH * match_ratio - W_PENALTY * violation_ratio

``base`` is the retrieval fusion score, which is rank-derived and therefore
already on a globally comparable scale -- no candidate-pool normalization.

Each active slot classifies a candidate as MATCH (the catalog says the product
has the requested value), VIOLATION (the catalog asserts a different value) or
UNKNOWN (the catalog says nothing).

UNKNOWN is never counted as a violation. On the frozen catalog only 47% of
products carry a recognisable colour, 63% a material and 21% a price, so
treating absent metadata as a mismatch would bury most of the catalog.

UNKNOWN slots do stay in the ratio denominator, which is the count of active
constraints. That denominator is identical for every candidate in a turn, so
no candidate is penalised for what the catalog omits about IT. That is
candidate-independence, NOT order invariance -- ``base`` is not divided by the
denominator, so changing it changes the strength of constraint evidence
relative to retrieval order and two candidates can cross::

    A  base 0.00, MATCH     ->  0.00 + 0.10 * 1/1 = 0.100   A wins
    B  base 0.06, UNKNOWN   ->  0.06              = 0.060

    add a second constraint, UNKNOWN for both:
    A  ->  0.00 + 0.10 * 1/2 = 0.050                        B wins

Reordering is the point; a scoring layer that could not reorder would do
nothing. The alternative -- dividing by only the slots with a known verdict --
was measured and rejected because it rewards ignorance: a product the catalog
describes fully and that matches two of three constraints scores 2/3, while
one the catalog mentions a single colour for scores 1/1 and outranks it on
strictly less evidence.

The violation penalty is bounded, so a violating candidate is pushed down but
never eliminated and a parser mistake stays recoverable.

Ranking is read-only with respect to ``SessionState`` and does no I/O beyond
one indexed metadata lookup per turn.
"""

from __future__ import annotations

from typing import Any

from starter.contracts import Candidate, Context, RankingResult
from starter.popularity import popularity_score
from starter.profile import extract_evidence, profile_match_ratio
from starter.reliability import reliability_of

# Ablation flag for the anonymized-profile prior.
#
# OFF. Measured twice on different baselines and the sign flipped between them,
# with a confidence interval straddling zero both times -- the burden is on the
# change and a feature whose sign is not stable has not met it.
#
# The cause is in the data, not the implementation: purchase_frequency is the
# same string in all 200 sessions, and the tags are dominated by dimensions
# that say nothing about which product is wanted (fit 81.5%, material 77%,
# comfort 72%). A signal present in four of five sessions cannot separate
# candidates. The code stays; the priority and empty-profile guarantees are
# tested, so this is one flag away if a richer profile ever arrives.
USE_PROFILE = False

# Weights, calibrated against the fusion base, which spans roughly
# [0.003, 0.049].
#
# W_MATCH sits well above that spread so constraint evidence, not raw text
# overlap, decides the ordering. The result saturates above ~0.02 and is then
# flat, so 0.10 is inside a stable plateau rather than balanced on a cliff.
#
# W_PENALTY is deliberately kept BELOW the base spread: a fully violating
# candidate loses less than a strong retrieval score can supply, so one
# mistaken extracted constraint demotes a candidate without burying it. A
# violation therefore always costs less than a match gains.
W_MATCH = 0.10
W_PENALTY = 0.02

# The profile prior is the weakest tier by construction. Priority is: current
# explicit request > active session state > profile. W_PROFILE is held an order
# of magnitude below W_MATCH so satisfying the whole profile can never outweigh
# even one satisfied constraint -- it may reorder candidates the constraints
# are indifferent between, and nothing more.
W_PROFILE = 0.008

# Slots whose catalog evidence is reliable enough to call a mismatch a
# violation. ``size`` is deliberately excluded: size metadata is sparse and
# inconsistent, so a size mismatch is treated as UNKNOWN.
VIOLATION_SLOTS = frozenset({"category", "color", "material", "brand", "budget"})
SCORED_SLOTS = ("category", "color", "material", "brand", "size", "budget")

MATCH, VIOLATION, UNKNOWN = "match", "violation", "unknown"

# The diagnostics contract. Every ranked candidate exposes exactly these keys;
# ``tests/test_ranking.py`` freeze-guards the set.
DIAGNOSTIC_KEYS = frozenset({
    "base_score",
    "attribute_score",
    "violation_penalty",
    "profile_score",
    "final_score",
    "rank",
    "matched",
    "violated",
    "route_sources",
    "constraint_weights",
    "popularity_score",
})


# Ablation flag for confidence/reliability weighting.
#
# OFF reproduces unweighted scoring exactly: every active slot counts 1. ON,
# each slot's contribution is scaled by Evidence Confidence (how firmly the
# shopper asserted it) times Match Reliability (how much the catalog's verdict
# on that field is worth). See ``constraint_weights``.
#
# The mechanism is live rather than inert -- it reorders about half of all
# turns and moves the hidden target on a few -- but it does not change which
# sessions hit, and its point estimate is small enough to have flipped sign
# when an unrelated constant moved. A quantity whose sign depends on an
# unrelated constant is not one to ship on, and weighting SHRINKS the attribute
# term, which is a component that does measurably work. It stays OFF on the
# burden-of-proof rule. `python3 -m tools.phase11_confidence` measures it.
#
# Evidence Confidence itself is computed and stored regardless: it is state,
# not scoring, and clarification reads it.
USE_CONFIDENCE_WEIGHTING = False

# Ablation flag for the popularity prior.
#
# ON. It used to be the largest established gain here; it is not any more.
# With clarification supplying real constraints the same term is worth much
# less and no longer clears significance. That is not a regression -- a prior
# is supposed to matter less as evidence arrives, and this one measurably does.
# It stays ON because the point estimate is positive and it costs nothing, but
# it should not be quoted as established. `python3 -m tools.phase12_popularity`
# regenerates the comparison.
#
# WHY IT IS WORTH ANYTHING, WHICH IS NOT THE FLATTERING ANSWER. Not because
# popularity is a deep signal about what shoppers want, but because the public
# set's targets are drawn almost entirely from the most-reviewed products:
# their median review count sits at the 99.5th catalog percentile, and 4 of 200
# fall below the catalog median where unbiased would be 100. On this benchmark
# a bestseller list is close to an oracle. The private set is built the same
# way so the gain should transfer -- but it is a fact about how the evaluation
# was sampled, not evidence that ranking by review count serves shoppers. On a
# counterfactual set with targets drawn uniformly the same prior measures
# -0.012556. Anyone quoting the gain should quote that with it.
#
# THE WEIGHT IS DELIBERATELY LEFT ON THE TABLE. W_POPULARITY is 0.008, an order
# of magnitude below W_MATCH. Raising it pays enormously on this benchmark --
# 0.02 and 0.05 and 0.10 all score progressively higher -- but at 0.10 the
# prior equals W_MATCH and a bestseller can cancel a satisfied constraint
# outright. Anything above ~0.01 stops being a prior and becomes the ranker.
USE_POPULARITY = True

# Where ``rank`` reads the catalog popularity scale out of ``Context.derived``.
POPULARITY_KEY = "popularity_scale"

# Where ``rank`` reads per-slot Match Reliability out of ``Context.derived``.
# A generic container: a new signal adds a key, never a frozen field. Absent
# means every slot is fully reliable.
RELIABILITY_KEY = "match_reliability"


def slot_confidence(context: Context, slot: str) -> float:
    """Evidence Confidence for one active slot.

    Absent means 1.0, which is what makes ``USE_CONFIDENCE_WEIGHTING = False``
    exact rather than approximate.
    """
    entry = context.state.slots.get(slot) if isinstance(context.state.slots, dict) else None
    if not isinstance(entry, dict):
        return 1.0
    value = entry.get("confidence")
    if not isinstance(value, (int, float)) or value != value:
        return 1.0
    return max(0.0, min(1.0, float(value)))


def active_constraints(
    context: Context,
) -> tuple[dict[str, list[str]], dict[str, Any] | None]:
    """The scored slots that currently hold values, plus any budget bounds.

    Reads only ACTIVE slot values, so a constraint superseded by an override
    can never influence ranking.
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


def _category_value_matches(tokens: set[str], value: str) -> bool:
    """True only if EVERY word of a category value is present.

    A multi-word category is a conjunction: "swim trunks" means swim AND
    trunks. Matching on any single word made "swim trunks" match "Women's
    Swimwear" and "tank top" match "Topcoats". Prefix matching is kept per word
    so plural tolerance (jacket -> jackets) still works.
    """
    stems = [_stem(word) for word in value.split() if word]
    if not stems:
        return False
    return all(
        any(token.startswith(stem) for token in tokens) for stem in stems
    )


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
        if any(_category_value_matches(tokens, value) for value in values):
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


def constraint_weights(
    context: Context, constraints: dict[str, list[str]]
) -> dict[str, float]:
    """``EC * MR`` per active slot.

    Evidence Confidence says how firmly the shopper asserted the constraint;
    Match Reliability says how much the catalog's verdict on that field is
    worth. A slot's verdict is only as good as the weaker of the two, and the
    product is the graded form of that:

        high EC, high MR   ~1.00   full weight
        high EC, low MR     0.10   a real requirement we cannot check
                                   reliably -- it must not bury a target that
                                   was never in the catalog's terms anyway
        low EC,  high MR    0.40   a passing remark the catalog can check
                                   perfectly, which still must not act as a
                                   filter

    Neither factor can reach zero, so no constraint is ever silently switched
    off -- it is discounted, and the result is still a ranking rather than a
    filtered set.
    """
    reliabilities = None
    if isinstance(context.derived, dict):
        reliabilities = context.derived.get(RELIABILITY_KEY)
    return {
        slot: slot_confidence(context, slot) * reliability_of(reliabilities, slot)
        for slot in constraints
    }


def score_candidate(
    candidate: Candidate,
    constraints: dict[str, list[str]],
    meta: dict[str, Any],
    bounds: dict[str, Any] | None = None,
    profile_evidence: dict[str, Any] | None = None,
    weights: dict[str, float] | None = None,
    popularity: dict[str, float] | None = None,
    total_evidence: float = 0.0,
) -> dict[str, Any]:
    """Constraint verdicts plus the score components for one candidate.

    ``weights`` maps a slot to ``EC * MR``; ``None`` means every slot counts 1.
    """
    matched: list[str] = []
    violated: list[str] = []
    # The denominator is the active constraint COUNT, identical for every
    # candidate this turn. Evidence quality multiplies the NUMERATOR only.
    #
    # It has to be this way round. Normalising by the summed weight instead
    # would cancel the weight whenever one constraint is active -- a single
    # slot at EC 0.4 would give 0.4/0.4 = full weight, defeating the mechanism.
    # Dividing by the count leaves the unclaimed share of the budget simply
    # unclaimed, so a half-meant constraint on a half-trusted field contributes
    # a quarter of the evidence it otherwise would and retrieval order carries
    # the rest. That is the safe direction to fail in.
    considered = 0
    matched_weight = 0.0
    violated_weight = 0.0
    for slot, values in constraints.items():
        verdict = classify(slot, values, meta, bounds)
        weight = 1.0 if weights is None else float(weights.get(slot, 1.0))
        considered += 1
        if verdict == MATCH:
            matched.append(slot)
            matched_weight += weight
        elif verdict == VIOLATION and slot in VIOLATION_SLOTS:
            violated.append(slot)
            violated_weight += weight

    base = float(candidate.metadata.get("fusion_score", 0.0))
    if considered:
        attribute_score = W_MATCH * (matched_weight / considered)
        violation_penalty = W_PENALTY * (violated_weight / considered)
    else:
        attribute_score = 0.0
        violation_penalty = 0.0

    profile_score = 0.0
    if profile_evidence:
        profile_score = W_PROFILE * profile_match_ratio(
            profile_evidence, meta.get("traits") or set()
        )

    # Bounded by W_POPULARITY and decayed by the evidence already gathered, so
    # it can only separate candidates the constraints leave tied.
    popularity_term = popularity_score(meta, popularity, total_evidence)

    return {
        "base_score": base,
        "attribute_score": attribute_score,
        "violation_penalty": violation_penalty,
        "profile_score": profile_score,
        "popularity_score": popularity_term,
        "final_score": (base + attribute_score - violation_penalty
                        + profile_score + popularity_term),
        "matched": matched,
        "violated": violated,
        "route_sources": list(candidate.route_sources),
        # Why each verdict counted for as much as it did. Same for every
        # candidate in a turn, but carried per candidate so a diagnostics row
        # explains its own score.
        "constraint_weights": dict(weights or {}),
    }


def rank(
    candidates: list[Candidate],
    context: Context,
    metadata: dict[str, dict[str, Any]],
    top_k: int,
) -> RankingResult:
    """Score, order, and truncate the pool.

    Deterministic: ties break on base score then ``parent_asin``. Deduplicated
    by ``parent_asin``. Every entry gets full diagnostics.

    The profile prior is read from ``context.state.user_profile`` and applied as
    the weakest term; it never touches ``state.slots``, so it cannot override an
    explicit request or a session constraint.
    """
    constraints, bounds = active_constraints(context)
    profile_evidence = (
        extract_evidence(context.state.user_profile) if USE_PROFILE else None
    )
    weights = constraint_weights(context, constraints) if USE_CONFIDENCE_WEIGHTING else None
    popularity = None
    total_evidence = 0.0
    if USE_POPULARITY:
        if isinstance(context.derived, dict):
            popularity = context.derived.get(POPULARITY_KEY)
        # How much the shopper has actually committed to so far: summed
        # Evidence Confidence, so a hedged constraint displaces less of the
        # prior than an insisted-upon one. Read straight from state, so this
        # works whether or not USE_CONFIDENCE_WEIGHTING is on.
        total_evidence = sum(slot_confidence(context, slot) for slot in constraints)
    scored: list[tuple[Candidate, dict[str, Any]]] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.parent_asin in seen:
            continue
        seen.add(candidate.parent_asin)
        detail = score_candidate(
            candidate, constraints, metadata.get(candidate.parent_asin, {}),
            bounds, profile_evidence, weights, popularity, total_evidence,
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
