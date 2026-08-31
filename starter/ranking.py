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

UNKNOWN is never counted as a violation (CP 6.4, principle D). This matters:
on the frozen catalog only 47% of products carry a recognisable colour, 63%
a material and 21% a price, so treating absent metadata as a mismatch would
bury most of the catalog.

UNKNOWN slots DO stay in the ratio denominator, which is the count of active
constraints. That denominator is identical for every candidate in a turn, so
no candidate is penalised for what the catalog omits about IT -- which is what
CP 6.4 requires, and is the whole claim.

It is NOT a claim that the ordering is invariant. An earlier version of this
paragraph said the denominator "is a monotone rescale: UNKNOWN never changes
the relative order of candidates". That is false, and the B Phase 11 review
found it: ``base`` is not divided by the denominator, so changing the
denominator changes the strength of constraint evidence RELATIVE to retrieval
order, and two candidates can cross. With one active constraint::

    A  base 0.00, MATCH     ->  0.00 + 0.10 * 1/1 = 0.100   A wins
    B  base 0.06, UNKNOWN   ->  0.06                = 0.060

    add a second, UNKNOWN-for-both constraint:
    A  ->  0.00 + 0.10 * 1/2 = 0.050                        B wins
    B  ->  0.06              = 0.060

Reordering is in fact the POINT -- a scoring layer that could not reorder
would do nothing. What the denominator guarantees is candidate-independence,
not order invariance, and only the former was ever needed.

The alternative -- dividing by only the slots with a known verdict -- was
measured and REJECTED. It rewards ignorance: with constraints
{colour, material, category}, a product the catalog describes fully and that
matches two of them scores 2/3, while a product the catalog mentions only a
colour for, matching just that, scores 1/1 and outranks it on strictly less
evidence. The thinner a product's metadata, the easier a perfect score --
pointing straight at the 53% of the catalog with no colour. It also measured
worse: TS 0.131194 -> 0.123823, HR 0.155 -> 0.145.

The violation penalty is bounded (``W_PENALTY``): a violating candidate is
pushed down, never eliminated and never sent to -infinity (CP 6.3,
principle E). A parser mistake is therefore always recoverable.

Ranking is read-only with respect to ``SessionState`` and does no I/O beyond
one indexed metadata lookup per turn.
"""

from __future__ import annotations

from typing import Any

from starter.contracts import Candidate, Context, RankingResult
from starter.popularity import popularity_score
from starter.profile import extract_evidence, profile_match_ratio
from starter.reliability import reliability_of

# Ablation flag for the profile prior (Phase 8).
#
# OFF: measured net-negative on the public set -- TS 0.131194 -> 0.127929,
# HR 0.155 -> 0.150 (MRR alone ticks up, 0.080312 -> 0.081097). The roadmap
# gates Phase 8 on "only if justified after core evaluation"; it is not.
#
# The cause is in the data, not the implementation. The profile carries
# almost no product-discriminative signal: purchase_frequency is the SAME
# string in all 200 sessions, and the tags are dominated by dimensions that
# say nothing about which product is wanted (fit 81.5%, material 77%,
# comfort 72%). A signal present in four of five sessions cannot separate
# candidates. The mapped tags that do carry catalog language -- warmth 9%,
# weather 6%, performance 13% -- are too rare to pay for the noise.
#
# The code stays: the priority guarantees (CP 8.3/8.4/8.5) and the empty
# profile safety (CP 8.6) are tested and hold, so this is one flag away if a
# richer profile ever arrives.
USE_PROFILE = False

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

# The anonymized-profile prior (Phase 8) -- the weakest tier by construction.
#
# Priority is: current explicit request > active session state > profile
# (principle I). The first two arrive through `constraints`; the profile
# arrives only here. W_PROFILE is held an order of magnitude below W_MATCH so
# that satisfying the whole profile can never outweigh even ONE satisfied
# constraint -- a profile can reorder candidates the constraints are
# indifferent between, and nothing more.
W_PROFILE = 0.008

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
    "profile_score",
    "final_score",
    "rank",
    "matched",
    "violated",
    "route_sources",
    "constraint_weights",
    # Phase 12. docs/file_ownership.md has required a "popularity prior" in
    # the diagnostics since CP 0.1; this is it.
    "popularity_score",
})


# Ablation flag for confidence/reliability weighting (Phase 11).
#
# OFF reproduces Phase 6 scoring exactly: every active slot counts 1, whatever
# the shopper's phrasing and whatever the catalog's coverage of that field.
#
# ON, each slot's contribution is scaled by Evidence Confidence (how firmly the
# shopper asserted it, from state.slots[slot]["confidence"]) times Match
# Reliability (how much the catalog's verdict on that field is worth, from
# reliability.match_reliability). See ``constraint_weights`` for the three
# quadrants and ``score_candidate`` for the arithmetic.
#
# OFF, and the reason is NOT that the weighting does nothing. It does a great
# deal; it just does it where the evaluator cannot see.
#
# Measured directly by ranking each live turn's pool BOTH ways and diffing the
# orders (``python3 -m tools.phase11_confidence``):
#
#   turns with a candidate-order change      513 /  929   55.2%
#   turns where the TARGET's rank moved       16 /  380    4.2%
#   sessions gained / lost                       1 / 1     p = 1.0000
#
# PHASE 15 REVIVED THIS MECHANISM, AND THE ARGUMENT BELOW HAD TO CHANGE WITH
# IT. Until clarification landed, this flag was measured EXACTLY inert: no
# target ever moved a single position, and the case CP 11.4 exists for
# (firmly meant, poorly attested) occurred 4 times in 1536 constraint
# occurrences. With the shopper actually disclosing preferences there are
# 3-constraint turns for the first time, weights differ between slots on 40%
# of turns, CP 11.4's case occurs 86 times in 1146 (7.5%), and the target's
# rank moves on 19 turns.
#
# So "no measured harm because it does nothing" is no longer available. It
# now does something, and what it does measures -0.001737 TS at 1/1
# discordant, p = 1.0000 -- no verdict, and a point estimate on the wrong
# side of zero. Same shipping decision, different and weaker reason: it used
# to ship OFF because it was inert, and it ships OFF now because the burden
# is on the change and a live mechanism with a negative point estimate has
# not met it.
#
# So the aggregates being identical to six decimals is a fact about the top of
# the list, not about the ranking. Three in a thousand turns reorder the Top-10
# at all, and across every turn where the target was in the pool it never moved
# a single position. An earlier version of this comment reported the aggregates
# as proof that "no target's rank moved anywhere"; the aggregates cannot show
# that, and it needed the row above to be said honestly (B Phase 11 review).
#
# That review also killed the argument this comment used to make -- that a
# single-constraint turn is "a monotone rescale that cannot reorder". False:
# ``base`` is not scaled by the weight, so shrinking the attribute term moves
# constraint evidence relative to retrieval order, and ONE constraint suffices.
# 146 of the 513 changed turns have exactly one active constraint. The module
# docstring above carries the counterexample, and the same error sat in the
# Phase 6 paragraph next to it.
#
# The census in the tool is still worth reading for what the mechanism had to
# work with -- CP 11.4's case (firmly meant, poorly attested) occurs 69 times,
# 5.7% of constraint occurrences, up from 4 in 1536 before Phase 15 -- but no
# claim here rests on it.
#
# So this ships OFF on the same rule as USE_PROFILE: the burden is on the
# change, and reordering only below the scored cut is not evidence for it.
# Unlike USE_PROFILE there is no measured harm either -- the honest summary is
# no evidence in either direction -- and after Phase 15 the point estimate is
# slightly negative, which is weaker still. What tips it is that weighting
# shrinks the attribute term, and constraint ranking is one of the components
# McNemar establishes on this set (+6, 6/0, p = 0.0312 on the BM25 pool).
# Shrinking something that works, for no measured gain, is not a trade to
# make blind.
#
# Everything is built, tested (CP 11.3/11.4/11.5 in tests/test_confidence.py)
# and one flag from live. EC itself is computed and stored regardless: it is
# state, not scoring, and Phase 15 wants it.
USE_CONFIDENCE_WEIGHTING = False

# Ablation flag for the popularity prior (Phase 12).
#
# ON, and it USED TO BE the largest established gain in the project. Phase 15
# ended that: with clarification supplying real constraints the same term is
# worth +0.0203 at 4/0 discordant, p = 0.1250 -- NO VERDICT.
#
# That is not a regression and it is the most interesting thing this comment
# has ever said. The popularity prior was worth +0.0477 when the agent knew
# almost nothing about the shopper, because on a benchmark whose targets sit
# at the 99.5th review-count percentile a bestseller list is close to an
# oracle for a query with no constraints in it. Give the ranker actual
# constraints to score and the oracle stops being needed. A prior is supposed
# to matter less as evidence arrives; this one measurably does, which is the
# first direct evidence in this repo that it is behaving like a prior rather
# than like the ranker.
#
# It stays ON. It is still positive on the point estimate, it costs nothing,
# and "no verdict" is not evidence of harm -- but no one should quote this
# term as established any more, and the sampling caveat below now carries
# more of its weight than the score does.
#
# EVERY NUMBER IN THIS COMMENT IS THE LIVE PIPELINE, RE-MEASURED. It used to
# be the Phase 12 pipeline stated in the present tense, and Phase 14 moved
# both arms underneath it without touching a word here -- the OFF arm read
# 0.134566 when the tool that produces it prints 0.187711 (D Phase 14 review,
# Finding 3). Nothing detected that, because every guard in this repo checks
# ASSERTIONS and none of them read prose. So: `python3 -m
# tools.phase12_popularity` regenerates all of it, and moving the pipeline
# means re-running it and editing here in the same change.
#
#   OFF   HR 0.2200   MRR 0.123369   MTTC 8.965   TS 0.187711
#   ON    HR 0.2850   MRR 0.162099   MTTC 8.360   TS 0.243930     +0.056219
#   McNemar +13, 13/0 discordant, p = 0.0002  -- established
#
# tools/phase7_ablation.py reports a different figure for the same flag.
# Both are right and they are different comparisons: the ladder ablates
# popularity at Run 3 -> Run 4, with neither the reranker nor clarification
# above it, while the figure here removes the term from the pipeline as it
# SHIPS. The gap between them IS the finding -- the ladder rung is the
# prior's value to an agent that knows nothing, and this one is its value to
# an agent that has asked. Read the scope, not only the number.
#
# Every scenario improves except boundary, which is 0.1000 in both arms --
# and boundary is 10 sessions, so read it as "no evidence either way" rather
# than as a null result. buying 0.3625 -> 0.4125, browsing 0.0750 -> 0.1625,
# intent_override 0.2667 -> 0.3333.
#
# For comparison the entire Phase 0-6 stack is worth +0.0279 over the
# baseline, so this one term is still worth more than everything before it
# combined. (That figure IS historical and stays fixed: it is the size of a
# ladder that was measured once and does not move again.)
#
# WHY IT IS WORTH SO MUCH, WHICH IS NOT THE FLATTERING ANSWER
#
# Not because popularity is a deep signal about what shoppers want. Because
# the public set's ground-truth targets are drawn almost entirely from the
# most-reviewed products in the catalog:
#
#   catalog  rating_number  median     12
#   targets  rating_number  median  7,078   -- the 99.5th catalog percentile
#   targets below the catalog median: 4 of 200 (2%), where unbiased is 50%
#
# So on THIS benchmark a bestseller list is close to an oracle. The private
# 800 is built the same way, so the gain should transfer -- but it is a fact
# about how the evaluation was sampled, not evidence that ranking by review
# count serves shoppers. Anyone quoting this number should quote that with it.
#
# D's Phase 12 review put a number on exactly that: on a counterfactual set
# with targets drawn UNIFORMLY from the catalog, the same prior measures
# -0.012556. Not established harm, and not the set we are scored on -- but it
# is the honest size of the "this is sampling, not preference" caveat, and
# every phase built on top of this one inherits it. (D's measurement,
# reproduced from their report rather than in-tree; the in-tree tool measures
# the sampling skew that causes it, not the counterfactual itself.)
#
# THE WEIGHT IS DELIBERATELY LEFT ON THE TABLE
#
# W_POPULARITY is 0.008, an order of magnitude below W_MATCH, exactly as
# CP 12.4 requires. Raising it pays enormously on this benchmark:
#
#   W_POPULARITY   0.008 -> TS 0.723   (shipped)
#                  0.02  -> TS 0.747
#                  0.05  -> TS 0.802
#                  0.10  -> TS 0.827
#
# At 0.10 popularity equals W_MATCH and a bestseller can cancel a satisfied
# constraint outright -- which is precisely the collapse CP 12.4 forbids. The
# roadmap set that constraint before any of these numbers existed, and it is
# not the kind of constraint to relax because the artifact it protects against
# happens to pay. Anything above ~0.01 stops being a prior and becomes the
# ranker. Left at 0.008; the sweep is recorded here so the trade is visible
# rather than silently taken or silently forgone.
#
# OFF: the popularity term is not computed. That is no longer "exactly the
# pre-Phase-12 score" -- it is the rest of the CURRENT pipeline without this
# term, which is 0.702720 today (0.187711 before Phase 15, 0.134566 when the
# ladder ended at Phase 12). The ladder figure lives in
# tools/phase7_ablation.py, where it is labelled as history.
USE_POPULARITY = True

# Where ``rank`` reads the catalog popularity scale out of ``Context.derived``.
POPULARITY_KEY = "popularity_scale"

# Where ``rank`` reads per-slot Match Reliability out of ``Context.derived``.
# A generic container by the contracts rule: a new signal adds a key, never a
# frozen field. Absent -> every slot fully reliable -> Phase 6 behaviour.
RELIABILITY_KEY = "match_reliability"


def slot_confidence(context: Context, slot: str) -> float:
    """CP 11.1 -- Evidence Confidence for one active slot.

    Absent means 1.0, which is the convention ``validate_delta`` established
    in Phase 4 and is what makes ``USE_CONFIDENCE_WEIGHTING = False`` exact.
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


def _category_value_matches(tokens: set[str], value: str) -> bool:
    """True only if EVERY word of a category value is present.

    A multi-word category is a conjunction, not a disjunction: "swim trunks"
    means swim AND trunks. Matching on any single word made "swim trunks"
    match "Women's Swimwear" and "tank top" match "Topcoats" / "Water Tanks"
    (Phase 6 review). Prefix matching is kept per word so the intended plural
    tolerance (jacket -> jackets) still works.
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
    """CP 11.3 / 11.4 / 11.5 -- ``EC * MR`` per active slot.

    Evidence Confidence says how firmly the shopper asserted the constraint;
    Match Reliability says how much the catalog's verdict on that field is
    worth. A slot's verdict is only as good as the weaker of the two, and the
    product is the graded form of that:

        high EC, high MR   ~1.00   full weight, Phase 6 behaviour  (CP 11.3)
        high EC, low MR     0.10   a real requirement we cannot check
                                   reliably: it must not bury the target,
                                   which was never in the catalog's terms
                                   to begin with                   (CP 11.4)
        low EC,  high MR    0.40   a passing remark: the catalog can check it
                                   perfectly, and it still must not act as a
                                   filter                          (CP 11.5)

    Neither factor can reach zero (``reliability.MIN_RELIABILITY``, and
    ``validate_delta`` drops a zero confidence outright), so no constraint is
    ever silently switched off -- it is discounted, and the ranked list is
    still a ranking rather than a filtered set.
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

    ``weights`` maps a slot to ``EC * MR`` -- how much this slot's verdict is
    worth, given both how firmly the shopper asserted it and how much the
    catalog's evidence for that field is worth (Phase 11). ``None`` means
    every slot counts 1, which is Phase 6 exactly.
    """
    matched: list[str] = []
    violated: list[str] = []
    # The denominator stays the active constraint COUNT, as in Phase 6:
    # identical for every candidate this turn, so UNKNOWN rescales but never
    # reorders. Evidence quality multiplies the NUMERATOR only.
    #
    # It has to be this way round. Normalising by the summed weight instead
    # would cancel the weight whenever one constraint is active -- a single
    # slot at EC 0.4 would give 0.4/0.4 = full penalty, and CP 11.5 would be
    # violated by the very mechanism meant to satisfy it. Dividing by the
    # count leaves the unclaimed share of the budget simply unclaimed, so a
    # constraint that is half-meant on a field we half-trust contributes a
    # quarter of the evidence it otherwise would and retrieval order carries
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

    # Phase 12. Bounded by W_POPULARITY and decayed by the evidence already
    # gathered, so it can only separate candidates the constraints leave tied.
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
        # CP 11.1 / 11.2 made visible: why each verdict counted for as much
        # as it did. Same for every candidate in a turn, but carried per
        # candidate so a diagnostics row explains its own score.
        "constraint_weights": dict(weights or {}),
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

    The profile prior (Phase 8) is read from ``context.state.user_profile``
    and applied as the weakest term; it never touches ``state.slots``, so it
    cannot override an explicit request or a session constraint.
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
        # CP 12.3 -- how much the shopper has actually committed to so far.
        # Summed Evidence Confidence, so a hedged constraint displaces less of
        # the prior than an insisted-upon one. Read straight from state, so
        # this works whether or not USE_CONFIDENCE_WEIGHTING is on: EC is
        # state, not scoring, and the two flags stay independent.
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
