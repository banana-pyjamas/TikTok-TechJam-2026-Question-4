"""Popularity prior (Phase 12).

A weak tiebreak for candidates the shopper's own constraints cannot separate,
and NOTHING more. When we know nothing about what someone wants, the product
50,000 people reviewed is a better guess than the one nobody did; the moment
they tell us something, that guess is worth less than what they said.

Four checkpoints:

  CP 12.1  feature   ``log1p(rating_number)``, precomputed per product into
                     ``product_meta`` and normalised against the catalog.
  CP 12.2  missing   a product with no usable review count is treated as
                     TYPICAL, not as unpopular.
  CP 12.3  decay     the prior fades as the shopper gives evidence -- it is a
                     prior, and a prior is what you use before evidence.
  CP 12.4  no domination  a specific query must never collapse into a
                     bestseller list.

This module is deliberately PURE -- it imports nothing from ``starter``.
``catalog_meta`` calls ``popularity_feature`` while building its rows, so any
import back the other way would be a cycle; the catalog-wide statistics that
normalise the feature therefore live in ``catalog_meta.popularity_scale``,
beside the table they query. (``reliability`` queries the same table from
outside for the same purpose, and can only do so because nothing in
``catalog_meta`` depends on it.)

WHY log1p

Review counts span 1 to 408,371 on the frozen catalog, with a median of 12 --
four orders of magnitude, concentrated at the bottom. Raw counts would make
the ranking a function of one outlier. ``log1p`` compresses that to 0.69-12.92
with a median of 2.56 and, unlike ``log``, is defined at zero, so a product
with no reviews yet is representable rather than an error.

WHAT CP 12.4 ACTUALLY RESTS ON -- AND WHAT IT DOES NOT

Stated precisely, because an earlier version of this docstring overstated it
(C Phase 12 review). What holds is a COMPONENT BOUND:

    W_POPULARITY (0.008)  <<  W_MATCH (0.10)

so the entire popularity term is worth less than the attribute term a single
satisfied constraint contributes, and CP 12.3's decay shrinks it further the
moment any constraint exists.

That is a bound on two components, NOT a proof about final ranking outcomes.
``base_score`` varies per candidate and is no part of the bound, so a
sufficiently large retrieval-score gap can still put a non-matching candidate
above a matching one -- correctly, and for reasons that have nothing to do
with popularity. The claim "a bestseller cannot out-rank a product that
matches" was wrong and is withdrawn.

The empirical claim is separate and narrower: checked at FULL-POOL scope
across the live dialogue, popularity never inverted a match/non-match
ordering. That is an observed absence on this dataset, not a structural
proof for every base-score distribution
(``python3 -m tools.phase12_popularity``).

THE BOUND IS CONDITIONAL ON USE_CONFIDENCE_WEIGHTING BEING OFF

With Phase 11 weighting ON, a match's attribute term is scaled by EC * MR and
divided by the constraint count, so it is no longer bounded below by
``W_MATCH``. Two hedged constraints on poorly-attested slots give a matching
candidate ``0.10 * 0.04 / 2 = 0.002`` while the prior can still supply
``0.008 / 1.8 = 0.0044`` -- the inequality inverts, and CP 12.4's arithmetic
guarantee does not hold (D Phase 12 review, P4). USE_CONFIDENCE_WEIGHTING
ships False, so this is latent rather than live; ``tests/test_popularity.py``
pins it so it cannot be enabled without someone seeing it. Turning that flag
on requires re-deriving W_POPULARITY, not just flipping a boolean.

The same construction as the Phase 8 profile prior (``ranking.W_PROFILE``),
and it inherits the same caveat.
"""

from __future__ import annotations

import math
from typing import Any

# The weight of the whole prior, before decay. An order of magnitude below
# W_MATCH (0.10) so that satisfying one constraint always beats being the most
# reviewed product in the catalog -- that inequality IS CP 12.4, and it holds
# by construction rather than by luck.
W_POPULARITY = 0.008

# Normalisation reference, when the catalog cannot supply one.
DEFAULT_SCALE = 13.0

# What a product with no usable review count is worth, as a fraction of the
# normalised range, when the catalog supplies no median either. Mid-scale:
# absent data means "typical", never "worst" (CP 12.2, principle D).
DEFAULT_MISSING = 0.5


def popularity_feature(rating_number: object) -> float | None:
    """CP 12.1 / 12.2 -- ``log1p(rating_number)``, or ``None`` if unusable.

    ``None`` rather than ``0.0`` on purpose: zero is a real, meaningful value
    (a product with no reviews yet) and must stay distinguishable from "the
    catalog did not say". Conflating them is exactly how missing metadata
    turns into a penalty.

    Negative and non-finite counts are unusable, not clamped -- a negative
    review count means the field is not what we think it is, and guessing is
    worse than declining to score it.
    """
    if isinstance(rating_number, bool) or rating_number is None:
        return None
    try:
        count = float(rating_number)
    except (TypeError, ValueError):
        return None
    if count != count or math.isinf(count) or count < 0.0:
        return None
    return math.log1p(count)


def normalized(feature: object, scale: dict[str, float] | None) -> float:
    """A product's popularity in ``[0, 1]``.

    ``feature`` is the stored ``log1p`` value, or ``None``/absent when the
    catalog gave no usable count -- in which case the result is the catalog
    MEDIAN, so an unrated product sits with the typical ones instead of at the
    bottom (CP 12.2).

    Total: any malformed scale or feature degrades to a neutral value rather
    than raising, because this sits on the scoring path.
    """
    if not isinstance(scale, dict) or not scale:
        return 0.0
    reference = scale.get("scale")
    if not isinstance(reference, (int, float)) or reference != reference or reference <= 0:
        reference = DEFAULT_SCALE
    if isinstance(feature, bool) or not isinstance(feature, (int, float)) \
            or feature != feature:
        fallback = scale.get("missing")
        if not isinstance(fallback, (int, float)) or fallback != fallback:
            return DEFAULT_MISSING
        feature = fallback
    return max(0.0, min(1.0, float(feature) / float(reference)))


def evidence_decay(total_evidence: float) -> float:
    """CP 12.3 -- how much of the prior survives, given the evidence so far.

    ``1 / (1 + evidence)``. With nothing said the prior is at full strength;
    one firmly-stated constraint halves it, two thirds it, and it approaches
    zero without ever reaching it.

    Parameter-free on purpose. Any curve with the right shape needs a rate
    constant, and nothing measured in this repo would justify a particular
    one; the reciprocal is the shape with no knob. ``total_evidence`` is the
    summed Evidence Confidence of the active constraints, so a hedged
    constraint displaces less of the prior than an insisted-upon one -- which
    is the same reading of EC that Phase 11 uses, and works whether or not
    Phase 11's scoring flag is on, since EC is state rather than scoring.
    """
    if not isinstance(total_evidence, (int, float)) or total_evidence != total_evidence:
        return 1.0
    return 1.0 / (1.0 + max(0.0, float(total_evidence)))


def popularity_score(
    meta: dict[str, Any] | None,
    scale: dict[str, float] | None,
    total_evidence: float,
) -> float:
    """The additive popularity term for one candidate.

    ``W_POPULARITY * normalized(feature) * decay(evidence)``. Bounded above by
    ``W_POPULARITY``, which is an order of magnitude below one satisfied
    constraint -- see the module docstring on CP 12.4.
    """
    if not isinstance(scale, dict) or not scale:
        return 0.0
    feature = meta.get("popularity") if isinstance(meta, dict) else None
    return (W_POPULARITY
            * normalized(feature, scale)
            * evidence_decay(total_evidence))
