"""Anonymized-profile evidence (Phase 8).

The profile the evaluator supplies at ``reset`` is deliberately thin:

    purchase_frequency   constant "3-4 prior purchases" across all 200 public
                         sessions -- zero discriminative signal
    average_prior_rating / rating_style
                         how this shopper rates, not what they want
    preference_tags      the only product-facing field: 9 abstract dimensions
                         (fit 81.5%, material 77%, comfort 72%, style 50.5%,
                         durability 23.5%, performance 13%, warmth 9%,
                         weather 6%)
    summary              prose restatement of the tags

Two consequences shape this module.

First, a tag says which DIMENSION the shopper cares about, never which VALUE
they want. "cares about material" does not imply leather. So a tag can only
ever be a weak prior over product language, never a constraint -- and tags
that carry no catalog-language signal at all (``fit``, ``material``,
``style``) are deliberately mapped to nothing rather than guessed at.

Second, and structurally: profile evidence NEVER reaches ``state.slots``.
It is read at ranking time as a separate, weakest-tier term. That is what
makes principle I hold by construction rather than by careful bookkeeping --
a profile cannot override an explicit request (CP 8.3), a session constraint
(CP 8.4), or a previous turn's state (CP 8.5), because it never enters the
channel those live in.
"""

from __future__ import annotations

from typing import Any

# Profile tag -> catalog language that evidences it. Only tags with real
# product-text signal are mapped; the rest resolve to nothing on purpose.
TAG_TERMS: dict[str, frozenset[str]] = {
    "warmth": frozenset({
        "warm", "insulated", "thermal", "fleece", "sherpa", "winter", "cozy",
    }),
    "weather": frozenset({
        "waterproof", "water", "resistant", "rain", "windproof", "weatherproof",
    }),
    "performance": frozenset({
        "athletic", "performance", "training", "sport", "running", "workout",
        "wicking", "breathable",
    }),
    "durability": frozenset({
        "durable", "rugged", "reinforced", "sturdy", "heavy", "lasting",
    }),
    "comfort": frozenset({
        "comfortable", "comfort", "soft", "cushioned", "padded", "lightweight",
    }),
    # Intentionally unmapped -- too abstract to ground in catalog text without
    # inventing intent: "fit", "material", "style", "general shopping".
}

# Every term any tag can contribute, for cheap catalog-side extraction.
ALL_TRAIT_TERMS: frozenset[str] = frozenset(
    term for terms in TAG_TERMS.values() for term in terms
)


def extract_evidence(user_profile: object) -> dict[str, Any]:
    """CP 8.2 -- the usable part of a profile, normalized.

    Returns ``{"tags": [...], "terms": frozenset(...)}``. Safe on ``None``,
    ``{}``, a non-dict, or a profile whose fields are the wrong type
    (CP 8.6): anything unusable yields empty evidence, never an exception.
    """
    if not isinstance(user_profile, dict):
        return {"tags": [], "terms": frozenset()}

    raw = user_profile.get("preference_tags")
    tags = [
        tag.strip().lower()
        for tag in raw
        if isinstance(tag, str) and tag.strip()
    ] if isinstance(raw, list) else []

    # Deduplicate, keep declaration order, and keep only mapped tags.
    seen: list[str] = []
    for tag in tags:
        if tag in TAG_TERMS and tag not in seen:
            seen.append(tag)

    terms = frozenset(term for tag in seen for term in TAG_TERMS[tag])
    return {"tags": seen, "terms": terms}


def profile_match_ratio(evidence: dict[str, Any], traits: set[str]) -> float:
    """Fraction of the shopper's mapped tags this product evidences.

    ``traits`` is the product's recognised trait vocabulary from
    ``catalog_meta``. A product the catalog says nothing about scores 0.0 --
    absent evidence is never counted against it, matching the UNKNOWN
    treatment in constraint ranking.
    """
    tags = evidence.get("tags") or []
    if not tags or not traits:
        return 0.0
    hits = sum(1 for tag in tags if TAG_TERMS[tag] & traits)
    return hits / len(tags)
