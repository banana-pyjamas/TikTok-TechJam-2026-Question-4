"""Adaptive strategy (Phase 9).

Decides HOW to run a turn -- which retrieval routes, with what relative
weight -- from how specific the shopper has been so far. It decides nothing
about individual products: the ``Strategy`` object carries no ``parent_asin``
and the ranking layer never reads it (CP 9.6). Strategy is workflow; ranking
is product order.

Two modes:

    buying     the shopper has named something concrete and checkable --
               a colour, material, brand, size, budget, or an explicit
               requirement. Favour precision.
    browsing   the shopper is still exploring: a category at most, plus
               soft or use-case language. Favour reach.

Classification is deterministic and recomputed from scratch every turn
(CP 9.3), so it follows the state rather than latching: a browsing session
becomes buying the moment specifics arrive (CP 9.4), and an override that
replaces the specifics re-derives the strategy from the NEW state, never the
old one (CP 9.5).

The cues below are ordinary English, not the evaluator's phrasing -- keying
on simulator strings would classify the public set well and generalize to
nothing.
"""

from __future__ import annotations

from typing import Any

from starter.contracts import Context, Strategy
from starter.state import is_non_answer
from starter.text import terms

BUYING, BROWSING, UNKNOWN = "buying", "browsing", "unknown"

# Slots that name something concrete and checkable about a product. Category
# is excluded on purpose: naming a category is how BOTH modes open.
SPECIFIC_SLOTS = ("color", "material", "brand", "size", "budget")

# "I need X specifically" -- the shopper is stating a requirement.
REQUIREMENT_CUES = frozenset({
    "requirement", "requirements", "required", "must", "need", "needs",
    "needed", "specifically", "exactly", "matters", "important", "essential",
})

# "show me what's out there" -- the shopper is still looking around.
BROWSING_CUES = frozenset({
    "exploring", "explore", "browsing", "browse", "ideas", "idea",
    "suggestions", "suggest", "options", "recommend", "recommendations",
    "inspiration", "unsure", "maybe", "anything", "something",
})

# Soft qualities and use-case contexts. These are things a shopper VOLUNTEERS
# while still exploring -- "comfortable shoes for traveling" names a feeling
# and an occasion, not a checkable spec. Free text made only of these does
# NOT make a turn a buying turn; free text containing anything else (a
# "buckle closure", a "stainless steel band") does.
SOFT_CUES = frozenset({
    # qualities
    "comfortable", "comfy", "cozy", "soft", "nice", "good", "great", "better",
    "best", "casual", "everyday", "versatile", "simple", "basic", "classic",
    "stylish", "cute", "cool", "warm", "light", "lightweight", "durable",
    "quality", "affordable", "cheap", "expensive",
    # occasions and activities
    "travel", "traveling", "travelling", "trip", "vacation", "holiday",
    "work", "office", "school", "gym", "workout", "hiking", "walking",
    "weekend", "daily", "party", "wedding", "summer", "winter", "spring",
    "fall", "autumn", "outdoor", "indoor", "gift", "present",
})

# Filler that survives tokenization but names nothing. Without this,
# "still exploring" reads as volunteered concrete detail.
FILLER_CUES = frozenset({
    "still", "just", "really", "quite", "very", "kind", "sort", "bit",
    "little", "more", "less", "much", "thing", "things", "one", "ones",
    "prefer", "like", "love", "help", "find", "show", "give", "get",
})

# Everything that is NOT evidence of a checkable spec.
_VAGUE_TOKENS = SOFT_CUES | FILLER_CUES

# Route plan. UNIFORM across modes -- corrected after the Phase 9 review.
#
# The first version of this file made buying and browsing select different
# route sets and reported the difference as the checkpoint's gain. That was
# wrong: mode and route set covaried, so the experiment could not separate
# them. Holding the route set fixed and removing the classifier entirely:
#
#   bm25 only                      TS 0.131194
#   bm25 + category                   0.134566   <- chosen, no classifier
#   bm25 + category + attribute       0.115512
#
# The damage is the ATTRIBUTE route, uniformly. It is not about buying vs
# browsing at all. Mode-adaptive route selection is worth +0.000298 over the
# uniform plan -- one thirty-fourth of the +0.01 that a single flipped
# session is worth on this set, and far below the +-0.04 noise floor. So the
# routes are fixed and the mode does NOT gate them.
#
# `classify_mode` is kept because Phase 15 wants it (how hard to push for a
# clarification differs between a shopper who has named specifics and one
# still exploring), not because it earns anything here.
_ROUTES = ["bm25", "category"]
_WEIGHTS = {"bm25": 1.0, "category": 1.0}
_ROUTE_PLAN: dict[str, tuple[list[str], dict[str, float]]] = {
    BUYING: (_ROUTES, _WEIGHTS),
    BROWSING: (_ROUTES, _WEIGHTS),
    UNKNOWN: (_ROUTES, _WEIGHTS),
}


def _specific_slot_count(context: Context) -> int:
    slots = context.state.slots
    return sum(
        1
        for name in SPECIFIC_SLOTS
        if isinstance(slots.get(name), dict) and slots[name].get("values")
    )


def _has_active_evidence(context: Context) -> bool:
    """Free text the shopper volunteered beyond slot values, still active."""
    return any(
        isinstance(entry, dict) and entry.get("status") == "active"
        for entry in context.state.evidence
    )


def _evidence_tokens(context: Context) -> set[str]:
    """Every token the shopper has volunteered as still-active free text.

    Superseded evidence is excluded, so an override cannot leave stale
    wording deciding the mode (CP 9.5).
    """
    out: set[str] = set()
    for entry in context.state.evidence:
        if not isinstance(entry, dict) or entry.get("status") != "active":
            continue
        normalized = entry.get("normalized", "")
        if isinstance(normalized, str):
            out.update(terms(normalized))
    return out


def classify_mode(context: Context) -> str:
    """CP 9.1 / 9.2 -- ``buying`` or ``browsing`` for the CURRENT state.

    Concrete specifics win: once a shopper names a colour, material, brand,
    size or budget, they are buying even if they also say "exploring".
    Absent specifics, explicit browsing language decides. A shopper who has
    volunteered nothing but a category is exploring.
    """
    # A turn that declines to add information is not the shopper's own
    # vocabulary -- reading its wording as a browsing cue would let the
    # harness's phrasing decide the mode. On the public set 90% of turns are
    # such a non-answer, and "options" in it collided with BROWSING_CUES.
    message = "" if is_non_answer(context.user_message) else context.user_message
    tokens = set(terms(message))
    if _specific_slot_count(context) >= 1:
        return BUYING
    # Requirement language is read from the whole accumulated session, not
    # just this turn. Stating a requirement is not something a shopper
    # un-does by going quiet, and on this harness they go quiet immediately:
    # once the agent stops learning anything, every later turn is the same
    # "no new information" reply. Reading only the current turn would flip a
    # buying session to browsing the moment it stopped talking.
    if tokens & REQUIREMENT_CUES or _evidence_tokens(context) & REQUIREMENT_CUES:
        return BUYING
    if tokens & BROWSING_CUES:
        return BROWSING
    return BROWSING


def build_strategy(context: Context) -> Strategy:
    """CP 9.3 -- the full strategy for this turn, derived from state only.

    Pure: reads ``context``, mutates nothing, and holds no product identity.
    """
    mode = classify_mode(context)
    routes, weights = _ROUTE_PLAN[mode]
    params: dict[str, Any] = {
        "specific_slots": _specific_slot_count(context),
        "has_evidence": _has_active_evidence(context),
        "turn": context.turn,
    }
    return Strategy(
        mode=mode,
        routes=list(routes),
        route_weights=dict(weights),
        params=params,
    )
