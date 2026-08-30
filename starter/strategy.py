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

Four rules, strongest evidence first (``classify_mode_with_reason`` names
which one fired):

    specific slot filled       a colour / material / brand / size / budget we
                               recognised and stored          -> buying
    requirement language       "a key requirement is", "must"  -> buying
    browsing language          "still exploring", "just ideas" -> browsing
    concrete unslotted detail  volunteered free text that is not soft,
                               filler or a restated category   -> buying

The last rule ranks BELOW the browsing declaration on purpose: it is a guess
about text we failed to parse, not something we recognised, and it is the one
rule that can misfire on a category name the extractor did not know. The
first three are what "concrete specifics win" means -- a recognised spec beats
"exploring", an unrecognised one does not.

The cues below are ordinary English, not the evaluator's phrasing -- keying
on simulator strings would classify the public set well and generalize to
nothing. Accuracy against the live dialogue is measured, not asserted:
``python3 -m tools.phase9_mode_accuracy``. It is currently 100% on all 200
sessions, which -- as that tool says in its own output -- is template coverage
on four opening templates, not evidence of generalization.

Nothing in the shipped agent reads the mode yet (see ``agent.respond``);
Phase 15 is the first consumer.
"""

from __future__ import annotations

from typing import Any

from starter.contracts import Context, Strategy
from starter.retrieval import DEFAULT_ROUTES
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
# "buckle closure", a "stainless steel band") does -- see
# ``_has_concrete_evidence``.
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

# Everything that is NOT evidence of a checkable spec. Browsing cues belong
# here too: "still exploring" is filler plus a browsing cue and must not read
# as volunteered concrete detail.
_VAGUE_TOKENS = SOFT_CUES | FILLER_CUES | BROWSING_CUES

# Route plan. UNIFORM across modes -- corrected after the Phase 9 review.
#
# The first version of this file made buying and browsing select different
# route sets and reported the difference as the checkpoint's gain. That was
# wrong: mode and route set covaried, so the experiment could not separate
# them. Holding the route set fixed and removing the classifier entirely
# showed the damage was the ATTRIBUTE route, uniformly -- nothing to do with
# buying vs browsing. Mode-adaptive route selection is worth +0.000298 over
# the uniform plan: one thirty-fourth of what a single flipped session is
# worth on this set. So the routes are fixed and the mode does NOT gate them.
#
# The route set itself, and what is and is not established about it, is stated
# once in ``retrieval.DEFAULT_ROUTES`` -- and taken from there rather than
# restated, so the two cannot drift apart.
#
# `classify_mode` is kept because Phase 15 wants it (how hard to push for a
# clarification differs between a shopper who has named specifics and one
# still exploring), not because it earns anything here.
_ROUTES = list(DEFAULT_ROUTES)
_WEIGHTS = {name: 1.0 for name in _ROUTES}
_ROUTE_PLAN: dict[str, tuple[list[str], dict[str, float]]] = {
    BUYING: (_ROUTES, _WEIGHTS),
    BROWSING: (_ROUTES, _WEIGHTS),
    UNKNOWN: (_ROUTES, _WEIGHTS),
}


def _slots(context: Context) -> dict:
    """``state.slots``, or an empty mapping if it is not one.

    The single-writer invariant means only ``state.update_state`` builds this,
    and it always builds a dict -- so this is unreachable through the shipped
    path. It is here because classification must degrade to "nothing known"
    rather than raise: a mode is a workflow hint, and no hint is worth a
    500 on a turn (principle E). Same reasoning as the ``normalized`` type
    guard in ``_evidence_tokens``.
    """
    return context.state.slots if isinstance(context.state.slots, dict) else {}


def _specific_slot_count(context: Context) -> int:
    slots = _slots(context)
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


def _restates_a_slot(token: str, slot_tokens: set[str]) -> bool:
    """True if ``token`` is just a surface form of an already-captured value.

    ``state.update_evidence`` strips extracted slot VALUES from the residual,
    but it strips the CANONICAL value: "I'm looking for jackets" stores
    ``category=jacket`` and leaves ``jackets`` in the evidence text. Naming a
    category is how both modes open (see ``SPECIFIC_SLOTS``), so counting that
    leftover plural as volunteered detail would make every session buying.

    Prefix matching in both directions covers the plural/singular pairs the
    canonical form differs by, without a stemmer that would have to agree with
    the three others in this codebase.
    """
    return any(
        token.startswith(value) or value.startswith(token)
        for value in slot_tokens
    )


def _slot_tokens(context: Context) -> set[str]:
    """Every token of every value already captured in a slot."""
    out: set[str] = set()
    for slot in _slots(context).values():
        if not isinstance(slot, dict):
            continue
        for value in slot.get("values", ()) or ():
            out.update(terms(str(value)))
    return out


def _has_concrete_evidence(context: Context) -> bool:
    """True when volunteered free text names something checkable.

    Restored after the Phase 9 review (C). ``normalized`` evidence has already
    had the extracted slot values, the override plumbing and the slot-marker
    words removed by ``state.update_evidence``, so what survives is close to
    genuine volunteered detail. A surviving token is concrete unless it is

      * soft quality / occasion language, filler, or an explicit browsing cue
        -- which is what keeps "still exploring" from reading as a spec, the
        gap the first version of this function had; or
      * a restatement of a value already captured in a slot.

    Superseded evidence is excluded via ``_evidence_tokens``, so an override
    that replaces the detail also withdraws the buying signal (CP 9.5).
    """
    slot_tokens = _slot_tokens(context)
    for token in _evidence_tokens(context):
        if token in _VAGUE_TOKENS:
            continue
        if _restates_a_slot(token, slot_tokens):
            continue
        return True
    return False


def classify_mode_with_reason(context: Context) -> tuple[str, str]:
    """``classify_mode``, plus the name of the rule that decided.

    The reason is for auditing, not for control flow: it lets
    ``tools/phase9_mode_accuracy.py`` report WHICH rule carries the accuracy
    instead of quoting a single percentage. An accuracy figure produced by one
    rule firing on one message template says much less than the same figure
    spread across several, and the difference is invisible in the percentage.
    """
    # A turn that declines to add information is not the shopper's own
    # vocabulary -- reading its wording as a browsing cue would let the
    # harness's phrasing decide the mode. On the public set 90% of turns are
    # such a non-answer, and "options" in it collided with BROWSING_CUES.
    message = "" if is_non_answer(context.user_message) else context.user_message
    tokens = set(terms(message))
    if _specific_slot_count(context) >= 1:
        return BUYING, "specific slot filled"
    # Requirement language is read from the whole accumulated session, not
    # just this turn. Stating a requirement is not something a shopper
    # un-does by going quiet, and on this harness they go quiet immediately:
    # once the agent stops learning anything, every later turn is the same
    # "no new information" reply. Reading only the current turn would flip a
    # buying session to browsing the moment it stopped talking.
    evidence = _evidence_tokens(context)
    if tokens & REQUIREMENT_CUES or evidence & REQUIREMENT_CUES:
        return BUYING, "requirement language"
    # An explicit "I'm still exploring" outranks the fallback below, and is
    # read from accumulated evidence for the same reason requirement language
    # is. It cannot pick up harness phrasing: update_evidence refuses to store
    # a non-answer, so only the shopper's own words reach here.
    if tokens & BROWSING_CUES or evidence & BROWSING_CUES:
        return BROWSING, "browsing language"
    # A concrete detail the extraction vocabulary has no slot for -- "buckle
    # closure", "stainless steel band". Still a spec, and the CP 9.2 case the
    # override openings are made of.
    #
    # This ranks BELOW the browsing declaration, unlike a filled slot, because
    # it is a guess about text we failed to parse rather than something we
    # recognised. The harness opens every browsing session by naming a raw
    # catalog category ("Basketball Men"), which the curated category
    # vocabulary does not extract; those leftover words are indistinguishable
    # from volunteered detail by inspection, and reading them as a spec
    # classified 94% of browsing turns as buying
    # (``python3 -m tools.phase9_mode_accuracy``). Naming a category is how
    # BOTH modes open, so when the shopper has also said they are exploring,
    # the declaration is the better evidence.
    if _has_concrete_evidence(context):
        return BUYING, "concrete unslotted detail"
    return BROWSING, "nothing volunteered"


def classify_mode(context: Context) -> str:
    """CP 9.1 / 9.2 -- ``buying`` or ``browsing`` for the CURRENT state.

    Recognised specifics win: once a shopper names a colour, material, brand,
    size or budget, they are buying even if they also say "exploring". Next,
    requirement language. Then an explicit browsing declaration. Only then the
    weakest rule -- concrete-looking free text we could not slot. A shopper who
    has volunteered nothing but a category is exploring.
    """
    return classify_mode_with_reason(context)[0]


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
