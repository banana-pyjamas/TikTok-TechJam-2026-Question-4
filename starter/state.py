"""Deterministic session-state manager (Phase 2 + Phase 3 + Phase 4).

Single authoritative writer of ``SessionState``. Given a raw user message it:

* extracts structured slot values with keyword / regex rules only -- no LLM,
  no network (principle J: the core must work offline);
* validates the delta before it can touch state -- unknown slots, invalid
  operations, and out-of-range / NaN confidences are rejected; extractor
  failure yields an empty delta and the previous state is kept intact
  (Phase 4). ``validate_delta`` is a no-op on the deterministic extractor's
  output; it guards a future / fuzzy / LLM delta source;
* detects intent-override cues deterministically ("actually" -> REPLACE,
  "also" -> ADD, "not leather" -> REMOVE) and applies the operation
  slot-specifically -- an override to one slot never disturbs the others
  (CP 3.2 golden);
* accumulates the result into ``state.slots``; superseded values are marked
  in ``state.provenance``;
* records every change in ``state.provenance``;
* keeps distilled residual free-text as ``state.evidence``. ``normalized`` is
  content tokens minus override cues, plumbing words, slot-markers and every
  extracted slot value, so an entry never carries a structured constraint --
  superseding one slot value cannot erase unrelated intent in the same
  entry (CP 2.8 / CP 3.6 / D Finding 1). Entries:
  ``{"turn", "text", "normalized", "status": "active"}``.

Slot value schema (approved at CP 2.1):

    state.slots[name] = {"values": list[str], "cardinality": "single" | "multi"}

``budget`` additionally carries ``"bounds": {"min": float|None, "max": float|None}``.
Single-valued slots hold 0-1 entries in ``values``; the uniform list shape lets
Phase 3 REPLACE / ADD / REMOVE operate without isinstance branching, and Phase 11
EC / MR attach as extra keys on the same dict.

Retrieval and ranking never call into this module -- they treat ``SessionState``
as read-only.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from starter.contracts import SessionState
from starter.text import STOPWORDS, TOKEN_RE, terms

_VALID_OPS = {"SET", "REPLACE", "ADD", "REMOVE"}
_VALID_CUES = {"replace", "add"}

# Slot-marker words that carry no standalone free-text signal, so they never
# make a message "residual" for evidence purposes.
_EVIDENCE_MARKER_TOKENS = {
    "size", "budget", "brand", "color", "colour", "material", "priced",
    "price", "cost", "costs", "dollars", "usd", "bucks", "under", "around",
    "about", "approximately", "roughly", "over", "above", "below",
}

# Conversational non-answers -- the customer declining to add information or
# telling the agent to proceed. These carry no user intent and must not be
# stored as evidence (D Finding 2). Generic phrasing, not tied to any one
# simulator string.
_NON_ANSWER_RE = re.compile(
    r"\bask me about"
    r"|\buse your judg"
    r"|\bnot quite right"
    r"|\bthose options"
    r"|\b(?:do not|don't|no|any|without) (?:really )?(?:have )?(?:an? )?"
    r"(?:additional |particular |strong |specific |real )?preference",
    re.I,
)

# Phase 3 override cues, detected deterministically on the raw message.
# REPLACE cue -> the new value supersedes the current one; ADD cue -> keep
# both. Negation directly before a known value -> REMOVE it.
_REPLACE_CUE_RE = re.compile(
    r"\b(?:actually|instead|rather|"
    r"scratch that|forget (?:the|that|my|about)|"
    r"changed? my mind|second thought|"
    r"no,? wait|wait,? no|"
    r"make it|change (?:it |that |this )?to|switch to|swap|replace)\b",
    re.I,
)
_ADD_CUE_RE = re.compile(
    r"\b(?:also|plus|as well|in addition|additionally|and also|along with)\b", re.I
)
# Removal verbs. "drop" is deliberately excluded -- it is also a product noun
# ("drop earrings", "drop-waist dress"), and a false REMOVE deletes a wanted
# constraint (D Finding 3). "remove" / "without" / "not" / "no" still cover it.
_STRONG_NEGATIONS = {
    "without", "remove", "lose", "skip", "minus", "nix", "exclude", "ditch",
}
_ADJACENT_NEGATIONS = {"not", "no"}

# Override-plumbing words: they appear in "actually, ignore my earlier
# preference, ..." style messages and are not free-text evidence.
_OVERRIDE_PLUMBING = {
    "actually", "instead", "rather", "ignore", "forget", "earlier", "previous",
    "prior", "preference", "preferences", "mind", "wait", "scratch",
    "nevermind", "disregard", "what", "need", "needed",
}

def is_non_answer(message: object) -> bool:
    """True when a turn declines to add information rather than stating a
    preference -- "ask me about one specific attribute", "no preference".

    Public because more than one layer needs it: evidence distillation must
    not store such a turn, and the strategy classifier must not read its
    wording as the shopper's own vocabulary.
    """
    return isinstance(message, str) and bool(_NON_ANSWER_RE.search(message))


SLOT_CARDINALITY: dict[str, str] = {
    "category": "single",
    "size": "single",
    "brand": "single",
    "budget": "single",
    "color": "multi",
    "material": "multi",
}


# --------------------------------------------------------------------------
# Deterministic extractors (keyword / regex only).
# --------------------------------------------------------------------------

# Curated keyword -> canonical category. Weak by design; Phase 5 grounds
# category against the catalog. First matching token in the message wins.
_CATEGORY_KEYWORDS: dict[str, str] = {
    "jacket": "jacket", "jackets": "jacket", "coat": "coat", "coats": "coat",
    "parka": "coat", "blazer": "blazer", "suit": "suit", "vest": "vest",
    "shirt": "shirt", "shirts": "shirt", "tshirt": "shirt", "tee": "shirt",
    "blouse": "blouse", "top": "top", "tank": "tank top",
    "sweater": "sweater", "sweaters": "sweater", "pullover": "sweater",
    "cardigan": "cardigan", "hoodie": "hoodie", "hoodies": "hoodie",
    "sweatshirt": "sweatshirt",
    "dress": "dress", "dresses": "dress", "gown": "dress",
    "skirt": "skirt", "skirts": "skirt",
    "pants": "pants", "trousers": "pants", "chinos": "pants",
    "jeans": "jeans", "shorts": "shorts", "leggings": "leggings",
    "tights": "tights", "joggers": "joggers", "sweatpants": "sweatpants",
    "shoe": "shoes", "shoes": "shoes", "sneaker": "sneakers",
    "sneakers": "sneakers", "trainers": "sneakers",
    "boot": "boots", "boots": "boots", "sandal": "sandals",
    "sandals": "sandals", "heel": "heels", "heels": "heels",
    "loafers": "loafers", "flats": "flats", "slippers": "slippers",
    "sock": "socks", "socks": "socks",
    "hat": "hat", "hats": "hat", "cap": "cap", "beanie": "beanie",
    "glove": "gloves", "gloves": "gloves", "mittens": "gloves",
    "scarf": "scarf", "scarves": "scarf", "belt": "belt", "belts": "belt",
    "tie": "tie", "ties": "tie",
    "watch": "watch", "watches": "watch", "ring": "ring", "rings": "ring",
    "necklace": "necklace", "necklaces": "necklace", "pendant": "necklace",
    "bracelet": "bracelet", "bracelets": "bracelet",
    "earring": "earrings", "earrings": "earrings",
    "bag": "bag", "bags": "bag", "backpack": "backpack",
    "purse": "purse", "handbag": "handbag", "tote": "tote",
    "wallet": "wallet", "wallets": "wallet",
    "bra": "bra", "bras": "bra", "underwear": "underwear",
    "swimsuit": "swimsuit", "bikini": "bikini", "trunks": "swim trunks",
    "pajamas": "pajamas", "robe": "robe",
}

_COLOR_ALIASES = {"grey": "gray"}
_COLORS = {
    "black", "white", "blue", "navy", "red", "pink", "green", "olive",
    "brown", "tan", "beige", "gray", "grey", "silver", "gold", "purple",
    "violet", "yellow", "orange", "maroon", "teal", "cream", "ivory",
    "khaki", "burgundy", "charcoal", "turquoise", "coral", "lavender",
}

_MATERIALS = {
    "cotton", "polyester", "nylon", "leather", "wool", "cashmere", "silk",
    "linen", "denim", "spandex", "elastane", "rayon", "suede", "fleece",
    "velvet", "satin", "chiffon", "corduroy", "canvas", "merino", "acrylic",
    "microfiber", "mesh", "flannel", "tweed", "jersey",
}

_SIZE_PREFIXED_RE = re.compile(
    r"\bsize\s+(\d{1,2}(?:\.5)?|xx-?small|x-?small|xs|xx-?large|x-?large|xl|xxl|xxxl|small|medium|large|s|m|l)\b",
    re.I,
)
_SIZE_STANDALONE_RE = re.compile(
    r"\b(xx-?small|x-?small|xs|xx-?large|x-?large|xxl|xxxl|xl)\b", re.I
)
_SIZE_NORMALIZE = {
    "xx-small": "xxs", "xxsmall": "xxs", "x-small": "xs", "xsmall": "xs",
    "xx-large": "xxl", "xxlarge": "xxl", "x-large": "xl", "xlarge": "xl",
    "small": "s", "medium": "m", "large": "l",
}

# Known brands; multi-word entries matched as phrases (longest first).
# Brands that are also ordinary English words (gap, lee, coach, champion,
# polo, columbia, fossil) are deliberately excluded -- on this dataset they
# are pure false-positive surface for near-zero yield (D Finding 3).
_BRANDS = {
    "nike", "adidas", "puma", "reebok", "new balance", "asics", "brooks",
    "saucony", "converse", "vans", "skechers", "crocs", "timberland", "ugg",
    "birkenstock", "clarks", "dr martens", "under armour", "the north face",
    "patagonia", "carhartt", "levi", "levis", "wrangler",
    "old navy", "uniqlo", "zara", "h&m", "hanes", "fila",
    "calvin klein", "tommy hilfiger", "ralph lauren", "lacoste",
    "gucci", "prada", "michael kors", "kate spade",
    "citizen", "seiko", "casio", "timex", "swatch",
}
_BRAND_NORMALIZE = {"levis": "levi"}
_BRAND_PHRASE_RE = re.compile(
    r"(?<![a-z])(" + "|".join(
        re.escape(b) for b in sorted(_BRANDS, key=len, reverse=True)
    ) + r")(?![a-z])",
    re.I,
)
_BRAND_KEYWORD_RE = re.compile(r"\bbrand\s+([a-z][a-z0-9&'\-]{1,20})\b", re.I)

_MONEY = r"\$?\s*(\d+(?:\.\d{1,2})?)"
_BUDGET_RANGE_RE = re.compile(rf"{_MONEY}\s*(?:-|to|through)\s*{_MONEY}", re.I)
_BUDGET_UNDER_RE = re.compile(
    rf"(?:under|below|less than|up to|at most|no more than|max(?:imum)?|within)\s*{_MONEY}",
    re.I,
)
_BUDGET_OVER_RE = re.compile(
    rf"(?:over|above|more than|at least|min(?:imum)?|starting at)\s*{_MONEY}", re.I
)
_BUDGET_AROUND_RE = re.compile(
    rf"(?:around|about|approx(?:imately)?|roughly|~)\s*{_MONEY}", re.I
)
_BUDGET_DOLLAR_RE = re.compile(r"\$\s*(\d+(?:\.\d{1,2})?)")
_BUDGET_DOLLARS_WORD_RE = re.compile(r"\b(\d+(?:\.\d{1,2})?)\s*(?:dollars|usd|bucks)\b", re.I)


def _content_tokens(message: str) -> list[tuple[str, int]]:
    """``(lowercased token, start offset)`` for content tokens (len > 1, not
    a stopword) -- same filter as ``text.terms`` but position-aware."""
    out: list[tuple[str, int]] = []
    for match in TOKEN_RE.finditer(message):
        token = match.group(0).lower()
        if len(token) > 1 and token not in STOPWORDS:
            out.append((token, match.start()))
    return out


def _extract_category(message: str) -> list[tuple[str, int]]:
    for token, offset in _content_tokens(message):
        canonical = _CATEGORY_KEYWORDS.get(token)
        if canonical:
            return [(canonical, offset)]
    return []


def _extract_multi(
    message: str, vocabulary: set[str], aliases: dict[str, str]
) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    seen: set[str] = set()
    for token, offset in _content_tokens(message):
        if token not in vocabulary:
            continue
        value = aliases.get(token, token)
        if value not in seen:
            seen.add(value)
            out.append((value, offset))
    return out


def _extract_size(message: str) -> list[tuple[str, int]]:
    match = _SIZE_PREFIXED_RE.search(message) or _SIZE_STANDALONE_RE.search(message)
    if not match:
        return []
    raw = match.group(1).lower().replace(" ", "")
    return [(_SIZE_NORMALIZE.get(raw, raw), match.start(1))]


def _extract_brand(message: str) -> list[tuple[str, int]]:
    match = _BRAND_PHRASE_RE.search(message) or _BRAND_KEYWORD_RE.search(message)
    if not match:
        return []
    value = match.group(1).lower()
    return [(_BRAND_NORMALIZE.get(value, value), match.start(1))]


def _extract_budget(message: str) -> dict[str, Any] | None:
    match = _BUDGET_RANGE_RE.search(message)
    if match:
        low, high = sorted((float(match.group(1)), float(match.group(2))))
        return {"raw": match.group(0).strip(), "offset": match.start(),
                "bounds": {"min": low, "max": high}}
    match = _BUDGET_UNDER_RE.search(message)
    if match:
        return {"raw": match.group(0).strip(), "offset": match.start(),
                "bounds": {"min": None, "max": float(match.group(1))}}
    match = _BUDGET_OVER_RE.search(message)
    if match:
        return {"raw": match.group(0).strip(), "offset": match.start(),
                "bounds": {"min": float(match.group(1)), "max": None}}
    match = _BUDGET_AROUND_RE.search(message)
    if match:
        return {"raw": match.group(0).strip(), "offset": match.start(),
                "bounds": {"min": None, "max": float(match.group(1))}}
    match = _BUDGET_DOLLARS_WORD_RE.search(message) or _BUDGET_DOLLAR_RE.search(message)
    if match:
        return {"raw": match.group(0).strip(), "offset": match.start(),
                "bounds": {"min": None, "max": float(match.group(1))}}
    return None


def _cue_spans(message: str) -> list[tuple[int, int, str]]:
    """Every override-cue match as ``(start, end, kind)``, kind in
    ``{"replace", "add"}``, sorted by position."""
    spans: list[tuple[int, int, str]] = []
    for match in _REPLACE_CUE_RE.finditer(message):
        spans.append((match.start(), match.end(), "replace"))
    for match in _ADD_CUE_RE.finditer(message):
        spans.append((match.start(), match.end(), "add"))
    return sorted(spans)


def _governing_cue(cues: list[tuple[int, int, str]], position: int) -> str | None:
    """The cue that governs a value at ``position``: the nearest cue,
    preferring one that ends before the value (a modifier precedes its
    target); a following cue is considered but penalised."""
    best_kind: str | None = None
    best_distance: float | None = None
    for start, end, kind in cues:
        distance = position - end if end <= position else (start - position) + 50
        if best_distance is None or distance < best_distance:
            best_distance, best_kind = distance, kind
    return best_kind


def _is_negated(message: str, value: str) -> bool:
    """True if ``value`` appears negated in ``message`` ("not leather",
    "without the leather", "don't want leather")."""
    tokens = re.findall(r"[a-z']+", message.lower())
    value_tokens = value.lower().split()
    span = len(value_tokens)
    for index in range(len(tokens) - span + 1):
        if tokens[index:index + span] != value_tokens:
            continue
        window = tokens[max(0, index - 3):index]
        if not window:
            continue
        if any(word in _STRONG_NEGATIONS for word in window):
            return True
        if window[-1] in _ADJACENT_NEGATIONS:
            return True
        if {"don't", "dont"} & set(window) and {"want", "need"} & set(window):
            return True
    return False


# --------------------------------------------------------------------------
# Evidence Confidence (Phase 11, CP 11.1) -- how much the shopper meant it.
#
# Phase 4 already built the RECEIVING half of this: validate_delta cleans and
# clamps a `confidence` on a delta entry, and stores it when below 1.0. What
# was missing is a producer and a persister. Extraction now assigns one and
# apply_delta carries it onto the slot entry, where the contract has always
# said it would live ("Evidence Confidence ... live inside slot entries from
# Phase 11", contracts.py).
#
# Absent means 1.0 throughout, which is Phase 4's convention and is what keeps
# every pre-existing caller and stored state unchanged.
#
# Confidence is per MESSAGE, not per value: a turn is one speech act, and the
# phrasing that marks it as a requirement or as thinking-out-loud governs the
# whole turn. The known cost is that "I'm looking for jackets. A key
# requirement is: leather" gives the CATEGORY the same 1.0 as the material.
# Per-value attribution by cue proximity is possible -- `_governing_cue`
# already does exactly that for operations -- and is the obvious refinement if
# a measurement ever shows it matters.
# --------------------------------------------------------------------------

EC_REQUIREMENT = 1.0   # "a key requirement is", "what matters is", "must"
EC_CORRECTION = 0.9    # an override: the shopper stopped to correct us
EC_STATED = 0.7        # a plain declarative mention
EC_HEDGED = 0.4        # "maybe", "something", "I guess", "still exploring"

# Thinking-out-loud markers. Overlaps ``strategy.BROWSING_CUES`` and
# ``strategy.FILLER_CUES`` in spirit and partly in content, but is NOT shared
# with them: that vocabulary answers "is this shopper browsing?" about a whole
# session, this one answers "did they commit to this constraint?" about one
# turn, and the two have already been observed to want different words.
# Importing across would also invert the dependency -- strategy imports state.
_HEDGE_CUES = frozenset({
    "maybe", "perhaps", "possibly", "probably", "might", "guess",
    "something", "anything", "some", "kind", "sort", "ish", "like",
    "prefer", "preferably", "ideally", "leaning", "considering", "thinking",
    "exploring", "browsing", "unsure", "open", "whatever", "either",
})

# Requirement phrasing. Kept here rather than imported from ``strategy`` for
# the dependency reason above; the two lists are expected to drift apart, and
# ``tests/test_confidence.py`` pins the overlap that matters.
_REQUIREMENT_TOKENS = frozenset({
    "requirement", "requirements", "required", "require", "must", "need",
    "needs", "needed", "specifically", "exactly", "matter", "matters",
    "important", "essential", "critical", "necessary",
})

# Negations that can turn a requirement word into its opposite. Contraction
# STEMS ("isn", "doesn") because TOKEN_RE splits on the apostrophe.
_NEGATION_TOKENS = frozenset({
    "not", "no", "never", "cannot", "cant", "nope", "isn", "aren", "wasn",
    "weren", "doesn", "don", "didn", "won", "wouldn", "shouldn", "couldn",
})

# How far back a negation can reach. "leather is not required" puts one token
# between them once stopwords are dropped; "not a hard requirement" puts two.
_NEGATION_WINDOW = 3


def _has_unnegated(tokens: list[str], vocabulary: frozenset) -> bool:
    """True if ``vocabulary`` appears in ``tokens`` without a negation before it.

    "A key requirement is: leather" is a requirement. "Leather is not
    required" is the shopper WITHDRAWING one, and reading it as a requirement
    scored the constraint at maximum confidence precisely when they had just
    relaxed it (D Phase 12 review, P2).
    """
    for index, token in enumerate(tokens):
        if token not in vocabulary:
            continue
        window = tokens[max(0, index - _NEGATION_WINDOW):index]
        if not any(word in _NEGATION_TOKENS for word in window):
            return True
    return False


def evidence_confidence(message: object) -> float:
    """CP 11.1 -- how firmly this turn asserts whatever it mentions.

    Deterministic and offline, read from the shopper's phrasing, strongest
    signal first:

        1.0  requirement language -- "a key requirement is", "what matters is"
        0.4  hedged -- "maybe", "something", "still exploring"
        0.9  a correction -- they stopped to replace something we had
        0.7  otherwise: a plain declarative mention

    Requirement language outranks a hedge: "I need something black" is a
    requirement that happens to contain "something", not a hesitation.

    A hedge outranks a CORRECTION, which is the one ordering that is not
    obvious. Requirement words and hedges both describe how committed the
    shopper is to the VALUE; a correction cue ("actually", "instead")
    describes the EDIT, and is only a proxy for commitment. So "actually,
    maybe denim" is a hedged 0.4 -- the shopper is replacing one thing with
    something they are unsure of -- while "actually, make it denim" is a
    confident 0.9. Ordering correction above hedge scored that first message
    0.9, which the D Phase 11 interleaving test caught.
    """
    if not isinstance(message, str) or not message.strip():
        return EC_STATED
    ordered = terms(message)
    tokens = set(ordered)
    if _has_unnegated(ordered, _REQUIREMENT_TOKENS):
        return EC_REQUIREMENT
    # A NEGATED requirement is the shopper relaxing something, which is the
    # opposite of insisting on it -- so it reads as a hedge, not as a plain
    # statement and certainly not as a requirement.
    #
    # It does NOT drop the slot. "Leather is not required" leaves leather in
    # play at low confidence rather than deleting it: a false REMOVE destroys
    # a constraint the shopper still wants, and this codebase already chose
    # the conservative side of that trade once (see _STRONG_NEGATIONS, where
    # "drop" is excluded for the same reason). Whether relaxation should
    # eventually retire the value is an extraction question for the override
    # layer, not a confidence question, and is deliberately left alone here.
    if tokens & _REQUIREMENT_TOKENS:
        return EC_HEDGED
    if tokens & _HEDGE_CUES:
        return EC_HEDGED
    if _REPLACE_CUE_RE.search(message):
        return EC_CORRECTION
    return EC_STATED


def extract_delta(message: str) -> dict[str, dict[str, Any]]:
    """Deterministic slot + operation extraction for one message.

    Returns ``{slot: entry}`` where ``entry`` has ``values`` (positive,
    non-negated), ``cardinality``, an optional ``cue`` (``"replace"`` /
    ``"add"``), an optional ``remove`` list (negated values), and ``bounds``
    for budget.

    Operation cues are attributed **per value**, by proximity to the
    governing cue -- so "Actually denim, but also navy" gives material a
    ``replace`` cue and color an ``add`` cue in the same turn. A slot with
    any ``replace`` value takes ``replace``; else any ``add`` -> ``add``.
    No state is touched here.
    """
    cues = _cue_spans(message)
    delta: dict[str, dict[str, Any]] = {}

    positioned: list[tuple[str, list[tuple[str, int]]]] = [
        ("category", _extract_category(message)),
        ("color", _extract_multi(message, _COLORS, _COLOR_ALIASES)),
        ("material", _extract_multi(message, _MATERIALS, {})),
        ("size", _extract_size(message)),
        ("brand", _extract_brand(message)),
    ]

    for slot, pairs in positioned:
        if not pairs:
            continue
        removed = [value for value, _ in pairs if _is_negated(message, value)]
        kept = [(value, offset) for value, offset in pairs if value not in removed]
        entry: dict[str, Any] = {
            "values": [value for value, _ in kept],
            "cardinality": SLOT_CARDINALITY[slot],
        }
        if removed:
            entry["remove"] = removed
        kinds = {_governing_cue(cues, offset) for _, offset in kept}
        if "replace" in kinds:
            entry["cue"] = "replace"
        elif "add" in kinds:
            entry["cue"] = "add"
        delta[slot] = entry

    budget = _extract_budget(message)
    if budget:
        entry = {
            "values": [budget["raw"]],
            "cardinality": "single",
            "bounds": budget["bounds"],
        }
        cue = _governing_cue(cues, budget["offset"])
        if cue == "replace":
            entry["cue"] = "replace"
        delta["budget"] = entry

    # CP 11.1 -- one confidence for the turn, on every entry it produced.
    # validate_delta clamps it and drops it again when it is 1.0, so a fully
    # confident turn produces exactly the delta it produced before Phase 11.
    confidence = evidence_confidence(message)
    for entry in delta.values():
        entry["confidence"] = confidence

    return delta


# --------------------------------------------------------------------------
# Delta validation (Phase 4) -- the guard between any delta source and the
# state manager.
# --------------------------------------------------------------------------


def _clean_str_list(value: object) -> list[str]:
    """Non-empty, whitespace-trimmed strings from ``value`` if it is a list,
    else ``[]`` (D Finding 2: an empty string must not reach ``state.slots``)."""
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _clean_confidence(value: object) -> float | None:
    """Confidence in ``(0, 1]``, or ``None`` meaning "drop this entry".

    ``None`` -> ``1.0`` (an absent confidence is trusted). ``NaN`` or
    non-numeric -> drop. Out of range -> clamped to ``[0, 1]``. Zero -> drop
    (CP 4.3).
    """
    if value is None:
        return 1.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    number = min(1.0, max(0.0, number))
    return number if number > 0.0 else None


def validate_delta(delta: object) -> dict[str, dict[str, Any]]:
    """Sanitize a delta from any source before ``apply_delta`` sees it.

    * non-dict input, or an entry that is not a dict -> dropped;
    * unknown slot name -> dropped (CP 4.1);
    * explicit ``op`` outside ``{SET, REPLACE, ADD, REMOVE}`` -> entry
      dropped (CP 4.2);
    * ``confidence`` coerced per ``_clean_confidence``; drop-worthy -> entry
      dropped (CP 4.3);
    * ``values`` / ``remove`` kept only if a list of ``str``; entry with
      neither -> dropped;
    * ``cardinality`` forced to the known value; unknown ``cue`` -> dropped;
    * ``bounds`` kept only if a dict of numeric-or-None min/max.

    A no-op on ``extract_delta``'s deterministic output.
    """
    if not isinstance(delta, dict):
        return {}
    clean: dict[str, dict[str, Any]] = {}
    for slot, entry in delta.items():
        if slot not in SLOT_CARDINALITY or not isinstance(entry, dict):
            continue
        if "op" in entry and entry["op"] not in _VALID_OPS:
            continue
        confidence = _clean_confidence(entry.get("confidence"))
        if confidence is None:
            continue
        cardinality = SLOT_CARDINALITY[slot]
        values = _clean_str_list(entry.get("values"))
        remove = _clean_str_list(entry.get("remove"))
        if cardinality == "single":
            # A single-valued slot holds at most one value (D Finding 1).
            values = values[:1]
        if not values and not remove:
            continue
        cleaned: dict[str, Any] = {"values": values, "cardinality": cardinality}
        if remove:
            cleaned["remove"] = remove
        if entry.get("op") in _VALID_OPS:
            cleaned["op"] = entry["op"]
        if entry.get("cue") in _VALID_CUES:
            cleaned["cue"] = entry["cue"]
        if confidence < 1.0:
            cleaned["confidence"] = confidence
        bounds = entry.get("bounds")
        if isinstance(bounds, dict):
            low, high = bounds.get("min"), bounds.get("max")
            if all(x is None or (isinstance(x, (int, float)) and x == x) for x in (low, high)):
                cleaned["bounds"] = {"min": low, "max": high}
        clean[slot] = cleaned
    return clean


def safe_extract_delta(message: str) -> dict[str, dict[str, Any]]:
    """``extract_delta`` guarded to never raise -- any failure yields an
    empty delta (CP 4.4).

    A standalone helper for callers that want a non-throwing extract.
    ``update_state`` does not use it: it calls ``extract_delta`` inside its
    own try/except so a failure in validation or application rolls state
    back too, not just an extractor failure.
    """
    try:
        return extract_delta(message)
    except Exception:
        return {}


# --------------------------------------------------------------------------
# State manager: the single authoritative writer.
# --------------------------------------------------------------------------


def _record(
    state: SessionState,
    turn: int,
    slot: str,
    op: str,
    value: str,
    superseded: list[str] | None = None,
) -> None:
    entry: dict[str, Any] = {
        "turn": turn, "slot": slot, "op": op, "value": value, "source": "extractor",
    }
    if superseded:
        entry["superseded"] = list(superseded)
    state.provenance.append(entry)


def _supersede_evidence(state: SessionState, dead_values: list[str]) -> None:
    """Invalidate the stale assertion of a now-dead constraint in free-text
    evidence, WITHOUT touching unrelated intent in the same entry
    (CP 3.6 / D Finding 1).

    Evidence is distilled to residual tokens at creation, so a dead slot
    value normally never reaches ``normalized``. This is the safety net for
    the case where extraction missed it: strip the dead token(s); keep the
    entry active if meaningful residual survives, else mark the whole entry
    superseded.
    """
    dead_tokens: set[str] = set()
    for value in dead_values:
        dead_tokens.update(terms(str(value)))
    if not dead_tokens:
        return
    for entry in state.evidence:
        if not isinstance(entry, dict) or entry.get("status") != "active":
            continue
        entry_tokens = terms(entry.get("normalized", ""))
        overlap = set(entry_tokens) & dead_tokens
        if not overlap:
            continue
        live = [token for token in entry_tokens if token not in dead_tokens]
        entry["superseded_parts"] = sorted(
            set(entry.get("superseded_parts", [])) | overlap
        )
        if live:
            entry["normalized"] = " ".join(live)
        else:
            entry["status"] = "superseded"


def superseded_values(state: SessionState, slot: str) -> set[str]:
    """Every value ever displaced from ``slot`` (from provenance)."""
    out: set[str] = set()
    for record in state.provenance:
        if record.get("slot") == slot:
            out.update(record.get("superseded", []))
            if record.get("op") == "REMOVE":
                out.add(record.get("value"))
    return out


def _confidence_of(entry: object) -> float:
    """A slot or delta entry's Evidence Confidence; absent means fully meant.

    The absent-is-1.0 convention is Phase 4's (``_clean_confidence``), kept so
    that state written before Phase 11 -- and any entry from a turn the
    shopper stated plainly -- reads back identically.
    """
    if not isinstance(entry, dict):
        return 1.0
    value = entry.get("confidence")
    if not isinstance(value, (int, float)) or value != value:
        return 1.0
    return max(0.0, min(1.0, float(value)))


def _carry_confidence(entry: dict, incoming: object, existing: object,
                      combine) -> None:
    """Put the surviving Evidence Confidence on a rebuilt slot entry.

    ``combine=max`` for a union (ADD), which keeps the firmest thing the
    shopper has said about the slot; ``combine=None`` for SET / REPLACE, where
    the new statement stands alone and the old confidence dies with the old
    value.

    Stored only when below 1.0, mirroring ``validate_delta``: a fully
    confident slot entry is byte-identical to its pre-Phase-11 self, which is
    what makes the ranking flag's OFF position exact.
    """
    confidence = _confidence_of(incoming)
    # Combine only against a slot that actually existed. "Absent means 1.0" is
    # right for reading a stored entry but wrong as a starting value for a
    # max: a brand-new hedged slot would combine against the 1.0 of the
    # nothing that preceded it and come out fully confident.
    if combine is not None and isinstance(existing, dict):
        confidence = combine(confidence, _confidence_of(existing))
    if confidence < 1.0:
        entry["confidence"] = confidence


def apply_delta(state: SessionState, delta: dict[str, dict[str, Any]], turn: int) -> None:
    """Apply a delta to ``state.slots`` -- the single authoritative write.

    Operation, highest priority first:

    * an explicit validated ``op`` (SET / REPLACE / ADD / REMOVE) is
      authoritative -- a future parser drives operations directly (CP 4.2);
    * else the Phase 3 cue / ``remove`` semantics ("actually" -> REPLACE,
      "also" -> ADD, negation -> REMOVE);
    * else the Phase 2 default: ADD for multi-valued, SET for single-valued.

    Semantics:
    SET / REPLACE -> slot value set becomes exactly ``values`` (multi-valued
    slots are NOT unioned); old active values are superseded. ADD -> union,
    deduplicated; a single-valued slot never accumulates -> ADD collapses to
    REPLACE (D Finding 1). REMOVE -> strip ``values`` (explicit op) or the
    negated ``remove`` values (cue path) from the slot; drop the slot if
    nothing survives; superseded evidence is invalidated (CP 3.6).

    A slot the delta does not mention is never touched (CP 2.2). Superseded
    values do not spontaneously return (CP 3.6). Defensive on a raw delta:
    unknown slot / invalid explicit ``op`` -> skipped (CP 4.1 / 4.2).
    """
    for slot, incoming in delta.items():
        if not isinstance(incoming, dict) or slot not in SLOT_CARDINALITY:
            continue
        explicit_op = incoming.get("op")
        if explicit_op is not None and explicit_op not in _VALID_OPS:
            continue
        cardinality = incoming.get("cardinality") or SLOT_CARDINALITY[slot]
        cue = incoming.get("cue")
        new_bounds = incoming.get("bounds")
        existing = state.slots.get(slot)
        existing_values = list(existing["values"]) if existing else []

        # An explicit REMOVE carries its targets in `values`; the cue path
        # carries negated values in `remove`.
        if explicit_op == "REMOVE":
            remove = list(incoming.get("values") or [])
            positive: list[str] = []
        else:
            remove = list(incoming.get("remove") or [])
            positive = list(incoming.get("values") or [])
        if cardinality == "single":
            positive = positive[:1]

        # 1. REMOVE phase.
        if remove:
            for value in remove:
                _record(state, turn, slot, "REMOVE", value)
            _supersede_evidence(state, remove)
            existing_values = [v for v in existing_values if v not in remove]
            if not positive:
                if existing_values:
                    # Rebuilt, so anything not copied across is LOST. Removing
                    # one value of a multi-valued slot says nothing about the
                    # shopper's commitment to the ones that remain: "maybe
                    # leather and denim" then "no leather" must leave denim
                    # hedged, not promote it to maximum insistence by dropping
                    # its confidence (D Phase 12 review, Q1). Bounds are
                    # carried for the same reason -- defensively, since a
                    # single-valued budget cannot reach here with survivors.
                    survivor: dict[str, Any] = {
                        "values": existing_values, "cardinality": cardinality,
                    }
                    if isinstance(existing, dict):
                        if existing.get("bounds") is not None:
                            survivor["bounds"] = existing["bounds"]
                        _carry_confidence(survivor, existing, None, combine=None)
                    state.slots[slot] = survivor
                else:
                    state.slots.pop(slot, None)
                continue

        if not positive:
            continue

        # 2. Decide the operation for the positive values.
        if explicit_op in ("SET", "REPLACE", "ADD"):
            operation = explicit_op
        elif cue == "add":
            operation = "ADD"
        elif remove:
            operation = "REPLACE"
        elif cue == "replace" and existing_values:
            operation = "REPLACE"
        elif cardinality == "single":
            operation = "REPLACE"
        else:
            operation = "ADD"
        if operation == "ADD" and cardinality == "single":
            operation = "REPLACE"

        if operation == "ADD":
            added = [v for v in positive if v not in existing_values]
            if not added:
                # Nothing new to union -- but the shopper may have restated an
                # existing value MORE firmly ("maybe leather" -> "leather is a
                # hard requirement"). That is not a no-op: the escalation is
                # the whole content of the turn (CP 11.1). Confidence only
                # ever rises here; a hedged restatement of something already
                # insisted on does not soften it.
                if existing is not None:
                    firmer = max(_confidence_of(incoming), _confidence_of(existing))
                    if firmer > _confidence_of(existing):
                        existing.pop("confidence", None)
                        if firmer < 1.0:
                            existing["confidence"] = firmer
                        _record(state, turn, slot, "ADD", positive[0])
                continue
            entry: dict[str, Any] = {
                "values": existing_values + added, "cardinality": cardinality,
            }
            if new_bounds is not None:
                entry["bounds"] = new_bounds
            # CP 11.1 -- a union keeps the STRONGEST statement made about the
            # slot. Adding a hedged value to a slot the shopper already
            # insisted on does not soften the insistence.
            _carry_confidence(entry, incoming, existing, combine=max)
            state.slots[slot] = entry
            for value in added:
                _record(state, turn, slot, "ADD", value)
            continue

        # SET / REPLACE: the slot's value set becomes exactly `positive`.
        superseded = [v for v in existing_values if v not in positive]
        unchanged = (
            existing is not None
            and existing["values"] == positive
            and existing.get("bounds") == new_bounds
            and not remove
            # Restating the same constraint MORE firmly is not a no-op: the
            # shopper escalated from "maybe leather" to "it must be leather"
            # and the slot has to record that (CP 11.1).
            and _confidence_of(incoming) <= _confidence_of(existing)
        )
        if unchanged:
            continue
        entry = {"values": list(positive), "cardinality": cardinality}
        if new_bounds is not None:
            entry["bounds"] = new_bounds
        # CP 11.1 -- a replacement replaces the confidence too: the old
        # statement is gone, so its confidence must not outlive it.
        _carry_confidence(entry, incoming, existing, combine=None)
        state.slots[slot] = entry
        if operation == "SET":
            op_name = "SET"
        else:
            op_name = "REPLACE" if (existing_values or remove) else "SET"
        for value in positive:
            _record(state, turn, slot, op_name, value, superseded or None)
        if superseded:
            _supersede_evidence(state, superseded)


def update_evidence(
    state: SessionState, message: str, delta: dict[str, dict[str, Any]], turn: int
) -> None:
    """CP 2.8 / CP 3.6 - keep residual free-text as distilled evidence.

    ``normalized`` is the DISTILLED content: message tokens minus override
    cues, override-plumbing words, slot-marker words, and every extracted
    slot value. So an entry never contains a structured constraint token --
    superseding ``leather`` cannot touch an entry that only says
    ``winter hiking`` (B Phase 3 blocker fix). An override message is not
    skipped wholesale: its genuine residual ("actually denim for winter
    hiking" -> ``winter hiking``) is retained.

    Not stored when nothing meaningful remains, or on a conversational
    non-answer. Each entry is
    ``{"turn", "text", "normalized", "status": "active"}``; ``text`` keeps
    the raw message for audit, ``normalized`` shrinks as overrides invalidate
    parts of it, ``status`` flips only when nothing live is left.
    """
    if not message or not message.strip():
        return
    if is_non_answer(message):
        return
    # Negation words are grammatical plumbing too: after "not leather" has
    # been applied as a REMOVE, the bare "not" left behind carries no product
    # intent and must not become standalone evidence.
    consumed: set[str] = (
        set(_EVIDENCE_MARKER_TOKENS)
        | _OVERRIDE_PLUMBING
        | _STRONG_NEGATIONS
        | _ADJACENT_NEGATIONS
    )
    for incoming in delta.values():
        for value in incoming.get("values", ()):
            consumed.update(terms(value))
        for value in incoming.get("remove", ()):
            consumed.update(terms(value))
    # Drop multi-word override cue phrases ("no wait", "scratch that", ...).
    stripped = _ADD_CUE_RE.sub(" ", _REPLACE_CUE_RE.sub(" ", message))
    residual = [token for token in terms(stripped) if token not in consumed]
    if not residual:
        return
    normalized = " ".join(residual)
    for entry in state.evidence:
        if (
            isinstance(entry, dict)
            and entry.get("status") == "active"
            and entry.get("normalized") == normalized
        ):
            return
    state.evidence.append(
        {"turn": turn, "text": message, "normalized": normalized, "status": "active"}
    )


def update_state(state: SessionState, message: object, turn: object) -> SessionState:
    """Single per-turn entry point the Agent calls before retrieval.

    Deterministic and offline. Mutates ``state`` in place and returns it.

    Phase 4 robustness: the delta is validated before it can touch state,
    and the whole update is wrapped -- if extraction, validation, or
    application fails for any reason, ``state`` is rolled back to exactly
    what it was before this turn (CP 4.4). A no-op turn (nothing extracted,
    everything rejected) is not a failure; it simply leaves state unchanged.
    """
    text = message if isinstance(message, str) else ""
    turn_value = turn if isinstance(turn, int) else state.turn
    snapshot = (
        copy.deepcopy(state.slots),
        copy.deepcopy(state.evidence),
        copy.deepcopy(state.provenance),
        state.turn,
    )
    try:
        delta = validate_delta(extract_delta(text))
        apply_delta(state, delta, turn_value)
        update_evidence(state, text, delta, turn_value)
    except Exception:
        state.slots, state.evidence, state.provenance, state.turn = snapshot
        return state
    if isinstance(turn, int) and turn > state.turn:
        state.turn = turn
    return state
