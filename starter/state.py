"""Deterministic session-state manager (Phase 2).

Single authoritative writer of ``SessionState``. Given a raw user message it:

* extracts structured slot values with keyword / regex rules only -- no LLM,
  no network (principle J: the core must work offline);
* accumulates them into ``state.slots`` (Phase 2 semantics: SET single-valued,
  ADD multi-valued -- no override / supersession yet, that is Phase 3);
* records every change in ``state.provenance``;
* keeps residual free-text as ``state.evidence`` (CP 2.8).

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

import re
from typing import Any

from starter.contracts import SessionState
from starter.text import terms

# Slot-marker words that carry no standalone free-text signal, so they never
# make a message "residual" for evidence purposes.
_EVIDENCE_MARKER_TOKENS = {
    "size", "budget", "brand", "color", "colour", "material", "priced",
    "price", "cost", "costs", "dollars", "usd", "bucks", "under", "around",
    "about", "approximately", "roughly", "over", "above", "below",
}

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
_BRANDS = {
    "nike", "adidas", "puma", "reebok", "new balance", "asics", "brooks",
    "saucony", "converse", "vans", "skechers", "crocs", "timberland", "ugg",
    "birkenstock", "clarks", "dr martens", "under armour", "the north face",
    "patagonia", "columbia", "carhartt", "levi", "levis", "wrangler", "lee",
    "gap", "old navy", "uniqlo", "zara", "h&m", "hanes", "champion", "fila",
    "calvin klein", "tommy hilfiger", "ralph lauren", "polo", "lacoste",
    "gucci", "prada", "coach", "michael kors", "kate spade", "fossil",
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


def _extract_category(message: str) -> str | None:
    for token in terms(message):
        canonical = _CATEGORY_KEYWORDS.get(token)
        if canonical:
            return canonical
    return None


def _extract_multi(message: str, vocabulary: set[str], aliases: dict[str, str]) -> list[str]:
    seen: list[str] = []
    for token in terms(message):
        value = aliases.get(token, token)
        if token in vocabulary and value not in seen:
            seen.append(value)
    return seen


def _extract_size(message: str) -> str | None:
    match = _SIZE_PREFIXED_RE.search(message) or _SIZE_STANDALONE_RE.search(message)
    if not match:
        return None
    raw = match.group(1).lower().replace(" ", "")
    return _SIZE_NORMALIZE.get(raw, raw)


def _extract_brand(message: str) -> str | None:
    match = _BRAND_PHRASE_RE.search(message)
    if match:
        value = match.group(1).lower()
        return _BRAND_NORMALIZE.get(value, value)
    match = _BRAND_KEYWORD_RE.search(message)
    if match:
        value = match.group(1).lower()
        return _BRAND_NORMALIZE.get(value, value)
    return None


def _extract_budget(message: str) -> dict[str, Any] | None:
    match = _BUDGET_RANGE_RE.search(message)
    if match:
        low, high = sorted((float(match.group(1)), float(match.group(2))))
        return {"raw": match.group(0).strip(), "bounds": {"min": low, "max": high}}
    match = _BUDGET_UNDER_RE.search(message)
    if match:
        return {"raw": match.group(0).strip(), "bounds": {"min": None, "max": float(match.group(1))}}
    match = _BUDGET_OVER_RE.search(message)
    if match:
        return {"raw": match.group(0).strip(), "bounds": {"min": float(match.group(1)), "max": None}}
    match = _BUDGET_AROUND_RE.search(message)
    if match:
        return {"raw": match.group(0).strip(), "bounds": {"min": None, "max": float(match.group(1))}}
    match = _BUDGET_DOLLARS_WORD_RE.search(message) or _BUDGET_DOLLAR_RE.search(message)
    if match:
        return {"raw": match.group(0).strip(), "bounds": {"min": None, "max": float(match.group(1))}}
    return None


def extract_delta(message: str) -> dict[str, dict[str, Any]]:
    """Deterministic slot extraction for one message.

    Returns ``{slot: {"values": [...], "cardinality": ...}}`` for slots that
    matched. ``budget`` also carries ``"bounds"``. No state is touched here.
    """
    delta: dict[str, dict[str, Any]] = {}

    category = _extract_category(message)
    if category:
        delta["category"] = {"values": [category], "cardinality": "single"}

    colors = _extract_multi(message, _COLORS, _COLOR_ALIASES)
    if colors:
        delta["color"] = {"values": colors, "cardinality": "multi"}

    materials = _extract_multi(message, _MATERIALS, {})
    if materials:
        delta["material"] = {"values": materials, "cardinality": "multi"}

    size = _extract_size(message)
    if size:
        delta["size"] = {"values": [size], "cardinality": "single"}

    brand = _extract_brand(message)
    if brand:
        delta["brand"] = {"values": [brand], "cardinality": "single"}

    budget = _extract_budget(message)
    if budget:
        delta["budget"] = {
            "values": [budget["raw"]],
            "cardinality": "single",
            "bounds": budget["bounds"],
        }

    return delta


# --------------------------------------------------------------------------
# State manager: the single authoritative writer.
# --------------------------------------------------------------------------


def _record(state: SessionState, turn: int, slot: str, op: str, value: str) -> None:
    state.provenance.append(
        {"turn": turn, "slot": slot, "op": op, "value": value, "source": "extractor"}
    )


def apply_delta(state: SessionState, delta: dict[str, dict[str, Any]], turn: int) -> None:
    """Accumulate a delta into ``state.slots`` (Phase 2 semantics).

    * multi-valued slot: ADD new values (union, dedup, retrieval order kept);
    * single-valued slot: SET (last write wins) unless the value and bounds
      are already what is stored.

    No slot the delta does not mention is touched -- that is what makes a
    stored constraint persist across later turns (CP 2.2).
    """
    for slot, incoming in delta.items():
        cardinality = incoming["cardinality"]
        existing = state.slots.get(slot)

        if cardinality == "multi":
            current: list[str] = list(existing["values"]) if existing else []
            added = [v for v in incoming["values"] if v not in current]
            if not added:
                continue
            state.slots[slot] = {
                "values": current + added,
                "cardinality": "multi",
            }
            for value in added:
                _record(state, turn, slot, "ADD", value)
            continue

        new_value = incoming["values"][0]
        new_bounds = incoming.get("bounds")
        if (
            existing
            and existing["values"] == [new_value]
            and existing.get("bounds") == new_bounds
        ):
            continue
        entry: dict[str, Any] = {"values": [new_value], "cardinality": "single"}
        if new_bounds is not None:
            entry["bounds"] = new_bounds
        state.slots[slot] = entry
        _record(state, turn, slot, "SET", new_value)


def update_evidence(
    state: SessionState, message: str, delta: dict[str, dict[str, Any]], turn: int
) -> None:
    """CP 2.8 - keep residual free-text as evidence.

    Residual = message content tokens minus tokens already consumed by an
    extracted slot value. If anything is left (``"gift for my dad"`` ->
    ``{gift, dad}``), store the raw message once, deduped by normalized form.
    A message that is purely a slot mention (``"jacket"``) leaves no residual
    and is not stored.
    """
    if not message or not message.strip():
        return
    consumed: set[str] = set(_EVIDENCE_MARKER_TOKENS)
    for incoming in delta.values():
        for value in incoming["values"]:
            consumed.update(terms(value))
    residual = [token for token in terms(message) if token not in consumed]
    if not residual:
        return
    normalized = " ".join(terms(message))
    for entry in state.evidence:
        if isinstance(entry, dict) and entry.get("normalized") == normalized:
            return
    state.evidence.append({"turn": turn, "text": message, "normalized": normalized})


def update_state(state: SessionState, message: object, turn: object) -> SessionState:
    """Single per-turn entry point the Agent calls before retrieval.

    Deterministic and offline. Mutates ``state`` in place and returns it.
    """
    text = message if isinstance(message, str) else ""
    delta = extract_delta(text)
    apply_delta(state, delta, turn if isinstance(turn, int) else state.turn)
    update_evidence(state, text, delta, turn if isinstance(turn, int) else state.turn)
    if isinstance(turn, int) and turn > state.turn:
        state.turn = turn
    return state
