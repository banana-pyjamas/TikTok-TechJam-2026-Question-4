"""Deterministic session-state manager (Phase 2 + Phase 3).

Single authoritative writer of ``SessionState``. Given a raw user message it:

* extracts structured slot values with keyword / regex rules only -- no LLM,
  no network (principle J: the core must work offline);
* detects intent-override cues deterministically ("actually" -> REPLACE,
  "also" -> ADD, "not leather" -> REMOVE) and applies the operation
  slot-specifically -- an override to one slot never disturbs the others
  (CP 3.2 golden);
* accumulates the result into ``state.slots``; superseded values are marked
  in ``state.provenance`` and any evidence entry asserting them flips to
  ``status: "superseded"`` so it cannot resurrect (CP 3.6 / D Finding 1);
* records every change in ``state.provenance``;
* keeps distilled residual free-text as ``state.evidence`` (CP 2.8). Entries
  are ``{"turn", "text", "normalized", "status": "active"}``; ``status`` is
  the hook Phase 3 uses to supersede free-text with the same mechanism it
  supersedes slots, so a superseded free-text constraint cannot resurrect if
  Phase 5 later feeds evidence into the query (D Finding 1).

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
_STRONG_NEGATIONS = {
    "without", "drop", "remove", "lose", "skip", "minus", "nix", "exclude", "ditch",
}
_ADJACENT_NEGATIONS = {"not", "no"}

# Override-plumbing words: they appear in "actually, ignore my earlier
# preference, ..." style messages and are not free-text evidence.
_OVERRIDE_PLUMBING = {
    "actually", "instead", "rather", "ignore", "forget", "earlier", "previous",
    "prior", "preference", "preferences", "mind", "wait", "scratch",
    "nevermind", "disregard", "what", "need", "needed",
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


def extract_delta(message: str) -> dict[str, dict[str, Any]]:
    """Deterministic slot + operation extraction for one message.

    Returns ``{slot: entry}`` where ``entry`` has ``values`` (positive,
    non-negated), ``cardinality``, an optional ``cue`` (``"replace"`` /
    ``"add"``), an optional ``remove`` list (negated values), and ``bounds``
    for budget. No state is touched here -- the final REPLACE/ADD/REMOVE
    decision is made by ``apply_delta`` against the current state.
    """
    replace_cue = bool(_REPLACE_CUE_RE.search(message))
    add_cue = bool(_ADD_CUE_RE.search(message))
    delta: dict[str, dict[str, Any]] = {}

    raw: list[tuple[str, list[str]]] = []
    category = _extract_category(message)
    if category:
        raw.append(("category", [category]))
    colors = _extract_multi(message, _COLORS, _COLOR_ALIASES)
    if colors:
        raw.append(("color", colors))
    materials = _extract_multi(message, _MATERIALS, {})
    if materials:
        raw.append(("material", materials))
    size = _extract_size(message)
    if size:
        raw.append(("size", [size]))
    brand = _extract_brand(message)
    if brand:
        raw.append(("brand", [brand]))

    for slot, values in raw:
        removed = [value for value in values if _is_negated(message, value)]
        kept = [value for value in values if value not in removed]
        entry: dict[str, Any] = {
            "values": kept,
            "cardinality": SLOT_CARDINALITY[slot],
        }
        if removed:
            entry["remove"] = removed
        if add_cue:
            entry["cue"] = "add"
        elif replace_cue:
            entry["cue"] = "replace"
        delta[slot] = entry

    budget = _extract_budget(message)
    if budget:
        entry = {
            "values": [budget["raw"]],
            "cardinality": "single",
            "bounds": budget["bounds"],
        }
        if replace_cue and not add_cue:
            entry["cue"] = "replace"
        delta["budget"] = entry

    return delta


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
    """Mark active evidence entries that assert a now-dead constraint as
    superseded, so they cannot resurrect it if a later phase reads evidence
    into the query (CP 3.6 / D Finding 1)."""
    dead_tokens: set[str] = set()
    for value in dead_values:
        dead_tokens.update(terms(str(value)))
    if not dead_tokens:
        return
    for entry in state.evidence:
        if not isinstance(entry, dict) or entry.get("status") != "active":
            continue
        if set(terms(entry.get("normalized", ""))) & dead_tokens:
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


def apply_delta(state: SessionState, delta: dict[str, dict[str, Any]], turn: int) -> None:
    """Apply a delta to ``state.slots`` -- the single authoritative write.

    Phase 2 accumulation plus Phase 3 operations:

    * ``remove`` values -> REMOVE (strip from the slot; supersede evidence);
    * positive values with an ``add`` cue -> ADD (union);
    * positive values with a ``replace`` cue over an existing value, or any
      new value for a single-valued slot -> REPLACE (old values superseded);
    * otherwise the Phase 2 default: ADD for multi-valued, first-time SET for
      single-valued.

    A slot the delta does not mention is never touched (CP 2.2). Superseded
    values do not spontaneously return (CP 3.6): later turns only re-add a
    value if the message states it again.
    """
    for slot, incoming in delta.items():
        cardinality = incoming["cardinality"]
        remove = list(incoming.get("remove") or [])
        values = list(incoming.get("values") or [])
        cue = incoming.get("cue")
        new_bounds = incoming.get("bounds")
        existing = state.slots.get(slot)
        existing_values = list(existing["values"]) if existing else []

        # 1. REMOVE negated values.
        if remove:
            survivors = [v for v in existing_values if v not in remove]
            for value in remove:
                _record(state, turn, slot, "REMOVE", value)
            _supersede_evidence(state, remove)
            existing_values = survivors
            if not values:
                if survivors:
                    state.slots[slot] = {"values": survivors, "cardinality": cardinality}
                else:
                    state.slots.pop(slot, None)
                continue

        if not values:
            continue

        # 2. Decide the operation for the positive values.
        if cue == "add":
            operation = "ADD"
        elif remove:
            operation = "REPLACE"
        elif cue == "replace" and existing_values:
            operation = "REPLACE"
        elif cardinality == "single":
            operation = "REPLACE"
        else:
            operation = "ADD"

        if operation == "ADD":
            added = [v for v in values if v not in existing_values]
            if not added:
                continue
            entry: dict[str, Any] = {
                "values": existing_values + added, "cardinality": cardinality,
            }
            if new_bounds is not None:
                entry["bounds"] = new_bounds
            state.slots[slot] = entry
            for value in added:
                _record(state, turn, slot, "ADD", value)
            continue

        # REPLACE (or first-time SET when nothing was there).
        superseded = [v for v in existing_values if v not in values]
        unchanged = (
            existing is not None
            and existing["values"] == values
            and existing.get("bounds") == new_bounds
            and not remove
        )
        if unchanged:
            continue
        entry = {"values": list(values), "cardinality": cardinality}
        if new_bounds is not None:
            entry["bounds"] = new_bounds
        state.slots[slot] = entry
        op_name = "REPLACE" if (existing_values or remove) else "SET"
        for value in values:
            _record(state, turn, slot, op_name, value, superseded or None)
        if superseded:
            _supersede_evidence(state, superseded)


def update_evidence(
    state: SessionState, message: str, delta: dict[str, dict[str, Any]], turn: int
) -> None:
    """CP 2.8 - keep residual free-text as distilled evidence.

    Stored only when the turn carries genuine user signal that did not become
    a slot. Skipped when the message is empty, is purely a slot mention
    (``"jacket"``), or is a conversational non-answer (``"ask me about one
    specific attribute"``, ``"I don't have a preference..."``).

    Residual = message content tokens minus tokens already consumed by an
    extracted slot value or a slot-marker word. ``"gift for my dad"`` ->
    ``{gift, dad}`` -> stored once (deduped by normalized form).

    Each entry is ``{"turn", "text", "normalized", "status": "active"}``.
    The ``status`` field is the hook Phase 3 uses to supersede free-text
    evidence with the same mechanism it supersedes slots (D Finding 1).
    """
    if not message or not message.strip():
        return
    if _NON_ANSWER_RE.search(message):
        return
    # An override instruction ("actually, ignore my earlier preference, ...")
    # whose slots were applied is plumbing, not new free-text evidence.
    if _REPLACE_CUE_RE.search(message) and delta:
        return
    consumed: set[str] = set(_EVIDENCE_MARKER_TOKENS) | _OVERRIDE_PLUMBING
    for incoming in delta.values():
        for value in incoming.get("values", ()):
            consumed.update(terms(value))
        for value in incoming.get("remove", ()):
            consumed.update(terms(value))
    residual = [token for token in terms(message) if token not in consumed]
    if not residual:
        return
    normalized = " ".join(terms(message))
    for entry in state.evidence:
        if isinstance(entry, dict) and entry.get("normalized") == normalized:
            return
    state.evidence.append(
        {"turn": turn, "text": message, "normalized": normalized, "status": "active"}
    )


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
