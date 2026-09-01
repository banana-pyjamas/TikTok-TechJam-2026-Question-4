"""Candidate-scoped vocabulary.

The words that matter are the ones the CURRENT candidates actually use. A
global catalog vocabulary would answer "what words exist in clothing retail";
this answers "what words distinguish the 300 products still in play", which is
the question the clarification layer needs and the question a
free-text term has to be grounded against.

Four properties:

  extraction     per-product terms are indexed once (``product_terms``, stored
                 by ``catalog_meta``); a turn's vocabulary is the document
                 frequency of those terms WITHIN the candidate pool.

  empty pool     every function here is total, on ``None`` as well as on empty.
                 An empty pool, a pool of products with no indexed terms, a
                 pool whose rows are missing from the side table, and a
                 ``None`` where a pool or a vocabulary was expected all yield
                 an empty result rather than an exception. ``None`` matters
                 because the consumer reads this out of ``Context.derived``,
                 where a missing key is ``None``.

  noise control  in-pool frequency bounds, applied only when the pool is big
                 enough for a frequency to mean anything.

  grounding      a free-form word ("warm") maps to the catalog words this pool
                 MATERIALLY uses ("insulated", "thermal", "fleece", "lined").
                 A materiality test, not word-sense disambiguation -- see
                 ``ground`` for the counterexample that survives it.

WHY IN-POOL FREQUENCY, NOT GLOBAL IDF

Global IDF would need a second pass over the 50k catalog at index build, and
it answers the wrong question: "waterproof" is rare catalog-wide and therefore
high-IDF, but in a pool of rain jackets it is in every candidate and
distinguishes nothing. Scoping the statistics to the pool is not a compromise
forced by cost -- it is the phase.

DETERMINISM

No embeddings, no network, no model. Term order is fully determined: document
frequency descending, then the term itself. The grounding map is ordinary
shopping English, not the evaluator's phrasing -- keying on simulator strings
would ground the public set well and generalize to nothing, the same rule the
The cue vocabularies follow.

Nothing in the shipped agent calls this yet; see ``build_vocabulary`` for why
that is deliberate.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from typing import Any, Iterable

from starter.contracts import Candidate
from starter.text import flatten_text, terms

# Per-product cap. The stored list is title-first, so the cap keeps the most
# salient words: a title and category path plus the opening features are what
# name a product, while the tail of a long description is marketing prose.
INDEX_TERM_LIMIT = 40

# Below this many candidates, an in-pool document frequency is not a
# statistic. Two of three candidates sharing a word says nothing about whether
# the word discriminates, so the frequency bounds are skipped entirely
# and only structural filtering applies.
NOISE_FLOOR_POOL = 8

# A term in exactly one candidate cannot separate the pool and is usually a
# typo, a model number, or one seller's marketing word.
MIN_DOCUMENT_FREQUENCY = 2

# A term in nearly every candidate describes the pool, not a candidate within
# it. "Clothing" in a pool of clothing is not information.
#
# Set at 0.8, not 0.5, and the difference matters. A 0.5 ceiling collapses
# ``most_discriminative`` into plain frequency order: if no term may exceed
# half the pool, then "closest to half" and "most frequent" are the same
# ordering, and the function returns exactly the frequency ranking. Measured
# on real pools, the two lists were identical. Removing near-universal terms
# and finding even splitters are different jobs, and one threshold cannot do
# both -- so the ceiling only removes what describes the pool, and
# ``most_discriminative`` does the splitting.
MAX_DOCUMENT_RATIO = 0.8

# A mapped word must describe at least this share of the pool before grounding
# will propose it. See ``ground`` for what this does and does not buy.
#
# 5% of a 300-candidate pool is 15 products: enough that the word names a real
# option among these candidates rather than one seller's phrasing. Measured
# against a hostile grid (21 map keys x 4 non-clothing
# pools), the floor cuts non-empty results from 59/84 to 19/84 while keeping
# 6 of 6 of the senses this phase claims. Those 6 survive unchanged from a 0%
# floor to a 10% one and break only at 15%, so 5% sits mid-plateau rather than
# on a cliff -- the standard W_MATCH was set by.
MIN_MAPPED_SUPPORT = 0.05

# The cap exists to bound memory, not to choose -- so it is set above what a
# real turn produces. Measured on the live dialogue, a 300-candidate pool
# yields ~1250 terms after noise control; at the original 400 the cap bound on
# 100% of turns and discarded 31% of all surviving term observations, which
# made it the second-largest filter in the phase and a silent one, since it
# cuts by frequency and so removes exactly the rare-but-precise words grounding
# needs ("sherpa", "ripstop"). A per-turn dict of ints is not what constrains
# this system. Consumers wanting the few most useful terms call
# ``most_discriminative``.
VOCABULARY_LIMIT = 1500

# Structural noise -- words describing the LISTING rather than the product.
# These survive the frequency bounds because they are common but not
# universal, and they are never what a shopper means. Deliberately short: the
# frequency bounds do the general work, and this covers only the catalog's own
# schema vocabulary, which no amount of in-pool statistics can recognise.
#
# The care-instruction group ("hand", "machine", "wash", ...) is here on
# measured evidence, not suspicion: before it was added, "hand", "machine" and
# "only" -- from "Hand Wash Only" -- ranked in the top ten terms of a winter
# jacket pool, above "hooded" and "puffer".
#
# "material" is excluded as a schema word while the materials themselves
# ("polyester", "fleece") are kept: the catalog writes "Material: Polyester",
# and it is the second word that names the product.
BOILERPLATE = frozenset({
    "imported", "manufacturer", "discontinued", "asin", "dimensions",
    "inches", "ounces", "pounds", "package", "packaging", "item", "model",
    "number", "date", "available", "department", "shipping", "returns",
    "warranty", "amazon", "seller", "please", "click", "buy", "purchase",
    "customer", "service", "contact", "email", "guarantee", "satisfaction",
    "quality", "products", "product", "brand", "new", "free", "day", "days",
    "size", "sizes", "chart", "check", "note", "due", "may", "vary",
    "different", "monitor", "screen", "picture", "actual", "color",
    # care instructions and listing prose
    "hand", "machine", "wash", "washable", "dry", "clean", "cleaning",
    "care", "instructions", "bleach", "iron", "tumble", "only", "made",
    "usa", "material", "materials", "made-in", "occasion", "suitable",
})

# free-form shopper language to the catalog words that express it.
#
# The shopper says "warm"; the catalog says "insulated", "thermal", "fleece".
# This map only PROPOSES; ``ground`` keeps just the words the current
# candidates actually use, so a proposal that is wrong for this pool costs
# nothing. That is what makes it grounding rather than query expansion.
#
# Ordinary shopping English, kept deliberately small. Every entry is a word a
# shopper volunteers and the catalog does not use, or uses differently.
GROUNDING: dict[str, tuple[str, ...]] = {
    "warm": ("insulated", "thermal", "fleece", "sherpa", "quilted", "down",
             "wool", "lined", "padded", "warmth"),
    "warmth": ("insulated", "thermal", "fleece", "wool", "lined", "padded"),
    "cold": ("insulated", "thermal", "fleece", "wool", "windproof"),
    "winter": ("insulated", "thermal", "fleece", "wool", "down", "parka"),
    "summer": ("breathable", "lightweight", "mesh", "linen", "short", "airy"),
    "cool": ("breathable", "mesh", "ventilated", "lightweight"),
    "comfortable": ("soft", "stretch", "cushioned", "padded", "breathable",
                    "relaxed", "comfort"),
    "comfy": ("soft", "stretch", "cushioned", "padded", "relaxed", "comfort"),
    "soft": ("cotton", "fleece", "cashmere", "plush", "velvet", "brushed"),
    "lightweight": ("light", "breathable", "thin", "mesh", "packable"),
    "breathable": ("mesh", "ventilated", "moisture", "wicking", "airy"),
    "durable": ("reinforced", "ripstop", "rugged", "heavyweight", "sturdy",
                "resistant", "tear"),
    "waterproof": ("water", "resistant", "rain", "sealed", "repellent"),
    "stretchy": ("stretch", "spandex", "elastane", "elastic", "flexible"),
    "casual": ("everyday", "relaxed", "weekend"),
    "formal": ("dress", "business", "tailored", "suit", "professional"),
    "work": ("business", "office", "professional", "dress"),
    "gym": ("athletic", "performance", "training", "moisture", "wicking"),
    "workout": ("athletic", "performance", "training", "moisture", "wicking"),
    "hiking": ("trail", "outdoor", "traction", "rugged", "hike"),
    "travel": ("packable", "wrinkle", "lightweight", "compact"),
}


def product_terms(product: dict) -> list[str]:
    """the bounded term list stored for one product.

    Title first, then the category path, then features, then description:
    the order the cap truncates against, so what survives is what names the
    product rather than the tail of its marketing copy.

    Structural filtering only. Tokens carrying digits are dropped (sizes,
    model numbers, dates), as are very short tokens and catalog boilerplate.
    Frequency-based filtering is NOT done here -- it cannot be, because it
    depends on the pool this product later lands in.
    """
    ordered: list[str] = []
    seen: set[str] = set()
    for field in ("title", "categories", "features", "description"):
        for token in terms(flatten_text(product.get(field))):
            if token in seen:
                continue
            if len(token) < 3 or any(character.isdigit() for character in token):
                continue
            if token in BOILERPLATE:
                continue
            seen.add(token)
            ordered.append(token)
            if len(ordered) >= INDEX_TERM_LIMIT:
                return ordered
    return ordered


def pool_terms(
    connection: sqlite3.Connection, parent_asins: Iterable[str]
) -> dict[str, list[str]]:
    """Indexed terms for the given products, keyed by ``parent_asin``.

    Its own query rather than a widened ``catalog_meta.lookup`` so ranking --
    which runs every turn and does not need this column -- keeps paying for
    exactly what it reads.

    A product with no row simply comes back absent; an empty input
    issues no query at all.
    """
    # Filter BEFORE stringifying: ``str(None)`` is "None", which is truthy and
    # would be sent to SQLite as a literal asin to look up.
    asins = [str(asin) for asin in parent_asins if asin]
    if not asins:
        return {}
    placeholders = ",".join("?" * len(asins))
    rows = connection.execute(
        f"SELECT parent_asin, vocab FROM product_meta "
        f"WHERE parent_asin IN ({placeholders})",
        asins,
    ).fetchall()
    return {str(row[0]): (row[1].split() if row[1] else []) for row in rows}


def build_vocabulary(
    connection: sqlite3.Connection,
    candidates: list[Candidate],
    limit: int = VOCABULARY_LIMIT,
) -> dict[str, Any]:
    """the vocabulary of one turn's candidate pool.

    Returns a plain dict, not a new type: per the contracts rule, a new signal
    adds keys to a generic container (``Context.derived``) rather than a
    frozen field or a sixth dataclass.

        pool_size   candidates considered
        terms       ``{term: in-pool document frequency}``, noise-filtered and
                    capped, ordered by frequency descending then term
        dropped     why terms were discarded, for diagnostics

    Not called from ``agent.respond``. Clarification is the consumer, and this
    repo
    just deleted a ``build_strategy`` call for computing a value nothing read;
    wiring an unread vocabulary in would repeat that. It costs one indexed
    query per turn when a consumer arrives.
    """
    pool = [c.parent_asin for c in (candidates or ()) if isinstance(c, Candidate)]
    unique = list(dict.fromkeys(asin for asin in pool if asin))
    dropped = {"rare": 0, "ubiquitous": 0, "over_limit": 0}
    if not unique:
        return {"pool_size": 0, "terms": {}, "dropped": dropped}

    indexed = pool_terms(connection, unique)
    frequency: Counter = Counter()
    for asin in unique:
        # A term counts once per PRODUCT, never per mention: document
        # frequency, so a word repeated through one long description cannot
        # look like agreement across the pool.
        frequency.update(set(indexed.get(asin, ())))

    pool_size = len(unique)
    if pool_size >= NOISE_FLOOR_POOL:
        minimum = MIN_DOCUMENT_FREQUENCY
        maximum = pool_size * MAX_DOCUMENT_RATIO
    else:
        minimum, maximum = 1, float(pool_size)

    kept: list[tuple[str, int]] = []
    for term, count in frequency.items():
        if count < minimum:
            dropped["rare"] += 1
        elif count > maximum:
            dropped["ubiquitous"] += 1
        else:
            kept.append((term, count))

    kept.sort(key=lambda pair: (-pair[1], pair[0]))
    if len(kept) > limit:
        dropped["over_limit"] = len(kept) - limit
        kept = kept[:limit]
    return {"pool_size": pool_size, "terms": dict(kept), "dropped": dropped}


def most_discriminative(
    vocabulary: dict[str, Any], count: int = 10
) -> tuple[str, ...]:
    """The terms that would split this pool most evenly.

    A term present in half the candidates divides them in two; one present in
    99% or 1% barely moves. So the ordering is by distance from a half, not by
    frequency -- which is what a clarification layer wants when choosing what
    to ask about, and is why ``terms`` is capped generously rather than
    pre-filtered to the "top" terms by frequency.

    Total on an empty vocabulary.
    """
    vocabulary = vocabulary if isinstance(vocabulary, dict) else {}
    pool_size = vocabulary.get("pool_size") or 0
    entries = vocabulary.get("terms") or {}
    if not pool_size or not entries or count <= 0:
        return ()
    ordered = sorted(
        entries.items(),
        key=lambda pair: (abs(pair[1] / pool_size - 0.5), -pair[1], pair[0]),
    )
    return tuple(term for term, _ in ordered[:count])


def ground(
    term: str,
    vocabulary: dict[str, Any],
    min_support: float = MIN_MAPPED_SUPPORT,
) -> tuple[str, ...]:
    """a free-form word to the pool's own words for it.

    Two sources: the word itself when the candidates use it, then the mapped
    catalog words the candidates use MATERIALLY -- in at least ``min_support``
    of the pool. An unmapped word the pool does not use grounds to nothing,
    deliberately, because a guess would send the layers above chasing a word
    no candidate contains.

    WHAT THIS GUARANTEES, AND WHAT IT DOES NOT

    Guaranteed: every word returned is one these candidates actually use, and
    a mapped word describes a real share of them rather than one listing.

    NOT guaranteed: that the pool means what the shopper meant. This is a
    materiality test, not word-sense disambiguation. The counterexample is
    concrete and survives the floor -- on a pool of sunglasses::

        ground("waterproof", sunglasses)  ->  ("resistant",)

    because sunglasses are scratch- and impact-resistant, and "resistant" is
    the same string either way. An earlier version of this docstring claimed
    "the same word on a watches pool grounds to nothing... a proposal wrong
    for this pool costs nothing". That was false: it was verified against a
    synthetic pool, and on the real one ``ground("warm", watches)`` returned
    ("padded", "down") from padded straps (D-V1). The floor removes that case
    and most like it, but polysemy is not solvable at this layer without a
    model, and no caller should be written as though it were.

    So a consumer must treat a grounded word as a CANDIDATE
    phrasing to consider, never as evidence that the pool shares the
    shopper's sense. ``grounding_support`` exposes the numbers for a consumer
    that wants a stricter bar than the default.

    Matching is exact. No stemming: three ``_stem`` helpers already exist in
    this codebase and a fourth that disagreed with them would be worse than
    none. Plurals are handled where they matter by listing both forms in
    ``GROUNDING``.
    """
    vocabulary = vocabulary if isinstance(vocabulary, dict) else {}
    entries = vocabulary.get("terms") or {}
    if not isinstance(term, str) or not entries:
        return ()
    token = term.strip().lower()
    if not token:
        return ()
    pool_size = vocabulary.get("pool_size") or 0
    # The shopper's own word is not an inference about meaning, so the
    # materiality floor -- which exists to suppress incidental inferences --
    # does not apply to it. A rare word the shopper actually said is still
    # worth surfacing; a rare word we guessed is not.
    found = [token] if token in entries else []
    for word in GROUNDING.get(token, ()):
        if word == token or word not in entries:
            continue
        if pool_size and entries[word] / pool_size < min_support:
            continue
        found.append(word)
    # Present-in-most-candidates first: of several words the pool uses for one
    # idea, the commonest is the pool's own way of saying it.
    return tuple(sorted(dict.fromkeys(found),
                        key=lambda word: (-entries[word], word)))


def grounding_support(
    term: str, vocabulary: dict[str, Any]
) -> dict[str, float]:
    """Share of the pool using each word ``term`` could ground to.

    Unfiltered, so a consumer can apply its own bar rather than inheriting
    this module's. Includes words below ``MIN_MAPPED_SUPPORT`` -- seeing that
    a mapped word sits at 1% is how a caller learns the grounding is thin.
    """
    vocabulary = vocabulary if isinstance(vocabulary, dict) else {}
    entries = vocabulary.get("terms") or {}
    pool_size = vocabulary.get("pool_size") or 0
    if not isinstance(term, str) or not entries or not pool_size:
        return {}
    token = term.strip().lower()
    if not token:
        return {}
    words = ([token] if token in entries else []) + [
        word for word in GROUNDING.get(token, ())
        if word in entries and word != token
    ]
    return {word: entries[word] / pool_size for word in dict.fromkeys(words)}


def ground_all(
    tokens: Iterable[str], vocabulary: dict[str, Any]
) -> dict[str, tuple[str, ...]]:
    """``ground`` over many words, keeping only those that grounded.

    Useful for a whole evidence string: the words that ground are the ones
    this pool can actually act on, and the ones that do not are exactly the
    shopper's language the catalog has no expression for.
    """
    grounded: dict[str, tuple[str, ...]] = {}
    for token in tokens or ():
        if not isinstance(token, str):
            continue
        result = ground(token, vocabulary)
        if result:
            grounded[token.strip().lower()] = result
    return grounded
