"""Compact per-product constraint signals.

Constraint ranking needs to know, for each candidate, what the catalog
actually asserts about colour, material, category, brand, size and price.
Keeping the full product text in Python costs ~87 MB for the frozen 50k
catalog; the recognised signals alone are ~83 bytes per product, so they live
in an indexed SQLite side-table and are looked up only for the <=300
candidates in a turn's pool (~3 ms).

Storing what the catalog ASSERTS -- rather than only what it matches -- is
what lets ranking tell a real mismatch (the product says "blue", we asked for
"black") apart from missing metadata (the product names no colour at all).
The latter is UNKNOWN, never a violation.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from starter.popularity import DEFAULT_SCALE, popularity_feature
from starter.profile import ALL_TRAIT_TERMS
from starter.state import _COLOR_ALIASES, _COLORS, _MATERIALS
from starter.vocabulary import product_terms

_TOKEN = re.compile(r"[a-z0-9.]+")
# Sizes as they appear in catalog metadata, e.g. "size 10", "size: XL".
_SIZE_RE = re.compile(
    r"\bsize\b[:\s]*([0-9]{1,2}(?:\.5)?|xxs|xs|s|m|l|xl|xxl|xxxl)\b", re.I
)

TABLE = "product_meta"


def create_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        f"CREATE TABLE IF NOT EXISTS {TABLE} ("
        "parent_asin TEXT PRIMARY KEY, colors TEXT, materials TEXT, "
        "cats TEXT, store TEXT, sizes TEXT, price REAL, traits TEXT, "
        "vocab TEXT, popularity REAL)"
    )


def signals(
    product: dict,
) -> tuple[str, str, str, str, str, float | None, str, str, float | None]:
    """The row this product contributes: colours, materials, category tokens,
    store, sizes, price, profile-trait vocabulary, candidate vocabulary,
    popularity.

    New signals are APPENDED. Existing positions are part of how callers read
    this tuple (``tests/test_ranking.py`` indexes ``[2]`` for category tokens),
    and ``agent._build_index`` derives its INSERT placeholders from the row
    length, so appending is the one safe way to grow it.
    """
    categories = " ".join(str(value) for value in (product.get("categories") or [])).lower()
    store = str(product.get("store") or "").lower()
    text = " ".join([
        str(product.get("title") or ""),
        categories,
        store,
        " ".join(str(value) for value in (product.get("features") or [])),
        " ".join(f"{key} {value}" for key, value in (product.get("details") or {}).items()),
        " ".join(str(value) for value in (product.get("description") or [])),
    ]).lower()

    tokens = set(_TOKEN.findall(text))
    colors = sorted({_COLOR_ALIASES.get(token, token) for token in tokens & _COLORS})
    materials = sorted(tokens & _MATERIALS)
    sizes = sorted({match.group(1).lower() for match in _SIZE_RE.finditer(text)})
    try:
        price: float | None = float(product.get("price"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        price = None

    return (
        " ".join(colors),
        " ".join(materials),
        " ".join(sorted(set(_TOKEN.findall(categories)))),
        store,
        " ".join(sizes),
        price,
        " ".join(sorted(tokens & ALL_TRAIT_TERMS)),
        # Order-sensitive (title first), so unlike the sets above it
        # is stored as produced rather than sorted.
        " ".join(product_terms(product)),
        # log1p of the review count, precomputed once here
        # rather than per candidate per turn. NULL when the catalog gives no
        # usable count -- absence is a missing signal, never a zero.
        popularity_feature(product.get("rating_number")),
    )


def popularity_scale(connection: sqlite3.Connection) -> dict[str, float]:
    """Catalog-wide popularity normalisation, computed once per agent.

    Returns ``{"scale": ..., "missing": ...}``:

      scale    the largest ``log1p`` review count in the catalog, so the
               normalised feature lands in ``[0, 1]`` with no hand-set
               reference;
      missing  the MEDIAN, which is what a product with no usable count scores
               -- absent data is typical, never worst.

    An empty or absent side table yields ``{}``, which consumers read as "no
    statistics" and therefore "no popularity signal": the prior switches
    itself off rather than inventing a scale.

    Lives here rather than in ``popularity`` because ``popularity`` must not
    import this module -- see that module's docstring.
    """
    try:
        rows = connection.execute(
            f"SELECT popularity FROM {TABLE} WHERE popularity IS NOT NULL "
            f"ORDER BY popularity"
        ).fetchall()
    except sqlite3.Error:
        return {}
    values = [float(row[0]) for row in rows]
    if not values:
        return {}
    return {"scale": values[-1] or DEFAULT_SCALE,
            "missing": values[len(values) // 2]}


def lookup(
    connection: sqlite3.Connection, parent_asins: list[str]
) -> dict[str, dict[str, Any]]:
    """Constraint signals for the given products, keyed by ``parent_asin``.

    A product with no row simply comes back absent, which ranking treats as
    UNKNOWN for every slot -- never as a violation.
    """
    if not parent_asins:
        return {}
    placeholders = ",".join("?" * len(parent_asins))
    rows = connection.execute(
        f"SELECT parent_asin, colors, materials, cats, store, sizes, price, "
        f"traits, popularity FROM {TABLE} WHERE parent_asin IN ({placeholders})",
        parent_asins,
    ).fetchall()
    return {
        str(row[0]): {
            "color": set(row[1].split()) if row[1] else set(),
            "material": set(row[2].split()) if row[2] else set(),
            "cats": set(row[3].split()) if row[3] else set(),
            "store": row[4] or "",
            "sizes": set(row[5].split()) if row[5] else set(),
            "price": row[6],
            "traits": set(row[7].split()) if row[7] else set(),
            "popularity": row[8],
        }
        for row in rows
    }
