"""Match Reliability (Phase 11, CP 11.2).

How much a catalog verdict about one SLOT is worth. Not how much the shopper
meant the constraint -- that is Evidence Confidence, and it lives on the slot
entry in ``SessionState`` (CP 11.1). The two are independent, which is the
whole point of the phase: strong intent against weak catalog evidence
(CP 11.4) and weak intent against strong catalog evidence (CP 11.5) are
different situations and must not be collapsed into one number.

WHERE THE NUMBER COMES FROM

Coverage: the share of the frozen catalog for which we can recognise ANY
value for that slot. Measured, not assigned::

    category  100.0%      brand      99.4%
    material   62.6%      color      46.8%
    budget     20.8%      size        9.6%

A field the catalog populates for nearly every product is one it actually
curates. A field present for one product in ten is incidental text that
happened to match a pattern, and a mismatch on it means correspondingly
little. That is not a new argument: it is the reason ``ranking.VIOLATION_SLOTS``
already excludes ``size`` by hand, with the comment "size metadata is sparse
and inconsistent". Phase 11 keeps that judgement and makes it continuous and
derived rather than binary and hardcoded.

Deliberately label-free. Reliability could instead be fitted to the 200 public
targets -- "how often does this slot condemn the right product" -- and that
would score better on the public set by construction. It would also be fitting
the answers rather than measuring the catalog, and would not transfer to the
private 800. The label-based version is computed by
``tools/phase11_confidence.py`` as a CHECK on this one, never as its source.

Absent statistics, every slot is fully reliable, so a caller that never
computes coverage gets exactly the pre-Phase-11 behaviour.
"""

from __future__ import annotations

import sqlite3

from starter.catalog_meta import TABLE

# Slot -> the product_meta column that carries its evidence. ``budget`` is a
# REAL column, so it is counted by NOT NULL rather than by being non-empty.
_SLOT_COLUMN = {
    "category": "cats",
    "color": "colors",
    "material": "materials",
    "brand": "store",
    "size": "sizes",
}
_NUMERIC_SLOTS = {"budget": "price"}

# A slot we have no statistic for is trusted. An unknown slot must not be
# silently discounted: that would be a hard filter arriving through the back
# door, which is exactly what CP 11.5 forbids.
DEFAULT_RELIABILITY = 1.0

# Floor under the derived value. A slot is never worth ZERO -- a zero would
# make its verdicts unreachable and turn "unreliable" into "ignored", losing
# the real signal that survives in the 9.6% of products that do declare a
# size. It also keeps the weighting continuous: nothing falls off a cliff.
MIN_RELIABILITY = 0.1


def slot_coverage(connection: sqlite3.Connection) -> dict[str, float]:
    """Share of the catalog carrying a recognisable value, per slot.

    One aggregate query per slot over the indexed side table, run once when
    the agent builds its index -- never per turn.

    Returns an empty mapping if the side table is missing or empty, which the
    caller reads as "no statistics" and therefore "trust everything".
    """
    try:
        total = connection.execute(f"SELECT count(*) FROM {TABLE}").fetchone()[0]
    except sqlite3.Error:
        return {}
    if not total:
        return {}
    coverage: dict[str, float] = {}
    for slot, column in _SLOT_COLUMN.items():
        found = connection.execute(
            f"SELECT count(*) FROM {TABLE} WHERE {column} != ''"
        ).fetchone()[0]
        coverage[slot] = found / total
    for slot, column in _NUMERIC_SLOTS.items():
        found = connection.execute(
            f"SELECT count(*) FROM {TABLE} WHERE {column} IS NOT NULL"
        ).fetchone()[0]
        coverage[slot] = found / total
    return coverage


def match_reliability(coverage: dict[str, float] | None) -> dict[str, float]:
    """Coverage -> per-slot reliability in ``[MIN_RELIABILITY, 1.0]``.

    The mapping is the identity, floored. Coverage is already a share, and
    inventing a curve over it -- a power, a logistic, a hand-drawn table --
    would add parameters that no measurement in this repo justifies. When
    something eventually does justify one, it belongs here and nowhere else.
    """
    if not coverage:
        return {}
    return {
        slot: max(MIN_RELIABILITY, min(1.0, float(share)))
        for slot, share in coverage.items()
        if isinstance(share, (int, float)) and share == share
    }


def reliability_of(reliabilities: dict[str, float] | None, slot: str) -> float:
    """One slot's reliability, defaulting to fully trusted.

    Total on ``None`` and on an unknown slot: the caller is a scoring path
    that must never raise, and an absent statistic means "no reason to
    discount", not "discount to zero".
    """
    if not isinstance(reliabilities, dict):
        return DEFAULT_RELIABILITY
    value = reliabilities.get(slot, DEFAULT_RELIABILITY)
    if not isinstance(value, (int, float)) or value != value:
        return DEFAULT_RELIABILITY
    return max(MIN_RELIABILITY, min(1.0, float(value)))
