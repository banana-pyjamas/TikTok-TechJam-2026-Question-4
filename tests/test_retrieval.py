"""CP 5.1 - 5.4, 5.8 — multi-route retrieval on a tiny catalog.

Recall@50/100/300 (CP 5.5-5.7) is measured on the real catalog in
``test_retrieval_recall.py``.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.agent import Agent
from starter.contracts import Context, SessionState
from starter.retrieval import (
    attribute_route,
    bm25_route,
    category_route,
    retrieve,
)
from starter.state import update_state


_CATALOG_ROWS = [
    {"parent_asin": "B0RUN", "title": "Blue running shoe",
     "categories": ["Clothing", "Shoes", "Athletic"], "features": ["breathable mesh"],
     "details": {"department": "womens"}, "store": "Ex", "description": ["road running"]},
    {"parent_asin": "B0BOOT", "title": "Black leather boot",
     "categories": ["Clothing", "Boots"], "features": ["full grain leather", "waterproof"],
     "details": {"department": "mens"}, "store": "Nike", "description": ["winter"]},
    {"parent_asin": "B0SANDAL", "title": "Brown leather sandal",
     "categories": ["Clothing", "Sandals"], "features": ["leather strap"],
     "details": {"department": "womens"}, "store": "Ex", "description": ["summer"]},
    {"parent_asin": "B0SOCK", "title": "Wool hiking sock",
     "categories": ["Clothing", "Socks"], "features": ["merino wool"],
     "details": {"department": "unisex"}, "store": "Ex", "description": ["cushioned"]},
    {"parent_asin": "B0BARE", "title": "Minimalist boot",  # missing features/desc/price
     "categories": ["Clothing", "Boots"], "features": [], "details": {},
     "store": "", "description": []},
]


def _ctx(message: str = "", **slots: list[str]) -> Context:
    state = SessionState(session_id="s")
    for name, values in slots.items():
        cardinality = "single" if name in ("category", "size", "brand", "budget") else "multi"
        state.slots[name] = {"values": list(values), "cardinality": cardinality}
    return Context(session_id="s", turn=1, user_message=message, state=state)


class _CatalogFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        path = Path(cls._tmp.name) / "catalog.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in _CATALOG_ROWS), encoding="utf-8"
        )
        cls._agent = Agent(path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _asins(self, candidates) -> list[str]:
        return [c.parent_asin for c in candidates]


class CategoryRouteTest(_CatalogFixture):
    """CP 5.1."""

    def test_returns_products_in_the_category_slot(self) -> None:
        rows = category_route(self._agent.connection, _ctx(category=["boots"]), 50)
        asins = {a for a, _ in rows}
        self.assertEqual(asins, {"B0BOOT", "B0BARE"})

    def test_singular_plural_are_both_matched(self) -> None:
        rows = category_route(self._agent.connection, _ctx(category=["sock"]), 50)
        self.assertEqual({a for a, _ in rows}, {"B0SOCK"})

    def test_no_category_slot_returns_empty(self) -> None:
        self.assertEqual(category_route(self._agent.connection, _ctx(), 50), [])


class AttributeRouteTest(_CatalogFixture):
    """CP 5.3."""

    def test_material_slot(self) -> None:
        rows = attribute_route(self._agent.connection, _ctx(material=["leather"]), 50)
        self.assertEqual({a for a, _ in rows}, {"B0BOOT", "B0SANDAL"})

    def test_brand_slot_matches_store(self) -> None:
        rows = attribute_route(self._agent.connection, _ctx(brand=["nike"]), 50)
        self.assertIn("B0BOOT", {a for a, _ in rows})

    def test_union_of_several_attributes(self) -> None:
        rows = attribute_route(
            self._agent.connection, _ctx(color=["brown"], material=["wool"]), 50
        )
        self.assertEqual({a for a, _ in rows}, {"B0SANDAL", "B0SOCK"})

    def test_no_attribute_slots_returns_empty(self) -> None:
        self.assertEqual(attribute_route(self._agent.connection, _ctx(), 50), [])


class UnionTest(_CatalogFixture):
    """CP 5.2 / 5.4."""

    def test_bm25_plus_category_union_is_not_an_intersection(self) -> None:
        # BM25 on "running" hits only B0RUN; category "boots" hits B0BOOT/B0BARE.
        ctx = _ctx("running", category=["boots"])
        pool = self._asins(retrieve(self._agent.connection, ctx, 300))
        self.assertIn("B0RUN", pool)
        self.assertIn("B0BOOT", pool)
        self.assertIn("B0BARE", pool)

    def test_candidate_carries_every_contributing_route(self) -> None:
        ctx = _ctx("black leather boot", category=["boots"], material=["leather"])
        pool = retrieve(self._agent.connection, ctx, 300)
        boot = next(c for c in pool if c.parent_asin == "B0BOOT")
        self.assertEqual(set(boot.route_sources), {"bm25", "category", "attribute"})
        for route in ("bm25", "category", "attribute"):
            self.assertIsInstance(boot.route_scores[route], float)

    def test_pool_is_deduplicated(self) -> None:
        ctx = _ctx("leather boot", category=["boots"], material=["leather"])
        pool = self._asins(retrieve(self._agent.connection, ctx, 300))
        self.assertEqual(len(pool), len(set(pool)))

    def test_three_route_union_ordering_is_deterministic(self) -> None:
        ctx = _ctx("leather", category=["boots"], material=["leather"])
        a = self._asins(retrieve(self._agent.connection, ctx, 300))
        b = self._asins(retrieve(self._agent.connection, ctx, 300))
        self.assertEqual(a, b)

    def test_limit_is_respected(self) -> None:
        ctx = _ctx("leather boot sock shoe sandal", category=["boots"])
        self.assertLessEqual(len(retrieve(self._agent.connection, ctx, 2)), 2)


class MissingMetadataSafetyTest(_CatalogFixture):
    """CP 5.8 — a product missing fields is never eliminated by the UNION."""

    def test_product_with_empty_features_price_desc_still_retrievable(self) -> None:
        # B0BARE has features=[], details={}, store="", description=[], no price.
        ctx = _ctx("boot", category=["boots"], color=["black"], material=["leather"])
        pool = self._asins(retrieve(self._agent.connection, ctx, 300))
        self.assertIn("B0BARE", pool, "missing metadata must not eliminate a product")

    def test_missing_attribute_does_not_drop_a_category_match(self) -> None:
        # Ask for a red boot; nothing is red, but boots must still be in the pool.
        ctx = _ctx("", category=["boots"], color=["red"])
        pool = self._asins(retrieve(self._agent.connection, ctx, 300))
        self.assertIn("B0BOOT", pool)
        self.assertIn("B0BARE", pool)

    def test_retrieval_does_not_mutate_state(self) -> None:
        import copy
        state = SessionState(session_id="s")
        update_state(state, "black leather boot", 1)
        snapshot = copy.deepcopy(state)
        ctx = Context(session_id="s", turn=1, user_message="black leather boot", state=state)
        retrieve(self._agent.connection, ctx, 300)
        self.assertEqual(state, snapshot)


class RespondPoolIntegrationTest(_CatalogFixture):
    def test_respond_still_returns_bm25_top10_unchanged(self) -> None:
        self._agent.reset("s", {"summary": "x"})
        payload = self._agent.respond("s", "black leather waterproof boot", 1, 10)
        self.assertEqual(payload["recommendations"][0]["parent_asin"], "B0BOOT")
        self.assertLessEqual(len(payload["recommendations"]), 10)


if __name__ == "__main__":
    unittest.main()
