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


# --------------------------------------------------------------------------
# B Phase 5 review: the union must remain a real union when BM25 alone can
# fill the whole pool budget.
# --------------------------------------------------------------------------

_SATURATED_ROWS = (
    # 40 products that all match the BM25 query "widget" -- enough to fill a
    # pool budget of 20 on their own.
    [
        {"parent_asin": f"B0W{index:02d}", "title": f"widget number {index}",
         "categories": ["Clothing", "Widgets"], "features": ["widget"],
         "details": {}, "store": "Ex", "description": ["a widget"]}
        for index in range(40)
    ]
    # findable only by the category route (in category, never says "widget")
    + [{"parent_asin": "B0CATONLY", "title": "plain moccasin",
        "categories": ["Clothing", "Boots"], "features": [], "details": {},
        "store": "Ex", "description": []}]
    # findable only by the attribute route (has the material, wrong category)
    + [{"parent_asin": "B0ATTRONLY", "title": "cashmere pouch",
        "categories": ["Clothing", "Pouches"], "features": ["cashmere"],
        "details": {}, "store": "Ex", "description": []}]
)


class UnionBudgetTest(unittest.TestCase):
    """CP 5.2/5.4 blocker regression: BM25 saturating the budget must not
    starve the auxiliary routes out of the FINAL pool."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        path = Path(cls._tmp.name) / "catalog.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in _SATURATED_ROWS), encoding="utf-8"
        )
        cls._agent = Agent(path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _ctx(self) -> Context:
        return _ctx("widget", category=["boots"], material=["cashmere"])

    def test_bm25_alone_saturates_the_pool_budget(self) -> None:
        rows = bm25_route(self._agent.connection, self._ctx(), 20)
        self.assertGreaterEqual(len(rows), 20, "precondition: BM25 fills the budget")
        self.assertNotIn("B0CATONLY", {a for a, _ in rows})
        self.assertNotIn("B0ATTRONLY", {a for a, _ in rows})

    def test_auxiliary_unique_candidates_survive_the_final_cap(self) -> None:
        pool = [c.parent_asin for c in retrieve(self._agent.connection, self._ctx(), 20)]
        self.assertEqual(len(pool), 20)
        self.assertIn("B0CATONLY", pool, "category-only candidate starved out of the pool")
        self.assertIn("B0ATTRONLY", pool, "attribute-only candidate starved out of the pool")

    def test_route_provenance_is_exact_for_auxiliary_uniques(self) -> None:
        pool = retrieve(self._agent.connection, self._ctx(), 20)
        by_asin = {c.parent_asin: c for c in pool}
        self.assertEqual(set(by_asin["B0CATONLY"].route_sources), {"category"})
        self.assertEqual(set(by_asin["B0ATTRONLY"].route_sources), {"attribute"})

    def test_pool_is_deduplicated_and_capped(self) -> None:
        pool = [c.parent_asin for c in retrieve(self._agent.connection, self._ctx(), 20)]
        self.assertEqual(len(pool), len(set(pool)))
        self.assertLessEqual(len(pool), 20)


_OVERRIDE_ROWS = [
    {"parent_asin": "B0LEATHER", "title": "leather pouch", "categories": ["Clothing", "Pouches"],
     "features": ["leather"], "details": {}, "store": "Ex", "description": []},
    {"parent_asin": "B0DENIM", "title": "denim pouch", "categories": ["Clothing", "Pouches"],
     "features": ["denim"], "details": {}, "store": "Ex", "description": []},
    {"parent_asin": "B0BLACKDENIM", "title": "black denim jacket", "categories": ["Clothing", "Jackets"],
     "features": ["denim"], "details": {}, "store": "Ex", "description": []},
    {"parent_asin": "B0BLACKLEATHER", "title": "black leather jacket", "categories": ["Clothing", "Jackets"],
     "features": ["leather"], "details": {}, "store": "Ex", "description": []},
]


class OverrideQueryTransitionTest(unittest.TestCase):
    """B item 10 / D item 1: after 'black leather jacket' -> 'actually denim'
    retrieval must use only the new active state."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        path = Path(cls._tmp.name) / "catalog.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in _OVERRIDE_ROWS), encoding="utf-8"
        )
        cls._agent = Agent(path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _after_override(self) -> Context:
        state = SessionState(session_id="o")
        update_state(state, "black leather jacket", 1)
        update_state(state, "actually denim", 2)
        return Context(session_id="o", turn=2, user_message="actually denim", state=state)

    def test_active_state_after_override(self) -> None:
        ctx = self._after_override()
        self.assertEqual(ctx.state.slots["color"]["values"], ["black"])
        self.assertEqual(ctx.state.slots["category"]["values"], ["jacket"])
        self.assertEqual(ctx.state.slots["material"]["values"], ["denim"])

    def test_attribute_route_drops_the_superseded_material(self) -> None:
        ctx = self._after_override()
        found = {a for a, _ in attribute_route(self._agent.connection, ctx, 50)}
        self.assertIn("B0DENIM", found, "denim is an active constraint")
        self.assertNotIn("B0LEATHER", found, "leather must not drive retrieval after override")

    def test_category_route_still_uses_the_preserved_category(self) -> None:
        ctx = self._after_override()
        found = {a for a, _ in category_route(self._agent.connection, ctx, 50)}
        self.assertEqual(found, {"B0BLACKDENIM", "B0BLACKLEATHER"})

    def test_pool_contains_the_new_intent(self) -> None:
        ctx = self._after_override()
        pool = [c.parent_asin for c in retrieve(self._agent.connection, ctx, 50)]
        self.assertIn("B0BLACKDENIM", pool)
        self.assertIn("B0DENIM", pool)
        self.assertNotIn("B0LEATHER", pool, "leather-only product must not be retrieved")


if __name__ == "__main__":
    unittest.main()
