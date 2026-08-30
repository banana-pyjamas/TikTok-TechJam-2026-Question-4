"""CP 9.1 - 9.6 — adaptive strategy.

Strategy decides HOW to run a turn (which routes), never WHICH product wins.
"""

from __future__ import annotations

import copy
import unittest

from starter.contracts import Context, SessionState, Strategy
from starter.state import update_state
from starter.strategy import (
    BROWSING,
    BUYING,
    SPECIFIC_SLOTS,
    build_strategy,
    classify_mode,
)


def _ctx(*turns: str) -> Context:
    """Replay turns through the real state manager, then wrap the last one."""
    state = SessionState(session_id="s")
    message = ""
    for index, message in enumerate(turns, start=1):
        update_state(state, message, index)
    return Context(session_id="s", turn=len(turns), user_message=message, state=state)


class CP91BuyingClassification(unittest.TestCase):
    def test_roadmap_example(self) -> None:
        self.assertEqual(classify_mode(_ctx("black Nike size 9")), BUYING)

    def test_any_single_concrete_slot_is_enough(self) -> None:
        for message in ("a leather jacket", "something in black",
                        "Adidas please", "size 10", "under $100"):
            self.assertEqual(classify_mode(_ctx(message)), BUYING, message)

    def test_requirement_language_without_a_slot(self) -> None:
        # The constraint value is outside our vocabulary, but the phrasing
        # still says "this is a requirement".
        self.assertEqual(
            classify_mode(_ctx("I need a necklace. A key requirement is: alloy.")),
            BUYING,
        )

    def test_concrete_volunteered_detail_counts(self) -> None:
        self.assertEqual(classify_mode(_ctx("A belt. Buckle closure")), BUYING)

    def test_specifics_beat_browsing_language(self) -> None:
        # Naming something checkable wins even while saying "exploring".
        self.assertEqual(
            classify_mode(_ctx("still exploring, but it must be leather")), BUYING)


class CP92BrowsingClassification(unittest.TestCase):
    def test_roadmap_example(self) -> None:
        self.assertEqual(classify_mode(_ctx("comfortable shoes for traveling")),
                         BROWSING)

    def test_explicit_exploring_language(self) -> None:
        self.assertEqual(
            classify_mode(_ctx("I'm looking for jackets, but I'm still exploring")),
            BROWSING)

    def test_category_alone_is_browsing(self) -> None:
        self.assertEqual(classify_mode(_ctx("I'm looking for shoes")), BROWSING)

    def test_soft_qualities_and_occasions_are_not_specifics(self) -> None:
        for message in ("something comfortable for the gym",
                        "a nice gift for the weekend",
                        "casual everyday shoes"):
            self.assertEqual(classify_mode(_ctx(message)), BROWSING, message)

    def test_empty_message_is_browsing_not_a_crash(self) -> None:
        self.assertEqual(classify_mode(_ctx("")), BROWSING)


class CP93RecomputeEveryTurn(unittest.TestCase):
    def test_strategy_is_derived_fresh_from_current_state(self) -> None:
        ctx = _ctx("I'm looking for shoes")
        self.assertEqual(build_strategy(ctx).mode, BROWSING)
        # Same context object, now with a specific slot added.
        update_state(ctx.state, "in black", 2)
        ctx.turn = 2
        self.assertEqual(build_strategy(ctx).mode, BUYING)

    def test_params_report_the_evidence_behind_the_decision(self) -> None:
        strategy = build_strategy(_ctx("black leather jacket"))
        self.assertGreaterEqual(strategy.params["specific_slots"], 2)
        self.assertIn("turn", strategy.params)

    def test_build_strategy_is_pure(self) -> None:
        ctx = _ctx("black leather jacket")
        before = copy.deepcopy(ctx.state)
        build_strategy(ctx)
        build_strategy(ctx)
        self.assertEqual(ctx.state, before)


class CP94BrowsingToBuyingTransition(unittest.TestCase):
    def test_roadmap_transition(self) -> None:
        state = SessionState(session_id="t")
        update_state(state, "shoes for traveling", 1)
        first = Context(session_id="t", turn=1,
                        user_message="shoes for traveling", state=state)
        self.assertEqual(classify_mode(first), BROWSING)

        update_state(state, "black Adidas size 9 under $100", 2)
        second = Context(session_id="t", turn=2,
                         user_message="black Adidas size 9 under $100", state=state)
        self.assertEqual(classify_mode(second), BUYING)

    def test_routes_change_with_the_mode(self) -> None:
        browsing = build_strategy(_ctx("shoes for traveling"))
        buying = build_strategy(_ctx("shoes for traveling", "black size 9"))
        self.assertNotEqual(browsing.routes, buying.routes)
        self.assertIn("bm25", browsing.routes)
        self.assertIn("bm25", buying.routes)


class CP95OverrideReOrchestration(unittest.TestCase):
    def test_strategy_rebuilds_from_the_new_state_after_an_override(self) -> None:
        ctx = _ctx("leather jacket", "actually denim")
        strategy = build_strategy(ctx)
        self.assertEqual(strategy.mode, BUYING)
        # Derived from the CURRENT state: denim is active, leather is gone.
        self.assertEqual(ctx.state.slots["material"]["values"], ["denim"])

    def test_removing_the_last_specific_returns_to_browsing(self) -> None:
        # Strategy follows state rather than latching on the earlier decision.
        ctx = _ctx("leather boots")
        self.assertEqual(classify_mode(ctx), BUYING)
        update_state(ctx.state, "not leather", 2)
        ctx.user_message = "not leather"
        self.assertNotIn("material", ctx.state.slots)
        self.assertEqual(classify_mode(ctx), BROWSING)


class CP96StrategyDoesNotRankProducts(unittest.TestCase):
    def test_strategy_carries_no_product_identity(self) -> None:
        strategy = build_strategy(_ctx("black leather jacket size 10"))
        blob = repr(strategy)
        self.assertNotIn("parent_asin", blob)
        self.assertEqual(
            set(vars(strategy)), {"mode", "routes", "route_weights", "params"})

    def test_ranking_does_not_consume_strategy(self) -> None:
        import inspect

        from starter import ranking

        source = inspect.getsource(ranking)
        self.assertNotIn("strategy", source.lower(),
                         "ranking must not read the strategy (CP 9.6)")

    def test_strategy_only_names_known_routes(self) -> None:
        from starter.retrieval import ROUTES

        for message in ("black leather jacket", "just browsing", ""):
            for route in build_strategy(_ctx(message)).routes:
                self.assertIn(route, ROUTES, message)

    def test_returns_the_frozen_contract_type(self) -> None:
        self.assertIsInstance(build_strategy(_ctx("shoes")), Strategy)


class SpecificSlotVocabularyTest(unittest.TestCase):
    def test_category_is_not_a_specific_slot(self) -> None:
        # Naming a category is how BOTH modes open, so it must not decide.
        self.assertNotIn("category", SPECIFIC_SLOTS)


if __name__ == "__main__":
    unittest.main()
