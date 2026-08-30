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

    def test_requirement_language_is_sticky_across_silent_turns(self) -> None:
        # Stating a requirement is not un-done by going quiet. On this
        # harness the shopper goes quiet immediately once we stop asking.
        ctx = _ctx("I need a necklace. A key requirement is: alloy.")
        self.assertEqual(classify_mode(ctx), BUYING)
        ctx.user_message = "Those options are not quite right yet."
        ctx.turn = 2
        self.assertEqual(classify_mode(ctx), BUYING)

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

    def test_the_mode_changes_but_the_route_plan_is_uniform(self) -> None:
        # Corrected after the Phase 9 review. The first version varied routes
        # by mode and reported the difference as this checkpoint's gain; a
        # controlled test showed the damage was the ATTRIBUTE route,
        # uniformly, and mode-adaptive selection is worth +0.000298. Routes
        # are now fixed; the mode is kept for Phase 15, not for retrieval.
        browsing = build_strategy(_ctx("shoes for traveling"))
        buying = build_strategy(_ctx("shoes for traveling", "black size 9"))
        self.assertEqual(browsing.mode, BROWSING)
        self.assertEqual(buying.mode, BUYING)
        self.assertEqual(browsing.routes, buying.routes)
        self.assertEqual(buying.routes, ["bm25", "category"])
        self.assertNotIn("attribute", buying.routes)


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


class EvaluatorBoilerplateMustNotDecideModeTest(unittest.TestCase):
    """Phase 9 review: 90% of turns are the harness's "no new information"
    reply, and the word "options" in it collided with BROWSING_CUES -- so the
    harness's phrasing, not the shopper, was deciding the mode."""

    STUCK = ("Those options are not quite right yet. "
             "Ask me about one specific attribute.")

    def test_a_non_answer_does_not_flip_a_buying_session_to_browsing(self) -> None:
        state = SessionState(session_id="b")
        update_state(state, "I need a necklace. A key requirement is: alloy.", 1)
        self.assertEqual(
            classify_mode(Context(session_id="b", turn=1,
                                  user_message="alloy necklace", state=state)),
            BUYING)
        # Same state, next turn carries no new information.
        self.assertEqual(
            classify_mode(Context(session_id="b", turn=2,
                                  user_message=self.STUCK, state=state)),
            BUYING, "a non-answer must not overwrite what state already knows")

    def test_a_non_answer_leaves_a_browsing_session_browsing(self) -> None:
        state = SessionState(session_id="w")
        update_state(state, "I'm looking for shoes, but I'm still exploring", 1)
        self.assertEqual(
            classify_mode(Context(session_id="w", turn=2,
                                  user_message=self.STUCK, state=state)),
            BROWSING)

    def test_the_shoppers_own_browsing_words_still_count(self) -> None:
        # The cue mechanism itself is intact -- only harness text is ignored.
        self.assertEqual(classify_mode(_ctx("just show me some options")), BROWSING)


class MalformedStateDegradesRatherThanRaisesTest(unittest.TestCase):
    """Unreachable through the shipped path, closed anyway.

    Only ``state.update_state`` writes ``slots``, and it always writes a dict.
    But a mode is a workflow hint, and no hint is worth a raised exception on a
    live turn (principle E) -- classification must degrade to "nothing known".
    D carried this one open across three rounds.
    """

    def test_non_dict_slots_classify_browsing(self) -> None:
        for broken in (None, [], "slots", 7):
            state = SessionState(session_id="m")
            state.slots = broken
            context = Context(session_id="m", turn=1,
                              user_message="a leather jacket", state=state)
            self.assertEqual(classify_mode(context), BROWSING, repr(broken))

    def test_build_strategy_survives_it_too(self) -> None:
        state = SessionState(session_id="m")
        state.slots = None
        strategy = build_strategy(
            Context(session_id="m", turn=1, user_message="black", state=state))
        self.assertEqual(strategy.mode, BROWSING)
        self.assertEqual(strategy.params["specific_slots"], 0)


class ConcreteUnslottedDetailTest(unittest.TestCase):
    """The CP 9.2 case the extraction vocabulary has no slot for.

    The module docstring promises that free text naming something checkable --
    "a buckle closure", "a stainless steel band" -- makes a turn buying. C
    found the promise outliving its implementation: ``_has_concrete_evidence``
    was deleted while fixing the ``options`` collision, and the docstring was
    left claiming behaviour that no longer existed. These tests are the
    docstring, executable.
    """

    def test_a_concrete_detail_with_no_slot_is_buying(self) -> None:
        for message in ("It has a stainless steel band",
                        "Buckle closure",
                        "I'm looking for Watches. Stainless Steel Band"):
            self.assertEqual(classify_mode(_ctx(message)), BUYING, message)

    def test_the_real_override_openings_are_buying(self) -> None:
        # Verbatim shape of the intent_override turn-1 message: the harness
        # emits "I'm looking for {category}. {soft_preference}", and the
        # preference is a raw catalog feature string.
        self.assertEqual(
            classify_mode(_ctx("I'm looking for Watches. Buckle closure")),
            BUYING)

    def test_a_concrete_detail_survives_a_silent_turn(self) -> None:
        ctx = _ctx("I'm looking for Watches. Buckle closure")
        ctx.user_message = ("Those options are not quite right yet. "
                            "Ask me about one specific attribute.")
        ctx.turn = 2
        self.assertEqual(classify_mode(ctx), BUYING)

    def test_soft_and_filler_text_alone_is_still_browsing(self) -> None:
        # The gap the first version of this fallback had: generic filler read
        # as volunteered detail.
        for message in ("I'm looking for jackets, but I'm still exploring",
                        "just something nice for the weekend",
                        "I'd prefer something comfortable"):
            self.assertEqual(classify_mode(_ctx(message)), BROWSING, message)

    def test_a_plural_restating_the_category_is_not_evidence(self) -> None:
        # update_state stores the CANONICAL "jacket" but leaves the surface
        # "jackets" in the evidence residual. Naming a category is how both
        # modes open, so that leftover must not decide the mode.
        ctx = _ctx("I'm looking for jackets, but I'm still exploring")
        self.assertEqual(ctx.state.slots["category"]["values"], ["jacket"])
        self.assertIn("jackets", ctx.state.evidence[0]["normalized"])
        self.assertEqual(classify_mode(ctx), BROWSING)

    def test_the_soft_and_filler_vocabularies_are_live(self) -> None:
        # D-N4: SOFT_CUES and FILLER_CUES fed only _VAGUE_TOKENS, whose only
        # consumer had been deleted. If this fallback is ever removed again,
        # remove them with it rather than leaving 81 dead words behind.
        import inspect

        from starter import strategy

        source = inspect.getsource(strategy)
        self.assertIn("_VAGUE_TOKENS", source.split("_VAGUE_TOKENS =")[1],
                      "_VAGUE_TOKENS must have a consumer")

    def test_an_override_withdraws_the_buying_signal_with_the_detail(self) -> None:
        # CP 9.5: superseded evidence cannot keep a session buying.
        ctx = _ctx("I'm looking for jackets. Buckle closure")
        self.assertEqual(classify_mode(ctx), BUYING)
        for entry in ctx.state.evidence:
            entry["status"] = "superseded"
        self.assertEqual(classify_mode(ctx), BROWSING)


if __name__ == "__main__":
    unittest.main()
