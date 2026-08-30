"""Phase 5 integration / adversarial verification (D review).

Covers the agent-level behavior of multi-route retrieval: override
integration, the empty-result path, session isolation, response invariants,
a full 10-turn session, and the documented auxiliary-route failure policy.
"""

from __future__ import annotations

import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from starter import agent as agent_module
from starter.agent import Agent
from starter.contracts import Context, SessionState
from starter.retrieval import POOL_LIMIT, retrieve, run_routes
from starter.state import update_state

_ALLOWED_ASK = {
    None, "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
}

_CATALOG_ROWS = [
    {"parent_asin": "B0BLACKLEATHER", "title": "black leather jacket",
     "categories": ["Clothing", "Jackets"], "features": ["full grain leather"],
     "details": {"department": "mens"}, "store": "Ex", "description": ["biker"]},
    {"parent_asin": "B0BLACKDENIM", "title": "black denim jacket",
     "categories": ["Clothing", "Jackets"], "features": ["denim"],
     "details": {"department": "mens"}, "store": "Ex", "description": ["trucker"]},
    {"parent_asin": "B0WOOLSOCK", "title": "wool hiking sock",
     "categories": ["Clothing", "Socks"], "features": ["merino wool"],
     "details": {}, "store": "Ex", "description": ["cushioned"]},
    {"parent_asin": "B0SPARSE", "title": "plain boot",
     "categories": ["Clothing", "Boots"], "features": [], "details": {},
     "store": "", "description": []},
]

_PROFILE = {
    "purchase_frequency": "3-4 prior purchases",
    "average_prior_rating": 4.5,
    "rating_style": "usually positive",
    "preference_tags": ["fit", "comfort"],
    "summary": "Prior purchases emphasize fit and comfort.",
}


class _AgentFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        path = Path(cls._tmp.name) / "catalog.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in _CATALOG_ROWS), encoding="utf-8"
        )
        cls._catalog_path = path
        cls._agent = Agent(path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def assert_valid_payload(self, payload: object, top_k: int = 10) -> None:
        self.assertIsInstance(payload, dict)
        assert isinstance(payload, dict)
        self.assertIsInstance(payload["message"], str)
        self.assertTrue(payload["message"])
        self.assertIn(payload["ask_attribute"], _ALLOWED_ASK)
        recommendations = payload["recommendations"]
        self.assertIsInstance(recommendations, list)
        self.assertLessEqual(len(recommendations), top_k)
        seen: set[str] = set()
        valid = {row["parent_asin"] for row in _CATALOG_ROWS}
        for entry in recommendations:
            asin = entry["parent_asin"]
            self.assertIn(asin, valid, "recommendation must be a catalog ASIN")
            self.assertNotIn(asin, seen, "duplicate parent_asin")
            seen.add(asin)


class OverrideIntegrationTest(_AgentFixture):
    """D item 1 — after an override, retrieval uses the new active state only."""

    def test_respond_across_an_override_uses_new_state(self) -> None:
        self._agent.reset("ov", _PROFILE)
        self.assert_valid_payload(self._agent.respond("ov", "black leather jacket", 1, 10))
        self.assert_valid_payload(self._agent.respond("ov", "actually denim", 2, 10))

        state = self._agent._states["ov"]
        self.assertEqual(state.slots["color"]["values"], ["black"])
        self.assertEqual(state.slots["category"]["values"], ["jacket"])
        self.assertEqual(state.slots["material"]["values"], ["denim"])

    def test_superseded_material_is_absent_from_every_route_query(self) -> None:
        state = SessionState(session_id="ov")
        update_state(state, "black leather jacket", 1)
        update_state(state, "actually denim", 2)
        ctx = Context(session_id="ov", turn=2, user_message="actually denim", state=state)

        active = {value for slot in state.slots.values() for value in slot["values"]}
        self.assertIn("denim", active)
        self.assertNotIn("leather", active)

        pool = {c.parent_asin for c in retrieve(self._agent.connection, ctx, POOL_LIMIT)}
        self.assertIn("B0BLACKDENIM", pool)


class EmptyResultPathTest(_AgentFixture):
    """D item 2 — every route empty must not crash and must stay schema-valid."""

    def test_empty_message_and_empty_state_yields_no_candidates(self) -> None:
        state = SessionState(session_id="e")
        ctx = Context(session_id="e", turn=1, user_message="", state=state)
        per_route = run_routes(self._agent.connection, ctx, POOL_LIMIT)
        for name, rows in per_route.items():
            self.assertEqual(rows, [], name)
        self.assertEqual(retrieve(self._agent.connection, ctx, POOL_LIMIT), [])

    def test_respond_with_no_candidates_is_valid_and_empty(self) -> None:
        self._agent.reset("e", _PROFILE)
        payload = self._agent.respond("e", "", 1, 10)
        self.assert_valid_payload(payload)
        self.assertEqual(payload["recommendations"], [])

    def test_respond_with_stopword_only_message_is_valid(self) -> None:
        self._agent.reset("e", _PROFILE)
        payload = self._agent.respond("e", "the to a of and", 1, 10)
        self.assert_valid_payload(payload)
        self.assertEqual(payload["recommendations"], [])

    def test_unmatchable_message_is_valid(self) -> None:
        self._agent.reset("e", _PROFILE)
        self.assert_valid_payload(self._agent.respond("e", "zzzqqxx", 1, 10))


class SessionIsolationThroughRetrievalTest(_AgentFixture):
    """D item 3 — interleaved sessions must not leak."""

    def test_interleaved_sessions_keep_separate_state_and_results(self) -> None:
        self._agent.reset("A", _PROFILE)
        self._agent.reset("B", _PROFILE)

        self._agent.respond("A", "black leather jacket", 1, 10)
        self._agent.respond("B", "wool hiking sock", 1, 10)
        self._agent.respond("A", "actually denim", 2, 10)
        self._agent.respond("B", "size 10", 2, 10)

        a, b = self._agent._states["A"], self._agent._states["B"]
        self.assertEqual(a.slots["category"]["values"], ["jacket"])
        self.assertEqual(a.slots["material"]["values"], ["denim"])
        self.assertNotIn("size", a.slots)
        self.assertEqual(b.slots["category"]["values"], ["socks"])
        self.assertEqual(b.slots["material"]["values"], ["wool"])
        self.assertEqual(b.slots["size"]["values"], ["10"])
        self.assertNotIn("color", b.slots)

    def test_retrieval_never_mutates_session_state(self) -> None:
        state = SessionState(session_id="ro")
        update_state(state, "black leather jacket size 10 under $200", 1)
        snapshot = copy.deepcopy(state)
        ctx = Context(session_id="ro", turn=1, user_message="black leather jacket",
                      state=state)
        run_routes(self._agent.connection, ctx, POOL_LIMIT)
        retrieve(self._agent.connection, ctx, POOL_LIMIT)
        self.assertEqual(state, snapshot)

    def test_a_reset_mid_flight_does_not_disturb_the_other_session(self) -> None:
        self._agent.reset("A", _PROFILE)
        self._agent.reset("B", _PROFILE)
        self._agent.respond("A", "black leather jacket", 1, 10)
        self._agent.reset("B", _PROFILE)
        self.assertEqual(self._agent._states["A"].slots["category"]["values"], ["jacket"])
        self.assertEqual(self._agent._states["B"].slots, {})


class AuxiliaryRouteFailureTest(_AgentFixture):
    """D item 4 — DOCUMENTED LIMITATION.

    These exercise the multi-route path specifically, so they enable
    ``USE_MULTI_ROUTE`` explicitly rather than depending on the default
    (which Phase 7 measured OFF).

    ``category_route`` and ``attribute_route`` are core deterministic
    components (principle J lists category and attribute retrieval among the
    parts that must work offline), not optional enrichment. A raise inside
    one is a programming error, so retrieval deliberately does NOT wrap them
    in a broad ``except`` -- swallowing it would hide the bug and silently
    degrade recall. The exception propagates to ``Agent.respond``.

    These tests pin that decision so a future change to silent degradation is
    explicit rather than accidental. The evaluator already treats a raising
    turn as a miss for that turn and continues the session, so a route bug
    costs recall but cannot corrupt state (the state manager has already
    committed or rolled back before retrieval runs).
    """

    def setUp(self) -> None:
        self._multi_route = mock.patch.object(agent_module, "USE_MULTI_ROUTE", True)
        self._multi_route.start()
        self.addCleanup(self._multi_route.stop)

    @staticmethod
    def _broken(name: str):
        def raiser(*_args, **_kwargs):
            raise RuntimeError(f"{name} is broken")

        # ROUTES binds the function objects at import time, so the dict entry
        # is the live dispatch point.
        return mock.patch.dict("starter.retrieval.ROUTES", {name: raiser})

    def test_auxiliary_route_failure_propagates_rather_than_degrading(self) -> None:
        self._agent.reset("f", _PROFILE)
        for name in ("category", "attribute"):
            with self._broken(name):
                with self.assertRaises(RuntimeError, msg=name):
                    self._agent.respond("f", "black leather jacket", 1, 10)

    def test_state_is_not_corrupted_by_a_route_failure(self) -> None:
        self._agent.reset("f", _PROFILE)
        self._agent.respond("f", "black leather jacket", 1, 10)
        before = copy.deepcopy(self._agent._states["f"])
        with self._broken("attribute"):
            with self.assertRaises(RuntimeError):
                self._agent.respond("f", "actually denim", 2, 10)
        after = self._agent._states["f"]
        # The state manager runs to completion before retrieval, so the turn's
        # state update is committed; it is coherent, not half-written.
        self.assertEqual(after.slots["material"]["values"], ["denim"])
        self.assertEqual(after.slots["color"]["values"], before.slots["color"]["values"])
        self.assertEqual(after.slots["category"]["values"], ["jacket"])

    def test_the_session_recovers_on_the_next_turn(self) -> None:
        self._agent.reset("f", _PROFILE)
        with self._broken("category"):
            with self.assertRaises(RuntimeError):
                self._agent.respond("f", "black leather jacket", 1, 10)
        self.assert_valid_payload(self._agent.respond("f", "denim jacket", 2, 10))


class ResponseInvariantsTest(_AgentFixture):
    """D item 6."""

    _MESSAGES = [
        "I'm looking for a jacket", "black", "actually denim", "size 10",
        "under $200", "wool socks too", "", "the to a of", "🧥 emoji", "$$$",
    ]

    def test_invariants_hold_across_many_message_shapes(self) -> None:
        self._agent.reset("inv", _PROFILE)
        for turn, message in enumerate(self._MESSAGES, start=1):
            self.assert_valid_payload(self._agent.respond("inv", message, turn, 10))

    def test_oversized_top_k_is_still_capped_at_10(self) -> None:
        self._agent.reset("inv", _PROFILE)
        payload = self._agent.respond("inv", "black denim leather jacket sock boot", 1, 100)
        self.assertLessEqual(len(payload["recommendations"]), 10)

    def test_respond_before_reset_still_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            self._agent.respond("never-reset", "jacket", 1, 10)


class TenTurnSmokeTest(_AgentFixture):
    """D item 7 — a full 10-turn session with state changes and an override."""

    _SCRIPT = [
        "I'm looking for a jacket",
        "black",
        "leather",
        "actually denim",
        "size 10",
        "under $200",
        "also navy",
        "not navy",
        "",
        "something warm for winter",
    ]

    def test_ten_turns_no_crash_and_state_stays_coherent(self) -> None:
        self._agent.reset("ten", _PROFILE)
        for turn, message in enumerate(self._SCRIPT, start=1):
            self.assert_valid_payload(self._agent.respond("ten", message, turn, 10))

        state = self._agent._states["ten"]
        self.assertEqual(state.turn, 10)
        self.assertEqual(state.slots["category"]["values"], ["jacket"])
        self.assertEqual(state.slots["material"]["values"], ["denim"])
        self.assertEqual(state.slots["color"]["values"], ["black"])
        self.assertEqual(state.slots["size"]["values"], ["10"])
        for slot in state.slots.values():
            self.assertIsInstance(slot["values"], list)
            self.assertTrue(all(isinstance(v, str) and v for v in slot["values"]))


class AblationFlagDefaultsTest(unittest.TestCase):
    """The committed flag defaults carry a measured score, so pin them --
    a change should be deliberate, not accidental.

    Measured by ``python3 -m tools.phase7_ablation`` on the public set:
      state ON, multi-route OFF, ranking ON -> HR 0.1550 / TS 0.131194
      turning multi-route back ON           -> HR 0.1350 / TS 0.115512
    """

    def test_committed_defaults(self) -> None:
        from starter import ranking

        self.assertTrue(agent_module.USE_STATE)
        self.assertTrue(
            agent_module.USE_MULTI_ROUTE,
            "re-enabled in Phase 9: net-negative applied uniformly, net-positive "
            "when gated by the adaptive strategy",
        )
        self.assertTrue(agent_module.USE_CONSTRAINT_RANKING)
        self.assertTrue(
            agent_module.USE_ADAPTIVE_STRATEGY,
            "the union is only a win while the strategy gates it (+0.0037 TS); "
            "ungated it costs -0.0157",
        )
        self.assertFalse(
            ranking.USE_PROFILE,
            "profile prior measured net-negative in Phase 8 (-0.0033 TS)",
        )


class RetrievalCostTest(_AgentFixture):
    """D item 8 — FTS queries per turn and retrieval latency."""

    def _count_queries(self, multi_route: bool) -> tuple[int, int]:
        agent = Agent(self._catalog_path)
        agent.reset("cost", _PROFILE)
        with mock.patch.object(agent_module, "USE_MULTI_ROUTE", multi_route):
            agent.respond("cost", "black leather jacket", 1, 10)  # warm the state
            statements: list[str] = []
            agent.connection.set_trace_callback(statements.append)
            try:
                agent.respond("cost", "actually denim size 10", 2, 10)
            finally:
                agent.connection.set_trace_callback(None)
        selects = [s for s in statements if s.lstrip().upper().startswith("SELECT")]
        return (
            len([s for s in selects if "products MATCH" in s]),
            len([s for s in selects if "product_meta" in s]),
        )

    def test_query_count_per_turn_in_both_pool_configurations(self) -> None:
        multi_fts, multi_meta = self._count_queries(True)
        single_fts, single_meta = self._count_queries(False)
        print(f"\nqueries per turn: multi-route {multi_fts} FTS + {multi_meta} meta; "
              f"bm25-only {single_fts} FTS + {single_meta} meta")
        self.assertEqual(multi_fts, 3, "one FTS query per route, no N+1")
        self.assertEqual(single_fts, 1, "bm25-only pool issues a single FTS query")
        # The batched metadata lookup is one query either way -- no N+1.
        self.assertEqual(multi_meta, 1)
        self.assertEqual(single_meta, 1)

    def test_retrieval_latency_is_reported(self) -> None:
        state = SessionState(session_id="lat")
        update_state(state, "black leather jacket size 10", 1)
        ctx = Context(session_id="lat", turn=1, user_message="black leather jacket",
                      state=state)
        start = time.perf_counter()
        for _ in range(20):
            retrieve(self._agent.connection, ctx, POOL_LIMIT)
        per_turn_ms = (time.perf_counter() - start) / 20 * 1000
        print(f"\nretrieval per turn (4-row catalog): {per_turn_ms:.3f} ms")
        self.assertLess(per_turn_ms, 100.0)


if __name__ == "__main__":
    unittest.main()
