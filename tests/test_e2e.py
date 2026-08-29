"""CP 1.7 - 1.9 — end-to-end smoke over the Agent.

* CP 1.7 — reset -> respond, one turn, valid payload.
* CP 1.8 — two consecutive turns do not crash.
* CP 1.9 — 5 sessions, up to 10 turns each, zero crashes, every payload valid,
  sessions stay isolated.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.agent import Agent

_ALLOWED_ASK = {
    None, "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
}

_CATALOG_ROWS = [
    {"parent_asin": "B0RUN", "title": "Blue running shoe",
     "categories": ["Clothing", "Shoes"], "features": ["breathable mesh"],
     "details": {"department": "womens"}, "store": "Ex", "description": ["walking shoe"]},
    {"parent_asin": "B0BOOT", "title": "Black leather boot",
     "categories": ["Clothing", "Boots"], "features": ["full grain leather", "waterproof"],
     "details": {"department": "mens"}, "store": "Ex", "description": ["winter boot"]},
    {"parent_asin": "B0SOCK", "title": "Wool hiking sock",
     "categories": ["Clothing", "Socks"], "features": ["merino wool"],
     "details": {"department": "unisex"}, "store": "Ex", "description": ["crew sock"]},
    {"parent_asin": "B0HAT", "title": "Red wool beanie hat",
     "categories": ["Clothing", "Hats"], "features": ["ribbed knit"],
     "details": {"department": "unisex"}, "store": "Ex", "description": ["warm winter hat"]},
]

_PROFILE = {
    "purchase_frequency": "3-4 prior purchases",
    "average_prior_rating": 4.5,
    "rating_style": "usually positive",
    "preference_tags": ["fit", "comfort"],
    "summary": "Prior purchases emphasize fit and comfort.",
}


class _E2EBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        path = Path(cls._tmp.name) / "catalog.jsonl"
        path.write_text(
            "".join(json.dumps(r) + "\n" for r in _CATALOG_ROWS), encoding="utf-8"
        )
        cls._agent = Agent(path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def assert_valid_payload(self, payload: object, top_k: int = 10) -> None:
        self.assertIsInstance(payload, dict)
        assert isinstance(payload, dict)
        self.assertIsInstance(payload["message"], str)
        self.assertIn(payload["ask_attribute"], _ALLOWED_ASK)
        recs = payload["recommendations"]
        self.assertIsInstance(recs, list)
        self.assertLessEqual(len(recs), top_k)
        seen: set[str] = set()
        for rec in recs:
            self.assertIsInstance(rec, dict)
            self.assertIsInstance(rec["parent_asin"], str)
            self.assertTrue(rec["parent_asin"])
            seen.add(rec["parent_asin"])
        self.assertEqual(len(seen), len(recs), "duplicate parent_asin in recommendations")


class OneTurnE2ETest(_E2EBase):
    """CP 1.7."""

    def test_reset_then_respond_returns_valid_payload(self) -> None:
        self._agent.reset("one", _PROFILE)
        self.assert_valid_payload(self._agent.respond("one", "black leather boot", 1, 10))

    def test_reset_then_respond_finds_target_in_tiny_catalog(self) -> None:
        self._agent.reset("one", _PROFILE)
        payload = self._agent.respond("one", "black leather waterproof boot", 1, 10)
        self.assertEqual(payload["recommendations"][0]["parent_asin"], "B0BOOT")

    def test_respond_caps_recommendations_at_10_for_oversized_top_k(self) -> None:
        self._agent.reset("one", _PROFILE)
        payload = self._agent.respond("one", "shoe boot sock hat wool leather", 1, 100)
        self.assertLessEqual(len(payload["recommendations"]), 10)


class TwoTurnE2ETest(_E2EBase):
    """CP 1.8."""

    def test_two_consecutive_turns_do_not_crash(self) -> None:
        self._agent.reset("two", _PROFILE)
        p1 = self._agent.respond("two", "I want a wool sock", 1, 10)
        p2 = self._agent.respond("two", "actually a boot", 2, 10)
        self.assert_valid_payload(p1)
        self.assert_valid_payload(p2)

    def test_second_turn_reflects_new_query(self) -> None:
        self._agent.reset("two", _PROFILE)
        self._agent.respond("two", "wool sock", 1, 10)
        p2 = self._agent.respond("two", "black leather boot", 2, 10)
        self.assertEqual(p2["recommendations"][0]["parent_asin"], "B0BOOT")

    def test_empty_then_real_turn(self) -> None:
        self._agent.reset("two", _PROFILE)
        self.assert_valid_payload(self._agent.respond("two", "", 1, 10))
        self.assert_valid_payload(self._agent.respond("two", "wool hat", 2, 10))


class FiveSessionSmokeTest(_E2EBase):
    """CP 1.9."""

    _MESSAGES = [
        "I'm looking for shoes",
        "",                       # empty
        "the to a of and",        # stopword-only
        "black leather boot",
        "wool hiking sock size 10",
        "something warm for winter",
        "red beanie hat",
        "waterproof",
        "comfortable running shoe",
        "boot sock hat shoe",
    ]

    def test_five_sessions_ten_turns_zero_crashes(self) -> None:
        for s in range(5):
            session_id = f"smoke_{s}"
            self._agent.reset(session_id, _PROFILE)
            for turn in range(1, 11):
                message = self._MESSAGES[(s + turn) % len(self._MESSAGES)]
                payload = self._agent.respond(session_id, message, turn, 10)
                self.assert_valid_payload(payload)

    def test_sessions_remain_isolated_across_the_smoke_run(self) -> None:
        for s in range(5):
            self._agent.reset(f"iso_{s}", _PROFILE)
        for turn in range(1, 11):
            self._agent.respond("iso_0", "black leather boot", turn, 10)
        # every other session's state is untouched
        for s in range(1, 5):
            state = self._agent._states[f"iso_{s}"]
            self.assertEqual(state.turn, 0)
            self.assertEqual(state.slots, {})
            self.assertEqual(state.evidence, [])


if __name__ == "__main__":
    unittest.main()
