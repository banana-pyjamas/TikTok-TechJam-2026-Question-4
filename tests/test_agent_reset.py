"""CP 1.1 — Agent.reset() session initialization.

Acceptance (roadmap):
* new session               -> fresh authoritative state exists
* same session reset         -> starts clean, nothing carried over
* separate sessions          -> independent state objects
* no state leakage           -> mutation of one session's state (or the
                                profile) cannot reach another session or the
                                caller's profile dict

Tests build the Agent against a tiny temporary catalog to avoid the ~20s
real FTS index build. reset() itself does not touch the index.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.agent import Agent
from starter.contracts import SessionState


_CATALOG_ROWS = [
    {
        "parent_asin": "B0PA",
        "title": "Blue running shoe",
        "categories": ["Clothing", "Shoes"],
        "features": ["breathable mesh"],
        "details": {"department": "womens"},
        "store": "Example",
        "description": ["walking shoe"],
    },
    {
        "parent_asin": "B0PB",
        "title": "Black leather boot",
        "categories": ["Clothing", "Boots"],
        "features": ["full grain leather"],
        "details": {"department": "mens"},
        "store": "Example",
        "description": ["winter boot"],
    },
]


def _profile(**overrides: object) -> dict:
    base = {
        "purchase_frequency": "3-4 prior purchases",
        "average_prior_rating": 4.5,
        "rating_style": "usually positive",
        "preference_tags": ["fit", "comfort"],
        "summary": "Prior purchases emphasize fit and comfort.",
    }
    base.update(overrides)
    return base


class AgentResetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        catalog_path = Path(cls._tmp.name) / "catalog.jsonl"
        catalog_path.write_text(
            "".join(json.dumps(row) + "\n" for row in _CATALOG_ROWS), encoding="utf-8"
        )
        cls._catalog_path = catalog_path

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _agent(self) -> Agent:
        return Agent(self._catalog_path)

    # --- new session -----------------------------------------------------

    def test_reset_creates_fresh_session_state(self) -> None:
        agent = self._agent()
        agent.reset("s1", _profile())
        state = agent._states["s1"]
        self.assertIsInstance(state, SessionState)
        self.assertEqual(state.session_id, "s1")
        self.assertEqual(state.turn, 0)
        self.assertEqual(state.slots, {})
        self.assertEqual(state.evidence, [])
        self.assertEqual(state.provenance, [])
        self.assertEqual(state.user_profile["preference_tags"], ["fit", "comfort"])

    def test_reset_tolerates_none_profile(self) -> None:
        agent = self._agent()
        agent.reset("s1", None)  # type: ignore[arg-type]
        self.assertEqual(agent._states["s1"].user_profile, {})

    # --- same session reset -------------------------------------------------

    def test_reset_same_session_starts_clean(self) -> None:
        agent = self._agent()
        agent.reset("s1", _profile())
        state = agent._states["s1"]
        state.turn = 4
        state.slots["category"] = "boot"
        state.evidence.append("gift for my dad")
        state.provenance.append({"turn": 1, "slot": "category"})

        agent.reset("s1", _profile(summary="new"))
        fresh = agent._states["s1"]
        self.assertIsNot(fresh, state)
        self.assertEqual(fresh.turn, 0)
        self.assertEqual(fresh.slots, {})
        self.assertEqual(fresh.evidence, [])
        self.assertEqual(fresh.provenance, [])
        self.assertEqual(fresh.user_profile["summary"], "new")

    # --- separate sessions ------------------------------------------------

    def test_separate_sessions_have_independent_state(self) -> None:
        agent = self._agent()
        agent.reset("a", _profile())
        agent.reset("b", _profile())
        self.assertIsNot(agent._states["a"], agent._states["b"])

        agent._states["a"].slots["color"] = "black"
        agent._states["a"].evidence.append("for hiking")
        self.assertEqual(agent._states["b"].slots, {})
        self.assertEqual(agent._states["b"].evidence, [])

    def test_reset_does_not_disturb_other_existing_sessions(self) -> None:
        agent = self._agent()
        agent.reset("a", _profile())
        agent._states["a"].slots["category"] = "shoe"
        agent.reset("b", _profile())
        self.assertEqual(agent._states["a"].slots, {"category": "shoe"})

    # --- no state leakage -----------------------------------------------

    def test_profile_is_deep_copied_no_leak_into_caller(self) -> None:
        agent = self._agent()
        caller_profile = _profile()
        agent.reset("s1", caller_profile)
        state = agent._states["s1"]

        state.user_profile["summary"] = "mutated"
        state.user_profile["preference_tags"].append("style")

        self.assertEqual(caller_profile["summary"], _profile()["summary"])
        self.assertEqual(caller_profile["preference_tags"], ["fit", "comfort"])

    def test_profile_not_shared_between_sessions_reset_from_same_dict(self) -> None:
        agent = self._agent()
        shared = _profile()
        agent.reset("a", shared)
        agent.reset("b", shared)

        agent._states["a"].user_profile["preference_tags"].append("style")
        self.assertEqual(
            agent._states["b"].user_profile["preference_tags"], ["fit", "comfort"]
        )
        self.assertEqual(shared["preference_tags"], ["fit", "comfort"])

    def test_two_agents_do_not_share_session_registries(self) -> None:
        a1 = self._agent()
        a2 = self._agent()
        a1.reset("s1", _profile())
        self.assertNotIn("s1", a2._states)

    # --- respond guard regression --------------------------------------

    def test_respond_before_reset_still_raises(self) -> None:
        agent = self._agent()
        with self.assertRaises(RuntimeError):
            agent.respond("never-reset", "hello", 1, 10)

    def test_respond_after_reset_returns_valid_payload(self) -> None:
        agent = self._agent()
        agent.reset("s1", _profile())
        response = agent.respond("s1", "black leather boot", 1, 10)
        self.assertIsInstance(response, dict)
        self.assertIsInstance(response["message"], str)
        self.assertIn(response["ask_attribute"], (None, "category", "material",
                                                  "color", "size", "style", "brand",
                                                  "budget", "feature", "use_case", "other"))
        self.assertIsInstance(response["recommendations"], list)
        self.assertLessEqual(len(response["recommendations"]), 10)

    def test_respond_isolated_between_sessions_after_reset(self) -> None:
        agent = self._agent()
        agent.reset("a", _profile())
        agent.reset("b", _profile())
        agent.respond("a", "boot", 1, 10)
        # b was never touched by a's turn
        self.assertEqual(agent._states["b"].turn, 0)
        self.assertEqual(agent._states["b"].slots, {})


if __name__ == "__main__":
    unittest.main()
