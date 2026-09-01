"""Phase 2 verification pass (B follow-up review).

Focused checks for CP 2.5 / 2.6 / 2.7 / 2.8 in the exact shape B asked for:
an explicit constraint on turn 1, an unrelated turn 2, assert persistence.
Plus cross-session isolation and the two small fixes approved alongside
(D Finding 2 boilerplate filter, D Finding 3 brand-list trim, D Finding 1
evidence status hook).
"""

from __future__ import annotations

import unittest

from starter.contracts import SessionState
from starter.state import extract_delta, update_state


def _session(*turns: str) -> SessionState:
    state = SessionState(session_id="s")
    for index, message in enumerate(turns, start=1):
        update_state(state, message, index)
    return state


class CP25SizeVerification(unittest.TestCase):
    def test_explicit_size_is_stored_and_persists(self) -> None:
        state = _session("black running shoes size 10")
        self.assertIn("size", state.slots)
        self.assertEqual(state.slots["size"]["values"], ["10"])
        self.assertEqual(state.slots["size"]["cardinality"], "single")

        update_state(state, "I also want them lightweight", 2)  # unrelated turn
        self.assertEqual(state.slots["size"]["values"], ["10"])


class CP26BrandVerification(unittest.TestCase):
    def test_explicit_brand_is_stored_and_persists(self) -> None:
        state = _session("Adidas running shoes")
        self.assertIn("brand", state.slots)
        self.assertEqual(state.slots["brand"]["values"], ["adidas"])

        update_state(state, "something for the gym", 2)  # unrelated turn
        self.assertEqual(state.slots["brand"]["values"], ["adidas"])


class CP27BudgetVerification(unittest.TestCase):
    def test_explicit_budget_upper_bound_is_stored_and_persists(self) -> None:
        state = _session("running shoes under $150")
        self.assertIn("budget", state.slots)
        self.assertEqual(state.slots["budget"]["bounds"]["max"], 150.0)
        self.assertIsNone(state.slots["budget"]["bounds"]["min"])

        update_state(state, "in a neutral color", 2)  # unrelated turn
        self.assertEqual(state.slots["budget"]["bounds"]["max"], 150.0)


class CP28EvidenceVerification(unittest.TestCase):
    def test_free_text_evidence_persists_when_new_slot_is_added(self) -> None:
        state = SessionState(session_id="s")
        update_state(state, "gift for my dad", 1)
        self.assertEqual([e["text"] for e in state.evidence], ["gift for my dad"])

        update_state(state, "black jacket", 2)
        self.assertIn("gift for my dad", [e["text"] for e in state.evidence])
        self.assertEqual(state.slots["category"]["values"], ["jacket"])
        self.assertEqual(state.slots["color"]["values"], ["black"])

    def test_evidence_entries_carry_an_active_status_hook(self) -> None:
        state = _session("gift for my dad")
        self.assertEqual(state.evidence[0]["status"], "active")

    def test_simulator_non_answer_boilerplate_is_not_stored(self) -> None:
        # D Finding 2: these carry zero user intent.
        for stuck in (
            "Those options are not quite right yet. Ask me about one specific attribute.",
            "I don't have a preference for color; please use your judgment.",
            "I don't have an additional preference for material.",
        ):
            state = _session(stuck)
            self.assertEqual(state.evidence, [], stuck)

    def test_genuine_residual_after_a_slot_is_still_stored(self) -> None:
        state = _session("warm waterproof jacket for hiking")
        self.assertEqual(state.slots["category"]["values"], ["jacket"])
        self.assertEqual(len(state.evidence), 1)


class BrandTrimVerification(unittest.TestCase):
    """D Finding 3 — English-word brands removed to kill false positives."""

    def test_ambiguous_english_word_brands_no_longer_match(self) -> None:
        for phrase in ("mind the gap", "a coach ticket", "playing polo",
                       "the fossil record", "lee side of the boat"):
            self.assertNotIn("brand", extract_delta(phrase), phrase)

    def test_real_brands_still_match(self) -> None:
        self.assertEqual(extract_delta("Nike shoes")["brand"]["values"], ["nike"])
        self.assertEqual(extract_delta("New Balance 990")["brand"]["values"], ["new balance"])


class SessionIsolationVerification(unittest.TestCase):
    def test_slots_and_evidence_do_not_leak_between_sessions(self) -> None:
        a = SessionState(session_id="A")
        b = SessionState(session_id="B")
        update_state(a, "Adidas leather boots size 11 under $200, gift for my dad", 1)

        self.assertEqual(b.slots, {})
        self.assertEqual(b.evidence, [])
        self.assertEqual(b.provenance, [])
        self.assertEqual(b.turn, 0)

        update_state(b, "cotton socks", 1)
        self.assertNotIn("brand", b.slots)
        self.assertNotIn("size", b.slots)
        self.assertNotIn("budget", b.slots)
        self.assertEqual(b.evidence, [])
        # A is untouched by B's turn
        self.assertEqual(a.slots["brand"]["values"], ["adidas"])
        self.assertEqual([e["text"] for e in a.evidence],
                         ["Adidas leather boots size 11 under $200, gift for my dad"])


if __name__ == "__main__":
    unittest.main()
