"""CP 3.1 - 3.7 — intent override in the deterministic state manager.

REPLACE / ADD / REMOVE are detected from the raw message ("actually" ->
REPLACE, "also" -> ADD, "not X" -> REMOVE) and applied slot-specifically.
"""

from __future__ import annotations

import unittest

from starter.contracts import SessionState
from starter.state import superseded_values, update_state


def _run(*turns: str) -> SessionState:
    state = SessionState(session_id="s")
    for index, message in enumerate(turns, start=1):
        update_state(state, message, index)
    return state


def _vals(state: SessionState, slot: str) -> list[str]:
    entry = state.slots.get(slot)
    return list(entry["values"]) if entry else []


def _ops(state: SessionState, slot: str) -> list[str]:
    return [p["op"] for p in state.provenance if p["slot"] == slot]


class CP31ReplaceOperation(unittest.TestCase):
    def test_material_replace_leather_to_denim(self) -> None:
        state = _run("leather jacket", "actually denim")
        self.assertEqual(_vals(state, "material"), ["denim"])
        self.assertIn("leather", superseded_values(state, "material"))
        self.assertIn("REPLACE", _ops(state, "material"))

    def test_replace_via_instead(self) -> None:
        self.assertEqual(_vals(_run("wool coat", "cotton instead"), "material"), ["cotton"])

    def test_replace_via_make_it(self) -> None:
        self.assertEqual(_vals(_run("a red shirt", "make it blue"), "color"), ["blue"])

    def test_replace_cue_with_no_existing_value_is_a_plain_first_write(self) -> None:
        state = _run("actually a leather jacket")
        self.assertEqual(_vals(state, "material"), ["leather"])
        # nothing was superseded -- multi-valued first write is an ADD
        self.assertEqual(_ops(state, "material"), ["ADD"])
        self.assertEqual(superseded_values(state, "material"), set())
        self.assertEqual(_ops(state, "category"), ["SET"])

    def test_single_valued_slot_replaces_without_a_cue(self) -> None:
        self.assertEqual(_vals(_run("size 8", "size 10"), "size"), ["10"])


class CP32PreserveUnrelatedEvidence(unittest.TestCase):
    """GOLDEN: black leather jacket -> actually denim."""

    def test_golden(self) -> None:
        state = _run("black leather jacket", "actually denim")
        self.assertEqual(_vals(state, "color"), ["black"], "color must survive")
        self.assertEqual(_vals(state, "category"), ["jacket"], "category must survive")
        self.assertEqual(_vals(state, "material"), ["denim"], "material replaced")
        self.assertNotIn("leather", _vals(state, "material"), "leather superseded")
        self.assertIn("leather", superseded_values(state, "material"))

    def test_override_does_not_add_spurious_provenance_to_other_slots(self) -> None:
        state = _run("black leather jacket", "actually denim")
        self.assertEqual(_ops(state, "color"), ["ADD"])
        self.assertEqual(_ops(state, "category"), ["SET"])


class CP33AddOperation(unittest.TestCase):
    def test_also_keeps_both_values(self) -> None:
        state = _run("black jacket", "also navy")
        self.assertEqual(_vals(state, "color"), ["black", "navy"])

    def test_plus_keeps_both(self) -> None:
        self.assertEqual(_vals(_run("wool sweater", "plus cashmere"), "material"),
                         ["wool", "cashmere"])

    def test_add_cue_beats_replace_cue_when_both_modify_one_value(self) -> None:
        state = _run("black jacket", "actually also navy")
        self.assertEqual(_vals(state, "color"), ["black", "navy"])

    def test_multi_valued_default_across_turns_is_still_add(self) -> None:
        self.assertEqual(_vals(_run("black shoes", "navy too? navy"), "color"),
                         ["black", "navy"])


class CP33MixedOperations(unittest.TestCase):
    """B CP 3.3 fix: cue is attributed per value, not message-globally."""

    def _mixed(self, message: str) -> SessionState:
        return _run("black leather jacket", message)

    def test_1_golden_mixed_operation(self) -> None:
        state = self._mixed("Actually denim, but also navy.")
        self.assertEqual(_vals(state, "material"), ["denim"])
        self.assertEqual(_vals(state, "color"), ["black", "navy"])
        self.assertIn("REPLACE", _ops(state, "material"))
        self.assertIn("ADD", _ops(state, "color"))
        self.assertIn("leather", superseded_values(state, "material"))

    def test_2_reverse_clause_order(self) -> None:
        state = self._mixed("Also navy, but actually denim.")
        self.assertEqual(_vals(state, "material"), ["denim"])
        self.assertEqual(_vals(state, "color"), ["black", "navy"])

    def test_3_two_replace_operations(self) -> None:
        state = self._mixed("Actually denim and make it blue.")
        self.assertEqual(_vals(state, "material"), ["denim"])
        self.assertEqual(_vals(state, "color"), ["blue"])

    def test_4_two_add_operations(self) -> None:
        state = _run("black wool sweater", "Also cashmere and also navy.")
        self.assertEqual(_vals(state, "material"), ["wool", "cashmere"])
        self.assertEqual(_vals(state, "color"), ["black", "navy"])

    def test_cue_is_not_borrowed_by_an_unrelated_slot(self) -> None:
        # "also" governs navy only; material must still REPLACE.
        state = self._mixed("Actually denim, also navy.")
        self.assertNotIn("leather", _vals(state, "material"))


class CP34RemoveOperation(unittest.TestCase):
    def test_not_leather_removes_it(self) -> None:
        state = _run("leather boots", "not leather")
        self.assertEqual(_vals(state, "material"), [])
        self.assertNotIn("material", state.slots)
        self.assertIn("REMOVE", _ops(state, "material"))

    def test_remove_one_of_several_keeps_the_rest(self) -> None:
        state = _run("black and white sneakers", "not white")
        self.assertEqual(_vals(state, "color"), ["black"])

    def test_without_phrasing(self) -> None:
        self.assertEqual(_vals(_run("wool and cashmere", "without wool"), "material"),
                         ["cashmere"])

    def test_remove_then_new_value_is_a_replace(self) -> None:
        state = _run("leather bag", "not leather, canvas")
        self.assertEqual(_vals(state, "material"), ["canvas"])
        self.assertIn("leather", superseded_values(state, "material"))


class CP35RepeatedOverride(unittest.TestCase):
    def test_leather_denim_cotton(self) -> None:
        state = _run("leather jacket", "actually denim", "no wait cotton")
        self.assertEqual(_vals(state, "material"), ["cotton"])
        superseded = superseded_values(state, "material")
        self.assertIn("leather", superseded)
        self.assertIn("denim", superseded)

    def test_each_override_is_recorded(self) -> None:
        state = _run("leather", "actually denim", "actually cotton")
        replaces = [p for p in state.provenance
                    if p["slot"] == "material" and p["op"] == "REPLACE"]
        self.assertEqual(len(replaces), 2)


class CP36NoStaleResurrection(unittest.TestCase):
    def test_superseded_value_does_not_return_on_an_unrelated_turn(self) -> None:
        state = _run("leather jacket", "actually denim", "black")
        self.assertEqual(_vals(state, "material"), ["denim"])
        self.assertEqual(_vals(state, "color"), ["black"])
        self.assertNotIn("leather", _vals(state, "material"))

    def test_superseded_value_stays_gone_over_many_turns(self) -> None:
        state = _run("leather jacket", "actually denim",
                     "size 10", "under $200", "for the weekend", "black")
        self.assertNotIn("leather", _vals(state, "material"))

    def _active_tokens(self, state: SessionState) -> set[str]:
        out: set[str] = set()
        for entry in state.evidence:
            if entry.get("status") == "active":
                out.update(entry["normalized"].split())
        return out

    def test_A_mixed_old_evidence_preservation(self) -> None:
        # B TEST A: superseding "leather" must not erase "winter hiking".
        state = _run("something in leather for winter hiking", "actually denim")
        self.assertEqual(_vals(state, "material"), ["denim"])
        self.assertIn("leather", superseded_values(state, "material"))
        active = self._active_tokens(state)
        self.assertNotIn("leather", active, "no active evidence may still assert leather")
        self.assertIn("winter", active)
        self.assertIn("hiking", active)

    def test_B_new_residual_information_on_override_turn(self) -> None:
        # B TEST B: override message's genuine residual is retained.
        state = _run("leather jacket", "actually denim for winter hiking")
        self.assertEqual(_vals(state, "material"), ["denim"])
        self.assertEqual(_vals(state, "category"), ["jacket"])
        active = self._active_tokens(state)
        self.assertIn("winter", active)
        self.assertIn("hiking", active)
        self.assertNotIn("leather", active)
        self.assertNotIn("denim", active, "slot value is not free-text evidence")
        self.assertNotIn("actually", active, "override plumbing is not evidence")

    def test_evidence_not_mentioning_the_dead_value_stays_active(self) -> None:
        state = SessionState(session_id="s")
        update_state(state, "a warm gift for my dad", 1)
        update_state(state, "leather jacket", 2)
        update_state(state, "actually denim", 3)
        gift = [e for e in state.evidence if "dad" in e["normalized"]]
        self.assertEqual([e["status"] for e in gift], ["active"])
        self.assertIn("warm", gift[0]["normalized"])

    def test_re_stating_a_value_explicitly_is_allowed(self) -> None:
        # B TEST E: not resurrection -- a fresh explicit request (principle I).
        state = _run("leather", "actually denim", "actually leather")
        self.assertEqual(_vals(state, "material"), ["leather"])


class CP37SessionIsolation(unittest.TestCase):
    def test_override_in_one_session_does_not_touch_another(self) -> None:
        a = SessionState(session_id="A")
        b = SessionState(session_id="B")
        update_state(a, "leather jacket", 1)
        update_state(b, "leather boots", 1)
        update_state(a, "actually denim", 2)

        self.assertEqual(_vals(a, "material"), ["denim"])
        self.assertEqual(_vals(b, "material"), ["leather"])
        self.assertEqual(superseded_values(b, "material"), set())
        self.assertEqual(_ops(b, "material"), ["ADD"])

    def test_F_evidence_supersession_is_per_session(self) -> None:
        import copy
        a = SessionState(session_id="A")
        b = SessionState(session_id="B")
        update_state(a, "something in leather for winter hiking", 1)
        update_state(b, "something in leather for winter hiking", 1)
        b_snapshot = copy.deepcopy(b)
        update_state(a, "actually denim", 2)
        self.assertEqual(_vals(a, "material"), ["denim"])
        self.assertEqual(b, b_snapshot, "session B is completely untouched")


if __name__ == "__main__":
    unittest.main()
