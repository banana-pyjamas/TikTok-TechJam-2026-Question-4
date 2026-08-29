"""CP 2.1 - 2.8 — deterministic session-state manager.

Slot value schema (approved at CP 2.1):
    state.slots[name] = {"values": list[str], "cardinality": "single" | "multi"}
    budget also carries "bounds": {"min": float|None, "max": float|None}
"""

from __future__ import annotations

import unittest

from starter.contracts import SessionState
from starter.state import (
    SLOT_CARDINALITY,
    apply_delta,
    extract_delta,
    update_state,
)


def _run(*turns: str) -> SessionState:
    state = SessionState(session_id="s")
    for index, message in enumerate(turns, start=1):
        update_state(state, message, index)
    return state


def _values(state: SessionState, slot: str) -> list[str]:
    entry = state.slots.get(slot)
    return list(entry["values"]) if entry else []


class SchemaTest(unittest.TestCase):
    def test_every_stored_slot_matches_the_approved_schema(self) -> None:
        state = _run("black leather jacket size 10 nike under $100")
        self.assertTrue(state.slots)
        for name, entry in state.slots.items():
            self.assertIn(name, SLOT_CARDINALITY)
            self.assertEqual(set(entry) >= {"values", "cardinality"}, True)
            self.assertIsInstance(entry["values"], list)
            self.assertTrue(all(isinstance(v, str) for v in entry["values"]))
            self.assertEqual(entry["cardinality"], SLOT_CARDINALITY[name])
            if name != "budget":
                self.assertEqual(set(entry), {"values", "cardinality"})

    def test_single_valued_slots_hold_at_most_one_value(self) -> None:
        state = _run("jacket", "coat", "dress")  # category re-stated each turn
        self.assertEqual(len(_values(state, "category")), 1)


class CategoryStorageTest(unittest.TestCase):
    """CP 2.1."""

    def test_show_me_jackets_stores_category_jacket(self) -> None:
        state = _run("show me jackets")
        self.assertEqual(_values(state, "category"), ["jacket"])
        self.assertEqual(state.slots["category"]["cardinality"], "single")

    def test_no_category_keyword_stores_nothing(self) -> None:
        state = _run("something nice please")
        self.assertNotIn("category", state.slots)

    def test_category_write_is_recorded_in_provenance(self) -> None:
        state = _run("I need a dress")
        self.assertIn(
            {"turn": 1, "slot": "category", "op": "SET", "value": "dress",
             "source": "extractor"},
            state.provenance,
        )


class CategoryPersistenceTest(unittest.TestCase):
    """CP 2.2."""

    def test_category_survives_a_later_turn_that_does_not_mention_it(self) -> None:
        state = _run("I want a jacket", "make it black")
        self.assertEqual(_values(state, "category"), ["jacket"])
        self.assertEqual(_values(state, "color"), ["black"])

    def test_category_persists_across_many_unrelated_turns(self) -> None:
        state = _run("jeans", "size 32", "under $80", "for the weekend")
        self.assertEqual(_values(state, "category"), ["jeans"])

    def test_restating_same_category_does_not_duplicate_provenance(self) -> None:
        state = _run("a jacket", "the jacket again")
        sets = [p for p in state.provenance if p["slot"] == "category"]
        self.assertEqual(len(sets), 1)


class ColorStorageTest(unittest.TestCase):
    """CP 2.3."""

    def test_single_color(self) -> None:
        self.assertEqual(_values(_run("a black coat"), "color"), ["black"])

    def test_multiple_colors_accumulate_in_order(self) -> None:
        self.assertEqual(_values(_run("black and white sneakers"), "color"), ["black", "white"])

    def test_colors_add_across_turns_without_dropping_earlier_ones(self) -> None:
        state = _run("black jacket", "actually also navy")
        self.assertEqual(_values(state, "color"), ["black", "navy"])

    def test_grey_is_normalized_to_gray(self) -> None:
        self.assertEqual(_values(_run("grey hoodie"), "color"), ["gray"])

    def test_repeated_color_is_not_duplicated(self) -> None:
        state = _run("black shoes", "black again")
        self.assertEqual(_values(state, "color"), ["black"])


class MaterialStorageTest(unittest.TestCase):
    """CP 2.4."""

    def test_single_material(self) -> None:
        self.assertEqual(_values(_run("leather boots"), "material"), ["leather"])

    def test_multiple_materials(self) -> None:
        self.assertEqual(_values(_run("cotton and linen shirt"), "material"), ["cotton", "linen"])

    def test_material_is_multi_valued_and_accumulates(self) -> None:
        state = _run("wool sweater", "with some cashmere")
        self.assertEqual(_values(state, "material"), ["wool", "cashmere"])


class SizeStorageTest(unittest.TestCase):
    """CP 2.5."""

    def test_numeric_size(self) -> None:
        self.assertEqual(_values(_run("size 10 running shoes"), "size"), ["10"])

    def test_half_size(self) -> None:
        self.assertEqual(_values(_run("size 9.5"), "size"), ["9.5"])

    def test_size_word_after_keyword_is_normalized(self) -> None:
        self.assertEqual(_values(_run("size medium"), "size"), ["m"])
        self.assertEqual(_values(_run("size large"), "size"), ["l"])

    def test_standalone_xl_is_recognized(self) -> None:
        self.assertEqual(_values(_run("an XL hoodie"), "size"), ["xl"])

    def test_size_is_single_valued_last_write_wins(self) -> None:
        state = _run("size 8", "no wait size 9")
        self.assertEqual(_values(state, "size"), ["9"])


class BrandStorageTest(unittest.TestCase):
    """CP 2.6."""

    def test_known_brand(self) -> None:
        self.assertEqual(_values(_run("I want Nike shoes"), "brand"), ["nike"])

    def test_multi_word_brand(self) -> None:
        self.assertEqual(_values(_run("a New Balance sneaker"), "brand"), ["new balance"])

    def test_brand_keyword_pattern(self) -> None:
        self.assertEqual(_values(_run("brand Rothys please"), "brand"), ["rothys"])

    def test_levis_is_normalized(self) -> None:
        self.assertEqual(_values(_run("Levis jeans"), "brand"), ["levi"])

    def test_brand_is_single_valued(self) -> None:
        state = _run("Nike", "actually Adidas")
        self.assertEqual(_values(state, "brand"), ["adidas"])


class BudgetStorageTest(unittest.TestCase):
    """CP 2.7."""

    def test_under_budget(self) -> None:
        entry = _run("a coat under $100").slots["budget"]
        self.assertEqual(entry["values"], ["under $100"])
        self.assertEqual(entry["bounds"], {"min": None, "max": 100.0})

    def test_range_budget(self) -> None:
        entry = _run("shoes $50 to $120").slots["budget"]
        self.assertEqual(entry["bounds"], {"min": 50.0, "max": 120.0})

    def test_over_budget(self) -> None:
        entry = _run("something above $200").slots["budget"]
        self.assertEqual(entry["bounds"], {"min": 200.0, "max": None})

    def test_bare_dollar_amount(self) -> None:
        entry = _run("around $75").slots["budget"]
        self.assertEqual(entry["bounds"]["max"], 75.0)

    def test_no_amount_no_budget(self) -> None:
        self.assertNotIn("budget", _run("something cheap").slots)

    def test_budget_is_single_valued(self) -> None:
        state = _run("under $100", "make it under $150")
        self.assertEqual(state.slots["budget"]["bounds"]["max"], 150.0)


class FreeTextEvidenceTest(unittest.TestCase):
    """CP 2.8."""

    def test_gift_for_my_dad_is_kept(self) -> None:
        state = _run("gift for my dad")
        self.assertEqual(len(state.evidence), 1)
        self.assertEqual(state.evidence[0]["text"], "gift for my dad")

    def test_gift_evidence_persists_across_later_turns(self) -> None:
        state = _run("gift for my dad", "jacket", "black")
        texts = [e["text"] for e in state.evidence]
        self.assertIn("gift for my dad", texts)

    def test_pure_slot_message_is_not_stored_as_evidence(self) -> None:
        state = _run("jacket")
        self.assertEqual(state.evidence, [])

    def test_slot_plus_residual_message_is_stored(self) -> None:
        state = _run("warm winter jacket")
        self.assertEqual(len(state.evidence), 1)
        self.assertEqual(_values(state, "category"), ["jacket"])

    def test_evidence_is_deduplicated(self) -> None:
        state = _run("gift for my dad", "a gift for my dad")
        self.assertEqual(len(state.evidence), 1)

    def test_empty_and_whitespace_messages_are_ignored(self) -> None:
        state = _run("", "   ", "\t")
        self.assertEqual(state.evidence, [])


class SingleWriterAndIsolationTest(unittest.TestCase):
    def test_turn_counter_advances(self) -> None:
        state = _run("jacket", "black", "leather")
        self.assertEqual(state.turn, 3)

    def test_delta_extraction_does_not_mutate_state(self) -> None:
        state = SessionState(session_id="s")
        extract_delta("black leather jacket size 10")
        self.assertEqual(state.slots, {})
        self.assertEqual(state.provenance, [])

    def test_apply_delta_only_touches_slots_in_the_delta(self) -> None:
        state = _run("black jacket")
        before_category = dict(state.slots["category"])
        apply_delta(state, {"color": {"values": ["red"], "cardinality": "multi"}}, 2)
        self.assertEqual(state.slots["category"], before_category)
        self.assertEqual(_values(state, "color"), ["black", "red"])

    def test_non_string_message_does_not_crash(self) -> None:
        state = SessionState(session_id="s")
        update_state(state, None, 1)  # type: ignore[arg-type]
        update_state(state, 123, 2)  # type: ignore[arg-type]
        self.assertEqual(state.slots, {})

    def test_hostile_inputs_do_not_crash(self) -> None:
        for message in ('""""', "$$$", "size size size", "$", "under $",
                        "🧥👖", "-" * 40, "12345", "a", "  \n  "):
            update_state(SessionState(session_id="s"), message, 1)


if __name__ == "__main__":
    unittest.main()
