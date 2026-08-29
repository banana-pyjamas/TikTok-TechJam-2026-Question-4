"""CP 4.1 - 4.4 — delta validation.

``validate_delta`` guards the state manager against any malformed delta
(unknown slot, bad op, bad confidence). ``safe_extract_delta`` and the
snapshot/rollback in ``update_state`` guarantee a failing parser leaves the
previous valid state intact.
"""

from __future__ import annotations

import copy
import math
import unittest
from unittest import mock

from starter.contracts import SessionState
from starter.state import (
    _clean_confidence,
    apply_delta,
    safe_extract_delta,
    update_state,
    validate_delta,
)


class CP41SchemaValidation(unittest.TestCase):
    def test_unknown_slot_is_dropped(self) -> None:
        delta = {
            "material": {"values": ["denim"], "cardinality": "multi"},
            "vibe": {"values": ["cool"], "cardinality": "single"},
            "": {"values": ["x"]},
        }
        clean = validate_delta(delta)
        self.assertIn("material", clean)
        self.assertNotIn("vibe", clean)
        self.assertNotIn("", clean)

    def test_non_dict_delta_and_entries_are_dropped(self) -> None:
        self.assertEqual(validate_delta(None), {})
        self.assertEqual(validate_delta("garbage"), {})
        self.assertEqual(validate_delta(["material"]), {})
        self.assertEqual(validate_delta({"material": "denim"}), {})
        self.assertEqual(validate_delta({"material": {}}), {})

    def test_cardinality_is_forced_to_the_known_value(self) -> None:
        clean = validate_delta({"color": {"values": ["black"], "cardinality": "single"}})
        self.assertEqual(clean["color"]["cardinality"], "multi")

    def test_values_and_remove_must_be_lists_of_str(self) -> None:
        clean = validate_delta({
            "color": {"values": ["black", 7, None], "remove": "white"},
        })
        self.assertEqual(clean["color"]["values"], ["black"])
        self.assertNotIn("remove", clean["color"])

    def test_unknown_cue_is_stripped_entry_kept(self) -> None:
        clean = validate_delta({"material": {"values": ["denim"], "cue": "DESTROY"}})
        self.assertIn("material", clean)
        self.assertNotIn("cue", clean["material"])

    def test_real_extractor_output_passes_through_unchanged(self) -> None:
        raw = safe_extract_delta("black leather jacket size 10 under $100")
        self.assertEqual(validate_delta(raw), raw)


class CP42InvalidOperation(unittest.TestCase):
    def test_invalid_op_drops_the_entry(self) -> None:
        clean = validate_delta({
            "material": {"values": ["denim"], "op": "DESTROY"},
            "color": {"values": ["black"], "op": "REPLACE"},
        })
        self.assertNotIn("material", clean)
        self.assertIn("color", clean)

    def test_apply_delta_ignores_an_invalid_op_directly(self) -> None:
        state = SessionState(session_id="s")
        state.slots["material"] = {"values": ["leather"], "cardinality": "multi"}
        before = copy.deepcopy(state)
        apply_delta(state, {"material": {"values": ["denim"], "op": "DESTROY",
                                         "cardinality": "multi"}}, 1)
        self.assertEqual(state.slots, before.slots)
        self.assertEqual(state.provenance, [])

    def test_apply_delta_ignores_an_unknown_slot_directly(self) -> None:
        state = SessionState(session_id="s")
        apply_delta(state, {"vibe": {"values": ["cool"], "cardinality": "single"}}, 1)
        self.assertEqual(state.slots, {})

    def test_valid_ops_are_kept(self) -> None:
        for op in ("SET", "REPLACE", "ADD", "REMOVE"):
            clean = validate_delta({"color": {"values": ["black"], "op": op}})
            self.assertIn("color", clean, op)


class CP43ConfidenceBounds(unittest.TestCase):
    def test_clean_confidence_cases(self) -> None:
        self.assertEqual(_clean_confidence(None), 1.0)
        self.assertEqual(_clean_confidence(0.7), 0.7)
        self.assertEqual(_clean_confidence(1.5), 1.0)
        self.assertEqual(_clean_confidence(-0.3), None)   # clamps to 0 -> drop
        self.assertEqual(_clean_confidence(0.0), None)
        self.assertEqual(_clean_confidence(float("nan")), None)
        self.assertEqual(_clean_confidence("high"), None)
        self.assertEqual(_clean_confidence(math.inf), 1.0)

    def test_out_of_range_high_is_clamped_entry_kept(self) -> None:
        clean = validate_delta({"color": {"values": ["black"], "confidence": 2.0}})
        self.assertIn("color", clean)
        self.assertNotIn("confidence", clean["color"])  # clamped to 1.0 -> not stored

    def test_low_or_nan_confidence_drops_the_entry(self) -> None:
        for bad in (-1, 0, float("nan"), "x"):
            clean = validate_delta({"color": {"values": ["black"], "confidence": bad}})
            self.assertEqual(clean, {}, bad)

    def test_mid_confidence_is_recorded_on_the_entry(self) -> None:
        clean = validate_delta({"color": {"values": ["black"], "confidence": 0.6}})
        self.assertEqual(clean["color"]["confidence"], 0.6)

    def test_absent_confidence_is_trusted(self) -> None:
        clean = validate_delta({"color": {"values": ["black"]}})
        self.assertIn("color", clean)
        self.assertNotIn("confidence", clean["color"])


class CP44ParserFailure(unittest.TestCase):
    def _prior(self) -> SessionState:
        state = SessionState(session_id="s")
        update_state(state, "black leather jacket for my dad", 1)
        return state

    def test_safe_extract_delta_swallows_exceptions(self) -> None:
        with mock.patch("starter.state.extract_delta", side_effect=RuntimeError("boom")):
            self.assertEqual(safe_extract_delta("anything"), {})

    def test_state_survives_an_extractor_failure(self) -> None:
        state = self._prior()
        snapshot = copy.deepcopy(state)
        with mock.patch("starter.state.extract_delta", side_effect=RuntimeError("boom")):
            update_state(state, "actually denim", 2)
        self.assertEqual(state.slots, snapshot.slots)
        self.assertEqual(state.evidence, snapshot.evidence)
        self.assertEqual(state.provenance, snapshot.provenance)

    def test_state_survives_a_downstream_failure(self) -> None:
        state = self._prior()
        snapshot = copy.deepcopy(state)
        with mock.patch("starter.state.apply_delta", side_effect=RuntimeError("boom")):
            update_state(state, "actually denim", 2)
        self.assertEqual(state.slots, snapshot.slots)
        self.assertEqual(state.provenance, snapshot.provenance)

    def test_a_noop_turn_is_not_a_failure(self) -> None:
        state = self._prior()
        turns_before = state.turn
        update_state(state, "hmm let me think", 2)  # nothing extractable
        self.assertEqual(state.turn, 2)
        self.assertGreater(state.turn, turns_before)
        self.assertEqual(state.slots["material"]["values"], ["leather"])

    def test_hostile_and_malformed_inputs_never_corrupt_state(self) -> None:
        state = self._prior()
        material_before = list(state.slots["material"]["values"])
        for turn, message in enumerate(["", None, 123, "$" * 50, "🧥" * 20], start=2):
            update_state(state, message, turn)  # type: ignore[arg-type]
        self.assertEqual(state.slots["material"]["values"], material_before)


if __name__ == "__main__":
    unittest.main()
