"""CP 0.2 / 0.3 / 0.4 — shared type contract tests.

Covers:
* CP 0.2 — the five shared types exist and are dataclasses.
* CP 0.3 — freeze guard: each type's field set is pinned. An unapproved
  change to ``starter/contracts.py`` fails here on purpose.
* CP 0.4 — every type is constructible with no arguments, empty instances
  are safe to use, and mutable defaults are not shared between instances.
"""

from __future__ import annotations

import copy
import unittest
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace

from starter.contracts import (
    Candidate,
    Context,
    RankingResult,
    SessionState,
    Strategy,
)


FROZEN_FIELDS = {
    SessionState: {
        "session_id",
        "user_profile",
        "turn",
        "slots",
        "evidence",
        "provenance",
    },
    Context: {"session_id", "turn", "user_message", "state"},
    Strategy: {"mode", "routes"},
    Candidate: {
        "parent_asin",
        "route_sources",
        "bm25_score",
        "category_score",
        "attribute_score",
        "metadata",
    },
    RankingResult: {"ranked", "diagnostics"},
}


class SharedTypesExistTest(unittest.TestCase):
    def test_all_five_types_are_dataclasses(self) -> None:
        for type_ in FROZEN_FIELDS:
            self.assertTrue(is_dataclass(type_), f"{type_.__name__} must be a dataclass")


class FreezeGuardTest(unittest.TestCase):
    """CP 0.3 — changing any field set requires an approved INTERFACE CHANGE
    REQUEST and an update to this mapping in the same commit."""

    def test_field_sets_match_frozen_contract(self) -> None:
        for type_, expected in FROZEN_FIELDS.items():
            actual = {f.name for f in fields(type_)}
            self.assertEqual(
                actual,
                expected,
                f"{type_.__name__} field set changed without updating the freeze guard",
            )

    def test_field_order_is_stable(self) -> None:
        # Positional construction is part of the contract; pin the order too.
        self.assertEqual(
            [f.name for f in fields(SessionState)],
            ["session_id", "user_profile", "turn", "slots", "evidence", "provenance"],
        )
        self.assertEqual(
            [f.name for f in fields(Context)],
            ["session_id", "turn", "user_message", "state"],
        )
        self.assertEqual([f.name for f in fields(Strategy)], ["mode", "routes"])
        self.assertEqual(
            [f.name for f in fields(Candidate)],
            [
                "parent_asin",
                "route_sources",
                "bm25_score",
                "category_score",
                "attribute_score",
                "metadata",
            ],
        )
        self.assertEqual(
            [f.name for f in fields(RankingResult)], ["ranked", "diagnostics"]
        )


class EmptyObjectsTest(unittest.TestCase):
    """CP 0.4 — empty/default instances construct and are safe to use."""

    def test_every_type_constructs_with_no_arguments(self) -> None:
        for type_ in FROZEN_FIELDS:
            instance = type_()  # must not raise
            self.assertIsInstance(instance, type_)

    def test_default_values(self) -> None:
        state = SessionState()
        self.assertEqual(state.session_id, "")
        self.assertEqual(state.user_profile, {})
        self.assertEqual(state.turn, 0)
        self.assertEqual(state.slots, {})
        self.assertEqual(state.evidence, [])
        self.assertEqual(state.provenance, [])

        ctx = Context()
        self.assertEqual(ctx.session_id, "")
        self.assertEqual(ctx.turn, 0)
        self.assertEqual(ctx.user_message, "")
        self.assertIsInstance(ctx.state, SessionState)

        strategy = Strategy()
        self.assertEqual(strategy.mode, "unknown")
        self.assertEqual(strategy.routes, [])

        candidate = Candidate()
        self.assertEqual(candidate.parent_asin, "")
        self.assertEqual(candidate.route_sources, [])
        self.assertEqual(candidate.bm25_score, 0.0)
        self.assertEqual(candidate.category_score, 0.0)
        self.assertEqual(candidate.attribute_score, 0.0)
        self.assertEqual(candidate.metadata, {})

        result = RankingResult()
        self.assertEqual(result.ranked, [])
        self.assertEqual(result.diagnostics, {})

    def test_mutable_defaults_are_not_shared_between_instances(self) -> None:
        a = SessionState()
        b = SessionState()
        a.slots["category"] = "jacket"
        a.evidence.append("gift for my dad")
        a.provenance.append({"turn": 1})
        a.user_profile["k"] = "v"
        self.assertEqual(b.slots, {})
        self.assertEqual(b.evidence, [])
        self.assertEqual(b.provenance, [])
        self.assertEqual(b.user_profile, {})

        c = Candidate()
        d = Candidate()
        c.route_sources.append("bm25")
        c.metadata["price"] = None
        self.assertEqual(d.route_sources, [])
        self.assertEqual(d.metadata, {})

        r1 = RankingResult()
        r2 = RankingResult()
        r1.ranked.append(Candidate(parent_asin="B1"))
        r1.diagnostics["B1"] = {"rank": 1}
        self.assertEqual(r2.ranked, [])
        self.assertEqual(r2.diagnostics, {})

    def test_context_nested_state_is_a_distinct_instance_per_context(self) -> None:
        c1 = Context()
        c2 = Context()
        self.assertIsNot(c1.state, c2.state)
        c1.state.slots["color"] = "black"
        self.assertEqual(c2.state.slots, {})

    def test_empty_instances_are_deepcopyable(self) -> None:
        # Session isolation later relies on being able to snapshot state.
        for type_ in FROZEN_FIELDS:
            clone = copy.deepcopy(type_())
            self.assertEqual(clone, type_())

    def test_dataclasses_are_not_frozen_instances(self) -> None:
        # The interface is frozen; instances are intentionally mutable.
        state = SessionState()
        try:
            state.turn = 2
        except FrozenInstanceError:  # pragma: no cover - guards against regression
            self.fail("SessionState instances must be mutable")
        self.assertEqual(state.turn, 2)
        self.assertEqual(replace(Candidate(), parent_asin="B9").parent_asin, "B9")


if __name__ == "__main__":
    unittest.main()
