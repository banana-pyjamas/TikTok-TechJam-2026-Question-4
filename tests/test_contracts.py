"""CP 0.2 / 0.3 / 0.4 — shared type contract tests.

Covers:
* CP 0.2 — the five shared types exist, are dataclasses, and keep
  route/strategy knobs in generic containers (no hardcoded per-route field).
* CP 0.3 — freeze guard: each type's field name set, field order, AND field
  type string are pinned. An unapproved change to ``starter/contracts.py``
  fails here on purpose.
* CP 0.4 — every type is constructible with no arguments, empty instances
  are safe to use, mutable defaults are not shared between instances, and
  the frozen None rule (container fields normalize ``None`` -> empty; scalar
  fields pass ``None`` through untouched) holds.
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


def _norm(type_string: str) -> str:
    """Whitespace-insensitive form of a dataclass field's annotation string."""
    return " ".join(type_string.split())


# Frozen contract: {type: {field_name: annotation_string}} in declaration order.
FROZEN_FIELDS: dict[type, dict[str, str]] = {
    SessionState: {
        "session_id": "str",
        "user_profile": "dict[str, Any]",
        "turn": "int",
        "slots": "dict[str, Any]",
        "evidence": "list[Any]",
        "provenance": "list[dict[str, Any]]",
    },
    Context: {
        "session_id": "str",
        "turn": "int",
        "user_message": "str",
        "state": "SessionState",
        "derived": "dict[str, Any]",
    },
    Strategy: {
        "mode": "str",
        "routes": "list[str]",
        "route_weights": "dict[str, float]",
        "params": "dict[str, Any]",
    },
    Candidate: {
        "parent_asin": "str",
        "route_scores": "dict[str, float]",
        "metadata": "dict[str, Any]",
    },
    RankingResult: {
        "ranked": "list[Candidate]",
        "diagnostics": "dict[str, dict[str, Any]]",
    },
}

# Container fields governed by the CP 0.4 None rule (None -> empty container).
CONTAINER_FIELDS: dict[type, list[str]] = {
    SessionState: ["user_profile", "slots", "evidence", "provenance"],
    Context: ["state", "derived"],
    Strategy: ["routes", "route_weights", "params"],
    Candidate: ["route_scores", "metadata"],
    RankingResult: ["ranked", "diagnostics"],
}


class SharedTypesExistTest(unittest.TestCase):
    def test_all_five_types_are_dataclasses(self) -> None:
        for type_ in FROZEN_FIELDS:
            self.assertTrue(is_dataclass(type_), f"{type_.__name__} must be a dataclass")

    def test_no_hardcoded_per_route_score_fields(self) -> None:
        # CP 0.2 blocker fix: route scores must be a generic container so a
        # new route (dense, ...) does not force a contract mutation.
        candidate_field_names = {f.name for f in fields(Candidate)}
        for banned in ("bm25_score", "category_score", "attribute_score", "dense_score"):
            self.assertNotIn(banned, candidate_field_names)
        self.assertIn("route_scores", candidate_field_names)

    def test_strategy_has_generic_extension_points(self) -> None:
        strategy_field_names = {f.name for f in fields(Strategy)}
        self.assertIn("route_weights", strategy_field_names)
        self.assertIn("params", strategy_field_names)


class FreezeGuardTest(unittest.TestCase):
    """CP 0.3 — changing any field name, order, or type requires an approved
    INTERFACE CHANGE REQUEST and an update to this mapping in the same commit."""

    def test_field_name_sets_match_frozen_contract(self) -> None:
        for type_, expected in FROZEN_FIELDS.items():
            actual = {f.name for f in fields(type_)}
            self.assertEqual(
                actual,
                set(expected),
                f"{type_.__name__} field-name set changed without updating the freeze guard",
            )

    def test_field_order_matches_frozen_contract(self) -> None:
        for type_, expected in FROZEN_FIELDS.items():
            self.assertEqual(
                [f.name for f in fields(type_)],
                list(expected),
                f"{type_.__name__} field order changed without updating the freeze guard",
            )

    def test_field_types_match_frozen_contract(self) -> None:
        for type_, expected in FROZEN_FIELDS.items():
            actual = {f.name: _norm(str(f.type)) for f in fields(type_)}
            wanted = {name: _norm(t) for name, t in expected.items()}
            self.assertEqual(
                actual,
                wanted,
                f"{type_.__name__} field type(s) changed without updating the freeze guard",
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
        self.assertEqual(ctx.derived, {})

        strategy = Strategy()
        self.assertEqual(strategy.mode, "unknown")
        self.assertEqual(strategy.routes, [])
        self.assertEqual(strategy.route_weights, {})
        self.assertEqual(strategy.params, {})

        candidate = Candidate()
        self.assertEqual(candidate.parent_asin, "")
        self.assertEqual(candidate.route_scores, {})
        self.assertEqual(candidate.metadata, {})
        self.assertEqual(candidate.route_sources, ())

        result = RankingResult()
        self.assertEqual(result.ranked, [])
        self.assertEqual(result.diagnostics, {})

    def test_route_sources_property_derives_from_route_scores(self) -> None:
        candidate = Candidate()
        self.assertEqual(candidate.route_sources, ())
        candidate.route_scores["bm25"] = 1.2
        candidate.route_scores["category"] = 0.0
        self.assertEqual(candidate.route_sources, ("bm25", "category"))

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
        c.route_scores["bm25"] = 1.0
        c.metadata["price"] = None
        self.assertEqual(d.route_scores, {})
        self.assertEqual(d.metadata, {})

        s1 = Strategy()
        s2 = Strategy()
        s1.routes.append("bm25")
        s1.route_weights["bm25"] = 1.0
        s1.params["threshold"] = 0.5
        self.assertEqual(s2.routes, [])
        self.assertEqual(s2.route_weights, {})
        self.assertEqual(s2.params, {})

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


class NoneHandlingTest(unittest.TestCase):
    """CP 0.4 blocker fix — frozen None rule."""

    def test_none_for_container_fields_normalizes_to_empty(self) -> None:
        self.assertEqual(SessionState(user_profile=None).user_profile, {})
        self.assertEqual(SessionState(slots=None).slots, {})
        self.assertEqual(SessionState(evidence=None).evidence, [])
        self.assertEqual(SessionState(provenance=None).provenance, [])

        self.assertIsInstance(Context(state=None).state, SessionState)
        self.assertEqual(Context(derived=None).derived, {})

        self.assertEqual(Strategy(routes=None).routes, [])
        self.assertEqual(Strategy(route_weights=None).route_weights, {})
        self.assertEqual(Strategy(params=None).params, {})

        self.assertEqual(Candidate(route_scores=None).route_scores, {})
        self.assertEqual(Candidate(metadata=None).metadata, {})

        self.assertEqual(RankingResult(ranked=None).ranked, [])
        self.assertEqual(RankingResult(diagnostics=None).diagnostics, {})

    def test_container_fields_listed_for_none_rule_match_contract(self) -> None:
        # Every non-scalar field must be covered by the None rule.
        for type_, covered in CONTAINER_FIELDS.items():
            container_like = [
                f.name
                for f in fields(type_)
                if _norm(str(f.type)) not in {"str", "int", "float", "bool"}
            ]
            self.assertEqual(sorted(container_like), sorted(covered), type_.__name__)

    def test_normalized_none_containers_are_independent_instances(self) -> None:
        a = SessionState(slots=None)
        b = SessionState(slots=None)
        a.slots["x"] = 1
        self.assertEqual(b.slots, {})

        c = Candidate(route_scores=None)
        d = Candidate(route_scores=None)
        c.route_scores["bm25"] = 1.0
        self.assertEqual(d.route_scores, {})

    def test_scalar_none_passes_through_untouched(self) -> None:
        # Frozen rule: scalar fields are the caller's responsibility.
        # Construction must not raise; the value is not coerced.
        self.assertIsNone(SessionState(session_id=None).session_id)
        self.assertIsNone(SessionState(turn=None).turn)
        self.assertIsNone(Context(user_message=None).user_message)
        self.assertIsNone(Strategy(mode=None).mode)
        self.assertIsNone(Candidate(parent_asin=None).parent_asin)


if __name__ == "__main__":
    unittest.main()
