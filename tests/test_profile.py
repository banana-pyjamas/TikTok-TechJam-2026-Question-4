"""CP 8.1 - 8.6 — anonymized-profile context.

The load-bearing property is a PRIORITY one (principle I):

    current explicit request  >  active session state  >  profile prior

It holds structurally, not by bookkeeping: profile evidence never reaches
``state.slots``, and ``W_PROFILE`` is an order of magnitude below
``W_MATCH``, so satisfying an entire profile cannot outweigh one satisfied
constraint.
"""

from __future__ import annotations

import copy
import unittest
from unittest import mock

from starter import ranking
from starter.contracts import Candidate, Context, SessionState
from starter.profile import TAG_TERMS, extract_evidence, profile_match_ratio
from starter.ranking import rank
from starter.state import update_state

_EMPTY_META = {
    "color": set(), "material": set(), "cats": set(),
    "store": "", "sizes": set(), "price": None, "traits": set(),
}

_REAL_PROFILE = {
    "purchase_frequency": "3-4 prior purchases",
    "average_prior_rating": 5.0,
    "rating_style": "usually positive",
    "preference_tags": ["fit", "comfort", "durability"],
    "summary": "Prior purchases emphasize fit, comfort, durability.",
}


def _meta(**overrides) -> dict:
    return {**_EMPTY_META, **overrides}


def _candidate(asin: str, fusion: float = 0.01) -> Candidate:
    return Candidate(parent_asin=asin, route_scores={"bm25": 1.0},
                     metadata={"fusion_score": fusion})


def _context(profile: object = None, **slots) -> Context:
    state = SessionState(session_id="s", user_profile=profile)
    for name, values in slots.items():
        cardinality = "single" if name in ("category", "size", "brand", "budget") else "multi"
        state.slots[name] = {"values": list(values), "cardinality": cardinality}
    return Context(session_id="s", turn=1, user_message="", state=state)


class _ProfileEnabled(unittest.TestCase):
    """These exercise the profile scoring path, so they enable USE_PROFILE
    explicitly rather than depending on the committed default (measured
    net-negative in Phase 8, so it ships OFF)."""

    def setUp(self) -> None:
        patcher = mock.patch.object(ranking, "USE_PROFILE", True)
        patcher.start()
        self.addCleanup(patcher.stop)


class CP81StoreUserProfile(unittest.TestCase):
    """Storage landed at CP 1.1; these pin it as a Phase 8 requirement."""

    def test_reset_stores_a_deep_copy_on_session_state(self) -> None:
        state = SessionState(session_id="s", user_profile=copy.deepcopy(_REAL_PROFILE))
        self.assertEqual(state.user_profile["preference_tags"],
                         ["fit", "comfort", "durability"])

    def test_state_updates_never_touch_the_stored_profile(self) -> None:
        state = SessionState(session_id="s", user_profile=copy.deepcopy(_REAL_PROFILE))
        before = copy.deepcopy(state.user_profile)
        update_state(state, "black leather jacket", 1)
        update_state(state, "actually denim", 2)
        self.assertEqual(state.user_profile, before)


class CP82ExtractProfileEvidence(unittest.TestCase):
    def test_mapped_tags_yield_catalog_terms(self) -> None:
        evidence = extract_evidence({"preference_tags": ["warmth", "weather"]})
        self.assertEqual(evidence["tags"], ["warmth", "weather"])
        self.assertIn("insulated", evidence["terms"])
        self.assertIn("waterproof", evidence["terms"])

    def test_abstract_tags_are_deliberately_unmapped(self) -> None:
        # "cares about material" says nothing about WHICH material.
        evidence = extract_evidence({"preference_tags": ["fit", "material", "style"]})
        self.assertEqual(evidence["tags"], [])
        self.assertEqual(evidence["terms"], frozenset())

    def test_unknown_tags_are_ignored_not_guessed(self) -> None:
        self.assertEqual(extract_evidence({"preference_tags": ["telepathy"]})["tags"], [])

    def test_tags_are_deduplicated_and_order_preserving(self) -> None:
        evidence = extract_evidence(
            {"preference_tags": ["warmth", "comfort", "warmth"]})
        self.assertEqual(evidence["tags"], ["warmth", "comfort"])

    def test_match_ratio_counts_evidenced_tags(self) -> None:
        evidence = extract_evidence({"preference_tags": ["warmth", "weather"]})
        self.assertEqual(profile_match_ratio(evidence, {"insulated"}), 0.5)
        self.assertEqual(profile_match_ratio(evidence, {"insulated", "waterproof"}), 1.0)
        self.assertEqual(profile_match_ratio(evidence, set()), 0.0)

    def test_profile_evidence_is_read_only(self) -> None:
        profile = copy.deepcopy(_REAL_PROFILE)
        extract_evidence(profile)
        self.assertEqual(profile, _REAL_PROFILE)


class CP83CurrentIntentBeatsProfile(_ProfileEnabled):
    """GOLDEN: profile says Nike, the user asks for Adidas -> Adidas wins.

    Note the real evaluator profiles contain no brands at all (only 9
    abstract tags), so this is a SAFETY guard against a profile ever being
    allowed to act like a constraint -- not an observed scenario.
    """

    def test_explicit_brand_request_beats_a_brand_flavoured_profile(self) -> None:
        profile = {"preference_tags": ["comfort"], "summary": "loves Nike"}
        ctx = _context(profile, brand=["adidas"])
        metadata = {
            # The "profile favourite": comfortable, but the wrong brand.
            "NIKE": _meta(store="nike", traits={"comfortable", "cushioned"}),
            # What the user actually asked for, with no profile appeal at all.
            "ADIDAS": _meta(store="adidas"),
        }
        result = rank([_candidate("NIKE", 0.02), _candidate("ADIDAS", 0.02)],
                      ctx, metadata, 10)
        self.assertEqual(result.ranked[0].parent_asin, "ADIDAS")

    def test_profile_never_writes_into_slots(self) -> None:
        # The structural guarantee: a profile cannot masquerade as a
        # constraint because it never enters the slot channel.
        ctx = _context(_REAL_PROFILE)
        rank([_candidate("A", 0.01)], ctx, {"A": _meta()}, 10)
        self.assertEqual(ctx.state.slots, {})

    def test_a_whole_profile_cannot_outweigh_one_constraint(self) -> None:
        self.assertLess(ranking.W_PROFILE, ranking.W_MATCH,
                        "profile must be the weakest tier (principle I)")
        ctx = _context({"preference_tags": ["warmth", "weather", "comfort"]},
                       color=["black"])
        metadata = {
            # satisfies the entire profile, violates the one real constraint
            "PROFILE": _meta(color={"blue"},
                             traits={"insulated", "waterproof", "cushioned"}),
            # satisfies the constraint, appeals to the profile not at all
            "CONSTRAINT": _meta(color={"black"}),
        }
        result = rank([_candidate("PROFILE", 0.02), _candidate("CONSTRAINT", 0.02)],
                      ctx, metadata, 10)
        self.assertEqual(result.ranked[0].parent_asin, "CONSTRAINT")


class CP84SessionBeatsProfile(_ProfileEnabled):
    def test_accumulated_session_constraint_outranks_the_profile(self) -> None:
        state = SessionState(session_id="s",
                             user_profile={"preference_tags": ["warmth"]})
        update_state(state, "I want a denim jacket", 1)  # session evidence
        ctx = Context(session_id="s", turn=1, user_message="", state=state)
        metadata = {
            "WARM": _meta(traits={"insulated", "fleece"}),
            "DENIM": _meta(material={"denim"}, cats={"jackets"}),
        }
        result = rank([_candidate("WARM", 0.02), _candidate("DENIM", 0.02)],
                      ctx, metadata, 10)
        self.assertEqual(result.ranked[0].parent_asin, "DENIM")


class CP85CurrentTurnBeatsPreviousState(_ProfileEnabled):
    def test_override_wins_over_the_earlier_turn_and_the_profile(self) -> None:
        state = SessionState(session_id="s",
                             user_profile={"preference_tags": ["warmth"]})
        update_state(state, "leather jacket", 1)
        update_state(state, "actually denim", 2)          # current turn
        ctx = Context(session_id="s", turn=2, user_message="actually denim",
                      state=state)
        metadata = {
            "LEATHER": _meta(material={"leather"}, cats={"jackets"},
                             traits={"insulated"}),
            "DENIM": _meta(material={"denim"}, cats={"jackets"}),
        }
        result = rank([_candidate("LEATHER", 0.02), _candidate("DENIM", 0.02)],
                      ctx, metadata, 10)
        self.assertEqual(result.ranked[0].parent_asin, "DENIM")
        self.assertNotIn("leather", ranking.active_constraints(ctx)[0].get("material", []))


class CP86EmptyProfileRegression(_ProfileEnabled):
    def test_extract_evidence_survives_every_degenerate_profile(self) -> None:
        for profile in ({}, None, [], "nope", 42,
                        {"preference_tags": None},
                        {"preference_tags": "fit"},
                        {"preference_tags": [None, 3, ""]},
                        {"preference_tags": []}):
            evidence = extract_evidence(profile)
            self.assertEqual(evidence["tags"], [], repr(profile))
            self.assertEqual(evidence["terms"], frozenset(), repr(profile))

    def test_ranking_is_unchanged_by_an_empty_profile(self) -> None:
        metadata = {"A": _meta(color={"black"}, traits={"insulated"}),
                    "B": _meta(color={"blue"})}
        candidates = [_candidate("A", 0.02), _candidate("B", 0.01)]
        reference = rank(candidates, _context({}, color=["black"]), metadata, 10)
        for profile in (None, {}, {"preference_tags": []}, "garbage"):
            result = rank(candidates, _context(profile, color=["black"]), metadata, 10)
            self.assertEqual([c.parent_asin for c in result.ranked],
                             [c.parent_asin for c in reference.ranked], repr(profile))
            for asin, detail in result.diagnostics.items():
                self.assertEqual(detail["profile_score"], 0.0)

    def test_profile_score_is_always_bounded(self) -> None:
        ctx = _context({"preference_tags": list(TAG_TERMS)})
        metadata = {"A": _meta(traits=set().union(*TAG_TERMS.values()))}
        result = rank([_candidate("A", 0.01)], ctx, metadata, 10)
        self.assertAlmostEqual(result.diagnostics["A"]["profile_score"],
                               ranking.W_PROFILE)


if __name__ == "__main__":
    unittest.main()
