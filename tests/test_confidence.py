"""CP 11.1 - 11.5 — Evidence Confidence x Match Reliability.

EC says how firmly the shopper asserted a constraint. MR says how much the
catalog's verdict on that field is worth. They are independent, and the phase
exists because the two off-diagonal cases need opposite reassurances:

    CP 11.4  high EC, low MR   a real requirement we cannot check reliably
                               must not bury the right product
    CP 11.5  low EC, high MR   a passing remark must not act as a filter

Both are stated as "must not" properties, so they are tested as bounds on what
can happen, not as particular numbers.
"""

from __future__ import annotations

import sqlite3
import unittest
from unittest import mock

from starter import ranking
from starter.catalog_meta import TABLE, create_table, signals
from starter.contracts import Candidate, Context, SessionState
from starter.ranking import (RELIABILITY_KEY, VIOLATION_SLOTS, W_MATCH,
                             W_PENALTY, active_constraints,
                             constraint_weights, rank, score_candidate,
                             slot_confidence)
from starter.reliability import (DEFAULT_RELIABILITY, MIN_RELIABILITY,
                                 match_reliability, reliability_of,
                                 slot_coverage)
from starter.state import (EC_CORRECTION, EC_HEDGED, EC_REQUIREMENT, EC_STATED,
                           evidence_confidence, update_state, validate_delta)


def _state(*turns: str) -> SessionState:
    state = SessionState(session_id="s")
    for index, message in enumerate(turns, start=1):
        update_state(state, message, index)
    return state


def _context(state: SessionState, reliabilities=None) -> Context:
    context = Context(session_id="s", turn=state.turn, user_message="",
                      state=state)
    if reliabilities is not None:
        context.derived[RELIABILITY_KEY] = reliabilities
    return context


def _candidate(fusion: float = 0.02) -> Candidate:
    return Candidate(parent_asin="A", route_scores={"bm25": 1.0},
                     metadata={"fusion_score": fusion})


_EMPTY_META = {"color": set(), "material": set(), "cats": set(),
               "store": "", "sizes": set(), "price": None, "traits": set()}


def _meta(**overrides):
    return {**_EMPTY_META, **overrides}


class CP111EvidenceConfidenceTest(unittest.TestCase):
    def test_phrasing_sets_the_level(self) -> None:
        cases = [
            ("I need a jacket. A key requirement is: leather.", EC_REQUIREMENT),
            ("For that, what matters is: cotton.", EC_REQUIREMENT),
            ("Actually, make it denim.", EC_CORRECTION),
            # A correction cue marks the EDIT; a hedge marks commitment to the
            # VALUE, so a hedged correction is hedged. Found by the D Phase 11
            # interleaving test, which expected 0.4 and got 0.9.
            ("actually maybe denim", EC_HEDGED),
            ("Actually, ignore my earlier preference. What I need is: denim.",
             EC_REQUIREMENT),
            ("black leather jacket", EC_STATED),
            ("maybe something in black", EC_HEDGED),
            ("I'm looking for shoes, but I'm still exploring", EC_HEDGED),
        ]
        for message, expected in cases:
            self.assertEqual(evidence_confidence(message), expected, message)

    def test_requirement_language_outranks_a_hedge(self) -> None:
        # "I need something black" is a requirement that happens to contain a
        # hedge word, not a hedge.
        self.assertEqual(evidence_confidence("I need something black"),
                         EC_REQUIREMENT)

    def test_bad_input_does_not_raise(self) -> None:
        for bad in (None, 7, "", "   ", []):
            self.assertEqual(evidence_confidence(bad), EC_STATED, repr(bad))

    def test_confidence_lands_on_the_slot_entry(self) -> None:
        state = _state("maybe something in black")
        self.assertAlmostEqual(state.slots["color"]["confidence"], EC_HEDGED)

    def test_full_confidence_is_stored_as_absence(self) -> None:
        # Phase 4's convention, kept so a fully-meant slot is byte-identical
        # to its pre-Phase-11 self and the ranking flag's OFF position is exact.
        state = _state("I need a jacket. A key requirement is: leather.")
        self.assertNotIn("confidence", state.slots["material"])

    def test_a_union_keeps_the_firmest_statement(self) -> None:
        # colour is multi-valued: adding a hedged value to a slot the shopper
        # insisted on must not soften the insistence.
        state = _state("I need black. It is a requirement.", "also maybe red")
        self.assertEqual(set(state.slots["color"]["values"]), {"black", "red"})
        self.assertEqual(slot_confidence(_context(state), "color"), 1.0)

    def test_a_new_hedged_slot_is_not_rescued_by_the_absent_default(self) -> None:
        # Regression: combining against a slot that did not exist used
        # "absent means 1.0" as the starting value of the max, so a brand-new
        # hedged slot came out fully confident.
        state = _state("maybe something in black")
        self.assertLess(slot_confidence(_context(state), "color"), 1.0)

    def test_a_replacement_replaces_the_confidence_too(self) -> None:
        state = _state("I need leather. A key requirement.", "actually maybe denim")
        self.assertEqual(state.slots["material"]["values"], ["denim"])
        self.assertLess(slot_confidence(_context(state), "material"), 1.0,
                        "the old statement's confidence must not outlive it")

    def test_restating_more_firmly_is_not_a_no_op(self) -> None:
        state = _state("maybe leather")
        self.assertAlmostEqual(state.slots["material"]["confidence"], EC_HEDGED)
        update_state(state, "leather is a hard requirement", 2)
        self.assertEqual(slot_confidence(_context(state), "material"), 1.0)

    def test_the_validator_still_guards_the_field(self) -> None:
        for bad in (0.0, -1.0, float("nan"), "high"):
            self.assertEqual(
                validate_delta({"color": {"values": ["black"], "confidence": bad}}),
                {}, repr(bad))


class CP112MatchReliabilityTest(unittest.TestCase):
    @staticmethod
    def _connection(products):
        connection = sqlite3.connect(":memory:")
        create_table(connection)
        rows = [(str(p["parent_asin"]), *signals(p)) for p in products]
        connection.executemany(
            f"INSERT OR REPLACE INTO {TABLE} VALUES "
            f"({', '.join('?' * len(rows[0]))})", rows)
        return connection

    def test_coverage_is_read_from_the_catalog(self) -> None:
        products = [
            {"parent_asin": f"P{n}", "title": "Black Cotton Shirt" if n < 5
             else "Shirt", "categories": ["Clothing"], "features": [],
             "details": {}, "store": "acme", "description": []}
            for n in range(10)
        ]
        coverage = slot_coverage(self._connection(products))
        self.assertAlmostEqual(coverage["color"], 0.5)
        self.assertAlmostEqual(coverage["material"], 0.5)
        self.assertAlmostEqual(coverage["category"], 1.0)

    def test_an_empty_or_missing_table_yields_no_statistics(self) -> None:
        empty = sqlite3.connect(":memory:")
        create_table(empty)
        self.assertEqual(slot_coverage(empty), {})
        self.assertEqual(slot_coverage(sqlite3.connect(":memory:")), {})

    def test_no_statistics_means_fully_trusted(self) -> None:
        # An unknown slot must never be silently discounted -- that would be a
        # hard filter arriving through the back door.
        self.assertEqual(match_reliability({}), {})
        self.assertEqual(reliability_of(None, "color"), DEFAULT_RELIABILITY)
        self.assertEqual(reliability_of({}, "color"), DEFAULT_RELIABILITY)
        self.assertEqual(reliability_of({"color": 0.5}, "size"),
                         DEFAULT_RELIABILITY)

    def test_reliability_is_floored_and_clamped(self) -> None:
        table = match_reliability({"a": 0.0, "b": 5.0, "c": -1.0, "d": 0.5})
        self.assertEqual(table["a"], MIN_RELIABILITY)
        self.assertEqual(table["b"], 1.0)
        self.assertEqual(table["c"], MIN_RELIABILITY)
        self.assertAlmostEqual(table["d"], 0.5)

    def test_nothing_reaches_zero(self) -> None:
        # A zero would make a slot's verdicts unreachable, turning
        # "unreliable" into "ignored" and losing the real signal that survives.
        for value in (0.0, -3.0, 1e-9):
            self.assertGreaterEqual(reliability_of({"x": value}, "x"),
                                    MIN_RELIABILITY)

    def test_bad_values_do_not_raise(self) -> None:
        self.assertEqual(reliability_of({"x": None}, "x"), DEFAULT_RELIABILITY)
        self.assertEqual(reliability_of({"x": float("nan")}, "x"),
                         DEFAULT_RELIABILITY)
        self.assertEqual(match_reliability({"x": float("nan")}), {})


class QuadrantTest(unittest.TestCase):
    """CP 11.3 / 11.4 / 11.5 -- the three cases the phase is named for."""

    CONSTRAINTS = {"material": ["leather"]}
    VIOLATING = property(lambda self: _meta(material={"denim"}))
    MATCHING = property(lambda self: _meta(material={"leather"}))

    def _score(self, meta, weight):
        return score_candidate(_candidate(), self.CONSTRAINTS, meta,
                               weights={"material": weight})

    def test_cp_113_high_ec_high_mr_is_phase_6_behaviour(self) -> None:
        weighted = self._score(self.MATCHING, 1.0)
        unweighted = score_candidate(_candidate(), self.CONSTRAINTS,
                                     self.MATCHING, weights=None)
        self.assertAlmostEqual(weighted["final_score"],
                               unweighted["final_score"])
        self.assertAlmostEqual(weighted["attribute_score"], W_MATCH)

    def test_cp_114_low_mr_cannot_bury_a_violating_candidate(self) -> None:
        """Strong intent, unreliable catalog evidence: the penalty must scale
        down with the reliability, not stay at full strength."""
        reliable = self._score(self.VIOLATING, 1.0)
        unreliable = self._score(self.VIOLATING, MIN_RELIABILITY)
        self.assertAlmostEqual(reliable["violation_penalty"], W_PENALTY)
        self.assertAlmostEqual(unreliable["violation_penalty"],
                               W_PENALTY * MIN_RELIABILITY)
        self.assertLess(unreliable["violation_penalty"],
                        reliable["violation_penalty"])

    def test_cp_114_the_target_is_never_deleted(self) -> None:
        """The load-bearing guarantee: a violation demotes, never removes.

        True at every weight, including 1.0 -- it comes from the penalty being
        bounded (Phase 6, principle E), and Phase 11 only makes the size of
        the demotion proportionate to the evidence.
        """
        state = SessionState(session_id="s")
        state.slots["material"] = {"values": ["leather"], "cardinality": "multi"}
        context = _context(state, {"material": MIN_RELIABILITY})
        pool = [Candidate(parent_asin="TARGET", metadata={"fusion_score": 0.02}),
                Candidate(parent_asin="OTHER", metadata={"fusion_score": 0.02})]
        # Must hold in BOTH flag positions: the guarantee comes from Phase 6's
        # bounded penalty, and Phase 11 must not weaken it in either state.
        for enabled in (False, True):
            with mock.patch.object(ranking, "USE_CONFIDENCE_WEIGHTING", enabled):
                result = rank(pool, context,
                              {"TARGET": _meta(material={"denim"})}, 10)
            with self.subTest(weighting=enabled):
                self.assertIn("TARGET", [c.parent_asin for c in result.ranked])
                detail = result.diagnostics["TARGET"]
                # Demoted, never deleted, and by a bounded amount. Note the
                # loss can reach the whole base score at the bottom of the
                # fusion range -- that is Phase 6's bound, unchanged here.
                self.assertGreaterEqual(detail["final_score"],
                                        detail["base_score"] - W_PENALTY)

    def test_cp_115_a_hedged_constraint_is_not_a_filter(self) -> None:
        """Weak evidence, reliable catalog: a passing remark must not decide
        the ranking."""
        firm = self._score(self.VIOLATING, EC_REQUIREMENT)
        hedged = self._score(self.VIOLATING, EC_HEDGED)
        self.assertAlmostEqual(hedged["violation_penalty"],
                               W_PENALTY * EC_HEDGED)
        self.assertLess(hedged["violation_penalty"], firm["violation_penalty"])

    def test_cp_115_weight_survives_a_single_active_constraint(self) -> None:
        """The arithmetic bug this design had to avoid.

        Normalising by the SUMMED WEIGHT instead of the count cancels the
        weight whenever one constraint is active: 0.4/0.4 = full penalty, and
        CP 11.5 would be violated by the mechanism meant to satisfy it.
        """
        self.assertEqual(len(self.CONSTRAINTS), 1)
        hedged = self._score(self.VIOLATING, EC_HEDGED)
        self.assertLess(hedged["violation_penalty"], W_PENALTY,
                        "a single hedged constraint must still be discounted")

    def test_a_hedged_constraint_moves_the_ranking_less_than_a_firm_one(self) -> None:
        """What CP 11.5 can actually promise.

        NOT that retrieval outranks a weak constraint: W_MATCH is deliberately
        set above the fusion-score spread so that constraint evidence, not raw
        text overlap, decides the order (see ranking's W_MATCH comment). A
        satisfied constraint is SUPPOSED to beat a better-retrieved candidate.

        What weighting promises is narrower and is the thing "must not become
        a hard filter" actually asks for: the gap a constraint can open
        between satisfying and violating it shrinks with the evidence behind
        it.
        """
        def spread(weight):
            matching = score_candidate(_candidate(), self.CONSTRAINTS,
                                       self.MATCHING, weights={"material": weight})
            violating = score_candidate(_candidate(), self.CONSTRAINTS,
                                        self.VIOLATING, weights={"material": weight})
            return matching["final_score"] - violating["final_score"]

        self.assertLess(spread(EC_HEDGED), spread(EC_REQUIREMENT))
        # And it shrinks in proportion, not by an arbitrary amount.
        self.assertAlmostEqual(spread(EC_HEDGED),
                               spread(EC_REQUIREMENT) * EC_HEDGED)

    def test_a_violation_never_costs_more_than_the_bounded_penalty(self) -> None:
        # principle E, unchanged by Phase 11: a parser mistake stays
        # recoverable because the loss is bounded, whatever the weight.
        for weight in (1.0, EC_HEDGED, MIN_RELIABILITY):
            detail = self._score(self.VIOLATING, weight)
            self.assertGreaterEqual(detail["final_score"],
                                    detail["base_score"] - W_PENALTY)

    def test_weights_never_reorder_via_the_denominator(self) -> None:
        # The denominator is the constraint COUNT, identical for every
        # candidate in a turn, so weighting rescales but cannot reorder for
        # reasons unrelated to the candidate (the CP 6.4 property).
        constraints = {"material": ["leather"], "color": ["black"]}
        weights = {"material": 0.6, "color": 0.4}
        both = score_candidate(_candidate(), constraints,
                               _meta(material={"leather"}, color={"black"}),
                               weights=weights)
        self.assertAlmostEqual(both["attribute_score"], W_MATCH * (1.0 / 2))


class SingleConstraintReordersTest(unittest.TestCase):
    """B Phase 11 review: the "cannot reorder" claim was false.

    The comments and the census tool argued that a turn with one active
    constraint scales every candidate's attribute term by the same factor and
    is therefore a monotone rescale that cannot change the order. It is not:
    ``base`` is NOT multiplied by the weight, so changing the weight changes
    the strength of constraint evidence RELATIVE to retrieval order, and one
    constraint is enough to cross two candidates.

    Pinned so the wording cannot drift back.
    """

    CONSTRAINTS = {"material": ["leather"]}

    def _final(self, base, meta, weight):
        return score_candidate(
            Candidate(parent_asin="X", metadata={"fusion_score": base}),
            self.CONSTRAINTS, meta, weights={"material": weight},
        )["final_score"]

    def test_one_constraint_flips_two_candidates(self) -> None:
        matching_no_base = _meta(material={"leather"})   # base 0.00, MATCH
        unknown_high_base = _meta()                      # base 0.06, UNKNOWN

        at_full = (self._final(0.00, matching_no_base, 1.0),
                   self._final(0.06, unknown_high_base, 1.0))
        at_hedged = (self._final(0.00, matching_no_base, EC_HEDGED),
                     self._final(0.06, unknown_high_base, EC_HEDGED))

        self.assertGreater(at_full[0], at_full[1],
                           "at full weight the matching candidate wins")
        self.assertLess(at_hedged[0], at_hedged[1],
                        "at a hedged weight the better-retrieved one wins -- "
                        "a SINGLE constraint reordered the pool")

    def test_the_reorder_is_visible_through_rank(self) -> None:
        # Same thing end to end, so the property is pinned at the API a
        # consumer actually calls and not only in the scoring helper.
        state = SessionState(session_id="s")
        state.slots["material"] = {"values": ["leather"], "cardinality": "multi",
                                   "confidence": EC_HEDGED}
        pool = [Candidate(parent_asin="MATCHES", metadata={"fusion_score": 0.00}),
                Candidate(parent_asin="RETRIEVED", metadata={"fusion_score": 0.06})]
        metadata = {"MATCHES": _meta(material={"leather"}), "RETRIEVED": _meta()}

        orders = {}
        for enabled in (False, True):
            with mock.patch.object(ranking, "USE_CONFIDENCE_WEIGHTING", enabled):
                result = rank(pool, _context(state, {"material": 1.0}),
                              metadata, 10)
            orders[enabled] = [c.parent_asin for c in result.ranked]

        self.assertEqual(orders[False], ["MATCHES", "RETRIEVED"])
        self.assertEqual(orders[True], ["RETRIEVED", "MATCHES"])

    def test_the_same_holds_for_the_phase_6_denominator(self) -> None:
        # The identical error sat in ranking's module docstring one phase
        # earlier ("UNKNOWN never changes the relative order of candidates").
        # Adding a second constraint that is UNKNOWN for both candidates
        # halves the attribute term and crosses them, with no weighting at all.
        one = {"material": ["leather"]}
        two = {"material": ["leather"], "brand": ["acme"]}
        matching = _meta(material={"leather"})

        def final(constraints, base, meta):
            return score_candidate(
                Candidate(parent_asin="X", metadata={"fusion_score": base}),
                constraints, meta, weights=None)["final_score"]

        self.assertGreater(final(one, 0.00, matching), final(one, 0.06, _meta()))
        self.assertLess(final(two, 0.00, matching), final(two, 0.06, _meta()))


class WeightAssemblyTest(unittest.TestCase):
    def test_weight_is_ec_times_mr(self) -> None:
        state = _state("maybe something in black")
        context = _context(state, {"color": 0.5})
        constraints, _ = active_constraints(context)
        weights = constraint_weights(context, constraints)
        self.assertAlmostEqual(weights["color"], EC_HEDGED * 0.5)

    def test_missing_reliability_leaves_confidence_alone(self) -> None:
        state = _state("maybe something in black")
        weights = constraint_weights(_context(state),
                                     {"color": ["black"]})
        self.assertAlmostEqual(weights["color"], EC_HEDGED)

    def test_absent_confidence_leaves_reliability_alone(self) -> None:
        state = _state("I need black. A key requirement.")
        weights = constraint_weights(_context(state, {"color": 0.47}),
                                     {"color": ["black"]})
        self.assertAlmostEqual(weights["color"], 0.47)

    def test_malformed_state_does_not_raise(self) -> None:
        state = SessionState(session_id="s")
        state.slots = None  # type: ignore[assignment]
        self.assertEqual(slot_confidence(_context(state), "color"), 1.0)

    def test_size_is_still_excluded_from_violations(self) -> None:
        # Phase 11 grades the magnitude; it does not re-open the Phase 6
        # decision about which slots may violate at all. Changing both at once
        # would confound them.
        self.assertNotIn("size", VIOLATION_SLOTS)


if __name__ == "__main__":
    unittest.main()


class AgentLevelConfidenceIsolationTest(unittest.TestCase):
    """D1 / D2 / D5 — confidence is per-session state, exercised through the
    real ``Agent`` rather than by hand-built ``SessionState`` objects.

    Phase 11 put a new mutable field inside ``state.slots``. Everything the
    single-writer invariant already promised has to keep holding for it:
    ``reset`` clears it, interleaved sessions never see each other's, and the
    catalog-global reliability table is never written to.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from pathlib import Path

        from starter.agent import Agent

        catalog = Path("data/catalog.jsonl")
        if not catalog.exists():
            raise unittest.SkipTest("catalog not available")
        cls.agent = Agent(str(catalog))

    PROFILE = {"purchase_frequency": "often", "average_prior_rating": 4.0,
               "rating_style": "generous", "preference_tags": ["fit"],
               "summary": "likes fitted clothes"}
    OTHER_PROFILE = {"purchase_frequency": "rarely", "average_prior_rating": 2.0,
                     "rating_style": "harsh", "preference_tags": ["warmth"],
                     "summary": "wants warm things"}

    def _slots(self, session_id: str) -> dict:
        return self.agent._states[session_id].slots

    def test_d1_reset_clears_confidence_and_history(self) -> None:
        self.agent.reset("D1", self.PROFILE)
        self.agent.respond("D1", "maybe leather jacket", 1, 10)
        slots = self._slots("D1")
        self.assertEqual(slots["material"]["values"], ["leather"])
        self.assertAlmostEqual(slots["material"]["confidence"], EC_HEDGED)

        self.agent.reset("D1", self.OTHER_PROFILE)
        state = self.agent._states["D1"]
        self.assertEqual(state.slots, {}, "old slots and confidence must be gone")
        self.assertEqual(state.evidence, [])
        self.assertEqual(state.provenance, [])
        self.assertEqual(state.turn, 0)
        self.assertEqual(state.user_profile, self.OTHER_PROFILE)

        # The new session's confidence comes only from the new session.
        self.agent.respond("D1", "I need cotton. A key requirement.", 1, 10)
        self.assertEqual(self._slots("D1")["material"]["values"], ["cotton"])
        self.assertNotIn("confidence", self._slots("D1")["material"],
                         "a requirement is EC 1.0, stored as absence")

    def test_d1_reset_does_not_leak_into_the_callers_profile(self) -> None:
        profile = dict(self.PROFILE)
        self.agent.reset("D1b", profile)
        self.agent.respond("D1b", "maybe leather", 1, 10)
        self.agent._states["D1b"].user_profile["preference_tags"].append("x")
        self.assertEqual(profile["preference_tags"], ["fit"])

    def test_d2_interleaved_sessions_keep_their_own_confidence(self) -> None:
        self.agent.reset("A", self.PROFILE)
        self.agent.reset("B", self.OTHER_PROFILE)

        self.agent.respond("A", "maybe leather jacket", 1, 10)
        self.agent.respond("B", "I need cotton. A key requirement.", 1, 10)

        self.assertAlmostEqual(self._slots("A")["material"]["confidence"],
                               EC_HEDGED)
        self.assertNotIn("confidence", self._slots("B")["material"])
        self.assertEqual(self._slots("A")["material"]["values"], ["leather"])
        self.assertEqual(self._slots("B")["material"]["values"], ["cotton"])

        # Escalating A must not touch B.
        self.agent.respond("A", "leather is a hard requirement", 2, 10)
        self.assertNotIn("confidence", self._slots("A")["material"])
        self.assertEqual(self._slots("B")["material"]["values"], ["cotton"])
        self.assertNotIn("confidence", self._slots("B")["material"])

        # Replacing in B must not touch A.
        self.agent.respond("B", "actually maybe denim", 2, 10)
        self.assertEqual(self._slots("B")["material"]["values"], ["denim"])
        self.assertAlmostEqual(self._slots("B")["material"]["confidence"],
                               EC_HEDGED)
        self.assertEqual(self._slots("A")["material"]["values"], ["leather"])
        self.assertNotIn("confidence", self._slots("A")["material"])

    def test_d2_no_slot_object_is_shared_between_sessions(self) -> None:
        self.agent.reset("A2", self.PROFILE)
        self.agent.reset("B2", self.PROFILE)
        self.agent.respond("A2", "maybe leather", 1, 10)
        self.agent.respond("B2", "maybe leather", 1, 10)
        self.assertIsNot(self._slots("A2")["material"],
                         self._slots("B2")["material"])

    def test_d5_the_shared_reliability_table_is_never_mutated(self) -> None:
        import copy as _copy

        before = _copy.deepcopy(self.agent._reliability)
        identity = id(self.agent._reliability)
        self.agent.reset("R", self.PROFILE)
        for turn, message in enumerate(
                ["maybe leather jacket", "I need black. A requirement.",
                 "actually denim", "size 10"], start=1):
            self.agent.respond("R", message, turn, 10)
        self.assertEqual(self.agent._reliability, before,
                         "catalog-global reliability is read-only state")
        self.assertEqual(id(self.agent._reliability), identity,
                         "and it is shared, not rebuilt per turn")


class ReliabilityFailureBoundaryTest(unittest.TestCase):
    """D3 — the documented fallback boundary, and where it deliberately ends.

    "No statistics" means "trust everything", because discounting an unknown
    slot would be a hard filter arriving through the back door. But a table
    that exists with the WRONG SHAPE is a programming error, not a missing
    statistic, and it is left to propagate rather than be swallowed into a
    silently-degraded reliability table.
    """

    def test_missing_table_yields_no_statistics(self) -> None:
        self.assertEqual(slot_coverage(sqlite3.connect(":memory:")), {})

    def test_empty_table_yields_no_statistics(self) -> None:
        connection = sqlite3.connect(":memory:")
        create_table(connection)
        self.assertEqual(slot_coverage(connection), {})

    def test_no_statistics_means_every_slot_fully_trusted(self) -> None:
        self.assertEqual(match_reliability(None), {})
        self.assertEqual(match_reliability({}), {})
        for slot in ("color", "size", "not_a_slot"):
            self.assertEqual(reliability_of(None, slot), DEFAULT_RELIABILITY)
            self.assertEqual(reliability_of({}, slot), DEFAULT_RELIABILITY)

    def test_malformed_values_fall_back_deterministically(self) -> None:
        for bad in (None, float("nan"), "0.5", [], {}):
            self.assertEqual(reliability_of({"color": bad}, "color"),
                             DEFAULT_RELIABILITY, repr(bad))
        self.assertEqual(match_reliability({"color": float("nan"),
                                            "size": "x"}), {})

    def test_a_corrupt_schema_propagates_rather_than_degrading(self) -> None:
        # The boundary, pinned deliberately. A product_meta that exists but is
        # missing a column we aggregate is a broken build, not a catalog
        # without colour data. Swallowing it would hand back a reliability
        # table that is quietly wrong, and every score derived from it would
        # be quietly wrong too.
        connection = sqlite3.connect(":memory:")
        connection.execute(f"CREATE TABLE {TABLE} (parent_asin TEXT, colors TEXT)")
        connection.execute(f"INSERT INTO {TABLE} VALUES ('A', 'black')")
        with self.assertRaises(sqlite3.Error):
            slot_coverage(connection)


class NegatedRequirementTest(unittest.TestCase):
    """D Phase 12 review, P2 — a withdrawn requirement scored as a maximal one.

    "Leather is not required" contains "required", so requirement detection
    fired and the slot was recorded at EC 1.0 -- maximum insistence, at the
    exact moment the shopper relaxed it. Inert while Phase 11 weighting was
    the only consumer; score-bearing from Phase 12, because CP 12.3's decay
    reads EC whatever that flag says.
    """

    WITHDRAWN = (
        "leather is not required",
        "leather is not a requirement",
        "leather does not matter",
        "leather is no longer required",
        "cotton isn't essential",
        "the material doesn't matter",
    )

    def test_a_withdrawn_requirement_is_not_a_requirement(self) -> None:
        for message in self.WITHDRAWN:
            self.assertEqual(evidence_confidence(message), EC_HEDGED, message)

    def test_a_real_requirement_still_reads_as_one(self) -> None:
        for message in ("I need a jacket. A key requirement is: leather.",
                        "For that, what matters is: cotton.",
                        "it must be leather",
                        "leather is essential"):
            self.assertEqual(evidence_confidence(message), EC_REQUIREMENT,
                             message)

    def test_a_negated_verb_is_not_a_negated_requirement(self) -> None:
        # "must not be leather" is still a firm assertion -- the shopper is
        # insisting, just exclusively. The negation follows the requirement
        # word rather than preceding it.
        self.assertEqual(evidence_confidence("it must not be leather"),
                         EC_REQUIREMENT)

    def test_the_slot_survives_at_lower_confidence(self) -> None:
        # Deliberately NOT a removal: a false REMOVE destroys a constraint the
        # shopper still wants, and this codebase already took the conservative
        # side of that trade once (state._STRONG_NEGATIONS excludes "drop").
        state = _state("maybe a leather jacket", "leather is not required")
        self.assertEqual(state.slots["material"]["values"], ["leather"])
        self.assertAlmostEqual(
            slot_confidence(_context(state), "material"), EC_HEDGED)

    def test_the_escalation_path_no_longer_fires_on_a_withdrawal(self) -> None:
        # The exact end-to-end shape D reported: hedged, then "withdrawn",
        # must not come back as fully confident.
        state = _state("maybe a leather jacket")
        self.assertAlmostEqual(state.slots["material"]["confidence"], EC_HEDGED)
        update_state(state, "leather is not required", 2)
        self.assertLess(slot_confidence(_context(state), "material"), 1.0,
                        "withdrawing a requirement must never raise confidence")


class PartialRemovalConfidenceTest(unittest.TestCase):
    """D Phase 12 review, Q1 — the sibling escalation path.

    ``apply_delta`` REBUILDS the slot entry when a removal leaves survivors,
    so anything not copied across is lost. ``confidence`` was not copied, so
    removing one value of a multi-valued slot promoted the remaining ones from
    a hedge to maximum insistence -- the same defect class as P2, at a
    different site, and score-bearing for the same reason (CP 12.3's decay
    reads EC whatever USE_CONFIDENCE_WEIGHTING says).
    """

    def test_removing_a_sibling_does_not_promote_the_survivor(self) -> None:
        state = _state("maybe leather and denim jacket", "no leather")
        self.assertEqual(state.slots["material"]["values"], ["denim"])
        self.assertAlmostEqual(
            slot_confidence(_context(state), "material"), EC_HEDGED,
            msg="removing leather says nothing about commitment to denim")

    def test_the_same_holds_for_colour(self) -> None:
        state = _state("maybe black and navy", "not black")
        self.assertEqual(state.slots["color"]["values"], ["navy"])
        self.assertAlmostEqual(
            slot_confidence(_context(state), "color"), EC_HEDGED)

    def test_a_firm_survivor_stays_firm(self) -> None:
        state = _state("I need leather and denim. A requirement.", "no leather")
        self.assertEqual(state.slots["material"]["values"], ["denim"])
        self.assertEqual(slot_confidence(_context(state), "material"), 1.0)

    def test_removing_everything_still_drops_the_slot(self) -> None:
        state = _state("maybe leather and denim jacket", "no leather", "no denim")
        self.assertNotIn("material", state.slots)

    def test_bounds_survive_a_partial_removal_too(self) -> None:
        # Same rebuild, same class of loss. Defensive: a single-valued budget
        # cannot currently reach this branch with survivors.
        state = SessionState(session_id="s")
        state.slots["budget"] = {"values": ["under $100", "under $50"],
                                 "cardinality": "multi",
                                 "bounds": {"min": None, "max": 100.0},
                                 "confidence": EC_HEDGED}
        from starter.state import apply_delta
        apply_delta(state, {"budget": {"values": [], "remove": ["under $50"],
                                       "cardinality": "multi"}}, 2)
        self.assertEqual(state.slots["budget"]["values"], ["under $100"])
        self.assertEqual(state.slots["budget"]["bounds"], {"min": None, "max": 100.0})
        self.assertAlmostEqual(state.slots["budget"]["confidence"], EC_HEDGED)
