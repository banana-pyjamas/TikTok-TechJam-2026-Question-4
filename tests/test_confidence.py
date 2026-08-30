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
