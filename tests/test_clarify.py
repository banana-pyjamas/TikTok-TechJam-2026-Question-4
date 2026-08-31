"""CP 15.1 - 15.7 -- the clarification layer.

CP 15.8 (the OFF/ON ablation) is measured on the real dataset by
``tools/phase15_clarification.py``; it is not a unit test.

The two checkpoints that are easiest to write a green test for and hardest to
get right are CP 15.6 and CP 15.7, so they get the most adversarial coverage
here: a question loop that never terminates would not fail any assertion about
one turn, and on a 10-turn harness it would not fail the score either -- it
would just quietly stop learning anything after turn 2.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from starter import clarify
from starter.clarify import (ALLOWED_ATTRIBUTES, ASK_VALUE_FLOOR,
                             EVIDENCE_ATTRIBUTES, SCORABLE_ATTRIBUTES,
                             WILDCARD, ClarificationLedger, attribute_value,
                             choose, rank_attributes)
from starter.contracts import Candidate, Context, SessionState

_CONTRACT = Path("docs/agent_api_contract.json")


def _context(slots: dict | None = None) -> Context:
    state = SessionState(session_id="s", turn=1, slots=slots or {})
    return Context(session_id="s", turn=1, user_message="hi", state=state)


def _pool(*rows: dict) -> tuple[list[Candidate], dict[str, dict]]:
    """Candidates plus the ``catalog_meta`` rows a real turn would look up."""
    blank = {"color": set(), "material": set(), "cats": set(), "store": "",
             "sizes": set(), "price": None}
    candidates, metadata = [], {}
    for index, row in enumerate(rows):
        asin = f"P{index}"
        candidates.append(Candidate(parent_asin=asin))
        metadata[asin] = {**blank, **row}
    return candidates, metadata


class NoAskBaselineTest(unittest.TestCase):
    """CP 15.1 -- the flag OFF is a true no-op, checked at the payload."""

    def test_to_response_defaults_to_no_question(self) -> None:
        from starter.agent import _to_response
        from starter.contracts import RankingResult

        payload = _to_response(RankingResult(ranked=[Candidate("A")]), 10)
        self.assertIsNone(payload["ask_attribute"])

    def test_the_flag_is_read_through_the_module(self) -> None:
        # `from starter.clarify import USE_CLARIFICATION` in agent.py would
        # bind a COPY that config_guard.set_flag could never flip -- the arm
        # would report OFF and run ON. Phase 12 closed that hole once and
        # Phase 14 wrote it down; this is the test neither of them had.
        source = Path("starter/agent.py").read_text(encoding="utf-8")
        self.assertIn("clarify.USE_CLARIFICATION", source)
        self.assertNotIn("from starter.clarify import", source)


class AllowedValuesTest(unittest.TestCase):
    """CP 15.3 -- only the organizer's enum ever reaches the wire."""

    def test_allowed_matches_the_organizer_contract(self) -> None:
        contract = json.loads(_CONTRACT.read_text(encoding="utf-8"))
        enum = contract["turn_response"]["properties"]["ask_attribute"]["enum"]
        self.assertEqual(set(ALLOWED_ATTRIBUTES),
                         {value for value in enum if value is not None})
        self.assertIn(None, enum)

    def test_the_wildcard_is_legal(self) -> None:
        self.assertIn(WILDCARD, ALLOWED_ATTRIBUTES)

    def test_scorable_is_a_subset_of_allowed(self) -> None:
        self.assertTrue(set(SCORABLE_ATTRIBUTES) <= set(ALLOWED_ATTRIBUTES))

    def test_evidence_is_a_subset_of_allowed_and_disjoint_from_scorable(self) -> None:
        self.assertTrue(set(EVIDENCE_ATTRIBUTES) <= set(ALLOWED_ATTRIBUTES))
        self.assertEqual(
            set(EVIDENCE_ATTRIBUTES) & set(SCORABLE_ATTRIBUTES), set())

    def test_the_two_tiers_plus_the_wildcard_are_the_whole_enum(self) -> None:
        # Nothing legal is silently unaskable. If the organizer adds a value,
        # this fails until someone decides which tier it belongs in.
        self.assertEqual(
            set(SCORABLE_ATTRIBUTES) | set(EVIDENCE_ATTRIBUTES) | {WILDCARD},
            set(ALLOWED_ATTRIBUTES))

    def test_scorable_is_exactly_what_the_pipeline_can_act_on(self) -> None:
        # The claim in the module docstring, asserted rather than asserted-in-
        # prose: an attribute is scorable iff an answer can become a slot AND
        # be checked against a catalog_meta column. Adding a slot without
        # deciding whether it is askable breaks here.
        from starter.reliability import _NUMERIC_SLOTS, _SLOT_COLUMN
        from starter.state import SLOT_CARDINALITY

        checkable = set(_SLOT_COLUMN) | set(_NUMERIC_SLOTS)
        self.assertEqual(set(SCORABLE_ATTRIBUTES),
                         set(SLOT_CARDINALITY) & checkable)

    def test_choose_never_returns_an_illegal_value(self) -> None:
        candidates, metadata = _pool({"color": {"black"}}, {"color": {"red"}})
        for ledger in (ClarificationLedger(), ClarificationLedger()):
            chosen = choose(_context(), ledger, candidates, metadata, None)
            self.assertIn(chosen, (*ALLOWED_ATTRIBUTES, None))


class QuestionValueTest(unittest.TestCase):
    """CP 15.5 -- a question is worth what knowing the answer would split."""

    def test_an_attribute_nobody_declares_is_worthless(self) -> None:
        _, metadata = _pool({"color": {"black"}}, {"color": {"red"}})
        rows = list(metadata.values())
        self.assertEqual(attribute_value("material", rows, {}, None, False), 0.0)

    def test_an_attribute_everyone_agrees_on_is_worthless(self) -> None:
        # The pool has already answered the question. This is the case an
        # "ask about the first empty slot" policy cannot see.
        _, metadata = _pool({"color": {"black"}}, {"color": {"black"}},
                            {"color": {"black"}})
        rows = list(metadata.values())
        self.assertEqual(attribute_value("color", rows, {}, None, False), 0.0)

    def test_a_split_attribute_is_worth_more_than_a_lopsided_one(self) -> None:
        _, split = _pool({"color": {"black"}}, {"color": {"red"}})
        _, lopsided = _pool({"color": {"black"}}, {"color": {"black"}},
                            {"color": {"black"}}, {"color": {"red"}})
        self.assertGreater(
            attribute_value("color", list(split.values()), {}, None, False),
            attribute_value("color", list(lopsided.values()), {}, None, False))

    def test_partial_declaration_discounts_the_value(self) -> None:
        _, all_declared = _pool({"color": {"black"}}, {"color": {"red"}})
        _, half_declared = _pool({"color": {"black"}}, {"color": {"red"}},
                                 {}, {})
        self.assertGreater(
            attribute_value("color", list(all_declared.values()), {}, None,
                            False),
            attribute_value("color", list(half_declared.values()), {}, None,
                            False))

    def test_reliability_discounts_the_value(self) -> None:
        _, metadata = _pool({"color": {"black"}}, {"color": {"red"}})
        rows = list(metadata.values())
        trusted = attribute_value("color", rows, {}, None, False)
        doubted = attribute_value("color", rows, {}, {"color": 0.5}, False)
        self.assertAlmostEqual(doubted, trusted * 0.5)

    def test_a_known_attribute_is_not_a_question(self) -> None:
        _, metadata = _pool({"color": {"black"}}, {"color": {"red"}})
        rows = list(metadata.values())
        self.assertEqual(attribute_value("color", rows, {}, None, True), 0.0)

    def test_an_empty_pool_is_worth_nothing_rather_than_raising(self) -> None:
        self.assertEqual(attribute_value("color", [], {}, None, False), 0.0)

    def test_budget_is_scored_on_pool_relative_buckets(self) -> None:
        # Absolute price bands would call both of these pools "agreed", for
        # opposite reasons. Relative buckets see the spread in both.
        cheap = _pool(*({"price": value} for value in (5.0, 6.0, 7.0, 40.0)))[1]
        dear = _pool(*({"price": value} for value in (80.0, 120.0, 300.0,
                                                      400.0)))[1]
        for metadata in (cheap, dear):
            rows = list(metadata.values())
            buckets = clarify._price_buckets(rows)
            self.assertGreater(
                attribute_value("budget", rows, buckets, None, False), 0.0)

    def test_ties_break_deterministically(self) -> None:
        candidates, metadata = _pool(
            {"color": {"black"}, "material": {"cotton"}},
            {"color": {"red"}, "material": {"wool"}})
        first = rank_attributes(_context(), candidates, metadata, None)
        second = rank_attributes(_context(), candidates, metadata, None)
        self.assertEqual(first, second)
        self.assertEqual([name for _, name in first][:2], ["color", "material"])


class AskDecisionTest(unittest.TestCase):
    """CP 15.2 -- and the fallback to the open question."""

    def test_a_discriminating_attribute_beats_the_wildcard(self) -> None:
        candidates, metadata = _pool({"color": {"black"}}, {"color": {"red"}})
        self.assertEqual(
            choose(_context(), ClarificationLedger(), candidates, metadata,
                   None),
            "color")

    def test_a_pool_with_no_signal_asks_an_evidence_question(self) -> None:
        # NOT the wildcard. A specific question is always preferable to an
        # open one, and the evidence tier is the specific question that is
        # still available when nothing is value-scorable.
        candidates, metadata = _pool({}, {})
        self.assertEqual(
            choose(_context(), ClarificationLedger(), candidates, metadata,
                   None),
            EVIDENCE_ATTRIBUTES[0])

    def test_missing_metadata_is_a_question_not_a_crash(self) -> None:
        candidates = [Candidate(parent_asin="A")]
        self.assertEqual(
            choose(_context(), ClarificationLedger(), candidates, None, None),
            EVIDENCE_ATTRIBUTES[0])

    def test_a_value_below_the_floor_falls_through_the_first_tier(self) -> None:
        # 1 of 40 declares, so answerable is 0.025 and the product cannot
        # reach the floor however split the declaring candidates are.
        rows = [{"color": {"black"}}] + [{} for _ in range(39)]
        candidates, metadata = _pool(*rows)
        self.assertLess(
            rank_attributes(_context(), candidates, metadata, None)[0][0],
            ASK_VALUE_FLOOR)
        self.assertEqual(
            choose(_context(), ClarificationLedger(), candidates, metadata,
                   None),
            EVIDENCE_ATTRIBUTES[0])

    def test_the_evidence_tier_is_exhausted_before_the_open_question(self) -> None:
        # The property that made bounding the wildcard free: every specific
        # question is tried first, so the open one is genuinely a last resort.
        candidates, metadata = _pool({}, {})
        ledger = ClarificationLedger()
        asked = []
        for _ in range(len(EVIDENCE_ATTRIBUTES)):
            chosen = choose(_context(), ledger, candidates, metadata, None)
            asked.append(chosen)
            ledger.record(chosen)
            ledger.observe(f"I don't have a preference for {chosen}.")
        self.assertEqual(asked, list(EVIDENCE_ATTRIBUTES))
        self.assertEqual(
            choose(_context(), ledger, candidates, metadata, None), WILDCARD)


class NoRepeatedQuestionsTest(unittest.TestCase):
    """CP 15.6."""

    def test_a_filled_slot_is_never_asked_about(self) -> None:
        candidates, metadata = _pool({"color": {"black"}}, {"color": {"red"}})
        context = _context({"color": {"values": ["black"],
                                      "cardinality": "multi"}})
        self.assertNotEqual(
            choose(context, ClarificationLedger(), candidates, metadata, None),
            "color")

    def test_a_closed_attribute_is_never_asked_again(self) -> None:
        candidates, metadata = _pool({"color": {"black"}}, {"color": {"red"}})
        ledger = ClarificationLedger()
        ledger.closed.add("color")
        self.assertNotEqual(
            choose(_context(), ledger, candidates, metadata, None), "color")

    def test_a_productive_specific_question_may_be_asked_again(self) -> None:
        # "And what else about colour?" is not a repeated question. Only an
        # attribute that answered EMPTY is closed -- see CP 15.7 below.
        candidates, metadata = _pool({"color": {"black"}}, {"color": {"red"}})
        ledger = ClarificationLedger()
        ledger.record(choose(_context(), ledger, candidates, metadata, None))
        ledger.observe("For that, what matters is: black.")
        self.assertEqual(
            choose(_context(), ledger, candidates, metadata, None), "color")

    def test_a_productive_open_question_is_NOT_farmed(self) -> None:
        # The B Phase 15 blocker. The wildcard used to be re-askable while it
        # kept paying, which on a harness whose "other" matches a superset of
        # every specific attribute is farming the simulator. It is now capped
        # by MAX_OPEN_QUESTIONS however productive it is.
        candidates, metadata = _pool({}, {})
        ledger = ClarificationLedger()
        ledger.closed.update(EVIDENCE_ATTRIBUTES)
        for _ in range(clarify.MAX_OPEN_QUESTIONS):
            self.assertEqual(
                choose(_context(), ledger, candidates, metadata, None),
                WILDCARD)
            ledger.record(WILDCARD)
            ledger.observe("For that, what matters is: merino wool.")
        self.assertNotEqual(
            choose(_context(), ledger, candidates, metadata, None), WILDCARD)


class NoPreferenceAndLoopTerminationTest(unittest.TestCase):
    """CP 15.7 -- and the reason the question loop cannot run forever."""

    def test_a_declined_question_is_closed(self) -> None:
        ledger = ClarificationLedger()
        ledger.record("color")
        ledger.observe("I don't have a preference for color; "
                       "please use your judgment.")
        self.assertIn("color", ledger.closed)

    def test_the_other_decline_wording_is_also_closed(self) -> None:
        ledger = ClarificationLedger()
        ledger.record("material")
        ledger.observe("I don't have an additional preference for material.")
        self.assertIn("material", ledger.closed)

    def test_a_substantive_answer_does_not_close_anything(self) -> None:
        ledger = ClarificationLedger()
        ledger.record("color")
        ledger.observe("For that, what matters is: black.")
        self.assertEqual(ledger.closed, set())

    def test_an_answer_with_no_pending_question_closes_nothing(self) -> None:
        ledger = ClarificationLedger()
        ledger.observe("I don't have a preference for color.")
        self.assertEqual(ledger.closed, set())

    def test_the_loop_terminates_when_every_answer_declines(self) -> None:
        # The adversarial session: the shopper declines everything. The
        # policy must run out of questions rather than cycle, and it must do
        # so within the number of legal attributes -- not within the
        # evaluator's 10-turn cap, which would leave the bound accidental.
        candidates, metadata = _pool({"color": {"black"}}, {"color": {"red"}},
                                     {"material": {"wool"}},
                                     {"material": {"cotton"}})
        ledger = ClarificationLedger()
        asked: list[str] = []
        for _ in range(len(ALLOWED_ATTRIBUTES) + 1):
            chosen = choose(_context(), ledger, candidates, metadata, None)
            if chosen is None:
                break
            asked.append(chosen)
            ledger.record(chosen)
            ledger.observe(f"I don't have a preference for {chosen}.")
        else:
            self.fail(f"never stopped asking; asked {asked}")
        self.assertEqual(len(asked), len(set(asked)), f"repeated: {asked}")
        self.assertLessEqual(len(asked), len(ALLOWED_ATTRIBUTES))

    def test_nothing_left_to_ask_is_a_clean_none(self) -> None:
        ledger = ClarificationLedger()
        ledger.closed.update(ALLOWED_ATTRIBUTES)
        candidates, metadata = _pool({"color": {"black"}}, {"color": {"red"}})
        self.assertIsNone(
            choose(_context(), ledger, candidates, metadata, None))


class OpenQuestionBudgetTest(unittest.TestCase):
    """CP 15.2 -- the wildcard is a bounded last resort, not a fallback.

    The B Phase 15 blocker. Everything here is about the FOURTH rung of
    ``choose`` existing: without "ask the best specific attribute even below
    the floor", capping the wildcard does not produce a policy that asks
    specific questions, it produces one that goes silent -- which is what the
    0.413290 "no-fallback" arm measured and what made the open question look
    load-bearing.
    """

    def _weak_pool(self):
        # One candidate in forty declares a colour: real signal, but far under
        # ASK_VALUE_FLOOR. This is the state where the old policy reached for
        # the wildcard on every turn.
        rows = [{"color": {"black"}}] + [{"color": {"red"}}] + [
            {} for _ in range(38)]
        return _pool(*rows)

    def test_below_the_floor_a_specific_question_still_comes_first(self) -> None:
        candidates, metadata = self._weak_pool()
        ledger = ClarificationLedger()
        chosen = choose(_context(), ledger, candidates, metadata, None)
        self.assertNotEqual(chosen, WILDCARD)
        self.assertIn(chosen, EVIDENCE_ATTRIBUTES)

    def test_a_spent_budget_falls_back_to_a_real_question(self) -> None:
        # The last rung. NOT to None: a policy whose only fallback is silence
        # makes the wildcard load-bearing by construction.
        candidates, metadata = self._weak_pool()
        ledger = ClarificationLedger()
        ledger.wildcard_uses = clarify.MAX_OPEN_QUESTIONS
        ledger.closed.update(EVIDENCE_ATTRIBUTES)
        chosen = choose(_context(), ledger, candidates, metadata, None)
        self.assertIsNotNone(chosen)
        self.assertNotEqual(chosen, WILDCARD)
        self.assertIn(chosen, SCORABLE_ATTRIBUTES)

    def test_a_zero_budget_never_asks_the_open_question(self) -> None:
        # MAX_OPEN_QUESTIONS = 0 is policy B, the strictly generic one. It
        # must remain reachable by moving the constant alone.
        candidates, metadata = self._weak_pool()
        ledger = ClarificationLedger()
        original = clarify.MAX_OPEN_QUESTIONS
        clarify.MAX_OPEN_QUESTIONS = 0
        try:
            for _ in range(10):
                chosen = choose(_context(), ledger, candidates, metadata, None)
                if chosen is None:
                    break
                self.assertNotEqual(chosen, WILDCARD)
                ledger.record(chosen)
                ledger.observe(f"I don't have a preference for {chosen}.")
        finally:
            clarify.MAX_OPEN_QUESTIONS = original

    def test_the_budget_counts_only_the_wildcard(self) -> None:
        ledger = ClarificationLedger()
        ledger.record("color")
        ledger.record("material")
        self.assertEqual(ledger.wildcard_uses, 0)
        ledger.record(WILDCARD)
        self.assertEqual(ledger.wildcard_uses, 1)

    def test_the_budget_is_per_session(self) -> None:
        self.assertEqual(ClarificationLedger().wildcard_uses, 0)


class DegradesRatherThanRaisesTest(unittest.TestCase):
    """Every untrusted layer degrades rather than raises.

    Phase 14 got ``safe_build_scorer`` for exactly this and Phase 15 shipped
    ``clarify.choose`` bare one phase later (D Phase 15 review). The rule is
    written down now; these are the tests that keep it.
    """

    class _Exploding:
        @property
        def parent_asin(self):
            raise RuntimeError("candidate metadata unavailable")

    def test_a_raising_candidate_yields_no_question(self) -> None:
        self.assertIsNone(clarify.safe_choose(
            _context(), ClarificationLedger(), [self._Exploding()], {}, None))

    _ADVERSARIAL = (
        (None, None), ("abc", {}), (7, {}), ([], None),
        (["not a candidate"], {}),
        ([Candidate("A")], {"A": None}),
        ([Candidate("A")], {"A": {"price": "free"}}),
        ([Candidate("A")], {"A": {"color": "black"}}),
        ([Candidate("A")], {"A": {"cats": 5}}),
        ([Candidate("A")], "not a dict"),
    )

    def test_adversarial_inputs_never_raise_through_the_guard(self) -> None:
        ledger = ClarificationLedger()
        for ranked, metadata in self._ADVERSARIAL:
            chosen = clarify.safe_choose(_context(), ledger, ranked, metadata,
                                         None)
            self.assertIn(chosen, (*ALLOWED_ATTRIBUTES, None))

    def test_the_guard_is_not_decoration(self) -> None:
        # Pins the claim in safe_choose's docstring: the bare function DOES
        # raise on some of these. An earlier version of that docstring said
        # it did not -- written from reading the code rather than running it.
        # If `choose` is ever made total, this test says so out loud instead
        # of letting the docstring drift the other way.
        raising = 0
        for ranked, metadata in self._ADVERSARIAL:
            try:
                clarify.choose(_context(), ClarificationLedger(), ranked,
                               metadata, None)
            except Exception:
                raising += 1
        self.assertEqual(raising, 4,
                         "the number of adversarial inputs that raise from "
                         "the bare `choose` has changed; update the count in "
                         "safe_choose's docstring in the same edit")

    def test_a_raising_observe_does_not_escape(self) -> None:
        class _Exploding:
            def __contains__(self, item):
                raise RuntimeError

        ledger = ClarificationLedger()
        ledger.record("color")
        clarify.safe_observe(ledger, _Exploding())
        self.assertIsNone(ledger.pending)

    def test_the_agent_calls_the_guarded_entry_points(self) -> None:
        source = Path("starter/agent.py").read_text(encoding="utf-8")
        self.assertIn("clarify.safe_choose(", source)
        self.assertIn("clarify.safe_observe(", source)
        self.assertNotIn("clarify.choose(", source)
        self.assertNotIn("ledger.observe(", source)


class LedgerIsAgentOwnedTest(unittest.TestCase):
    """The single-writer invariant: clarification never writes SessionState."""

    def test_choosing_does_not_mutate_state(self) -> None:
        context = _context()
        before = (dict(context.state.slots), list(context.state.evidence),
                  list(context.state.provenance))
        candidates, metadata = _pool({"color": {"black"}}, {"color": {"red"}})
        ledger = ClarificationLedger()
        ledger.record(choose(context, ledger, candidates, metadata, None))
        ledger.observe("I don't have a preference for color.")
        self.assertEqual(
            (context.state.slots, context.state.evidence,
             context.state.provenance),
            before)

    def test_a_reset_session_starts_with_a_clean_ledger(self) -> None:
        ledger = ClarificationLedger()
        self.assertEqual(ledger.asked, [])
        self.assertEqual(ledger.closed, set())
        self.assertIsNone(ledger.pending)


if __name__ == "__main__":
    unittest.main()
