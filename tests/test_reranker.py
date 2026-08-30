"""CP 14.1 - 14.5 -- the reranker contract, and every way a scorer can fail.

The scorer is the untrusted layer of this pipeline: it may be absent, slow,
raise, or return nonsense, and in all four cases the turn must come back with
the order ranking already produced. So most of this file is adversarial
scorers rather than happy paths.

CP 14.6 (the ON/OFF ablation) is measured on the real dataset by
``tools/phase14_reranker.py``; it is not a unit test.
"""

from __future__ import annotations

import time
import unittest

from starter import reranker
from starter.contracts import Candidate, Context, RankingResult, SessionState
from starter.reranker import (OUTCOMES, RERANK_KEY, PoolTermScorer,
                              _evidence_terms, build_scorer,
                              load_encoder_scorer, rerank)


def _candidates(*asins: str) -> list[Candidate]:
    return [
        Candidate(parent_asin=asin, route_scores={"bm25": 1.0},
                  metadata={"fusion_score": 1.0 / (index + 1)})
        for index, asin in enumerate(asins)
    ]


def _result(*asins: str) -> RankingResult:
    candidates = _candidates(*asins)
    return RankingResult(
        ranked=candidates,
        diagnostics={
            candidate.parent_asin: {"rank": index + 1, "final_score": 1.0}
            for index, candidate in enumerate(candidates)
        },
    )


def _context(evidence: list[str] | None = None) -> Context:
    state = SessionState(session_id="s", turn=1)
    for text in evidence or ():
        state.evidence.append(
            {"turn": 1, "text": text, "normalized": text, "status": "active"}
        )
    return Context(session_id="s", turn=1, user_message="hi", state=state)


class _Scorer:
    """A scorer returning whatever it was constructed with."""

    name = "stub"

    def __init__(self, output, delay: float = 0.0) -> None:
        self.output = output
        self.delay = delay
        self.seen: list[str] | None = None

    def order(self, parent_asins, context, deadline):
        self.seen = list(parent_asins)
        if self.delay:
            time.sleep(self.delay)
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


def _asins(result: RankingResult) -> list[str]:
    return [candidate.parent_asin for candidate in result.ranked]


class TopNInputOnlyTest(unittest.TestCase):
    """CP 14.1 -- the scorer sees the head and only the head."""

    def test_scorer_receives_at_most_top_n(self) -> None:
        scorer = _Scorer([])
        rerank(_result("A", "B", "C", "D"), _context(), scorer, top_n=2)
        self.assertEqual(scorer.seen, ["A", "B"])

    def test_tail_keeps_its_order_and_position(self) -> None:
        scorer = _Scorer(["B", "A"])
        result = rerank(_result("A", "B", "C", "D"), _context(), scorer, top_n=2)
        self.assertEqual(_asins(result), ["B", "A", "C", "D"])

    def test_tail_is_never_promoted_by_the_scorer(self) -> None:
        # "D" is in the pool but outside the window, so naming it must not
        # move it -- otherwise top_n would not bound anything.
        scorer = _Scorer(["D", "B", "A"])
        result = rerank(_result("A", "B", "C", "D"), _context(), scorer, top_n=2)
        self.assertEqual(_asins(result), ["B", "A", "C", "D"])

    def test_window_larger_than_the_list_is_harmless(self) -> None:
        scorer = _Scorer(["C", "B", "A"])
        result = rerank(_result("A", "B", "C"), _context(), scorer, top_n=999)
        self.assertEqual(_asins(result), ["C", "B", "A"])

    def test_zero_window_reranks_nothing(self) -> None:
        scorer = _Scorer(["C", "B", "A"])
        result = rerank(_result("A", "B", "C"), _context(), scorer, top_n=0)
        self.assertEqual(_asins(result), ["A", "B", "C"])
        self.assertEqual(result.diagnostics["A"]["rank"], 1)


class ConstantsAreReadAtCallTimeTest(unittest.TestCase):
    """The defaults must follow the module constants, not freeze at import.

    ``def rerank(..., top_n=RERANK_TOP_N)`` binds 50 once, forever. With that
    signature a tool sweeping ``reranker.RERANK_TOP_N`` moves the ranking
    depth in ``agent.respond`` while ``rerank`` keeps reranking the first 50 --
    so the sweep reports one window's behaviour under every window's label.
    The first CP 14.6 sweep did exactly that and its numbers were withdrawn.
    """

    def test_top_n_follows_the_module_constant(self) -> None:
        original = reranker.RERANK_TOP_N
        try:
            reranker.RERANK_TOP_N = 2
            scorer = _Scorer([])
            reranker.rerank(_result("A", "B", "C", "D"), _context(), scorer)
            self.assertEqual(scorer.seen, ["A", "B"])
            reranker.RERANK_TOP_N = 4
            scorer = _Scorer([])
            reranker.rerank(_result("A", "B", "C", "D"), _context(), scorer)
            self.assertEqual(scorer.seen, ["A", "B", "C", "D"])
        finally:
            reranker.RERANK_TOP_N = original

    def test_budget_follows_the_module_constant(self) -> None:
        original = reranker.RERANK_BUDGET_MS
        try:
            reranker.RERANK_BUDGET_MS = 0.0
            context = _context()
            reranker.rerank(_result("A", "B"), context,
                            _Scorer(["B", "A"], delay=0.002))
            self.assertEqual(context.derived[RERANK_KEY]["outcome"], "timeout")
        finally:
            reranker.RERANK_BUDGET_MS = original

    def test_an_explicit_argument_still_wins(self) -> None:
        scorer = _Scorer([])
        reranker.rerank(_result("A", "B", "C"), _context(), scorer, top_n=1)
        self.assertEqual(scorer.seen, ["A"])


class CandidateIdPreservationTest(unittest.TestCase):
    """CP 14.2 -- the semantic model cannot invent ASINs."""

    def test_invented_asin_is_discarded(self) -> None:
        scorer = _Scorer(["B0FAKE", "B", "A"])
        result = rerank(_result("A", "B", "C"), _context(), scorer)
        self.assertEqual(_asins(result), ["B", "A", "C"])
        self.assertNotIn("B0FAKE", _asins(result))

    def test_invented_asins_are_counted(self) -> None:
        scorer = _Scorer(["B0FAKE", "B0ALSOFAKE", "B", "A", "C"])
        context = _context()
        rerank(_result("A", "B", "C"), context, scorer)
        self.assertEqual(context.derived[RERANK_KEY]["invented"], 2)

    def test_output_is_always_a_permutation_of_the_input(self) -> None:
        for output in (["C"], ["C", "C", "C"], ["B0FAKE", "C"],
                       ["C", "B", "A", "B0FAKE"], [None, "C", 7, "A"]):
            result = rerank(_result("A", "B", "C"), _context(), _Scorer(output))
            self.assertEqual(sorted(_asins(result)), ["A", "B", "C"], output)

    def test_omitted_candidates_are_appended_not_dropped(self) -> None:
        # A scorer that returns one id must not be able to shorten the list.
        scorer = _Scorer(["C"])
        result = rerank(_result("A", "B", "C"), _context(), scorer)
        self.assertEqual(_asins(result), ["C", "A", "B"])

    def test_duplicates_in_the_proposal_collapse(self) -> None:
        scorer = _Scorer(["C", "C", "A", "C"])
        result = rerank(_result("A", "B", "C"), _context(), scorer)
        self.assertEqual(_asins(result), ["C", "A", "B"])

    def test_every_ranked_candidate_still_has_diagnostics(self) -> None:
        result = rerank(_result("A", "B", "C"), _context(), _Scorer(["C", "A"]))
        self.assertEqual({c.parent_asin for c in result.ranked},
                         set(result.diagnostics))

    def test_ranks_are_renumbered_to_the_new_order(self) -> None:
        result = rerank(_result("A", "B", "C"), _context(), _Scorer(["C", "B", "A"]))
        self.assertEqual([result.diagnostics[a]["rank"] for a in ("C", "B", "A")],
                         [1, 2, 3])

    def test_the_input_result_is_not_mutated(self) -> None:
        source = _result("A", "B", "C")
        rerank(source, _context(), _Scorer(["C", "B", "A"]))
        self.assertEqual(_asins(source), ["A", "B", "C"])
        self.assertEqual(source.diagnostics["A"]["rank"], 1)


class MalformedOutputFallbackTest(unittest.TestCase):
    """CP 14.3 -- garbage in, ranking's own order out."""

    def test_non_list_output_falls_back(self) -> None:
        for output in (None, "ABC", 42, {"a": 1}, object()):
            context = _context()
            result = rerank(_result("A", "B", "C"), context, _Scorer(output))
            self.assertEqual(_asins(result), ["A", "B", "C"], output)
            self.assertEqual(context.derived[RERANK_KEY]["outcome"], "malformed")

    def test_all_invented_output_falls_back(self) -> None:
        context = _context()
        result = rerank(_result("A", "B", "C"), context,
                        _Scorer(["B0X", "B0Y"]))
        self.assertEqual(_asins(result), ["A", "B", "C"])
        self.assertEqual(context.derived[RERANK_KEY]["outcome"], "malformed")

    def test_empty_output_falls_back(self) -> None:
        context = _context()
        result = rerank(_result("A", "B", "C"), context, _Scorer([]))
        self.assertEqual(_asins(result), ["A", "B", "C"])
        self.assertEqual(context.derived[RERANK_KEY]["outcome"], "malformed")

    def test_raising_scorer_falls_back(self) -> None:
        for error in (ValueError("bad"), RuntimeError("worse"),
                      KeyError("worst"), MemoryError()):
            context = _context()
            result = rerank(_result("A", "B", "C"), context, _Scorer(error))
            self.assertEqual(_asins(result), ["A", "B", "C"], error)
            self.assertEqual(context.derived[RERANK_KEY]["outcome"], "error")

    def test_scorer_without_an_order_method_falls_back(self) -> None:
        context = _context()
        result = rerank(_result("A", "B"), context, object())
        self.assertEqual(_asins(result), ["A", "B"])
        self.assertEqual(context.derived[RERANK_KEY]["outcome"], "error")


class TimeoutFallbackTest(unittest.TestCase):
    """CP 14.4 -- a late answer is a wrong answer."""

    def test_overrunning_scorer_is_discarded_even_when_valid(self) -> None:
        context = _context()
        scorer = _Scorer(["C", "B", "A"], delay=0.05)
        result = rerank(_result("A", "B", "C"), context, scorer, budget_ms=1.0)
        self.assertEqual(_asins(result), ["A", "B", "C"])
        self.assertEqual(context.derived[RERANK_KEY]["outcome"], "timeout")

    def test_a_scorer_inside_budget_is_honoured(self) -> None:
        context = _context()
        result = rerank(_result("A", "B", "C"), context,
                        _Scorer(["C", "B", "A"]), budget_ms=5000.0)
        self.assertEqual(_asins(result), ["C", "B", "A"])
        self.assertEqual(context.derived[RERANK_KEY]["outcome"], "applied")

    def test_zero_budget_falls_back(self) -> None:
        context = _context()
        result = rerank(_result("A", "B", "C"), context,
                        _Scorer(["C", "B", "A"], delay=0.002), budget_ms=0.0)
        self.assertEqual(_asins(result), ["A", "B", "C"])
        self.assertEqual(context.derived[RERANK_KEY]["outcome"], "timeout")

    def test_the_deadline_is_handed_to_the_scorer(self) -> None:
        seen: list[float] = []

        class _Deadline:
            name = "deadline"

            def order(self, parent_asins, context, deadline):
                seen.append(deadline)
                return list(parent_asins)

        rerank(_result("A", "B"), _context(), _Deadline(), budget_ms=100.0)
        self.assertEqual(len(seen), 1)
        self.assertGreater(seen[0], time.monotonic())


class OfflineFallbackTest(unittest.TestCase):
    """CP 14.5 -- no model, no reranking, no error."""

    def test_none_scorer_falls_back(self) -> None:
        context = _context()
        result = rerank(_result("A", "B", "C"), context, None)
        self.assertEqual(_asins(result), ["A", "B", "C"])
        self.assertEqual(context.derived[RERANK_KEY]["outcome"], "offline")

    def test_no_encoder_is_vendored_in_this_repository(self) -> None:
        # Not a mock: this is the real state of the tree, and it is what makes
        # CP 14.5 the live path rather than a defensive branch.
        self.assertIsNone(load_encoder_scorer())

    def test_the_encoder_loader_does_not_import_or_download(self) -> None:
        # It must not make the shipped path depend on what happens to be
        # installed on the scoring machine (the claim Phase 13 was corrected
        # for). Cheap enough to assert on the clock.
        started = time.monotonic()
        load_encoder_scorer()
        self.assertLess(time.monotonic() - started, 0.05)

    def test_empty_ranked_list_is_total(self) -> None:
        context = _context()
        result = rerank(RankingResult(), context, _Scorer(["A"]))
        self.assertEqual(result.ranked, [])
        self.assertEqual(context.derived[RERANK_KEY]["outcome"], "empty")

    def test_every_reported_outcome_is_declared(self) -> None:
        seen = set()
        for scorer, budget in ((None, 150.0), (_Scorer([]), 150.0),
                               (_Scorer(ValueError()), 150.0),
                               (_Scorer(["A", "B"], delay=0.02), 1.0),
                               (_Scorer(["B", "A"]), 150.0),
                               (_Scorer(["A", "B"]), 150.0)):
            context = _context()
            rerank(_result("A", "B"), context, scorer, budget_ms=budget)
            seen.add(context.derived[RERANK_KEY]["outcome"])
        self.assertTrue(seen <= set(OUTCOMES), seen - set(OUTCOMES))
        self.assertEqual(
            seen, {"offline", "malformed", "error", "timeout", "applied", "identity"})


class PoolTermScorerTest(unittest.TestCase):
    """The shippable scorer: lexical, pool-scoped, and deterministic."""

    def _scorer(self) -> PoolTermScorer:
        # "strap" is in 1 of 4, "leather" in 3 of 4: the rare word discriminates.
        return PoolTermScorer(
            {"pool_size": 4, "terms": {"leather": 3, "strap": 1, "boot": 2}},
            {"A": ["leather", "boot"], "B": ["leather", "strap"],
             "C": ["leather"], "D": ["boot"]},
        )

    def test_a_rare_pool_term_outranks_a_common_one(self) -> None:
        order = self._scorer().order(["A", "B", "C", "D"],
                                     _context(["strap"]), time.monotonic() + 10)
        self.assertEqual(order[0], "B")

    def test_no_evidence_is_the_identity_permutation(self) -> None:
        order = self._scorer().order(["A", "B", "C"], _context(),
                                     time.monotonic() + 10)
        self.assertEqual(order, ["A", "B", "C"])

    def test_evidence_the_pool_does_not_use_is_the_identity(self) -> None:
        order = self._scorer().order(["A", "B", "C"], _context(["helicopter"]),
                                     time.monotonic() + 10)
        self.assertEqual(order, ["A", "B", "C"])

    def test_ties_keep_the_incoming_rank(self) -> None:
        # All three match "leather" equally, so ranking's order must survive.
        order = self._scorer().order(["C", "A", "B"], _context(["leather"]),
                                     time.monotonic() + 10)
        self.assertEqual(order, ["C", "A", "B"])

    def test_it_is_deterministic(self) -> None:
        scorer = self._scorer()
        context = _context(["strap", "leather"])
        first = scorer.order(["A", "B", "C", "D"], context, time.monotonic() + 10)
        second = scorer.order(["A", "B", "C", "D"], context, time.monotonic() + 10)
        self.assertEqual(first, second)

    def test_superseded_evidence_is_ignored(self) -> None:
        # Phase 3 marks a withdrawn preference superseded; reranking on it
        # would be exactly the stale-evidence resurrection that phase prevents.
        context = _context(["strap"])
        context.state.evidence[0]["status"] = "superseded"
        self.assertEqual(_evidence_terms(context), set())
        order = self._scorer().order(["A", "B", "C"], context,
                                     time.monotonic() + 10)
        self.assertEqual(order, ["A", "B", "C"])

    def test_it_is_total_on_empty_and_missing_inputs(self) -> None:
        for scorer in (PoolTermScorer(None, None), PoolTermScorer({}, {}),
                       PoolTermScorer({"pool_size": 0, "terms": {}}, {})):
            self.assertEqual(
                scorer.order(["A", "B"], _context(["strap"]),
                             time.monotonic() + 10),
                ["A", "B"],
            )

    def test_a_term_outside_the_pool_vocabulary_weighs_nothing(self) -> None:
        self.assertEqual(self._scorer().weight("helicopter"), 0.0)
        self.assertGreater(self._scorer().weight("strap"),
                           self._scorer().weight("leather"))


class BuildScorerTest(unittest.TestCase):
    """The per-turn scorer choice, including when NOT to build one."""

    def test_no_free_text_means_no_scorer(self) -> None:
        # PoolTermScorer would be the identity, and building it costs two
        # indexed queries. This repo has twice deleted a mechanism that
        # computed a value nothing used.
        self.assertIsNone(build_scorer(None, _candidates("A"), _context()))

    def test_free_text_builds_the_lexical_scorer_without_a_catalog(self) -> None:
        # `None` connection would raise if build_scorer queried eagerly for a
        # turn it should have skipped; it must only query when it will score.
        with self.assertRaises(Exception):
            build_scorer(None, _candidates("A"), _context(["strap"]))


if __name__ == "__main__":
    unittest.main()
