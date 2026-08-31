"""The shared distribution summary the measurement tools report from.

D-R4: the Phase 13 commit reported "median 0.95 filled slots" for a number the
tool had computed as a MEAN. The tool printed it correctly; positional
unpacking of a ``(mean, median, p90)`` tuple is how the label came off the
wrong value on the way into the commit message. These tests pin the field
names and the order statistics so that mistake cannot be made silently again.
"""

from __future__ import annotations

import unittest

from tools.summaries import percentiles


class PercentilesTest(unittest.TestCase):
    def test_mean_and_median_are_not_interchangeable(self) -> None:
        # A skewed sample, so a mean printed under a median's label is visible.
        shape = percentiles([1, 1, 1, 1, 96])
        self.assertEqual(shape.median, 1)
        self.assertEqual(shape.mean, 20.0)

    def test_fields_are_named_not_positional(self) -> None:
        shape = percentiles([5, 1, 3])
        self.assertEqual(shape.mean, 3.0)
        self.assertEqual(shape.median, 3)
        self.assertEqual(shape.maximum, 5)

    def test_order_statistics_are_observed_values(self) -> None:
        # P90 is a member of the sample, never an interpolation between two.
        values = list(range(1, 11))
        self.assertIn(percentiles(values).p90, values)
        self.assertEqual(percentiles(values).p90, 10)

    def test_input_is_not_reordered_in_place(self) -> None:
        values = [3, 1, 2]
        percentiles(values)
        self.assertEqual(values, [3, 1, 2])

    def test_empty_is_all_zeros_rather_than_an_error(self) -> None:
        self.assertEqual(tuple(percentiles([])), (0.0, 0.0, 0.0, 0.0))



class SessionCompositeTest(unittest.TestCase):
    """The composite must BE the score, not a proxy for it.

    A paired permutation over per-session composites is only a test about
    TechnicalScore if averaging the composites reproduces TechnicalScore. If
    the two ever drift, the permutation quietly starts answering a different
    question and nothing else in the repo would notice.
    """

    def _sessions(self):
        return [
            {"sample_id": "a", "hit": True, "first_hit_turn": 1,
             "reciprocal_rank": 1.0},
            {"sample_id": "b", "hit": True, "first_hit_turn": 7,
             "reciprocal_rank": 0.125},
            {"sample_id": "c", "hit": False, "first_hit_turn": None,
             "reciprocal_rank": 0.0},
            {"sample_id": "d", "hit": True, "first_hit_turn": 10,
             "reciprocal_rank": 0.5},
        ]

    def test_the_mean_composite_is_the_technical_score(self) -> None:
        from evaluator.local_evaluator import metric_summary
        from tools.significance import session_composite

        sessions = self._sessions()
        overall = metric_summary(sessions)
        efficiency = max(0.0, min(1.0, (11.0 - float(overall["mttc"])) / 10.0))
        evaluator_ts = (0.50 * overall["hit_rate_at_10"]
                        + 0.30 * overall["mrr"] + 0.20 * efficiency)
        mean_composite = sum(session_composite(s) for s in sessions) / len(sessions)
        self.assertAlmostEqual(mean_composite, evaluator_ts, places=9)

    def test_the_weights_match_the_evaluator(self) -> None:
        # The weights are duplicated in significance.py so it stays importable
        # without the evaluator. Duplication is fine; drift is not.
        import inspect

        from evaluator import local_evaluator
        from tools import significance

        source = inspect.getsource(local_evaluator.evaluate)
        self.assertIn("0.50 * overall[\"hit_rate_at_10\"]", source)
        self.assertIn("0.30 * overall[\"mrr\"]", source)
        self.assertIn("0.20 * efficiency", source)
        self.assertEqual(
            (significance._W_HIT, significance._W_MRR, significance._W_EFF),
            (0.50, 0.30, 0.20))
        self.assertEqual(significance._MAX_TURNS, local_evaluator.MAX_TURNS)

    def test_a_miss_is_charged_the_evaluator_penalty(self) -> None:
        from tools.significance import session_composite

        miss = session_composite({"hit": False, "first_hit_turn": None,
                                  "reciprocal_rank": 0.0})
        self.assertEqual(miss, 0.0)

    def test_permutation_and_bootstrap_are_reproducible(self) -> None:
        from tools.significance import bootstrap_ci, paired_permutation

        before = {str(i): float(i % 3) for i in range(200)}
        after = {str(i): float(i % 3) + (0.1 if i % 5 else -0.05)
                 for i in range(200)}
        self.assertEqual(paired_permutation(before, after),
                         paired_permutation(before, after))
        self.assertEqual(bootstrap_ci(before, after),
                         bootstrap_ci(before, after))

    def test_permutation_never_claims_certainty(self) -> None:
        # (1 + extreme) / (1 + rounds) can never be 0.0 -- 10k resamples do
        # not license a p of exactly zero.
        from tools.significance import paired_permutation

        before = {str(i): 0.0 for i in range(200)}
        after = {str(i): 1.0 for i in range(200)}
        self.assertGreater(paired_permutation(before, after)["p"], 0.0)

    def test_an_unchanged_run_is_no_verdict(self) -> None:
        from tools.significance import bootstrap_ci, paired_permutation

        same = {str(i): float(i) for i in range(200)}
        test = paired_permutation(same, dict(same))
        self.assertEqual(test["moved"], 0)
        self.assertEqual(test["p"], 1.0)
        interval = bootstrap_ci(same, dict(same))
        self.assertEqual((interval["low"], interval["high"]), (0.0, 0.0))

if __name__ == "__main__":
    unittest.main()
