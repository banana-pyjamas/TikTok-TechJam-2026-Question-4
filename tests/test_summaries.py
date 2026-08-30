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


if __name__ == "__main__":
    unittest.main()
