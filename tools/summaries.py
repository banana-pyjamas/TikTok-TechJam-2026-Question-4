"""Distribution summaries shared by the measurement tools.

Third copy of the same six lines, so it stops being a copy. ``_percentiles``
lived privately in ``tools/phase9_retrieval_evidence.py`` and again in
``tools/phase10_vocabulary.py``; Phase 13 needed a third, which is the point at
which this repo has agreed to extract rather than paste (D-N2, D-P1).

The return type is a NAMED tuple rather than a bare ``(float, int, int)``.
That is not cosmetic: the Phase 13 commit reported "median 0.95 filled slots"
for a number the tool had computed as a MEAN (D-R4). Positional unpacking is
how a mean gets printed under a median's label; ``summary.mean`` cannot.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence


class Summary(NamedTuple):
    """mean / median / P90 / max of one distribution."""

    mean: float
    median: float
    p90: float
    maximum: float


def percentiles(values: Sequence[float]) -> Summary:
    """Summarize a distribution. Empty input is all zeros.

    ``median`` and ``p90`` are order statistics of the sample -- an actual
    observed value, not an interpolation -- which is what the existing tables
    report and what makes "P90 300" readable as "a real turn hit the cap".
    """
    ordered = sorted(values)
    if not ordered:
        return Summary(0.0, 0.0, 0.0, 0.0)
    return Summary(
        mean=sum(ordered) / len(ordered),
        median=ordered[len(ordered) // 2],
        p90=ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))],
        maximum=ordered[-1],
    )
