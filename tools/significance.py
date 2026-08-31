"""Paired significance for two evaluator runs over the same sessions.

The public set is 200 sessions, so one flipped session is worth ~0.005 hit
rate and ~0.01 TS. A difference of a few thousandths is a difference of one or
two sessions, and a point estimate alone cannot tell that apart from noise.
Every score comparison this repo BANKS should therefore be reported with the
discordant counts that produced it.

McNemar's exact test is the right one here: the two runs see the SAME sessions,
so the pairing is real and only the sessions that changed verdict carry
information. Sessions both runs hit, or both miss, are uninformative by
construction and are excluded -- that is the whole point of the test.

Exact binomial rather than the chi-square approximation because the discordant
counts here are single digits, where the approximation is not trustworthy.

WHY THERE ARE THREE TESTS HERE AND NOT ONE
------------------------------------------
McNemar answers ONE question: did the set of sessions that HIT change? It is
blind to everything else the score is made of. TS = 0.5*HR + 0.3*MRR +
0.2*efficiency, so a change that moves every hit from rank 7 to rank 2, or
finds the same targets three turns sooner, is invisible to it -- discordant
count zero, p = 1.0000, "no verdict" -- while moving TS substantially.

That blindness has now mispriced a verdict in two consecutive phases (D
Phase 15 review, item 2). The project's own tuning protocol asks for a paired
permutation over per-session COMPOSITES, which is the test that sees all
three terms; it was written in a vault note, which is a place no checkpoint
reads, so every phase reached for the test that was actually built here.

So all three now live here and are the default:

    mcnemar          did the HIT SET change? Exact, and the right test for a
                     binary verdict. Keep quoting it -- a hit is what the
                     benchmark ultimately rewards.
    paired_permutation   did the SCORE change? Exact-in-the-limit over
                     per-session composites, sensitive to rank and turn
                     movement that McNemar cannot see.
    bootstrap_ci     how big is the change, with what uncertainty? A p-value
                     is not an effect size and this repo quotes effect sizes
                     constantly.

Report all three for anything banked. They disagree in informative ways: a
permutation that clears while McNemar does not means the phase improved ranks
without converting misses, which is a real and different finding.

Both new tests are seeded (``RANDOM_SEED``) and therefore reproducible run to
run -- the Phase 14 placebo shipped a "deterministic" control seeded from a
per-run uuid, and nothing in this file will repeat that.
"""

from __future__ import annotations

import random
import statistics
from math import comb

# Fixed so every run of every tool reports the same p-value and the same
# interval. See the module docstring.
RANDOM_SEED = 20260831

# Resamples. 10k gives a p-value resolution of 1e-4, which is finer than any
# claim this repo makes, and costs milliseconds on 200 sessions.
PERMUTATIONS = 10000
BOOTSTRAPS = 10000


def hits_by_sample(result: dict) -> dict[str, bool]:
    """``sample_id -> hit`` for one ``evaluate()`` result."""
    return {str(s["sample_id"]): bool(s["hit"]) for s in result["sessions"]}


def mcnemar(before: dict[str, bool], after: dict[str, bool]) -> dict:
    """Paired comparison of two runs' per-session hit verdicts.

    Returns the discordant counts and the exact two-sided p-value.

    ``gained``  sessions the second run hits and the first does not
    ``lost``    sessions the first run hits and the second does not
    ``p``       exact two-sided binomial p over the discordant pairs
    """
    shared = sorted(set(before) & set(after))
    gained = sum(1 for key in shared if after[key] and not before[key])
    lost = sum(1 for key in shared if before[key] and not after[key])
    total = gained + lost
    if total == 0:
        p = 1.0
    else:
        smaller = min(gained, lost)
        tail = sum(comb(total, i) for i in range(smaller + 1)) * (0.5 ** total)
        p = min(1.0, 2.0 * tail)
    return {
        "n": len(shared),
        "gained": gained,
        "lost": lost,
        "discordant": total,
        "net": gained - lost,
        "p": p,
    }


def verdict(test: dict, alpha: float = 0.05) -> str:
    """The word this repo is allowed to use about a difference.

    Deliberately blunt. "no verdict" is not a soft "probably real": on this
    set it means the comparison does not establish the difference in EITHER
    direction, and the number must not be banked as a gain.

    Per-comparison, with no multiplicity adjustment. When a caller prints a
    family of related tests -- the same paired data at several thresholds, say
    -- several "established" labels are ONE finding seen from several angles,
    not several findings, and the caller must say so (D-P3). What survives
    multiplicity is a consistent direction across the family; a single label
    that clears alpha in isolation does not.
    """
    if test["discordant"] == 0:
        return "identical (no session changed verdict)"
    if test["p"] < alpha:
        return "established" if test["net"] > 0 else "established (negative)"
    return "no verdict"


def format_test(label: str, test: dict) -> str:
    return (
        f"{label:34}{test['net']:>+6}  "
        f"{test['gained']}/{test['lost']} discordant   "
        f"p = {test['p']:.4f}   {verdict(test)}"
    )


def mttc_given_hit(result: dict) -> float | None:
    """Mean turns-to-first-hit over the sessions that HIT.

    Reported alongside the official MTTC, which charges a miss as
    ``MAX_TURNS + 1`` and is therefore dominated by the hit rate: a run can
    "improve" MTTC purely by hitting more, or worsen it by hitting sessions
    that take longer. This is the conditional number, which moves only when
    the agent gets slower or faster on the sessions it actually solves
    (D-F8, kept open across two commits).
    """
    turns = [s["first_hit_turn"] for s in result["sessions"] if s["hit"]]
    return sum(turns) / len(turns) if turns else None


# The evaluator's own weights (local_evaluator.evaluate). Duplicated here
# rather than imported because this module must stay importable without the
# evaluator; tests/test_summaries.py asserts the two agree.
_W_HIT, _W_MRR, _W_EFF = 0.50, 0.30, 0.20
_MAX_TURNS = 10


def session_composite(session: dict) -> float:
    """One session's contribution to TechnicalScore, on the same scale as TS.

    ``0.5*hit + 0.3*reciprocal_rank + 0.2*efficiency``, where efficiency uses
    the evaluator's own ``(11 - turns) / 10`` with a miss charged at
    ``MAX_TURNS + 1``. Averaging this over the sessions reproduces the
    reported TS exactly, which is what makes a paired test over it a test
    about the score rather than about a proxy for it.
    """
    hit = 1.0 if session.get("hit") else 0.0
    reciprocal = float(session.get("reciprocal_rank") or 0.0)
    turn = session.get("first_hit_turn")
    turns = float(_MAX_TURNS + 1 if turn is None else turn)
    efficiency = max(0.0, min(1.0, (11.0 - turns) / 10.0))
    return _W_HIT * hit + _W_MRR * reciprocal + _W_EFF * efficiency


def composites_by_sample(result: dict) -> dict[str, float]:
    """``sample_id -> session_composite`` for one ``evaluate()`` result."""
    return {str(s["sample_id"]): session_composite(s)
            for s in result["sessions"]}


def _paired(before: dict[str, float], after: dict[str, float]) -> list[float]:
    shared = sorted(set(before) & set(after))
    return [after[key] - before[key] for key in shared]


def paired_permutation(before: dict[str, float], after: dict[str, float],
                       rounds: int = PERMUTATIONS) -> dict:
    """Exact-in-the-limit paired permutation over per-session composites.

    The null is "the sign of each session's change is arbitrary", which is
    the right null for a paired design: it holds the sessions fixed and asks
    whether the DIRECTION of the differences could have come out this way by
    chance. Sessions whose composite did not move contribute a zero and are
    carried through the permutation rather than dropped -- unlike McNemar,
    where a session both runs hit is genuinely uninformative, a session whose
    score did not move IS information about the effect's consistency.

    The p-value is ``(1 + #{|permuted mean| >= |observed mean|}) / (1 + n)``.
    The +1 on both sides is not a rounding nicety: it keeps the estimate from
    ever returning exactly 0.0, which would claim more certainty than 10k
    resamples can support.
    """
    differences = _paired(before, after)
    if not differences:
        return {"n": 0, "mean": 0.0, "p": 1.0, "moved": 0}
    observed = statistics.fmean(differences)
    rng = random.Random(RANDOM_SEED)
    extreme = 0
    for _ in range(rounds):
        total = 0.0
        for difference in differences:
            total += difference if rng.random() < 0.5 else -difference
        if abs(total / len(differences)) >= abs(observed) - 1e-15:
            extreme += 1
    return {
        "n": len(differences),
        "mean": observed,
        "moved": sum(1 for d in differences if d != 0.0),
        "p": (1 + extreme) / (1 + rounds),
    }


def bootstrap_ci(before: dict[str, float], after: dict[str, float],
                 rounds: int = BOOTSTRAPS, alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI for the mean paired difference.

    Resamples SESSIONS, not differences-within-a-session: the pairing is the
    design and breaking it would answer a question nobody asked. The interval
    is the effect size with its uncertainty, which is what a reader needs to
    decide whether a "+0.0068" is worth taking and which no p-value supplies.
    """
    differences = _paired(before, after)
    if not differences:
        return {"n": 0, "mean": 0.0, "low": 0.0, "high": 0.0}
    rng = random.Random(RANDOM_SEED)
    count = len(differences)
    means = sorted(
        statistics.fmean(rng.choices(differences, k=count))
        for _ in range(rounds)
    )
    low = means[int((alpha / 2) * rounds)]
    high = means[min(rounds - 1, int((1 - alpha / 2) * rounds))]
    return {"n": count, "mean": statistics.fmean(differences),
            "low": low, "high": high}


def format_composite(label: str, before: dict[str, float],
                     after: dict[str, float], alpha: float = 0.05) -> str:
    """One line carrying the permutation p AND the bootstrap interval.

    Both, always. A p-value without an effect size invites "established" to
    be read as "large", and an interval without a p invites the reverse.
    """
    test = paired_permutation(before, after)
    interval = bootstrap_ci(before, after)
    call = "established" if test["p"] < alpha else "no verdict"
    if test["p"] < alpha and test["mean"] < 0:
        call = "established (negative)"
    return (
        f"{label:34}{test['mean']:>+9.6f}  "
        f"95% CI [{interval['low']:+.6f}, {interval['high']:+.6f}]  "
        f"{test['moved']}/{test['n']} moved   "
        f"p = {test['p']:.4f}   {call}"
    )
