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
"""

from __future__ import annotations

from math import comb


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
