"""Phase 9 — mode-classifier accuracy on the live dialogue.

Commit 2b37007 claimed "100% accuracy" from a throwaway script. Commit 66d127c
called that out, then reported "LIVE PER-TURN MODE ACCURACY (committed
methodology this time) ... intent_override 6.7%, OVERALL 86.0%" -- and did not
commit the methodology either. Two independent reproductions of "the same"
measurement came back 56.7% and 57.2% on the override cell. Three numbers, no
script. This is the script.

Phase 7 set the precedent (``tools/phase7_ablation.py``): a number that is
quoted is a number anyone can re-run.

WHAT IS MEASURED

The mode ``starter.strategy.classify_mode`` returns, on every turn of the REAL
dialogue. The loop is not reimplemented here -- ``evaluator.local_evaluator``
drives it, and the agent is subclassed only to record what the classifier saw
after ``update_state`` ran. So the transcript measured is exactly the shipped
agent's transcript, including the fact that the dialogue stops at the first hit
and that the agent never asks a question, so the customer's replies are
non-answers from turn 2 on.

GROUND TRUTH

Read off the harness's own opening line (``local_evaluator.initial_message``),
which is the only place the scenario is expressed as shopper language:

    buying            "A key requirement is: X."          -> buying
    intent_override   "{category}. {a concrete detail}"   -> buying
    browsing          "but I'm still exploring"           -> browsing
    boundary          "but I'm still exploring"           -> browsing

``boundary`` opens with the browsing line, so it is scored as browsing; its
distinguishing behaviour is how it answers a question, not how it opens.

THREE AGGREGATIONS, ALL PRINTED

The reason the previous numbers could not be compared is that "accuracy" was
never defined. All three definitions are printed, so a reader can see which one
any given claim refers to instead of guessing:

    per-turn       correct turns / all turns          (turns are the unit)
    turn 1 only    correct openings / sessions        (what 2b37007 measured)
    all-turns      sessions where EVERY turn is right (strictest)

Usage:  python3 -m tools.phase9_mode_accuracy
        python3 -m tools.phase9_mode_accuracy --without-fallback

``--without-fallback`` disables the concrete-unslotted-detail rule, which
reproduces the classifier as 66d127c shipped it. It is here so the disagreement
in that commit's message is settled by measurement rather than argument: run it
both ways and the 86.0% / 6.7% pair turns up in the turn-1 column, not the
per-turn one. That is the whole finding -- the numbers were right and the
column heading was wrong.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter import strategy as strategy_module
from starter.agent import Agent
from starter.contracts import Context
from starter.strategy import BROWSING, BUYING, classify_mode_with_reason
from tools import config_guard

CATALOG = "data/catalog.jsonl"
DATASET = "data/public_set.jsonl"

# Derived from local_evaluator.initial_message -- see the module docstring.
EXPECTED_MODE = {
    "buying": BUYING,
    "intent_override": BUYING,
    "browsing": BROWSING,
    "boundary": BROWSING,
}


class ModeRecordingAgent(Agent):
    """The shipped agent, plus a record of the mode on every turn.

    ``respond`` is not reimplemented: it delegates, then classifies the state
    the real turn left behind. ``classify_mode`` is pure and reads only the
    context, so observing it here cannot change what the agent does -- the
    scores this run produces must match a plain evaluator run exactly.
    """

    def __init__(self, catalog_path: str) -> None:
        super().__init__(catalog_path)
        self.order: list[str] = []
        self.observations: list[dict] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        super().reset(session_id, user_profile)
        self.order.append(session_id)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        response = super().respond(session_id, user_message, turn, top_k)
        context = Context(
            session_id=session_id,
            turn=turn,
            user_message=user_message,
            state=self._states[session_id],
        )
        mode, reason = classify_mode_with_reason(context)
        self.observations.append({
            "session_id": session_id,
            "turn": turn,
            "message": user_message,
            "mode": mode,
            "reason": reason,
        })
        return response


def _rate(correct: int, total: int) -> str:
    return f"{100.0 * correct / total:6.1f}%  ({correct}/{total})" if total else "     --"


def main() -> None:
    config_guard.assert_all_flags_pinned(set(config_guard.COMMITTED_FLAGS))
    config_guard.assert_committed_constants()
    config_guard.restore_committed_flags()

    if "--without-fallback" in sys.argv:
        strategy_module._has_concrete_evidence = lambda context: False
        print("!! concrete-unslotted-detail rule DISABLED "
              "(reproducing 66d127c's classifier)\n")

    print("building index...", flush=True)
    started = time.time()
    agent = ModeRecordingAgent(CATALOG)
    samples = load_jsonl(DATASET)
    catalog_ids, categories, products = catalog_index(CATALOG)
    print(f"ready in {time.time() - started:.1f}s", flush=True)

    started = time.time()
    result = evaluate(agent, samples, catalog_ids, categories, products)
    print(f"live dialogue: {len(agent.observations)} turns over "
          f"{len(agent.order)} sessions in {time.time() - started:.0f}s")
    print(f"score during this run: HR {result['hit_rate_at_10']:.4f} "
          f"TS {result['recommended_technical_score']:.6f} "
          "(must equal a plain evaluator run -- observing is read-only)")

    scenario_of = {
        session_id: str(sample["scenario_type"])
        for session_id, sample in zip(agent.order, samples)
    }
    by_session: dict[str, list[dict]] = defaultdict(list)
    for observation in agent.observations:
        by_session[observation["session_id"]].append(observation)

    scenarios = sorted(EXPECTED_MODE)
    turns_total: dict[str, int] = defaultdict(int)
    turns_right: dict[str, int] = defaultdict(int)
    first_total: dict[str, int] = defaultdict(int)
    first_right: dict[str, int] = defaultdict(int)
    all_total: dict[str, int] = defaultdict(int)
    all_right: dict[str, int] = defaultdict(int)

    for session_id, observations in by_session.items():
        scenario = scenario_of[session_id]
        expected = EXPECTED_MODE[scenario]
        observations.sort(key=lambda o: o["turn"])
        right = [o["mode"] == expected for o in observations]
        turns_total[scenario] += len(right)
        turns_right[scenario] += sum(right)
        first_total[scenario] += 1
        first_right[scenario] += int(right[0])
        all_total[scenario] += 1
        all_right[scenario] += int(all(right))

    print(f"\n{'scenario':18}{'expected':>10}{'per-turn':>20}"
          f"{'turn 1 only':>20}{'all-turns':>20}")
    for scenario in scenarios:
        print(f"{scenario:18}{EXPECTED_MODE[scenario]:>10}"
              f"{_rate(turns_right[scenario], turns_total[scenario]):>20}"
              f"{_rate(first_right[scenario], first_total[scenario]):>20}"
              f"{_rate(all_right[scenario], all_total[scenario]):>20}")
    print(f"{'OVERALL':18}{'':>10}"
          f"{_rate(sum(turns_right.values()), sum(turns_total.values())):>20}"
          f"{_rate(sum(first_right.values()), sum(first_total.values())):>20}"
          f"{_rate(sum(all_right.values()), sum(all_total.values())):>20}")

    # Turn 1 is the only turn that carries shopper information: the agent never
    # asks a question, so every later customer reply is a harness non-answer.
    # Splitting it out shows how much of any "accuracy" figure is just that.
    later_total = sum(turns_total.values()) - sum(first_total.values())
    later_right = sum(turns_right.values()) - sum(first_right.values())
    print(f"\nturn 1 vs the rest: turn 1 "
          f"{_rate(sum(first_right.values()), sum(first_total.values()))}   "
          f"turns 2+ {_rate(later_right, later_total)}")
    print("  turns 2+ carry no shopper information -- the agent asks nothing, "
          "so every\n  reply is a harness non-answer. They test stability, "
          "not classification.")

    misses = [
        (scenario_of[o["session_id"]], o["turn"], o["mode"], o["message"])
        for observations in by_session.values() for o in observations
        if o["mode"] != EXPECTED_MODE[scenario_of[o["session_id"]]]
    ]
    print(f"\nfirst 10 of {len(misses)} misclassified turns:")
    for scenario, turn, mode, message in misses[:10]:
        print(f"  {scenario:16} t{turn:<3} got {mode:9} {message[:64]!r}")

    # WHICH rule earns the accuracy, and over how many distinct openings. A
    # single percentage hides the difference between a classifier that
    # generalises and one that covers a handful of templates; on this harness
    # it is the latter, and the table below is what says so.
    print("\nwhich rule decided (turn 1 only -- the only turn carrying "
          "shopper language):")
    rules: dict[tuple[str, str], int] = defaultdict(int)
    openings: dict[str, set[str]] = defaultdict(set)
    for session_id, observations in by_session.items():
        scenario = scenario_of[session_id]
        first = observations[0]
        rules[(scenario, first["reason"])] += 1
        openings[scenario].add(first["message"])
    for scenario in scenarios:
        for (name, reason), count in sorted(rules.items()):
            if name == scenario:
                print(f"  {scenario:18}{reason:28}{count:>5}")
    distinct = sum(len(v) for v in openings.values())
    print(f"\nthe public set contains {distinct} distinct opening messages "
          f"built from {len(scenarios)} templates\n"
          "(local_evaluator.initial_message), and every non-opening turn is a "
          "harness\nnon-answer. A high score here is TEMPLATE COVERAGE, not "
          "evidence of\ngeneralization -- read it as 'no known case is "
          "misread', nothing stronger.")

    print(f"\nconfig: {config_guard.describe()}")
    print("note: the mode gates nothing in the shipped agent -- it is measured "
          "here, not tuned\n      against. Phase 15 is the first consumer.")


if __name__ == "__main__":
    main()
