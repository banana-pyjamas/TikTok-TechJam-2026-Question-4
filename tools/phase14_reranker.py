"""Phase 14 CP 14.6 -- semantic reranking OFF vs ON, and what the stage did.

An ablation on its own would answer "did the score move" and nothing else,
which is not enough to justify a stage that sits in the middle of every turn.
So this also reports what the reranker DID: which fallback each turn took,
how far it moved the hidden target, and what it cost.

Four questions:

  1. NO-OP        does the flag's OFF position reproduce the committed score
                  EXACTLY? A flag whose OFF arm drifts makes every comparison
                  below meaningless.
  2. SCORE        OFF vs ON, paired McNemar on per-session hit verdicts.
  3. BEHAVIOUR    the CP 14.1 - 14.5 paths, counted over the real dialogue --
                  including how often the stage is a no-op because the
                  shopper volunteered no free text at all.
  4. MOVEMENT     for the turns where the target was inside the rerank
                  window, where did it start and where did it end up? This is
                  the question Phase 13 left for Phase 14: 176 eligible-turn
                  pool hits sit at final rank 11-50, and moving them is the
                  only reason this stage exists.

The CP 14.1 - 14.5 CONTRACTS are unit-tested in tests/test_reranker.py. What
this tool adds is that they hold on live data -- every reranked output is
checked to be a permutation of its input, so "the model cannot invent ASINs"
is verified against the real pipeline and not only against stubs.

Usage:  python3 -m tools.phase14_reranker      (~2 min: two evaluator runs)
"""

from __future__ import annotations

import hashlib
import time
from collections import Counter

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter import reranker
from starter.contracts import Context
from starter.reranker import RERANK_KEY, PoolTermScorer, _evidence_terms
from tools import config_guard
from tools.capture import CapturingAgent
from tools.phase13_dense_gate import first_scoring_turn
from tools.significance import (format_test, hits_by_sample, mcnemar,
                                mttc_given_hit)
from tools.summaries import percentiles

CATALOG = "data/catalog.jsonl"
DATASET = "data/public_set.jsonl"

# The score with USE_SEMANTIC_RERANK OFF: the Phase 12 pipeline exactly. A
# fixed historical reference, NOT the committed score -- the committed score is
# now the ON arm, and it lives in config_guard. Keeping the two apart is the
# whole lesson of phase10/phase11, which each pinned "the committed score" as a
# private literal and then spent two phases raising on their own staleness.
PRE_RERANK_TECHNICAL_SCORE = 0.182258


class PlaceboScorer(PoolTermScorer):
    """The control arm: the same machinery, ranking on the WRONG words.

    Everything is held constant except the one thing under test. Same pool
    vocabulary, same in-pool IDF weighting, same tie-break, same number of
    query terms per turn -- but the terms are drawn deterministically from the
    pool instead of from what the shopper said.

    This exists because the first ON run measured +0.024 TS while the target's
    mean rank got WORSE (21.0 -> 21.9, up on 15 turns and down on 43). A score
    gain whose stated mechanism runs backwards is exactly the shape of a
    perturbation gain: the evaluator stops at the first hit, so ANY reordering
    reshuffles which sessions happen to land a target in the top 10, and on
    200 sessions that can look like a win. If the placebo gains too, the
    lexical signal is doing nothing and the ON arm must not ship.

    Deterministic across processes: ``hashlib``, never ``hash()``, whose string
    seed is randomized per interpreter and would make this unreproducible.
    """

    name = "placebo"

    def query_terms(self, context: Context) -> set[str]:
        real = _evidence_terms(context)
        vocabulary = sorted(self.frequencies)
        if not real or not vocabulary:
            return set()
        seed = f"{context.session_id}:{context.turn}"
        picked: set[str] = set()
        attempt = 0
        while len(picked) < len(real) and attempt < 100 * len(real):
            digest = hashlib.md5(f"{seed}:{attempt}".encode("utf-8")).hexdigest()
            picked.add(vocabulary[int(digest, 16) % len(vocabulary)])
            attempt += 1
        return picked


def make_placebo_builder(original):
    """``build_scorer`` with the placebo substituted for the real scorer.

    Closes over ``original`` rather than reading ``reranker.build_scorer``:
    the arm loop patches that name, so reading it here would call THIS
    function again. The first version did exactly that, recursed until
    RecursionError, and -- because the evaluator swallows a raising turn into
    an empty response -- reported a placebo arm of HR 0.0000 that looked like
    a devastating result instead of a broken one.
    """

    def build(connection, candidates, context):
        real = original(connection, candidates, context)
        if not isinstance(real, PoolTermScorer):
            return real
        placebo = PlaceboScorer(None, real.indexed_terms)
        placebo.pool_size = real.pool_size
        placebo.frequencies = real.frequencies
        return placebo

    return build


def instrument(agent: CapturingAgent, targets: list[str], eligible_from: list[int]):
    """Wrap ``reranker.rerank`` with observation. Read-only.

    Patching the MODULE attribute, which is also how ``agent.respond`` reaches
    it -- so this observes the real call, not a reimplementation of it. The
    wrapper never changes the returned order; a tool that measured a different
    reranker than the one that ships would be worthless.

    Session identity: ``reset`` runs in dataset order, so the first session id
    the wrapper sees belongs to sample 0, and so on. Same bridge every tool in
    this repo uses to get from an opaque uuid back to its sample.
    """
    original = reranker.rerank
    stats = {
        "outcomes": Counter(),
        "invented": 0,
        "not_a_permutation": 0,
        "tail_disturbed": 0,
        "elapsed": [],
        "moved": [],
        "target_before": [],
        "target_after": [],
        "target_into_top10": 0,
        "target_out_of_top10": 0,
        "target_in_window": 0,
    }
    target_of: dict[str, str] = {}
    eligible_of: dict[str, int] = {}

    def counting(result, context, scorer, *args, **kwargs):
        before = [candidate.parent_asin for candidate in result.ranked]
        output = original(result, context, scorer, *args, **kwargs)
        after = [candidate.parent_asin for candidate in output.ranked]

        diagnostics = (context.derived or {}).get(RERANK_KEY) or {}
        stats["outcomes"][diagnostics.get("outcome", "?")] += 1
        stats["invented"] += int(diagnostics.get("invented") or 0)
        stats["elapsed"].append(float(diagnostics.get("elapsed_ms") or 0.0))
        stats["moved"].append(int(diagnostics.get("moved") or 0))

        # CP 14.2 on live data: same multiset in, same multiset out.
        if sorted(before) != sorted(after):
            stats["not_a_permutation"] += 1
        # CP 14.1 on live data: everything past the window is untouched.
        window = int(diagnostics.get("considered") or 0)
        if window and before[window:] != after[window:]:
            stats["tail_disturbed"] += 1

        if context.session_id not in target_of:
            index = len(target_of)
            target_of[context.session_id] = targets[index]
            eligible_of[context.session_id] = eligible_from[index]
        target = target_of[context.session_id]
        if context.turn >= eligible_of[context.session_id] and target in before:
            position_before = before.index(target) + 1
            position_after = after.index(target) + 1
            stats["target_in_window"] += 1
            stats["target_before"].append(position_before)
            stats["target_after"].append(position_after)
            if position_before > 10 >= position_after:
                stats["target_into_top10"] += 1
            elif position_after > 10 >= position_before:
                stats["target_out_of_top10"] += 1
        return output

    reranker.rerank = counting
    return stats, original


def main() -> None:
    config_guard.assert_all_flags_pinned(set(config_guard.COMMITTED_FLAGS))
    config_guard.assert_committed_constants()
    config_guard.assert_committed_flags_match_source()
    config_guard.restore_committed_flags()

    print("building index...", flush=True)
    started = time.time()
    agent = CapturingAgent(CATALOG)
    samples = load_jsonl(DATASET)
    catalog_ids, categories, products = catalog_index(CATALOG)
    print(f"ready in {time.time() - started:.1f}s", flush=True)

    targets = [str(s["ground_truth"]["parent_asin"]) for s in samples]
    eligible_from = [first_scoring_turn(s, products) for s in samples]

    # Three arms. The placebo is not decoration: without it, "ON gained 6
    # sessions" cannot be told apart from "any reordering gains ~6 sessions",
    # because the evaluator stops at the first hit and is therefore sensitive
    # to permutation as such.
    original_build = reranker.build_scorer
    arms = (("OFF", False, original_build),
            ("ON", True, original_build),
            ("PLACEBO", True, make_placebo_builder(original_build)))
    results: dict[str, dict] = {}
    stats: dict[str, dict] = {}

    for label, enabled, builder in arms:
        config_guard.set_flag("reranker", "USE_SEMANTIC_RERANK", enabled)
        reranker.build_scorer = builder
        agent.order.clear()
        agent.captured.clear()
        arm_stats, original_rerank = instrument(agent, targets, eligible_from)
        started = time.time()
        results[label] = evaluate(agent, samples, catalog_ids, categories,
                                  products)
        reranker.rerank = original_rerank
        reranker.build_scorer = original_build
        elapsed = time.time() - started
        stats[label] = arm_stats
        result = results[label]

        # The evaluator turns a raising `respond` into an empty response and
        # carries on, so a totally broken arm reports HR 0.0000 and looks like
        # a finding rather than a crash. `CapturingAgent.respond` records only
        # AFTER the real call returns, which makes the turn count the cheapest
        # honest liveness check there is.
        turn_count = len(agent.captured)
        if not turn_count:
            raise SystemExit(
                f"arm {label!r} captured 0 turns: every respond() raised and "
                "the evaluator swallowed it. Its numbers are a crash, not a "
                "measurement.")
        if enabled and arm_stats["outcomes"] and \
                sum(arm_stats["outcomes"].values()) != turn_count:
            raise SystemExit(
                f"arm {label!r} reranked "
                f"{sum(arm_stats['outcomes'].values())} of {turn_count} turns; "
                "the stage must run on every turn or the comparison is not "
                "controlled.")

        print(f"\n{label:8} USE_SEMANTIC_RERANK={enabled!s:5} "
              f"HR {result['hit_rate_at_10']:.4f} MRR {result['mrr']:.6f} "
              f"MTTC {result['mttc']:.3f} "
              f"TS {result['recommended_technical_score']:.6f} "
              f"({turn_count} turns, {elapsed:.0f}s, "
              f"{1000 * elapsed / turn_count:.1f} ms/turn)", flush=True)
    config_guard.restore_committed_flags()

    off, on, placebo = results["OFF"], results["ON"], results["PLACEBO"]
    stats_on = stats["ON"]
    turns = len(agent.captured)

    # -- 1 -------------------------------------------------------------------
    print("\n1. NO-OP -- is the OFF arm a true no-op?")
    # Two different pins, and conflating them is how phase10 and phase11 went
    # stale for two whole phases. PRE_RERANK is a HISTORICAL reference -- the
    # score of this flag's OFF position, which is the Phase 12 pipeline and
    # never moves again. COMMITTED_TECHNICAL_SCORE is the score of what ships,
    # which is now the ON arm. (phase12_popularity pins its OFF arm the same
    # way, for the same reason.)
    committed = config_guard.COMMITTED_TECHNICAL_SCORE
    checks = (
        ("OFF reproduces the pre-Phase-14 pipeline", off, PRE_RERANK_TECHNICAL_SCORE),
        ("ON reproduces the committed score       ", on, committed),
    )
    failed = []
    for label, result, expected in checks:
        actual = result["recommended_technical_score"]
        exact = abs(actual - expected) <= 1e-9
        print(f"   {label} {expected}   {'PASS' if exact else f'FAIL ({actual})'}")
        if not exact:
            failed.append(label.strip())
    if failed:
        raise SystemExit(
            "these arms no longer reproduce their pinned scores, so nothing "
            "below is a controlled comparison: " + "; ".join(failed))
    print("   With the flag OFF the reranker is not imported into the hot path,")
    print("   the ranked list is not deepened, and no scorer is built -- the")
    print("   turn is byte-for-byte the Phase 12 pipeline.")

    # -- 2 -------------------------------------------------------------------
    print("\n2. SCORE -- OFF vs ON, against a placebo that reorders on the")
    print("   WRONG words (same machinery, same term count, same weighting)")
    for label in ("OFF", "ON", "PLACEBO"):
        result = results[label]
        conditional = mttc_given_hit(result)
        print(f"   {label:8}HR {result['hit_rate_at_10']:.4f}  "
              f"MRR {result['mrr']:.6f}  MTTC {result['mttc']:.3f}  "
              f"MTTC|hit {0.0 if conditional is None else conditional:.3f}"
              f"  TS {result['recommended_technical_score']:.6f}"
              f"  {result['recommended_technical_score'] - off['recommended_technical_score']:+.6f}")
    print()
    print("   " + format_test("ON vs OFF",
                              mcnemar(hits_by_sample(off), hits_by_sample(on))))
    print("   " + format_test("PLACEBO vs OFF",
                              mcnemar(hits_by_sample(off), hits_by_sample(placebo))))
    print("   " + format_test("ON vs PLACEBO",
                              mcnemar(hits_by_sample(placebo), hits_by_sample(on))))
    print("\n   HOW TO READ THIS. The evaluator stops a session at its first")
    print("   hit, so ANY reordering of the top 50 reshuffles which sessions")
    print("   happen to land a target in the top 10. If the placebo gains as")
    print("   much as ON, the shopper's words carried nothing and the ON gain")
    print("   is perturbation. Only 'ON vs PLACEBO' isolates the signal.")
    print(f"\n   {'scenario':18}{'OFF':>10}{'ON':>10}{'PLACEBO':>10}"
          f"{'ON-OFF':>10}")
    for scenario in ("buying", "browsing", "intent_override", "boundary"):
        a = off["scenario_metrics"][scenario]["hit_rate_at_10"]
        b = on["scenario_metrics"][scenario]["hit_rate_at_10"]
        c = placebo["scenario_metrics"][scenario]["hit_rate_at_10"]
        print(f"   {scenario:18}{a:>10.4f}{b:>10.4f}{c:>10.4f}{b - a:>+10.4f}")

    # -- 3 -------------------------------------------------------------------
    assert stats_on is not None
    outcomes = stats_on["outcomes"]
    calls = sum(outcomes.values())
    print(f"\n3. BEHAVIOUR -- which path each of the {calls} reranked turns took")
    for outcome in reranker.OUTCOMES:
        count = outcomes[outcome]
        note = {
            "offline": "   <- CP 14.5: no scorer at all (no encoder AND no free text)",
            "malformed": "   <- CP 14.3",
            "error": "   <- CP 14.3",
            "timeout": "   <- CP 14.4",
            "identity": "   <- scorer ran and changed nothing",
            "applied": "   <- scorer reordered the window",
        }.get(outcome, "")
        print(f"   {outcome:12}{count:>6}{count / max(calls, 1):>9.1%}{note}")
    unexpected = set(outcomes) - set(reranker.OUTCOMES)
    print(f"   undeclared outcomes            "
          f"{'none  PASS' if not unexpected else sorted(unexpected)}")
    print(f"\n   CP 14.5, stated precisely. `load_encoder_scorer()` returns "
          f"{reranker.load_encoder_scorer()!r} on")
    print("   EVERY one of these turns -- the semantic scorer is unavailable "
          "100% of\n   the time, which is why the lexical one is what ran at "
          "all. The 'offline'\n   row above is the narrower case where there "
          "is no scorer of ANY kind,\n   which needs the shopper to have "
          "volunteered no free text either.")

    print("\n   contract checks, on live data rather than on stubs")
    print(f"   CP 14.1  tail outside the window disturbed   "
          f"{stats_on['tail_disturbed']:>5}  "
          f"{'PASS' if not stats_on['tail_disturbed'] else 'FAIL'}")
    print(f"   CP 14.2  outputs that were not a permutation {stats_on['not_a_permutation']:>5}  "
          f"{'PASS' if not stats_on['not_a_permutation'] else 'FAIL'}")
    print(f"   CP 14.2  invented ASINs discarded            "
          f"{stats_on['invented']:>5}")
    print(f"   CP 14.4  budget overruns                     "
          f"{outcomes['timeout']:>5}  "
          f"{'PASS' if not outcomes['timeout'] else 'the ON numbers are not reproducible'}")
    cost = percentiles(stats_on["elapsed"])
    print(f"   scorer cost per turn   mean {cost.mean:.3f} ms   "
          f"P90 {cost.p90:.3f} ms   max {cost.maximum:.3f} ms   "
          f"(budget {reranker.RERANK_BUDGET_MS:.0f} ms)")

    # -- 4 -------------------------------------------------------------------
    print("\n4. MOVEMENT -- what happened to the hidden target")
    in_window = stats_on["target_in_window"]
    print(f"   eligible turns with the target inside the rerank window "
          f"{in_window:>5}")
    if in_window:
        before = percentiles(stats_on["target_before"])
        after = percentiles(stats_on["target_after"])
        print(f"   target rank BEFORE rerank   mean {before.mean:>7.1f}   "
              f"median {before.median:>5.0f}   P90 {before.p90:>5.0f}")
        print(f"   target rank AFTER  rerank   mean {after.mean:>7.1f}   "
              f"median {after.median:>5.0f}   P90 {after.p90:>5.0f}")
        improved = sum(1 for b, a in zip(stats_on["target_before"],
                                         stats_on["target_after"]) if a < b)
        worsened = sum(1 for b, a in zip(stats_on["target_before"],
                                         stats_on["target_after"]) if a > b)
        print(f"   target moved UP {improved}, DOWN {worsened}, "
              f"unchanged {in_window - improved - worsened}")
        print(f"   crossed INTO the top 10   {stats_on['target_into_top10']:>5}")
        print(f"   pushed OUT of the top 10  {stats_on['target_out_of_top10']:>5}")
        print("   Turn-level, so one session can contribute several rows; the")
        print("   session-level consequence is the McNemar test in (2), which is")
        print("   the number that decides the flag.")

    # -- 5 -------------------------------------------------------------------
    print("\n5. ROBUSTNESS -- does the gain survive a different window?")
    print("   A real effect should not depend on RERANK_TOP_N being exactly")
    print(f"   {reranker.RERANK_TOP_N}; a lucky permutation would. top_n=10 is a "
          "control rather than a\n   setting: the window is then the response "
          "itself, so the stage can only\n   reorder what was already being "
          "shown and can pull nothing up from below.")
    print(f"   {'top_n':>8}{'HR':>10}{'MRR':>12}{'TS':>12}{'vs OFF':>11}"
          "   McNemar vs OFF")
    committed_top_n = reranker.RERANK_TOP_N
    config_guard.set_flag("reranker", "USE_SEMANTIC_RERANK", True)
    try:
        for top_n in (10, 20, 50, 100, 200):
            reranker.RERANK_TOP_N = top_n
            agent.order.clear()
            agent.captured.clear()
            swept = evaluate(agent, samples, catalog_ids, categories, products)
            test = mcnemar(hits_by_sample(off), hits_by_sample(swept))
            print(f"   {top_n:>8}{swept['hit_rate_at_10']:>10.4f}"
                  f"{swept['mrr']:>12.6f}"
                  f"{swept['recommended_technical_score']:>12.6f}"
                  f"{swept['recommended_technical_score'] - off['recommended_technical_score']:>+11.6f}"
                  f"   {test['gained']}/{test['lost']} discordant, "
                  f"p = {test['p']:.4f}", flush=True)
    finally:
        reranker.RERANK_TOP_N = committed_top_n
        config_guard.restore_committed_flags()
    print("   Five comparisons against one baseline, so read the family rather")
    print("   than any single p-value (D-P3): what carries the conclusion is")
    print("   that the control is null and the direction holds across the")
    print("   plateau, not that a particular row cleared 0.05.")

    print(f"\nturns: {turns}   config: {config_guard.describe()}")


if __name__ == "__main__":
    main()
