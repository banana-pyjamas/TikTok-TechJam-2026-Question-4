"""Phase 16 final config freeze -- the RERANK_TOP_N depth gate.

The last open decision in the roadmap. Phase 14 chose 50 a priori, before any
sweep existed, from Phase 13's measured rank distribution. Every sweep since
has shown a deeper window scoring better, and each time the value was left
alone because reading a maximum off the same 200 sessions the score is then
reported on is how a benchmark gets fitted. That refusal is only defensible
while the alternative is *unexamined*, and this tool examines it.

WHAT WOULD MAKE 200 A REAL CHOICE RATHER THAN A PEAK

Three things, and the score is the least of them:

  a MECHANISM     8 of the 25 in-pool ranking failures are targets the top-50
                  window truncates. If depth is the cause, those specific 8
                  recover at 200 -- a prediction about NAMED sessions that can
                  be wrong, not a correlation.
  a COST          the stage has a budget (CP 14.4). A deeper window that
                  overran it would fall back silently, and the "gain" would
                  be an artifact of the fallback rather than of ranking.
  NO REGRESSION   not on any scenario, not on the adversarial contracts, and
                  not on the sessions already being won.

Everything below is reported for all three depths so the decision is legible
rather than asserted. Reuses ``phase15_recovery.RecoveryProbe`` instead of
growing a second harness: the pool, the ranked list and the per-turn
diagnostics it already observes are exactly what a depth question needs.

Usage:  python3 -m tools.phase16_depth      (~4 min: three evaluator runs)
"""

from __future__ import annotations

import statistics
import time
from collections import Counter

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter import reranker
from starter.reranker import RERANK_KEY
from tools import config_guard
from tools.phase13_dense_gate import first_scoring_turn
from tools.phase15_recovery import RecoveryProbe, _position
from tools.significance import (composites_by_sample, format_composite,
                                format_test, hits_by_sample, mcnemar)

CATALOG = "data/catalog.jsonl"
DATASET = "data/public_set.jsonl"
DEPTHS = (50, 100, 200)


def _percentiles(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(0.9 * len(ordered)))
    return (statistics.fmean(ordered), ordered[len(ordered) // 2],
            ordered[index])


def measure(depth, agent, samples, catalog_ids, categories, products):
    """One full evaluator run at ``depth``, with everything the gate needs."""
    reranker.RERANK_TOP_N = depth
    agent.records.clear()
    agent.order.clear()
    agent.fired.clear()
    agent.install()
    started = time.time()
    try:
        result = evaluate(agent, samples, catalog_ids, categories, products)
    finally:
        agent.uninstall()
    elapsed = time.time() - started
    if not agent.fired["response"]:
        raise SystemExit(f"depth {depth}: the response probe never fired")

    # session_id -> sample_id. The evaluator regenerates session ids as fresh
    # uuid4s on EVERY run, so comparing failure sets across depths by session
    # id compares two disjoint sets and reports "all recovered, all new".
    # The first version of this tool did exactly that. `reset` runs in
    # dataset order, which is the bridge every tool here uses.
    sample_of = {s: str(sample["sample_id"])
                 for s, sample in zip(agent.order, samples)}
    target_of = {s: str(sample["ground_truth"]["parent_asin"])
                 for s, sample in zip(agent.order, samples)}
    eligible_of = {s: first_scoring_turn(sample, products)
                   for s, sample in zip(agent.order, samples)}
    hit_of = {str(s["sample_id"]): bool(s["hit"]) for s in result["sessions"]}
    hit_by_session = {s: hit_of[str(sample["sample_id"])]
                      for s, sample in zip(agent.order, samples)}

    by_session: dict[str, list[dict]] = {}
    order_ms: list[float] = []
    outcomes: Counter = Counter()
    for record in agent.records:
        session = record["session_id"]
        target = target_of[session]
        record["eligible"] = record["turn"] >= eligible_of[session]
        record["pool_position"] = _position(record["pool"], target)
        record["final_rank"] = _position(record["ranked"], target)
        record["in_top10"] = (record["final_rank"] is not None
                              and record["final_rank"] <= 10)
        diagnostics = (getattr(record["context"], "derived", None)
                       or {}).get(RERANK_KEY) or {}
        order_ms.append(float(diagnostics.get("elapsed_ms") or 0.0))
        outcomes[diagnostics.get("outcome", "?")] += 1
        by_session.setdefault(session, []).append(record)
    for records in by_session.values():
        records.sort(key=lambda item: item["turn"])

    def first(records, predicate):
        for record in records:
            if record["eligible"] and predicate(record):
                return record["turn"]
        return None

    sessions = sorted(by_session)
    in_pool = {s: first(by_session[s],
                        lambda r: r["pool_position"] is not None)
               for s in sessions}
    top10 = {s: first(by_session[s], lambda r: r["in_top10"])
             for s in sessions}
    latency: Counter = Counter()
    for session in sessions:
        if in_pool[session] is None:
            latency["never in pool"] += 1
        elif top10[session] is None:
            latency["never"] += 1
        else:
            delta = top10[session] - in_pool[session]
            latency["same turn" if delta <= 0 else "+1" if delta == 1
                    else "+2" if delta == 2 else "3+"] += 1
    downstream = [top10[s] - in_pool[s] for s in sessions
                  if top10[s] is not None and in_pool[s] is not None]

    ranks = [s["best_rank"] for s in result["sessions"] if s["best_rank"]]
    return {
        "depth": depth, "result": result, "seconds": elapsed,
        "rank1": sum(1 for r in ranks if r == 1),
        "top3": sum(1 for r in ranks if r <= 3),
        "top5": sum(1 for r in ranks if r <= 5),
        "top10": sum(1 for r in ranks if r <= 10),
        "retrieval_failures": {sample_of[s] for s in sessions
                               if not hit_by_session[s] and in_pool[s] is None},
        "ranking_failures": {sample_of[s] for s in sessions
                             if not hit_by_session[s] and in_pool[s] is not None},
        # For each session, the best pool position the target ever reached on
        # a scoring-eligible turn. This is what decides whether a deeper
        # window could even SEE it: a target that never got past pool
        # position 50 was inside the old window all along, and its loss is
        # not a truncation.
        "best_pool_position": {
            sample_of[s]: min(
                (r["pool_position"] for r in by_session[s]
                 if r["eligible"] and r["pool_position"] is not None),
                default=None)
            for s in sessions},
        "recall300": sum(1 for s in sessions if in_pool[s] is not None),
        "latency": latency, "downstream": downstream,
        "order_ms": order_ms, "outcomes": outcomes,
        "turns": len(agent.records),
    }


def main() -> None:
    config_guard.assert_everything()
    print("building index...", flush=True)
    started = time.time()
    agent = RecoveryProbe(CATALOG)
    samples = load_jsonl(DATASET)
    catalog_ids, categories, products = catalog_index(CATALOG)
    print(f"ready in {time.time() - started:.1f}s\n", flush=True)

    committed = reranker.RERANK_TOP_N
    runs = {}
    try:
        for depth in DEPTHS:
            runs[depth] = measure(depth, agent, samples, catalog_ids,
                                  categories, products)
            r = runs[depth]["result"]
            print(f"   depth {depth:>3}   HR {r['hit_rate_at_10']:.4f}  "
                  f"MRR {r['mrr']:.6f}  MTTC {r['mttc']:.3f}  "
                  f"eff {r['efficiency']:.4f}  TS "
                  f"{r['recommended_technical_score']:.6f}  "
                  f"({runs[depth]['seconds']:.0f}s)", flush=True)
    finally:
        reranker.RERANK_TOP_N = committed
        config_guard.assert_committed_constants()

    base = runs[50]
    print("\n" + "=" * 74)
    print("1. SCORE AND RANK QUALITY")
    print("=" * 74)
    print(f"   {'depth':>6}{'HR@10':>9}{'MRR':>11}{'MTTC':>8}{'eff':>8}"
          f"{'TS':>11}{'vs 50':>11}{'R1':>5}{'T3':>5}{'T5':>5}{'T10':>5}")
    for depth in DEPTHS:
        run = runs[depth]
        r = run["result"]
        print(f"   {depth:>6}{r['hit_rate_at_10']:>9.4f}{r['mrr']:>11.6f}"
              f"{r['mttc']:>8.3f}{r['efficiency']:>8.4f}"
              f"{r['recommended_technical_score']:>11.6f}"
              f"{r['recommended_technical_score'] - base['result']['recommended_technical_score']:>+11.6f}"
              f"{run['rank1']:>5}{run['top3']:>5}{run['top5']:>5}"
              f"{run['top10']:>5}")
    print()
    for depth in (100, 200):
        print("   " + format_test(f"depth {depth} vs 50 (hits)",
                                  mcnemar(hits_by_sample(base["result"]),
                                          hits_by_sample(runs[depth]["result"]))))
        print("   " + format_composite(
            f"depth {depth} vs 50 (score)",
            composites_by_sample(base["result"]),
            composites_by_sample(runs[depth]["result"])))

    print("\n" + "=" * 74)
    print("2. SCENARIO BREAKDOWN -- a regression anywhere disqualifies")
    print("=" * 74)
    print(f"   {'scenario':18}" + "".join(f"{'@' + str(d):>12}" for d in DEPTHS)
          + f"{'200 vs 50':>12}")
    regressions = []
    for scenario in ("buying", "browsing", "intent_override", "boundary"):
        values = [runs[d]["result"]["scenario_metrics"][scenario]["hit_rate_at_10"]
                  for d in DEPTHS]
        delta = values[-1] - values[0]
        if delta < 0:
            regressions.append((scenario, delta))
        print(f"   {scenario:18}" + "".join(f"{v:>12.4f}" for v in values)
              + f"{delta:>+12.4f}")
    print(f"   scenarios regressing at 200: "
          f"{'none' if not regressions else regressions}")

    print("\n" + "=" * 74)
    print("3. THE MECHANISM -- do the truncated sessions actually recover?")
    print("=" * 74)
    print("   The claim under test is that the top-50 window CUTS specific")
    print("   sessions, not that a bigger number scores better. So: which")
    print("   ranking failures at 50 stop being failures at 200, and are they")
    print("   the ones a depth explanation predicts?")
    for depth in DEPTHS:
        run = runs[depth]
        print(f"   depth {depth:>3}   retrieval failures {len(run['retrieval_failures']):>3}"
              f"   ranking failures {len(run['ranking_failures']):>3}"
              f"   recall@300 {run['recall300']}/200")
    lost_at_50 = base["ranking_failures"]
    positions = base["best_pool_position"]
    for depth in (100, 200):
        failures = runs[depth]["ranking_failures"] | runs[depth]["retrieval_failures"]
        recovered = lost_at_50 - failures
        introduced = runs[depth]["ranking_failures"] - lost_at_50
        # The mechanism: only a target the OLD window could not reach is a
        # truncation. Sessions whose target sat inside the top 50 all along
        # were seen by the reranker and lost anyway; depth cannot be their
        # explanation, and if they are the ones "recovering" then the gain is
        # reshuffling rather than reach.
        reachable = {sample for sample in lost_at_50
                     if positions.get(sample) is not None
                     and 50 < positions[sample] <= depth}
        unreachable = lost_at_50 - reachable
        print(f"\n   depth {depth:>3}   {len(recovered)} of the "
              f"{len(lost_at_50)} recover, {len(introduced)} new appear")
        print(f"               targets the top-50 window could NOT reach but "
              f"top-{depth} can: {len(reachable)}")
        print(f"               of those, recovered: "
              f"{len(reachable & recovered)}/{len(reachable)}")
        print(f"               of the {len(unreachable)} it could already "
              f"reach, recovered: {len(unreachable & recovered)}")

    print("\n" + "=" * 74)
    print("4. CANDIDATE -> TOP 10, and the delay between them")
    print("=" * 74)
    buckets = ("same turn", "+1", "+2", "3+", "never", "never in pool")
    print(f"   {'depth':>6}" + "".join(f"{b:>15}" for b in buckets))
    for depth in DEPTHS:
        print(f"   {depth:>6}" + "".join(
            f"{runs[depth]['latency'][b]:>15}" for b in buckets))
    print(f"\n   {'depth':>6}{'downstream mean':>18}{'median':>10}{'P90':>8}")
    for depth in DEPTHS:
        mean, median, p90 = _percentiles(
            [float(v) for v in runs[depth]["downstream"]])
        print(f"   {depth:>6}{mean:>18.2f}{median:>10.1f}{p90:>8.1f}")

    print("\n" + "=" * 74)
    print("5. COST -- a deeper window that overran the budget would fall back")
    print("=" * 74)
    print(f"   {'depth':>6}{'mean ms':>10}{'median':>9}{'P90':>8}{'max':>8}"
          f"{'budget':>9}{'timeouts':>10}{'fallbacks':>11}")
    for depth in DEPTHS:
        run = runs[depth]
        mean, median, p90 = _percentiles(run["order_ms"])
        fallbacks = sum(run["outcomes"][name] for name in
                        ("timeout", "error", "malformed", "empty"))
        print(f"   {depth:>6}{mean:>10.3f}{median:>9.3f}{p90:>8.3f}"
              f"{max(run['order_ms'], default=0.0):>8.3f}"
              f"{reranker.RERANK_BUDGET_MS:>9.0f}"
              f"{run['outcomes']['timeout']:>10}{fallbacks:>11}")
    print("   A nonzero timeout or fallback count at any depth would mean the")
    print("   stage silently returned ranking's order on those turns, and the")
    print("   'gain' at that depth would be an artifact rather than a result.")

    print(f"\nconfig: {config_guard.describe()}")


if __name__ == "__main__":
    main()
