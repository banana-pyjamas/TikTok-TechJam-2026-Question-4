"""Phase 12 — the popularity prior, measured.

This is the largest gain the project has produced, and the first one where the
honest explanation is less flattering than the number. So the order here is:
the effect, then WHY the effect is that big, then what is deliberately being
left on the table because of it.

  1. how popular are the ground-truth targets, compared to the catalog?
  2. what does the prior do to the score, and is it established?
  3. what would a larger weight be worth -- i.e. what does CP 12.4 cost?
  4. what does CP 12.3's decay cost?
  5. does a bestseller ever actually beat a matching product? (CP 12.4, live)

Question 1 comes first on purpose. Without it, question 2's answer invites
exactly the wrong conclusion.

Usage:  python3 -m tools.phase12_popularity
        python3 -m tools.phase12_popularity --quick   (skip the weight sweep)
"""

from __future__ import annotations

import bisect
import json
import statistics
import sys
import time
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter import popularity as popularity_module
from starter import ranking
from starter.catalog_meta import lookup as meta_lookup
from starter.contracts import Context
from starter.popularity import W_POPULARITY, popularity_feature
from starter.ranking import (POPULARITY_KEY, W_MATCH, active_constraints,
                             classify, rank)
from starter.retrieval import DEFAULT_ROUTES, POOL_LIMIT, retrieve
from tools import config_guard
from tools.capture import CapturingAgent
from tools.significance import format_test, hits_by_sample, mcnemar

CATALOG = "data/catalog.jsonl"
DATASET = "data/public_set.jsonl"

# Weights to sweep. 0.10 equals W_MATCH, where a bestseller can cancel a
# satisfied constraint outright -- included to show where the cliff is, not as
# a candidate setting.
SWEEP = (0.008, 0.02, 0.05, 0.10)


def main() -> None:
    quick = "--quick" in sys.argv
    config_guard.assert_all_flags_pinned(set(config_guard.COMMITTED_FLAGS))
    config_guard.assert_committed_constants()
    config_guard.restore_committed_flags()

    print("building index...", flush=True)
    started = time.time()
    agent = CapturingAgent(CATALOG)
    samples = load_jsonl(DATASET)
    catalog_ids, categories, products = catalog_index(CATALOG)
    print(f"ready in {time.time() - started:.1f}s", flush=True)

    # -- 1. the sampling fact that explains everything below ---------------
    counts = {}
    with Path(CATALOG).open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            counts[str(product["parent_asin"])] = float(
                product.get("rating_number") or 0)
    catalog_sorted = sorted(counts.values())
    targets = [counts[str(s["ground_truth"]["parent_asin"])] for s in samples
               if str(s["ground_truth"]["parent_asin"]) in counts]
    target_sorted = sorted(targets)
    median_target = target_sorted[len(target_sorted) // 2]
    percentile = bisect.bisect_left(catalog_sorted, median_target) / len(catalog_sorted)
    below = sum(1 for value in targets
                if value < catalog_sorted[len(catalog_sorted) // 2])

    print("\n1. HOW THE TARGETS WERE SAMPLED -- read this before the scores")
    print(f"   catalog  n={len(catalog_sorted):<6} median rating_number "
          f"{catalog_sorted[len(catalog_sorted) // 2]:>10.0f}   "
          f"mean {statistics.fmean(catalog_sorted):>9.0f}")
    print(f"   targets  n={len(target_sorted):<6} median rating_number "
          f"{median_target:>10.0f}   "
          f"mean {statistics.fmean(target_sorted):>9.0f}")
    print(f"   the median target sits at the {percentile:.1%} percentile of "
          "the catalog")
    print(f"   targets below the catalog median: {below}/{len(targets)} "
          f"({below / len(targets):.1%}) -- unbiased would be 50%")
    print("   So on this benchmark a bestseller list is close to an oracle.")
    print("   The gain below is real and should transfer to the private set,")
    print("   which is built the same way. It is NOT evidence that ranking by")
    print("   review count serves shoppers. Quote it with this paragraph.")

    # -- 2. the effect -----------------------------------------------------
    def run(label: str) -> dict:
        started = time.time()
        result = evaluate(agent, samples, catalog_ids, categories, products)
        print(f"   {label:34}HR {result['hit_rate_at_10']:.4f}  "
              f"MRR {result['mrr']:.6f}  MTTC {result['mttc']:.3f}  "
              f"TS {result['recommended_technical_score']:.6f}  "
              f"({time.time() - started:.0f}s)", flush=True)
        return result

    print("\n2. WHAT IT DOES TO THE SCORE")
    results = {}
    for enabled in (False, True):
        config_guard.set_flag("ranking", "USE_POPULARITY", enabled)
        results[enabled] = run(f"USE_POPULARITY={enabled}")
    config_guard.restore_committed_flags()
    # Pinned to the CURRENT pipeline, not to a frozen historical arm. This
    # tool measures what USE_POPULARITY is worth NOW, and Phase 14 changed
    # what "now" is: its OFF arm was 0.134566 when the ladder ended at Phase
    # 12 and is 0.178651 with the reranker in front of it. Asserting the old
    # literal made this tool exit on its own staleness, which is the failure
    # phase10 and phase11 each spent two phases in.
    if abs(results[True]["recommended_technical_score"]
           - config_guard.COMMITTED_TECHNICAL_SCORE) > 1e-9:
        raise SystemExit(
            "the ON arm no longer reproduces the committed score, so this is "
            "not a measurement of the shipped pipeline")
    print("   " + format_test("popularity ON vs OFF",
                              mcnemar(hits_by_sample(results[False]),
                                      hits_by_sample(results[True]))))
    for scenario in ("buying", "browsing", "intent_override", "boundary"):
        before = results[False]["scenario_metrics"][scenario]["hit_rate_at_10"]
        after = results[True]["scenario_metrics"][scenario]["hit_rate_at_10"]
        print(f"     {scenario:18}{before:.4f} -> {after:.4f}")

    # -- 3. what CP 12.4 costs --------------------------------------------
    if not quick:
        print("\n3. WHAT THE CP 12.4 CONSTRAINT COSTS (weight sweep)")
        print(f"   W_MATCH is {W_MATCH}; a prior at or above it can cancel a")
        print("   satisfied constraint outright, which is the collapse CP 12.4")
        print("   forbids. Shipped weight is an order of magnitude below it.")
        config_guard.set_flag("ranking", "USE_POPULARITY", True)
        original = popularity_module.W_POPULARITY
        for weight in SWEEP:
            popularity_module.W_POPULARITY = weight
            note = "  <- shipped" if weight == original else (
                "  <- == W_MATCH, CP 12.4 violated" if weight >= W_MATCH else "")
            run(f"W_POPULARITY={weight}{note}")
        popularity_module.W_POPULARITY = original

        # -- 4. what CP 12.3 costs ----------------------------------------
        print("\n4. WHAT THE CP 12.3 DECAY COSTS")
        decay = popularity_module.evidence_decay
        popularity_module.evidence_decay = lambda evidence: 1.0
        run("decay disabled (not shipped)")
        popularity_module.evidence_decay = decay
        print("   The decay costs score on this benchmark, because the signal")
        print("   it decays away is the strongest predictor the set contains.")
        print("   Kept anyway: CP 12.3 asks for a prior, and a prior that does")
        print("   not yield to evidence is not a prior. Recorded, not hidden.")
        config_guard.restore_committed_flags()

    # -- 5. CP 12.4, measured on the live dialogue -------------------------
    print("\n5. CP 12.4 ON THE LIVE DIALOGUE: does a bestseller ever beat a match?")
    print("   Scope is the FULL ranked pool, not the Top-10. An earlier version")
    print("   passed top_k=10 to rank(), so the check never looked past rank 10")
    print("   and the reported figure was far narrower than it sounded")
    print("   (C Phase 12 review). Full-pool is a strictly stronger check.")
    agent.order.clear()
    agent.captured.clear()
    evaluate(agent, samples, catalog_ids, categories, products)
    inversions = 0
    checked = 0
    eligible = 0
    for session_id, captures in agent.by_session().items():
        for capture in captures:
            context = Context(session_id=session_id, turn=capture["turn"],
                              user_message=capture["message"],
                              state=capture["state"])
            context.derived[POPULARITY_KEY] = agent._popularity
            constraints, bounds = active_constraints(context)
            if not constraints:
                continue
            pool = retrieve(agent.connection, context, POOL_LIMIT, DEFAULT_ROUTES)
            if not pool:
                continue
            metadata = meta_lookup(agent.connection,
                                   [c.parent_asin for c in pool])
            # len(pool), not 10: rank truncates to top_k before returning.
            ranked = rank(pool, context, metadata, len(pool)).ranked
            # A "match" satisfies at least one constraint; a "non-match"
            # satisfies none.
            verdicts = {}
            for candidate in ranked:
                meta = metadata.get(candidate.parent_asin, {})
                verdicts[candidate.parent_asin] = any(
                    classify(slot, values, meta, bounds) == ranking.MATCH
                    for slot, values in constraints.items())
            checked += 1
            # A turn can only EXHIBIT an inversion if it contains both a match
            # and a non-match. Counting turns that could never have shown one
            # inflates the denominator and makes the evidence look far thicker
            # than it is (D Phase 12 review).
            values = list(verdicts.values())
            if not (any(values) and not all(values)):
                continue
            eligible += 1
            order = [c.parent_asin for c in ranked]
            for index, asin in enumerate(order):
                if verdicts[asin]:
                    continue
                if any(verdicts[later] for later in order[index + 1:]):
                    inversions += 1
                    break
    print(f"   turns with >=1 active constraint and a non-empty pool: {checked}")
    print(f"   of those, turns that COULD show an inversion (mixed pool): "
          f"{eligible}")
    print(f"   turns where a NON-matching candidate outranks a matching one: "
          f"{inversions} ({inversions / max(eligible, 1):.1%} of eligible)")
    print("   Read the eligible row as the denominator. The checked row counts")
    print("   turns whose pool was all-match or all-non-match, where no")
    print("   inversion was possible and a zero means nothing.")
    print("   And read the result as what it is: an observed absence on this")
    print("   dataset, not a structural proof. The component bound "
          f"(W_POPULARITY {W_POPULARITY}")
    print(f"   << W_MATCH {W_MATCH}) constrains two terms; base_score varies")
    print("   independently and is no part of it. See starter/popularity.py.")

    print(f"\nconfig: {config_guard.describe()}")


if __name__ == "__main__":
    main()
