"""Phase 7 — first core evaluation.

Controlled comparison of the layers built in Phases 0-6, toggling the REAL
code path via the ablation flags in ``starter`` rather than a reimplemented
pipeline, so what is measured is what ships.

    Run 0  official baseline        state OFF, multi-route OFF, ranking OFF
    Run 1  state only               state ON,  multi-route OFF, ranking OFF
    Run 2  state + retrieval        state ON,  multi-route ON,  ranking OFF
    Run 3  state + retr + ranking   state ON,  multi-route ON,  ranking ON

Run 0 must reproduce ``docs/baseline_results.json`` exactly; that is the
validity check on the whole ablation.

Every difference is reported with the paired discordant counts and an exact
McNemar p-value beside it. A TS delta of a few thousandths is one or two
flipped sessions out of 200; without the pairing there is no way to tell a
real effect from noise, and this repo has now twice banked a point estimate
that the pairing does not support.

Usage:  python3 -m tools.phase7_ablation
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent
from tools import config_guard
from tools.significance import (format_test, hits_by_sample, mcnemar,
                                mttc_given_hit)

CATALOG = "data/catalog.jsonl"
DATASET = "data/public_set.jsonl"

# The roadmap's linear ladder, PLUS the cell it skips (Run 3b). A linear
# 0->1->2->3 walk can only answer "does each layer help given the previous
# one?", which hides the case where an earlier layer is a net cost that a
# later one merely compensates for. Run 3b isolates that.
RUNS = [
    # label, USE_STATE, USE_MULTI_ROUTE, USE_CONSTRAINT_RANKING, USE_POPULARITY
    ("Run 0   baseline", False, False, False, False),
    ("Run 1   +state", True, False, False, False),
    ("Run 2   +retrieval", True, True, False, False),
    ("Run 3   +ranking", True, True, True, False),
    ("Run 4   +popularity", True, True, True, True),
    ("Run 3b  ranking, BM25 pool", True, False, True, False),
]

# Run 4 is the stack that SHIPS. Popularity gets its own rung rather than
# riding with USE_CONSTRAINT_RANKING because it is worth more than every layer
# before it combined, and folding it into another rung would hide that.

# Every ablation flag in the WHOLE package, so a run pins the entire
# configuration. Scoped package-wide via ``tools.config_guard`` after D-N2:
# the first version of this guard scanned ``starter.agent`` only, so a flag
# added to ``ranking.py`` -- or a change to ``retrieval.DEFAULT_ROUTES``, which
# is not a flag at all -- drifted every number here silently while Run 0's
# validity check still passed.
PINNED_FLAGS = {
    ("agent", "USE_STATE"),
    ("agent", "USE_MULTI_ROUTE"),
    ("agent", "USE_CONSTRAINT_RANKING"),
    ("ranking", "USE_PROFILE"),
    ("ranking", "USE_CONFIDENCE_WEIGHTING"),
    ("ranking", "USE_POPULARITY"),
}


def main() -> None:
    config_guard.assert_all_flags_pinned(PINNED_FLAGS)
    config_guard.assert_committed_constants()

    print("building index...", flush=True)
    started = time.time()
    agent = Agent(CATALOG)
    samples = load_jsonl(DATASET)
    catalog_ids, categories, products = catalog_index(CATALOG)
    print(f"ready in {time.time() - started:.1f}s\n")

    results = []
    for label, use_state, use_routes, use_ranking, use_popularity in RUNS:
        config_guard.set_flag("agent", "USE_STATE", use_state)
        config_guard.set_flag("agent", "USE_MULTI_ROUTE", use_routes)
        config_guard.set_flag("agent", "USE_CONSTRAINT_RANKING", use_ranking)
        config_guard.set_flag("ranking", "USE_POPULARITY", use_popularity)
        config_guard.set_flag("ranking", "USE_PROFILE", False)
        # Pinned to its COMMITTED value, not to the rung. This ladder measures
        # the stack that ships, and Phase 11 weighting ships OFF (measured
        # exactly inert -- see ranking.USE_CONFIDENCE_WEIGHTING). Its own arm
        # is measured by tools/phase11_confidence.py, which is where a claim
        # about it belongs.
        config_guard.set_flag("ranking", "USE_CONFIDENCE_WEIGHTING", False)
        started = time.time()
        result = evaluate(agent, samples, catalog_ids, categories, products)
        result["_label"] = label
        result["_seconds"] = time.time() - started
        results.append(result)
        print(f"{label:26} TS={result['recommended_technical_score']:.6f} "
              f"({result['_seconds']:.0f}s)", flush=True)

    config_guard.restore_committed_flags()

    by_label = {r["_label"]: r for r in results}
    base = results[0]["recommended_technical_score"]

    print(f"\n{'run':28}{'HR@10':>8}{'MRR':>10}{'MTTC':>8}{'MTTC|hit':>10}"
          f"{'TS':>10}{'d prev':>9}{'d base':>9}")
    previous = base
    for result in results[:5]:  # the linear ladder
        score = result["recommended_technical_score"]
        conditional = mttc_given_hit(result)
        print(f"{result['_label']:28}{result['hit_rate_at_10']:>8.4f}"
              f"{result['mrr']:>10.6f}{result['mttc']:>8.3f}"
              f"{(conditional if conditional is not None else float('nan')):>10.3f}"
              f"{score:>10.6f}{score - previous:>+9.6f}{score - base:>+9.6f}")
        previous = score
    off_ladder = results[5]
    score = off_ladder["recommended_technical_score"]
    conditional = mttc_given_hit(off_ladder)
    print(f"{off_ladder['_label']:28}{off_ladder['hit_rate_at_10']:>8.4f}"
          f"{off_ladder['mrr']:>10.6f}{off_ladder['mttc']:>8.3f}"
          f"{(conditional if conditional is not None else float('nan')):>10.3f}"
          f"{score:>10.6f}{'':>9}{score - base:>+9.6f}")
    print("  MTTC charges a miss as MAX_TURNS+1, so it tracks the hit rate; "
          "MTTC|hit is the\n  conditional number and moves only on the "
          "sessions actually solved (D-F8).")

    print(f"\n{'run':28}{'buying':>9}{'browsing':>10}{'override':>10}{'boundary':>10}")
    for result in results:
        scenario = result["scenario_metrics"]
        print(f"{result['_label']:28}"
              + "".join(f"{scenario[name]['hit_rate_at_10']:>{width}.4f}"
                        for name, width in (("buying", 9), ("browsing", 10),
                                            ("intent_override", 10), ("boundary", 10))))

    # The 2x2 that actually answers "is each layer worth it?"
    print(f"\n2x2 interaction (TS){'':6}{'ranking OFF':>14}{'ranking ON':>13}{'effect':>10}")
    cells = {
        ("bm25", False): by_label["Run 1   +state"],
        ("bm25", True): by_label["Run 3b  ranking, BM25 pool"],
        ("union", False): by_label["Run 2   +retrieval"],
        ("union", True): by_label["Run 3   +ranking"],
    }
    for pool in ("bm25", "union"):
        off = cells[(pool, False)]["recommended_technical_score"]
        on = cells[(pool, True)]["recommended_technical_score"]
        print(f"  {pool + ' pool':24}{off:>14.6f}{on:>13.6f}{on - off:>+10.6f}")
    print(f"  {'effect of multi-route':24}"
          f"{cells[('union', False)]['recommended_technical_score'] - cells[('bm25', False)]['recommended_technical_score']:>+14.6f}"
          f"{cells[('union', True)]['recommended_technical_score'] - cells[('bm25', True)]['recommended_technical_score']:>+13.6f}")

    # Paired significance. The arm that SHIPS is ranking ON; a difference that
    # only exists in the ranking-OFF arm is not a property of the shipped
    # agent, and must not be quoted as one (D-N1).
    print("\npaired significance (McNemar exact, per-session hit verdicts)")
    print("  the shipped arm is ranking ON -- read that column, not the other")
    pairs = [
        ("Phase 0-6 stack vs baseline", "Run 0   baseline", "Run 3   +ranking"),
        ("state alone", "Run 0   baseline", "Run 1   +state"),
        ("multi-route | ranking OFF", "Run 1   +state", "Run 2   +retrieval"),
        ("multi-route | ranking ON  *", "Run 3b  ranking, BM25 pool", "Run 3   +ranking"),
        ("ranking | bm25 pool", "Run 1   +state", "Run 3b  ranking, BM25 pool"),
        ("ranking | union pool  *", "Run 2   +retrieval", "Run 3   +ranking"),
        ("popularity | full stack  *", "Run 3   +ranking", "Run 4   +popularity"),
        ("whole shipped stack", "Run 0   baseline", "Run 4   +popularity"),
    ]
    for label, before_label, after_label in pairs:
        test = mcnemar(hits_by_sample(by_label[before_label]),
                       hits_by_sample(by_label[after_label]))
        print("  " + format_test(label, test))
    print("  * = the arm that ships. n=200 sessions; one session is ~0.005 HR "
          "/ ~0.01 TS.")

    reference = json.loads(Path("docs/baseline_results.json").read_text())
    run0 = results[0]
    ok = (abs(run0["hit_rate_at_10"] - reference["hit_rate_at_10"]) < 1e-9
          and abs(run0["mrr"] - reference["mrr"]) < 1e-6)
    print(f"\nvalidity: Run 0 reproduces docs/baseline_results.json -> "
          f"{'PASS' if ok else 'FAIL'} "
          f"(HR {run0['hit_rate_at_10']} vs {reference['hit_rate_at_10']}, "
          f"MRR {run0['mrr']} vs {reference['mrr']})")
    print(f"\ncommitted config: {config_guard.describe()}")


if __name__ == "__main__":
    main()
