"""Phase 7 — first core evaluation.

Controlled comparison of the layers built in Phases 0-6, toggling the REAL
code path via the ablation flags in ``starter.agent`` rather than a
reimplemented pipeline, so what is measured is what ships.

    Run 0  official baseline        state OFF, multi-route OFF, ranking OFF
    Run 1  state only               state ON,  multi-route OFF, ranking OFF
    Run 2  state + retrieval        state ON,  multi-route ON,  ranking OFF
    Run 3  state + retr + ranking   state ON,  multi-route ON,  ranking ON

Run 0 must reproduce ``docs/baseline_results.json`` exactly; that is the
validity check on the whole ablation.

Usage:  python3 -m tools.phase7_ablation
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from starter import agent as agent_module
from starter.agent import Agent
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl

CATALOG = "data/catalog.jsonl"
DATASET = "data/public_set.jsonl"

# The roadmap's linear ladder, PLUS the cell it skips (Run 3b). A linear
# 0->1->2->3 walk can only answer "does each layer help given the previous
# one?", which hides the case where an earlier layer is a net cost that a
# later one merely compensates for. Run 3b isolates that.
RUNS = [
    ("Run 0   baseline", False, False, False),
    ("Run 1   +state", True, False, False),
    ("Run 2   +retrieval", True, True, False),
    ("Run 3   +ranking", True, True, True),
    ("Run 3b  ranking, BM25 pool", True, False, True),
]


def main() -> None:
    print("building index...", flush=True)
    started = time.time()
    agent = Agent(CATALOG)
    samples = load_jsonl(DATASET)
    catalog_ids, categories, products = catalog_index(CATALOG)
    print(f"ready in {time.time() - started:.1f}s\n")

    results = []
    for label, use_state, use_routes, use_ranking in RUNS:
        agent_module.USE_STATE = use_state
        agent_module.USE_MULTI_ROUTE = use_routes
        agent_module.USE_CONSTRAINT_RANKING = use_ranking
        started = time.time()
        result = evaluate(agent, samples, catalog_ids, categories, products)
        result["_label"] = label
        result["_seconds"] = time.time() - started
        results.append(result)
        print(f"{label:26} TS={result['recommended_technical_score']:.6f} "
              f"({result['_seconds']:.0f}s)", flush=True)

    # restore committed defaults
    agent_module.USE_STATE = True
    agent_module.USE_MULTI_ROUTE = True
    agent_module.USE_CONSTRAINT_RANKING = True

    by_label = {r["_label"]: r for r in results}
    base = results[0]["recommended_technical_score"]

    print(f"\n{'run':28}{'HR@10':>8}{'MRR':>10}{'MTTC':>8}{'TS':>10}"
          f"{'d prev':>9}{'d base':>9}")
    previous = base
    for result in results[:4]:  # the linear ladder
        score = result["recommended_technical_score"]
        print(f"{result['_label']:28}{result['hit_rate_at_10']:>8.4f}"
              f"{result['mrr']:>10.6f}{result['mttc']:>8.3f}{score:>10.6f}"
              f"{score - previous:>+9.6f}{score - base:>+9.6f}")
        previous = score
    off_ladder = results[4]
    score = off_ladder["recommended_technical_score"]
    print(f"{off_ladder['_label']:28}{off_ladder['hit_rate_at_10']:>8.4f}"
          f"{off_ladder['mrr']:>10.6f}{off_ladder['mttc']:>8.3f}{score:>10.6f}"
          f"{'':>9}{score - base:>+9.6f}")

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

    reference = json.loads(Path("docs/baseline_results.json").read_text())
    run0 = results[0]
    ok = (abs(run0["hit_rate_at_10"] - reference["hit_rate_at_10"]) < 1e-9
          and abs(run0["mrr"] - reference["mrr"]) < 1e-6)
    print(f"\nvalidity: Run 0 reproduces docs/baseline_results.json -> "
          f"{'PASS' if ok else 'FAIL'} "
          f"(HR {run0['hit_rate_at_10']} vs {reference['hit_rate_at_10']}, "
          f"MRR {run0['mrr']} vs {reference['mrr']})")

    print("\ncommitted config: "
          f"USE_STATE={agent_module.USE_STATE}, "
          f"USE_MULTI_ROUTE={agent_module.USE_MULTI_ROUTE}, "
          f"USE_CONSTRAINT_RANKING={agent_module.USE_CONSTRAINT_RANKING}")


if __name__ == "__main__":
    main()
