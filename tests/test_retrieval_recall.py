"""CP 5.5 / 5.6 / 5.7 — candidate-pool recall on the real catalog.

Measures, per public session, whether the hidden target lands in the top-K of
the retrieval pool as the disclosed constraints accumulate turn by turn, and
prints the per-route / loss-stage / candidate-count diagnostics the retrieval
review asks for.

Skipped automatically when the full catalog / public set is not present.
Builds the FTS index once (~2s) then runs ~3 FTS queries per turn.
"""

from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

from starter.agent import Agent
from starter.contracts import Context, SessionState
from starter.retrieval import POOL_LIMIT, ROUTES, fuse, run_routes
from starter.state import update_state

_CATALOG = Path("data/catalog.jsonl")
_DATASET = Path("data/public_set.jsonl")
_K = (50, 100, 300)


@unittest.skipUnless(_CATALOG.exists() and _DATASET.exists(), "catalog/public_set not available")
class RetrievalRecallTest(unittest.TestCase):
    """One pass over the public set; every assertion reads the same probe."""

    @classmethod
    def setUpClass(cls) -> None:
        from evaluator.local_evaluator import catalog_index, load_jsonl

        cls.agent = Agent(str(_CATALOG))
        cls.samples = load_jsonl(str(_DATASET))
        _, _, cls.products = catalog_index(str(_CATALOG))
        cls.probe = cls._probe_all()

    # -- session replay ---------------------------------------------------

    @classmethod
    def _session_messages(cls, sample: dict) -> list[str]:
        from evaluator.local_evaluator import coarse_category, materialize_hidden_fields

        card, _ = materialize_hidden_fields(sample, cls.products)
        target = str(sample["ground_truth"]["parent_asin"])
        cats = cls.products.get(target, {}).get("categories") or []
        opening = f"I'm looking for {coarse_category([str(c) for c in cats])}"
        return [opening,
                *[str(c) for c in card.get("hard_constraints", [])],
                *[str(c) for c in card.get("soft_preferences", [])]]

    @classmethod
    def _probe_all(cls) -> list[dict]:
        """For every scored session: best rank per pool, per-route presence,
        pre-cap union presence, and pool sizes."""
        out: list[dict] = []
        for sample in cls.samples:
            target = str(sample["ground_truth"]["parent_asin"])
            if target not in cls.products:
                continue
            state = SessionState(session_id="r")
            record = {
                "scenario": sample["scenario_type"],
                "bm25_rank": None,
                "union_rank": None,
                "routes": {name: False for name in ROUTES},
                "precap": False,
                "sizes": [],
            }
            for turn, message in enumerate(cls._session_messages(sample), start=1):
                update_state(state, message, turn)
                ctx = Context(session_id="r", turn=turn, user_message=message, state=state)
                per_route = run_routes(cls.agent.connection, ctx, POOL_LIMIT)

                for name, rows in per_route.items():
                    if any(asin == target for asin, _ in rows):
                        record["routes"][name] = True

                precap = {asin for rows in per_route.values() for asin, _ in rows}
                record["sizes"].append(len(precap))
                record["precap"] = record["precap"] or target in precap

                bm25 = [asin for asin, _ in per_route["bm25"]]
                if target in bm25:
                    rank = bm25.index(target)
                    record["bm25_rank"] = rank if record["bm25_rank"] is None \
                        else min(record["bm25_rank"], rank)

                pool = [c.parent_asin for c in fuse(per_route, POOL_LIMIT)]
                if target in pool:
                    rank = pool.index(target)
                    record["union_rank"] = rank if record["union_rank"] is None \
                        else min(record["union_rank"], rank)
            out.append(record)
        return out

    # -- helpers ----------------------------------------------------------

    def _recall(self, key: str, k: int, records=None) -> float:
        records = self.probe if records is None else records
        if not records:
            return 0.0
        hits = sum(1 for r in records if r[key] is not None and r[key] < k)
        return hits / len(records)

    # -- CP 5.5 / 5.6 / 5.7 ----------------------------------------------

    def test_union_recall_beats_bm25_at_every_k(self) -> None:
        print(f"\n{'pool':10}" + "".join(f"{'@' + str(k):>9}" for k in _K))
        for label, key in (("bm25", "bm25_rank"), ("union", "union_rank")):
            print(f"{label:10}" + "".join(f"{self._recall(key, k):>9.3f}" for k in _K))

        for k in _K:
            bm25, union = self._recall("bm25_rank", k), self._recall("union_rank", k)
            self.assertGreater(union, bm25, f"union must beat bm25 @{k}")

    def test_recall_is_monotonic_in_k(self) -> None:
        values = [self._recall("union_rank", k) for k in _K]
        self.assertEqual(values, sorted(values))

    def test_recall_at_300_clears_the_review_floor(self) -> None:
        self.assertGreater(self._recall("union_rank", 300), 0.90)

    # -- per-route presence and loss stage -------------------------------

    def test_every_route_contributes_target_presence(self) -> None:
        n = len(self.probe)
        print("\nper-route target presence (any turn):")
        for name in ROUTES:
            present = sum(1 for r in self.probe if r["routes"][name]) / n
            print(f"  {name:12}{present:>8.3f}")
            self.assertGreater(present, 0.0, f"{name} never surfaced any target")
        precap = sum(1 for r in self.probe if r["precap"]) / n
        print(f"  {'pre-cap':12}{precap:>8.3f}")
        # the union of routes must reach at least as many targets as bm25 alone
        self.assertGreaterEqual(precap, sum(1 for r in self.probe if r["routes"]["bm25"]) / n)

    def test_loss_stage_accounting_is_complete(self) -> None:
        stages = Counter()
        for record in self.probe:
            if record["union_rank"] is not None and record["union_rank"] < POOL_LIMIT:
                stages["in_final_pool"] += 1
            elif record["precap"]:
                stages["lost_at_fusion_cap"] += 1
            else:
                stages["never_retrieved_by_any_route"] += 1
        n = len(self.probe)
        print(f"\ncandidate loss stage (n={n}):")
        for stage, count in stages.most_common():
            print(f"  {stage:32}{count:>5}{count / n:>9.3f}")
        self.assertEqual(sum(stages.values()), n, "every session must be accounted for")

    def test_candidate_count_distribution(self) -> None:
        sizes = sorted(size for record in self.probe for size in record["sizes"])
        mean = sum(sizes) / len(sizes)
        median = sizes[len(sizes) // 2]
        p90 = sizes[int(len(sizes) * 0.9)]
        print(f"\npre-cap union candidate count: mean={mean:.1f} "
              f"median={median} p90={p90} max={sizes[-1]}")
        self.assertGreater(median, POOL_LIMIT,
                           "pre-cap union should exceed the pool cap, else the "
                           "cap is not exercised")

    def test_recall_split_by_scenario(self) -> None:
        scenarios = sorted({r["scenario"] for r in self.probe})
        print(f"\n{'scenario':18}{'n':>4}{'bm25@300':>10}{'union@300':>11}")
        for scenario in scenarios:
            rows = [r for r in self.probe if r["scenario"] == scenario]
            bm25 = self._recall("bm25_rank", 300, rows)
            union = self._recall("union_rank", 300, rows)
            print(f"{scenario:18}{len(rows):>4}{bm25:>10.3f}{union:>11.3f}")
            self.assertGreaterEqual(union, bm25, scenario)


if __name__ == "__main__":
    unittest.main()
