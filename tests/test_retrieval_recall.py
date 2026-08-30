"""CP 5.5 / 5.6 / 5.7 — candidate-pool recall on the real catalog.

Measures, per public session, whether the hidden target ever lands in the
top-K of the retrieval pool as the disclosed constraints accumulate turn by
turn. Reports the UNION pool against a BM25-only baseline so the value of the
category + attribute routes is visible.

Skipped automatically when the full catalog / public set is not present.
Builds the FTS index once (~20s).
"""

from __future__ import annotations

import unittest
from pathlib import Path

from starter.agent import Agent
from starter.contracts import Context, SessionState
from starter.retrieval import ROUTES, bm25_route, retrieve
from starter.state import update_state

_CATALOG = Path("data/catalog.jsonl")
_DATASET = Path("data/public_set.jsonl")
_K = (50, 100, 300)


def _bm25_only_pool(connection, context, limit):
    return [asin for asin, _ in bm25_route(connection, context, limit)]


@unittest.skipUnless(_CATALOG.exists() and _DATASET.exists(), "catalog/public_set not available")
class RetrievalRecallTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from evaluator.local_evaluator import catalog_index, load_jsonl, materialize_hidden_fields

        cls.agent = Agent(str(_CATALOG))
        cls.samples = load_jsonl(str(_DATASET))
        _, _, cls.products = catalog_index(str(_CATALOG))
        cls._materialize = staticmethod(materialize_hidden_fields)

    def _session_messages(self, sample: dict) -> list[str]:
        card, _ = self._materialize(sample, self.products)
        target = str(sample["ground_truth"]["parent_asin"])
        cats = self.products.get(target, {}).get("categories") or []
        from evaluator.local_evaluator import coarse_category

        opening = f"I'm looking for {coarse_category([str(c) for c in cats])}"
        return [opening, *[str(c) for c in card.get("hard_constraints", [])],
                *[str(c) for c in card.get("soft_preferences", [])]]

    def _best_rank(self, sample: dict, pool_fn) -> int | None:
        target = str(sample["ground_truth"]["parent_asin"])
        state = SessionState(session_id="r")
        best: int | None = None
        for turn, message in enumerate(self._session_messages(sample), start=1):
            update_state(state, message, turn)
            ctx = Context(session_id="r", turn=turn, user_message=message, state=state)
            pool = pool_fn(self.agent.connection, ctx, max(_K))
            if target in pool:
                rank = pool.index(target)
                best = rank if best is None else min(best, rank)
        return best

    def _recall(self, pool_fn) -> dict[int, float]:
        scored = [s for s in self.samples
                  if str(s["ground_truth"]["parent_asin"]) in self.products]
        hits = {k: 0 for k in _K}
        for sample in scored:
            best = self._best_rank(sample, pool_fn)
            if best is None:
                continue
            for k in _K:
                if best < k:
                    hits[k] += 1
        return {k: hits[k] / len(scored) for k in _K}

    def test_union_pool_recall_never_below_bm25_and_is_monotonic(self) -> None:
        union = self._recall(lambda c, x, k: [cand.parent_asin for cand in retrieve(c, x, k)])
        bm25 = self._recall(_bm25_only_pool)

        print(f"\n{'':12}{'@50':>8}{'@100':>8}{'@300':>8}{'  in-pool':>10}")
        print(f"{'bm25':12}" + "".join(f"{bm25[k]:>8.3f}" for k in _K)
              + f"{bm25[max(_K)]:>10.3f}")
        print(f"{'union':12}" + "".join(f"{union[k]:>8.3f}" for k in _K)
              + f"{union[max(_K)]:>10.3f}")

        # monotonic
        self.assertLessEqual(union[50], union[100])
        self.assertLessEqual(union[100], union[300])
        # append-only union: pool head is BM25's, so Recall@K can never fall
        # below BM25's for any measured K.
        for k in _K:
            self.assertGreaterEqual(union[k], bm25[k], f"@{k}")
        self.assertGreater(union[300], 0.30)

    def test_finding_retrieval_recall_is_not_the_bottleneck(self) -> None:
        # BM25 alone reaches essentially every target if the pool is deep
        # enough -- the gap between Recall@300 (~0.94) and Recall@10 (HR 0.125)
        # is a RANKING gap, not a retrieval gap. Documents why the extra
        # routes buy no top-K recall here (Phase 6 uses their per-route
        # signals instead).
        deep = 800

        def reach(sample: dict) -> bool:
            target = str(sample["ground_truth"]["parent_asin"])
            state = SessionState(session_id="r")
            for turn, message in enumerate(self._session_messages(sample), start=1):
                update_state(state, message, turn)
                ctx = Context(session_id="r", turn=turn, user_message=message, state=state)
                if target in _bm25_only_pool(self.agent.connection, ctx, deep):
                    return True
            return False

        scored = [s for s in self.samples
                  if str(s["ground_truth"]["parent_asin"]) in self.products]
        deep_reach = sum(reach(s) for s in scored) / len(scored)
        print(f"\nbm25 reach @{deep}: {deep_reach:.3f}")
        self.assertGreater(deep_reach, 0.95)

    def test_every_route_is_reachable(self) -> None:
        self.assertEqual(set(ROUTES), {"bm25", "category", "attribute"})


if __name__ == "__main__":
    unittest.main()
