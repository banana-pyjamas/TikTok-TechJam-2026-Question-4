"""CP 1.2 - 1.6 — the minimum end-to-end turn pipeline, stage by stage.

    user message -> Context -> BM25 -> Candidate[] -> RankingResult -> respond()

Stages that need the FTS index build against a tiny 3-row temporary catalog.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.agent import (
    Agent,
    _bm25_search,
    _build_context,
    _rank,
    _terms,
    _to_candidates,
    _to_response,
)
from starter.contracts import Candidate, Context, RankingResult, SessionState


_CATALOG_ROWS = [
    {
        "parent_asin": "B0RUN",
        "title": "Blue running shoe",
        "categories": ["Clothing", "Shoes"],
        "features": ["breathable mesh"],
        "details": {"department": "womens"},
        "store": "Example",
        "description": ["lightweight walking shoe"],
    },
    {
        "parent_asin": "B0BOOT",
        "title": "Black leather boot",
        "categories": ["Clothing", "Boots"],
        "features": ["full grain leather", "waterproof"],
        "details": {"department": "mens"},
        "store": "Example",
        "description": ["insulated winter boot"],
    },
    {
        "parent_asin": "B0SOCK",
        "title": "Wool hiking sock",
        "categories": ["Clothing", "Socks"],
        "features": ["merino wool"],
        "details": {"department": "unisex"},
        "store": "Example",
        "description": ["cushioned crew sock"],
    },
]

# Exact legacy inline query from before the CP 1.3 wrapper (parent_asin only).
_LEGACY_QUERY = (
    "SELECT parent_asin FROM products WHERE products MATCH ? "
    "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?"
)


class _CatalogFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        path = Path(cls._tmp.name) / "catalog.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in _CATALOG_ROWS), encoding="utf-8"
        )
        cls._agent = Agent(path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()


class BuildContextTest(unittest.TestCase):
    """CP 1.2 — Empty Context."""

    def test_minimal_fields_are_populated(self) -> None:
        state = SessionState(session_id="s1")
        ctx = _build_context("s1", "black boots", 3, state)
        self.assertIsInstance(ctx, Context)
        self.assertEqual(ctx.session_id, "s1")
        self.assertEqual(ctx.turn, 3)
        self.assertEqual(ctx.user_message, "black boots")
        self.assertEqual(ctx.derived, {})

    def test_state_is_passed_by_reference_not_copied(self) -> None:
        state = SessionState(session_id="s1")
        ctx = _build_context("s1", "hi", 1, state)
        self.assertIs(ctx.state, state)
        state.slots["category"] = "boot"
        self.assertEqual(ctx.state.slots, {"category": "boot"})

    def test_empty_message_is_still_a_valid_context(self) -> None:
        ctx = _build_context("s1", "", 1, SessionState())
        self.assertEqual(ctx.user_message, "")

    def test_non_string_message_is_coerced_to_empty(self) -> None:
        ctx = _build_context("s1", None, 1, SessionState())  # type: ignore[arg-type]
        self.assertEqual(ctx.user_message, "")
        ctx = _build_context("s1", 123, 1, SessionState())  # type: ignore[arg-type]
        self.assertEqual(ctx.user_message, "")


class Bm25SearchTest(_CatalogFixture):
    """CP 1.3 — Baseline BM25 wrapper, behavior unchanged."""

    def test_matches_expected_product(self) -> None:
        rows = _bm25_search(self._agent.connection, "black leather boot", 10)
        self.assertEqual([pa for pa, _ in rows][:1], ["B0BOOT"])

    def test_empty_or_stopword_only_query_returns_empty(self) -> None:
        self.assertEqual(_bm25_search(self._agent.connection, "", 10), [])
        self.assertEqual(_bm25_search(self._agent.connection, "the to a of", 10), [])

    def test_order_matches_legacy_inline_query(self) -> None:
        for message in ("black leather boot", "wool shoe", "shoe boot sock", "waterproof"):
            terms = list(dict.fromkeys(_terms(message)))[:40]
            expression = " OR ".join(f'"{t}"' for t in terms)
            legacy = [
                str(r[0])
                for r in self._agent.connection.execute(_LEGACY_QUERY, (expression, 10))
            ]
            wrapped = [pa for pa, _ in _bm25_search(self._agent.connection, message, 10)]
            self.assertEqual(wrapped, legacy, message)

    def test_respects_limit(self) -> None:
        rows = _bm25_search(self._agent.connection, "shoe boot sock wool leather", 2)
        self.assertLessEqual(len(rows), 2)

    def test_raw_bm25_value_is_returned(self) -> None:
        rows = _bm25_search(self._agent.connection, "black leather boot", 10)
        self.assertTrue(rows)
        for _, raw in rows:
            self.assertIsInstance(raw, float)


class ToCandidatesTest(unittest.TestCase):
    """CP 1.4 — Candidate conversion."""

    def test_preserves_parent_asin_and_order(self) -> None:
        rows = [("A", -1.0), ("B", -2.0), ("C", -0.5)]
        candidates = _to_candidates(rows)
        self.assertEqual([c.parent_asin for c in candidates], ["A", "B", "C"])

    def test_bm25_score_is_negated_raw_value(self) -> None:
        candidates = _to_candidates([("A", -1.5), ("B", -3.0)])
        self.assertEqual(candidates[0].route_scores, {"bm25": 1.5})
        self.assertEqual(candidates[1].route_scores, {"bm25": 3.0})

    def test_route_sources_is_bm25(self) -> None:
        (candidate,) = _to_candidates([("A", -1.0)])
        self.assertEqual(candidate.route_sources, ("bm25",))

    def test_empty_rows_give_empty_list(self) -> None:
        self.assertEqual(_to_candidates([]), [])


class RankTest(unittest.TestCase):
    """CP 1.5 — Basic ranking."""

    def _candidates(self, *scored: tuple[str, float]) -> list[Candidate]:
        return [Candidate(parent_asin=pa, route_scores={"bm25": s}) for pa, s in scored]

    def test_sorts_by_bm25_descending(self) -> None:
        result = _rank(self._candidates(("A", 1.0), ("B", 3.0), ("C", 2.0)), 10)
        self.assertEqual([c.parent_asin for c in result.ranked], ["B", "C", "A"])

    def test_stable_for_equal_scores(self) -> None:
        result = _rank(self._candidates(("A", 1.0), ("B", 1.0), ("C", 1.0)), 10)
        self.assertEqual([c.parent_asin for c in result.ranked], ["A", "B", "C"])

    def test_truncates_to_top_k(self) -> None:
        result = _rank(self._candidates(*[(f"P{i}", float(i)) for i in range(20)]), 10)
        self.assertEqual(len(result.ranked), 10)

    def test_returns_ranking_result_with_rank_diagnostics(self) -> None:
        result = _rank(self._candidates(("A", 2.0), ("B", 1.0)), 10)
        self.assertIsInstance(result, RankingResult)
        self.assertEqual(result.diagnostics["A"]["rank"], 1)
        self.assertEqual(result.diagnostics["B"]["rank"], 2)

    def test_empty_candidates_give_empty_result(self) -> None:
        result = _rank([], 10)
        self.assertEqual(result.ranked, [])
        self.assertEqual(result.diagnostics, {})


class ToResponseTest(unittest.TestCase):
    """CP 1.6 — Official response."""

    def _result(self, *asins: str) -> RankingResult:
        return _rank([Candidate(parent_asin=a, route_scores={"bm25": -i})
                      for i, a in enumerate(asins)], 100)

    def test_schema_shape(self) -> None:
        payload = _to_response(self._result("A", "B"), 10)
        self.assertEqual(
            set(payload), {"message", "ask_attribute", "recommendations", "usage"}
        )
        self.assertIsInstance(payload["message"], str)
        self.assertIsNone(payload["ask_attribute"])
        self.assertEqual(payload["usage"], {"prompt_tokens": 0, "completion_tokens": 0})

    def test_message_is_baseline_string(self) -> None:
        payload = _to_response(self._result("A"), 10)
        self.assertEqual(payload["message"], "Here are the closest matches I found.")

    def test_recommendations_are_capped_parent_asin_dicts(self) -> None:
        payload = _to_response(self._result(*[f"P{i}" for i in range(25)]), 10)
        self.assertEqual(len(payload["recommendations"]), 10)
        self.assertTrue(
            all(set(r) == {"parent_asin"} and isinstance(r["parent_asin"], str)
                for r in payload["recommendations"])
        )

    def test_empty_ranked_gives_empty_recommendations(self) -> None:
        payload = _to_response(RankingResult(), 10)
        self.assertEqual(payload["recommendations"], [])


if __name__ == "__main__":
    unittest.main()
