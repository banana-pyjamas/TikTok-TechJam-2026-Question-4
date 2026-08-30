"""CP 6.1 - 6.7 — constraint-aware ranking.

Scoring is ``base + W_MATCH * match_ratio - W_PENALTY * violation_ratio``,
where ``base`` is the retrieval fusion score.
"""

from __future__ import annotations

import math
import unittest

from starter import ranking
from starter.contracts import Candidate, Context, RankingResult, SessionState
from starter.ranking import (
    DIAGNOSTIC_KEYS,
    MATCH,
    UNKNOWN,
    VIOLATION,
    active_constraints,
    classify,
    rank,
)

# CP 6.7 freeze guard: the diagnostics contract reviewers consume. Changing
# this set is an interface change and must be deliberate.
FROZEN_DIAGNOSTIC_KEYS = {
    "base_score",
    "attribute_score",
    "violation_penalty",
    "profile_score",
    "final_score",
    "rank",
    "matched",
    "violated",
    "route_sources",
}

_EMPTY_META = {
    "color": set(), "material": set(), "cats": set(),
    "store": "", "sizes": set(), "price": None, "traits": set(),
}


def _meta(**overrides) -> dict:
    return {**_EMPTY_META, **overrides}


def _candidate(asin: str, fusion: float = 0.01, routes=("bm25",)) -> Candidate:
    return Candidate(
        parent_asin=asin,
        route_scores={name: 1.0 for name in routes},
        metadata={"fusion_score": fusion},
    )


def _context(**slots) -> Context:
    state = SessionState(session_id="s")
    for name, values in slots.items():
        cardinality = "single" if name in ("category", "size", "brand", "budget") else "multi"
        entry = {"values": list(values), "cardinality": cardinality}
        if name == "budget":
            entry["bounds"] = slots[name][1] if isinstance(values, tuple) else None
        state.slots[name] = entry
    return Context(session_id="s", turn=1, user_message="", state=state)


def _budget_context(raw: str, low, high) -> Context:
    state = SessionState(session_id="s")
    state.slots["budget"] = {
        "values": [raw], "cardinality": "single", "bounds": {"min": low, "max": high},
    }
    return Context(session_id="s", turn=1, user_message="", state=state)


class CP61OneAttributeBoost(unittest.TestCase):
    def test_matching_candidate_outranks_comparable_non_match(self) -> None:
        ctx = _context(color=["black"])
        candidates = [_candidate("MISS", 0.01), _candidate("HIT", 0.01)]
        metadata = {"MISS": _meta(color={"blue"}), "HIT": _meta(color={"black"})}
        result = rank(candidates, ctx, metadata, 10)
        self.assertEqual([c.parent_asin for c in result.ranked], ["HIT", "MISS"])

    def test_boost_can_overcome_a_better_base_score(self) -> None:
        ctx = _context(color=["black"])
        candidates = [_candidate("STRONGBASE", 0.049), _candidate("MATCHES", 0.003)]
        metadata = {"STRONGBASE": _meta(color={"blue"}), "MATCHES": _meta(color={"black"})}
        result = rank(candidates, ctx, metadata, 10)
        self.assertEqual(result.ranked[0].parent_asin, "MATCHES")

    def test_with_no_active_constraints_base_order_is_preserved(self) -> None:
        ctx = _context()
        candidates = [_candidate("LOW", 0.003), _candidate("HIGH", 0.049)]
        result = rank(candidates, ctx, {"LOW": _meta(), "HIGH": _meta()}, 10)
        self.assertEqual([c.parent_asin for c in result.ranked], ["HIGH", "LOW"])


class CP62MultipleMatches(unittest.TestCase):
    def test_three_match_candidate_outranks_one_match(self) -> None:
        ctx = _context(color=["black"], material=["denim"], category=["jacket"])
        candidates = [_candidate("ONE", 0.02), _candidate("THREE", 0.02)]
        metadata = {
            "ONE": _meta(cats={"jackets"}),
            "THREE": _meta(color={"black"}, material={"denim"}, cats={"jackets"}),
        }
        result = rank(candidates, ctx, metadata, 10)
        self.assertEqual([c.parent_asin for c in result.ranked], ["THREE", "ONE"])
        self.assertEqual(len(result.diagnostics["THREE"]["matched"]), 3)
        self.assertEqual(len(result.diagnostics["ONE"]["matched"]), 1)

    def test_match_count_orders_monotonically(self) -> None:
        ctx = _context(color=["black"], material=["denim"], category=["jacket"])
        metadata = {
            "M0": _meta(),
            "M1": _meta(cats={"jackets"}),
            "M2": _meta(color={"black"}, cats={"jackets"}),
            "M3": _meta(color={"black"}, material={"denim"}, cats={"jackets"}),
        }
        candidates = [_candidate(a, 0.02) for a in ("M0", "M1", "M2", "M3")]
        result = rank(candidates, ctx, metadata, 10)
        self.assertEqual([c.parent_asin for c in result.ranked], ["M3", "M2", "M1", "M0"])


class CP63BoundedViolationPenalty(unittest.TestCase):
    def test_violation_is_finite_and_bounded(self) -> None:
        ctx = _context(color=["black"], material=["denim"])
        candidates = [_candidate("BAD", 0.02)]
        metadata = {"BAD": _meta(color={"blue"}, material={"wool"})}
        result = rank(candidates, ctx, metadata, 10)
        detail = result.diagnostics["BAD"]
        self.assertTrue(math.isfinite(detail["final_score"]))
        self.assertLessEqual(detail["violation_penalty"], ranking.W_PENALTY)
        self.assertGreaterEqual(detail["violation_penalty"], 0.0)

    def test_violating_candidate_is_not_eliminated(self) -> None:
        ctx = _context(color=["black"])
        candidates = [_candidate("BAD", 0.02)]
        result = rank(candidates, ctx, {"BAD": _meta(color={"blue"})}, 10)
        self.assertEqual([c.parent_asin for c in result.ranked], ["BAD"])

    def test_a_violator_still_outranks_nothing_relevant_when_base_is_strong(self) -> None:
        # One violation must not send a candidate to the bottom irreversibly:
        # a large enough base advantage still wins.
        ctx = _context(color=["black"])
        candidates = [_candidate("VIOLATOR", 0.049), _candidate("UNKNOWN", 0.0001)]
        metadata = {"VIOLATOR": _meta(color={"blue"}), "UNKNOWN": _meta()}
        result = rank(candidates, ctx, metadata, 10)
        self.assertEqual(result.ranked[0].parent_asin, "VIOLATOR")

    def test_all_violations_still_yield_a_finite_score(self) -> None:
        ctx = _context(color=["black"], material=["denim"], category=["jacket"])
        metadata = {"ALLBAD": _meta(color={"blue"}, material={"wool"}, cats={"socks"})}
        result = rank([_candidate("ALLBAD", 0.01)], ctx, metadata, 10)
        detail = result.diagnostics["ALLBAD"]
        self.assertTrue(math.isfinite(detail["final_score"]))
        self.assertAlmostEqual(detail["violation_penalty"], ranking.W_PENALTY)


class CP64UnknownIsNotViolation(unittest.TestCase):
    def test_missing_colour_is_unknown_not_violation(self) -> None:
        self.assertEqual(classify("color", ["black"], _meta()), UNKNOWN)
        self.assertEqual(classify("color", ["black"], _meta(color={"blue"})), VIOLATION)
        self.assertEqual(classify("color", ["black"], _meta(color={"black"})), MATCH)

    def test_missing_material_price_store_are_unknown(self) -> None:
        self.assertEqual(classify("material", ["denim"], _meta()), UNKNOWN)
        self.assertEqual(classify("brand", ["nike"], _meta()), UNKNOWN)
        self.assertEqual(
            classify("budget", ["under $50"], _meta(), {"min": None, "max": 50.0}),
            UNKNOWN,
        )

    def test_unknown_candidate_outranks_a_violating_one(self) -> None:
        ctx = _context(color=["black"])
        candidates = [_candidate("VIOLATES", 0.01), _candidate("SILENT", 0.01)]
        metadata = {"VIOLATES": _meta(color={"blue"}), "SILENT": _meta()}
        result = rank(candidates, ctx, metadata, 10)
        self.assertEqual([c.parent_asin for c in result.ranked], ["SILENT", "VIOLATES"])

    def test_unknown_is_never_counted_as_a_violation(self) -> None:
        ctx = _context(color=["black"], material=["denim"])
        metadata = {"HALF": _meta(color={"black"})}  # material unknown
        result = rank([_candidate("HALF", 0.01)], ctx, metadata, 10)
        detail = result.diagnostics["HALF"]
        self.assertEqual(detail["matched"], ["color"])
        self.assertEqual(detail["violated"], [])
        self.assertEqual(detail["violation_penalty"], 0.0)
        self.assertGreater(detail["attribute_score"], 0.0)

    def test_denominator_is_the_active_constraint_count(self) -> None:
        """The reported gap: the previous assertion (attribute_score > 0)
        held under either denominator convention, so it could not tell them
        apart. Pin the exact value instead."""
        metadata = {"HALF": _meta(color={"black"})}  # material is UNKNOWN

        def attribute_score(ctx) -> float:
            result = rank([_candidate("HALF", 0.01)], ctx, metadata, 10)
            return result.diagnostics["HALF"]["attribute_score"]

        self.assertAlmostEqual(
            attribute_score(_context(color=["black"])), ranking.W_MATCH)
        self.assertAlmostEqual(
            attribute_score(_context(color=["black"], material=["denim"])),
            ranking.W_MATCH / 2)

    def test_unknown_rescales_but_never_reorders(self) -> None:
        """Because the denominator is the active constraint count, it is
        identical for every candidate in a turn -- so an UNKNOWN slot cannot
        change who outranks whom. This is what makes CP 6.4 hold."""
        metadata = {
            "TWO": _meta(color={"black"}, material={"denim"}),
            "ONE": _meta(color={"black"}),
            "ZERO": _meta(),
        }
        candidates = [_candidate(a, 0.01) for a in ("ZERO", "ONE", "TWO")]
        expected = ["TWO", "ONE", "ZERO"]
        for ctx in (
            _context(color=["black"], material=["denim"]),
            # A third, wholly unknown slot rescales every score; order holds.
            _context(color=["black"], material=["denim"], brand=["nike"]),
        ):
            result = rank(candidates, ctx, metadata, 10)
            self.assertEqual([c.parent_asin for c in result.ranked], expected)

    def test_thin_metadata_does_not_outrank_stronger_evidence(self) -> None:
        """Why per-candidate "known verdicts only" was rejected: it would let
        a product the catalog barely describes beat one that demonstrably
        matches more constraints."""
        ctx = _context(color=["black"], material=["denim"], category=["jacket"])
        metadata = {
            # demonstrates two matches; catalog describes it fully
            "RICH": _meta(color={"black"}, material={"denim"}, cats={"socks"}),
            # demonstrates one match; catalog says nothing else about it
            "SPARSE": _meta(color={"black"}),
        }
        result = rank([_candidate("RICH", 0.02), _candidate("SPARSE", 0.02)],
                      ctx, metadata, 10)
        self.assertGreater(
            result.diagnostics["RICH"]["attribute_score"],
            result.diagnostics["SPARSE"]["attribute_score"],
            "more demonstrated matches must earn a higher attribute score",
        )

    def test_size_mismatch_is_never_a_violation(self) -> None:
        # Size metadata is too sparse to treat a miss as evidence against.
        self.assertEqual(classify("size", ["10"], _meta(sizes={"12"})), UNKNOWN)
        self.assertEqual(classify("size", ["10"], _meta(sizes={"10"})), MATCH)
        self.assertEqual(classify("size", ["10"], _meta()), UNKNOWN)

    def test_product_absent_from_metadata_is_all_unknown(self) -> None:
        ctx = _context(color=["black"], material=["denim"])
        result = rank([_candidate("GHOST", 0.01)], ctx, {}, 10)
        detail = result.diagnostics["GHOST"]
        self.assertEqual(detail["matched"], [])
        self.assertEqual(detail["violated"], [])
        self.assertEqual(detail["final_score"], detail["base_score"])


class MultiWordCategoryTest(unittest.TestCase):
    """Phase 6 review Finding 2 -- a multi-word category value is a
    conjunction. Matching on any single word made "swim trunks" match
    "Women's Swimwear" and "tank top" match "Topcoats"."""

    @staticmethod
    def _cats(*names: str) -> dict:
        from starter.catalog_meta import signals

        cats = signals({
            "parent_asin": "X", "title": "", "categories": list(names),
            "features": [], "details": {}, "store": "", "description": [],
        })[2]
        return _meta(cats=set(cats.split()))

    def test_shared_first_word_is_not_a_match(self) -> None:
        for names, value in (
            (("Clothing", "Women's Swimwear"), "swim trunks"),
            (("Clothing", "Swim Goggles"), "swim trunks"),
            (("Clothing", "Tops & Tees"), "tank top"),
            (("Clothing", "Topcoats"), "tank top"),
            (("Clothing", "Water Tanks"), "tank top"),
        ):
            self.assertEqual(
                classify("category", [value], self._cats(*names)), VIOLATION,
                msg=f"{value!r} must not match {names}",
            )

    def test_genuine_multi_word_category_still_matches(self) -> None:
        self.assertEqual(
            classify("category", ["swim trunks"],
                     self._cats("Clothing", "Men's Swim Trunks")), MATCH)
        self.assertEqual(
            classify("category", ["tank top"],
                     self._cats("Clothing", "Tank Tops")), MATCH)

    def test_single_word_plural_tolerance_is_unaffected(self) -> None:
        self.assertEqual(
            classify("category", ["jacket"], self._cats("Clothing", "Jackets")), MATCH)
        self.assertEqual(
            classify("category", ["sock"], self._cats("Clothing", "Socks")), MATCH)
        self.assertEqual(
            classify("category", ["jacket"], self._cats("Clothing", "Socks")), VIOLATION)

    def test_every_multi_word_canonical_category_is_covered(self) -> None:
        # Guard: if state.py gains another multi-word canonical category,
        # this conjunction rule must be considered for it too.
        from starter.state import _CATEGORY_KEYWORDS

        multi_word = {v for v in _CATEGORY_KEYWORDS.values() if " " in v}
        self.assertEqual(multi_word, {"swim trunks", "tank top"})


class CP65StableSorting(unittest.TestCase):
    def test_identical_input_gives_identical_output(self) -> None:
        ctx = _context(color=["black"])
        candidates = [_candidate(f"P{i}", 0.01) for i in range(20)]
        metadata = {f"P{i}": _meta(color={"black"}) for i in range(20)}
        first = [c.parent_asin for c in rank(candidates, ctx, metadata, 10).ranked]
        second = [c.parent_asin for c in rank(candidates, ctx, metadata, 10).ranked]
        self.assertEqual(first, second)

    def test_full_ties_break_deterministically_on_parent_asin(self) -> None:
        ctx = _context()
        candidates = [_candidate(a, 0.01) for a in ("CCC", "AAA", "BBB")]
        metadata = {a: _meta() for a in ("CCC", "AAA", "BBB")}
        result = rank(candidates, ctx, metadata, 10)
        self.assertEqual([c.parent_asin for c in result.ranked], ["AAA", "BBB", "CCC"])

    def test_input_order_does_not_change_the_result(self) -> None:
        ctx = _context(material=["denim"])
        metadata = {"A": _meta(material={"denim"}), "B": _meta(material={"wool"}),
                    "C": _meta()}
        forward = [_candidate(a, 0.01) for a in ("A", "B", "C")]
        reverse = [_candidate(a, 0.01) for a in ("C", "B", "A")]
        self.assertEqual(
            [c.parent_asin for c in rank(forward, ctx, metadata, 10).ranked],
            [c.parent_asin for c in rank(reverse, ctx, metadata, 10).ranked],
        )


class CP66UniqueTopTen(unittest.TestCase):
    def test_duplicate_candidates_are_collapsed(self) -> None:
        ctx = _context()
        candidates = [_candidate("DUP", 0.02), _candidate("DUP", 0.01),
                      _candidate("OTHER", 0.015)]
        result = rank(candidates, ctx, {"DUP": _meta(), "OTHER": _meta()}, 10)
        asins = [c.parent_asin for c in result.ranked]
        self.assertEqual(len(asins), len(set(asins)))
        self.assertEqual(sorted(asins), ["DUP", "OTHER"])

    def test_top_k_is_respected(self) -> None:
        ctx = _context()
        candidates = [_candidate(f"P{i:03d}", 0.01) for i in range(50)]
        metadata = {f"P{i:03d}": _meta() for i in range(50)}
        result = rank(candidates, ctx, metadata, 10)
        self.assertEqual(len(result.ranked), 10)
        self.assertEqual(len(result.diagnostics), 10)

    def test_zero_and_negative_top_k(self) -> None:
        ctx = _context()
        candidates = [_candidate("A", 0.01)]
        self.assertEqual(rank(candidates, ctx, {"A": _meta()}, 0).ranked, [])
        self.assertEqual(rank(candidates, ctx, {"A": _meta()}, -5).ranked, [])

    def test_empty_pool(self) -> None:
        result = rank([], _context(color=["black"]), {}, 10)
        self.assertIsInstance(result, RankingResult)
        self.assertEqual(result.ranked, [])
        self.assertEqual(result.diagnostics, {})


class CP67RankingDiagnostics(unittest.TestCase):
    def _result(self) -> RankingResult:
        ctx = _context(color=["black"], material=["denim"])
        candidates = [_candidate("A", 0.02, ("bm25", "category")),
                      _candidate("B", 0.01, ("attribute",))]
        metadata = {"A": _meta(color={"black"}, material={"denim"}),
                    "B": _meta(color={"blue"})}
        return rank(candidates, ctx, metadata, 10)

    def test_diagnostics_key_set_is_frozen(self) -> None:
        # Mirrors the contracts freeze guard: changing the diagnostics
        # contract requires updating this set deliberately.
        self.assertEqual(set(DIAGNOSTIC_KEYS), FROZEN_DIAGNOSTIC_KEYS)
        for entry in self._result().diagnostics.values():
            self.assertEqual(set(entry), FROZEN_DIAGNOSTIC_KEYS)

    def test_every_ranked_candidate_has_diagnostics(self) -> None:
        result = self._result()
        self.assertEqual(
            {c.parent_asin for c in result.ranked}, set(result.diagnostics)
        )

    def test_score_components_reconcile(self) -> None:
        for entry in self._result().diagnostics.values():
            self.assertAlmostEqual(
                entry["final_score"],
                entry["base_score"] + entry["attribute_score"]
                - entry["violation_penalty"] + entry["profile_score"],
            )

    def test_rank_is_one_based_and_contiguous(self) -> None:
        result = self._result()
        ranks = sorted(e["rank"] for e in result.diagnostics.values())
        self.assertEqual(ranks, list(range(1, len(ranks) + 1)))

    def test_diagnostics_expose_matched_violated_and_routes(self) -> None:
        result = self._result()
        self.assertEqual(sorted(result.diagnostics["A"]["matched"]), ["color", "material"])
        self.assertEqual(result.diagnostics["A"]["violated"], [])
        self.assertEqual(sorted(result.diagnostics["A"]["route_sources"]),
                         ["bm25", "category"])
        self.assertEqual(result.diagnostics["B"]["violated"], ["color"])


class BudgetConstraintTest(unittest.TestCase):
    def test_price_within_bounds_matches(self) -> None:
        ctx = _budget_context("under $100", None, 100.0)
        result = rank([_candidate("CHEAP", 0.01)], ctx, {"CHEAP": _meta(price=50.0)}, 10)
        self.assertEqual(result.diagnostics["CHEAP"]["matched"], ["budget"])

    def test_price_above_max_violates(self) -> None:
        ctx = _budget_context("under $100", None, 100.0)
        result = rank([_candidate("PRICEY", 0.01)], ctx, {"PRICEY": _meta(price=500.0)}, 10)
        self.assertEqual(result.diagnostics["PRICEY"]["violated"], ["budget"])

    def test_price_below_min_violates(self) -> None:
        ctx = _budget_context("over $200", 200.0, None)
        result = rank([_candidate("CHEAP", 0.01)], ctx, {"CHEAP": _meta(price=10.0)}, 10)
        self.assertEqual(result.diagnostics["CHEAP"]["violated"], ["budget"])

    def test_missing_price_is_unknown(self) -> None:
        ctx = _budget_context("under $100", None, 100.0)
        result = rank([_candidate("NOPRICE", 0.01)], ctx, {"NOPRICE": _meta()}, 10)
        self.assertEqual(result.diagnostics["NOPRICE"]["matched"], [])
        self.assertEqual(result.diagnostics["NOPRICE"]["violated"], [])


class ActiveConstraintsTest(unittest.TestCase):
    def test_only_active_slot_values_are_used(self) -> None:
        state = SessionState(session_id="s")
        from starter.state import update_state

        update_state(state, "black leather jacket", 1)
        update_state(state, "actually denim", 2)
        ctx = Context(session_id="s", turn=2, user_message="actually denim", state=state)
        constraints, _ = active_constraints(ctx)
        self.assertEqual(constraints["material"], ["denim"])
        self.assertNotIn("leather", constraints["material"])
        self.assertEqual(constraints["color"], ["black"])
        self.assertEqual(constraints["category"], ["jacket"])

    def test_ranking_does_not_mutate_state(self) -> None:
        import copy

        ctx = _context(color=["black"])
        snapshot = copy.deepcopy(ctx.state)
        rank([_candidate("A", 0.01)], ctx, {"A": _meta()}, 10)
        self.assertEqual(ctx.state, snapshot)


if __name__ == "__main__":
    unittest.main()
