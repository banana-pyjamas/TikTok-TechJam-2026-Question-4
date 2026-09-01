"""CP 12.1 - 12.4 — the popularity prior.

A prior is what you use BEFORE evidence. Everything here is about keeping it
in that role: bounded below one satisfied constraint, decaying as the shopper
says more, and neutral -- never punitive -- when the catalog says nothing.
"""

from __future__ import annotations

import math
import sqlite3
import unittest
from unittest import mock

from starter import ranking
from starter.catalog_meta import TABLE, create_table, popularity_scale, signals
from starter.contracts import Candidate, Context, SessionState
from starter.popularity import (DEFAULT_MISSING, DEFAULT_SCALE, W_POPULARITY,
                                evidence_decay, normalized, popularity_feature,
                                popularity_score)
from starter.ranking import POPULARITY_KEY, W_MATCH, rank

_EMPTY_META = {"color": set(), "material": set(), "cats": set(), "store": "",
               "sizes": set(), "price": None, "traits": set(),
               "popularity": None}


def _meta(**overrides):
    return {**_EMPTY_META, **overrides}


def _product(asin: str, rating_number, title: str = "Jacket") -> dict:
    return {"parent_asin": asin, "title": title, "categories": ["Clothing"],
            "features": [], "details": {}, "store": "acme", "description": [],
            "rating_number": rating_number}


def _connection(products) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    create_table(connection)
    rows = [(str(p["parent_asin"]), *signals(p)) for p in products]
    if rows:
        connection.executemany(
            f"INSERT OR REPLACE INTO {TABLE} VALUES "
            f"({', '.join('?' * len(rows[0]))})", rows)
    return connection


def _context(scale, **slots) -> Context:
    state = SessionState(session_id="s")
    for name, values in slots.items():
        cardinality = "multi" if name in ("color", "material") else "single"
        state.slots[name] = {"values": list(values), "cardinality": cardinality}
    context = Context(session_id="s", turn=1, user_message="", state=state)
    context.derived[POPULARITY_KEY] = scale
    return context


def _candidate(asin: str, fusion: float = 0.02) -> Candidate:
    return Candidate(parent_asin=asin, metadata={"fusion_score": fusion})


class CP121FeatureTest(unittest.TestCase):
    def test_the_feature_is_log1p_of_the_review_count(self) -> None:
        for count in (0, 1, 12, 3332, 408371):
            self.assertAlmostEqual(popularity_feature(count), math.log1p(count))

    def test_log1p_is_defined_at_zero(self) -> None:
        # A product with no reviews YET is a real, representable value -- and
        # must stay distinguishable from "the catalog did not say".
        self.assertEqual(popularity_feature(0), 0.0)
        self.assertIsNone(popularity_feature(None))

    def test_it_compresses_the_four_order_of_magnitude_range(self) -> None:
        smallest = popularity_feature(1)
        largest = popularity_feature(408371)
        self.assertLess(largest / smallest, 20,
                        "raw counts would differ by 400,000x")

    def test_the_scale_is_read_from_the_catalog(self) -> None:
        connection = _connection([_product(f"P{n}", n) for n in (0, 5, 50, 5000)])
        scale = popularity_scale(connection)
        self.assertAlmostEqual(scale["scale"], math.log1p(5000))
        self.assertAlmostEqual(scale["missing"], math.log1p(50))

    def test_normalized_lands_in_the_unit_interval(self) -> None:
        scale = {"scale": math.log1p(408371), "missing": math.log1p(12)}
        for count in (0, 1, 12, 3332, 408371):
            value = normalized(popularity_feature(count), scale)
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        self.assertAlmostEqual(normalized(popularity_feature(408371), scale), 1.0)

    def test_more_reviews_scores_higher(self) -> None:
        scale = {"scale": math.log1p(10000), "missing": math.log1p(10)}
        ordered = [normalized(popularity_feature(n), scale)
                   for n in (0, 10, 100, 1000, 10000)]
        self.assertEqual(ordered, sorted(ordered))


class CP122MissingSafetyTest(unittest.TestCase):
    """A missing review count means "unknown", and unknown must be TYPICAL.

    Note this is a robustness property, not a live gap: every product in the
    frozen catalog carries a rating_number. It is tested because the private
    800 and any future catalog need not be so tidy, and because scoring an
    unrated product as unpopular is exactly the "absent metadata becomes a
    penalty" failure that CP 6.4 rules out elsewhere.
    """

    SCALE = {"scale": math.log1p(10000), "missing": math.log1p(12)}

    def test_unusable_counts_become_none_not_zero(self) -> None:
        for bad in (None, "many", [], {}, float("nan"), float("inf"), -1, True):
            self.assertIsNone(popularity_feature(bad), repr(bad))

    def test_a_missing_count_scores_as_the_catalog_median(self) -> None:
        expected = normalized(self.SCALE["missing"], self.SCALE)
        self.assertAlmostEqual(normalized(None, self.SCALE), expected)

    def test_missing_beats_genuinely_unpopular(self) -> None:
        # The load-bearing direction: absence must not rank below a product
        # the catalog says nobody reviewed.
        self.assertGreater(normalized(None, self.SCALE),
                           normalized(popularity_feature(0), self.SCALE))

    def test_missing_loses_to_genuinely_popular(self) -> None:
        self.assertLess(normalized(None, self.SCALE),
                        normalized(popularity_feature(10000), self.SCALE))

    def test_a_scale_with_no_median_falls_back_mid_range(self) -> None:
        self.assertAlmostEqual(normalized(None, {"scale": 10.0}), DEFAULT_MISSING)
        self.assertAlmostEqual(
            normalized(None, {"scale": 10.0, "missing": float("nan")}),
            DEFAULT_MISSING)

    def test_no_statistics_means_no_popularity_signal(self) -> None:
        # Rather than inventing a scale, the prior switches itself off.
        self.assertEqual(popularity_scale(sqlite3.connect(":memory:")), {})
        empty = sqlite3.connect(":memory:")
        create_table(empty)
        self.assertEqual(popularity_scale(empty), {})
        self.assertEqual(normalized(5.0, {}), 0.0)
        self.assertEqual(popularity_score(_meta(popularity=5.0), {}, 0.0), 0.0)
        self.assertEqual(popularity_score(_meta(), None, 0.0), 0.0)

    def test_a_catalog_where_nothing_has_a_count(self) -> None:
        connection = _connection([_product(f"P{n}", None) for n in range(5)])
        self.assertEqual(popularity_scale(connection), {})

    def test_a_malformed_scale_does_not_raise(self) -> None:
        for bad in ({"scale": 0}, {"scale": -1}, {"scale": float("nan")},
                    {"scale": "x"}, {"scale": None}):
            value = normalized(2.0, {**bad, "missing": 1.0})
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_score_is_total_on_junk_metadata(self) -> None:
        scale = {"scale": 10.0, "missing": 2.0}
        for meta in (None, {}, {"popularity": "x"}, {"popularity": None}):
            value = popularity_score(meta, scale, 0.0)
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, W_POPULARITY)


class CP123EvidenceDecayTest(unittest.TestCase):
    def test_the_prior_is_strongest_when_nothing_is_known(self) -> None:
        self.assertAlmostEqual(evidence_decay(0.0), 1.0)

    def test_evidence_shrinks_it_monotonically(self) -> None:
        values = [evidence_decay(e) for e in (0, 0.4, 1, 2, 3, 10)]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_one_firm_constraint_halves_it(self) -> None:
        self.assertAlmostEqual(evidence_decay(1.0), 0.5)
        self.assertAlmostEqual(evidence_decay(2.0), 1 / 3)

    def test_a_hedged_constraint_displaces_less_than_a_firm_one(self) -> None:
        self.assertGreater(evidence_decay(0.4), evidence_decay(1.0))

    def test_it_never_reaches_zero_or_goes_negative(self) -> None:
        for evidence in (0, 1, 100, 1e6):
            self.assertGreater(evidence_decay(evidence), 0.0)
        self.assertAlmostEqual(evidence_decay(-5), 1.0, msg="clamped at 0")
        self.assertAlmostEqual(evidence_decay(float("nan")), 1.0)

    def test_the_decay_reaches_the_score(self) -> None:
        scale = {"scale": 10.0, "missing": 5.0}
        meta = _meta(popularity=10.0)
        self.assertAlmostEqual(popularity_score(meta, scale, 0.0), W_POPULARITY)
        self.assertAlmostEqual(popularity_score(meta, scale, 1.0),
                               W_POPULARITY / 2)

    def test_decay_is_driven_by_state_not_by_the_phase_11_flag(self) -> None:
        # EC is state, not scoring, so CP 12.3 works with Phase 11's scoring
        # flag in either position. The two flags stay independent.
        state = SessionState(session_id="s")
        state.slots["material"] = {"values": ["leather"], "cardinality": "multi",
                                   "confidence": 0.4}
        context = Context(session_id="s", turn=1, user_message="", state=state)
        context.derived[POPULARITY_KEY] = {"scale": 10.0, "missing": 5.0}
        pool = [_candidate("A")]
        metadata = {"A": _meta(popularity=10.0)}
        scores = {}
        for weighting in (False, True):
            with mock.patch.object(ranking, "USE_POPULARITY", True), \
                 mock.patch.object(ranking, "USE_CONFIDENCE_WEIGHTING", weighting):
                scores[weighting] = rank(pool, context, metadata, 10) \
                    .diagnostics["A"]["popularity_score"]
        self.assertAlmostEqual(scores[False], scores[True])
        # 0.4 of evidence, so the prior keeps 1/1.4 of its strength.
        self.assertAlmostEqual(scores[False], W_POPULARITY / 1.4)


class CP124NoBestsellerDominationTest(unittest.TestCase):
    """The guarantee that makes the prior safe to ship.

    Held by arithmetic, not by tuning: the entire popularity term is bounded
    by W_POPULARITY, an order of magnitude below the W_MATCH a single
    satisfied constraint earns -- and CP 12.3's decay shrinks it further the
    moment any constraint exists at all.
    """

    SCALE = {"scale": math.log1p(408371), "missing": math.log1p(12)}

    def test_the_whole_prior_is_worth_less_than_one_match(self) -> None:
        self.assertLess(W_POPULARITY * 10, W_MATCH,
                        "popularity must be an order of magnitude below a match")

    def test_the_most_popular_product_cannot_outrank_a_match(self) -> None:
        context = _context(self.SCALE, material=["leather"])
        pool = [_candidate("BESTSELLER", 0.02), _candidate("MATCHES", 0.02)]
        metadata = {
            "BESTSELLER": _meta(popularity=popularity_feature(408371),
                                material={"denim"}),
            "MATCHES": _meta(popularity=popularity_feature(0),
                             material={"leather"}),
        }
        with mock.patch.object(ranking, "USE_POPULARITY", True):
            result = rank(pool, context, metadata, 10)
        self.assertEqual(result.ranked[0].parent_asin, "MATCHES")

    def test_a_specific_query_does_not_collapse_into_the_bestseller_order(self) -> None:
        # Six candidates whose popularity order is the exact reverse of their
        # constraint-satisfaction order. A specific query must follow the
        # constraints, not the review counts.
        context = _context(self.SCALE, material=["leather"])
        counts = [408371, 100000, 10000, 1000, 100, 0]
        pool = [_candidate(f"P{i}", 0.02) for i in range(6)]
        metadata = {
            f"P{i}": _meta(popularity=popularity_feature(counts[i]),
                           material={"leather"} if i >= 3 else {"denim"})
            for i in range(6)
        }
        with mock.patch.object(ranking, "USE_POPULARITY", True):
            order = [c.parent_asin for c in rank(pool, context, metadata, 10).ranked]
        bestseller_order = [f"P{i}" for i in range(6)]
        self.assertNotEqual(order, bestseller_order)
        # Every matching candidate outranks every violating one.
        self.assertEqual(set(order[:3]), {"P3", "P4", "P5"})

    def test_popularity_only_breaks_ties(self) -> None:
        # With nothing to separate them, the prior is allowed to decide -- that
        # is its entire job.
        context = _context(self.SCALE)
        pool = [_candidate("QUIET", 0.02), _candidate("LOUD", 0.02)]
        metadata = {"QUIET": _meta(popularity=popularity_feature(1)),
                    "LOUD": _meta(popularity=popularity_feature(408371))}
        with mock.patch.object(ranking, "USE_POPULARITY", True):
            result = rank(pool, context, metadata, 10)
        self.assertEqual(result.ranked[0].parent_asin, "LOUD")

    def test_it_cannot_overturn_a_clear_retrieval_advantage(self) -> None:
        context = _context(self.SCALE)
        pool = [_candidate("RELEVANT", 0.049), _candidate("POPULAR", 0.02)]
        metadata = {"RELEVANT": _meta(popularity=popularity_feature(0)),
                    "POPULAR": _meta(popularity=popularity_feature(408371))}
        with mock.patch.object(ranking, "USE_POPULARITY", True):
            result = rank(pool, context, metadata, 10)
        self.assertEqual(result.ranked[0].parent_asin, "RELEVANT")

    def test_the_term_is_bounded_everywhere(self) -> None:
        for count in (0, 1, 12, 408371, None):
            for evidence in (0.0, 0.4, 1.0, 5.0):
                value = popularity_score(
                    _meta(popularity=popularity_feature(count)),
                    self.SCALE, evidence)
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, W_POPULARITY)


class FlagOffIsExactTest(unittest.TestCase):
    def test_off_adds_no_popularity_term(self) -> None:
        context = _context({"scale": 10.0, "missing": 5.0}, material=["leather"])
        pool = [_candidate("A")]
        metadata = {"A": _meta(popularity=10.0, material={"leather"})}
        with mock.patch.object(ranking, "USE_POPULARITY", False):
            detail = rank(pool, context, metadata, 10).diagnostics["A"]
        self.assertEqual(detail["popularity_score"], 0.0)
        self.assertAlmostEqual(
            detail["final_score"],
            detail["base_score"] + detail["attribute_score"]
            - detail["violation_penalty"] + detail["profile_score"])

    def test_the_diagnostics_key_exists_in_both_positions(self) -> None:
        context = _context({"scale": 10.0, "missing": 5.0})
        pool = [_candidate("A")]
        for enabled in (False, True):
            with mock.patch.object(ranking, "USE_POPULARITY", enabled):
                detail = rank(pool, context, {"A": _meta()}, 10).diagnostics["A"]
            self.assertIn("popularity_score", detail)


if __name__ == "__main__":
    unittest.main()


class Phase11InteractionTest(unittest.TestCase):
    """D Phase 12 review, P4 — CP 12.4's arithmetic bound is CONDITIONAL.

    With USE_CONFIDENCE_WEIGHTING OFF, a satisfied constraint contributes
    W_MATCH / n and the prior is bounded by W_POPULARITY, so the inequality
    holds for any n. With it ON, the attribute term is additionally scaled by
    EC * MR, which has no lower bound near 1 -- and the inequality can invert.

    USE_CONFIDENCE_WEIGHTING ships False, so this is latent. It is pinned here
    so that turning that flag on cannot happen without this failing and
    forcing W_POPULARITY to be re-derived.
    """

    def test_the_bound_holds_while_phase_11_weighting_is_off(self) -> None:
        # Read the SOURCE, not the live attribute: another test in the same
        # process can write the committed value back over a source edit, which
        # is precisely how this guard passed in the full suite while failing
        # when run alone (D Phase 12 review, Q2).
        from tools import config_guard

        self.assertFalse(
            config_guard.source_flags()[("ranking", "USE_CONFIDENCE_WEIGHTING")],
            "if this flag ships True, re-derive W_POPULARITY")
        # Weakest possible match under OFF: one of n constraints matched.
        for n in (1, 2, 3, 6):
            weakest_match = W_MATCH / n
            strongest_prior = W_POPULARITY  # decay <= 1
            self.assertGreater(weakest_match, strongest_prior, f"n={n}")

    def test_the_bound_inverts_if_phase_11_weighting_is_turned_on(self) -> None:
        # Two hedged constraints on poorly-attested slots.
        weight = 0.4 * 0.1          # EC_HEDGED * MIN_RELIABILITY
        matched_term = W_MATCH * weight / 2
        prior_term = W_POPULARITY * evidence_decay(0.4 + 0.4)
        self.assertLess(matched_term, prior_term,
                        "this asserts the KNOWN defect: if it starts passing "
                        "the interaction was fixed and the docstring in "
                        "starter/popularity.py must be updated")

    def test_the_interaction_is_documented_where_the_flag_lives(self) -> None:
        import inspect

        from starter import popularity as popularity_module

        self.assertIn("USE_CONFIDENCE_WEIGHTING",
                      inspect.getdoc(popularity_module) or "",
                      "the conditional must be stated beside the bound")
