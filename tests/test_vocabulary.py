"""CP 10.1 - 10.4 — candidate-scoped vocabulary.

The vocabulary is a property of the CURRENT pool, not of the catalog. Most of
what these tests pin is that scoping: the same word must ground differently in
a pool of jackets and a pool of watches, and must ground to nothing in a pool
that has no word for it.
"""

from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path

from starter.catalog_meta import TABLE, create_table, signals
from starter.contracts import Candidate
from starter.vocabulary import (BOILERPLATE, GROUNDING, INDEX_TERM_LIMIT,
                                MAX_DOCUMENT_RATIO, NOISE_FLOOR_POOL,
                                VOCABULARY_LIMIT, build_vocabulary, ground,
                                ground_all, most_discriminative, pool_terms,
                                product_terms)

_CATALOG = Path("data/catalog.jsonl")


def _product(asin: str, title: str = "", features=(), description=(),
             categories=()) -> dict:
    return {
        "parent_asin": asin,
        "title": title,
        "categories": list(categories),
        "features": list(features),
        "details": {},
        "store": "",
        "description": list(description),
    }


def _connection(products: list[dict]) -> sqlite3.Connection:
    """A meta table populated through the real ``signals`` path."""
    connection = sqlite3.connect(":memory:")
    create_table(connection)
    rows = [(str(p["parent_asin"]), *signals(p)) for p in products]
    placeholders = ", ".join("?" * len(rows[0])) if rows else ""
    if rows:
        connection.executemany(
            f"INSERT OR REPLACE INTO {TABLE} VALUES ({placeholders})", rows)
    return connection


def _pool(*asins: str) -> list[Candidate]:
    return [Candidate(parent_asin=asin) for asin in asins]


class CP101ExtractionTest(unittest.TestCase):
    def test_title_comes_first_so_the_cap_keeps_what_names_the_product(self) -> None:
        extracted = product_terms(_product(
            "A", title="Merino Wool Hiking Socks",
            features=["cushioned"], description=["breathable"]))
        self.assertEqual(extracted[:4], ["merino", "wool", "hiking", "socks"])
        self.assertLess(extracted.index("cushioned"), extracted.index("breathable"))

    def test_the_per_product_list_is_capped(self) -> None:
        many = _product("A", title=" ".join(f"word{n}x" for n in range(200)),
                        features=["alpha beta gamma delta epsilon"] * 40)
        self.assertLessEqual(len(product_terms(many)), INDEX_TERM_LIMIT)

    def test_digits_short_tokens_and_boilerplate_are_dropped(self) -> None:
        extracted = product_terms(_product(
            "A", title="XL 2XL Jacket B07K34RX5J",
            features=["Imported", "Machine Wash Only", "Made in USA"]))
        self.assertIn("jacket", extracted)
        for absent in ("xl", "2xl", "b07k34rx5j", "imported", "machine",
                       "wash", "only", "made", "usa"):
            self.assertNotIn(absent, extracted, absent)

    def test_terms_are_deduplicated(self) -> None:
        extracted = product_terms(_product(
            "A", title="Fleece Fleece Jacket", features=["fleece"]))
        self.assertEqual(extracted.count("fleece"), 1)

    def test_document_frequency_counts_products_not_mentions(self) -> None:
        # One product saying "fleece" ten times must not look like agreement.
        connection = _connection([
            _product("A", title="Fleece", features=["fleece fleece fleece"],
                     description=["fleece fleece fleece fleece"]),
            _product("B", title="Cotton Shirt"),
        ])
        vocabulary = build_vocabulary(connection, _pool("A", "B"))
        # Pool of 2 is below the noise floor, so nothing is frequency-filtered.
        self.assertEqual(vocabulary["terms"]["fleece"], 1)

    def test_ordering_is_deterministic(self) -> None:
        products = [_product(f"P{n}", title="alpha beta gamma") for n in range(12)]
        connection = _connection(products)
        pool = _pool(*[f"P{n}" for n in range(12)])
        first = build_vocabulary(connection, pool)
        second = build_vocabulary(connection, list(reversed(pool)))
        self.assertEqual(list(first["terms"]), list(second["terms"]))


class CP102EmptyPoolSafetyTest(unittest.TestCase):
    """Every function total. A pool that yields nothing must yield an empty
    vocabulary, never an exception and never an un-grounded guess."""

    def test_empty_pool_issues_no_query_and_returns_empty(self) -> None:
        connection = _connection([_product("A", title="Jacket")])
        statements: list[str] = []
        connection.set_trace_callback(statements.append)
        try:
            vocabulary = build_vocabulary(connection, [])
        finally:
            connection.set_trace_callback(None)
        self.assertEqual(vocabulary["pool_size"], 0)
        self.assertEqual(vocabulary["terms"], {})
        self.assertEqual(statements, [], "an empty pool must issue no SQL")

    def test_candidates_missing_from_the_side_table(self) -> None:
        connection = _connection([_product("A", title="Jacket")])
        vocabulary = build_vocabulary(connection, _pool("GHOST", "ALSO_GONE"))
        self.assertEqual(vocabulary["pool_size"], 2)
        self.assertEqual(vocabulary["terms"], {})

    def test_products_with_no_indexed_terms(self) -> None:
        connection = _connection([_product("A"), _product("B")])
        self.assertEqual(build_vocabulary(connection, _pool("A", "B"))["terms"], {})

    def test_duplicate_and_malformed_candidates(self) -> None:
        connection = _connection([_product("A", title="Jacket")])
        pool = _pool("A", "A", "")
        vocabulary = build_vocabulary(connection, pool)
        self.assertEqual(vocabulary["pool_size"], 1, "duplicates count once")

    def test_consumers_are_total_on_an_empty_vocabulary(self) -> None:
        empty = {"pool_size": 0, "terms": {}, "dropped": {}}
        self.assertEqual(ground("warm", empty), ())
        self.assertEqual(most_discriminative(empty, 5), ())
        self.assertEqual(ground_all(["warm", "soft"], empty), {})
        # And on a dict that is not a vocabulary at all.
        self.assertEqual(ground("warm", {}), ())
        self.assertEqual(most_discriminative({}, 5), ())

    def test_pool_terms_with_no_asins_issues_no_query(self) -> None:
        connection = _connection([_product("A", title="Jacket")])
        statements: list[str] = []
        connection.set_trace_callback(statements.append)
        try:
            self.assertEqual(pool_terms(connection, []), {})
            # Empty and None must be filtered out, not stringified: str(None)
            # is "None", which is truthy and would be looked up as an asin.
            self.assertEqual(pool_terms(connection, ["", None]), {})  # type: ignore[list-item]
        finally:
            connection.set_trace_callback(None)
        self.assertEqual(statements, [], "nothing to look up must issue no SQL")


class CP103NoiseControlTest(unittest.TestCase):
    @staticmethod
    def _uniform_pool(count: int, shared: str, unique_prefix: str = "uniq"):
        """``count`` products all sharing ``shared``, each with one own word."""
        return [
            _product(f"P{n}", title=f"{shared} {unique_prefix}{'x' * (n + 1)}")
            for n in range(count)
        ]

    def test_a_term_in_one_candidate_is_dropped(self) -> None:
        products = self._uniform_pool(NOISE_FLOOR_POOL + 4, "jacket")
        connection = _connection(products)
        vocabulary = build_vocabulary(
            connection, _pool(*[p["parent_asin"] for p in products]))
        self.assertGreater(vocabulary["dropped"]["rare"], 0)
        for term in vocabulary["terms"]:
            self.assertFalse(term.startswith("uniq"), term)

    def test_a_near_universal_term_is_dropped(self) -> None:
        # "jacket" is in every candidate: it describes the pool.
        products = self._uniform_pool(20, "jacket")
        for index in range(0, 10):  # give half of them a second shared word
            products[index]["title"] += " hooded hooded2"
        connection = _connection(products)
        vocabulary = build_vocabulary(
            connection, _pool(*[p["parent_asin"] for p in products]))
        self.assertNotIn("jacket", vocabulary["terms"])
        self.assertGreater(vocabulary["dropped"]["ubiquitous"], 0)
        self.assertIn("hooded", vocabulary["terms"], "a half-pool term is kept")

    def test_small_pools_skip_the_frequency_bounds(self) -> None:
        # Below the floor a frequency is not a statistic: with MIN_DF=2 and a
        # 0.8 ceiling, a 3-candidate pool would otherwise be nearly empty.
        products = self._uniform_pool(3, "jacket")
        connection = _connection(products)
        vocabulary = build_vocabulary(
            connection, _pool(*[p["parent_asin"] for p in products]))
        self.assertLess(vocabulary["pool_size"], NOISE_FLOOR_POOL)
        self.assertIn("jacket", vocabulary["terms"])
        self.assertEqual(vocabulary["dropped"]["rare"], 0)
        self.assertEqual(vocabulary["dropped"]["ubiquitous"], 0)

    def test_boilerplate_never_reaches_the_vocabulary(self) -> None:
        products = [
            _product(f"P{n}", title="Jacket Coat",
                     features=["Imported", "Machine Wash", "Made in USA"])
            for n in range(NOISE_FLOOR_POOL + 2)
        ]
        connection = _connection(products)
        vocabulary = build_vocabulary(
            connection, _pool(*[p["parent_asin"] for p in products]))
        for term in vocabulary["terms"]:
            self.assertNotIn(term, BOILERPLATE, term)

    def test_the_cap_is_respected_and_counted(self) -> None:
        # Two halves with disjoint word sets, so every word sits at df 6/12 --
        # inside both bounds, leaving the cap as the only thing that can bind.
        # (Words are letters only: product_terms drops anything with a digit.)
        first = [f"alpha{'a' * n}" for n in range(15)]
        second = [f"beta{'b' * n}" for n in range(15)]
        products = self._uniform_pool(12, "jacket")
        for index, product in enumerate(products):
            product["title"] += " " + " ".join(first if index < 6 else second)
        connection = _connection(products)
        vocabulary = build_vocabulary(
            connection, _pool(*[p["parent_asin"] for p in products]), limit=5)
        self.assertEqual(len(vocabulary["terms"]), 5)
        self.assertGreater(vocabulary["dropped"]["over_limit"], 0)

    def test_the_ceiling_leaves_room_for_discriminative_ordering(self) -> None:
        # Regression on a real design defect: with a 0.5 ceiling, "closest to
        # half" and "most frequent" are the same ordering, and
        # most_discriminative returned exactly the frequency ranking.
        self.assertGreater(MAX_DOCUMENT_RATIO, 0.5)

    def test_most_discriminative_prefers_an_even_split(self) -> None:
        # 20 candidates: "split" in 10, "uncommon" in 3, "filler" in 2.
        products = [_product(f"P{n}", title="common") for n in range(20)]
        for index in range(18, 20):
            products[index]["title"] = "filler"
        for index in range(10):
            products[index]["title"] += " split"
        for index in range(3):
            products[index]["title"] += " uncommon"
        connection = _connection(products)
        vocabulary = build_vocabulary(
            connection, _pool(*[p["parent_asin"] for p in products]))
        ranked = most_discriminative(vocabulary, 4)
        self.assertEqual(ranked[0], "split",
                         "the term splitting the pool in half must rank first")
        self.assertLess(ranked.index("split"), ranked.index("uncommon"))
        # Frequency order alone would not produce this: "split" is both the
        # most frequent survivor and the evenest, so the discriminating case
        # is "uncommon" (3) ranking above "filler" (2).
        self.assertLess(ranked.index("uncommon"), ranked.index("filler"))


class CP104GroundingTest(unittest.TestCase):
    def _jacket_pool(self):
        products = [
            _product(f"J{n}", title="Winter Coat",
                     features=["insulated thermal fleece lined"])
            for n in range(6)
        ] + [
            _product(f"K{n}", title="Winter Coat", features=["shell"])
            for n in range(6)
        ]
        connection = _connection(products)
        pool = _pool(*[p["parent_asin"] for p in products])
        return build_vocabulary(connection, pool)

    def test_the_roadmap_example(self) -> None:
        # warm -> insulated / thermal / fleece-lined
        grounded = ground("warm", self._jacket_pool())
        for expected in ("insulated", "thermal", "fleece", "lined"):
            self.assertIn(expected, grounded, expected)

    def test_only_words_the_pool_actually_uses_come_back(self) -> None:
        vocabulary = self._jacket_pool()
        grounded = ground("warm", vocabulary)
        for word in grounded:
            self.assertIn(word, vocabulary["terms"])
        # These are in the map but not in this pool.
        for absent in ("sherpa", "quilted", "down"):
            self.assertNotIn(absent, grounded, absent)

    def test_the_same_word_grounds_differently_in_a_different_pool(self) -> None:
        # The point of the phase: vocabulary is scoped to the candidates.
        watches = _connection([
            _product(f"W{n}", title="Quartz Watch",
                     features=["water resistant stainless"])
            for n in range(12)
        ])
        watch_vocabulary = build_vocabulary(
            watches, _pool(*[f"W{n}" for n in range(12)]))
        self.assertEqual(ground("warm", watch_vocabulary), ())
        self.assertNotEqual(ground("warm", self._jacket_pool()), ())

    def test_a_word_the_pool_uses_grounds_to_itself(self) -> None:
        self.assertIn("fleece", ground("fleece", self._jacket_pool()))

    def test_an_unmapped_absent_word_grounds_to_nothing(self) -> None:
        self.assertEqual(ground("zzzznotaword", self._jacket_pool()), ())

    def test_ordering_is_by_pool_frequency(self) -> None:
        # "insulated" in 7 of 10, "thermal" in 4 -- both inside the ubiquity
        # ceiling (8), so the gap is a frequency gap and nothing else.
        products = [_product(f"J{n}", title="Coat") for n in range(10)]
        for index in range(7):
            products[index]["features"] = ["insulated"]
        for index in range(4):
            products[index]["features"] = ["insulated thermal"]
        connection = _connection(products)
        vocabulary = build_vocabulary(
            connection, _pool(*[p["parent_asin"] for p in products]))
        grounded = ground("warm", vocabulary)
        self.assertEqual(grounded[0], "insulated",
                         "the pool's commonest word for the idea comes first")

    def test_bad_input_is_safe(self) -> None:
        vocabulary = self._jacket_pool()
        for bad in ("", "   ", None, 7, []):
            self.assertEqual(ground(bad, vocabulary), (), repr(bad))  # type: ignore[arg-type]
        self.assertEqual(ground_all([None, 7, "zzz"], vocabulary), {})  # type: ignore[list-item]
        self.assertEqual(ground_all(None, vocabulary), {})  # type: ignore[arg-type]

    def test_ground_all_keeps_only_what_grounded(self) -> None:
        grounded = ground_all(["warm", "zzzznotaword"], self._jacket_pool())
        self.assertIn("warm", grounded)
        self.assertNotIn("zzzznotaword", grounded)

    def test_the_map_is_ordinary_english_not_harness_phrasing(self) -> None:
        # Same rule as the Phase 9 cue vocabularies: keying on simulator
        # strings would ground the public set well and generalize to nothing.
        for key, values in GROUNDING.items():
            self.assertTrue(key.isalpha() and key.islower(), key)
            for value in values:
                self.assertTrue(value.isalpha() and value.islower(), value)


@unittest.skipUnless(_CATALOG.exists(), "catalog not available")
class RealCatalogGroundingTest(unittest.TestCase):
    """The roadmap example against the real catalog and a real pool."""

    @classmethod
    def setUpClass(cls) -> None:
        from starter.agent import Agent
        from starter.contracts import Context, SessionState
        from starter.retrieval import DEFAULT_ROUTES, POOL_LIMIT, retrieve
        from starter.state import update_state

        cls.agent = Agent(str(_CATALOG))
        message = "I'm looking for a winter jacket, something warm"
        state = SessionState(session_id="v")
        update_state(state, message, 1)
        context = Context(session_id="v", turn=1, user_message=message,
                          state=state)
        pool = retrieve(cls.agent.connection, context, POOL_LIMIT, DEFAULT_ROUTES)
        cls.vocabulary = build_vocabulary(cls.agent.connection, pool)

    def test_warm_grounds_to_the_catalogs_words_for_warm(self) -> None:
        grounded = ground("warm", self.vocabulary)
        for expected in ("insulated", "thermal", "fleece"):
            self.assertIn(expected, grounded, expected)

    def test_the_vocabulary_is_product_language_not_listing_language(self) -> None:
        top = list(self.vocabulary["terms"])[:15]
        for noise in ("hand", "machine", "wash", "only", "imported", "material"):
            self.assertNotIn(noise, top, f"{noise} is listing boilerplate")

    def test_a_real_pool_fills_but_does_not_exceed_the_cap(self) -> None:
        self.assertLessEqual(len(self.vocabulary["terms"]), VOCABULARY_LIMIT)
        self.assertGreater(len(self.vocabulary["terms"]), 50)


if __name__ == "__main__":
    unittest.main()
