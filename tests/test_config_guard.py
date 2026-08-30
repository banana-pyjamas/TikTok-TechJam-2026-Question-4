"""The measurement tools' configuration guard.

D-N2: the first version of this guard lived inside ``tools/phase7_ablation.py``
and scanned ``starter.agent`` only. Adversarially tested, a new ``USE_`` flag
in ``agent.py`` tripped it and the same flag in ``ranking.py`` was silent --
and ``retrieval.DEFAULT_ROUTES``, which decides every row of the union pool,
was not pinned at all. These tests are that adversarial test, committed, so
the failure mode cannot relocate one module over again.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import starter
from starter import agent, ranking, retrieval, state, vocabulary
from tools import config_guard


class FlagDiscoveryIsPackageWideTest(unittest.TestCase):
    def test_it_finds_the_flags_that_exist_today(self) -> None:
        found = config_guard.discover_flags()
        for expected in (("agent", "USE_STATE"),
                         ("agent", "USE_MULTI_ROUTE"),
                         ("agent", "USE_CONSTRAINT_RANKING"),
                         ("ranking", "USE_PROFILE")):
            self.assertIn(expected, found)

    def test_every_discovered_flag_has_a_committed_value(self) -> None:
        self.assertEqual(config_guard.discover_flags(),
                         set(config_guard.COMMITTED_FLAGS))

    def test_a_new_flag_in_any_module_trips_the_guard(self) -> None:
        pinned = set(config_guard.COMMITTED_FLAGS)
        # The control: today's configuration is fully pinned.
        config_guard.assert_all_flags_pinned(pinned)
        # Every module, not just agent.py -- that was the regression.
        for module in (agent, ranking, retrieval, state):
            with self.subTest(module=module.__name__):
                setattr(module, "USE_SOMETHING_NEW", True)
                try:
                    with self.assertRaises(SystemExit) as caught:
                        config_guard.assert_all_flags_pinned(pinned)
                    self.assertIn("USE_SOMETHING_NEW", str(caught.exception))
                finally:
                    delattr(module, "USE_SOMETHING_NEW")

    def test_a_pinned_flag_that_no_longer_exists_also_trips(self) -> None:
        with self.assertRaises(SystemExit):
            config_guard.assert_all_flags_pinned(
                set(config_guard.COMMITTED_FLAGS) | {("agent", "USE_GONE")})


class ModuleListCompletenessTest(unittest.TestCase):
    """D-P1: a hand-maintained module list is the same failure one level up.

    Scoping the guard to ``starter.agent`` was the first version of this
    failure; listing modules by hand was the second. The list is now checked
    against the directory, so there is no third.
    """

    def test_the_list_matches_the_package_today(self) -> None:
        config_guard.assert_module_list_is_complete()

    def test_a_new_module_carrying_a_flag_is_not_silent(self) -> None:
        # This case used to name `reranker.py`, the module the roadmap adds at
        # Phase 13/14. Phase 14 shipped it, and the test kept passing only
        # because its own precondition failed first -- one assertion away from
        # writing over the real module and then unlink()ing it in `finally`.
        # The placeholder must therefore be a name the package does NOT have,
        # and the precondition is what keeps the cleanup safe. If Phase 15
        # ships `clarify.py`, move this to the next unused name.
        package = Path(starter.__file__).resolve().parent
        new_module = package / "clarify.py"
        self.assertFalse(
            new_module.exists(),
            f"{new_module.name} now exists -- point this test at a module the "
            "package does not have, or the cleanup below deletes real code",
        )
        new_module.write_text("USE_CLARIFY = True\n", encoding="utf-8")
        try:
            with self.assertRaises(SystemExit) as caught:
                config_guard.assert_module_list_is_complete()
            self.assertIn("clarify", str(caught.exception))
            # And it must also trip the flag guard, which calls it.
            with self.assertRaises(SystemExit):
                config_guard.assert_all_flags_pinned(
                    set(config_guard.COMMITTED_FLAGS))
        finally:
            new_module.unlink()
        # Clean again afterwards.
        config_guard.assert_module_list_is_complete()

    def test_a_listed_module_that_vanishes_also_trips(self) -> None:
        original = config_guard.STARTER_MODULES
        config_guard.STARTER_MODULES = original + ("no_such_module",)
        try:
            with self.assertRaises(SystemExit) as caught:
                config_guard.assert_module_list_is_complete()
            self.assertIn("no_such_module", str(caught.exception))
        finally:
            config_guard.STARTER_MODULES = original


class CommittedConstantsTest(unittest.TestCase):
    def test_the_tree_matches_its_committed_constants(self) -> None:
        config_guard.assert_committed_constants()

    def test_default_routes_is_pinned(self) -> None:
        # The specific unpinned constant D found: it determines every union
        # row, and moving it swings the measured score by up to 0.019.
        self.assertIn(("retrieval", "DEFAULT_ROUTES"),
                      config_guard.COMMITTED_CONSTANTS)
        self.assertEqual(config_guard.COMMITTED_CONSTANTS[("retrieval", "DEFAULT_ROUTES")],
                         retrieval.DEFAULT_ROUTES)

    def test_every_numeric_tunable_is_pinned(self) -> None:
        # D-V2: Phase 10 shipped six thresholds unpinned, two of them values a
        # measurement had just rejected. The registry is now derived, not
        # maintained.
        config_guard.assert_all_constants_pinned()

    def test_discovery_attributes_a_reexport_to_its_definer(self) -> None:
        # agent.py imports POOL_LIMIT from retrieval; it must be demanded once,
        # against the module that defines it.
        found = config_guard.discover_numeric_constants()
        self.assertIn(("retrieval", "POOL_LIMIT"), found)
        self.assertNotIn(("agent", "POOL_LIMIT"), found)

    def test_an_unregistered_numeric_constant_trips_the_guard(self) -> None:
        key = ("vocabulary", "VOCABULARY_LIMIT")
        original = config_guard.COMMITTED_CONSTANTS.pop(key)
        try:
            with self.assertRaises(SystemExit) as caught:
                config_guard.assert_committed_constants()
            self.assertIn("VOCABULARY_LIMIT", str(caught.exception))
        finally:
            config_guard.COMMITTED_CONSTANTS[key] = original

    def test_the_rejected_phase_10_values_cannot_come_back_silently(self) -> None:
        # The two values measurement overturned, by name.
        for name, rejected in (("MAX_DOCUMENT_RATIO", 0.5),
                               ("VOCABULARY_LIMIT", 400)):
            with self.subTest(name=name):
                original = getattr(vocabulary, name)
                setattr(vocabulary, name, rejected)
                try:
                    with self.assertRaises(SystemExit) as caught:
                        config_guard.assert_committed_constants()
                    self.assertIn(name, str(caught.exception))
                finally:
                    setattr(vocabulary, name, original)

    def test_string_vocabularies_are_not_demanded(self) -> None:
        # Content, not configuration: pinning them would mean re-approving the
        # registry on every word added.
        found = config_guard.discover_numeric_constants()
        for name in ("GROUNDING", "BOILERPLATE"):
            self.assertNotIn(("vocabulary", name), found)

    def test_moving_a_constant_trips_the_guard(self) -> None:
        original = retrieval.DEFAULT_ROUTES
        retrieval.DEFAULT_ROUTES = ("bm25", "category", "attribute")
        try:
            with self.assertRaises(SystemExit) as caught:
                config_guard.assert_committed_constants()
            self.assertIn("DEFAULT_ROUTES", str(caught.exception))
        finally:
            retrieval.DEFAULT_ROUTES = original

    def test_restore_puts_every_flag_back(self) -> None:
        # Snapshot and restore. This test writes committed values onto live
        # modules, and without the restore it silently reverted any
        # source-level flag edit for every test that ran after it -- which
        # defeated the Phase 11/12 interaction guard in the full suite while
        # it passed when run alone (D Phase 12 review, Q2).
        before = {
            (module, name): getattr(config_guard._module(module), name)
            for module, name in config_guard.COMMITTED_FLAGS
        }
        try:
            agent.USE_MULTI_ROUTE = False
            ranking.USE_PROFILE = True
            config_guard.restore_committed_flags()
            self.assertTrue(agent.USE_MULTI_ROUTE)
            self.assertFalse(ranking.USE_PROFILE)
        finally:
            for (module, name), value in before.items():
                setattr(config_guard._module(module), name, value)

    def test_committed_flags_match_what_the_source_declares(self) -> None:
        # The registry claims to describe the committed configuration; the
        # source IS it. If they drift, every "restored to committed values"
        # line in this package becomes unverifiable.
        config_guard.assert_committed_flags_match_source()

    def test_source_flags_are_read_from_disk_not_memory(self) -> None:
        original = ranking.USE_POPULARITY
        ranking.USE_POPULARITY = not original
        try:
            self.assertEqual(
                config_guard.source_flags()[("ranking", "USE_POPULARITY")],
                original,
                "source_flags must survive runtime mutation")
        finally:
            ranking.USE_POPULARITY = original

    def test_describe_names_the_whole_configuration(self) -> None:
        described = config_guard.describe()
        for name in ("USE_STATE", "USE_MULTI_ROUTE", "USE_CONSTRAINT_RANKING",
                     "USE_PROFILE", "DEFAULT_ROUTES"):
            self.assertIn(name, described)


if __name__ == "__main__":
    unittest.main()
