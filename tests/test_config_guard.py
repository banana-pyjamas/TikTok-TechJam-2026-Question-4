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

from starter import agent, ranking, retrieval, state
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
        agent.USE_MULTI_ROUTE = False
        ranking.USE_PROFILE = True
        config_guard.restore_committed_flags()
        self.assertTrue(agent.USE_MULTI_ROUTE)
        self.assertFalse(ranking.USE_PROFILE)

    def test_describe_names_the_whole_configuration(self) -> None:
        described = config_guard.describe()
        for name in ("USE_STATE", "USE_MULTI_ROUTE", "USE_CONSTRAINT_RANKING",
                     "USE_PROFILE", "DEFAULT_ROUTES"):
            self.assertIn(name, described)


if __name__ == "__main__":
    unittest.main()
