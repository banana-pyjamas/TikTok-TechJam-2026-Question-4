"""Phase 16 -- the staged-enablement ladder's own structure.

The tool itself takes six evaluator runs and is not a unit test. What IS
unit-testable is the thing that makes it worth running: that it covers every
step the roadmap names, that the features it declines to run are declined for
a stated reason rather than forgotten, and that its end state is checked
against the committed configuration rather than assumed to match it.

A ladder that cannot disagree with what ships would be the "104-item
checklist" ritual this phase is warned against -- nine green rows proving
nothing. These tests pin the parts that let it go red.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from tools import config_guard
from tools.phase16_integration import CORE, LADDER, _wrap


class LadderCoversTheRoadmapTest(unittest.TestCase):

    def test_all_nine_steps_are_present_and_in_order(self) -> None:
        self.assertEqual([step for step, _, _, _ in LADDER], list(range(1, 10)))

    def test_every_step_is_either_runnable_or_explained(self) -> None:
        # The whole point. A step with no flag must carry a note saying why,
        # so "we did not measure it" and "we forgot it" cannot look alike.
        for step, label, flag, note in LADDER:
            if flag is None and step != 1:
                self.assertTrue(
                    note.strip(),
                    f"step {step} ({label}) has no flag and no explanation")

    def test_the_skipped_steps_are_the_four_that_do_not_ship(self) -> None:
        skipped = {label for _, label, flag, _ in LADDER if flag is None}
        self.assertEqual(
            skipped,
            {"Core only", "+ Adaptive Strategy", "+ Candidate Vocabulary",
             "+ Dense"})

    def test_every_flag_in_the_ladder_is_a_real_pinned_flag(self) -> None:
        for _, label, flag, _ in LADDER:
            if flag is None:
                continue
            self.assertIn(flag, config_guard.COMMITTED_FLAGS,
                          f"{label} names {flag}, which config_guard does not "
                          "pin -- the ladder would move an unguarded flag")

    def test_every_optional_flag_gets_a_rung(self) -> None:
        # The core three are always on and are the baseline, not rungs. Every
        # OTHER pinned flag must be enabled by some rung, or the ladder is
        # silently shipping something it never gated.
        core_always_on = {flag for flag, value in CORE.items() if value}
        rungs = {flag for _, _, flag, _ in LADDER if flag is not None}
        ungated = set(config_guard.COMMITTED_FLAGS) - core_always_on - rungs
        self.assertEqual(ungated, set(),
                         f"these flags ship without a Phase 16 rung: "
                         f"{sorted(ungated)}")


class CoreIsTheMinimumWinningPathTest(unittest.TestCase):

    def test_core_turns_on_exactly_the_three_core_flags(self) -> None:
        self.assertEqual({flag for flag, value in CORE.items() if value},
                         {("agent", "USE_STATE"),
                          ("agent", "USE_MULTI_ROUTE"),
                          ("agent", "USE_CONSTRAINT_RANKING")})

    def test_core_names_every_pinned_flag(self) -> None:
        # If a new flag is added and CORE does not mention it, rung 1 would
        # inherit whatever the module happened to declare, and every delta
        # below it would be measured against an unstated baseline.
        self.assertEqual(set(CORE), set(config_guard.COMMITTED_FLAGS))


class TheLadderCanDisagreeTest(unittest.TestCase):
    """The checks that let this tool fail, pinned so they cannot be softened
    into decoration."""

    def _source(self) -> str:
        return Path("tools/phase16_integration.py").read_text(encoding="utf-8")

    def test_it_compares_its_end_state_against_the_committed_flags(self) -> None:
        source = self._source()
        self.assertIn("COMMITTED_FLAGS", source)
        self.assertIn("DISAGREES", source)

    def test_a_disagreement_raises_rather_than_prints(self) -> None:
        source = self._source()
        self.assertIn("the gated ladder does not reproduce the committed",
                      source)
        self.assertIn("raise SystemExit", source)

    def test_it_asserts_the_committed_score_at_the_end(self) -> None:
        source = self._source()
        self.assertIn("COMMITTED_TECHNICAL_SCORE", source)
        self.assertIn("does not reproduce the committed score", source)

    def test_a_failed_rung_is_reverted_not_merely_reported(self) -> None:
        source = self._source()
        self.assertIn("GATE: FAIL", source)
        self.assertIn("config_guard.set_flag(flag[0], flag[1], False)", source)

    def test_the_gate_reads_both_tests(self) -> None:
        # Quoting only McNemar is the failure D named twice; the gate must
        # pass on either test establishing a gain.
        source = self._source()
        self.assertIn("paired_permutation", source)
        self.assertIn('hits["p"] < 0.05', source)
        self.assertIn('score["p"] < 0.05', source)

    def test_flags_are_restored_even_if_a_rung_raises(self) -> None:
        source = self._source()
        self.assertIn("finally:", source)
        self.assertIn("config_guard.restore_committed_flags()", source)


class WrapTest(unittest.TestCase):

    def test_it_wraps_without_losing_words(self) -> None:
        text = " ".join(f"word{index}" for index in range(40))
        lines = _wrap(text, width=30)
        self.assertEqual(" ".join(lines), text)
        self.assertTrue(all(len(line) <= 30 for line in lines))

    def test_empty_text_yields_no_lines(self) -> None:
        self.assertEqual(_wrap(""), [])


if __name__ == "__main__":
    unittest.main()
