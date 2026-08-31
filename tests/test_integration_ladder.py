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
from tools.phase16_integration import (CORE, LADDER,
                                       disagreements, _wrap,
                                       passes_gate)


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


class TheGateCanFailTest(unittest.TestCase):
    """The gate's SEMANTICS, executed -- not its source text, grepped.

    The previous version of this class asserted that certain strings appeared
    in the tool's source. D's Phase 16 review put 29 semantic mutants through
    it and all 29 passed, including one that turned the ladder into a rubber
    stamp: a mutation can leave every quoted phrase in place and invert what
    the code does. A test that reads source text tests the comment.

    So the gate and the reconciliation are pure functions now, and these call
    them.
    """

    def test_a_gain_established_only_on_hits_passes(self) -> None:
        # McNemar clears, permutation does not. Real case: a rung that
        # converts misses without moving ranks much.
        self.assertTrue(passes_gate({"p": 0.01, "net": 5},
                                    {"p": 0.90, "mean": 0.001}))

    def test_a_gain_established_only_on_score_passes(self) -> None:
        # The case McNemar is blind to: every hit moves up three ranks and
        # not one session changes verdict. Reverting this rung would be the
        # mispricing D named across two phases.
        self.assertTrue(passes_gate({"p": 0.90, "net": 0},
                                    {"p": 0.01, "mean": 0.02}))

    def test_a_gain_established_on_neither_fails(self) -> None:
        self.assertFalse(passes_gate({"p": 0.90, "net": 5},
                                     {"p": 0.90, "mean": 0.02}))

    def test_an_ESTABLISHED_LOSS_fails(self) -> None:
        # The mutation that would be invisible in the tool's output: drop the
        # direction check and `p < alpha` alone keeps a feature that
        # significantly made things worse. Every rung would still print a
        # verdict and the ladder would still end somewhere.
        self.assertFalse(passes_gate({"p": 0.001, "net": -9},
                                     {"p": 0.001, "mean": -0.05}))

    def test_an_identical_rung_fails(self) -> None:
        # What rung 5 (EC/MR on core) actually produces.
        self.assertFalse(passes_gate({"p": 1.0, "net": 0},
                                     {"p": 1.0, "mean": 0.0}))

    def test_the_gate_is_not_a_rubber_stamp(self) -> None:
        # The named mutant. If `passes_gate` were replaced by `True`, every
        # assertion above that expects False would fail -- this one states
        # the property directly so the intent survives a refactor.
        outcomes = {
            passes_gate({"p": p, "net": n}, {"p": p, "mean": m})
            for p, n, m in ((0.001, 5, 0.02), (0.001, -5, -0.02),
                            (0.9, 5, 0.02), (1.0, 0, 0.0))
        }
        self.assertEqual(outcomes, {True, False},
                         "the gate returns a constant; it decides nothing")

    def test_alpha_is_honoured(self) -> None:
        self.assertFalse(passes_gate({"p": 0.06, "net": 5},
                                     {"p": 0.06, "mean": 0.02}))
        self.assertTrue(passes_gate({"p": 0.06, "net": 5},
                                    {"p": 0.06, "mean": 0.02}, alpha=0.10))


class ReconciliationCanDisagreeTest(unittest.TestCase):
    """The check the whole tool exists to be able to fail, executed."""

    _COMMITTED = {("m", "A"): True, ("m", "B"): False}

    def test_matching_state_reports_nothing(self) -> None:
        self.assertEqual(
            disagreements({("m", "A"): True, ("m", "B"): False},
                          self._COMMITTED),
            [])

    def test_a_flag_the_ladder_kept_but_we_ship_off_is_reported(self) -> None:
        self.assertEqual(
            disagreements({("m", "A"): True, ("m", "B"): True},
                          self._COMMITTED),
            [(("m", "B"), True, False)])

    def test_a_flag_the_ladder_reverted_but_we_ship_on_is_reported(self) -> None:
        self.assertEqual(
            disagreements({("m", "A"): False, ("m", "B"): False},
                          self._COMMITTED),
            [(("m", "A"), False, True)])

    def test_a_flag_missing_from_the_state_defaults_to_agreement(self) -> None:
        # A flag no rung touched cannot be a disagreement -- it was never
        # gated, which CORE-completeness above is what actually catches.
        self.assertEqual(disagreements({}, self._COMMITTED), [])

    def test_the_real_committed_flags_are_reconcilable(self) -> None:
        # Sanity: the function accepts the shape config_guard actually holds.
        self.assertEqual(
            disagreements(dict(config_guard.COMMITTED_FLAGS),
                          config_guard.COMMITTED_FLAGS),
            [])


class RestoreCoversEveryMutationTest(unittest.TestCase):
    """Rung 1 and the setup loop must be inside the try, not before it.

    This one is structural rather than behavioural, because the property is
    about control flow: it walks the AST instead of grepping, so it cannot be
    satisfied by a comment that mentions `finally`.
    """

    def test_no_flag_is_set_before_the_try_block(self) -> None:
        import ast

        source = Path("tools/phase16_integration.py").read_text(
            encoding="utf-8")
        tree = ast.parse(source)
        main = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "main")
        seen_try = False
        for node in main.body:
            if isinstance(node, ast.Try):
                seen_try = True
                continue
            if seen_try:
                continue
            for inner in ast.walk(node):
                if (isinstance(inner, ast.Call)
                        and isinstance(inner.func, ast.Attribute)
                        and inner.func.attr == "set_flag"):
                    self.fail("a flag is mutated before main()'s try block, "
                              "so the restore in `finally` cannot cover it")

    def test_the_try_has_a_finally_that_restores(self) -> None:
        import ast

        source = Path("tools/phase16_integration.py").read_text(
            encoding="utf-8")
        main = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "main")
        tries = [node for node in main.body if isinstance(node, ast.Try)]
        self.assertTrue(tries, "main() has no try block")
        restores = [
            inner for node in tries for statement in node.finalbody
            for inner in ast.walk(statement)
            if isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and inner.func.attr == "restore_committed_flags"
        ]
        self.assertTrue(restores,
                        "the try block's finally does not restore the flags")


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
