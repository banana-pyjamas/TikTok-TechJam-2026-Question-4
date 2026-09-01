"""Phase 16 -- staged final integration. Turn the features on one at a time.

NAMING. Step 8 is the "Free-Text Reranker", not a semantic one. The shipped
scorer is `reranker.PoolTermScorer`, which ranks on in-pool IDF over the
shopper's own free text. `load_encoder_scorer()` returns None on every turn
of every run in this repository: no encoder is vendored, none is downloaded,
and no model contributes to the score. The internal flag is still called
`USE_SEMANTIC_RERANK` -- renaming a pinned flag at a config freeze is a risk
with no benefit -- but nothing a human reads should imply a trained encoder
or an LLM is doing the ranking (B Phase 16 review, blocker 2).

The roadmap's last phase, and the one most likely to be performed rather than
run: "enable -> test -> review -> PASS -> next" for nine features is a
satisfying ritual whether or not any of it decides anything. So this tool is
built to be able to FAIL, and the thing it can fail is the claim that
matters:

    does a ladder that enables each feature only when the measurement
    justifies it arrive at the configuration we actually ship?

If yes, every flag's position is a measured consequence rather than an
accumulation of history. If no, something ships that its own gate does not
support, and the tool says which.

WHY THIS IS NOT tools/phase7_ablation.py

That tool walks the SHIPPED stack in the order the score was built up, to
attribute the score. This one walks the ROADMAP's feature order, including
the four features that do NOT ship, to audit the configuration. Different
order, different question, and the orders genuinely disagree: popularity
ablated from the full stack is +0.0203 at p = 0.1250 (no verdict), while
popularity added at its own rung -- before the reranker and clarification
exist to supply constraints it can be redundant with -- is a different and
much stronger comparison. A feature is justified GIVEN WHAT IS ALREADY ON,
which is exactly what staged enablement means and exactly why the sequence
cannot be reshuffled after the fact.

THE GATE

Each rung enables one feature on top of everything kept so far, measures, and
is judged on BOTH tests (see tools/significance.py): McNemar over per-session
hit verdicts, and a paired permutation over per-session composites. A rung
PASSES if either establishes a gain. A rung that does not pass is REVERTED,
and the next rung builds on the reverted configuration -- which is what
"PASS -> next" means when the answer is no.

FOUR OF THE NINE STEPS HAVE NO FLAG, and inventing one for the sake of a
complete-looking table would be the "104-item checklist" failure this phase
is explicitly warned against. They are reported with the measurement that
removed them. Three were built and rejected on evidence; one is not a
separable feature at all.

Usage:  python3 -m tools.phase16_integration     (~6 min: 6 evaluator runs)
"""

from __future__ import annotations

import time

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from tools import config_guard
from tools.capture import CapturingAgent
from tools.significance import (composites_by_sample, format_composite,
                                format_test, hits_by_sample, mcnemar,
                                mttc_given_hit, paired_permutation)

CATALOG = "data/catalog.jsonl"
DATASET = "data/public_set.jsonl"

# Step 1. Everything the "minimum winning path" builds before any advanced
# feature: state, override, validation, retrieval, ranking. Every later flag
# starts OFF and is switched on by its own rung.
CORE = {
    ("agent", "USE_STATE"): True,
    ("agent", "USE_MULTI_ROUTE"): True,
    ("agent", "USE_CONSTRAINT_RANKING"): True,
    ("ranking", "USE_PROFILE"): False,
    ("ranking", "USE_CONFIDENCE_WEIGHTING"): False,
    ("ranking", "USE_POPULARITY"): False,
    ("reranker", "USE_SEMANTIC_RERANK"): False,
    ("clarify", "USE_CLARIFICATION"): False,
}

# The roadmap's nine steps. ``flag`` is None where no flag exists, and the
# note says why -- those rungs are reported, not run.
LADDER = [
    (1, "Core only", None,
     "state + override + validation + retrieval + ranking. The baseline "
     "every rung below is measured against."),
    (2, "+ Profile", ("ranking", "USE_PROFILE"), ""),
    (3, "+ Adaptive Strategy", None,
     "NO FLAG: measured worth +0.000298 TS -- one thirty-fourth of a single "
     "flipped session -- and the flag was DELETED in Phase 9 rather than "
     "shipped OFF. starter/strategy.py is still built, tested and measured "
     "(tools/phase9_mode_accuracy.py); it gates nothing. A knob that changes "
     "nothing is the mechanism this repo has twice deleted."),
    (4, "+ Candidate Vocabulary", None,
     "NOT A SEPARABLE FEATURE: `reranker.build_scorer` imports "
     "`vocabulary.build_vocabulary` and `pool_terms` directly and has no "
     "other index. Disabling the vocabulary IS disabling the reranker, which "
     "is step 8. Measured on its own terms by tools/phase10_vocabulary.py."),
    (5, "+ EC/MR", ("ranking", "USE_CONFIDENCE_WEIGHTING"), ""),
    (6, "+ Popularity", ("ranking", "USE_POPULARITY"), ""),
    (7, "+ Dense", None,
     "NOT IMPLEMENTED, on measurement. The strongest vector retriever that "
     "can ship here is lexical TF-IDF cosine, and tools/phase13_dense_gate.py "
     "measures the union WORSE than the committed route set at recall@50 -- "
     "established negative, and re-established on every re-run. The exact "
     "counts are deliberately NOT quoted here: they are replayed over the "
     "live dialogue and move whenever the dialogue does, and an earlier "
     "version of this line quoted figures the gate itself contradicted at "
     "@100 by the time anyone read them (D Phase 16 review). Run the gate "
     "for the numbers; this rung carries the conclusion only. What a TRAINED "
     "encoder would be worth is not measured anywhere and is not claimed."),
    (8, "+ Free-Text Reranker", ("reranker", "USE_SEMANTIC_RERANK"), ""),
    (9, "+ Clarification", ("clarify", "USE_CLARIFICATION"), ""),
]


def passes_gate(hits: dict, score: dict, alpha: float = 0.05) -> bool:
    """Does this rung earn its place on top of the rung below it?

    PASSES if either test establishes a GAIN. Both directions of that matter:

      * either test -- McNemar sees the hit set and is blind to a change that
        moves every hit three ranks up; the paired permutation over
        per-session composites sees the score. Quoting only the first is the
        failure D named across two phases, and a gate that read only McNemar
        would have reverted rungs that improved the score without converting
        a miss.
      * a GAIN -- an established LOSS must fail, not pass. ``p < alpha``
        alone would keep a feature that significantly made things worse,
        which is the one mutation of this function that would be invisible in
        the tool's output: every rung would still print a verdict, and the
        ladder would still end somewhere.

    Pure and separately tested (tests/test_integration_ladder.py) precisely
    because the alternative is asserting on the tool's source text, which
    passes for any mutant that leaves the wording alone.
    """
    return ((hits["p"] < alpha and hits["net"] > 0)
            or (score["p"] < alpha and score["mean"] > 0))


def disagreements(state: dict, committed: dict) -> list[tuple]:
    """Flags where the gated ladder and the shipped configuration differ.

    Returns ``(flag, reached, committed)`` triples. Empty means every flag's
    position is a measured consequence; non-empty is the finding this whole
    tool exists to be able to produce, and ``main`` raises on it.
    """
    return [
        (flag, state.get(flag, value), value)
        for flag, value in sorted(committed.items())
        if state.get(flag, value) != value
    ]


def main() -> None:
    config_guard.assert_everything()

    print("building index...", flush=True)
    started = time.time()
    agent = CapturingAgent(CATALOG)
    samples = load_jsonl(DATASET)
    catalog_ids, categories, products = catalog_index(CATALOG)
    print(f"ready in {time.time() - started:.1f}s\n", flush=True)

    def run(label: str) -> dict:
        agent.order.clear()
        agent.captured.clear()
        began = time.time()
        result = evaluate(agent, samples, catalog_ids, categories, products)
        turns = len(agent.captured)
        if not turns:
            raise SystemExit(
                f"rung {label!r} captured 0 turns: every respond() raised and "
                "the evaluator swallowed it. Its numbers are a crash.")
        print(f"   {label:26}  HR {result['hit_rate_at_10']:.4f}  "
              f"MRR {result['mrr']:.6f}  MTTC {result['mttc']:.3f}  "
              f"TS {result['recommended_technical_score']:.6f}  "
              f"({turns} turns, {time.time() - began:.0f}s)", flush=True)
        return result

    state = dict(CORE)
    print("=" * 74)
    print("THE LADDER -- each rung enabled on top of everything KEPT so far")
    print("=" * 74)

    # EVERYTHING that mutates a flag is inside this try, including the setup
    # loop and rung 1. The first version opened the try AFTER rung 1, so a
    # raise while setting the core flags or while measuring core left the
    # process pinned to the core configuration -- and the tool's own crash
    # guard, which exists to restore them, could not run (D Phase 16 review).
    # A restore that only covers the rungs it happened to be wrapped around
    # is not a restore.
    try:
        for (module, name), value in state.items():
            config_guard.set_flag(module, name, value)
        previous = run("1. Core only")
        core = previous
        rows = [(1, "Core only", previous, None, "baseline")]

        for step, label, flag, note in LADDER[1:]:
            print()
            if flag is None:
                print(f"   {step}. {label:23} SKIPPED -- no flag exists")
                for line in _wrap(note):
                    print(f"      {line}")
                rows.append((step, label, None, None, "no flag"))
                continue

            config_guard.set_flag(flag[0], flag[1], True)
            result = run(f"{step}. {label}")
            hits = mcnemar(hits_by_sample(previous), hits_by_sample(result))
            score = paired_permutation(composites_by_sample(previous),
                                       composites_by_sample(result))
            print("      " + format_test("vs previous rung (hits)", hits))
            print("      " + format_composite("vs previous rung (score)",
                                              composites_by_sample(previous),
                                              composites_by_sample(result)))
            passed = passes_gate(hits, score)
            if passed:
                print(f"      GATE: PASS -- kept ON")
                state[flag] = True
                previous = result
            else:
                print(f"      GATE: FAIL -- reverted to OFF, the next rung "
                      f"builds on the configuration without it")
                config_guard.set_flag(flag[0], flag[1], False)
                state[flag] = False
            rows.append((step, label, result, passed, "kept" if passed
                         else "reverted"))
    finally:
        config_guard.restore_committed_flags()

    # -- the check this tool exists for --------------------------------------
    print("\n" + "=" * 74)
    print("DOES THE GATED LADDER LAND ON WHAT SHIPS?")
    print("=" * 74)
    disagreed = disagreements(state, config_guard.COMMITTED_FLAGS)
    disagreeing = {flag for flag, _, _ in disagreed}
    for flag, committed in sorted(config_guard.COMMITTED_FLAGS.items()):
        reached = state.get(flag, committed)
        mark = "DISAGREES" if flag in disagreeing else "same"
        name = f"{flag[0]}.{flag[1]}"
        print(f"   {name:34} ladder {str(reached):5}   "
              f"committed {str(committed):5}   {mark}")
    if disagreed:
        raise SystemExit(
            "the gated ladder does not reproduce the committed "
            "configuration: "
            + "; ".join(f"{m}.{n} ladder={r} committed={c}"
                        for (m, n), r, c in disagreed)
            + ". Either a shipped flag is not supported by its own gate, or "
            "the gate is wrong. Both are findings; neither is ignorable.")
    print("\n   Every flag's position is what its own rung measured. No flag")
    print("   ships on history, on intuition, or on a number from a")
    print("   configuration that no longer exists.")

    final = run("final (committed configuration)")
    committed_score = config_guard.COMMITTED_TECHNICAL_SCORE
    actual = final["recommended_technical_score"]
    exact = abs(actual - committed_score) <= 1e-9
    print(f"\n   reproduces COMMITTED_TECHNICAL_SCORE {committed_score}   "
          f"{'PASS' if exact else f'FAIL ({actual})'}")
    if not exact:
        raise SystemExit(
            "the ladder's end state does not reproduce the committed score, "
            "so the staged enablement and the shipped agent are not the same "
            "pipeline")

    # -- the summary table ---------------------------------------------------
    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print(f"   {'step':>4}  {'feature':22}{'TS':>10}{'vs core':>11}"
          f"{'vs prev':>11}  outcome")
    previous_ts = core["recommended_technical_score"]
    for step, label, result, passed, outcome in rows:
        if result is None:
            print(f"   {step:>4}  {label:22}{'--':>10}{'--':>11}{'--':>11}"
                  f"  {outcome}")
            continue
        technical = result["recommended_technical_score"]
        against_core = technical - core["recommended_technical_score"]
        against_prev = technical - previous_ts
        print(f"   {step:>4}  {label:22}{technical:>10.6f}"
              f"{against_core:>+11.6f}{against_prev:>+11.6f}  {outcome}")
        if outcome != "reverted":
            previous_ts = technical
    conditional = mttc_given_hit(final)
    print(f"\n   final    HR {final['hit_rate_at_10']:.4f}   "
          f"MRR {final['mrr']:.6f}   MTTC {final['mttc']:.3f}   "
          f"MTTC|hit {0.0 if conditional is None else conditional:.3f}   "
          f"TS {actual:.6f}")
    print("   " + format_test("whole ladder (hits)",
                              mcnemar(hits_by_sample(core),
                                      hits_by_sample(final))))
    print("   " + format_composite("whole ladder (score)",
                                   composites_by_sample(core),
                                   composites_by_sample(final)))

    print("\n   READ THE 'vs prev' COLUMN, NOT 'vs core'. A rung's own gate is")
    print("   the comparison against what was already on; 'vs core' is the")
    print("   accumulated total and double-counts anything two features both")
    print("   supply. The two disagree most where it matters: see the")
    print("   popularity rung against tools/phase12_popularity.py, which")
    print("   ablates the same flag from the FULL stack and gets a different")
    print("   and weaker answer, because by then the reranker and")
    print("   clarification supply constraints the prior was standing in for.")

    print(f"\nconfig: {config_guard.describe()}")


def _wrap(text: str, width: int = 68) -> list[str]:
    words, lines, line = text.split(), [], ""
    for word in words:
        if len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return lines


if __name__ == "__main__":
    main()
