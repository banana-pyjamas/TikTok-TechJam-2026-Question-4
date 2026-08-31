"""Phase 14 CP 14.6 -- semantic reranking OFF vs ON, and what the stage did.

An ablation on its own would answer "did the score move" and nothing else,
which is not enough to justify a stage that sits in the middle of every turn.
So this also reports what the reranker DID: which fallback each turn took,
how far it moved the hidden target, and what it cost.

Six questions:

  1. NO-OP        does the flag's OFF position reproduce the committed score
                  EXACTLY? A flag whose OFF arm drifts makes every comparison
                  below meaningless.
  2. SCORE        OFF vs ON, paired McNemar on per-session hit verdicts,
                  against PLACEBO_DRAWS independent draws of a control that
                  reorders on meaningless words.
  3. BEHAVIOUR    the CP 14.1 - 14.5 paths, counted over the real dialogue --
                  including how often the stage is a no-op because the
                  shopper volunteered no free text at all -- and what the
                  whole stage costs, build included.
  4. MOVEMENT     for the turns where the target was inside the rerank
                  window, where did it start and where did it end up? This is
                  the question Phase 13 left for Phase 14: 176 eligible-turn
                  pool hits sit at final rank 11-50, and moving them is the
                  only reason this stage exists. Reported for every arm.
  5. ROBUSTNESS   does the gain survive a different window?
  6. TERMS        WHAT is the scorer ranking on? An ablation says a stage
                  helps; only this says the stage reads the shopper rather
                  than the harness's sentence templates.

The CP 14.1 - 14.5 CONTRACTS are unit-tested in tests/test_reranker.py. What
this tool adds is that they hold on live data -- every reranked output is
checked to be a permutation of its input, so "the model cannot invent ASINs"
is verified against the real pipeline and not only against stubs. Where a
live check has no coverage it says so instead of printing a PASS; see the
CP 14.1 tail note in section 3.

Usage:  python3 -m tools.phase14_reranker
        (~10 min: 2 arms + PLACEBO_DRAWS control draws + 5 sweep rows)
"""

from __future__ import annotations

import hashlib
import time
from collections import Counter

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter import reranker
from starter.contracts import Context
from starter.reranker import RERANK_KEY, PoolTermScorer, _evidence_terms
from starter.state import _REQUEST_FRAMING as FRAMING_WORDS
from tools import config_guard
from tools.capture import CapturingAgent
from tools.phase13_dense_gate import first_scoring_turn
from tools.significance import (format_test, hits_by_sample, mcnemar,
                                mttc_given_hit)
from tools.summaries import percentiles

CATALOG = "data/catalog.jsonl"
DATASET = "data/public_set.jsonl"

# The score this flag's OFF position produced AT PHASE 14: the Phase 12
# pipeline exactly. A fixed HISTORICAL reference, NOT the committed score --
# the committed score is now the ON arm and it lives in config_guard. Keeping
# the two apart is the whole lesson of phase10/phase11, which each pinned "the
# committed score" as a private literal and then spent two phases raising on
# their own staleness.
#
# It is dated, not maintained. Phase 15 moved the OFF arm (clarification
# changes what the shopper says on every turn, reranker or no reranker), so
# section 1 now REPORTS the difference instead of exiting on it.
PRE_RERANK_TECHNICAL_SCORE = 0.182258


# How many independent placebo draws to run. One draw is not a control.
#
# THIS IS THE FIX FOR THE ONE BLOCKER IN THE PHASE 14 REVIEW (C and D, same
# finding). The first version seeded each draw on ``context.session_id``,
# believing it deterministic because it used ``hashlib`` rather than
# ``hash()``. It is not: ``evaluator/local_evaluator.py`` sets that field to
# ``f"public_{uuid.uuid4().hex}"``, freshly generated inside ``evaluate`` on
# every run. The seed had eliminated the small randomness and inherited a
# larger one, so the "deterministic" placebo was a different draw every run
# -- observed range 0.147 to 0.192 TS, a spread WIDER than the 0.0415 effect
# it was controlling for. The committed number was the most favourable of
# nine draws, and "ON vs PLACEBO: established" became "no verdict" on re-run.
#
# Two changes, and both are needed. The seed below is stable (see
# ``PlaceboScorer.query_terms``), so a single draw now reproduces. And a
# single draw is still one sample of a random control, so the tool runs
# several and makes every claim against the HIGHEST-SCORING of them -- if
# the gain survives the draw that got luckiest, it is not a fact about
# which draw happened to be quoted.
PLACEBO_DRAWS = 5


class PlaceboScorer(PoolTermScorer):
    """The control arm: the same machinery, ranking on the WRONG words.

    Everything is held constant except the one thing under test. Same pool
    vocabulary, same in-pool IDF weighting, same tie-break, same number of
    query terms per turn -- but the terms are drawn deterministically from the
    pool instead of from what the shopper said.

    This exists because the first ON run measured +0.024 TS while the target's
    mean rank got WORSE (21.0 -> 21.9, up on 15 turns and down on 43). A score
    gain whose stated mechanism runs backwards is exactly the shape of a
    perturbation gain: the evaluator stops at the first hit, so ANY reordering
    reshuffles which sessions happen to land a target in the top 10, and on
    200 sessions that can look like a win. If the placebo gains too, the
    lexical signal is doing nothing and the ON arm must not ship.
    """

    name = "placebo"

    def __init__(self, vocabulary, indexed_terms, salt: int = 0) -> None:
        super().__init__(vocabulary, indexed_terms)
        self.salt = salt

    def query_terms(self, context: Context) -> set[str]:
        """Meaningless terms, same cardinality, REPRODUCIBLE ACROSS RUNS.

        Seeded from the shopper's own words and the turn number -- both
        stable properties of the sample -- plus a per-draw salt. NOT from
        ``context.session_id``, which the evaluator regenerates as a fresh
        uuid4 on every run and which silently made this whole arm a fresh
        random draw each time (see PLACEBO_DRAWS above).

        Seeding on the real terms does not leak their meaning: the digest is
        used only to index a sorted pool vocabulary, so the drawn words are
        unrelated to the shopper's, which is the property the control needs.
        Two sessions that said the same thing draw the same placebo, which is
        a feature -- it is what "deterministic" means here.
        """
        real = _evidence_terms(context)
        vocabulary = sorted(self.frequencies)
        if not real or not vocabulary:
            return set()
        seed = f"{self.salt}:{context.turn}:{'|'.join(sorted(real))}"
        picked: set[str] = set()
        attempt = 0
        while len(picked) < len(real) and attempt < 100 * len(real):
            digest = hashlib.md5(f"{seed}:{attempt}".encode("utf-8")).hexdigest()
            picked.add(vocabulary[int(digest, 16) % len(vocabulary)])
            attempt += 1
        return picked


def make_placebo_builder(original, salt: int = 0):
    """``build_scorer`` with the placebo substituted for the real scorer.

    Closes over ``original`` rather than reading ``reranker.build_scorer``:
    the arm loop patches that name, so reading it here would call THIS
    function again. The first version did exactly that, recursed until
    RecursionError, and -- because the evaluator swallows a raising turn into
    an empty response -- reported a placebo arm of HR 0.0000 that looked like
    a devastating result instead of a broken one.
    """

    def build(connection, candidates, context):
        real = original(connection, candidates, context)
        if not isinstance(real, PoolTermScorer):
            return real
        placebo = PlaceboScorer(None, real.indexed_terms, salt)
        placebo.pool_size = real.pool_size
        placebo.frequencies = real.frequencies
        return placebo

    return build


def instrument(agent: CapturingAgent, targets: list[str], eligible_from: list[int]):
    """Wrap ``reranker.rerank`` and the scorer BUILD with observation.

    Patching the MODULE attributes, which is also how ``agent.respond``
    reaches them -- so this observes the real calls, not a reimplementation of
    them. The wrapper never changes the returned order; a tool that measured a
    different reranker than the one that ships would be worthless.

    ``safe_build_scorer`` is wrapped too, and that is a correction rather than
    extra detail: the reported cost of this stage used to be
    ``diagnostics["elapsed_ms"]``, which times ``scorer.order`` ALONE. The
    build -- two indexed catalog queries over the whole 300-candidate pool --
    is the expensive half and was outside the number the phase quoted
    (D Phase 14 review). Returns ``(stats, restore)``.

    Session identity: ``reset`` runs in dataset order, so the first session id
    the wrapper sees belongs to sample 0, and so on. Same bridge every tool in
    this repo uses to get from an opaque uuid back to its sample.
    """
    original = reranker.rerank
    original_safe_build = reranker.safe_build_scorer
    stats = {
        "outcomes": Counter(),
        "invented": 0,
        "not_a_permutation": 0,
        "tail_disturbed": 0,
        "tail_nonempty": 0,
        "elapsed": [],
        "build_elapsed": [],
        "term_weight": Counter(),
        "moved": [],
        "target_before": [],
        "target_after": [],
        "target_into_top10": 0,
        "target_out_of_top10": 0,
        "target_in_window": 0,
    }
    target_of: dict[str, str] = {}
    eligible_of: dict[str, int] = {}

    def timed_build(connection, candidates, context):
        started = time.monotonic()
        try:
            scorer = original_safe_build(connection, candidates, context)
        finally:
            stats["build_elapsed"].append(
                (time.monotonic() - started) * 1000.0)
        # What the scorer is about to rank ON, weighted as it will weight it.
        # Section 6 reads this. Recomputing `query_terms` here is a few
        # microseconds and keeps the accounting outside the shipped path.
        if isinstance(scorer, PoolTermScorer):
            for term in scorer.query_terms(context):
                stats["term_weight"][term] += scorer.weight(term)
        return scorer

    def counting(result, context, scorer, *args, **kwargs):
        before = [candidate.parent_asin for candidate in result.ranked]
        output = original(result, context, scorer, *args, **kwargs)
        after = [candidate.parent_asin for candidate in output.ranked]

        diagnostics = (context.derived or {}).get(RERANK_KEY) or {}
        stats["outcomes"][diagnostics.get("outcome", "?")] += 1
        stats["invented"] += int(diagnostics.get("invented") or 0)
        stats["elapsed"].append(float(diagnostics.get("elapsed_ms") or 0.0))
        stats["moved"].append(int(diagnostics.get("moved") or 0))

        # CP 14.2 on live data: same multiset in, same multiset out.
        if sorted(before) != sorted(after):
            stats["not_a_permutation"] += 1
        # CP 14.1 on live data: everything past the window is untouched.
        #
        # ``tail_nonempty`` is counted alongside, because without it this
        # check is a green light that cannot go red: `agent.respond` ranks to
        # exactly RERANK_TOP_N with the flag ON, so `before[window:]` and
        # `after[window:]` are both empty on every turn and the comparison is
        # `[] != []` (D Phase 14 review). A vacuous PASS is worse than no
        # check, so the count of turns where there was actually a tail to
        # disturb is printed next to the verdict.
        window = int(diagnostics.get("considered") or 0)
        if window and len(before) > window:
            stats["tail_nonempty"] += 1
        if window and before[window:] != after[window:]:
            stats["tail_disturbed"] += 1

        if context.session_id not in target_of:
            index = len(target_of)
            target_of[context.session_id] = targets[index]
            eligible_of[context.session_id] = eligible_from[index]
        target = target_of[context.session_id]
        if context.turn >= eligible_of[context.session_id] and target in before:
            position_before = before.index(target) + 1
            position_after = after.index(target) + 1
            stats["target_in_window"] += 1
            stats["target_before"].append(position_before)
            stats["target_after"].append(position_after)
            if position_before > 10 >= position_after:
                stats["target_into_top10"] += 1
            elif position_after > 10 >= position_before:
                stats["target_out_of_top10"] += 1
        return output

    reranker.rerank = counting
    reranker.safe_build_scorer = timed_build

    def restore() -> None:
        reranker.rerank = original
        reranker.safe_build_scorer = original_safe_build

    return stats, restore


def main() -> None:
    config_guard.assert_all_flags_pinned(set(config_guard.COMMITTED_FLAGS))
    config_guard.assert_committed_constants()
    config_guard.assert_committed_flags_match_source()
    config_guard.restore_committed_flags()

    print("building index...", flush=True)
    started = time.time()
    agent = CapturingAgent(CATALOG)
    samples = load_jsonl(DATASET)
    catalog_ids, categories, products = catalog_index(CATALOG)
    print(f"ready in {time.time() - started:.1f}s", flush=True)

    targets = [str(s["ground_truth"]["parent_asin"]) for s in samples]
    eligible_from = [first_scoring_turn(s, products) for s in samples]

    # Two arms plus PLACEBO_DRAWS control draws. The placebo is not
    # decoration: without it, "ON gained 6 sessions" cannot be told apart from
    # "any reordering gains ~6 sessions", because the evaluator stops at the
    # first hit and is therefore sensitive to permutation as such. Several
    # draws rather than one because a single draw of a random control is a
    # sample, not a control -- see PLACEBO_DRAWS.
    original_build = reranker.build_scorer
    arms = [("OFF", False, original_build), ("ON", True, original_build)]
    arms += [(f"PLACEBO{draw}", True, make_placebo_builder(original_build, draw))
             for draw in range(PLACEBO_DRAWS)]
    placebo_labels = [label for label, _, _ in arms if label.startswith("PLACEBO")]
    results: dict[str, dict] = {}
    stats: dict[str, dict] = {}

    for label, enabled, builder in arms:
        config_guard.set_flag("reranker", "USE_SEMANTIC_RERANK", enabled)
        reranker.build_scorer = builder
        agent.order.clear()
        agent.captured.clear()
        arm_stats, restore = instrument(agent, targets, eligible_from)
        started = time.time()
        results[label] = evaluate(agent, samples, catalog_ids, categories,
                                  products)
        restore()
        reranker.build_scorer = original_build
        elapsed = time.time() - started
        stats[label] = arm_stats
        result = results[label]

        # The evaluator turns a raising `respond` into an empty response and
        # carries on, so a totally broken arm reports HR 0.0000 and looks like
        # a finding rather than a crash. `CapturingAgent.respond` records only
        # AFTER the real call returns, which makes the turn count the cheapest
        # honest liveness check there is.
        turn_count = len(agent.captured)
        if not turn_count:
            raise SystemExit(
                f"arm {label!r} captured 0 turns: every respond() raised and "
                "the evaluator swallowed it. Its numbers are a crash, not a "
                "measurement.")
        if enabled and arm_stats["outcomes"] and \
                sum(arm_stats["outcomes"].values()) != turn_count:
            raise SystemExit(
                f"arm {label!r} reranked "
                f"{sum(arm_stats['outcomes'].values())} of {turn_count} turns; "
                "the stage must run on every turn or the comparison is not "
                "controlled.")

        print(f"\n{label:8} USE_SEMANTIC_RERANK={enabled!s:5} "
              f"HR {result['hit_rate_at_10']:.4f} MRR {result['mrr']:.6f} "
              f"MTTC {result['mttc']:.3f} "
              f"TS {result['recommended_technical_score']:.6f} "
              f"({turn_count} turns, {elapsed:.0f}s, "
              f"{1000 * elapsed / turn_count:.1f} ms/turn)", flush=True)
    config_guard.restore_committed_flags()

    off, on = results["OFF"], results["ON"]
    stats_on = stats["ON"]
    turns = len(agent.captured)

    # The draw that is HARDEST for ON to beat, chosen by score rather than by
    # which one flatters the conclusion. Every "ON vs PLACEBO" claim below is
    # made against this one; the whole table is printed underneath it so the
    # spread is visible and the choice is auditable.
    strongest_placebo = max(
        placebo_labels,
        key=lambda label: results[label]["recommended_technical_score"])
    placebo = results[strongest_placebo]

    # -- 1 -------------------------------------------------------------------
    print("\n1. NO-OP -- is the OFF arm a true no-op?")
    # Two different pins, and conflating them is how phase10 and phase11 went
    # stale for two whole phases. PRE_RERANK is a HISTORICAL reference -- the
    # score of this flag's OFF position, which is the Phase 12 pipeline and
    # never moves again. COMMITTED_TECHNICAL_SCORE is the score of what ships,
    # which is now the ON arm. (phase12_popularity pins its OFF arm the same
    # way, for the same reason.)
    # ASSERTED: the ON arm reproduces the committed score, which is what makes
    # everything below a measurement of the shipped pipeline.
    #
    # REPORTED, NEVER ASSERTED: the OFF arm against the pre-Phase-14 score.
    # It was exact until Phase 15, and then stopped being -- "this pipeline
    # minus reranking" is not a constant, it moves with everything upstream,
    # and clarification changed what the shopper says on every turn. The first
    # version of this block asserted BOTH and would have exited here, which is
    # the staleness-failure phase10 and phase11 each spent two phases in and
    # which phase12 had already been corrected for. Same fix, one phase later.
    committed = config_guard.COMMITTED_TECHNICAL_SCORE
    actual_on = on["recommended_technical_score"]
    exact = abs(actual_on - committed) <= 1e-9
    print(f"   ON reproduces the committed score        {committed}   "
          f"{'PASS' if exact else f'FAIL ({actual_on})'}")
    if not exact:
        raise SystemExit(
            "the ON arm no longer reproduces the committed score, so this is "
            "not a measurement of the shipped pipeline")
    actual_off = off["recommended_technical_score"]
    drift = actual_off - PRE_RERANK_TECHNICAL_SCORE
    print(f"   OFF vs the pre-Phase-14 pipeline         "
          f"{PRE_RERANK_TECHNICAL_SCORE}   "
          f"{'exact' if abs(drift) <= 1e-9 else f'{actual_off} ({drift:+.6f})'}"
          f"   [reported, not asserted]")
    print("   With the flag OFF the reranker is not imported into the hot path,")
    print("   the ranked list is not deepened, and no scorer is built -- the")
    print("   turn is byte-for-byte the Phase 12 pipeline.")

    # -- 2 -------------------------------------------------------------------
    print("\n2. SCORE -- OFF vs ON, against a placebo that reorders on the")
    print("   WRONG words (same machinery, same term count, same weighting)")
    for label in ["OFF", "ON"] + placebo_labels:
        result = results[label]
        conditional = mttc_given_hit(result)
        marker = "  <- highest-scoring draw" if label == strongest_placebo else ""
        print(f"   {label:10}HR {result['hit_rate_at_10']:.4f}  "
              f"MRR {result['mrr']:.6f}  MTTC {result['mttc']:.3f}  "
              f"MTTC|hit {0.0 if conditional is None else conditional:.3f}"
              f"  TS {result['recommended_technical_score']:.6f}"
              f"  {result['recommended_technical_score'] - off['recommended_technical_score']:+.6f}"
              f"{marker}")
    scores = [results[label]["recommended_technical_score"]
              for label in placebo_labels]
    print(f"   placebo spread across {len(placebo_labels)} draws: "
          f"{min(scores):.6f} .. {max(scores):.6f}   "
          f"(ON - strongest placebo {on['recommended_technical_score'] - max(scores):+.6f})")
    print()
    print("   " + format_test("ON vs OFF",
                              mcnemar(hits_by_sample(off), hits_by_sample(on))))
    print("   " + format_test(f"{strongest_placebo} vs OFF",
                              mcnemar(hits_by_sample(off), hits_by_sample(placebo))))
    for label in placebo_labels:
        test = mcnemar(hits_by_sample(results[label]), hits_by_sample(on))
        print("   " + format_test(f"ON vs {label}", test))
    print("\n   HOW TO READ THIS. The evaluator stops a session at its first")
    print("   hit, so ANY reordering of the top 50 reshuffles which sessions")
    print("   happen to land a target in the top 10. If the placebo gains as")
    print("   much as ON, the shopper's words carried nothing and the ON gain")
    print("   is perturbation. Only 'ON vs PLACEBO' isolates the signal.")
    print("   Read the whole placebo column, not one row of it. A single draw")
    print("   of a random control is a sample: the previous version of this")
    print("   tool seeded the draw on the evaluator's per-run uuid, quoted one")
    print("   draw as if it were the control, and the verdict flipped on")
    print("   re-run. Every draw here reproduces exactly, and the claim is")
    print("   made against the HIGHEST-SCORING of them.")
    print(f"\n   {'scenario':18}{'OFF':>10}{'ON':>10}{strongest_placebo:>10}"
          f"{'ON-OFF':>10}")
    for scenario in ("buying", "browsing", "intent_override", "boundary"):
        a = off["scenario_metrics"][scenario]["hit_rate_at_10"]
        b = on["scenario_metrics"][scenario]["hit_rate_at_10"]
        c = placebo["scenario_metrics"][scenario]["hit_rate_at_10"]
        print(f"   {scenario:18}{a:>10.4f}{b:>10.4f}{c:>10.4f}{b - a:>+10.4f}")

    # -- 3 -------------------------------------------------------------------
    assert stats_on is not None
    outcomes = stats_on["outcomes"]
    calls = sum(outcomes.values())
    print(f"\n3. BEHAVIOUR -- which path each of the {calls} reranked turns took")
    for outcome in reranker.OUTCOMES:
        count = outcomes[outcome]
        note = {
            "offline": "   <- CP 14.5: no scorer at all (no encoder AND no free text)",
            "malformed": "   <- CP 14.3",
            "error": "   <- CP 14.3",
            "timeout": "   <- CP 14.4",
            "identity": "   <- scorer ran and changed nothing",
            "applied": "   <- scorer reordered the window",
        }.get(outcome, "")
        print(f"   {outcome:12}{count:>6}{count / max(calls, 1):>9.1%}{note}")
    unexpected = set(outcomes) - set(reranker.OUTCOMES)
    print(f"   undeclared outcomes            "
          f"{'none  PASS' if not unexpected else sorted(unexpected)}")
    print(f"\n   CP 14.5, stated precisely. `load_encoder_scorer()` returns "
          f"{reranker.load_encoder_scorer()!r} on")
    print("   EVERY one of these turns -- the semantic scorer is unavailable "
          "100% of\n   the time, which is why the lexical one is what ran at "
          "all. The 'offline'\n   row above is the narrower case where there "
          "is no scorer of ANY kind,\n   which needs the shopper to have "
          "volunteered no free text either.")

    print("\n   contract checks, on live data rather than on stubs")
    tail_turns = stats_on["tail_nonempty"]
    print(f"   CP 14.1  turns with a non-empty tail        "
          f"{tail_turns:>6}  "
          f"{'VACUOUS -- see below' if not tail_turns else ''}")
    print(f"   CP 14.1  tail outside the window disturbed  "
          f"{stats_on['tail_disturbed']:>6}  "
          f"{('PASS' if not stats_on['tail_disturbed'] else 'FAIL') if tail_turns else 'no live coverage'}")
    print(f"   CP 14.2  outputs that were not a permutation {stats_on['not_a_permutation']:>5}  "
          f"{'PASS' if not stats_on['not_a_permutation'] else 'FAIL'}")
    print(f"   CP 14.2  invented ASINs discarded            "
          f"{stats_on['invented']:>5}")
    print(f"   CP 14.4  budget overruns                     "
          f"{outcomes['timeout']:>5}  "
          f"{'PASS' if not outcomes['timeout'] else 'the ON numbers are not reproducible'}")
    if not tail_turns:
        print("   The CP 14.1 tail check has NO live coverage and says so rather")
        print("   than printing a green PASS. `agent.respond` ranks to exactly")
        print("   RERANK_TOP_N with the flag ON, so the tail is empty on every")
        print("   turn and the comparison is `[] != []` -- a check that cannot")
        print("   go red (D Phase 14 review). CP 14.1 is covered where it can")
        print("   actually fail: tests/test_reranker.py hands `rerank` a list")
        print("   longer than the window. The live check stays because the")
        print("   ranking depth is not a constant of this stage -- it is set by")
        print("   a `max()` in agent.respond and a future top_k > 50 would give")
        print("   it teeth without anyone remembering to re-enable it.")
    cost = percentiles(stats_on["elapsed"])
    build = percentiles(stats_on["build_elapsed"])
    print(f"   scorer.order cost per turn   mean {cost.mean:.3f} ms   "
          f"P90 {cost.p90:.3f} ms   max {cost.maximum:.3f} ms   "
          f"(budget {reranker.RERANK_BUDGET_MS:.0f} ms)")
    print(f"   scorer BUILD cost per turn   mean {build.mean:.3f} ms   "
          f"P90 {build.p90:.3f} ms   max {build.maximum:.3f} ms   "
          f"(no budget -- see below)")
    print(f"   stage total per turn         mean "
          f"{cost.mean + build.mean:.3f} ms")
    print("   The build is the expensive half and the phase used to quote only")
    print("   the other one (D Phase 14 review). It is two indexed queries over")
    print("   the whole 300-candidate pool, and RERANK_BUDGET_MS does not cover")
    print("   it -- the budget is a deadline handed to `scorer.order`, and a")
    print("   deadline cannot bound work that happens before the scorer exists.")
    print("   `safe_build_scorer` makes the build total (it cannot take a turn")
    print("   down); making it FAST is a real encoder's problem, and whoever")
    print("   vendors one has to bound the load here, not only the scoring.")

    # -- 4 -------------------------------------------------------------------
    print("\n4. MOVEMENT -- what happened to the hidden target")
    print("   Every arm, not just ON. This table is the evidence the score")
    print("   comparison cannot give: the placebo's SCORE only says a random")
    print("   reordering did not happen to win on 200 sessions, while its")
    print("   MOVEMENT says what a random reordering does to the target every")
    print("   single time. The tool computed these rows for the placebo from")
    print("   the first version and printed only ON's (D Phase 14 review).")
    print(f"\n   {'arm':12}{'in window':>10}{'mean before':>13}"
          f"{'mean after':>12}{'up':>6}{'down':>6}{'->top10':>9}{'out':>6}")
    for label in ["ON"] + placebo_labels:
        arm = stats[label]
        in_window = arm["target_in_window"]
        if not in_window:
            print(f"   {label:12}{in_window:>10}")
            continue
        before = percentiles(arm["target_before"])
        after = percentiles(arm["target_after"])
        improved = sum(1 for b, a in zip(arm["target_before"],
                                         arm["target_after"]) if a < b)
        worsened = sum(1 for b, a in zip(arm["target_before"],
                                         arm["target_after"]) if a > b)
        print(f"   {label:12}{in_window:>10}{before.mean:>13.1f}"
              f"{after.mean:>12.1f}{improved:>6}{worsened:>6}"
              f"{arm['target_into_top10']:>9}{arm['target_out_of_top10']:>6}")
    print("   Turn-level, so one session can contribute several rows; the")
    print("   session-level consequence is the McNemar test in (2), which is")
    print("   the number that decides the flag.")

    # -- 5 -------------------------------------------------------------------
    print("\n5. ROBUSTNESS -- does the gain survive a different window?")
    print("   A real effect should not depend on RERANK_TOP_N being exactly")
    print(f"   {reranker.RERANK_TOP_N}; a lucky permutation would. top_n=10 is a "
          "control rather than a\n   setting: the window is then the response "
          "itself, so the stage can only\n   reorder what was already being "
          "shown and can pull nothing up from below.")
    print(f"   {'top_n':>8}{'HR':>10}{'MRR':>12}{'TS':>12}{'vs OFF':>11}"
          "   McNemar vs OFF")
    committed_top_n = reranker.RERANK_TOP_N
    config_guard.set_flag("reranker", "USE_SEMANTIC_RERANK", True)
    try:
        for top_n in (10, 20, 50, 100, 200):
            reranker.RERANK_TOP_N = top_n
            agent.order.clear()
            agent.captured.clear()
            swept = evaluate(agent, samples, catalog_ids, categories, products)
            test = mcnemar(hits_by_sample(off), hits_by_sample(swept))
            print(f"   {top_n:>8}{swept['hit_rate_at_10']:>10.4f}"
                  f"{swept['mrr']:>12.6f}"
                  f"{swept['recommended_technical_score']:>12.6f}"
                  f"{swept['recommended_technical_score'] - off['recommended_technical_score']:>+11.6f}"
                  f"   {test['gained']}/{test['lost']} discordant, "
                  f"p = {test['p']:.4f}", flush=True)
    finally:
        reranker.RERANK_TOP_N = committed_top_n
        config_guard.restore_committed_flags()
    print("   Five comparisons against one baseline, so read the family rather")
    print("   than any single p-value (D-P3): what carries the conclusion is")
    print("   that the control is null and the direction holds across the")
    print("   plateau, not that a particular row cleared 0.05.")

    # -- 6 -------------------------------------------------------------------
    print("\n6. WHAT THE SCORER IS ACTUALLY RANKING ON")
    print("   The one question an ablation cannot answer. OFF vs ON says the")
    print("   reordering helps; it does not say the reordering reads the")
    print("   SHOPPER. Total in-pool IDF weight applied per term over the run:")
    weights = stats_on["term_weight"]
    total = sum(weights.values()) or 1.0
    for term, weight in weights.most_common(12):
        flag = "  <- REQUEST FRAMING" if term in FRAMING_WORDS else ""
        print(f"   {term:24}{weight:>12.1f}{weight / total:>9.1%}{flag}")
    framing = sum(weight for term, weight in weights.items()
                  if term in FRAMING_WORDS)
    print(f"   {'-- framing total --':24}{framing:>12.1f}"
          f"{framing / total:>9.1%}")
    print("\n   This section exists because that share was 30.9% -- `still`")
    print("   alone 23.3%, 8.6x the top real product word -- and nothing in the")
    print("   phase would have shown it (D Phase 14 review, Finding 2). Those")
    print("   words come from the evaluator's own sentence templates:")
    print("   \"...but I'm still exploring.\", \"A key requirement is:\",")
    print("   \"For that, what matters is:\". They are not the shopper's")
    print("   vocabulary, they are the simulator's, and ranking on them means")
    print("   ranking on which catalog description happens to contain the word")
    print("   \"still\". starter/state.py now strips them at distillation")
    print("   (_REQUEST_FRAMING), which is worth +0.0202 TS on its own. A")
    print("   nonzero framing share here means a template got past that set.")

    print(f"\nturns: {turns}   config: {config_guard.describe()}")


if __name__ == "__main__":
    main()
