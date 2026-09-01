"""Phase 11 — Evidence Confidence x Match Reliability, measured.

Two questions, in this order, because the second only matters if the first
has an honest answer:

  1. does the weighting change the score, and is that change established?
  2. if not, WHY not -- is the mechanism wrong, or does this dialogue simply
     not contain the situations it exists for?

The second question is the one that keeps a neutral result informative. A
mechanism can be right and still be inert on a particular 200 sessions, and
the way to tell that apart from a broken mechanism is to count the situations
it is supposed to act on rather than to shrug at a flat score.

Also checks the Match Reliability table against the labels. MR is derived from
catalog coverage with no labels involved (see ``starter/reliability.py``);
this tool computes the label-based version -- how often the KNOWN target is
condemned on each slot -- purely as a check on that derivation. If a slot with
high derived reliability turns out to condemn targets constantly, the
derivation is wrong and this is where it shows.

Usage:  python3 -m tools.phase11_confidence
"""

from __future__ import annotations

import time
from collections import Counter

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter import ranking
from starter.catalog_meta import lookup as meta_lookup
from starter.contracts import Context
from starter.ranking import (RELIABILITY_KEY, SCORED_SLOTS, VIOLATION_SLOTS,
                             active_constraints, classify, constraint_weights,
                             rank, slot_confidence)
from starter.retrieval import DEFAULT_ROUTES, POOL_LIMIT, retrieve
from starter.reliability import match_reliability, reliability_of, slot_coverage
from starter.state import EC_HEDGED, EC_REQUIREMENT, EC_STATED
from tools import config_guard
from tools.capture import CapturingAgent
from tools.significance import format_test, hits_by_sample, mcnemar

CATALOG = "data/catalog.jsonl"
DATASET = "data/public_set.jsonl"

# Above this an assertion counts as firmly meant; below it, hedged. Only used
# to bucket the census below -- the scoring path uses the continuous value.
HIGH_EC = 0.65
HIGH_MR = 0.5


def main() -> None:
    config_guard.assert_all_flags_pinned(set(config_guard.COMMITTED_FLAGS))
    config_guard.assert_committed_constants()
    config_guard.restore_committed_flags()

    print("building index...", flush=True)
    started = time.time()
    agent = CapturingAgent(CATALOG)
    samples = load_jsonl(DATASET)
    catalog_ids, categories, products = catalog_index(CATALOG)
    print(f"ready in {time.time() - started:.1f}s", flush=True)

    reliabilities = match_reliability(slot_coverage(agent.connection))
    print("\nCP 11.2  Match Reliability, derived from catalog coverage")
    print(f"   {'slot':12}{'coverage':>10}{'MR':>8}   in VIOLATION_SLOTS?")
    coverage = slot_coverage(agent.connection)
    for slot in SCORED_SLOTS:
        print(f"   {slot:12}{coverage.get(slot, float('nan')):>10.1%}"
              f"{reliability_of(reliabilities, slot):>8.2f}"
              f"   {'yes' if slot in VIOLATION_SLOTS else 'no (hand-excluded in Phase 6)'}")

    # -- 1. does it change the score? -------------------------------------
    results = {}
    for enabled in (False, True):
        config_guard.set_flag("ranking", "USE_CONFIDENCE_WEIGHTING", enabled)
        # Two full evaluations run through one capturing agent, so the buffers
        # have to be cleared between them -- otherwise `order` holds 400
        # sessions against 200 samples and the census reads the wrong run.
        # The census uses the LAST run; state and constraints are identical in
        # both arms, since weighting affects scoring only.
        agent.order.clear()
        agent.captured.clear()
        started = time.time()
        results[enabled] = evaluate(agent, samples, catalog_ids, categories, products)
        result = results[enabled]
        print(f"\nUSE_CONFIDENCE_WEIGHTING={enabled!s:5} "
              f"HR {result['hit_rate_at_10']:.4f} MRR {result['mrr']:.6f} "
              f"MTTC {result['mttc']:.3f} "
              f"TS {result['recommended_technical_score']:.6f} "
              f"({time.time() - started:.0f}s)", flush=True)
    config_guard.restore_committed_flags()

    off, on = results[False], results[True]
    committed = config_guard.COMMITTED_TECHNICAL_SCORE
    if abs(off["recommended_technical_score"] - committed) > 1e-9:
        raise SystemExit(
            "OFF no longer reproduces the committed score. The flag's OFF "
            "position must be exact; fix that before reading anything else.")
    print(f"\n   OFF reproduces the committed {committed} exactly, so the "
          "flag's OFF position\n   is a true no-op and the comparison below is "
          "controlled.")
    print("   " + format_test("weighting ON vs OFF",
                              mcnemar(hits_by_sample(off), hits_by_sample(on))))

    # -- 2. why? the census of situations the phase exists for -------------
    print("\ncensus over the live dialogue: what the mechanism had to work with")
    per_turn_constraints: Counter = Counter()
    quadrants: Counter = Counter()
    ec_values: Counter = Counter()
    weight_spread_turns = 0
    turns = 0
    target_verdicts: dict[str, Counter] = {slot: Counter() for slot in SCORED_SLOTS}
    target_of = {
        session_id: str(sample["ground_truth"]["parent_asin"])
        for session_id, sample in zip(agent.order, samples)
    }

    for session_id, captures in agent.by_session().items():
        target = target_of[session_id]
        target_meta = meta_lookup(agent.connection, [target]).get(target, {})
        for capture in captures:
            turns += 1
            context = Context(
                session_id=session_id, turn=capture["turn"],
                user_message=capture["message"], state=capture["state"],
            )
            context.derived[RELIABILITY_KEY] = reliabilities
            constraints, bounds = active_constraints(context)
            per_turn_constraints[len(constraints)] += 1
            weights = constraint_weights(context, constraints)
            if len(set(round(w, 6) for w in weights.values())) > 1:
                weight_spread_turns += 1
            for slot in constraints:
                confidence = slot_confidence(context, slot)
                ec_values[round(confidence, 2)] += 1
                quadrants[(
                    "high EC" if confidence >= HIGH_EC else "low EC",
                    "high MR" if reliability_of(reliabilities, slot) >= HIGH_MR
                    else "low MR",
                )] += 1
                # The label-based check on MR: how often does this slot
                # condemn the product we KNOW is right?
                target_verdicts[slot][
                    classify(slot, constraints[slot], target_meta, bounds)] += 1

    print(f"   active constraints per turn (of {turns} turns):")
    for count in sorted(per_turn_constraints):
        share = per_turn_constraints[count] / turns
        print(f"     {count} constraint(s){per_turn_constraints[count]:>8}{share:>9.1%}")
    print(f"   turns where the weights DIFFER between slots: "
          f"{weight_spread_turns:>6}{weight_spread_turns / turns:>9.1%}")
    print("   These counts describe the INPUT, and nothing more. An earlier "
          "version of\n   this tool argued from them that a single-constraint "
          "turn 'cannot reorder'.\n   That was false -- `base` is not scaled "
          "by the weight, so one constraint is\n   enough to cross two "
          "candidates (B Phase 11 review). The effect is measured\n   "
          "directly below instead of inferred from here.")

    print("\n   Evidence Confidence actually assigned:")
    for value in sorted(ec_values, reverse=True):
        label = {EC_REQUIREMENT: "requirement", 0.9: "correction",
                 EC_STATED: "plain statement", EC_HEDGED: "hedged"}.get(value, "")
        total = sum(ec_values.values())
        print(f"     {value:<5} {label:16}{ec_values[value]:>8}"
              f"{ec_values[value] / total:>9.1%}")

    print("\n   quadrant census (constraint occurrences):")
    total = sum(quadrants.values()) or 1
    for ec in ("high EC", "low EC"):
        for mr in ("high MR", "low MR"):
            count = quadrants[(ec, mr)]
            cp = {("high EC", "high MR"): "CP 11.3",
                  ("high EC", "low MR"): "CP 11.4",
                  ("low EC", "high MR"): "CP 11.5",
                  ("low EC", "low MR"): "both weak"}[(ec, mr)]
            print(f"     {ec:9}/{mr:9}{cp:>12}{count:>9}{count / total:>9.1%}")

    # -- 2b. DIRECT rank sensitivity, not inferred ------------------------
    #
    # The aggregates above establish that no session changed its Top-10 hit
    # verdict. They cannot establish that nothing moved: a reorder below rank
    # 10, or one among non-targets above it, is invisible to HR, MRR and MTTC
    # alike. So rank each captured turn's pool BOTH ways and diff the orders.
    print("\nrank sensitivity, measured directly (same pool, ranked twice)")
    order_changed = 0
    top10_order_changed = 0
    top10_set_changed = 0
    target_eligible = 0
    target_moved = 0
    deltas: list[int] = []
    changed_by_constraints: Counter = Counter()
    eligible_turns = 0
    started = time.time()

    for session_id, captures in agent.by_session().items():
        target = target_of[session_id]
        for capture in captures:
            context = Context(
                session_id=session_id, turn=capture["turn"],
                user_message=capture["message"], state=capture["state"],
            )
            context.derived[RELIABILITY_KEY] = reliabilities
            pool = retrieve(agent.connection, context, POOL_LIMIT, DEFAULT_ROUTES)
            if not pool:
                continue
            eligible_turns += 1
            metadata = meta_lookup(agent.connection,
                                   [c.parent_asin for c in pool])
            orders = {}
            for enabled in (False, True):
                ranking.USE_CONFIDENCE_WEIGHTING = enabled
                orders[enabled] = [
                    c.parent_asin
                    for c in rank(pool, context, metadata, len(pool)).ranked
                ]
            ranking.USE_CONFIDENCE_WEIGHTING = False

            constraints, _ = active_constraints(context)
            if orders[False] != orders[True]:
                order_changed += 1
                changed_by_constraints[len(constraints)] += 1
            # Order and SET are different questions: a swap inside the top 10
            # changes the order a shopper sees while leaving the scored set
            # -- and therefore HR -- untouched. Reported separately.
            if orders[False][:10] != orders[True][:10]:
                top10_order_changed += 1
            if set(orders[False][:10]) != set(orders[True][:10]):
                top10_set_changed += 1
            if target in orders[False]:
                target_eligible += 1
                before = orders[False].index(target)
                after = orders[True].index(target)
                if before != after:
                    target_moved += 1
                    deltas.append(after - before)

    print(f"   turns with ANY candidate-order change   {order_changed:>7}"
          f"{order_changed / eligible_turns:>9.1%}  of {eligible_turns}")
    print(f"   turns with a Top-10 ORDER change        {top10_order_changed:>7}"
          f"{top10_order_changed / eligible_turns:>9.1%}")
    print(f"   turns with a Top-10 SET change          {top10_set_changed:>7}"
          f"{top10_set_changed / eligible_turns:>9.1%}  "
          "(what HR can even see)")
    print(f"   turns where the TARGET's rank moved     {target_moved:>7}"
          f"{(target_moved / target_eligible if target_eligible else 0):>9.1%}"
          f"  of {target_eligible} turns with the target in pool")
    if deltas:
        ordered = sorted(deltas)
        mean = sum(ordered) / len(ordered)
        print(f"   target rank delta (+ = worse): mean {mean:+.2f}  "
              f"median {ordered[len(ordered) // 2]:+d}  "
              f"min {ordered[0]:+d}  max {ordered[-1]:+d}")
    else:
        print("   target rank delta: n/a -- no target's rank moved")
    if changed_by_constraints:
        print("   order changes by active-constraint count:")
        for count in sorted(changed_by_constraints):
            print(f"     {count} constraint(s){changed_by_constraints[count]:>8}")
        if changed_by_constraints.get(1):
            print("     note the 1-constraint row: a single constraint DOES "
                  "reorder, which is\n     what makes the old census argument "
                  "wrong.")
    print(f"   ({time.time() - started:.0f}s)")

    # -- 3. the label-based check on the derived MR -----------------------
    print("\nCP 11.2 check: how often each slot CONDEMNS the known target")
    print("   (label-based, and used only to check the label-free derivation)")
    print(f"   {'slot':12}{'violations':>12}{'of decided':>12}{'rate':>9}"
          f"{'derived MR':>12}")
    for slot in SCORED_SLOTS:
        counts = target_verdicts[slot]
        decided = counts[ranking.MATCH] + counts[ranking.VIOLATION]
        if not decided:
            continue
        rate = counts[ranking.VIOLATION] / decided
        print(f"   {slot:12}{counts[ranking.VIOLATION]:>12}{decided:>12}"
              f"{rate:>9.1%}{reliability_of(reliabilities, slot):>12.2f}")
    print("   A slot that condemns the right product often is one whose "
          "verdicts are worth\n   little. If that ranking disagrees with the "
          "derived MR column, the coverage\n   derivation is wrong and should "
          "be replaced -- not quietly re-fitted here.")

    print(f"\nconfig: {config_guard.describe()}")


if __name__ == "__main__":
    main()
