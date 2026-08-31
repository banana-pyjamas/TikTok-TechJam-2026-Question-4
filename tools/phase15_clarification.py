"""Phase 15 CP 15.8 -- clarification OFF vs ON, and WHICH question to ask.

The ablation is the easy half. The hard half is that this benchmark cannot
tell a good question from a lucky one, and saying so is most of this tool.

``local_evaluator.customer_reply`` filters the undisclosed constraints with

    attribute == "other" or classify_constraint(value) == attribute

so the wildcard matches a strict SUPERSET of what any specific attribute
matches, on every turn, BY CONSTRUCTION. Asking ``"other"`` cannot harvest
fewer constraints than asking "color", and usually harvests more. Any
comparison of question policies on this harness is therefore scored on a rule
that structurally favours the least specific question, and a tool that just
printed the winner would be measuring the simulator for the fifth time in this
project (see tools/phase14_reranker.py section 6).

So the arms are laid out to separate the two things:

  1. NO-OP        does the flag's OFF position reproduce the committed
                  pre-Phase-15 score EXACTLY?
  2. SCORE        OFF vs ON, paired McNemar on per-session hit verdicts.
  3. POLICY       five question policies against each other: the wildcard
                  the harness structurally favours, the naive
                  first-empty-slot policy, the shipped one with its open
                  question removed, and the shipped one with the harness's
                  two unanswerable questions removed -- which is the size of
                  a prize this phase declines to take. What is interesting is
                  not which wins but WHY -- see section 3's note.
  4. BEHAVIOUR    what was actually asked, how often a question was declined,
                  how often the loop ran out (CP 15.6 / CP 15.7), and the
                  contract checks (CP 15.3 / CP 15.4) on live data.
  5. ROBUSTNESS   ASK_VALUE_FLOOR sweep: is the shipped value on a cliff?

Usage:  python3 -m tools.phase15_clarification
        (~6 min: 2 arms + 4 extra policies + 5 sweep rows)
"""

from __future__ import annotations

import time
from collections import Counter

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter import clarify
from starter.clarify import ALLOWED_ATTRIBUTES, SCORABLE_ATTRIBUTES, WILDCARD
from tools import config_guard
from tools.capture import CapturingAgent
from tools.significance import (format_test, hits_by_sample, mcnemar,
                                mttc_given_hit)

CATALOG = "data/catalog.jsonl"
DATASET = "data/public_set.jsonl"

# The score with USE_CLARIFICATION OFF: the Phase 14 pipeline exactly. A fixed
# historical reference, NOT the committed score -- the committed score is now
# the ON arm and it lives in config_guard. Same split, and for the same
# reason, as phase14's PRE_RERANK_TECHNICAL_SCORE.
PRE_CLARIFY_TECHNICAL_SCORE = 0.243930


def always_wildcard(context, ledger, ranked, metadata, reliabilities=None):
    """The policy this harness structurally favours. See the module docstring.

    Not gated on the ledger: the wildcard is the open question, and closing
    it here would only make this arm weaker than the thing it is meant to
    upper-bound.
    """
    return WILDCARD


def first_empty_slot(context, ledger, ranked, metadata, reliabilities=None):
    """The obvious policy, and the one to beat: ask about the first thing we
    do not know yet, in a fixed order, with no idea whether the answer would
    change anything.

    Worth running because it is what "ask a clarifying question" means before
    anyone measures it, and because the gap between it and the shipped policy
    is the only evidence that CP 15.5 does any work at all.
    """
    slots = getattr(getattr(context, "state", None), "slots", None)
    known = {name for name, entry in (slots or {}).items()
             if isinstance(entry, dict) and entry.get("values")}
    for attribute in SCORABLE_ATTRIBUTES:
        if attribute not in known and attribute not in ledger.closed:
            return attribute
    return WILDCARD if WILDCARD not in ledger.closed else None


def make_scored_no_fallback(original):
    """The shipped policy with the open question removed.

    The arm that isolates what the fallback is worth. If this scores close to
    the shipped policy, the wildcard is decoration; if it collapses, most of
    what Phase 15 buys is the open question and the scorer is choosing the
    ORDER rather than the content.

    Closes over ``original`` rather than reading ``clarify.choose``: the arm
    loop patches that name, so reading it here would call THIS function again.
    Phase 14's placebo builder made exactly that mistake, recursed until
    RecursionError, and -- because the evaluator swallows a raising turn into
    an empty response -- reported an arm of HR 0.0000 that read as a
    devastating result rather than a broken one.
    """

    def policy(context, ledger, ranked, metadata, reliabilities=None):
        chosen = original(context, ledger, ranked, metadata, reliabilities)
        return None if chosen == WILDCARD else chosen

    return policy


# The two attributes this harness can never answer. NOT a shipping decision --
# see ``make_harness_answerable`` -- but the fact needs a name.
#
# ``local_evaluator.classify_constraint`` maps a constraint string to one of
# budget / material / color / size / style / use_case / feature, and its final
# fallback is ``"feature"``. It has NO branch that returns "brand" and none
# that returns "category". So a question about either matches nothing in
# ``customer_reply``, returns "I don't have an additional preference for
# brand.", and is closed by CP 15.7 having cost exactly one turn.
UNANSWERABLE_BY_THE_HARNESS = ("brand", "category")


def make_harness_answerable(original):
    """The shipped policy with the two structurally-dead questions removed.

    THIS ARM IS THE SIMULATOR, MEASURED ON PURPOSE, AND IT DOES NOT SHIP.

    Asking a shopper about brand is a good question -- it is the single most
    discriminating thing the catalog knows (99.4% coverage, high diversity),
    which is exactly why the value scorer picks it. It is dead HERE because
    of a gap in the harness's own classifier, not because of anything about
    shopping, and a policy that skipped it would be fitted to
    ``classify_constraint``'s branch list. This project has measured the
    simulator four times already and written down what it costs each time.

    So the gap is quantified rather than exploited: this arm is the size of
    the prize, printed next to the shipped number, so the trade is visible
    to whoever reads it rather than silently taken or silently forgone.
    """

    def policy(context, ledger, ranked, metadata, reliabilities=None):
        for attribute in UNANSWERABLE_BY_THE_HARNESS:
            ledger.closed.add(attribute)
        return original(context, ledger, ranked, metadata, reliabilities)

    return policy


class AskAuditingAgent(CapturingAgent):
    """The shipped agent, plus an audit of the PAYLOAD it actually returned.

    CP 15.3 and CP 15.4 are properties of what goes on the wire, so they are
    checked there rather than at ``clarify.choose``'s return. The difference
    is not pedantic: a check on the policy function would pass even if
    ``_to_response`` dropped the question, mislabelled it, or emptied the
    recommendation list -- and "asked instead of answering" is precisely the
    bug CP 15.4 exists to forbid. A check that cannot go red is worse than no
    check (D Phase 14 review); this one can.
    """

    def __init__(self, catalog_path: str) -> None:
        super().__init__(catalog_path)
        self.audit: Counter = Counter()

    def respond(self, session_id: str, user_message: str, turn: int,
                top_k: int) -> dict:
        payload = super().respond(session_id, user_message, turn, top_k)
        ask = payload.get("ask_attribute")
        self.audit["turns"] += 1
        if ask is None:
            self.audit["silent"] += 1
            return payload
        self.audit["asked"] += 1
        if ask not in ALLOWED_ATTRIBUTES:
            self.audit["illegal"] += 1
        if not payload.get("recommendations"):
            self.audit["asked_without_recommendations"] += 1
        return payload


def instrument():
    """Wrap ``clarify.choose`` with observation. Read-only.

    Patching the module attribute, which is how ``agent.respond`` reaches it,
    so this observes the real decision rather than a reimplementation of it.
    Returns ``(stats, restore)``.
    """
    original = clarify.choose
    stats = {
        "asked": Counter(),
        "turns": 0,
        "silent": 0,
        "illegal": 0,
        "by_turn": Counter(),
        "closed_per_session": {},
        "asked_per_session": {},
        "repeat_after_close": 0,
    }

    def counting(context, ledger, ranked, metadata, reliabilities=None):
        chosen = original(context, ledger, ranked, metadata, reliabilities)
        stats["turns"] += 1
        # CP 15.3 on live data: nothing but the organizer's enum or None.
        if chosen is not None and chosen not in ALLOWED_ATTRIBUTES:
            stats["illegal"] += 1
        # CP 15.6 on live data: a closed attribute is never asked again.
        if chosen is not None and chosen in ledger.closed:
            stats["repeat_after_close"] += 1
        if chosen is None:
            stats["silent"] += 1
        else:
            stats["asked"][chosen] += 1
            stats["by_turn"][context.turn] += 1
        session = context.session_id
        stats["closed_per_session"][session] = len(ledger.closed)
        stats["asked_per_session"][session] = len(ledger.asked) + (
            0 if chosen is None else 1)
        return chosen

    clarify.choose = counting

    def restore() -> None:
        clarify.choose = original

    return stats, restore


def main() -> None:
    config_guard.assert_all_flags_pinned(set(config_guard.COMMITTED_FLAGS))
    config_guard.assert_committed_constants()
    config_guard.assert_committed_flags_match_source()
    config_guard.restore_committed_flags()

    print("building index...", flush=True)
    started = time.time()
    agent = AskAuditingAgent(CATALOG)
    samples = load_jsonl(DATASET)
    catalog_ids, categories, products = catalog_index(CATALOG)
    print(f"ready in {time.time() - started:.1f}s", flush=True)

    shipped = clarify.choose
    arms = (
        ("OFF", False, shipped),
        ("ON", True, shipped),
        ("wildcard", True, always_wildcard),
        ("first-empty", True, first_empty_slot),
        ("no-fallback", True, make_scored_no_fallback(shipped)),
        ("harness-fit", True, make_harness_answerable(shipped)),
    )
    results: dict[str, dict] = {}
    stats: dict[str, dict] = {}
    audits: dict[str, Counter] = {}

    for label, enabled, policy in arms:
        config_guard.set_flag("clarify", "USE_CLARIFICATION", enabled)
        clarify.choose = policy
        agent.order.clear()
        agent.captured.clear()
        agent.audit.clear()
        arm_stats, restore = instrument()
        started = time.time()
        results[label] = evaluate(agent, samples, catalog_ids, categories,
                                  products)
        restore()
        clarify.choose = shipped
        elapsed = time.time() - started
        stats[label] = arm_stats
        audits[label] = Counter(agent.audit)
        result = results[label]

        turn_count = len(agent.captured)
        if not turn_count:
            raise SystemExit(
                f"arm {label!r} captured 0 turns: every respond() raised and "
                "the evaluator swallowed it. Its numbers are a crash, not a "
                "measurement.")
        print(f"\n{label:12} USE_CLARIFICATION={enabled!s:5} "
              f"HR {result['hit_rate_at_10']:.4f} MRR {result['mrr']:.6f} "
              f"MTTC {result['mttc']:.3f} "
              f"TS {result['recommended_technical_score']:.6f} "
              f"({turn_count} turns, {elapsed:.0f}s)", flush=True)
    config_guard.restore_committed_flags()

    off, on = results["OFF"], results["ON"]

    # -- 1 -------------------------------------------------------------------
    print("\n1. NO-OP -- is the OFF arm a true no-op?")
    # ONE of these is an assertion and the other is a report, and which is
    # which is the lesson phase10/phase11/phase12 paid for three times over.
    #
    # ASSERTED: the ON arm reproduces the committed score. That is the check
    # that makes everything below a controlled comparison -- it says this run
    # measured the pipeline that ships.
    #
    # REPORTED, NEVER ASSERTED: the OFF arm against the pre-Phase-15 score.
    # It matches today. It will stop matching the moment any LATER phase
    # lands, because "this pipeline minus clarification" is not a constant --
    # it moves with everything upstream. Asserting it would make this tool
    # exit on its own staleness, which is exactly how phase10 and phase11
    # spent two phases each failing for the one reason a guard must never
    # fail. So drift is printed with its size and the run continues.
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
    drift = actual_off - PRE_CLARIFY_TECHNICAL_SCORE
    print(f"   OFF vs the pre-Phase-15 pipeline         "
          f"{PRE_CLARIFY_TECHNICAL_SCORE}   "
          f"{'exact' if abs(drift) <= 1e-9 else f'{actual_off} ({drift:+.6f})'}"
          f"   [reported, not asserted]")
    if abs(drift) > 1e-9:
        print("   The OFF arm has moved since Phase 15. That is not a failure:")
        print("   it means a later phase changed the pipeline this flag sits")
        print("   in, and the historical literal above is now history. The")
        print("   comparison that decides anything is the ON arm above.")
    print("   With the flag OFF no ledger is read, no question is chosen and")
    print("   `ask_attribute` is None on every turn -- the payload is")
    print("   byte-for-byte the Phase 14 one.")

    # -- 2 -------------------------------------------------------------------
    print("\n2. SCORE -- OFF vs ON")
    for label in ("OFF", "ON"):
        result = results[label]
        conditional = mttc_given_hit(result)
        print(f"   {label:12}HR {result['hit_rate_at_10']:.4f}  "
              f"MRR {result['mrr']:.6f}  MTTC {result['mttc']:.3f}  "
              f"MTTC|hit {0.0 if conditional is None else conditional:.3f}"
              f"  TS {result['recommended_technical_score']:.6f}"
              f"  {result['recommended_technical_score'] - off['recommended_technical_score']:+.6f}")
    print("   " + format_test("ON vs OFF",
                              mcnemar(hits_by_sample(off), hits_by_sample(on))))
    print(f"\n   {'scenario':18}{'OFF':>10}{'ON':>10}{'ON-OFF':>10}")
    for scenario in ("buying", "browsing", "intent_override", "boundary"):
        a = off["scenario_metrics"][scenario]["hit_rate_at_10"]
        b = on["scenario_metrics"][scenario]["hit_rate_at_10"]
        print(f"   {scenario:18}{a:>10.4f}{b:>10.4f}{b - a:>+10.4f}")
    print("\n   No placebo arm here, and the Phase 14 reason for having one")
    print("   does not apply: this stage does not REORDER anything. It changes")
    print("   what the shopper says, so a 'meaningless question' control would")
    print("   be a question that harvests no constraints -- which is exactly")
    print("   the OFF arm. OFF *is* the null.")

    # -- 3 -------------------------------------------------------------------
    print("\n3. POLICY -- which question, and what the choice is worth")
    print(f"   {'policy':14}{'HR':>9}{'MRR':>11}{'MTTC':>8}{'TS':>11}"
          f"{'vs OFF':>11}   McNemar vs OFF")
    for label in ("ON", "wildcard", "first-empty", "no-fallback",
                  "harness-fit"):
        result = results[label]
        test = mcnemar(hits_by_sample(off), hits_by_sample(result))
        note = "  <- shipped" if label == "ON" else ""
        print(f"   {label:14}{result['hit_rate_at_10']:>9.4f}"
              f"{result['mrr']:>11.6f}{result['mttc']:>8.3f}"
              f"{result['recommended_technical_score']:>11.6f}"
              f"{result['recommended_technical_score'] - off['recommended_technical_score']:>+11.6f}"
              f"   {test['gained']}/{test['lost']}, p = {test['p']:.4f}{note}")
    print("   " + format_test("shipped vs wildcard",
                              mcnemar(hits_by_sample(results["wildcard"]),
                                      hits_by_sample(on))))
    print("   " + format_test("shipped vs first-empty",
                              mcnemar(hits_by_sample(results["first-empty"]),
                                      hits_by_sample(on))))
    print("   " + format_test("harness-fit vs shipped",
                              mcnemar(hits_by_sample(on),
                                      hits_by_sample(results["harness-fit"]))))
    print("\n   READ THIS BEFORE THE RANKING. `\"other\"` is a WILDCARD in")
    print("   customer_reply -- `attribute == \"other\" or")
    print("   classify_constraint(value) == attribute` -- so it matches a")
    print("   strict superset of what any specific question matches, every")
    print("   turn, by construction. The wildcard arm therefore harvests the")
    print("   MOST constraints per turn that any policy can, and the harness")
    print("   structurally favours it.")
    print("\n   So if the shipped policy beats it, the gain cannot be 'asked")
    print("   more'. It is WHICH constraint arrives: a colour or a material")
    print("   becomes a SLOT the ranker scores on, while an arbitrary feature")
    print("   string becomes free-text evidence only. Quantity of disclosure")
    print("   is what the wildcard maximises; usable disclosure is what the")
    print("   value scorer maximises, and only the second is a fact about")
    print("   shopping rather than about this evaluator.")

    # -- 4 -------------------------------------------------------------------
    shipped_stats = stats["ON"]
    print("\n4. BEHAVIOUR -- what was asked, and did the loop stay bounded")
    asked = shipped_stats["asked"]
    total_asks = sum(asked.values())
    print(f"   questions asked over {shipped_stats['turns']} turns: "
          f"{total_asks}   silent turns: {shipped_stats['silent']}")
    for attribute, count in asked.most_common():
        print(f"   {attribute:14}{count:>6}{count / max(total_asks, 1):>9.1%}")
    print(f"\n   {'turn':>6}{'asked':>8}")
    for turn in sorted(shipped_stats["by_turn"]):
        print(f"   {turn:>6}{shipped_stats['by_turn'][turn]:>8}")
    per_session = shipped_stats["asked_per_session"]
    closed = shipped_stats["closed_per_session"]
    audit = audits["ON"]
    print(f"\n   checked on the PAYLOAD, not on the policy's return value")
    print(f"   turns that put a question on the wire            "
          f"{audit['asked']:>6}   silent {audit['silent']}")
    print(f"   CP 15.3  ask_attribute outside the contract enum "
          f"{audit['illegal']:>6}  "
          f"{'PASS' if not audit['illegal'] else 'FAIL'}")
    print(f"   CP 15.4  turns that asked without recommending   "
          f"{audit['asked_without_recommendations']:>6}  "
          f"{'PASS' if not audit['asked_without_recommendations'] else 'FAIL'}"
          f"   ({audit['asked']} live asks to fail on)")
    print(f"   CP 15.6  a closed attribute asked again          "
          f"{shipped_stats['repeat_after_close']:>6}  "
          f"{'PASS' if not shipped_stats['repeat_after_close'] else 'FAIL'}")
    print(f"   CP 15.7  most questions in one session           "
          f"{max(per_session.values(), default=0):>6}  "
          f"(bound is {len(ALLOWED_ATTRIBUTES)} attributes, "
          f"then None for the rest of the session)")
    print(f"   CP 15.7  most attributes closed in one session   "
          f"{max(closed.values(), default=0):>6}")
    print(f"   sessions that ran out of questions entirely       "
          f"{sum(1 for value in closed.values() if value >= len(ALLOWED_ATTRIBUTES)):>5}")

    # -- 5 -------------------------------------------------------------------
    print("\n5. ROBUSTNESS -- is ASK_VALUE_FLOOR on a cliff?")
    print("   The floor decides how good a specific question has to be before")
    print("   it is preferred to the open one. 0.0 asks a specific question")
    print("   whenever any attribute has ANY value; 1.0 never does and is the")
    print("   wildcard arm exactly. The shipped 0.10 was chosen to exclude the")
    print("   cases the formula already reports as near-zero, not read off")
    print("   this table.")
    print(f"   {'floor':>8}{'HR':>10}{'MRR':>12}{'TS':>12}{'vs OFF':>11}")
    committed_floor = clarify.ASK_VALUE_FLOOR
    config_guard.set_flag("clarify", "USE_CLARIFICATION", True)
    try:
        for floor in (0.0, 0.05, 0.10, 0.20, 0.40):
            clarify.ASK_VALUE_FLOOR = floor
            agent.order.clear()
            agent.captured.clear()
            swept = evaluate(agent, samples, catalog_ids, categories, products)
            marker = "  <- committed" if floor == committed_floor else ""
            print(f"   {floor:>8.2f}{swept['hit_rate_at_10']:>10.4f}"
                  f"{swept['mrr']:>12.6f}"
                  f"{swept['recommended_technical_score']:>12.6f}"
                  f"{swept['recommended_technical_score'] - off['recommended_technical_score']:>+11.6f}"
                  f"{marker}", flush=True)
    finally:
        clarify.ASK_VALUE_FLOOR = committed_floor
        config_guard.restore_committed_flags()

    print(f"\nturns: {len(agent.captured)}   config: {config_guard.describe()}")


if __name__ == "__main__":
    main()
