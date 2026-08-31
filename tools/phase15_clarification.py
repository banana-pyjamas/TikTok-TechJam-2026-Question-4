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
from tools.significance import (composites_by_sample, format_composite,
                                format_test, hits_by_sample, mcnemar,
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
        "specific_per_session": {},
        "wildcard_per_session": {},
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
        asked = list(ledger.asked) + ([] if chosen is None else [chosen])
        stats["asked_per_session"][session] = len(asked)
        stats["wildcard_per_session"][session] = asked.count(WILDCARD)
        stats["specific_per_session"][session] = len(asked) - asked.count(WILDCARD)
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
    # The four arms the B Phase 15 blocker asks for, plus the two diagnostics
    # that were already here. A/B/C differ ONLY in the open question's
    # per-session budget, which is the one knob under review.
    #
    # The budget is set on the MODULE CONSTANT, not smuggled through the
    # ledger. The first version of this loop wrapped the policy and force-
    # closed the wildcard once a session had spent its budget, which can only
    # ever LOWER the budget -- `choose` still read `MAX_OPEN_QUESTIONS` from
    # the module, so the "unbounded" arm silently ran at the committed value
    # of 1 and measured the arm next to it. The constant is restored in a
    # `finally`, and `config_guard.assert_committed_constants` at the end of
    # the run is what proves it was.
    unbounded = 10 ** 6
    arms = (
        ("OFF", False, shipped, None),
        ("A unbounded", True, shipped, unbounded),
        ("B no-other", True, shipped, 0),
        ("C one-other", True, shipped, 1),
        ("D wildcard", True, always_wildcard, None),
        ("first-empty", True, first_empty_slot, None),
        ("no-fallback", True, make_scored_no_fallback(shipped), None),
        ("harness-fit", True, make_harness_answerable(shipped), None),
    )
    # Which arm is the SHIPPED configuration. Derived from the constant rather
    # than hardcoded, so changing MAX_OPEN_QUESTIONS moves this label with it.
    shipped_arm = {0: "B no-other", 1: "C one-other"}.get(
        clarify.MAX_OPEN_QUESTIONS, "A unbounded")
    results: dict[str, dict] = {}
    stats: dict[str, dict] = {}
    audits: dict[str, Counter] = {}

    committed_budget = clarify.MAX_OPEN_QUESTIONS
    for label, enabled, policy, budget in arms:
        config_guard.set_flag("clarify", "USE_CLARIFICATION", enabled)
        clarify.choose = policy
        if budget is not None:
            clarify.MAX_OPEN_QUESTIONS = budget
        agent.order.clear()
        agent.captured.clear()
        agent.audit.clear()
        arm_stats, restore = instrument()
        started = time.time()
        try:
            results[label] = evaluate(agent, samples, catalog_ids, categories,
                                      products)
        finally:
            restore()
            clarify.choose = shipped
            clarify.MAX_OPEN_QUESTIONS = committed_budget
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
    # Proves the sweep put every pinned constant back, including the budget
    # the arms above moved. A tool that leaves a constant shifted reports the
    # next section under a configuration it never names.
    config_guard.assert_committed_constants()

    off, on = results["OFF"], results[shipped_arm]

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
    for label in ("OFF", shipped_arm):
        result = results[label]
        conditional = mttc_given_hit(result)
        print(f"   {label:12}HR {result['hit_rate_at_10']:.4f}  "
              f"MRR {result['mrr']:.6f}  MTTC {result['mttc']:.3f}  "
              f"MTTC|hit {0.0 if conditional is None else conditional:.3f}"
              f"  TS {result['recommended_technical_score']:.6f}"
              f"  {result['recommended_technical_score'] - off['recommended_technical_score']:+.6f}")
    print("   " + format_test("ON vs OFF (hits)",
                              mcnemar(hits_by_sample(off), hits_by_sample(on))))
    print("   " + format_composite("ON vs OFF (score)",
                                   composites_by_sample(off),
                                   composites_by_sample(on)))
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
    print("\n3. POLICY -- which question, and how much of the gain is the")
    print("   open question doing")
    print("   A / B / C differ in ONE knob: MAX_OPEN_QUESTIONS, the number of")
    print("   times a session may fall back to \"other\". D is a diagnostic")
    print("   upper bound, not a candidate.")
    print(f"\n   {'policy':14}{'HR':>9}{'MRR':>11}{'MTTC':>8}{'TS':>11}"
          f"{'vs OFF':>11}{'q/sess':>9}{'spec':>8}{'other':>8}{'used':>7}")
    order = ["A unbounded", "B no-other", "C one-other", "D wildcard",
             "first-empty", "no-fallback", "harness-fit"]
    for label in order:
        result = results[label]
        arm = stats[label]
        sessions = max(len(arm["asked_per_session"]), 1)
        per = sum(arm["asked_per_session"].values()) / sessions
        specific = sum(arm["specific_per_session"].values()) / sessions
        wildcard = sum(arm["wildcard_per_session"].values()) / sessions
        used = sum(1 for v in arm["wildcard_per_session"].values() if v)
        note = "  <- SHIPPED" if label == shipped_arm else ""
        print(f"   {label:14}{result['hit_rate_at_10']:>9.4f}"
              f"{result['mrr']:>11.6f}{result['mttc']:>8.3f}"
              f"{result['recommended_technical_score']:>11.6f}"
              f"{result['recommended_technical_score'] - off['recommended_technical_score']:>+11.6f}"
              f"{per:>9.2f}{specific:>8.2f}{wildcard:>8.2f}"
              f"{used:>5}/{sessions}{note}")

    print("\n   paired tests against OFF -- BOTH tests, because they answer")
    print("   different questions and this repo has now mispriced a verdict")
    print("   twice by quoting only the first (D Phase 15 review):")
    for label in order:
        test = mcnemar(hits_by_sample(off), hits_by_sample(results[label]))
        print("     " + format_test(f"{label} hits", test))
    print()
    for label in order:
        print("     " + format_composite(
            f"{label} score", composites_by_sample(off),
            composites_by_sample(results[label])))

    print("\n   the comparisons that decide the knob, A/B/C against each")
    print("   other -- and harness-fit against the SHIPPED arm, which is the")
    print("   comparison clarify.USE_CLARIFICATION's comment justifies a")
    print("   decision from. It was only ever measured against OFF, leaving")
    print("   that decision unpriced by its own instrument (D Phase 15")
    print("   review).")
    for left, right in (("A unbounded", "C one-other"),
                        ("B no-other", "C one-other"),
                        ("B no-other", "A unbounded"),
                        (shipped_arm, "harness-fit")):
        print("     " + format_test(
            f"{right} vs {left} hits",
            mcnemar(hits_by_sample(results[left]),
                    hits_by_sample(results[right]))))
        print("     " + format_composite(
            f"{right} vs {left} score",
            composites_by_sample(results[left]),
            composites_by_sample(results[right])))

    print(f"\n   {'scenario':18}" + "".join(f"{label[:11]:>13}" for label in
                                            ("OFF", "A unbounded",
                                             "B no-other", "C one-other",
                                             "D wildcard")))
    for scenario in ("buying", "browsing", "intent_override", "boundary"):
        row = "".join(
            f"{results[label]['scenario_metrics'][scenario]['hit_rate_at_10']:>13.4f}"
            for label in ("OFF", "A unbounded", "B no-other", "C one-other",
                          "D wildcard"))
        print(f"   {scenario:18}{row}")

    print("\n   READ THIS BEFORE THE RANKING. `\"other\"` is a WILDCARD in")
    print("   customer_reply -- `attribute == \"other\" or")
    print("   classify_constraint(value) == attribute` -- so it matches a")
    print("   strict superset of what any specific question matches, every")
    print("   turn, by construction. The harness structurally favours the")
    print("   least specific policy, so a policy chosen for topping this")
    print("   table would be chosen by the simulator.")
    print("\n   WHAT THE A/C ROW PAIR SAYS, which is the answer to the")
    print("   blocker: capping the open question is SCORE-free and NOT")
    print("   behaviour-free, and the difference matters. Read the 'other'")
    print("   column against 'used': A's 0.33 per session over 42 sessions is")
    print("   66 asks, so 24 sessions asked the open question more than once")
    print("   -- 10 of which go on to hit. C asks it exactly once in each of")
    print("   the same 42. Every metric is nonetheless equal to six decimals")
    print("   and 0 of 200 sessions moved, so the cap removes 24 repeat")
    print("   questions that changed no outcome.")
    print("\n   That is a better thing to be able to say than 'nothing was")
    print("   happening'. The thing the cap forbids -- re-asking a productive")
    print("   wildcard until it dries up -- WAS happening, on 24 sessions,")
    print("   and it is farming a harness whose rules favour it. It now")
    print("   cannot happen, at no measured cost.")
    print("\n   B, the strictly-generic policy, costs an ESTABLISHED 0.0370")
    print("   (9/0 discordant, p = 0.0039; score p = 0.0017). That is the")
    print("   honest price of removing the open question entirely, and it is")
    print("   not inside the noise. It is available by setting")
    print("   MAX_OPEN_QUESTIONS to 0 -- a one-line change with a measured,")
    print("   named cost rather than a judgement call.")
    print("\n   Note what the 'other' column now reads: 0.21 per session, 42")
    print("   of 200 sessions, against D's 3.72 and 200 of 200. The reliance")
    print("   the blocker was raised about is real and is now bounded and")
    print("   small. The reason it fell is the EVIDENCE tier, not the cap:")
    print("   `feature` / `use_case` / `style` are specific questions that")
    print("   reach constraints the six catalog-checkable attributes cannot,")
    print("   and adding them is what made the wildcard optional rather than")
    print("   load-bearing. See clarify.EVIDENCE_ATTRIBUTES for why they were")
    print("   wrongly excluded in the first place.")

    # -- 4 -------------------------------------------------------------------
    shipped_stats = stats[shipped_arm]
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
    audit = audits[shipped_arm]
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
    print("   it is preferred to an unscored one. 0.0 asks a value-scored")
    print("   question whenever any attribute has ANY value; 1.0 never does,")
    print("   and falls straight through to the evidence tier. The shipped")
    print("   0.10 was chosen to exclude the cases the formula already")
    print("   reports as near-zero, not read off this table -- it happens to")
    print("   top it, which is luck rather than method.")
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
