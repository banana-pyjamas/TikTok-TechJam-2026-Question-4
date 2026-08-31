"""Clarification (Phase 15) -- deciding what to ASK, and whether to ask at all.

Every phase before this one made the agent better at answering. This is the
first one that makes it participate in the conversation, and on this benchmark
it is the single largest lever there is: `ask_attribute` has been hardcoded to
``None`` in every commit of this repository, and the evaluator's customer only
ever discloses a hidden constraint in reply to a question. With no question,
1557 of 1787 turns are the same stuck sentence -- "Those options are not quite
right yet. Ask me about one specific attribute." -- and the shopper's real
requirements are never spoken at all.

WHAT THIS FILE DECIDES

    CP 15.2  whether to ask this turn at all
    CP 15.3  which values are legal to emit
    CP 15.5  which attribute is worth the most, if we do ask
    CP 15.6  not asking the same thing twice
    CP 15.7  what a "no preference" answer means, and why the loop ends

CP 15.1 (the no-ask baseline) is the ``USE_CLARIFICATION`` flag below, and
CP 15.4 (recommendations AND a question in the same turn) is a property of
``agent._to_response``, which builds both from the same ranked result and
cannot emit one without the other.

THE THING TO READ BEFORE THE ABLATION NUMBERS

``"other"`` is a WILDCARD in this evaluator, and it is not a close call --
``local_evaluator.customer_reply`` filters the undisclosed constraints with

    attribute == "other" or classify_constraint(value) == attribute

so ``"other"`` matches a strict SUPERSET of what any specific attribute
matches, on every turn, by construction. It cannot score worse than a specific
question and will usually score better, because it harvests up to two
constraints of ANY class per turn while "color" harvests only colours. That is
a fact about the harness's source, not a finding about question quality, and no
amount of measuring on this benchmark can separate "the wildcard is the best
question" from "the wildcard is the only question the simulator rewards".

This repository has now measured the simulator four times (see
``tools/phase14_reranker.py`` section 6 for the most recent). So the numbers
are reported both ways in ``tools/phase15_clarification.py``, the structural
argument above is stated next to them, and the choice of what ships is made
with both in view rather than by reading the larger number.

WHAT IT ACTUALLY ASKS

Over the 200 live sessions (918 questions on 932 turns, 14 turns silent):

    feature   282  30.7%     use_case   57   6.2%     budget   37   4.0%
    brand     200  21.8%     style      50   5.4%     size     19   2.1%
    color     116  12.6%     other      42   4.6%     category      never
    material  115  12.5%

``category`` is never asked because the opening message always fills it. The
open question is 4.6% of all questions and reaches 42 of 200 sessions, which
is the number the B Phase 15 review asked to see come down; see
MAX_OPEN_QUESTIONS for how.

TWO TIERS OF QUESTION, AND WHY THE SECOND ONE EXISTS

The contract allows ten values (CP 15.3). Nine of them are askable here, in
two tiers:

  SCORABLE    category, color, material, brand, size, budget -- exactly the
              six ``state.SLOT_CARDINALITY`` can store and exactly the six
              ``catalog_meta`` carries a column for. An answer becomes a slot
              and is checkable, so a VALUE can be computed for the question
              (see ``attribute_value``) and the best one is asked first.

  EVIDENCE    feature, use_case, style -- no slot, no catalog column, and so
              no computable value. Asked in fixed order after the scorable
              ones.

The first version of this file had only the first tier, on the argument that
an unslotted answer "cannot change the ranking, so asking for one buys a
sentence and nothing else". That argument was wrong, and Phase 14 is the
counterexample: ``reranker.PoolTermScorer`` ranks the window on the shopper's
still-active free-text evidence, which is precisely where an unslotted answer
lands. Such an answer changes the ranking through the reranker instead of
through ``score_candidate`` -- a different path, not no path.

The error was expensive rather than academic. With only six askable
attributes the open question was the only way to reach most of what a shopper
has to say, which is what made ``"other"`` load-bearing and drew the review.
"""

from __future__ import annotations

import statistics
from typing import Any, Iterable, Sequence

from starter.contracts import Candidate, Context
from starter.reliability import reliability_of
from starter.state import is_non_answer

# CP 15.1 / CP 15.8. The no-ask baseline is this flag OFF: `ask_attribute` is
# ``None`` on every turn and the pipeline is byte-for-byte the Phase 14 one.
#
# ON, and it is not close -- `python3 -m tools.phase15_clarification`
# reproduces all of this:
#
#   OFF   HR 0.2850  MRR 0.162099  MTTC 8.360  TS 0.243930
#   ON    HR 0.8500  MRR 0.580597  MTTC 4.810  TS 0.722979   +0.479049
#
#   ON vs OFF (hits)    +113   114/1 discordant       p = 0.0000  established
#   ON vs OFF (score)   +0.479049  95% CI [+0.4183, +0.5383]
#                       125/200 sessions moved        p = 0.0001  established
#
# BOTH tests, because they answer different questions and this project has
# now mispriced a verdict twice by quoting only the first (D Phase 15 review).
# McNemar sees the hit SET; the paired permutation over per-session
# composites sees the score, including rank and turn movement McNemar is
# blind to. They agree here. Where they disagree, say so.
#
# Every scenario, by a lot: buying +0.4500, browsing +0.6750, override
# +0.5000, boundary +0.8000. This one flag is worth 3.5x the whole of Phases
# 0-14 combined (+0.1372 over the baseline), which is not a compliment to
# this phase so much as a statement about what the previous fourteen were
# optimising. They made the agent better at answering a question it was never
# told the answer to.
#
# NO PLACEBO ARM, and the Phase 14 reason for having one genuinely does not
# apply. That phase REORDERED a list, so any disturbance could shuffle a
# target into the top 10 and a null had to be built. This stage does not
# reorder anything -- it changes what the shopper says. A "meaningless
# question" control is a question that harvests nothing, and that is the OFF
# arm. OFF is the null.
#
# WHICH QUESTION, WHICH IS THE PART THIS BENCHMARK CANNOT SETTLE
#
#   shipped (C, one open question)  TS 0.722979
#   unbounded open question (A)     TS 0.722979   vs shipped: IDENTICAL
#   no open question at all (B)     TS 0.686029   vs shipped: -0.0370, 9/0,
#                                                 p = 0.0039  established
#   wildcard every turn (D)         TS 0.719419
#   first empty slot                TS 0.704984
#   harness-fitted                  TS 0.726837   vs shipped: -2 hits, 0/2,
#                                                 p = 0.5000; score +0.003858,
#                                                 CI [-0.0088, +0.0132],
#                                                 p = 0.5031. No verdict.
#
# Read the A row first: capping the open question at one per session costs
# nothing measurable. It is not, however, a no-op -- A asks the wildcard 66
# times across those 42 sessions and C asks it 42 times, so the cap really
# does remove 24 repeat questions; they simply changed no session's outcome.
# See MAX_OPEN_QUESTIONS. Then read B: removing the open question entirely
# costs an established 0.0370. So it is no longer load-bearing and it is not
# free either, and both facts are stated rather than one of them.
#
# The shipped policy is ahead of the wildcard-every-turn arm and the naive
# first-empty-slot arm, and neither margin is established -- stated plainly,
# this benchmark does not show that the clever question beats the dumb one.
#
# It cannot show it, either, and that is the load-bearing sentence: `"other"`
# matches a strict superset of what any specific question matches (see the
# module docstring), so the harness structurally favours the least specific
# policy. A no-verdict result against a control the rules favour is not
# evidence for that control. What the scorer earns on the number that IS
# established -- MRR 0.582419 vs the wildcard's 0.575397 at an equal or
# better hit rate -- is that the constraints arrive in a more useful ORDER: a
# colour or a material becomes a SLOT the ranker scores on, while an
# arbitrary feature string becomes free-text evidence only.
#
# THE LAST ROW IS A PRIZE THIS PHASE DECLINES, and it is recorded here so the
# choice is auditable rather than quietly forgone. The evaluator's
# ``classify_constraint`` has no branch that returns "brand" and none that
# returns "category", so those two questions can never be answered on this
# harness: the scorer asks brand on 200 turns, every one is declined, and
# CP 15.7 closes it having cost one turn per session. Deleting them from
# ``SCORABLE_ATTRIBUTES`` is worth +0.003858 TS on a 95% CI of
# [-0.0088, +0.0132] that STRADDLES ZERO, and it LOSES two sessions on hits
# (0/2 discordant, p = 0.5000) -- no verdict on either test, from a
# comparison the tool only started making against the shipped arm after the
# D Phase 15 review pointed out it had only ever been made against OFF. The
# figures here read +0.0144 and 2/1 until then, which were against the wrong
# baseline. It remains a policy fitted to the branch list of a simulator's
# classifier. Brand is the single most
# discriminating thing this catalog knows and asking a real shopper about it
# is a good question. Not taken.
USE_CLARIFICATION = True

# CP 15.3. The enum from docs/agent_api_contract.json turn_response, plus the
# implicit ``None``. Duplicated here deliberately rather than parsed from the
# JSON at import: the contract file is the organizer's and is read-only to us,
# so a mismatch must be a TEST failure with a name (tests/test_clarify.py reads
# the JSON and compares), not a silent runtime dependency on a file we do not
# own being present and parseable inside a scored turn.
ALLOWED_ATTRIBUTES = (
    "category", "material", "color", "size", "style", "brand", "budget",
    "feature", "use_case", "other",
)

# The open question. Legal, and the only value that invites information the
# slot vocabulary has no name for. See the module docstring for what it is in
# THIS evaluator, which is a different and much stronger thing.
WILDCARD = "other"

# CP 15.5. The attributes the value scorer can actually score: an answer
# becomes a slot (``state.SLOT_CARDINALITY``) and is checkable against a
# ``catalog_meta`` column. Asserted against both in tests/test_clarify.py, so
# adding a slot without deciding whether it is askable breaks loudly.
SCORABLE_ATTRIBUTES = (
    "category", "color", "material", "brand", "size", "budget",
)

# CP 15.5, the second tier: askable but NOT value-scorable.
#
# THIS TIER EXISTS BECAUSE THE FIRST VERSION OF THIS FILE GOT ITS OWN
# ARGUMENT WRONG. It excluded these three on the grounds that an answer
# "cannot become a slot, cannot be checked against the catalog, and cannot
# change the ranking, so asking for one buys a sentence and nothing else".
# The first two clauses are true. The third is FALSE, and Phase 14 is the
# proof: ``reranker.PoolTermScorer`` ranks the window on the shopper's
# still-active free-text evidence, which is exactly where an unslotted answer
# lands. A "reinforced toe" answer changes the ranking through the reranker
# rather than through ``score_candidate``; it was never inert.
#
# The cost of that error was concrete. With only six askable attributes the
# open question was the ONLY way to reach the majority of what this shopper
# has to say -- ``local_evaluator.classify_constraint`` falls back to
# "feature" for anything it cannot bucket, so most constraints are
# feature-class -- which is what made the wildcard load-bearing and drew the
# B Phase 15 blocker. Bounding the wildcard without this tier costs 0.0466
# TS; with it, see MAX_OPEN_QUESTIONS.
#
# No value score, because there is no catalog column to compute one from.
# They are tried in a fixed order after every scorable attribute and before
# the open question: a specific question is always preferable to an open one,
# and this is the cheapest way to make that preference reachable.
EVIDENCE_ATTRIBUTES = (
    "feature", "use_case", "style",
)

# How deep into the ranked list a question's value is judged. The question
# "would knowing this split the answer" is about the candidates the shopper is
# about to be shown and the ones just behind them -- not about the 300-row
# retrieval pool, most of which the ranker has already rejected.
#
# Deliberately not RERANK_TOP_N. It happens to be the same number today, and
# reading it from `reranker` would silently retune this layer whenever that
# window moved, which is the coupling `reranker.rerank` was corrected for in
# Phase 14 (the top_n default bound at import).
ASK_POOL_DEPTH = 50

# Quantile buckets for the one continuous attribute (budget). Four, because
# the question is "roughly where in this pool's price range are you", and a
# finer grid would report spread that no shopper's answer could resolve.
PRICE_BUCKETS = 4

# CP 15.2. A specific question must beat this to be asked in preference to the
# open one. Value is a product of three shares (see ``attribute_value``), so
# the scale is 0..1 and this is "the answer would split a tenth of the visible
# pool, weighted by whether we could trust the check".
#
# NOT tuned on the public set. It is set to exclude the degenerate cases the
# formula already reports as near-zero -- an attribute nothing in the pool
# declares, or one every candidate agrees on -- and the sweep in
# tools/phase15_clarification.py exists to show the choice is not on a cliff,
# not to pick the peak.
#
#   floor    TS         vs OFF
#   0.00   0.701142   +0.457212   <- ask a value-scored question on any signal
#   0.05   0.717937   +0.474007
#   0.10   0.722979   +0.479049   <- committed, chosen a priori
#   0.20   0.704196   +0.460266
#   0.40   0.677156   +0.433226
#
# Not a cliff, and the a-priori value happens to top the table. That is luck
# and is recorded as luck: 0.05 is 0.005 away, which is half a session, and
# the choice was made before this sweep existed. Nobody should read the peak
# as evidence the value was tuned well.
#
# The shape changed direction when EVIDENCE_ATTRIBUTES landed. Before it, a
# higher floor scored better, because falling through the floor meant
# reaching the wildcard and the harness rewards the wildcard. Now falling
# through means reaching a specific evidence question, and the curve has an
# interior maximum -- which is what a threshold with a real trade-off on both
# sides looks like, rather than a proxy for "how often do we say other".
ASK_VALUE_FLOOR = 0.10


# CP 15.2. How many times ONE session may fall back to the open question.
#
# THIS IS THE B PHASE 15 BLOCKER, AND THE NUMBER IS THE ANSWER TO IT. The
# first version had no budget: the wildcard was the fallback on every turn no
# specific attribute cleared the floor, and a productive one could be re-asked
# indefinitely. On a harness whose ``customer_reply`` treats "other" as a
# wildcard matching a strict superset of every specific attribute, that is
# farming the simulator -- and the review was right that the evidence showed
# it, because removing the open question cost two thirds of the phase's gain.
#
# 1, and the measurement says the cap is FREE. The arms are in
# tools/phase15_clarification.py:
#
#   policy                    TS      other/session   sessions using other
#   A  unbounded           0.722979       0.33             42/200
#   B  no open question    0.686029       0.00              0/200
#   C  at most one  <-     0.722979       0.21             42/200
#   D  wildcard every turn 0.719419       3.72            200/200
#
# THE CAP IS SCORE-FREE, NOT BEHAVIOUR-FREE, and the first version of this
# comment got that wrong. It said "no session ever wanted a second open
# question", which the tool's own table contradicts thirty characters away:
# 0.33 per session over 42 sessions is 66 asks, not 42 (B Phase 15 review,
# E1). Measured directly:
#
#   A  66 wildcard asks over 42 sessions; 24 of those sessions asked twice
#      or more. 10 of the 24 go on to HIT, so the repeats are not confined
#      to sessions that were lost anyway.
#   C  42 asks over the same 42 sessions; exactly one each, by construction.
#
# And yet A and C score identically -- 0/0 discordant, 0 of 200 sessions
# moved, every metric equal to six decimals. So the cap removes 24 repeat
# questions that changed no session's outcome. It is a real behavioural
# change with no measurable price, which is a better thing to be able to say
# than "nothing was happening": the thing it forbids -- re-asking a
# productive wildcard until it dries up -- is farming a harness whose
# ``customer_reply`` treats "other" as matching a strict superset of every
# specific attribute, and it WAS happening on 24 sessions.
#
# B is the strictly-generic policy: no open question at any point. It costs
# an ESTABLISHED 0.0370 (9/0 discordant, p = 0.0039; paired permutation over
# per-session composites p = 0.0017). That is a real price, not noise, and it
# is stated as one. Anyone who wants zero wildcard dependence sets this to 0
# and pays it knowingly.
#
# WHAT MADE THE CAP CHEAP was not the cap. Before ``EVIDENCE_ATTRIBUTES``
# existed, bounding the wildcard cost 0.0466 and removing it cost 0.266,
# because the six catalog-checkable attributes could not reach the majority
# of what this shopper has to say. Adding the three evidence questions moved
# "other" from load-bearing to optional. The B Phase 15 blocker was right
# about the dependence and the fix was to give the policy better specific
# questions, not to force it to do without.
MAX_OPEN_QUESTIONS = 1


class ClarificationLedger:
    """What has been asked in one session, and what is exhausted.

    AGENT-OWNED, NOT SESSION STATE, and the distinction is the single-writer
    invariant rather than bookkeeping taste. ``SessionState`` holds what the
    SHOPPER has told us -- slots, evidence, provenance -- and only
    ``state.update_state`` may write it (``starter/contracts.py``). What the
    agent chose to ask is the agent's own conversation history, it is keyed on
    our question rather than on their answer, and storing it in ``slots`` would
    put a non-constraint in the constraint bag. So it lives beside
    ``Agent._states``, is created by ``Agent.reset``, and dies with the
    session.

    CP 15.6 / CP 15.7 are both enforced here:

      ``asked``     every attribute asked, in order. Diagnostics.
      ``closed``    attributes that answered EMPTY once. Never asked again.
      ``pending``   the attribute the last question asked, awaiting its reply.

    A question is not closed for having been asked -- an open question that
    keeps yielding constraints deserves to be asked again ("and what else?").
    It is closed the first time it yields NOTHING. That is what bounds the
    loop, and it bounds it twice over: an attribute can close only once, so
    after at most ``len(ALLOWED_ATTRIBUTES)`` unproductive questions there is
    nothing left to ask and ``choose`` returns ``None`` for the rest of the
    session. The evaluator's 10-turn cap is the looser of the two bounds.
    """

    __slots__ = ("asked", "closed", "pending", "wildcard_uses")

    def __init__(self) -> None:
        self.asked: list[str] = []
        self.closed: set[str] = set()
        self.pending: str | None = None
        # CP 15.2, the open question's per-session budget. Counted rather than
        # flagged because ``MAX_OPEN_QUESTIONS`` is a number and 0 / 1 / many
        # are all policies someone might want; see that constant.
        self.wildcard_uses: int = 0

    def observe(self, user_message: object) -> None:
        """CP 15.7 -- read the reply to the last question we asked.

        A reply that declines to add information closes the attribute it
        declined. ``state.is_non_answer`` is the test, and it is reused rather
        than rewritten for two reasons: it already recognises both shapes this
        evaluator produces ("I don't have a preference for X; please use your
        judgment", "I don't have an additional preference for X") through
        GENERIC phrasing rather than through its exact strings, and a second
        implementation of "the customer declined" would be the drift shape
        this repo keeps finding (D-N2).

        Message-based on purpose. The obvious alternative -- "close the
        attribute if session state did not grow" -- reads correct and couples
        this layer to ``agent.USE_STATE``: with the state manager ablated OFF
        nothing ever grows, so every question would close after one ask and the
        clarification arm of an unrelated ablation would quietly be measuring
        a different policy.
        """
        if self.pending and is_non_answer(user_message):
            self.closed.add(self.pending)
        self.pending = None

    def record(self, attribute: str | None) -> None:
        if attribute is None:
            return
        self.asked.append(attribute)
        self.pending = attribute
        if attribute == WILDCARD:
            self.wildcard_uses += 1

    def open_attributes(self) -> tuple[str, ...]:
        return tuple(name for name in ALLOWED_ATTRIBUTES
                     if name not in self.closed)


def _declared(row: dict[str, Any], attribute: str,
              price_bucket: dict[float, int]) -> set[str]:
    """What the catalog asserts about one candidate for one attribute.

    An empty set means UNKNOWN -- the catalog says nothing -- which is the
    same reading ``ranking`` gives it (CP 6.4) and is why an attribute nobody
    declares scores zero here rather than scoring as unanimous agreement.
    """
    if attribute == "category":
        value = row.get("cats")
    elif attribute == "brand":
        store = row.get("store")
        return {store} if isinstance(store, str) and store else set()
    elif attribute == "size":
        value = row.get("sizes")
    elif attribute == "budget":
        price = row.get("price")
        if not isinstance(price, (int, float)) or price != price:
            return set()
        return {f"q{price_bucket.get(float(price), 0)}"}
    else:
        value = row.get(attribute)
    return set(value) if isinstance(value, (set, frozenset)) else set()


def _price_buckets(rows: Iterable[dict[str, Any]]) -> dict[float, int]:
    """Pool-relative quantile index for every declared price.

    Pool-relative rather than absolute because the question is whether a
    budget answer would SPLIT THIS POOL. Fixed price bands would report a
    pool of $12 socks as perfectly agreed and a pool spanning $80 to $400 as
    perfectly agreed, for opposite reasons and with the same number.
    """
    prices = sorted(
        float(row["price"]) for row in rows
        if isinstance(row.get("price"), (int, float))
        and row["price"] == row["price"]
    )
    if not prices:
        return {}
    edges = [
        statistics.quantiles(prices, n=PRICE_BUCKETS)[index]
        for index in range(PRICE_BUCKETS - 1)
    ] if len(prices) >= PRICE_BUCKETS else []
    buckets: dict[float, int] = {}
    for price in prices:
        index = 0
        for edge in edges:
            if price > edge:
                index += 1
        buckets[price] = index
    return buckets


def attribute_value(
    attribute: str,
    rows: Sequence[dict[str, Any]],
    price_bucket: dict[float, int],
    reliabilities: dict[str, float] | None,
    known: bool,
) -> float:
    """CP 15.5 -- what one question is worth, in ``[0, 1]``.

    Three factors, multiplied, each a share and each answering a different
    reason a question can be worthless:

      ANSWERABLE   the share of the visible pool that declares this attribute
                   at all. A question about something the catalog is silent on
                   cannot be acted upon however willing the shopper is; this
                   is the local, per-pool form of the coverage argument
                   ``reliability.py`` makes catalog-wide.

      DISCRIMINATING   ``1 - dominance``, where dominance is the share of the
                   declaring candidates that carry the single most common
                   value. If every candidate in the window is black, "what
                   colour?" separates nothing -- the pool has already answered
                   it. This is the factor an "ask about the first empty slot"
                   policy has no way to see, and it is why that policy
                   measured 2.2x worse than the wildcard.

      TRUSTWORTHY  Phase 11's Match Reliability. Answerable measures whether
                   THIS pool declares the attribute; reliability measures
                   whether a verdict on it is worth anything catalog-wide.
                   They come apart exactly where it matters: ``size`` is
                   declared by 9.6% of the catalog, so a pool that happens to
                   contain a few sized items looks locally answerable while a
                   size verdict remains nearly worthless.

    ``known`` short-circuits to 0: an attribute already captured in a slot is
    not a question, it is a repeat (CP 15.6). Re-asking it would also be the
    one way this layer could talk over the shopper.
    """
    if known or not rows:
        return 0.0
    declaring = 0
    counts: dict[str, int] = {}
    for row in rows:
        values = _declared(row, attribute, price_bucket)
        if not values:
            continue
        declaring += 1
        for value in values:
            counts[value] = counts.get(value, 0) + 1
    if not declaring:
        return 0.0
    answerable = declaring / len(rows)
    dominance = max(counts.values()) / declaring
    return (answerable * (1.0 - dominance)
            * reliability_of(reliabilities, attribute))


def _filled_slots(context: Context) -> set[str]:
    slots = getattr(getattr(context, "state", None), "slots", None)
    if not isinstance(slots, dict):
        return set()
    return {
        name for name, entry in slots.items()
        if isinstance(entry, dict) and entry.get("values")
    }


def rank_attributes(
    context: Context,
    ranked: Sequence[Candidate],
    metadata: dict[str, dict[str, Any]] | None,
    reliabilities: dict[str, float] | None,
) -> list[tuple[float, str]]:
    """CP 15.5 -- every scorable attribute with its value, best first.

    Ties break on ``SCORABLE_ATTRIBUTES`` order, which is fixed, so the
    choice is deterministic on a pool where two attributes are worth exactly
    the same. Sorting on the float alone would leave that to whatever order
    the candidates arrived in.
    """
    metadata = metadata if isinstance(metadata, dict) else {}
    rows = [
        metadata.get(candidate.parent_asin) or {}
        for candidate in list(ranked)[:ASK_POOL_DEPTH]
    ]
    price_bucket = _price_buckets(rows)
    known = _filled_slots(context)
    scored = [
        (attribute_value(attribute, rows, price_bucket, reliabilities,
                         attribute in known),
         index,
         attribute)
        for index, attribute in enumerate(SCORABLE_ATTRIBUTES)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [(value, attribute) for value, _, attribute in scored]


def safe_choose(
    context: Context,
    ledger: ClarificationLedger,
    ranked: Sequence[Candidate],
    metadata: dict[str, dict[str, Any]] | None,
    reliabilities: dict[str, float] | None = None,
) -> str | None:
    """``choose`` that cannot take the turn down. THE CALL SITE.

    Phase 14 learned this and Phase 15 did not: ``build_scorer`` was passed as
    an argument to ``rerank``, so it was evaluated outside every fallback the
    reranker advertised, and the review's fix was ``safe_build_scorer``. One
    phase later ``clarify.choose`` shipped with exactly the same shape --
    called bare from ``agent.respond``, downstream of a metadata dict whose
    shape it trusts -- and a raise there does not produce a bad question, it
    produces an exception that ``evaluate`` swallows into an empty response
    and scores as zero for the whole turn (D Phase 15 review).

    And ``choose`` DOES raise. The first version of this docstring said it
    was "defensive throughout" and that no known input reached a raise; that
    was written from reading the code rather than from running it, and it is
    false. Of ten adversarial inputs, four raise from the bare function --
    ``ranked=None`` and ``ranked=7`` are TypeErrors, a string or a non-
    Candidate element is an AttributeError. None of them can occur through
    ``agent.respond`` today, because ``RankingResult.ranked`` is always a
    list of ``Candidate``. "Cannot happen through the current caller" is a
    property of the CALLER, and the caller is one edit away from changing.

    The rule this codebase has now demonstrated twice and never written down:
    EVERY UNTRUSTED LAYER DEGRADES RATHER THAN RAISES, where "untrusted"
    means "computed from something another module owns", not "known to be
    buggy". Clarification is strictly optional -- the turn is answerable
    without a question -- so the failure mode is ``None``, which is CP 15.1's
    baseline behaviour and costs nothing but the question.
    """
    try:
        return choose(context, ledger, ranked, metadata, reliabilities)
    except Exception:
        return None


def safe_observe(ledger: ClarificationLedger, user_message: object) -> None:
    """``ledger.observe`` under the same rule, for the same reason.

    It reads a message the harness owns and calls a regex over it. A raise
    here would lose the turn just as completely, and losing the LEDGER only
    costs the CP 15.7 bookkeeping for one turn.
    """
    try:
        ledger.observe(user_message)
    except Exception:
        ledger.pending = None


def choose(
    context: Context,
    ledger: ClarificationLedger,
    ranked: Sequence[Candidate],
    metadata: dict[str, dict[str, Any]] | None,
    reliabilities: dict[str, float] | None = None,
) -> str | None:
    """CP 15.2 / 15.3 / 15.5 / 15.6 -- the attribute to ask about, or ``None``.

    A four-rung ladder, and the ORDER is the whole policy:

      1. a scorable attribute that is still open, not already known, and worth
         at least ``ASK_VALUE_FLOOR`` -- the best one;
      2. otherwise an EVIDENCE attribute that is still open, in fixed order.
         No value score exists for these; what makes them rung 2 rather than
         rung 4 is that a SPECIFIC question is always preferable to an open
         one, and they are the only specific questions left;
      3. otherwise the open question, if the session has any of its
         ``MAX_OPEN_QUESTIONS`` budget left and it is not closed;
      4. otherwise the best scorable attribute that is worth ANYTHING at all,
         even below the floor;
      5. otherwise ``None``.

    RUNGS 2 AND 4 ARE WHAT MAKE RUNG 3 BOUNDABLE, and together they are the
    shape of the B Phase 15 blocker's fix. The first version of this function
    had only the equivalent of rungs 1, 3 and 5: below the floor it reached
    for the wildcard, and if the wildcard was gone it went silent. That made
    the open question load-bearing BY CONSTRUCTION -- removing it did not
    produce a policy that asked specific questions instead, it produced a
    policy that stopped asking (TS 0.413290, which the review correctly read
    as "the wildcard explains a material part of the gain"). With a real
    question below it in the ladder, the budget in rung 3 becomes a knob
    rather than a load-bearing beam.

    Rung 5 is the CP 15.2 "do not ask" case. On this harness there is no COST
    to asking -- the shopper answers a question and a recommendation list in
    the same turn (CP 15.4), so a question that fails costs nothing but the
    sentence. A deployment where questions cost patience wants a second gate
    here: turn budget, or the ``strategy.classify_mode`` distinction between
    a shopper who has named specifics and one still exploring, which is built
    and measured (``tools/phase9_mode_accuracy.py``) and deliberately not
    wired in on a benchmark that cannot see the difference.

    Returns a value from ``ALLOWED_ATTRIBUTES`` or ``None``, always: the
    return is a member check away from the contract enum, so a scorer bug
    cannot put an out-of-enum string on the wire (CP 15.3).
    """
    best_value, best_attribute = 0.0, None
    for value, attribute in rank_attributes(context, ranked, metadata,
                                            reliabilities):
        if attribute in ledger.closed:
            continue
        best_value, best_attribute = value, attribute
        break
    unscored = next((attribute for attribute in EVIDENCE_ATTRIBUTES
                     if attribute not in ledger.closed), None)
    if best_attribute is not None and best_value >= ASK_VALUE_FLOOR:
        chosen = best_attribute
    elif unscored is not None:
        chosen = unscored
    elif (ledger.wildcard_uses < MAX_OPEN_QUESTIONS
            and WILDCARD not in ledger.closed):
        chosen = WILDCARD
    elif best_attribute is not None and best_value > 0.0:
        chosen = best_attribute
    else:
        return None
    return chosen if chosen in ALLOWED_ATTRIBUTES else None
