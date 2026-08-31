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

Over the 200 live sessions (719 questions on 874 turns, 155 turns silent):

    other      301   41.9%      color      104   14.5%
    brand      200   27.8%      budget       2    0.3%
    material   112   15.6%      category / size, never

``category`` is never asked because the opening message always fills it, and
``size`` never because 9.6% catalog coverage keeps its value under the floor
-- both are the scorer declining a question rather than a gap in it. The
longest session asks 7 questions and none exhausts the attribute list, so
CP 15.7's bound is never the thing that stops the loop; the shopper running
out of preferences is.

WHY THE SCORER ONLY SCORES SIX ATTRIBUTES

The contract allows ten (CP 15.3). Six of them -- category, color, material,
brand, size, budget -- are exactly the six ``state.SLOT_CARDINALITY`` can
store and exactly the six ``catalog_meta`` carries a column for. That is not a
coincidence to work around; it is the boundary of what an answer can DO. An
answer about ``style`` cannot become a slot, cannot be checked against the
catalog, and cannot change the ranking, so asking for one buys a sentence and
nothing else. The scorer therefore scores what the pipeline can act on, and
``"other"`` is the open question it falls back to -- "is there anything else
that matters?" -- which is a real thing to ask a shopper and is the only legal
way to invite information the slot vocabulary has no name for.
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
#   ON    HR 0.8450  MRR 0.582419  MTTC 4.525  TS 0.726726   +0.482796
#
#   ON vs OFF   +112   114/2 discordant   p = 0.0000   established
#
# Every scenario, by a lot: buying +0.4500, browsing +0.6875, override
# +0.4667, boundary +0.7000. This one flag is worth 3.5x the whole of Phases
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
#   shipped (value-scored)   TS 0.726726
#   wildcard every turn      TS 0.719419     shipped vs it: +7, 9/2, p = 0.0654
#   first empty slot         TS 0.704984     shipped vs it: +0, 2/2, p = 1.0000
#   scored, no open question TS 0.413290
#   harness-fitted           TS 0.741172     vs shipped:    +1, 2/1, p = 1.0000
#
# Read those in the order they matter. The last row is the big one: with the
# open question removed the phase loses two thirds of its gain, so most of
# what clarification buys is ASKING AT ALL, and the value scorer is choosing
# the ORDER rather than the content. The shipped policy is nominally ahead of
# the wildcard and the naive policy, and neither margin is established --
# stated plainly, this benchmark does not show that the clever question beats
# the dumb one.
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
# ``SCORABLE_ATTRIBUTES`` is worth +0.0144 TS -- and it is worth ONE session
# (2/1 discordant, p = 1.0000, no verdict), while being a policy fitted to
# the branch list of a simulator's classifier. Brand is the single most
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
#   0.00   0.692042   +0.448112   <- ask a specific question on any signal
#   0.05   0.723034   +0.479104
#   0.10   0.726726   +0.482796   <- committed, chosen a priori
#   0.20   0.731636   +0.487706
#   0.40   0.733548   +0.489618
#
# Not a cliff anywhere above 0.05, which is what the sweep was for. It is
# also not the peak: 0.40 measures +0.0068 better, which is roughly one
# session on 200, and taking it would mean reading a maximum off a table
# produced by the same 200 sessions the score is then reported on. Same trade
# as reranker.RERANK_TOP_N and refused the same way. The direction is worth
# noting for whoever revisits it with a held-out split: a HIGHER floor asks
# fewer specific questions, so what the table is saying is that the wildcard
# is a strong default here -- which is exactly what the harness's own
# superset rule predicts, and exactly why it is not evidence.
ASK_VALUE_FLOOR = 0.10


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

    __slots__ = ("asked", "closed", "pending")

    def __init__(self) -> None:
        self.asked: list[str] = []
        self.closed: set[str] = set()
        self.pending: str | None = None

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


def choose(
    context: Context,
    ledger: ClarificationLedger,
    ranked: Sequence[Candidate],
    metadata: dict[str, dict[str, Any]] | None,
    reliabilities: dict[str, float] | None = None,
) -> str | None:
    """CP 15.2 / 15.3 / 15.5 / 15.6 -- the attribute to ask about, or ``None``.

    The decision, in order:

      1. a scorable attribute that is still open, not already known, and worth
         more than ``ASK_VALUE_FLOOR`` -- the best one;
      2. otherwise the open question, if it is still open;
      3. otherwise ``None`` -- everything has been asked and answered empty,
         and asking again would be noise.

    (3) is the CP 15.2 "do not ask" case and it is the only one this dataset
    reaches, because on this harness there is no COST to asking: the shopper
    answers a question and a recommendation list in the same turn (CP 15.4),
    so a question that fails costs nothing but the sentence. A deployment
    where questions cost patience wants a second gate here -- turn budget,
    or the ``strategy.classify_mode`` distinction between a shopper who has
    named specifics and one still exploring, which is built and measured
    (``tools/phase9_mode_accuracy.py``) and deliberately not wired in on a
    benchmark that cannot see the difference.

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
    if best_attribute is not None and best_value >= ASK_VALUE_FLOOR:
        chosen = best_attribute
    elif WILDCARD not in ledger.closed:
        chosen = WILDCARD
    else:
        return None
    return chosen if chosen in ALLOWED_ATTRIBUTES else None
