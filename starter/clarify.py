"""Clarification -- deciding what to ASK, and whether to ask at all.

Every other stage makes the agent better at answering. This one makes it
participate in the conversation, and on this benchmark it is the single
largest lever there is: the evaluator's customer only discloses a hidden
constraint in reply to a question, so with no question the shopper's real
requirements are never spoken at all.

This file decides whether to ask this turn, which values are legal to emit,
which attribute is worth the most, how not to ask the same thing twice, and
what a "no preference" answer means. Emitting recommendations AND a question in
the same turn is a property of ``agent._to_response``, which builds both from
the same ranked result and cannot emit one without the other.

THE THING TO READ BEFORE ANY COMPARISON OF QUESTION POLICIES

``"other"`` is a wildcard in this evaluator, and it is not a close call.
``local_evaluator.customer_reply`` filters undisclosed constraints with

    attribute == "other" or classify_constraint(value) == attribute

so ``"other"`` matches a strict SUPERSET of what any specific attribute
matches, on every turn, by construction. It cannot score worse than a specific
question and will usually score better, because it harvests constraints of any
class while "color" harvests only colours. That is a fact about the harness's
source, not a finding about question quality, and no amount of measuring here
can separate "the wildcard is the best question" from "the wildcard is the only
question the simulator rewards". The shipped policy bounds its use of the open
question and reports what that costs rather than maximising against the rule.

TWO TIERS OF QUESTION

Nine of the ten contract values are askable, in two tiers:

  SCORABLE   category, color, material, brand, size, budget -- exactly the six
             ``state.SLOT_CARDINALITY`` can store and exactly the six
             ``catalog_meta`` carries a column for. An answer becomes a slot
             and is checkable, so a value can be computed for the question and
             the best one is asked first.

  EVIDENCE   feature, use_case, style -- no slot, no catalog column, and so no
             computable value. Asked in fixed order after the scorable ones.

The second tier matters more than it looks. An unslotted answer still changes
the ranking, through ``reranker.PoolTermScorer``, which ranks the window on the
shopper's still-active free-text evidence -- a different path, not no path.
Without that tier the open question is the only way to reach most of what a
shopper has to say, which is what would make the wildcard load-bearing.
"""

from __future__ import annotations

import statistics
from typing import Any, Iterable, Sequence

from starter.contracts import Candidate, Context
from starter.reliability import reliability_of
from starter.state import is_non_answer

# Ablation flag. OFF is the no-ask baseline: ``ask_attribute`` is ``None`` on
# every turn. ON is worth more than every other stage combined, which is not a
# compliment to this one so much as a statement about what the others were
# optimising -- they made the agent better at answering a question it was never
# told the answer to. `python3 -m tools.phase15_clarification` measures it.
#
# No placebo arm, and the reranker's reason for having one does not apply here.
# That stage reordered a list, so any disturbance could shuffle a target into
# the top 10 and a null had to be built. This one does not reorder anything --
# it changes what the shopper says. A question that harvests nothing is the OFF
# arm, so OFF is the null.
USE_CLARIFICATION = True

# The enum from docs/agent_api_contract.json turn_response, plus the implicit
# ``None``. Duplicated here deliberately rather than parsed from the JSON at
# import: the contract file is the organizer's and read-only to us, so a
# mismatch must be a named TEST failure (tests/test_clarify.py reads the JSON
# and compares), not a silent runtime dependency on a file we do not own being
# present and parseable inside a scored turn.
ALLOWED_ATTRIBUTES = (
    "category", "material", "color", "size", "style", "brand", "budget",
    "feature", "use_case", "other",
)

# The open question. Legal, and the only value that invites information the
# slot vocabulary has no name for. See the module docstring for what it is in
# THIS evaluator, which is a different and much stronger thing.
WILDCARD = "other"

# The attributes the value scorer can score: an answer becomes a slot and is
# checkable against a ``catalog_meta`` column. Asserted against both in
# tests/test_clarify.py, so adding a slot without deciding whether it is
# askable breaks loudly.
SCORABLE_ATTRIBUTES = (
    "color", "material", "size", "budget",
)

# The second tier: askable but not value-scorable, because there is no catalog
# column to compute a value from. Tried in fixed order after every scorable
# attribute and before the open question -- a specific question is always
# preferable to an open one, and this is the cheapest way to make that
# preference reachable.
EVIDENCE_ATTRIBUTES = (
    "feature", "use_case", "style",
)

# How deep into the ranked list a question's value is judged. "Would knowing
# this split the answer" is about the candidates the shopper is about to be
# shown and the ones just behind them, not about the whole retrieval pool, most
# of which the ranker has already rejected.
#
# Deliberately not read from ``reranker.RERANK_TOP_N``. It happens to be a
# similar number, and reading it from there would silently retune this layer
# whenever that window moved.
ASK_POOL_DEPTH = 50

# Quantile buckets for the one continuous attribute (budget). Four, because the
# question is "roughly where in this pool's price range are you", and a finer
# grid would report spread that no shopper's answer could resolve.
PRICE_BUCKETS = 4

# A specific question must beat this to be asked in preference to the open one.
# Value is a product of three shares, so the scale is 0..1 and this is "the
# answer would split a tenth of the visible pool, weighted by whether we could
# trust the check".
#
# Chosen a priori, not tuned: it excludes the degenerate cases the formula
# already reports as near-zero -- an attribute nothing in the pool declares, or
# one every candidate agrees on. The sweep in tools/phase15_clarification.py
# exists to show the choice is not on a cliff, not to pick the peak.
ASK_VALUE_FLOOR = 0.10


# How many times one session may fall back to the open question.
#
# The cap is score-free but NOT behaviour-free: uncapped, a productive wildcard
# gets re-asked until it dries up, which happens on a real fraction of sessions
# and is farming a harness that treats "other" as a superset of every specific
# attribute. Capping removes those repeats at no measurable cost.
#
# Setting this to 0 is the strictly-generic policy -- no open question at any
# point -- and it costs an established amount. That is a real price, not noise.
# Anyone who wants zero wildcard dependence sets this to 0 and pays it
# knowingly. What made the cap cheap was not the cap but ``EVIDENCE_ATTRIBUTES``
# below it: with only six catalog-checkable attributes the open question was
# the only way to reach most of what this shopper has to say.
MAX_OPEN_QUESTIONS = 1


class ClarificationLedger:
    """What has been asked in one session, and what is exhausted.

    AGENT-OWNED, NOT SESSION STATE, and the distinction is the single-writer
    invariant rather than bookkeeping taste. ``SessionState`` holds what the
    SHOPPER has told us and only ``state.update_state`` may write it. What the
    agent chose to ask is the agent's own conversation history, keyed on our
    question rather than on their answer, and storing it in ``slots`` would put
    a non-constraint in the constraint bag. So it lives beside
    ``Agent._states``, is created by ``Agent.reset``, and dies with the session.

      ``asked``     every attribute asked, in order. Diagnostics.
      ``closed``    attributes that answered EMPTY once. Never asked again.
      ``pending``   the attribute the last question asked, awaiting its reply.

    A question is not closed for having been asked -- one that keeps yielding
    constraints deserves to be asked again. It is closed the first time it
    yields NOTHING. That bounds the loop twice over: an attribute can close only
    once, so after at most ``len(ALLOWED_ATTRIBUTES)`` unproductive questions
    there is nothing left to ask and ``choose`` returns ``None`` for the rest of
    the session. The evaluator's 10-turn cap is the looser of the two bounds.
    """

    __slots__ = ("asked", "closed", "pending", "wildcard_uses")

    def __init__(self) -> None:
        self.asked: list[str] = []
        self.closed: set[str] = set()
        self.pending: str | None = None
        # The open question's per-session budget. Counted rather than flagged
        # because 0 / 1 / many are all policies someone might want.
        self.wildcard_uses: int = 0

    def observe(self, user_message: object) -> None:
        """Read the reply to the last question we asked.

        A reply that declines to add information closes the attribute it
        declined. ``state.is_non_answer`` is the test, reused rather than
        rewritten: it already recognises both shapes this evaluator produces
        through generic phrasing rather than exact strings, and a second
        implementation of "the customer declined" would be free to drift from
        the first.

        Message-based on purpose. The obvious alternative -- close the
        attribute if session state did not grow -- reads correct and couples
        this layer to ``agent.USE_STATE``: with the state manager ablated off
        nothing ever grows, so every question would close after one ask and an
        unrelated ablation would quietly measure a different policy.
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

    An empty set means UNKNOWN -- the catalog says nothing -- which is the same
    reading ``ranking`` gives it, and is why an attribute nobody declares scores
    zero here rather than scoring as unanimous agreement.
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

    Pool-relative rather than absolute because the question is whether a budget
    answer would SPLIT THIS POOL. Fixed price bands would report a pool of $12
    socks as perfectly agreed and a pool spanning $80 to $400 as perfectly
    agreed, for opposite reasons and with the same number.
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
    """What one question is worth, in ``[0, 1]``.

    Three factors, multiplied, each a share and each answering a different
    reason a question can be worthless:

      ANSWERABLE   the share of the visible pool that declares this attribute
                   at all. A question about something the catalog is silent on
                   cannot be acted upon however willing the shopper is.

      DISCRIMINATING   ``1 - dominance``, where dominance is the share of the
                   declaring candidates carrying the single most common value.
                   If every candidate in the window is black, "what colour?"
                   separates nothing -- the pool has already answered it. This
                   is the factor an "ask about the first empty slot" policy has
                   no way to see.

      TRUSTWORTHY  Match Reliability. Answerable measures whether THIS pool
                   declares the attribute; reliability measures whether a
                   verdict on it is worth anything catalog-wide. They come
                   apart exactly where it matters: ``size`` is declared by 9.6%
                   of the catalog, so a pool that happens to contain a few
                   sized items looks locally answerable while a size verdict
                   remains nearly worthless.

    ``known`` short-circuits to 0: an attribute already captured in a slot is
    not a question, it is a repeat -- and re-asking it is the one way this layer
    could talk over the shopper.
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
    """Every scorable attribute with its value, best first.

    Ties break on ``SCORABLE_ATTRIBUTES`` order, so the choice is deterministic
    on a pool where two attributes are worth exactly the same. Sorting on the
    float alone would leave that to whatever order the candidates arrived in.
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
    """``choose`` that cannot take the turn down. The call site.

    ``choose`` does raise on malformed input -- ``ranked=None`` or a non-
    ``Candidate`` element are enough. None of that can occur through
    ``agent.respond`` today, because ``RankingResult.ranked`` is always a list
    of ``Candidate``, but "cannot happen through the current caller" is a
    property of the CALLER and the caller is one edit away from changing. A
    raise here does not produce a bad question; it produces an exception that
    ``evaluate`` swallows into an empty response, scoring zero for the turn.

    The rule: every untrusted layer degrades rather than raises, where
    "untrusted" means "computed from something another module owns". Since
    clarification is strictly optional -- the turn is answerable without a
    question -- the failure mode is ``None``, which costs nothing but the
    question.
    """
    try:
        return choose(context, ledger, ranked, metadata, reliabilities)
    except Exception:
        return None


def safe_observe(ledger: ClarificationLedger, user_message: object) -> None:
    """``ledger.observe`` under the same rule, for the same reason.

    It reads a message the harness owns and runs a regex over it. A raise here
    would lose the turn just as completely, while losing the ledger costs only
    one turn of bookkeeping.
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
    """The attribute to ask about, or ``None``.

    A ladder, and the ORDER is the whole policy:

      1. a scorable attribute that is still open, not already known, and worth
         at least ``ASK_VALUE_FLOOR`` -- the best one;
      2. otherwise an evidence attribute that is still open, in fixed order.
         No value score exists for these; what makes them rung 2 rather than
         rung 4 is that a specific question is always preferable to an open one,
         and they are the only specific questions left;
      3. otherwise the open question, if the session has budget left for it;
      4. otherwise the best scorable attribute worth anything at all, even
         below the floor;
      5. otherwise ``None``.

    Rungs 2 and 4 are what make rung 3 boundable. With only rungs 1, 3 and 5,
    falling below the floor reaches for the wildcard and removing the wildcard
    produces a policy that stops asking rather than one that asks specific
    questions -- which makes the open question load-bearing by construction.
    With a real question below it, the budget in rung 3 becomes a knob rather
    than a load-bearing beam.

    Rung 5 is the "do not ask" case. On this harness there is no COST to asking
    -- the shopper answers a question and a recommendation list in the same turn
    -- so a question that fails costs nothing but the sentence. A deployment
    where questions cost patience wants a second gate here: a turn budget, or
    the ``strategy.classify_mode`` distinction between a shopper who has named
    specifics and one still exploring, which is built and measured and
    deliberately not wired in on a benchmark that cannot see the difference.

    Returns a value from ``ALLOWED_ATTRIBUTES`` or ``None``, always: the return
    is a member check away from the contract enum, so a scorer bug cannot put an
    out-of-enum string on the wire.
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
