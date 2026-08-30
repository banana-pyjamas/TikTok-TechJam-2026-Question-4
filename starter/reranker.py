"""Semantic reranking (Phase 14).

Phase 13 measured where the score actually is. Retrieval delivered the target
into the pool on a scoring-eligible turn 149/200 times and only 42 of those
converted; 176 landed at final rank 11-50 and another 149 at 51-300. Getting
the target into the pool at the BOTTOM was worth +0.0000 TS, getting it to the
HEAD +0.2419. The problem is ORDER, and this file is the layer that reorders.

(Those are the pre-Phase-14 figures, quoted as the motivation they were. With
this stage ON the same gate now reports 53/149 converting and 106 rather than
176 at rank 11-50 -- which is this file having done its job. The retrieval
half of Phase 13's numbers is unchanged; see the note in retrieval.py about
which of them move when the ranker moves.)

WHAT THIS PHASE ACTUALLY DELIVERS, STATED UP FRONT

Two things, and they are not equally strong:

  the CONTRACT   ``rerank`` is scorer-agnostic and enforces CP 14.1 - 14.5
                 whatever is plugged into it: top-N input only, output IDs a
                 subset of input IDs, and three independent fallbacks to the
                 order ranking already produced. This is complete and tested.

  a SCORER       ``PoolTermScorer``, which is LEXICAL, not semantic. Naming it
                 accurately is not pedantry here: 7c52e87 shipped a TF-IDF
                 arm labelled as a proxy for dense retrieval and the review
                 correctly rejected the label. The SEMANTIC scorer is
                 ``load_encoder_scorer``, and on this machine it returns
                 ``None`` -- nothing is vendored, network may be off at
                 scoring time, and the submission is certified on python3.9.
                 So CP 14.5 is not a hypothetical branch here; it is the live
                 path for the only genuinely semantic scorer this repo has.

That asymmetry is the phase. The fallback machinery is built so that an
encoder can be dropped in later without touching the pipeline, and the
measured decision (CP 14.6) is made on the scorer that can actually run.

WHY A SEPARATE STAGE RATHER THAN ANOTHER TERM IN ``ranking.score_candidate``

Because a reranker is allowed to be untrusted. Every term in ranking is a
deterministic function this repo owns; a reranker may be a model that is
slow, offline, or wrong, and the contract has to survive all three without
the turn degrading. Folding it into the additive score would mean a bad
scorer silently moves candidates with no fallback to fall back TO.

DETERMINISM, HONESTLY

``PoolTermScorer`` is pure and deterministic. The TIMEOUT is not: which path
runs depends on wall clock, so a slow machine could in principle take the
fallback where a fast one did not. Two things bound that. The budget is
cooperative (the deadline is handed to the scorer, and a late result is
discarded afterwards) rather than preemptive -- stdlib has no way to kill a
running call that does not involve signals or unkillable threads, and
pretending otherwise would be the bigger lie. And the shipped scorer's work
is bounded by ``RERANK_TOP_N`` candidates of at most 40 indexed terms, which
is microseconds; ``tools/phase14_reranker.py`` reports the observed timeout
count, and it must be 0 for the ON arm's numbers to mean anything.
"""

from __future__ import annotations

import math
import time
from typing import Any, Iterable, Sequence

from starter.contracts import Candidate, Context, RankingResult
from starter.text import terms

# Ablation flag (CP 14.6). ON, measured -- `python3 -m tools.phase14_reranker`
# reproduces all of this.
#
#   OFF       HR 0.2100  MRR 0.126192  MTTC 9.030  TS 0.182258
#   ON        HR 0.2650  MRR 0.141468  MTTC 8.560  TS 0.223740   +0.041482
#   PLACEBO   HR 0.1900  MRR 0.065857  MTTC 9.380  TS 0.147157   -0.035101
#
#   ON vs OFF        +11   12/1 discordant   p = 0.0034   established
#   PLACEBO vs OFF    -4   9/13 discordant   p = 0.5235   no verdict
#   ON vs PLACEBO    +15   19/4 discordant   p = 0.0026   established
#
# WHY THE PLACEBO ARM EXISTS, AND WHY IT IS LOAD-BEARING. The evaluator stops
# a session at its first hit, so ANY reordering of the window reshuffles which
# sessions happen to land a target in the top 10, and on 200 sessions that can
# look like a win. The placebo is this same scorer ranking on meaningless
# terms -- same pool vocabulary, same in-pool IDF, same cardinality per turn,
# drawn deterministically instead of from what the shopper said. It does not
# gain; it loses 0.035 TS. So reordering as such is not free money, and the
# gain is attributable to the shopper's words rather than to the disturbance.
#
# The movement table agrees with the score, which is the check that matters:
# over the turns where the target is inside the window its mean final rank
# goes 21.6 -> 18.3, it moves up on 44 turns and down on 21, and it crosses
# INTO the top 10 12 times against 1 push-out -- exactly the 12/1 the
# session-level McNemar reports. An earlier version of this stage scored
# +0.024 while moving the target DOWN on average; that contradiction was a
# bug (see build_scorer), not a subtlety, and it is why this table is printed.
#
# THE WINDOW SWEEP, which is also where an earlier version of this comment was
# wrong. top_n=10 is the control: the window is then the response itself, so
# the stage can reorder only what was already being shown.
#
#   top_n     TS        vs OFF     McNemar vs OFF
#      10   0.182714   +0.000456   0/0    p = 1.0000  <- control, exactly null
#      20   0.216959   +0.034701   9/0    p = 0.0039
#      50   0.223740   +0.041482  12/1    p = 0.0034  <- committed
#     100   0.234563   +0.052305  18/3    p = 0.0015
#     200   0.213411   +0.031153  19/9    p = 0.0872
#
# Established at 20, 50 and 100; null at the control; degrading at 200 as the
# window admits candidates the constraint ranker judged much worse. Five
# comparisons, so at a Bonferroni alpha/5 = 0.01 the middle three still clear.
# What carries it beyond any single p-value is that the control is null and
# the direction is consistent across the plateau -- the same standard the
# category route is held to in retrieval.py.
USE_SEMANTIC_RERANK = True

# CP 14.1. The reranker sees the top N of the ranked list and nothing else.
#
# 50, which is NOT the best-scoring row above. 100 measures +0.011 TS better,
# and taking it would mean selecting a maximum on the same 200 sessions the
# score is then reported on. 50 was chosen a priori, before any sweep ran,
# from Phase 13's measured rank distribution -- 176 of the 367 eligible-turn
# pool hits sit at final rank 11-50, the largest single band a reordering
# layer can reach -- and it lands mid-plateau with the fewest sessions lost
# (12 gained, 1 lost). An a-priori value inside a broad established plateau
# generalizes better than the peak next to the region where the effect decays.
#
# The +0.052 at 100 is left on the table deliberately and recorded here so the
# choice is auditable rather than quietly optimal.
#
# Not 300. The window must stay a strict subset of the pool for CP 14.1 to
# mean anything: a stage that reorders everything is not a reranker with a
# fallback, it is the ranker.
RERANK_TOP_N = 50

# CP 14.4. Wall-clock budget for one scorer call, milliseconds. Generous by
# three orders of magnitude for the shipped scorer, because its job is to
# bound a MODEL that is not there yet, not to police arithmetic.
RERANK_BUDGET_MS = 150.0

# Context.derived key carrying this stage's diagnostics. A generic container,
# not a new frozen field and not a new key in RankingResult.diagnostics --
# that set is frozen and guarded (tests/test_ranking.py FROZEN_DIAGNOSTIC_KEYS).
RERANK_KEY = "rerank"

# Every outcome ``rerank`` can report. Enumerated so the ablation tool can
# assert it has seen the fallbacks fire rather than assuming they would.
OUTCOMES = (
    "applied",      # the scorer's order was used
    "identity",     # the scorer returned the order it was given
    "offline",      # CP 14.5 -- no scorer available
    "malformed",    # CP 14.3 -- output was not a usable list of ids
    "error",        # CP 14.3 -- the scorer raised
    "timeout",      # CP 14.4 -- the scorer overran its budget
    "empty",        # nothing to rerank
)


def _blank_diagnostics(scorer_name: str | None = None) -> dict[str, Any]:
    return {
        "scorer": scorer_name,
        "outcome": "empty",
        "considered": 0,
        "returned": 0,
        "invented": 0,
        "dropped": 0,
        "moved": 0,
        "elapsed_ms": 0.0,
    }


def load_encoder_scorer(vendor_root: str | None = None) -> None:
    """The SEMANTIC scorer. Returns ``None`` unless a model is vendored.

    CP 14.5 lives here, and it is the live path rather than a defensive
    branch: no encoder is vendored in this repository, ``docs/
    submission_rules.md`` notes the organizer "may disable network access"
    for final scoring, and the submission is certified on the bare system
    python3.9 that current torch builds do not support. Phase 13 recorded
    those constraints in detail beside ``retrieval.DEFAULT_ROUTES``.

    It deliberately does NOT attempt an import or a download. A scorer that
    tries ``import torch`` inside a turn would make the shipped path depend on
    what happens to be installed on the scoring machine -- exactly the claim
    Phase 13 was corrected for making. When a model is vendored, this function
    is the one place that changes, and ``rerank`` needs no edit.

    ``vendor_root`` exists so a test can point it somewhere; nothing ships
    that reads it.
    """
    return None


class PoolTermScorer:
    """LEXICAL reranking on pool-scoped term rarity. Not semantic.

    Scores each candidate by how many of the shopper's OWN still-active free
    text words it uses, weighted by how rare those words are IN THIS POOL.

    Two reasons this is the right shippable scorer rather than an arbitrary
    one:

    it reads a signal ranking does not
        ``ranking.score_candidate`` scores SLOTS. ``SessionState.evidence``
        holds the free text that was never promoted to a slot -- "gift for my
        dad", "something for hiking" -- and nothing in the scoring path has
        ever read it. Reranking on it is additive information, not a
        reweighting of what the constraint terms already saw.

    pool-scoped rarity is the discrimination signal
        Phase 13 concluded the misses are discrimination failures among
        products that match the query equally well. A word every candidate in
        the pool uses cannot discriminate between them however rare it is
        catalog-wide; a word four of them use splits the pool. That is
        precisely what ``vocabulary.build_vocabulary`` computes, and Phase 10
        built it, tested it, and left it unread pending a consumer. This is
        the consumer.

    Ordering is by score descending, then by the ORIGINAL rank -- so a
    candidate the scorer cannot separate keeps the position ranking gave it,
    and a zero-evidence turn is exactly the identity permutation.
    """

    name = "pool-term"

    def __init__(self, vocabulary: dict[str, Any] | None,
                 indexed_terms: dict[str, Sequence[str]] | None) -> None:
        vocabulary = vocabulary if isinstance(vocabulary, dict) else {}
        self.pool_size = int(vocabulary.get("pool_size") or 0)
        frequencies = vocabulary.get("terms")
        self.frequencies: dict[str, int] = (
            frequencies if isinstance(frequencies, dict) else {}
        )
        self.indexed_terms = indexed_terms if isinstance(indexed_terms, dict) else {}

    def weight(self, term: str) -> float:
        """In-pool inverse document frequency, 0 for a term the pool lacks.

        A term outside the pool vocabulary scores nothing rather than
        infinity: ``build_vocabulary`` has already dropped the ubiquitous and
        the vanishingly rare (CP 10.3), and a word no candidate uses cannot
        order candidates.
        """
        count = self.frequencies.get(term)
        if not count or self.pool_size <= 0:
            return 0.0
        return math.log(self.pool_size / (1.0 + count))

    def query_terms(self, context: Context) -> set[str]:
        """The words this scorer ranks on: the shopper's own free text.

        A seam, not indirection. ``tools/phase14_reranker.py`` overrides it to
        substitute meaningless terms of the same cardinality drawn from the
        same pool vocabulary, which is the only way to tell "the shopper's
        words carry signal" apart from "any reordering of this window moves
        the score". The placebo lives in the tool and never ships.
        """
        return _evidence_terms(context)

    def order(self, parent_asins: Sequence[str], context: Context,
              deadline: float) -> list[str]:
        """CP 14.1/14.2 input contract: ids in, a permutation of them out.

        Returns the input order unchanged when the shopper has said nothing
        free-form, which is the common case in this dataset and is why the
        ablation below has so little to work with.
        """
        wanted = self.query_terms(context)
        if not wanted:
            return list(parent_asins)
        weights = {term: self.weight(term) for term in wanted}
        if not any(weights.values()):
            return list(parent_asins)

        scored: list[tuple[float, int, str]] = []
        for position, parent_asin in enumerate(parent_asins):
            # Cooperative deadline (see the module docstring): checked between
            # candidates, which is the only preemption point a pure-Python
            # scorer has. Returning the input order here is a no-op that the
            # caller's post-hoc budget check will classify as a timeout.
            if time.monotonic() > deadline:
                return list(parent_asins)
            present = set(self.indexed_terms.get(parent_asin, ()))
            score = sum(weight for term, weight in weights.items()
                        if term in present)
            scored.append((-score, position, parent_asin))
        scored.sort()
        return [parent_asin for _, _, parent_asin in scored]


def _evidence_terms(context: Context) -> set[str]:
    """The shopper's still-active free text, tokenized.

    ``normalized`` rather than ``text``: the state manager writes both, and
    the normalized form is what every other consumer in this repo reads.
    Superseded evidence is excluded -- Phase 3 sets ``status`` to
    ``superseded`` on override, and reranking on a withdrawn preference is
    the stale-evidence resurrection that phase exists to prevent.
    """
    state = getattr(context, "state", None)
    found: set[str] = set()
    for entry in (getattr(state, "evidence", None) or ()):
        if not isinstance(entry, dict) or entry.get("status") != "active":
            continue
        normalized = entry.get("normalized")
        if isinstance(normalized, str):
            found.update(terms(normalized))
    return found


def _valid_ids(proposed: object, allowed: Iterable[str]) -> tuple[list[str], int]:
    """CP 14.2/14.3 -- coerce a scorer's output into ids it was given.

    Returns ``(ids, invented)``. The result is always a deduplicated subset of
    ``allowed``, in the proposed order. Anything else the scorer emitted --
    an ASIN it made up, an integer, a nested list, ``None`` -- is counted and
    discarded. This is enforcement by CONSTRUCTION rather than validation
    after the fact: there is no code path in which a parent_asin the caller
    did not supply can reach the returned list.
    """
    allowed_set = set(allowed)
    ids: list[str] = []
    seen: set[str] = set()
    invented = 0
    if not isinstance(proposed, (list, tuple)):
        return [], 0
    for item in proposed:
        if not isinstance(item, str):
            invented += 1
            continue
        if item in seen:
            continue
        if item not in allowed_set:
            invented += 1
            continue
        seen.add(item)
        ids.append(item)
    return ids, invented


def rerank(
    result: RankingResult,
    context: Context,
    scorer: Any | None,
    top_n: int | None = None,
    budget_ms: float | None = None,
) -> RankingResult:
    """Reorder the head of a ranked list, or return it untouched.

    TOTAL. Every failure mode of the scorer -- absent, slow, raising, or
    returning nonsense -- yields the ranking order it was handed, never a
    partial or empty list. That is the whole point of the stage: the layer
    below it has already produced a defensible answer, so the reranker is
    only ever allowed to IMPROVE on one, not to be required for one.

    CP 14.1  only ``result.ranked[:top_n]`` is shown to the scorer; the tail
             is re-appended in its original order and is never reordered.
    CP 14.2  the output is a permutation of the input ids. Enforced in
             ``_valid_ids``.
    CP 14.3  a scorer that raises, or returns something that is not a usable
             list of ids, falls back.
    CP 14.4  a scorer that overruns ``budget_ms`` falls back, even if what it
             returned was perfectly valid. A late answer is a wrong answer:
             accepting it would make the turn's latency unbounded.
    CP 14.5  ``scorer is None`` falls back. This is the path a missing
             encoder takes.

    Diagnostics go to ``context.derived[RERANK_KEY]`` -- a generic container,
    per the contracts rule; ``RankingResult.diagnostics`` keys are frozen and
    this stage adds none of them. ``rank`` values inside those diagnostics ARE
    renumbered, because a stale rank is worse than no rank.

    ``top_n`` and ``budget_ms`` default to ``None`` and are resolved from the
    module constants HERE, at call time, not in the signature. Writing
    ``top_n: int = RERANK_TOP_N`` binds the value once at import, so a tool
    sweeping ``reranker.RERANK_TOP_N`` would move the ranking depth in
    ``agent.respond`` while this function silently kept reranking the first
    50 -- which is precisely what the first version of the CP 14.6 sweep
    measured, and why its numbers were withdrawn.
    """
    top_n = RERANK_TOP_N if top_n is None else top_n
    budget_ms = RERANK_BUDGET_MS if budget_ms is None else budget_ms
    diagnostics = _blank_diagnostics(getattr(scorer, "name", None))

    def _finish(outcome: str, ranked: list[Candidate]) -> RankingResult:
        diagnostics["outcome"] = outcome
        if isinstance(getattr(context, "derived", None), dict):
            context.derived[RERANK_KEY] = diagnostics
        return RankingResult(ranked=ranked, diagnostics=_renumbered(result, ranked))

    ranked = list(getattr(result, "ranked", None) or ())
    if not ranked:
        return _finish("empty", ranked)
    if scorer is None:
        return _finish("offline", ranked)

    depth = max(0, int(top_n))
    head, tail = ranked[:depth], ranked[depth:]
    if not head:
        return _finish("empty", ranked)
    original = [candidate.parent_asin for candidate in head]
    diagnostics["considered"] = len(original)

    started = time.monotonic()
    deadline = started + max(0.0, float(budget_ms)) / 1000.0
    try:
        proposed = scorer.order(list(original), context, deadline)
    except Exception:
        # Deliberately broad. A reranker is the untrusted layer; a model
        # wrapper can raise anything at all, and the one behaviour that is
        # never acceptable is losing the turn over it.
        diagnostics["elapsed_ms"] = (time.monotonic() - started) * 1000.0
        return _finish("error", ranked)
    diagnostics["elapsed_ms"] = (time.monotonic() - started) * 1000.0

    if time.monotonic() > deadline:
        return _finish("timeout", ranked)

    ids, invented = _valid_ids(proposed, original)
    diagnostics["returned"] = len(ids)
    diagnostics["invented"] = invented
    if not ids:
        return _finish("malformed", ranked)

    by_asin = {candidate.parent_asin: candidate for candidate in head}
    # Anything the scorer omitted keeps its original relative order behind
    # what the scorer did rank. Dropping it instead would let a lazy scorer
    # shorten the recommendation list, which CP 14.2 does not permit.
    missing = [asin for asin in original if asin not in set(ids)]
    diagnostics["dropped"] = len(missing)
    new_head = [by_asin[asin] for asin in ids + missing]
    diagnostics["moved"] = sum(
        1 for before, after in zip(original, [c.parent_asin for c in new_head])
        if before != after
    )
    outcome = "identity" if diagnostics["moved"] == 0 else "applied"
    return _finish(outcome, new_head + tail)


def _renumbered(result: RankingResult, ranked: list[Candidate]) -> dict[str, dict]:
    """The original diagnostics with ``rank`` following the new order.

    Rows are copied, not mutated: the caller's ``RankingResult`` is an input
    and this stage does not own it. A candidate with no diagnostics row keeps
    none -- inventing a row would fake a score nothing computed.
    """
    source = getattr(result, "diagnostics", None) or {}
    updated: dict[str, dict] = {}
    for index, candidate in enumerate(ranked):
        row = source.get(candidate.parent_asin)
        if isinstance(row, dict):
            updated[candidate.parent_asin] = {**row, "rank": index + 1}
    return updated


def build_scorer(
    connection: Any, candidates: list[Candidate], context: Context
) -> Any | None:
    """The scorer for one turn, or ``None`` to take the CP 14.5 path.

    Prefers a vendored semantic encoder and falls back to the lexical
    pool-term scorer. The order matters and is the point: when an encoder
    becomes available, it wins here without any other edit.

    Returns ``None`` when the shopper has volunteered no free text, because
    ``PoolTermScorer`` would then be the identity permutation and building it
    costs an indexed query per turn. This repo has twice deleted a mechanism
    that computed a value nothing used; this is the same rule applied before
    the fact.
    """
    encoder = load_encoder_scorer()
    if encoder is not None:
        return encoder
    if not _evidence_terms(context):
        return None
    # Imported here rather than at module scope: `vocabulary` imports nothing
    # from this module today, but reranking is exactly the kind of layer that
    # grows a back-reference, and a cycle would surface as an import error at
    # agent construction rather than as anything diagnosable.
    from starter.vocabulary import build_vocabulary, pool_terms

    # BOTH over the whole pool, and the second one is a bug fix worth naming.
    #
    # Frequencies must be pool-wide because ``build_vocabulary`` drops terms
    # below MIN_DOCUMENT_FREQUENCY as noise (CP 10.3); over 50 candidates that
    # discards every term used by exactly one of them, which is the most
    # discriminating evidence there is. Over 300 the same floor discards far
    # less, and "how rare is this word among the candidates still in play" is
    # the question Phase 10 says the pool vocabulary answers.
    #
    # The TERM LISTS were originally scoped to ``candidates[:RERANK_TOP_N]``,
    # which was silently wrong: ``candidates`` arrives in RETRIEVAL order and
    # ``rerank`` operates on the RANKED order, so the two heads are different
    # sets. A candidate in the ranked window but outside the retrieval window
    # had no terms, scored 0, and was pushed down for having been ranked well.
    # Worse, how often that happened depended on RERANK_TOP_N, so the window
    # sweep was measuring the coverage gap rather than the window. Pool-wide
    # removes the coupling entirely: the scorer can score anything it is given.
    asins = [candidate.parent_asin for candidate in candidates]
    return PoolTermScorer(
        build_vocabulary(connection, candidates),
        pool_terms(connection, asins),
    )
