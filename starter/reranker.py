"""Semantic reranking (Phase 14).

Phase 13 measured where the score actually is. Retrieval delivered the target
into the pool on a scoring-eligible turn 149/200 times and only 42 of those
converted; 176 landed at final rank 11-50 and another 149 at 51-300. Getting
the target into the pool at the BOTTOM was worth +0.0000 TS, getting it to the
HEAD +0.2419. The problem is ORDER, and this file is the layer that reorders.

(Those are the pre-Phase-14 figures, quoted as the motivation they were, and
they are now history twice over. With this stage ON the same gate reported
57/149 converting; with Phase 15 in front of it, 169/194 convert and 89
rather than 176 sit at rank 11-50. Every one of those moves whenever
anything upstream moves, so they are regenerated rather than remembered:
`python3 -m tools.phase13_dense_gate` prints them.

The motivation has NOT survived intact, and pretending otherwise would be the
easy lie. Phase 13 said the problem was ORDER, not retrieval, on the evidence
that 51 of 200 sessions never got the target into the pool and 92 that did
still lost. With clarification supplying real constraints, 6 sessions never
retrieve the target and 25 in-pool sessions still lose. Retrieval headroom
went from +0.2550 recall to +0.0300. This stage still pays -- more than it
did, see the ablation below -- but it is no longer answering the question it
was built to answer, because that question has largely been dissolved by
asking the shopper what they want.)

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

WHAT THE BUDGET DOES NOT COVER, AND WHAT THIS STAGE ACTUALLY COSTS

``RERANK_BUDGET_MS`` is a deadline handed to ``scorer.order``. It cannot
bound work that happens before a scorer exists, and ``build_scorer`` is that
work: two indexed queries over the whole 300-candidate pool. Measured, they
are the expensive half by a factor of 35 --

    scorer.order   mean 0.106 ms/turn      (inside the budget)
    build_scorer   mean 3.422 ms/turn      (outside it)
    stage total    mean 3.528 ms/turn

-- and the phase first quoted only the first line, which understated the
stage's cost by 35x (D Phase 14 review). ``safe_build_scorer`` makes the
build TOTAL, so a failure there degrades to CP 14.5 instead of losing the
turn; making it FAST is the problem of whoever vendors a real encoder, and
they have to bound the load as well as the scoring. Both numbers are printed
by ``tools/phase14_reranker.py`` section 3 rather than remembered here.
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
#   OFF        HR 0.7400  MRR 0.519024  MTTC 5.325  TS 0.639207
#   ON         HR 0.8450  MRR 0.582419  MTTC 4.525  TS 0.726726   +0.087519
#   PLACEBO    5 draws, TS 0.407034 .. 0.448645   highest draw    -0.190562
#
#   ON vs OFF        +21   22/1 discordant   p = 0.0000   established
#   ON vs PLACEBO    established against ALL FIVE draws, p = 0.0000 on every
#                    one (47/1, 48/1, 49/0, 59/0, 51/0). ON beats the best
#                    draw by +0.2781.
#
# PHASE 15 MADE THIS STAGE STRONGER, NOT REDUNDANT, and the numbers above are
# the second measurement rather than the first. At Phase 14 this was +0.0617
# (15/0) and the ON-vs-placebo comparison could not clear 0.05 against the
# luckiest draw. With clarification in front of it the same code measures
# +0.0875 (22/1) and beats every placebo draw at p = 0.0000.
#
# The reason is the whole thesis of this file. PoolTermScorer ranks on the
# shopper's free text, and before Phase 15 the shopper barely produced any:
# the agent never asked, so 90% of turns were the same stuck non-answer. A
# reranker that reads what the shopper said gets better exactly when the
# shopper is given a reason to say something, and the placebo -- which reads
# meaningless words -- gets WORSE, because there is now real signal for it to
# destroy. A layer whose value grows when the layer above it improves is the
# opposite of the usual ablation story, and it is worth stating plainly:
# these two phases are complements, not substitutes.
#
# WHY THE PLACEBO ARM EXISTS, AND WHY IT IS LOAD-BEARING. The evaluator stops
# a session at its first hit, so ANY reordering of the window reshuffles which
# sessions happen to land a target in the top 10, and on 200 sessions that can
# look like a win. The placebo is this same scorer ranking on meaningless
# terms -- same pool vocabulary, same in-pool IDF, same cardinality per turn,
# drawn deterministically instead of from what the shopper said.
#
# FIVE DRAWS, NOT ONE, AND EVERY CLAIM IS MADE AGAINST THE BEST OF THEM.
# The first version of this comment quoted one draw as "the placebo" and called
# ON vs PLACEBO established at p = 0.0026. That draw was seeded, through
# ``context.session_id``, on a uuid4 the evaluator regenerates every run: the
# control was re-randomized on each execution and the published verdict was
# one favourable draw of nine, which flipped to "no verdict" on re-run (C and
# D, Phase 14 review -- the one blocker either raised). The seed is stable
# now and the claim is made against the HIGHEST-SCORING draw, which is the
# one draw that does not clear significance. Stated plainly: on this
# evidence the score comparison alone does not separate ON from a lucky
# reordering at the 0.05 level in every draw.
#
# WHAT DOES SEPARATE THEM IS THE MOVEMENT TABLE, and it is not close. Over
# the turns where the target sits inside the window:
#
#            in window   mean rank    up   down   into top 10   pushed out
#   ON             221   12.3-> 7.7   96      5            38            1
#   PLACEBO      294-320   ~12->~18  24-30 218-230        15-18        74-93
#
# Real terms move the target up 96 times and down 5. Every placebo draw moves
# it DOWN on 218-230 turns and pushes it out of the top 10 on 74 to 93.
# The scores are one sample of a noisy statistic; this is what the mechanism
# does on every turn, and the two arms are not the same phenomenon. An
# earlier version of this stage scored +0.024 while moving the target DOWN on
# average; that contradiction was a bug (see build_scorer), not a subtlety,
# and it is why this table is printed.
#
# THE WINDOW SWEEP, which is also where an earlier version of this comment was
# wrong. top_n=10 is the control: the window is then the response itself, so
# the stage can reorder only what was already being shown.
#
#   top_n     TS        vs OFF     McNemar vs OFF
#      10   0.659781   +0.020574   0/0    p = 1.0000  <- control, exactly null
#      20   0.677499   +0.038292   8/1    p = 0.0391
#      50   0.726726   +0.087519  22/1    p = 0.0000  <- committed
#     100   0.758418   +0.119211  29/0    p = 0.0000
#     200   0.771381   +0.132174  33/0    p = 0.0000
#
# Established at 20, 50, 100 and 200; null at the control. Five comparisons,
# so at a Bonferroni alpha/5 = 0.01 the top three still clear (20 does not). What carries it
# beyond any single p-value is that the control is null and the direction is
# consistent across the plateau -- the same standard the category route is
# held to in retrieval.py.
USE_SEMANTIC_RERANK = True

# CP 14.1. The reranker sees the top N of the ranked list and nothing else.
#
# 50, which is NOT the best-scoring row above -- and the gap has grown twice
# now. The sweep no longer decays at 200 at all: 100 measures +0.032 TS
# better than 50 and 200 another +0.013 on top. Taking either would mean
# selecting a maximum on the same 200 sessions the score is then reported on,
# AFTER seeing the sweep -- a worse version of the same trade each time the
# gap widens, not a better one. The widening is itself a reason for caution:
# a value that looks more and more attractive every time the pipeline changes
# is a value being fitted to this pipeline. 50 was chosen a priori, before any sweep ran, from Phase 13's
# measured rank distribution -- 176 of the 367 eligible-turn pool hits sit at
# final rank 11-50, the largest single band a reordering layer can reach --
# and it lands inside the established plateau with almost nothing lost (22
# gained, 1 lost). An a-priori value inside a broad established plateau
# generalizes better than a peak picked off the plateau's own chart.
#
# The +0.032 at 100 is left on the table deliberately and recorded here so the
# choice is auditable rather than quietly optimal. Whoever revisits it should
# revisit it with a held-out split, not with this table.
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
        # ``sorted``, not the set itself. The score below is a float SUM, and
        # float addition is not associative, so iterating a set of strings
        # made the last bits of two near-equal scores depend on
        # PYTHONHASHSEED: one real turn reordered ranks 26-28 on 42 of 400
        # seeds (D Phase 14 review). It never reached the scored top 10 and
        # results.json was byte-identical across seeds -- but "deterministic"
        # is a claim this module makes in its docstring, and a claim that
        # holds only outside the visible window is not the claim.
        #
        # It also holds only on the interpreter you happen to run: CPython
        # 3.12's ``sum`` compensates (Neumaier) and hides most of this, while
        # 3.9 -- the version docs/submission_rules.md certifies against, and
        # the one the observation above came from -- does not. A bug visible
        # only on the interpreter that scores the submission is the worst
        # place for one to hide, so it is fixed at the input rather than
        # relied on to stay invisible.
        wanted = sorted(self.query_terms(context))
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

    Reading ``normalized`` is also what keeps the HARNESS out of the ranking
    signal, and that is not automatic -- it is a property of what
    ``state.update_evidence`` strips. Until the Phase 14 review it stripped
    override plumbing and slot markers but not request framing, so `still`,
    `exploring`, `key` and `requirement` -- words the local evaluator's
    sentence templates put in every session -- arrived here as query terms
    and took 31% of this scorer's total applied weight. They are stripped at
    the source now (``state._REQUEST_FRAMING``); section 6 of
    ``tools/phase14_reranker.py`` prints the weight breakdown so a template
    that gets past that set is visible rather than inferred.
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


def _scorer_name(scorer: Any | None) -> str | None:
    """The scorer's name, or ``None``, without trusting the scorer.

    ``getattr(scorer, "name", None)`` looks total and is not: ``name`` on a
    model wrapper can be a property, and a property can raise. That line sat
    ABOVE ``rerank``'s try block, so a scorer that raised while being asked
    what it was called took the turn down before any fallback existed to
    catch it (D Phase 14 review). Nothing about a diagnostics label is worth
    a turn.
    """
    try:
        name = getattr(scorer, "name", None)
    except Exception:
        return None
    return name if isinstance(name, str) else None


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
    diagnostics = _blank_diagnostics(_scorer_name(scorer))

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


def safe_build_scorer(
    connection: Any, candidates: list[Candidate], context: Context
) -> Any | None:
    """``build_scorer`` that cannot take the turn down. The call site.

    CP 14.3/14.5 cover a scorer that raises while SCORING. They did not cover
    a scorer that raises while being BUILT, because ``agent.respond`` passed
    ``build_scorer(...)`` as an argument to ``rerank`` -- evaluated before
    ``rerank`` is entered, and so outside every fallback the phase advertises
    (D Phase 14 review). The one function the docstrings say a future encoder
    plugs into was the one function with no guard: loading a model is exactly
    the step that fails on a machine with no weights, no network, or no
    memory, and it would have failed OUTSIDE the machinery built for it.

    Returning ``None`` here is the CP 14.5 path, which is the correct
    degradation: no scorer, ranking's order stands, the turn is answered.
    """
    try:
        return build_scorer(connection, candidates, context)
    except Exception:
        return None


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
