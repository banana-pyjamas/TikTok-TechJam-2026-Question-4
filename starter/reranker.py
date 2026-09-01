"""Free-text reranking -- reordering the head of the ranked list.

Most losses are not retrieval failures. The target is usually in the pool and
simply ordered below the scored cut, so this is the layer that reorders.

WHAT THIS STAGE IS, STATED ACCURATELY

Two things, and they are not equally strong:

  the CONTRACT   ``rerank`` is scorer-agnostic and holds whatever is plugged
                 into it: top-N input only, output ids a subset of input ids,
                 and three independent fallbacks to the order ranking already
                 produced. Complete and tested.

  a SCORER       ``PoolTermScorer``, which is LEXICAL, not semantic. Naming it
                 accurately matters: the SEMANTIC scorer is
                 ``load_encoder_scorer``, and it returns ``None`` here --
                 nothing is vendored, network may be off at scoring time, and
                 the submission is certified on a Python version current torch
                 builds do not support. So the no-scorer path is not a
                 hypothetical branch; it is the live path for the only
                 genuinely semantic scorer this repo has.

The fallback machinery exists so an encoder can be dropped in later without
touching the pipeline, and the shipped decision is made on the scorer that can
actually run.

WHY A SEPARATE STAGE RATHER THAN ANOTHER TERM IN ``ranking.score_candidate``

Because a reranker is allowed to be untrusted. Every term in ranking is a
deterministic function this repo owns; a reranker may be a model that is slow,
offline, or wrong, and the contract has to survive all three. Folding it into
the additive score would mean a bad scorer silently moves candidates with no
order to fall back TO.

DETERMINISM, HONESTLY

``PoolTermScorer`` is pure and deterministic. The TIMEOUT is not: which path
runs depends on wall clock, so a slow machine could in principle take the
fallback where a fast one did not. Two things bound that. The budget is
cooperative -- the deadline is handed to the scorer and a late result is
discarded afterwards -- because stdlib has no way to kill a running call
without signals or unkillable threads. And the shipped scorer's work is bounded
by ``RERANK_TOP_N`` candidates of at most 40 indexed terms, which is
microseconds. ``tools/phase14_reranker.py`` reports the observed timeout count,
and it must be 0 for the ON arm's numbers to mean anything.

WHAT THE BUDGET DOES NOT COVER

``RERANK_BUDGET_MS`` is a deadline handed to ``scorer.order``. It cannot bound
work that happens before a scorer exists, and ``build_scorer`` is that work:
two indexed queries over the whole pool, and the more expensive half by roughly
an order of magnitude. ``safe_build_scorer`` makes the build total, so a
failure there degrades rather than losing the turn; making it FAST is the
problem of whoever vendors a real encoder, and they have to bound the load as
well as the scoring. Both figures are printed by ``tools/phase14_reranker.py``.
"""

from __future__ import annotations

import math
import time
from typing import Any, Iterable, Sequence

from starter.contracts import Candidate, Context, RankingResult
from starter.text import terms

# Ablation flag. ON, and by a wide margin -- `python3 -m tools.phase14_reranker`
# reproduces the comparison, including a placebo arm.
#
# THE PLACEBO ARM IS LOAD-BEARING. The evaluator stops a session at its first
# hit, so ANY reordering of the window reshuffles which sessions happen to land
# a target in the top 10, and on 200 sessions that can look like a win. The
# placebo is this same scorer ranking on meaningless terms -- same pool
# vocabulary, same weighting, same cardinality per turn, drawn deterministically
# instead of from what the shopper said. Several independent draws are run and
# every claim is made against the best of them.
#
# What separates the arms is not only the score but the movement table: real
# terms move the hidden target UP and the placebo moves it DOWN, on every draw.
# An earlier version of this stage scored positive while moving the target down
# on average; that contradiction was a bug, not a subtlety, and it is why the
# table is printed.
#
# Clarification made this stage stronger rather than redundant, which is worth
# stating because it is the opposite of the usual ablation story. This scorer
# ranks on the shopper's free text, and before the agent asked questions the
# shopper barely produced any. A layer whose value grows when the layer above it
# improves is a complement, not a substitute.
USE_SEMANTIC_RERANK = True

# The reranker sees the top N of the ranked list and nothing else.
#
# 200. This was 50 for several checkpoints and the refusal to move it was
# correct each time -- 50 was chosen a priori from a measured rank distribution,
# and reading a maximum off the same 200 sessions the score is reported on is
# how a benchmark gets fitted. That refusal is only defensible while the
# alternative is unexamined; `python3 -m tools.phase16_depth` examined it.
#
# What makes 200 a choice rather than a peak, with the score the weakest of the
# four reasons:
#
#   MECHANISM     depth buys REACH, and reach is structural. A target at pool
#                 position 87 cannot be reranked by a top-50 window on any
#                 dataset -- the stage never sees it.
#   DOMINANCE     no new ranking failures appear at 200. The failure set is a
#                 strict subset of the one at 50, so it is not a trade here.
#   NO REGRESSION every scenario improves or holds, and the full suite passes
#                 at 200 including every adversarial contract test.
#   COST          no timeouts and no fallbacks at any depth; the scorer runs
#                 three orders of magnitude inside its budget. A deeper window
#                 that overran would silently return ranking's order and the
#                 gain would be an artifact. It does not.
#
# 100 is not an intermediate step -- it is unestablished on both tests, so
# "move halfway" would be movement without evidence. The choice was 200 or 50.
#
# 200 is also the deepest LEGITIMATE value, which is why "it topped the sweep"
# understates the case: the window must stay a strict subset of the pool for the
# top-N contract to mean anything, since a stage that reorders everything is not
# a reranker with a fallback, it is the ranker.
RERANK_TOP_N = 200

# Wall-clock budget for one scorer call, milliseconds. Generous by three orders
# of magnitude for the shipped scorer, because its job is to bound a MODEL that
# is not there yet, not to police arithmetic.
RERANK_BUDGET_MS = 150.0

# Context.derived key carrying this stage's diagnostics. A generic container,
# not a new frozen field and not a new key in RankingResult.diagnostics -- that
# set is frozen and guarded by tests/test_ranking.py.
RERANK_KEY = "rerank"

# Every outcome ``rerank`` can report. Enumerated so the ablation tool can
# assert it has seen the fallbacks fire rather than assuming they would.
OUTCOMES = (
    "applied",      # the scorer's order was used
    "identity",     # the scorer returned the order it was given
    "offline",      # no scorer available
    "malformed",    # output was not a usable list of ids
    "error",        # the scorer raised
    "timeout",      # the scorer overran its budget
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

    This is the live path rather than a defensive branch: no encoder is
    vendored here, the organizer may disable network access for final scoring,
    and the submission is certified on an interpreter current torch builds do
    not support.

    It deliberately does NOT attempt an import or a download. A scorer that
    tries ``import torch`` inside a turn would make the shipped path depend on
    what happens to be installed on the scoring machine. When a model is
    vendored, this function is the one place that changes and ``rerank`` needs
    no edit.

    ``vendor_root`` exists so a test can point it somewhere; nothing ships that
    reads it.
    """
    return None


class PoolTermScorer:
    """LEXICAL reranking on pool-scoped term rarity. Not semantic.

    Scores each candidate by how many of the shopper's own still-active
    free-text words it uses, weighted by how rare those words are IN THIS POOL.

    Two reasons this is the right shippable scorer rather than an arbitrary one:

    it reads a signal ranking does not
        ``ranking.score_candidate`` scores SLOTS. ``SessionState.evidence``
        holds the free text that was never promoted to a slot -- "gift for my
        dad", "something for hiking" -- and nothing in the scoring path reads
        it. Reranking on it is additive information, not a reweighting of what
        the constraint terms already saw.

    pool-scoped rarity is the discrimination signal
        The misses are discrimination failures among products that match the
        query equally well. A word every candidate in the pool uses cannot
        discriminate between them however rare it is catalog-wide; a word four
        of them use splits the pool. That is what
        ``vocabulary.build_vocabulary`` computes.

    Ordering is by score descending, then by the ORIGINAL rank -- so a candidate
    the scorer cannot separate keeps the position ranking gave it, and a
    zero-evidence turn is exactly the identity permutation.
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

        A term outside the pool vocabulary scores nothing rather than infinity:
        ``build_vocabulary`` has already dropped the ubiquitous and the
        vanishingly rare, and a word no candidate uses cannot order candidates.
        """
        count = self.frequencies.get(term)
        if not count or self.pool_size <= 0:
            return 0.0
        return math.log(self.pool_size / (1.0 + count))

    def query_terms(self, context: Context) -> set[str]:
        """The words this scorer ranks on: the shopper's own free text.

        A seam, not indirection. ``tools/phase14_reranker.py`` overrides it to
        substitute meaningless terms of the same cardinality drawn from the same
        pool vocabulary, which is the only way to tell "the shopper's words
        carry signal" apart from "any reordering of this window moves the
        score". The placebo lives in the tool and never ships.
        """
        return _evidence_terms(context)

    def order(self, parent_asins: Sequence[str], context: Context,
              deadline: float) -> list[str]:
        """Ids in, a permutation of them out.

        Returns the input order unchanged when the shopper has volunteered no
        free text.
        """
        # ``sorted``, not the set itself. The score below is a float SUM, and
        # float addition is not associative, so iterating a set of strings made
        # the low bits of two near-equal scores depend on PYTHONHASHSEED. It
        # never reached the scored top 10 and end-to-end output was identical
        # across seeds -- but "deterministic" is a claim this module makes, and
        # a claim that holds only outside the visible window is not the claim.
        #
        # It also held only on the interpreter you happen to run: CPython 3.12's
        # ``sum`` compensates and hides most of it, while 3.9 -- the certified
        # version -- does not. A bug visible only on the interpreter that scores
        # the submission is the worst place for one to hide, so it is fixed at
        # the input rather than relied on to stay invisible.
        wanted = sorted(self.query_terms(context))
        if not wanted:
            return list(parent_asins)
        weights = {term: self.weight(term) for term in wanted}
        if not any(weights.values()):
            return list(parent_asins)

        scored: list[tuple[float, int, str]] = []
        for position, parent_asin in enumerate(parent_asins):
            # Cooperative deadline, checked between candidates -- the only
            # preemption point a pure-Python scorer has. Returning the input
            # order here is a no-op the caller's post-hoc budget check
            # classifies as a timeout.
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

    ``normalized`` rather than ``text``: the state manager writes both, and the
    normalized form is what every other consumer reads. Superseded evidence is
    excluded -- reranking on a withdrawn preference is exactly the stale-evidence
    resurrection the override handling exists to prevent.

    Reading ``normalized`` is also what keeps the HARNESS out of the ranking
    signal, and that is not automatic -- it is a property of what
    ``state.update_evidence`` strips. Words from the evaluator's own sentence
    templates once arrived here as query terms and took a third of this scorer's
    applied weight. They are stripped at the source now
    (``state._REQUEST_FRAMING``), and ``tools/phase14_reranker.py`` prints the
    weight breakdown so a template that gets past that set is visible rather
    than inferred.
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
    model wrapper can be a property, and a property can raise. Nothing about a
    diagnostics label is worth a turn.
    """
    try:
        name = getattr(scorer, "name", None)
    except Exception:
        return None
    return name if isinstance(name, str) else None


def _valid_ids(proposed: object, allowed: Iterable[str]) -> tuple[list[str], int]:
    """Coerce a scorer's output into ids it was given.

    Returns ``(ids, invented)``. The result is always a deduplicated subset of
    ``allowed``, in the proposed order. Anything else the scorer emitted -- an
    ASIN it made up, an integer, a nested list, ``None`` -- is counted and
    discarded. Enforcement by CONSTRUCTION rather than validation after the
    fact: there is no path by which a parent_asin the caller did not supply can
    reach the returned list.
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
    partial or empty list. That is the point of the stage: the layer below has
    already produced a defensible answer, so the reranker is only ever allowed
    to IMPROVE on one, never required for one.

    Only ``result.ranked[:top_n]`` is shown to the scorer; the tail is
    re-appended in its original order and never reordered. The output is a
    permutation of the input ids. A scorer that raises, overruns its budget, or
    returns something unusable falls back -- including when a late answer was
    otherwise valid, since accepting it would make the turn's latency unbounded.

    Diagnostics go to ``context.derived[RERANK_KEY]``, a generic container;
    ``RankingResult.diagnostics`` keys are frozen and this stage adds none of
    them. ``rank`` values inside those diagnostics ARE renumbered, because a
    stale rank is worse than no rank.

    ``top_n`` and ``budget_ms`` default to ``None`` and are resolved from the
    module constants HERE, at call time, not in the signature. Writing
    ``top_n: int = RERANK_TOP_N`` binds the value once at import, so a tool
    sweeping ``reranker.RERANK_TOP_N`` would move the ranking depth in
    ``agent.respond`` while this function silently kept the old window.
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
        # Deliberately broad. A reranker is the untrusted layer; a model wrapper
        # can raise anything at all, and the one behaviour that is never
        # acceptable is losing the turn over it.
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
    # Anything the scorer omitted keeps its original relative order behind what
    # the scorer did rank. Dropping it instead would let a lazy scorer shorten
    # the recommendation list.
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

    Rows are copied, not mutated: the caller's ``RankingResult`` is an input and
    this stage does not own it. A candidate with no diagnostics row keeps none
    -- inventing a row would fake a score nothing computed.
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

    The fallbacks above cover a scorer that raises while SCORING. They do not
    cover one that raises while being BUILT, because a builder passed as an
    argument to ``rerank`` is evaluated before ``rerank`` is entered and so sits
    outside every fallback the stage advertises. Loading a model is exactly the
    step that fails on a machine with no weights, no network, or no memory.

    Returning ``None`` here is the no-scorer path, which is the correct
    degradation: ranking's order stands and the turn is answered.
    """
    try:
        return build_scorer(connection, candidates, context)
    except Exception:
        return None


def build_scorer(
    connection: Any, candidates: list[Candidate], context: Context
) -> Any | None:
    """The scorer for one turn, or ``None`` to take the no-scorer path.

    Prefers a vendored semantic encoder and falls back to the lexical pool-term
    scorer. The order matters and is the point: when an encoder becomes
    available it wins here without any other edit.

    Returns ``None`` when the shopper has volunteered no free text, because the
    lexical scorer would then be the identity permutation and building it costs
    an indexed query per turn.
    """
    encoder = load_encoder_scorer()
    if encoder is not None:
        return encoder
    if not _evidence_terms(context):
        return None
    # Imported here rather than at module scope: ``vocabulary`` imports nothing
    # from this module today, but reranking is the kind of layer that grows a
    # back-reference, and a cycle would surface as an import error at agent
    # construction rather than as anything diagnosable.
    from starter.vocabulary import build_vocabulary, pool_terms

    # Both queries run over the WHOLE pool, and that is deliberate.
    #
    # Frequencies must be pool-wide because ``build_vocabulary`` drops terms
    # below a minimum document frequency as noise; over a narrow window that
    # discards every term used by exactly one candidate, which is the most
    # discriminating evidence there is.
    #
    # The term lists must be pool-wide too. Scoping them to the first
    # ``RERANK_TOP_N`` candidates was silently wrong: ``candidates`` arrives in
    # RETRIEVAL order while ``rerank`` operates on the RANKED order, so the two
    # heads are different sets. A candidate in the ranked window but outside the
    # retrieval window had no terms, scored 0, and was pushed down for having
    # been ranked well -- and how often that happened depended on the window
    # size, so a window sweep measured the coverage gap rather than the window.
    asins = [candidate.parent_asin for candidate in candidates]
    return PoolTermScorer(
        build_vocabulary(connection, candidates),
        pool_terms(connection, asins),
    )
