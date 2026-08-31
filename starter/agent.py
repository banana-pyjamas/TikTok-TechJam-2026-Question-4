from __future__ import annotations

import copy
import json
import sqlite3
from pathlib import Path

from starter.catalog_meta import TABLE as META_TABLE
from starter.catalog_meta import create_table as create_meta_table
from starter.catalog_meta import lookup as meta_lookup
from starter.catalog_meta import popularity_scale
from starter.catalog_meta import signals as meta_signals
from starter.contracts import Candidate, Context, RankingResult, SessionState
from starter.ranking import POPULARITY_KEY, RELIABILITY_KEY
from starter.ranking import rank as constraint_rank
from starter.reliability import match_reliability, slot_coverage
# The MODULE, not its names. `config_guard.set_flag` assigns to the defining
# module, so `from starter.reranker import USE_SEMANTIC_RERANK` would bind a
# COPY here that the ablation could never flip -- the flag would read ON in
# the guard's report and OFF in the code it governs. Phase 12 closed that
# exact hole once already; it is not being reopened one module over.
from starter import clarify, reranker
from starter.retrieval import (_BM25_RANK, DEFAULT_ROUTES, POOL_LIMIT,
                               bm25_route, fuse, retrieve)
from starter.state import update_state
from starter.text import flatten_text as _text
from starter.text import terms as _terms


# `_BM25_RANK` is IMPORTED from retrieval above, not restated here.
#
# It was a second copy of the same seven field weights until the D Phase 15
# review pointed at it. Two identical literals in two modules is the drift
# shape this repo keeps finding (D-N2, D-P1): nothing would have failed if one
# of them had been retuned, the Phase 1 baseline path and the retrieval route
# would silently have been scoring different documents, and every ablation
# comparing them would have been comparing two things. The constant exists so
# the SELECT projection and the ORDER BY cannot drift apart (CP 1.3); that
# argument applies across modules for exactly the same reason.
_RESPONSE_MESSAGE = "Here are the closest matches I found."

# --------------------------------------------------------------------------
# Ablation flags (Phase 7 controlled comparison; Phase 16 staged enablement).
#
# Turning one OFF restores the behaviour of the phase before it landed:
#
#   USE_STATE               run the deterministic state manager each turn
#   USE_MULTI_ROUTE         3-route UNION pool vs the BM25-only pool
#   USE_CONSTRAINT_RANKING  constraint scoring vs pure retrieval order
#
# With all three OFF the agent reproduces the official weak-BM25 baseline,
# which is the validity check on the whole ablation
# (``python3 -m tools.phase7_ablation``).
#
# Phase 7 measured the 3-route union net-NEGATIVE (TS 0.131194 -> 0.115512)
# and it was disabled. The Phase 9 review showed that conclusion was an
# artifact of WHICH routes: the cost is the attribute route alone. With the
# route set corrected to bm25 + category the union is ON.
#
# It is ON because it measurably improves CANDIDATE RECALL, not because its
# score effect is established -- end to end with ranking ON it is +0.0034 TS
# at McNemar p = 1.0000, which is no verdict. See retrieval.DEFAULT_ROUTES for
# the full statement of what is and is not established here; do not quote a
# score gain for this flag without re-reading it.
#
# There is deliberately no USE_ADAPTIVE_STRATEGY flag. The strategy is always
# computed but does not gate retrieval -- mode-adaptive route selection
# measured worth +0.000298, far below the noise floor, so a flag for it would
# have been a knob that changes nothing.
# --------------------------------------------------------------------------
USE_STATE = True
USE_MULTI_ROUTE = True
USE_CONSTRAINT_RANKING = True

# The evaluator scores at most this many recommendations (agent_api_contract
# turn_request pins top_k to 10). A larger top_k must never yield a longer
# list -- enforced at both the ranking and the response stage (CP 1.6).
_MAX_RECOMMENDATIONS = 10


def _effective_k(top_k: int) -> int:
    """Clamp a caller-supplied top_k to ``[0, _MAX_RECOMMENDATIONS]``."""
    return min(max(0, top_k), _MAX_RECOMMENDATIONS)


# --------------------------------------------------------------------------
# Minimum end-to-end turn pipeline (Phase 1).
#
#   user message -> Context -> BM25 -> Candidate[] -> RankingResult -> respond()
#
# Each stage is a small pure function so B / C / D can review and test it in
# isolation. Retrieval and ranking are still the weak baseline: BM25 only, no
# state-driven query rewriting. Nothing here mutates SessionState yet (Phase 2).
# --------------------------------------------------------------------------


def _build_context(
    session_id: str, user_message: str, turn: int, state: SessionState
) -> Context:
    """CP 1.2 - wrap a turn's inputs in a minimal, safe Context.

    No query derivation happens yet; downstream reads ``user_message``
    directly. ``state`` is passed by reference so later layers read live
    session state; they must treat it as read-only. Only the deterministic
    state manager mutates ``SessionState`` (single-writer invariant, see
    ``starter/contracts.py``).
    """
    return Context(
        session_id=session_id,
        turn=turn,
        user_message=user_message if isinstance(user_message, str) else "",
        state=state,
    )


def _bm25_search(
    connection: sqlite3.Connection, text: str, limit: int
) -> list[tuple[str, float]]:
    """CP 1.3 - the baseline BM25 query, behavior unchanged.

    Same tokenization, same ``OR`` expression, same weighted ``bm25`` ORDER
    BY, same ``LIMIT``. The only difference from the old inline query is that
    the raw ``bm25`` value is also selected (projection only - it does not
    affect which rows match or their order). Rows come back best-first;
    SQLite ``bm25`` is more negative for a better match.
    """
    terms = list(dict.fromkeys(_terms(text)))[:40]
    expression = " OR ".join(f'"{term}"' for term in terms)
    if not expression:
        return []
    rows = connection.execute(
        f"SELECT parent_asin, {_BM25_RANK} FROM products WHERE products MATCH ? "
        f"ORDER BY {_BM25_RANK} LIMIT ?",
        (expression, limit),
    ).fetchall()
    return [(str(parent_asin), float(raw)) for parent_asin, raw in rows]


def _to_candidates(rows: list[tuple[str, float]]) -> list[Candidate]:
    """CP 1.4 - BM25 rows -> Candidate[].

    Preserves ``parent_asin`` and retrieval order. The raw SQLite ``bm25``
    value is negated on the way in so ``route_scores["bm25"]`` follows the
    usual "higher is better" convention; ``route_sources`` becomes
    ``("bm25",)``.
    """
    return [
        Candidate(parent_asin=parent_asin, route_scores={"bm25": -raw})
        for parent_asin, raw in rows
    ]


def _rank(candidates: list[Candidate], top_k: int) -> RankingResult:
    """CP 1.5 - sort candidates by BM25 score (higher = better), keep Top-k.

    ``top_k`` is clamped to the frozen maximum (10); a larger value cannot
    produce a longer list. Python's sort is stable, so candidates with an
    equal score keep their retrieval order. ``diagnostics`` here is minimal
    and provisional; the full ranking-diagnostics schema is CP 6.7.
    """
    ordered = sorted(
        candidates,
        key=lambda candidate: candidate.route_scores.get("bm25", float("-inf")),
        reverse=True,
    )[: _effective_k(top_k)]
    diagnostics = {
        candidate.parent_asin: {
            "rank": index + 1,
            "bm25_score": candidate.route_scores.get("bm25", 0.0),
        }
        for index, candidate in enumerate(ordered)
    }
    return RankingResult(ranked=ordered, diagnostics=diagnostics)


def _to_response(result: RankingResult, top_k: int,
                 ask: str | None = None) -> dict:
    """CP 1.6 / CP 15.4 - RankingResult -> evaluator ``respond()`` payload.

    ``message`` is the baseline string and ``recommendations`` is at most 10
    ``{"parent_asin": ...}`` entries in ranked order -- the frozen maximum is
    enforced here regardless of ``top_k``.

    CP 15.4 IS STRUCTURAL HERE, not a rule this function follows. Both fields
    are built from the SAME ranked result in the same expression, so there is
    no code path that returns a question without a recommendation list, and
    none that suppresses recommendations because the agent decided to ask
    something. That matters more than it sounds: the evaluator scores the
    recommendations of EVERY turn, so an agent that asked instead of answering
    would score zero on the turn it asked, and the natural chat-shaped
    implementation -- "if unsure, ask; else recommend" -- is exactly that bug.

    ``ask`` defaults to ``None``, which is both the CP 15.1 no-ask baseline
    and the value every caller passed before Phase 15. It is validated against
    the contract enum by ``clarify.choose``; this function does not re-check
    it, because a second copy of the enum is a second thing to keep in step
    (CP 15.3 lives in one place, with a test that reads the organizer's JSON).
    """
    recommendations = [
        {"parent_asin": candidate.parent_asin}
        for candidate in result.ranked[: _effective_k(top_k)]
    ]
    return {
        "message": _RESPONSE_MESSAGE,
        "ask_attribute": ask,
        "recommendations": recommendations,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    }


class Agent:
    """Minimum end-to-end agent.

    Per-session ``SessionState`` plus baseline BM25 retrieval, wired through
    the shared ``Context`` / ``Candidate`` / ``RankingResult`` contracts.
    Retrieval and ranking are still the weak baseline (BM25 only, no
    state-driven query). No network or LLM dependency.
    """

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._states: dict[str, SessionState] = {}
        # Phase 15, per session, created by ``reset`` alongside the state.
        self._clarify: dict[str, clarify.ClarificationLedger] = {}
        self._build_index()
        # Phase 11 (CP 11.2). A property of the frozen catalog, so it is
        # computed once per agent and never per turn.
        #
        # Measured cost (D Phase 11 follow-up), since USE_CONFIDENCE_WEIGHTING
        # currently ships OFF and this therefore runs for a consumer that is
        # switched off:
        #
        #   init without it   5.680s     7 aggregate queries (1 count + 6 slots)
        #   init with it      5.766s     0 reliability queries per respond()
        #   delta            +0.086s     +1.52%
        #
        # Negligible, so it is left unconditional rather than gated. That
        # keeps flipping the flag a one-line change with no other edits, which
        # is the point of an ablation flag. Worth being explicit about what
        # OFF means: the scoring path and the response are a no-op, NOT that
        # no work happens. Gating this behind the flag would trade 86ms for a
        # second thing to remember when the flag moves.
        #
        # The result is catalog-global and READ-ONLY: it is handed to every
        # Context by reference, never copied per turn and never mutated.
        self._reliability = match_reliability(slot_coverage(self.connection))
        # Phase 12 (CP 12.1/12.2). Catalog-wide popularity normalisation, on
        # the same terms: computed once, read-only, shared by reference.
        self._popularity = popularity_scale(self.connection)

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        create_meta_table(self.connection)
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        meta_batch: list[tuple] = []

        def flush() -> None:
            if not batch:
                return
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
            # Placeholders derived from the row so adding a signal column to
            # catalog_meta cannot silently desync this INSERT.
            placeholders = ", ".join("?" * len(meta_batch[0]))
            cursor.executemany(
                f"INSERT OR REPLACE INTO {META_TABLE} VALUES ({placeholders})",
                meta_batch,
            )
            batch.clear()
            meta_batch.clear()

        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                batch.append(
                    (
                        parent_asin,
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                meta_batch.append((parent_asin, *meta_signals(product)))
                if len(batch) >= 1000:
                    flush()
        if batch:
            flush()
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        """Initialize (or re-initialize) authoritative state for a session.

        A fresh ``SessionState`` replaces any existing state for this
        ``session_id``, so a re-``reset`` starts clean with nothing carried
        over. The anonymized profile is deep-copied into the state so later
        state mutation cannot leak back into the caller's object or across
        sessions that were reset from the same dict.
        """
        self._states[session_id] = SessionState(
            session_id=session_id,
            user_profile=copy.deepcopy(user_profile),
        )
        # Phase 15. The agent's own side of the conversation -- what it has
        # asked, and what turned out to be exhausted. Reset with the session
        # for the same reason the state is: a re-``reset`` must start clean, and
        # a ledger that outlived its session would carry one shopper's dead
        # questions into the next one's. Deliberately NOT inside SessionState;
        # see ``clarify.ClarificationLedger`` for why.
        self._clarify[session_id] = clarify.ClarificationLedger()

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        """One turn end to end.

        State manager (single writer) -> Context -> multi-route retrieval
        UNION (Phase 5) -> constraint-aware ranking (Phase 6) -> semantic
        rerank (Phase 14) -> clarification (Phase 15) -> payload. Retrieval,
        ranking and clarification never touch state.
        """
        if session_id not in self._states:
            raise RuntimeError("reset must be called before respond")
        state = self._states[session_id]
        if USE_STATE:
            update_state(state, user_message, turn)
        # CP 15.7, and it must run BEFORE this turn's question is chosen: this
        # message is the answer to the LAST one, and an attribute the shopper
        # has just declined must not be a candidate again a few lines below.
        ledger = self._clarify.setdefault(
            session_id, clarify.ClarificationLedger())
        if clarify.USE_CLARIFICATION:
            clarify.safe_observe(ledger, user_message)
        context = _build_context(session_id, user_message, turn, state)
        # CP 11.2 -- per-slot Match Reliability reaches ranking through the
        # generic `derived` bag, not a new frozen field (contracts rule).
        context.derived[RELIABILITY_KEY] = self._reliability
        context.derived[POPULARITY_KEY] = self._popularity

        # No build_strategy() call here on purpose. The strategy does not gate
        # retrieval -- mode-adaptive route selection measured worth +0.000298,
        # so the route set is a retrieval constant (DEFAULT_ROUTES) instead --
        # and computing a value nothing reads is the same "mechanism that
        # changes nothing" this phase deleted USE_ADAPTIVE_STRATEGY for
        # (D-N4). starter.strategy stays: it is pure, tested, and measured by
        # tools/phase9_mode_accuracy.py.
        #
        # Phase 15 was expected to be its first consumer and is NOT. The mode
        # would gate how hard to push for a clarification -- and on this
        # harness there is no cost to asking, so a gate on it could only
        # subtract. Wiring it in to honour the plan would be a knob that
        # changes nothing, which is the same thing this comment already
        # deleted USE_ADAPTIVE_STRATEGY for. See `clarify.choose` for where
        # the second gate belongs when a deployment gives questions a price.

        if USE_MULTI_ROUTE:
            pool = retrieve(self.connection, context, POOL_LIMIT, DEFAULT_ROUTES)
        else:
            pool = fuse({"bm25": bm25_route(self.connection, context, POOL_LIMIT)},
                        POOL_LIMIT)

        if USE_CONSTRAINT_RANKING:
            metadata = meta_lookup(self.connection, [c.parent_asin for c in pool])
        else:
            # Every slot then classifies UNKNOWN, so the constraint terms are
            # zero and the final score is the retrieval score alone -- pure
            # retrieval order, through the same code path.
            metadata = {}

        # Phase 14. Ranking normally truncates to the 10 the response carries;
        # a reranker handed 10 rows cannot reach the 11-50 band where Phase 13
        # measured most recoverable targets sitting, so with the flag ON the
        # ranked list is kept RERANK_TOP_N deep and re-truncated by
        # `_to_response`. With the flag OFF the depth, the call and the cost
        # are all exactly what Phase 12 shipped.
        if reranker.USE_SEMANTIC_RERANK:
            result = constraint_rank(
                pool, context, metadata,
                max(_effective_k(top_k), reranker.RERANK_TOP_N))
            result = reranker.rerank(
                result, context,
                reranker.safe_build_scorer(self.connection, pool, context))
        else:
            result = constraint_rank(pool, context, metadata, _effective_k(top_k))

        # Phase 15. Chosen from the RANKED head, after reranking, because the
        # question is "what would split the answer I am about to give" -- and
        # from `metadata`, which the ranking path has already looked up for
        # this exact pool, so clarification adds no query to the turn.
        #
        # With the flag OFF this is `None` and the payload is byte-for-byte
        # the Phase 14 one (CP 15.1). `ask` is never computed in that case:
        # a no-op arm that still does the work is a no-op that can still be
        # wrong, and the ablation would stop being free.
        #
        # `safe_choose`, not `choose`. Phase 14 shipped `build_scorer` bare as
        # an argument to `rerank` -- outside every fallback that stage
        # advertised -- and got `safe_build_scorer` in review. This line was
        # the same shape one phase later (D Phase 15 review). Clarification is
        # strictly optional, so its failure mode is no question, never a lost
        # turn.
        ask = None
        if clarify.USE_CLARIFICATION:
            ask = clarify.safe_choose(context, ledger, result.ranked,
                                      metadata, self._reliability)
            ledger.record(ask)
        return _to_response(result, top_k, ask)
