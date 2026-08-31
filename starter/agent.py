"""The agent entrypoint: one deterministic turn, end to end.

    user message -> state -> Context -> retrieval UNION -> constraint ranking
                 -> free-text reranking -> clarification -> payload

Only the state manager writes ``SessionState``; retrieval, ranking, reranking
and clarification all read it. No network, no model, no LLM.
"""

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
# The MODULE, not its names. ``config_guard.set_flag`` assigns to the defining
# module, so ``from starter.reranker import USE_SEMANTIC_RERANK`` would bind a
# COPY here that an ablation could never flip -- the flag would read ON in the
# guard's report and OFF in the code it governs.
from starter import clarify, reranker
from starter.retrieval import (_BM25_RANK, DEFAULT_ROUTES, POOL_LIMIT,
                               bm25_route, fuse, retrieve)
from starter.state import update_state
from starter.text import flatten_text as _text
from starter.text import terms as _terms


# ``_BM25_RANK`` is imported from retrieval, not restated here. Two identical
# literals in two modules is a drift shape: nothing would fail if one were
# retuned, and the baseline path and the retrieval route would silently score
# different documents.
_RESPONSE_MESSAGE = "Here are the closest matches I found."

# Ablation flags. Turning one OFF restores the behaviour from before it landed:
#
#   USE_STATE               run the deterministic state manager each turn
#   USE_MULTI_ROUTE         multi-route UNION pool vs the BM25-only pool
#   USE_CONSTRAINT_RANKING  constraint scoring vs pure retrieval order
#
# With all three OFF the agent reproduces the official weak-BM25 baseline,
# which is the validity check on the whole ablation
# (``python3 -m tools.phase7_ablation``).
#
# USE_MULTI_ROUTE is ON because it measurably improves CANDIDATE RECALL, not
# because its score effect is established -- end to end it is no verdict. See
# ``retrieval.DEFAULT_ROUTES``; do not quote a score gain for this flag without
# reading it.
#
# There is deliberately no adaptive-strategy flag. The strategy is computed but
# does not gate retrieval: mode-adaptive route selection measured far below the
# noise floor, so a flag for it would be a knob that changes nothing.
USE_STATE = True
USE_MULTI_ROUTE = True
USE_CONSTRAINT_RANKING = True

# The evaluator scores at most this many recommendations. A larger ``top_k``
# must never yield a longer list.
_MAX_RECOMMENDATIONS = 10


def _effective_k(top_k: int) -> int:
    """Clamp a caller-supplied top_k to ``[0, _MAX_RECOMMENDATIONS]``."""
    return min(max(0, top_k), _MAX_RECOMMENDATIONS)


def _build_context(
    session_id: str, user_message: str, turn: int, state: SessionState
) -> Context:
    """Wrap a turn's inputs in a minimal Context.

    ``state`` is passed by reference so later layers read live session state;
    they must treat it as read-only. Only the state manager mutates it.
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
    """The baseline BM25 query.

    Rows come back best-first; SQLite ``bm25`` is more negative for a better
    match, so callers negate it.
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
    """BM25 rows -> ``Candidate[]``, preserving retrieval order.

    The raw SQLite score is negated so ``route_scores["bm25"]`` follows the
    usual "higher is better" convention.
    """
    return [
        Candidate(parent_asin=parent_asin, route_scores={"bm25": -raw})
        for parent_asin, raw in rows
    ]


def _rank(candidates: list[Candidate], top_k: int) -> RankingResult:
    """Sort by BM25 score, keep the top k.

    ``top_k`` is clamped to the frozen maximum. Python's sort is stable, so
    equal-scoring candidates keep their retrieval order.
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
    """``RankingResult`` -> the evaluator's ``respond()`` payload.

    RECOMMENDATIONS AND A QUESTION IN THE SAME TURN IS STRUCTURAL HERE, not a
    rule this function follows. Both fields are built from the SAME ranked
    result in the same expression, so there is no path that returns a question
    without a recommendation list, and none that suppresses recommendations
    because the agent decided to ask something. That matters more than it
    sounds: the evaluator scores the recommendations of EVERY turn, so an agent
    that asked instead of answering would score zero on the turn it asked --
    and the natural chat-shaped implementation, "if unsure ask, else
    recommend", is exactly that bug.

    ``ask`` is validated against the contract enum by ``clarify.choose``; this
    function does not re-check it, because a second copy of the enum is a second
    thing to keep in step.
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
    """The agent the evaluator imports. One in-memory index, state per session."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._states: dict[str, SessionState] = {}
        self._clarify: dict[str, clarify.ClarificationLedger] = {}
        self._build_index()
        # Catalog-global and read-only: computed once per agent, handed to every
        # Context by reference, never copied per turn and never mutated. Left
        # unconditional rather than gated behind USE_CONFIDENCE_WEIGHTING -- it
        # costs about 1.5% of index build time, and gating it would trade that
        # for a second thing to remember when the flag moves.
        self._reliability = match_reliability(slot_coverage(self.connection))
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
        ``session_id``, so a re-``reset`` starts clean. The profile is
        deep-copied so later mutation cannot leak back into the caller's object
        or across sessions reset from the same dict. The clarification ledger is
        reset alongside it, for the same reason -- one that outlived its session
        would carry a shopper's dead questions into the next one's.
        """
        self._states[session_id] = SessionState(
            session_id=session_id,
            user_profile=copy.deepcopy(user_profile),
        )
        self._clarify[session_id] = clarify.ClarificationLedger()

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        """One turn end to end."""
        if session_id not in self._states:
            raise RuntimeError("reset must be called before respond")
        state = self._states[session_id]
        if USE_STATE:
            update_state(state, user_message, turn)
        # This must run BEFORE the turn's question is chosen: the message is the
        # answer to the LAST question, and an attribute the shopper just
        # declined must not be a candidate again a few lines below.
        ledger = self._clarify.setdefault(
            session_id, clarify.ClarificationLedger())
        if clarify.USE_CLARIFICATION:
            clarify.safe_observe(ledger, user_message)
        context = _build_context(session_id, user_message, turn, state)
        # Catalog-wide signals reach ranking through the generic ``derived``
        # bag, not a new frozen field.
        context.derived[RELIABILITY_KEY] = self._reliability
        context.derived[POPULARITY_KEY] = self._popularity

        # No strategy call here on purpose. The mode does not gate anything --
        # mode-adaptive routing measured below the noise floor, and on this
        # harness asking is free so a mode gate on clarification could only
        # subtract. Computing a value nothing reads is a mechanism that changes
        # nothing. ``starter.strategy`` stays: it is pure, tested, and measured
        # by tools/phase9_mode_accuracy.py. See ``clarify.choose`` for where a
        # second gate belongs when a deployment gives questions a price.

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

        # Ranking normally truncates to the 10 the response carries. A reranker
        # handed 10 rows cannot reach the band where most recoverable targets
        # sit, so with the flag ON the ranked list is kept RERANK_TOP_N deep and
        # re-truncated by ``_to_response``. With it OFF the depth, the call and
        # the cost are exactly what they were before the stage existed.
        if reranker.USE_SEMANTIC_RERANK:
            result = constraint_rank(
                pool, context, metadata,
                max(_effective_k(top_k), reranker.RERANK_TOP_N))
            result = reranker.rerank(
                result, context,
                reranker.safe_build_scorer(self.connection, pool, context))
        else:
            result = constraint_rank(pool, context, metadata, _effective_k(top_k))

        # Chosen from the RANKED head, after reranking, because the question is
        # "what would split the answer I am about to give" -- and from
        # ``metadata``, which the ranking path already looked up for this exact
        # pool, so clarification adds no query to the turn.
        #
        # ``ask`` is never computed with the flag OFF: a no-op arm that still
        # does the work is a no-op that can still be wrong.
        #
        # ``safe_choose``, not ``choose`` -- clarification is strictly optional,
        # so its failure mode must be no question, never a lost turn.
        ask = None
        if clarify.USE_CLARIFICATION:
            ask = clarify.safe_choose(context, ledger, result.ranked,
                                      metadata, self._reliability)
            ledger.record(ask)
        return _to_response(result, top_k, ask)
