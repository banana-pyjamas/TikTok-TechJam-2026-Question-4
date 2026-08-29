from __future__ import annotations

import copy
import json
import re
import sqlite3
from pathlib import Path

from starter.contracts import Candidate, Context, RankingResult, SessionState


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}

# The baseline BM25 field-weight expression. Kept as one constant so the
# SELECT projection and the ORDER BY can never drift apart (CP 1.3).
_BM25_RANK = "bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)"
_RESPONSE_MESSAGE = "Here are the closest matches I found."

# The evaluator scores at most this many recommendations (agent_api_contract
# turn_request pins top_k to 10). A larger top_k must never yield a longer
# list -- enforced at both the ranking and the response stage (CP 1.6).
_MAX_RECOMMENDATIONS = 10


def _effective_k(top_k: int) -> int:
    """Clamp a caller-supplied top_k to ``[0, _MAX_RECOMMENDATIONS]``."""
    return min(max(0, top_k), _MAX_RECOMMENDATIONS)


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


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


def _to_response(result: RankingResult, top_k: int) -> dict:
    """CP 1.6 - RankingResult -> evaluator-compatible respond() payload.

    ``message`` is the baseline string, ``ask_attribute`` is ``None`` (no
    clarification yet), and ``recommendations`` is at most 10
    ``{"parent_asin": ...}`` entries in ranked order -- the frozen maximum
    is enforced here regardless of ``top_k``.
    """
    recommendations = [
        {"parent_asin": candidate.parent_asin}
        for candidate in result.ranked[: _effective_k(top_k)]
    ]
    return {
        "message": _RESPONSE_MESSAGE,
        "ask_attribute": None,
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
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                batch.append(
                    (
                        str(product["parent_asin"]),
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
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

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        """CP 1.7 - one turn end to end: Context -> BM25 -> Candidate[] ->
        RankingResult -> respond() payload. Read-only w.r.t. SessionState."""
        if session_id not in self._states:
            raise RuntimeError("reset must be called before respond")
        context = _build_context(
            session_id, user_message, turn, self._states[session_id]
        )
        rows = _bm25_search(self.connection, context.user_message, top_k)
        candidates = _to_candidates(rows)
        result = _rank(candidates, top_k)
        return _to_response(result, top_k)
