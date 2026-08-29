# CP 0.1 — File Ownership

Team model: Person A is the **sole production implementer**. B / C / D are
reviewers who report reproducible failures; they do not edit production code.

## Person A owns (may modify on `dev` only)

| Path | Role |
| --- | --- |
| `starter/agent.py` | The `Agent` entrypoint: `reset(session_id, user_profile)` and `respond(session_id, user_message, turn, top_k)`. Only file the evaluator imports. |
| `starter/contracts.py` | Frozen shared types (`SessionState`, `Context`, `Strategy`, `Candidate`, `RankingResult`). Change only via an approved INTERFACE CHANGE REQUEST — see `interface_mutation_rule.md`. |
| `starter/*.py` (future) | All later production modules — deterministic state manager, delta extractor/validator, retrieval routes, candidate union, constraint ranking, semantic reranker, clarification heuristic, feature-flag config. Created and modified only by A. |
| `tests/test_*.py` for A's modules | Checkpoint-specific tests A adds alongside each change (e.g. `tests/test_contracts.py`). |
| `docs/file_ownership.md`, `docs/interface_mutation_rule.md` | Team process contracts. |

## Not owned — must not modify

| Path | Reason |
| --- | --- |
| `main` branch | Development happens only on `dev`. Never commit, merge, or switch to `main`. |
| `evaluator/` | Organizer simulator and scorer. Treated as read-only; our score is only valid against the unmodified evaluator. |
| `data/` | Frozen catalog and public sessions. |
| `docs/competition_specification.md`, `docs/agent_api_contract.json`, `docs/evaluation_config.json`, `docs/baseline_results.json`, `README.md`, `DATA_ATTRIBUTION.md` | Frozen competition inputs / references. |
| `tests/test_evaluator.py` | Organizer-facing evaluator tests. A runs them for regression but does not weaken them. |
| `organizer/`, `secure/`, `docs/audits/` | Organizer-only, gitignored. |

## Reviewer-required outputs (what A must expose to B / C / D)

### B — Retrieval Guardian
- `Candidate` objects carrying `parent_asin`, `route_sources`, and separate
  `bm25_score` / `category_score` / `attribute_score`.
- The candidate pool **before ranking** must be inspectable (route diagnostics,
  candidate coverage, per-route contribution counts).
- Metrics B computes against that pool: Recall@50 / @100 / @300, candidate
  coverage.

### C — Ranking / Metrics Guardian
- `RankingResult.diagnostics` keyed by `parent_asin` exposing base score,
  attribute contribution, violation penalty, popularity prior, final score,
  and rank (populated from Phase 6 onward).
- Deterministic, stable ordering; unique `parent_asin` in the Top 10.
- Hooks to measure clarification impact on HitRate@10 / MRR / MTTC.

### D — QA / Integration Guardian
- `reset()` / `respond()` deterministic and side-effect-isolated per session.
- No cross-session state leakage.
- Core path (state, BM25, category, attribute, deterministic ranking,
  candidate vocabulary, clarification heuristic) runs with **no network**.
- Dense / LLM / semantic components have a deterministic fallback.
- `respond()` output always satisfies `docs/agent_api_contract.json`
  (`message` string, allowed `ask_attribute`, ≤ 10 unique valid `parent_asin`).
- Feature flags: OFF preserves prior behavior exactly.

## Debug output policy

Diagnostics (candidate route sources, ranking breakdown) are kept separate
from the official `respond()` payload. The user-facing `message` stays clean;
no excessive logging in the returned response.
