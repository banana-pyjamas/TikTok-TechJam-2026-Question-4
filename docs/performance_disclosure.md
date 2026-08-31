# Latency, token usage, and estimated model cost

Required by `docs/submission_rules.md` ("a disclosure of latency, token
usage, and estimated model cost").

Every figure below is produced by `python3 -m tools.disclosure`, which times
the shipped `Agent.respond` over the full 200-session public set and refuses
to print anything if timing has changed the score. Re-run it and update this
file in the same change — the numbers move when the pipeline moves, and a
disclosure written by hand goes stale silently.

## Token usage and cost

| | |
| --- | --- |
| prompt tokens | **0** |
| completion tokens | **0** |
| total tokens | **0** |
| estimated model cost | **$0.00** |

Not an estimate, and not a figure rounded down. **The agent makes no model
call of any kind.** There is no LLM in the turn path, no API client anywhere
in the package, and no credential to supply. The `usage` field of every
response reports a literal `{"prompt_tokens": 0, "completion_tokens": 0}`
rather than an untracked or unavailable number.

`tests/test_submission.py` fails if any module under `starter/` ever imports
a network or model library, so this section cannot quietly become false.

## Latency

Measured on the certified interpreter, end to end — state update, retrieval,
ranking, reranking, clarification and payload construction:

| | |
| --- | --- |
| turns measured | 932 |
| mean | **43.19 ms** |
| median | 36.59 ms |
| P90 | 76.17 ms |
| max | 184.08 ms |
| index build (once, at construction) | 16.74 s |
| whole 200-session run | 40.4 s |

The index build is a one-time startup cost: the 50k-product catalog is read
once and loaded into an in-memory SQLite FTS5 index plus a signals side
table. No per-turn work touches the disk, and nothing is written.

Latency is a property of the machine as much as of the code, so the machine
is stated rather than implied:

| | |
| --- | --- |
| python | 3.9.6 (CPython) |
| platform | macOS-26.6.2-arm64 |
| machine | arm64 |

On a faster development interpreter (3.12.2, same machine) the same run
takes roughly 28 s with a ~30 ms mean turn. Neither figure should be read as
a guarantee about the organizer's hardware. What **is** a property of the
code, and does transfer: the work per turn is bounded and fixed — two FTS
queries, one indexed metadata lookup over at most 300 candidates, two
vocabulary queries for the reranker, and pure-Python scoring. There is no
retry, no backoff, no timeout that can extend a turn, and no external call
that can hang one.

The one stage with an explicit budget is the reranker
(`RERANK_BUDGET_MS = 150`), which bounds the scorer call and falls back to
the ranking order if it overruns. Over the whole public set it overruns
**0** times; the scorer itself costs ~0.1 ms per turn, and its setup ~3.4 ms.

## Memory

| | |
| --- | --- |
| peak traced, after index build | 5 MiB |
| peak traced, whole run | 198 MiB |

These are Python-object allocations only. The catalog lives inside SQLite's
own in-memory pages, which `tracemalloc` does not see, so treat both figures
as a floor rather than a total.

## Network and credentials

| | |
| --- | --- |
| network calls | **0** |
| credentials required | **none** |
| offline fallback | not applicable — the offline path is the only path |

`docs/submission_rules.md` notes that organizer policy may disable network
access for final scoring. That is a no-op here. The submission does not
degrade offline; it has no online mode to degrade from.

## Reproducing this disclosure

```bash
python3 -m tools.disclosure
```

The tool asserts the whole pinned configuration before it starts and aborts
if the timed run does not reproduce `COMMITTED_TECHNICAL_SCORE`, so a
disclosure can never describe a pipeline other than the one that ships.
