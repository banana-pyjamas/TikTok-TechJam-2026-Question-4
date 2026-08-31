# Submission and setup

Everything `docs/submission_rules.md` asks for under "Reproducibility
Requirements", in one place. `README.md` is the organizer's file and is not
modified by this team.

## What this agent is, in one line

A deterministic, offline, stdlib-only conversational retrieval agent. No LLM,
no network, no credentials, no third-party package.

## Exact Python version

**Python 3.9.6.** This is stated because the requirement is non-default: the
organizer's README says "3.10 or later is recommended", and every measurement
this team reports is produced on 3.9.6 (`/usr/bin/python3` on macOS 15). The
full test suite and the public evaluator both pass on it.

The package is also exercised on 3.12.2 during development and behaves
identically, with one documented exception that is fixed rather than relied
on: CPython 3.12's `sum` uses compensated summation and 3.9's does not, so a
float sum over an unordered set could differ between them. `starter/reranker.py`
sorts its terms before summing so the result is the same on both. See the
`DETERMINISM, HONESTLY` section of that file.

Nothing in the package uses syntax newer than 3.9. Every module carries
`from __future__ import annotations`, so PEP 604 (`X | None`) annotations are
strings at runtime and never evaluated.

## Dependencies

None.

```bash
pip install -r requirements.txt   # succeeds and installs nothing
```

`requirements.txt` is intentionally empty of packages. It exists because "this
submission has no dependencies" and "nobody wrote the manifest" are otherwise
indistinguishable to whoever unpacks the bundle. `tests/test_submission.py`
fails if any module under `starter/` ever imports something outside the
standard library.

## Data the agent needs

| Path | Where it comes from |
| --- | --- |
| `data/catalog.jsonl` | The organizer's frozen 50k-product catalog. |
| `data/public_set.jsonl` | The organizer's frozen 200-session public set. |

`data/catalog.jsonl` is **gitignored in this repository** and is not part of
the source tree. A clone therefore cannot run until the organizer's catalog is
placed at that path. The agent reads it once at construction and builds an
in-memory SQLite FTS5 index; nothing is written to disk.

If the harness supplies the catalog at a different path, pass it to the
constructor — `Agent("/path/to/catalog.jsonl")`. The default is
`data/catalog.jsonl`.

## One command to run the agent in the official harness

```bash
python3 -m evaluator.local_evaluator
```

Writes `results.json` and prints the metric summary. With the frozen public
set this reproduces the committed score exactly; the value is pinned in
`tools/config_guard.py` as `COMMITTED_TECHNICAL_SCORE` and several tools
refuse to run if it has drifted.

Optional arguments, all defaulted:

```bash
python3 -m evaluator.local_evaluator \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --output results.json
```

## Environment variables

None. There are no non-obvious environment variables, and the agent reads none
at all.

`PYTHONHASHSEED` is worth one sentence because it is the variable most likely
to matter for a Python submission and here it does not: `results.json` is
byte-identical across `PYTHONHASHSEED=0`, `1` and `12345`, and
`tests/test_reranker.py` pins the one place where set-iteration order could
have reached the arithmetic.

## Verifying the submission

```bash
python3 -m unittest discover -s tests -t .   # 607 tests
python3 -m tools.config_guard                # every pinned flag and constant
python3 -m tools.phase16_integration         # staged enable, one feature at a time
python3 -m tools.disclosure                  # latency, tokens, cost
```

`tools/config_guard.py` is the configuration guard: it checks that every
`USE_` flag and every tunable constant in `starter/` is registered and
undrifted, and it fails loudly rather than exiting 0 on nothing.

## Network access

The agent makes no network call. `docs/submission_rules.md` notes that
organizer policy may disable network access for final scoring; that is a
no-op for this submission, because there is no online path to fall back
from. See `docs/method_and_limitations.md` for the full statement.

## Related documents

| File | Contents |
| --- | --- |
| `docs/method_and_limitations.md` | Method, model choice, and limitations |
| `docs/performance_disclosure.md` | Latency, token usage, estimated cost |
| `docs/file_ownership.md` | Who may modify what |
