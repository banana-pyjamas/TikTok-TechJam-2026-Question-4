# Clerk

A deterministic conversational shopping agent. **No LLM, no network, no model, no third-party
package.** It asks one good question and finds the shopper's hidden product in 4.4 turns of the 10
available.

TikTok TechJam 2026 — Problem 4, Conversational E-Commerce Search.

| | |
|---|---|
| **TechnicalScore** | **0.759659** |
| HitRate@10 | 0.9000 |
| MRR | 0.591530 |
| MTTC | 4.390 turns |
| vs. the provided baseline | **7.1×** (0.106710 → 0.759659) |
| Dependencies | none — Python standard library only |
| Determinism | `results.json` byte-identical across `PYTHONHASHSEED` 0 / 1 / 12345 |

---

## 1. Project overview

The evaluator gives an agent ten turns to surface a hidden target product from a 50,000-item
catalog, talking to a simulated shopper who **only discloses a requirement when asked for it**.

That last clause is the whole problem. For fourteen of our sixteen checkpoints `ask_attribute` was
hardcoded to `None` — the agent recommended without ever asking, so the shopper's real requirements
were never spoken and most turns were the same stuck non-answer. Turning the question on was worth
**+0.479 TS**, about **3.5× the entire retrieval and ranking stack beneath it** (which is worth
+0.137 over the baseline). Everything else in this repository is the machinery that makes a good
question possible and converts the answer once it arrives.

A turn runs six deterministic stages:

```
user message
  → state manager        deterministic slot extraction + free-text evidence
  → retrieval            BM25 + category routes over SQLite FTS5, RRF union
  → constraint ranking   match / violation / unknown per slot, + popularity prior
  → free-text reranking  in-pool IDF over the shopper's own words (lexical, no encoder)
  → clarification        which attribute to ask about, if any
  → response             10 recommendations AND a question, in the same turn
```

That last line matters: the evaluator scores the recommendations of *every* turn, so an agent that
asks *instead of* answering scores zero on the turn it asks. Both fields are built from the same
ranked result in one expression, so there is no code path that does that.

**Every stage sits behind an ablation flag and every flag was measured ON vs OFF before it shipped.**
Two features were measured, failed their gate, and were reverted. Four were measured and never
wired in. `tools/phase16_integration.py` re-derives the shipped configuration by enabling features
one at a time and reverting any rung that fails both a McNemar test and a paired permutation test —
and asserts that the ladder lands on exactly the flags we ship.

### What this is not

There is no language model anywhere in the system. `load_encoder_scorer()` returns `None` on every
turn of every run, and the reranker is called the *free-text* reranker rather than the semantic one
because nothing semantic ships. The seam for a vendored encoder exists and is tested; it is empty
on purpose, because the organizer's rules note that network access may be disabled for final
scoring and we wanted a submission that could not care.

---

## 2. Setup and installation

**Python 3.9.6 exactly.** This is stated because it is non-default: every number we report is
produced on 3.9.6 (`/usr/bin/python3` on macOS). The suite also passes on 3.12/3.13, but 3.9 is
what we certify — float summation differs between them, and we would rather name the interpreter
that produced the score.

```bash
git clone <this repository>
cd TikTok-TechJam-2026-Question-4
python3 --version          # expect 3.9.6
```

**There is nothing to install.** `requirements.txt` is deliberately empty of packages; it exists so
that "this submission needs nothing" is a stated fact rather than a missing file. `starter/` imports
only `collections`, `copy`, `dataclasses`, `json`, `math`, `pathlib`, `re`, `sqlite3`, `statistics`,
`time` and `typing`.

**One file must be supplied.** `data/catalog.jsonl` (50,000 products, ~60 MB) is gitignored, as the
organizer's kit intends. Download `catalog.jsonl.gz` from the competition GitHub Release and
decompress it into `data/`:

```bash
gunzip -c catalog.jsonl.gz > data/catalog.jsonl
wc -l data/catalog.jsonl   # expect 50000
```

`data/public_set.jsonl` (200 labelled development sessions) ships in the repository.

No environment variables. No credentials. No network access at any point.

---

## 3. Reproducing our results

### The headline number

```bash
python3 -m evaluator.local_evaluator
```

Writes `results.json` and prints the metric summary. Expect **`recommended_technical_score:
0.759659`** exactly — the value is pinned in `tools/config_guard.py` as
`COMMITTED_TECHNICAL_SCORE`, and several tools refuse to run if it has drifted. Takes about 30
seconds.

### The test suite

```bash
python3 -m unittest discover -s tests -t .     # 615 tests, ~32s
```

### The configuration guard

```bash
python3 -m tools.config_guard                  # instant
```

Checks that every `USE_` flag and every tunable constant in `starter/` is registered and undrifted,
and prints what it checked. It fails loudly rather than exiting 0 on nothing.

### The evidence behind each decision

Each checkpoint's claims are regenerated by its own tool rather than remembered in a comment. These
are slow — they run the full evaluator several times each:

```bash
python3 -m tools.phase16_integration   # ~4 min  the staged-enablement ladder
python3 -m tools.phase16_depth         # ~3 min  the rerank-depth gate
python3 -m tools.phase15_clarification # ~5 min  the question policy, all arms
python3 -m tools.phase14_reranker      # ~5 min  reranker ON/OFF vs placebo
python3 -m tools.disclosure            # ~1 min  latency, tokens, cost
```

`tools/phase7_ablation.py` and the remaining `phase9`–`phase13` tools regenerate the earlier rungs.
Thirteen tools in total; all exit 0.

### Determinism

```bash
for s in 0 1 12345; do
  PYTHONHASHSEED=$s python3 -m evaluator.local_evaluator --output /tmp/r$s.json
done
md5 /tmp/r0.json /tmp/r1.json /tmp/r12345.json   # three identical digests
```

### Clean-room check

The score does not depend on anything untracked. Exporting only the tracked files plus the catalog
and running the command above reproduces **0.759659 in 28.7 seconds**.

---

## 4. Limitations, and what we would do next

### What the agent cannot do

**No paraphrase or synonymy.** Retrieval is lexical. A shopper who says "warm" will not match a
catalog that says "insulated" unless another route happens to surface the product. This is the
single biggest structural gap, and it is why the reranker's plug-in seam exists.

**The remaining misses are a ranking problem, not a retrieval one.** Of 20 residual misses, **5** are
targets that never enter the 300-candidate pool and **15** are targets that sit in the pool and never
reach the scored top 10. Candidate recall is 195/200. More retrieval will not fix this; better
ordering would.

**One benchmark, one catalog.** Everything is measured on 200 public sessions drawn from a single
category. We report a bootstrapped estimate of **0.74–0.78** for the private 800, but every constant
was chosen with these 200 in view.

### What we know is a fact about the benchmark rather than about shopping

We would rather name these than have them found.

- **The harness structurally rewards vagueness.** `customer_reply` matches the wildcard `"other"`
  against a strict superset of what any specific question matches, on every turn, by construction.
  Asking `"other"` every turn scores **+0.025 higher** than our policy. We measured it, we did not
  take it, and we capped the open question at one per session instead.
- **The popularity prior is a fact about sampling.** The public set's targets sit at the 99.5th
  percentile of catalog review count, so a bestseller list is nearly an oracle *here*. On a
  counterfactual set with targets drawn uniformly the same prior measures **−0.0126**. We hold its
  weight an order of magnitude below the constraint weight and leave a large measured gain on the
  table.
- **Five separate metrics in this project turned out to be measuring the simulator.** The most
  expensive: the reranker was ranking partly on `still`, `exploring`, `key` and `requirement` —
  words from the evaluator's sentence templates, not from any shopper — which carried 31% of its
  scoring weight until they were stripped.

The two largest declined gains measure **+0.038** and **+0.025** individually, both statistically
established; they overlap, so they do not simply add. We think the refusals are the most defensible
thing here, but they are a deliberate cost and we would rather state the price than hide it.

### Given more time

1. **A trained sentence encoder in the reranker slot.** The interface exists, is tested against
   adversarial scorers, and degrades to the lexical path if a model fails to load. Dropping in a
   real encoder is the one change that would attack the synonymy gap directly.
2. **Held-out constant selection.** Our tuned constants were chosen on the same 200 sessions the
   score is reported on. With more sessions we would split them and pick on the held-out half.
3. **The 15 ranking failures.** These are targets the system retrieves and then buries. A learned
   ranker, or richer per-slot evidence, is where the next real gain is.
4. **A question policy that costs something.** On this harness asking is free, so our policy has no
   reason to stop. A deployment where questions cost patience needs a second gate — the mode
   classifier in `starter/strategy.py` is built and measured for exactly that and deliberately not
   wired in, because on this benchmark it could only subtract.

---

## 5. Team

Four members. Each owned one review perspective **and** contributed to the architecture and the
code; the perspectives are lanes of responsibility, not a division of authorship.

| Member | Lane | Contribution |
|---|---|---|
| **Kim Minjun** | **A — Implementation & Architecture** | Designed the six-stage pipeline and the frozen `contracts.py` types, implemented every checkpoint, and integrated all three review lanes' findings. Built the flag/constant registry and the staged-enablement ladder. |
| **Oh Changsung** | **B — Retrieval** | Owned the candidate pool: BM25 and category routes over SQLite FTS5, the RRF union and its cap. Drove the dense-retrieval evaluation that ended in a measured refusal, and the recovery analysis showing that asking genuinely *recovers* candidates rather than only reordering them. |
| **Jung Woohyeop** | **C — Ranking & Metrics** | Owned constraint scoring, the popularity prior, and the significance methodology. Ran the rerank-depth analysis that produced the final `RERANK_TOP_N = 200`, and the counterfactual sampling work that kept the popularity weight honest. |
| **Park Junseong** | **D — QA & Integration** | Owned correctness, robustness and reproducibility: contract conformance, session isolation, crash safety, MTTC, and scenario regression. Verified every checkpoint by execution rather than by review, which caught the irreproducible placebo control, the simulator words carrying 31% of the reranker's weight, and several checks that could not fail. |

Every finding in this repository was reported as OBSERVED / EXPECTED / ACTUAL / REPRODUCTION with a
severity, and closed by measurement rather than by discussion.

---

## Further reading

| Document | What it is |
|---|---|
| `SUBMISSION.md` | Setup, exact Python version, one run command, verification steps |
| `docs/method_and_limitations.md` | The full method and limitations report |
| `docs/performance_disclosure.md` | Measured latency, token usage, estimated model cost |
| `docs/starter_kit_README.md` | The organizer's original starter-kit README, unmodified |

Licensed data: see `DATA_ATTRIBUTION.md`. The catalog derives from Amazon Reviews 2023 (McAuley
Lab, UCSD) and is used under the source dataset's terms.
