# Method, model choice, and limitations

Required by `docs/submission_rules.md` ("a short report describing method,
model choice, and limitations"). Written to be read by someone who has not
seen the code.

## Method

A deterministic six-stage turn. No model, no network, no randomness.

```
user message
  -> state manager        deterministic slot extraction + free-text evidence
  -> retrieval            BM25 + category routes over SQLite FTS5, RRF union
  -> constraint ranking   match / violation / unknown per slot, + popularity
  -> lexical reranking    in-pool IDF over the shopper's own free text
  -> clarification        which attribute to ask about, if any
  -> response             10 recommendations AND a question, same turn
```

Each stage is behind an ablation flag and each was measured OFF vs ON on the
200-session public set. The ladder (`python3 -m tools.phase7_ablation`):

| run | TS | delta |
| --- | --- | --- |
| baseline (weak BM25) | 0.106710 | — |
| + deterministic state | 0.106710 | +0.000000 |
| + multi-route retrieval | 0.119431 | +0.012721 |
| + constraint ranking | 0.134566 | +0.015135 |
| + popularity prior | 0.182258 | +0.047692 |
| + lexical reranker | 0.243930 | +0.061672 |
| **+ clarification (ships)** | **0.722979** | **+0.479049** |

### Staged integration, and what it decided

`python3 -m tools.phase16_integration` enables the features one at a time in
the roadmap's order, gates each rung on both significance tests against the
rung below it, and **reverts any rung that does not establish a gain**. Four
features ship OFF or do not exist, and each is a measured decision rather
than an omission:

| step | feature | outcome |
| --- | --- | --- |
| 1 | Core only | baseline, TS 0.134566 |
| 2 | Profile | **reverted** — +0.0077, 7/5, p = 0.77; CI straddles zero |
| 3 | Adaptive Strategy | no flag — worth +0.000298, deleted in Phase 9 |
| 4 | Candidate Vocabulary | not separable — it *is* the reranker's index |
| 5 | EC/MR | **reverted** — exactly inert on core, 0/0 discordant |
| 6 | Popularity | kept — +0.047692, 10/0, p = 0.0020 |
| 7 | Dense | not implemented — measured worse, Phase 13 |
| 8 | Semantic Reranker | kept — +0.061672, 15/0, p = 0.0001 |
| 9 | Clarification | kept — +0.479049, 114/1, p = 0.0000 |

The ladder's end state matches the committed flags on all eight and
reproduces `COMMITTED_TECHNICAL_SCORE` exactly, so no flag ships on history
or intuition. Whole ladder: **+0.588413**, 139/1 discordant, p = 0.0000,
bootstrap CI [+0.532, +0.642].

## Model choice

**There is no model.** That is a choice, not an omission, and it was made
against three constraints stated in the organizer's own rules:

1. `docs/submission_rules.md` warns that "organizer policy may disable
   network access" for official scoring. A submission whose main path needs
   an API is a submission that may score zero.
2. Nothing may depend on undeclared external services for final scoring.
3. The bundle must be reproducible from the submitted files alone.

A deterministic agent satisfies all three trivially. The cost is that no
stage can generalise beyond what a rule can express, and the sections below
say where that cost lands.

A semantic reranker WAS built for a model and ships without one:
`starter/reranker.py` defines `load_encoder_scorer`, which returns `None` on
this machine, and the whole CP 14.1–14.5 fallback contract exists so an
encoder can be dropped in later without touching the pipeline. Phase 13
measured the strongest vector retriever that can actually ship here — TF-IDF
cosine, lexical, not semantic — and it made the candidate pool *worse*
(−11 sessions at recall@50, p = 0.0074), so it was rejected on measurement.
What a trained encoder would be worth here is **not measured anywhere** and
is not claimed.

## Limitations

### The benchmark structurally favours the least specific question

`local_evaluator.customer_reply` filters undisclosed constraints with
`attribute == "other" or classify_constraint(value) == attribute`. So asking
`"other"` matches a strict *superset* of what any specific question matches,
on every turn, by construction. Any comparison of question policies on this
harness is therefore scored under a rule that rewards vagueness, and no
amount of measuring here can separate "the wildcard is the best question"
from "the wildcard is the only question the simulator rewards". The shipped
policy bounds its use of the open question to once per session and reports
what that costs rather than maximising against the rule.

### The popularity prior is a fact about how the set was sampled

The public set's targets sit at the 99.5th percentile of catalog review
count (median 7,078 against a catalog median of 12; 4 of 200 targets are
below the catalog median, where unbiased would be 100). On this benchmark a
bestseller list is close to an oracle. The private set is built the same way,
so the gain should transfer — but it is **not** evidence that ranking by
review count serves shoppers, and on a counterfactual set with targets drawn
uniformly the same prior measures −0.012556. The weight is held an order of
magnitude below the constraint weight (`W_POPULARITY` 0.008 vs `W_MATCH`
0.10) for that reason, leaving a large measured gain on the table: raising
it to 0.10 pays enormously here and would be fitting the sampling.

### Several measured numbers are measurements of the simulator

The evaluator writes the shopper's turns from the target product's own
fields and from four sentence templates. Five separate metrics in this
project have turned out to be measuring that rather than the agent, and each
is recorded where it was found. The most expensive: the reranker was ranking
partly on `still`, `exploring`, `key` and `requirement` — words from the
templates, not from any shopper — which carried 31% of its scoring weight
until they were stripped.

### What the agent cannot do

- **No paraphrase or synonymy.** Retrieval is lexical. A shopper who says
  "warm" will not match a catalog that says "insulated" unless some other
  route surfaces the product.
- **No reasoning about the message.** Slot extraction is regex and
  vocabulary driven. Novel phrasings fall through to free-text evidence,
  which the reranker uses but the constraint ranker does not.
- **Six checkable attributes.** Colour, material, category, brand, size and
  budget are the only things the catalog can verify. Style, feature and
  use-case answers are kept as evidence and reach the ranking only through
  the reranker.
- **Fixed response text.** `message` is one constant string. The agent
  participates through `ask_attribute`, not through prose.
- **Catalog coverage bounds everything.** 47% of products declare a colour,
  63% a material, 21% a price, 10% a size. A constraint on a field the
  catalog omits is UNKNOWN, never a violation — which is deliberate, and
  means such a constraint cannot discriminate.

### Asking recovers candidates -- measured, not assumed

The obvious story for why clarification is worth +0.48 is that a disclosed
constraint enters the query and pulls the target into the pool. That is a
retrieval claim and it was untested until `tools/phase15_recovery.py`; the
competing explanation (the constraint only re-ranks a pool that already
contained the target) fits the same score.

The story holds. Of the 67 sessions where the target was not yet in the pool
when the first informative answer arrived, **62 recovered (92.5%)**, at a
median latency of **0 turns** -- 37 of them appear in the pool on the very
turn the answer lands, because that turn's query already carries it.
Override sessions recover within one turn of the override in 67% of cases
and fail in 1 of 30.

### What the residual actually is

30 sessions still miss. Of those, **5 are retrieval failures** (the target
never reaches the 300-candidate pool on a scoring-eligible turn) and **25 are
ranking failures** (it reaches the pool and never the top 10). The remaining
headroom is a ranking problem, not a retrieval one, and the downstream delay
confirms it: the ranker holds a target it already has for a mean of 1.65
turns before showing it.

None of the 5 retrieval misses is unreachable. Classified by re-running the
routes at depth 5000, every one is either a query miss (the words that turn
did not match) or a cap miss (the routes found it, `POOL_LIMIT` cut it); no
target is absent from the index. `python3 -m tools.phase15_recovery`
reproduces all of this.

## Reproducibility and determinism

- No randomness in the shipped path. `results.json` is byte-identical across
  `PYTHONHASHSEED` 0, 1 and 12345.
- Every tunable is registered in `tools/config_guard.py` and checked before
  any measurement runs; `python3 -m tools.config_guard` verifies the whole
  configuration and fails loudly.
- 607 unit tests, run on the certified Python 3.9.6.
- Every measured number in the source comments names the tool that
  regenerates it.

## Honest summary

The strongest claims this submission can make are that it is fully
deterministic, needs nothing installed, cannot fail for want of a network,
and that every component was measured rather than assumed — including the
four that were measured and then **rejected** (profile prior, confidence
weighting, TF-IDF retrieval, mode-adaptive routing). The weakest part is that
its largest single gain comes from a benchmark whose simulated shopper only
speaks when asked, and whose reward for asking well over asking vaguely is
something this benchmark cannot express.
