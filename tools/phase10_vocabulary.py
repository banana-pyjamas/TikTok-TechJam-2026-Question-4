"""Phase 10 — candidate-scoped vocabulary, measured on the live dialogue.

MEASUREMENT ONLY. The vocabulary layer is not wired into ``agent.respond``, so
this cannot change the score and the tool asserts that it did not.

What it answers, in the order the checkpoints ask it:

  CP 10.1  how big is a real turn's vocabulary, and what is in it
  CP 10.2  how often is a pool empty or unindexed in practice
  CP 10.3  how much does noise control remove, and for which reason
  CP 10.4  how much of what a shopper actually says can be grounded in the
           candidates in front of them

The last one is the number that decides whether Phase 15 can lean on this.
Grounding coverage is measured against the shopper's OWN words -- the active
evidence the state manager distilled -- not against a wordlist chosen here,
because a grounding map scored against its own vocabulary would score 100%
and mean nothing.

Usage:  python3 -m tools.phase10_vocabulary
"""

from __future__ import annotations

import time
from collections import Counter

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.contracts import Context
from starter.retrieval import DEFAULT_ROUTES, POOL_LIMIT, retrieve
from starter.text import terms as tokenize
from starter.vocabulary import (GROUNDING, VOCABULARY_LIMIT, build_vocabulary,
                                ground, most_discriminative)
from tools import config_guard
from tools.capture import CapturingAgent

CATALOG = "data/catalog.jsonl"
DATASET = "data/public_set.jsonl"

# A worked example per CP 10.4, run against whatever pool the real retrieval
# layer returns for it.
EXAMPLES = (
    ("I'm looking for a winter jacket, something warm", "warm"),
    ("I'm looking for running shoes, something lightweight", "lightweight"),
    ("I'm looking for a rain coat, waterproof", "waterproof"),
)


def _percentiles(values: list[int]) -> tuple[float, int, int]:
    ordered = sorted(values)
    mean = sum(ordered) / len(ordered)
    median = ordered[len(ordered) // 2]
    p90 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))]
    return mean, median, p90


def _evidence_tokens(state) -> list[str]:
    """The shopper's own still-active free text, tokenized."""
    out: list[str] = []
    for entry in getattr(state, "evidence", ()) or ():
        if isinstance(entry, dict) and entry.get("status") == "active":
            normalized = entry.get("normalized", "")
            if isinstance(normalized, str):
                out.extend(tokenize(normalized))
    return out


def main() -> None:
    config_guard.assert_all_flags_pinned(set(config_guard.COMMITTED_FLAGS))
    config_guard.assert_committed_constants()
    config_guard.restore_committed_flags()

    print("building index...", flush=True)
    started = time.time()
    agent = CapturingAgent(CATALOG)
    index_seconds = time.time() - started
    samples = load_jsonl(DATASET)
    catalog_ids, categories, products = catalog_index(CATALOG)
    print(f"ready in {index_seconds:.1f}s", flush=True)

    started = time.time()
    result = evaluate(agent, samples, catalog_ids, categories, products)
    turns = len(agent.captured)
    print(f"captured {turns} turns over {len(agent.order)} sessions "
          f"in {time.time() - started:.0f}s")
    print(f"score during capture: HR {result['hit_rate_at_10']:.4f} "
          f"MRR {result['mrr']:.6f} TS "
          f"{result['recommended_technical_score']:.6f}")
    if abs(result["recommended_technical_score"] - 0.134566) > 1e-9:
        raise SystemExit(
            "the vocabulary layer changed the score. It is not wired into "
            "agent.respond, so this should be impossible -- investigate "
            "before trusting anything below.")
    print("   unchanged from the committed score: the vocabulary layer is "
          "built and\n   measured here, but nothing in agent.respond calls it "
          "(Phase 15 is the consumer).")

    # -- replay: build the vocabulary for every real turn ------------------
    started = time.time()
    sizes: list[int] = []
    pool_sizes: list[int] = []
    dropped_total: Counter = Counter()
    kept_total = 0
    empty_pools = 0
    below_floor = 0
    grounded_tokens = 0
    identity_only = 0
    mapped_tokens = 0
    total_tokens = 0
    turns_with_evidence = 0
    turns_with_a_grounding = 0
    identity_examples: Counter = Counter()
    mapped_examples: Counter = Counter()
    query_seconds = 0.0

    for session_id, captures in agent.by_session().items():
        for capture in captures:
            context = Context(
                session_id=session_id,
                turn=capture["turn"],
                user_message=capture["message"],
                state=capture["state"],
            )
            pool = retrieve(agent.connection, context, POOL_LIMIT, DEFAULT_ROUTES)
            clock = time.time()
            vocabulary = build_vocabulary(agent.connection, pool)
            query_seconds += time.time() - clock

            pool_sizes.append(vocabulary["pool_size"])
            sizes.append(len(vocabulary["terms"]))
            kept_total += len(vocabulary["terms"])
            dropped_total.update(vocabulary["dropped"])
            if vocabulary["pool_size"] == 0:
                empty_pools += 1
            elif vocabulary["pool_size"] < 8:
                below_floor += 1

            tokens = _evidence_tokens(capture["state"])
            if tokens:
                turns_with_evidence += 1
                hit = False
                for token in tokens:
                    total_tokens += 1
                    result_terms = ground(token, vocabulary)
                    if not result_terms:
                        continue
                    grounded_tokens += 1
                    hit = True
                    # Two very different mechanisms hide behind one number.
                    # Identity: the pool already uses the shopper's word, and
                    # GROUNDING contributed nothing. Mapped: the shopper's word
                    # is not the catalog's, and the map bridged it -- which is
                    # the only part CP 10.4 is actually claiming.
                    bridged = [word for word in result_terms
                               if word != token.strip().lower()]
                    if bridged:
                        mapped_tokens += 1
                        mapped_examples[token] += 1
                    else:
                        identity_only += 1
                        identity_examples[token] += 1
                turns_with_a_grounding += int(hit)

    print(f"replayed {turns} turns in {time.time() - started:.0f}s "
          f"(vocabulary build alone: {query_seconds:.1f}s, "
          f"{1000 * query_seconds / turns:.2f} ms/turn)\n")

    # -- CP 10.1 ----------------------------------------------------------
    mean, median, p90 = _percentiles(sizes)
    pool_mean, pool_median, pool_p90 = _percentiles(pool_sizes)
    print("CP 10.1  vocabulary size per turn")
    print(f"   terms      mean {mean:>7.1f}   median {median:>5}   "
          f"P90 {p90:>5}   max {max(sizes):>5}   cap {VOCABULARY_LIMIT}")
    print(f"   pool       mean {pool_mean:>7.1f}   median {pool_median:>5}   "
          f"P90 {pool_p90:>5}   max {max(pool_sizes):>5}")

    # -- CP 10.2 ----------------------------------------------------------
    print("\nCP 10.2  degenerate pools encountered on the real dialogue")
    print(f"   empty pools (vocabulary empty, no crash)       {empty_pools:>6}")
    print(f"   pools below the noise floor (bounds skipped)   {below_floor:>6}")
    print(f"   turns where the vocabulary came back empty     "
          f"{sum(1 for size in sizes if size == 0):>6}")

    # -- CP 10.3 ----------------------------------------------------------
    total_seen = kept_total + sum(dropped_total.values())
    print(f"\nCP 10.3  noise control, summed over {turns} turns "
          f"({total_seen} term observations)")
    print(f"   kept                          {kept_total:>9}"
          f"{kept_total / total_seen:>9.1%}")
    for reason, label in (("rare", "dropped: in one candidate"),
                          ("ubiquitous", "dropped: in most candidates"),
                          ("over_limit", "dropped: over the cap")):
        count = dropped_total[reason]
        print(f"   {label:30}{count:>9}{count / total_seen:>9.1%}")

    # -- CP 10.4 ----------------------------------------------------------
    print("\nCP 10.4  grounding the shopper's own words in their own pool")
    if total_tokens:
        print(f"   any grounding                 {grounded_tokens:>9}"
              f"{grounded_tokens / total_tokens:>9.1%}  of {total_tokens} "
              "evidence tokens")
        print(f"     of which identity only      {identity_only:>9}"
              f"{identity_only / total_tokens:>9.1%}  the pool already uses "
              "the shopper's word")
        print(f"     of which MAPPED             {mapped_tokens:>9}"
              f"{mapped_tokens / total_tokens:>9.1%}  <- the only part "
              "GROUNDING earns")
    print(f"   turns with any grounding      {turns_with_a_grounding:>9}"
          f"{(turns_with_a_grounding / turns_with_evidence if turns_with_evidence else 0):>9.1%}"
          f"  of {turns_with_evidence} turns carrying evidence")
    print("   top identity words: "
          + ", ".join(f"{word} ({count})"
                      for word, count in identity_examples.most_common(8)))
    print("   top mapped words:   "
          + ", ".join(f"{word} ({count})"
                      for word, count in mapped_examples.most_common(8)))
    print(f"   Read the MAPPED row, not the headline. The headline is carried "
          f"by category\n   words the catalog already uses ('shirts', "
          "'blouses'), which need no map --\n   quoting it as evidence for the "
          "map would be attributing one mechanism's\n   result to another. "
          f"The map has {len(GROUNDING)} entries; coverage below 100% is "
          "correct, since\n   a word the candidates have no expression for "
          "must ground to nothing.")

    # -- worked examples --------------------------------------------------
    print("\nworked examples (fresh session, real retrieval, real pool)")
    from starter.contracts import SessionState
    from starter.state import update_state

    for message, word in EXAMPLES:
        state = SessionState(session_id="ex")
        update_state(state, message, 1)
        context = Context(session_id="ex", turn=1, user_message=message,
                          state=state)
        pool = retrieve(agent.connection, context, POOL_LIMIT, DEFAULT_ROUTES)
        vocabulary = build_vocabulary(agent.connection, pool)
        print(f"   {word:12} -> {', '.join(ground(word, vocabulary)) or '(nothing)'}")
        print(f"   {'':12}    pool {vocabulary['pool_size']}, "
              f"{len(vocabulary['terms'])} terms; would ask about: "
              f"{', '.join(most_discriminative(vocabulary, 6))}")

    print(f"\ncost: index build {index_seconds:.1f}s "
          f"(+~1.8s for the vocab column), ~11.5 MB of side table, "
          f"{1000 * query_seconds / turns:.2f} ms per turn when a consumer "
          "calls it")
    print(f"config: {config_guard.describe()}")


if __name__ == "__main__":
    main()
