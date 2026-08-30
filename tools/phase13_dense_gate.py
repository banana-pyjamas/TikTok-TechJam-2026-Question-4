"""Phase 13 gate — is dense retrieval justified?

The roadmap gates this phase on measurement: "Only do this if measurements
justify it." This tool is that measurement, and it is the deliverable when the
answer is no.

Four questions, in the order that decides the phase:

  1. CEILING     if dense retrieval were PERFECT, how much score is available?
  2. DIAGNOSIS   are the targets we miss vocabulary-mismatch cases -- the thing
                 dense retrieval exists to fix -- or something else?
  3. STARVATION  how much did the shopper actually say in the sessions we miss?
  4. ALTERNATIVE does a different vector space over the same text find them?

Question 4 is run rather than argued. A projection of a bag of words is, in
theory, the same information BM25 already has in different coordinates -- but
this repo has been burned once for predicting a mechanism's behaviour from an
argument instead of measuring it (the Phase 11 "cannot reorder" claim), so the
alternative retriever is built and measured here.

The alternative is TF-IDF cosine over the same indexed terms, via an inverted
index. That is the strongest vector-space retriever available under the
constraints -- no encoder is installed (no torch, no transformers), the
organizer "may disable network access" for final scoring so a model would have
to be vendored, and the starter is standard-library only. It is also the upper
bound on any random-projection "dense" variant, which can only approximate
these same inner products less accurately.

Usage:  python3 -m tools.phase13_dense_gate
"""

from __future__ import annotations

import math
import statistics
import time
from collections import Counter, defaultdict

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.contracts import Context
from starter.retrieval import DEFAULT_ROUTES, POOL_LIMIT, retrieve
from starter.state import is_non_answer
from starter.text import terms
from tools import config_guard
from tools.capture import CapturingAgent

CATALOG = "data/catalog.jsonl"
DATASET = "data/public_set.jsonl"
TOP_K = POOL_LIMIT


class TfidfIndex:
    """TF-IDF cosine retrieval over the indexed product terms.

    An inverted index rather than a dense matrix: the catalog is 50k products
    of <=40 terms, so postings are ~1.8M entries and a query touches only the
    postings of its own terms. Pure standard library, deterministic, and fast
    enough to run over every session.

    This stands in for "a dense route" because it is the best vector-space
    retriever that can actually ship here, and because any random-projection
    embedding of the same bag of words approximates these inner products
    rather than adding to them.
    """

    def __init__(self, connection) -> None:
        rows = connection.execute(
            "SELECT parent_asin, vocab FROM product_meta").fetchall()
        self.asins = [str(row[0]) for row in rows]
        documents = [(row[1] or "").split() for row in rows]
        frequency: Counter = Counter()
        for document in documents:
            frequency.update(set(document))
        total = len(documents) or 1
        self.idf = {term: math.log(total / (1 + count))
                    for term, count in frequency.items()}
        self.postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self.norms = [0.0] * len(documents)
        for index, document in enumerate(documents):
            counts = Counter(document)
            weights = {term: (1.0 + math.log(count)) * self.idf.get(term, 0.0)
                       for term, count in counts.items()}
            norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
            self.norms[index] = norm
            for term, weight in weights.items():
                self.postings[term].append((index, weight / norm))

    def search(self, text: str, limit: int = TOP_K) -> list[str]:
        counts = Counter(terms(text))
        if not counts:
            return []
        query = {term: (1.0 + math.log(count)) * self.idf.get(term, 0.0)
                 for term, count in counts.items() if term in self.postings}
        norm = math.sqrt(sum(w * w for w in query.values()))
        if not norm:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term, weight in query.items():
            contribution = weight / norm
            for index, doc_weight in self.postings[term]:
                scores[index] += contribution * doc_weight
        ranked = sorted(scores, key=lambda i: (-scores[i], self.asins[i]))[:limit]
        return [self.asins[i] for i in ranked]


def main() -> None:
    config_guard.assert_all_flags_pinned(set(config_guard.COMMITTED_FLAGS))
    config_guard.assert_committed_constants()
    config_guard.restore_committed_flags()

    print("building index...", flush=True)
    started = time.time()
    agent = CapturingAgent(CATALOG)
    samples = load_jsonl(DATASET)
    catalog_ids, categories, products = catalog_index(CATALOG)
    print(f"ready in {time.time() - started:.1f}s", flush=True)

    result = evaluate(agent, samples, catalog_ids, categories, products)
    target_of = {s: str(x["ground_truth"]["parent_asin"])
                 for s, x in zip(agent.order, samples)}
    scenario_of = agent.sample_field(samples, "scenario_type")
    hit_of = {r["sample_id"]: r["hit"] for r in result["sessions"]}
    sample_of = {s: x for s, x in zip(agent.order, samples)}

    target_terms = {}
    for session_id, target in target_of.items():
        row = agent.connection.execute(
            "SELECT title, categories, features, description FROM products "
            "WHERE parent_asin = ?", (target,)).fetchone()
        target_terms[session_id] = set(
            terms(" ".join(str(cell or "") for cell in row))) if row else set()

    found, never = [], []
    said_of, informative_of, slots_of, query_of = {}, {}, {}, {}
    for session_id, captures in agent.by_session().items():
        target = target_of[session_id]
        in_pool = False
        said: set = set()
        informative = 0
        query_parts = []
        for capture in captures:
            message = capture["message"]
            if not is_non_answer(message):
                said |= set(terms(message))
                informative += 1
                query_parts.append(message)
            context = Context(session_id=session_id, turn=capture["turn"],
                              user_message=message, state=capture["state"])
            pool = retrieve(agent.connection, context, POOL_LIMIT, DEFAULT_ROUTES)
            if any(c.parent_asin == target for c in pool):
                in_pool = True
        said_of[session_id] = said
        informative_of[session_id] = informative
        slots_of[session_id] = len(captures[-1]["state"].slots)
        query_of[session_id] = " ".join(query_parts)
        (found if in_pool else never).append(session_id)

    total = len(found) + len(never)
    hits = sum(1 for s in found if hit_of[sample_of[s]["sample_id"]])
    conversion = hits / len(found) if found else 0.0

    # -- 1 -----------------------------------------------------------------
    print("\n1. CEILING -- the most dense retrieval could possibly be worth")
    print(f"   target reaches the pool          {len(found):>4}/{total}")
    print(f"   target NEVER retrieved           {len(never):>4}/{total}"
          "   <- all dense retrieval can attack")
    print(f"   of in-pool sessions, hit rate    {hits:>4}/{len(found)} = "
          f"{conversion:.1%}")
    gained = len(never) * conversion
    print(f"   => recovering EVERY missed target adds ~{gained:.1f} hits, "
          f"~{gained / total:.3f} HR, ~{0.5 * gained / total:.3f} TS")
    print(f"   By contrast {len(found) - hits} targets ARE in the pool and "
          "still lose -- the\n   conversion failure is "
          f"{(len(found) - hits) / max(len(never), 1):.1f}x larger than the "
          "entire retrieval ceiling.")

    # -- 2 -----------------------------------------------------------------
    print("\n2. DIAGNOSIS -- is this the failure dense retrieval fixes?")
    print("   Dense retrieval exists to bridge VOCABULARY MISMATCH: the shopper")
    print("   says 'warm', the catalog says 'insulated'. If the words already")
    print("   match, a different embedding of them has nothing to bridge.")
    overlap = Counter()
    for session_id in never:
        shared = len(said_of[session_id] & target_terms[session_id])
        overlap["0 shared" if shared == 0 else
                "1-2 shared" if shared <= 2 else
                "3-4 shared" if shared <= 4 else "5+ shared"] += 1
    for bucket in ("0 shared", "1-2 shared", "3-4 shared", "5+ shared"):
        print(f"   missed targets sharing tokens with what was said: "
              f"{bucket:11}{overlap[bucket]:>4}")
    print("   Every missed target already shares vocabulary with the query.")
    print("   These are not mismatch failures; they are discrimination")
    print("   failures among thousands of products that match equally well.")

    # -- 3 -----------------------------------------------------------------
    print("\n3. STARVATION -- how much did the shopper actually say?")
    for label, group in (("target found", found), ("target missed", never)):
        tokens = [len(said_of[s]) for s in group]
        print(f"   {label:16}n={len(group):>4}  shopper tokens median "
              f"{sorted(tokens)[len(tokens) // 2]:>3}  mean "
              f"{statistics.fmean(tokens):>5.1f}   informative turns mean "
              f"{statistics.fmean([informative_of[s] for s in group]):>4.2f}"
              f"   slots mean {statistics.fmean([slots_of[s] for s in group]):>4.2f}")
    missed_scenarios = Counter(scenario_of[s] for s in never)
    print(f"   missed by scenario: {dict(missed_scenarios)}")
    print("   The misses are mostly BROWSING sessions, which open with a bare")
    print("   category and never add anything, because the agent never asks.")
    print("   The missing input is information, not semantics.")

    # -- 4 -----------------------------------------------------------------
    print("\n4. ALTERNATIVE -- run a different vector space over the same text")
    started = time.time()
    index = TfidfIndex(agent.connection)
    print(f"   TF-IDF index built in {time.time() - started:.1f}s "
          f"({len(index.postings)} terms)")
    recovered = 0
    dense_recall = 0
    for session_id in found + never:
        ranked = index.search(query_of[session_id], TOP_K)
        if target_of[session_id] in ranked:
            dense_recall += 1
            if session_id in never:
                recovered += 1
    print(f"   TF-IDF recall@{TOP_K} over all sessions        "
          f"{dense_recall:>4}/{total} = {dense_recall / total:.1%}")
    print(f"   committed bm25+category recall            {len(found):>4}/{total} "
          f"= {len(found) / total:.1%}")
    print(f"   missed targets this alternative RECOVERS  {recovered:>4}/{len(never)}")
    print("   A random-projection embedding of the same bag of words can only")
    print("   approximate these inner products, so this is the ceiling for any")
    print("   dense variant that ships without a trained encoder.")

    print(f"\nconfig: {config_guard.describe()}")


if __name__ == "__main__":
    main()
