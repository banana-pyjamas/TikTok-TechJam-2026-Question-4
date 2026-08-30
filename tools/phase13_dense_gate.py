"""Phase 13 gate -- is a dense retrieval route justified?

The roadmap gates this phase on measurement: "Only do this if measurements
justify it." This tool is that measurement, and it is the deliverable when the
answer is no.

WHAT CHANGED AFTER REVIEW B/D
-----------------------------
The first version of this tool answered the question but overstated its own
evidence in four places. The corrections are structural, not editorial:

B1/D-R3  CEILING.  "38 of 200 never retrieved" is a fact about CANDIDATE
         RECALL: the headroom is +0.19 recall, and that is the number
         retrieval owns. The old "~+0.025 TS" was a downstream extrapolation
         that moved the HitRate term ONLY, while TS = 0.5·HR + 0.3·MRR +
         0.2·eff and a recovered hit moves all three. So the TS consequence
         is now (a) extrapolated across all three terms and labelled as an
         extrapolation, and (b) MEASURED, by re-running the evaluator with an
         oracle route that hands retrieval the answer. Two oracle positions
         bracket it: the target injected at the FLOOR of the pool, and at the
         HEAD. Perfect retrieval cannot be worth more than the head arm.

B2/D-R1  DIAGNOSIS.  The old "zero missed targets share NO vocabulary" was a
         measurement of the SIMULATOR, not of retrieval.
         ``evaluator/local_evaluator.py`` builds the shopper's opening line
         out of ``coarse_category(target.categories)`` -- the target's own
         taxonomy string -- and the gate then counted the target's categories
         field among its terms. Overlap >= 1 is an identity, not a finding.
         It is now measured three ways (full text / taxonomy field removed /
         title only) against a CONTROL group (the sessions retrieval finds),
         and the tool reports whether the metric discriminates at all. It
         does not, under the original definition.

         Nor would a zero count have licensed the old conclusion. "No
         pure vocabulary-mismatch cases were observed" is what the data can
         say. It is not the same claim as "semantic retrieval cannot help",
         and that claim is not made here.

B3/D-R5  ALTERNATIVE.  TF-IDF cosine is a LEXICAL vector-space retriever. It
         is not a semantic dense retriever, and it is not an upper bound on a
         trained encoder -- a trained encoder places related-but-disjoint
         vocabulary near each other, which no reweighting of the same terms
         can do. It is reported here as what it is: the strongest retriever
         that can actually ship under the constraints. The old run also
         indexed ``product_meta.vocab`` (32.8 terms/product, title-first,
         capped at 40) while calling it "the same indexed terms" as BM25
         (87.6 terms/product across six weighted columns). Both are built
         now, and the STRONGER one is carried into the ablation, because the
         claim under test is that TF-IDF does not help.

B4       COMPLEMENTARITY.  Standalone recall cannot answer "does adding this
         route help" -- a worse retriever can still contribute unique
         candidates. Four arms are measured over the same captured dialogue,
         with the full pool-shape and per-scenario breakdown, and the union
         goes through the shipped RRF ``fuse`` with route provenance intact.

Nothing here changes production. ``DEFAULT_ROUTES`` is pinned by
``tools.config_guard`` and the tool refuses to run if it has drifted.

Usage:  python3 -m tools.phase13_dense_gate     (~6 min: three evaluator runs,
        two lexical indexes, and a four-arm replay over every captured turn)
"""

from __future__ import annotations

import importlib
import math
import sys
import time
from array import array
from collections import Counter, defaultdict

from evaluator.local_evaluator import (catalog_index, coarse_category,
                                       evaluate, load_jsonl)
from starter import agent as agent_module
from starter.agent import Agent
from starter.contracts import Candidate, Context
from starter.retrieval import DEFAULT_ROUTES, POOL_LIMIT, ROUTES, fuse
from starter.state import is_non_answer
from starter.text import terms
from tools import config_guard
from tools.capture import CapturingAgent
from tools.significance import format_test, mcnemar
from tools.summaries import percentiles

CATALOG = "data/catalog.jsonl"
DATASET = "data/public_set.jsonl"
K_VALUES = (50, 100, 300)

# BM25's column weights (``retrieval._BM25_RANK``), reused as term-frequency
# multipliers so the lexical vector arm reads the catalog with the same
# emphasis the committed retriever does. Without this the comparison would
# handicap the alternative on document representation rather than on the
# scoring function, which is not the question.
_COLUMN_WEIGHTS = (("title", 6.0), ("categories", 4.0), ("features", 2.5),
                   ("details", 2.5), ("store", 1.5), ("description", 1.0))

# The lexical-vector route's name in the fusion dict. Not registered in
# ``retrieval.ROUTES``: it is measured here, it does not ship.
TFIDF = "tfidf"

ARMS: dict[str, tuple[str, ...]] = {
    "bm25": ("bm25",),
    "bm25+category (committed)": ("bm25", "category"),
    "tfidf": (TFIDF,),
    "bm25+category+tfidf": ("bm25", "category", TFIDF),
}
COMMITTED_ARM = "bm25+category (committed)"
UNION_ARM = "bm25+category+tfidf"


class LexicalVectorIndex:
    """TF-IDF cosine over the catalog, as an inverted index.

    A LEXICAL vector space: every dimension is a term the catalog literally
    contains, so two products with disjoint wording are orthogonal here no
    matter how related they are. That is precisely the thing a trained dense
    encoder does differently, and precisely why this is a floor for "a
    different vector space over the same words", NOT a ceiling for "a dense
    retriever". The docstring says so because the commit that shipped the
    first version of this tool did not.

    ``source`` picks the document text:

      ``"full"``   the six FTS columns BM25 itself scores, with BM25's column
                   weights applied to the term frequencies (87.6 terms/doc).
      ``"vocab"``  ``product_meta.vocab`` -- the Phase 10 indexed terms,
                   title-first and capped at ``INDEX_TERM_LIMIT``
                   (32.8 terms/doc).

    Postings are ``array`` pairs rather than lists of tuples: 4.4M postings as
    Python tuples is ~400 MB, as two typed arrays it is ~35 MB, and the query
    loop iterates them at C speed through ``zip``.

    Deterministic: ties break on ``parent_asin``.
    """

    def __init__(self, connection, source: str) -> None:
        self.source = source
        self._cache: dict[str, list[tuple[str, float]]] = {}

        document_frequency: Counter = Counter()
        documents = 0
        for _, counts in self._documents(connection):
            document_frequency.update(counts.keys())
            documents += 1
        self.documents = documents
        self.idf = {term: math.log(documents / (1 + count))
                    for term, count in document_frequency.items()}

        self.asins: list[str] = []
        self.postings: dict[str, tuple[array, array]] = {}
        self.postings_count = 0
        self.terms_per_document = 0.0
        for parent_asin, counts in self._documents(connection):
            index = len(self.asins)
            self.asins.append(parent_asin)
            weights = {term: (1.0 + math.log(count)) * self.idf[term]
                       for term, count in counts.items()}
            norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
            for term, weight in weights.items():
                posting = self.postings.get(term)
                if posting is None:
                    posting = self.postings[term] = (array("i"), array("f"))
                posting[0].append(index)
                posting[1].append(weight / norm)
            self.postings_count += len(weights)
        self.terms_per_document = self.postings_count / max(documents, 1)

    def _documents(self, connection):
        """``(parent_asin, {term: weighted frequency})`` for every product.

        A generator run twice (document frequencies, then postings) rather
        than a materialized list: the full catalog text is ~240 MB of Python
        strings and there is no reason to hold it.
        """
        if self.source == "vocab":
            cursor = connection.execute(
                "SELECT parent_asin, vocab FROM product_meta ORDER BY parent_asin")
            for parent_asin, vocab in cursor:
                counts: dict[str, float] = defaultdict(float)
                for token in (vocab or "").split():
                    counts[token] += 1.0
                yield str(parent_asin), counts
            return
        cursor = connection.execute(
            "SELECT parent_asin, title, categories, features, details, store, "
            "description FROM products")
        for row in cursor:
            counts = defaultdict(float)
            for offset, (_, weight) in enumerate(_COLUMN_WEIGHTS, start=1):
                for token in terms(str(row[offset] or "")):
                    counts[token] += weight
            yield str(row[0]), counts

    def search(self, text: str, limit: int) -> list[tuple[str, float]]:
        """Top-``limit`` ``(parent_asin, cosine)``, highest first.

        Memoized on the query string. The shipped dialogue repeats a small set
        of non-answers verbatim from turn 2 onward ("Those options are not
        quite right yet..."), so a cache turns ~1800 turns into a few hundred
        distinct queries.
        """
        cached = self._cache.get(text)
        if cached is not None:
            return cached[:limit]
        counts = Counter(terms(text))
        query = {term: (1.0 + math.log(count)) * self.idf[term]
                 for term, count in counts.items() if term in self.postings}
        norm = math.sqrt(sum(weight * weight for weight in query.values()))
        if not norm:
            self._cache[text] = []
            return []
        scores: dict[int, float] = defaultdict(float)
        for term, weight in query.items():
            contribution = weight / norm
            indexes, values = self.postings[term]
            for index, value in zip(indexes, values):
                scores[index] += contribution * value
        ranked = sorted(scores, key=lambda i: (-scores[i], self.asins[i]))[:POOL_LIMIT]
        result = [(self.asins[i], scores[i]) for i in ranked]
        self._cache[text] = result
        return result[:limit]


# ---------------------------------------------------------------------------
# 1. CEILING -- measured, not extrapolated
# ---------------------------------------------------------------------------


class _OracleAgent(Agent):
    """The shipped agent, with the target's session id mapped for the oracle.

    ``reset`` is called once per sample in dataset order, which is the only
    bridge from the evaluator's opaque uuid back to the sample it came from.
    Sessions outside ``eligible`` are never mapped, so the oracle cannot reach
    them.
    """

    def __init__(self, catalog_path: str, samples: list[dict],
                 eligible: set[str]) -> None:
        super().__init__(catalog_path)
        self._samples = samples
        self._eligible = eligible
        self._reset_count = 0
        self.target_of: dict[str, str] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        super().reset(session_id, user_profile)
        sample = self._samples[self._reset_count]
        self._reset_count += 1
        if str(sample["sample_id"]) in self._eligible:
            self.target_of[session_id] = str(sample["ground_truth"]["parent_asin"])


def oracle_run(position: str, samples, eligible: set[str],
               catalog_ids, categories, products) -> dict:
    """Re-run the evaluator with retrieval handed the answer.

    This is the ceiling measured rather than argued. A retrieval route cannot
    do better than putting the target in the pool, so whatever the score does
    here bounds what ANY route -- dense, lexical, or otherwise -- could be
    worth downstream, GIVEN the ranking that ships.

    SCOPE MATTERS, and getting it wrong is how a ceiling gets inflated. The
    oracle fires ONLY on the ``eligible`` samples: the ones whose target the
    committed route set never retrieves on any turn. Those are the only
    sessions a better retriever could rescue, and they are exactly the +0.19
    recall headroom this section is about. Injecting into every session would
    also hand the ranker a rank-1 target on turns where retrieval had already
    delivered it on some other turn, which measures a perfect retriever AND a
    perfect within-session ordering -- a different, much larger claim.

    ``position`` decides where the injected candidate enters, which is the
    honest uncertainty in the measurement:

      ``"floor"``  ``fusion_score`` 0.0 -- it arrives at the bottom of the
                   pool and ranking has to lift it 300 places. A LOWER bound:
                   no real route would place a match there.
      ``"head"``   the pool's best ``fusion_score`` -- as if retrieval had
                   ranked it first. An UPPER bound.

    The dialogue is NOT held fixed here, on purpose: a session that hits ends
    early, which changes MTTC and therefore the efficiency term. That is a
    real consequence of better retrieval and belongs in the number.
    """
    agent = _OracleAgent(CATALOG, samples, eligible)
    original = agent_module.retrieve

    def oracle_retrieve(connection, context, limit=POOL_LIMIT, routes=None):
        pool = original(connection, context, limit, routes)
        target = agent.target_of.get(context.session_id)
        if target is None or any(c.parent_asin == target for c in pool):
            return pool
        if position == "head":
            score = max((c.metadata.get("fusion_score", 0.0) for c in pool),
                        default=1.0 / 61)
            injected = Candidate(parent_asin=target, route_scores={"oracle": 0.0},
                                 metadata={"fusion_score": score})
            return ([injected] + pool)[:limit]
        injected = Candidate(parent_asin=target, route_scores={"oracle": 0.0},
                             metadata={"fusion_score": 0.0})
        if len(pool) < limit:
            return pool + [injected]
        return pool[:limit - 1] + [injected]

    agent_module.retrieve = oracle_retrieve
    try:
        return evaluate(agent, samples, catalog_ids, categories, products)
    finally:
        agent_module.retrieve = original


def extrapolated_ceiling(result: dict, missed: int, total: int) -> dict:
    """The old "+0.025 TS", corrected to move all three TS terms.

    TS = 0.5·HR + 0.3·MRR + 0.2·efficiency. Recovering a target that never
    reached the pool adds a hit (HR), that hit's reciprocal rank (MRR), and
    a first-hit turn well inside the 11-turn miss charge (efficiency). The
    published figure moved HR alone, which understates its own claim by
    roughly a factor of two. Reported here as an EXTRAPOLATION -- it assumes
    recovered sessions behave like the sessions that already convert -- and
    is superseded by ``oracle_run``, which measures instead of assuming.
    """
    sessions = result["sessions"]
    hit_sessions = [s for s in sessions if s["hit"]]
    hits = len(hit_sessions)
    conversion = hits / (total - missed) if total > missed else 0.0
    recovered = missed * conversion
    mean_rr = (sum(s["reciprocal_rank"] for s in hit_sessions) / hits) if hits else 0.0
    mean_turn = (sum(s["first_hit_turn"] for s in hit_sessions) / hits) if hits else 11.0

    hit_rate = (hits + recovered) / total
    mrr = (sum(s["reciprocal_rank"] for s in sessions) + recovered * mean_rr) / total
    mttc_total = sum(s["first_hit_turn"] if s["first_hit_turn"] is not None else 11
                     for s in sessions)
    mttc = (mttc_total - recovered * (11 - mean_turn)) / total
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    return {
        "conversion": conversion,
        "recovered": recovered,
        "hit_rate": hit_rate,
        "mrr": mrr,
        "efficiency": efficiency,
        "technical_score": 0.5 * hit_rate + 0.3 * mrr + 0.2 * efficiency,
    }


# ---------------------------------------------------------------------------
# 4. COMPLEMENTARITY
# ---------------------------------------------------------------------------


def _tfidf_query(capture: dict) -> str:
    """The text the lexical vector route scores for one turn.

    The raw message plus every value the state manager has accumulated, which
    is strictly MORE than the bm25 route sees (raw message only) and about
    what the category route sees. The alternative is given the better query on
    purpose: the claim under test is that it does not help.
    """
    slots = capture["state"].slots
    values = [str(value)
              for slot in slots.values() if isinstance(slot, dict)
              for value in slot.get("values", ()) if str(value)]
    return " ".join([capture["message"], *values])


def recall(records: list[dict], arm: str, k: int) -> float:
    if not records:
        return 0.0
    hits = sum(1 for r in records
               if r["best_rank"][arm] is not None and r["best_rank"][arm] < k)
    return hits / len(records)


def _recalled(records: list[dict], arm: str, k: int) -> set[str]:
    return {r["session_id"] for r in records
            if r["best_rank"][arm] is not None and r["best_rank"][arm] < k}


def main() -> None:
    config_guard.assert_all_flags_pinned(set(config_guard.COMMITTED_FLAGS))
    config_guard.assert_committed_constants()
    config_guard.assert_committed_flags_match_source()
    config_guard.restore_committed_flags()
    assert ARMS[COMMITTED_ARM] == tuple(DEFAULT_ROUTES), (
        "this tool's committed arm must be retrieval.DEFAULT_ROUTES")
    assert TFIDF not in ROUTES, (
        "the lexical vector route is measured here, not shipped; it must not "
        "be registered in retrieval.ROUTES")

    print("building index...", flush=True)
    started = time.time()
    agent = CapturingAgent(CATALOG)
    samples = load_jsonl(DATASET)
    catalog_ids, categories, products = catalog_index(CATALOG)
    print(f"ready in {time.time() - started:.1f}s", flush=True)

    started = time.time()
    result = evaluate(agent, samples, catalog_ids, categories, products)
    technical_score = result["recommended_technical_score"]
    committed = config_guard.COMMITTED_TECHNICAL_SCORE
    drift = "" if abs(technical_score - committed) < 5e-7 else \
        f"   <- DRIFT from the committed {committed}"
    print(f"captured {len(agent.captured)} turns over {len(agent.order)} "
          f"sessions in {time.time() - started:.0f}s")
    print(f"score during capture: HR {result['hit_rate_at_10']:.4f} "
          f"MRR {result['mrr']:.6f} MTTC {result['mttc']:.3f} "
          f"TS {technical_score:.6f}{drift}")

    target_of = {session_id: str(sample["ground_truth"]["parent_asin"])
                 for session_id, sample in zip(agent.order, samples)}
    scenario_of = agent.sample_field(samples, "scenario_type")
    by_session = agent.by_session()
    total = len(by_session)

    # What the shopper said, what the catalog says, and the opening line the
    # simulator generated -- section 2 needs all three.
    said_of, opening_of, slots_of, informative_of = {}, {}, {}, {}
    for session_id, captures in by_session.items():
        said: set = set()
        informative = 0
        for capture in captures:
            if not is_non_answer(capture["message"]):
                said |= set(terms(capture["message"]))
                informative += 1
        said_of[session_id] = said
        opening_of[session_id] = captures[0]["message"]
        slots_of[session_id] = len(captures[-1]["state"].slots)
        informative_of[session_id] = informative

    # -- lexical vector indexes ---------------------------------------------
    print("\nbuilding lexical vector indexes...", flush=True)
    indexes = {}
    for source, label in (("vocab", "product_meta.vocab (Phase 10 terms)"),
                          ("full", "full FTS text, BM25 column weights")):
        started = time.time()
        index = LexicalVectorIndex(agent.connection, source)
        indexes[source] = index
        print(f"   {label:38} {index.terms_per_document:5.1f} terms/doc  "
              f"{len(index.postings):>7} distinct  {index.postings_count:>9} "
              f"postings  {time.time() - started:.0f}s", flush=True)

    # -- replay -------------------------------------------------------------
    # Same discipline as tools/phase9_retrieval_evidence.py: the dialogue is
    # held FIXED at the one the shipped agent actually produced, and only the
    # route set varies. Re-running the loop per arm would change the transcript
    # as well as the pool, and the two could not be separated.
    print("\nreplaying arms over the captured dialogue...", flush=True)
    started = time.time()

    # Which of the two lexical indexes gets carried into the ablation is
    # decided by measurement, on standalone session-level recall@300, and the
    # STRONGER one wins -- the claim under test is that this route does not
    # help, so it is given its best shot.
    standalone = {}
    for source, index in indexes.items():
        recalled = 0
        for session_id, captures in by_session.items():
            target = target_of[session_id]
            for capture in captures:
                query = _tfidf_query(capture)
                if any(asin == target
                       for asin, _ in index.search(query, POOL_LIMIT)):
                    recalled += 1
                    break
        standalone[source] = recalled / total
    chosen = max(standalone, key=lambda source: standalone[source])
    print(f"   standalone recall@{POOL_LIMIT}: "
          + ", ".join(f"{source} {value:.4f}" for source, value in standalone.items())
          + f"   -> carrying '{chosen}' into the ablation")
    index = indexes[chosen]

    records: list[dict] = []
    precap: dict[str, list[int]] = {arm: [] for arm in ARMS}
    final: dict[str, list[int]] = {arm: [] for arm in ARMS}
    unique_to_tfidf = 0
    union_members = 0
    union_with_tfidf = 0
    bad_provenance = 0

    for session_id, captures in by_session.items():
        target = target_of[session_id]
        record = {
            "session_id": session_id,
            "scenario": scenario_of[session_id],
            "best_rank": {arm: None for arm in ARMS},
        }
        for capture in captures:
            context = Context(session_id=session_id, turn=capture["turn"],
                              user_message=capture["message"],
                              state=capture["state"])
            per_route = {
                "bm25": ROUTES["bm25"](agent.connection, context, POOL_LIMIT),
                "category": ROUTES["category"](agent.connection, context, POOL_LIMIT),
                TFIDF: index.search(_tfidf_query(capture), POOL_LIMIT),
            }
            lexical_only = {asin for asin, _ in per_route[TFIDF]} - {
                asin for name in DEFAULT_ROUTES for asin, _ in per_route[name]}
            unique_to_tfidf += len(lexical_only)

            for arm, route_names in ARMS.items():
                selected = {name: per_route[name] for name in route_names}
                precap[arm].append(len({asin for rows in selected.values()
                                        for asin, _ in rows}))
                pool = fuse(selected, POOL_LIMIT)
                final[arm].append(len(pool))
                asins = [candidate.parent_asin for candidate in pool]
                if target in asins:
                    rank = asins.index(target)
                    current = record["best_rank"][arm]
                    record["best_rank"][arm] = rank if current is None \
                        else min(current, rank)
                if arm != UNION_ARM:
                    continue
                # B5: the union must go through the shipped fusion with
                # provenance intact, not through a hand-rolled merge.
                union_members += len(pool)
                for candidate in pool:
                    sources = set(candidate.route_sources)
                    if not sources or not sources <= set(route_names):
                        bad_provenance += 1
                    if TFIDF in sources:
                        union_with_tfidf += 1
        records.append(record)

    print(f"   replayed {len(ARMS)} arms over {len(agent.captured)} turns "
          f"in {time.time() - started:.0f}s")

    found = _recalled(records, COMMITTED_ARM, POOL_LIMIT)
    never = sorted({r["session_id"] for r in records} - found)
    hit_of = {r["sample_id"]: r["hit"] for r in result["sessions"]}
    sample_of = {session_id: sample
                 for session_id, sample in zip(agent.order, samples)}
    hits = sum(1 for s in found if hit_of[sample_of[s]["sample_id"]])

    # -- 1 -------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("1. CEILING -- what perfect retrieval is worth")
    print("=" * 76)
    print(f"   target reaches the pool          {len(found):>4}/{total}")
    print(f"   target NEVER retrieved           {len(never):>4}/{total}"
          "   <- the whole surface a route can attack")
    print(f"\n   In retrieval's OWN metric the headroom is exact and it is the")
    print(f"   only number here that is not an inference:")
    print(f"      candidate recall@{POOL_LIMIT}             "
          f"{len(found) / total:.4f}  ->  1.0000 "
          f"= +{len(never) / total:.4f} recall")
    print(f"\n   Downstream in TS it is an EXTRAPOLATION, and the published")
    print(f"   '+0.025 TS' moved the HitRate term only. TS = 0.5*HR + 0.3*MRR")
    print(f"   + 0.2*eff, and a recovered hit moves all three:")
    ceiling = extrapolated_ceiling(result, len(never), total)
    print(f"      in-pool conversion               {hits}/{len(found)} = "
          f"{ceiling['conversion']:.1%}")
    print(f"      recovering all {len(never)} at that rate  "
          f"~{ceiling['recovered']:.1f} hits")
    print(f"      extrapolated TS                  {technical_score:.6f} -> "
          f"{ceiling['technical_score']:.6f}  "
          f"(+{ceiling['technical_score'] - technical_score:.4f})")
    print(f"      of which HitRate term only       "
          f"+{0.5 * ceiling['recovered'] / total:.4f}   <- the old figure")

    eligible = {str(sample_of[s]["sample_id"]) for s in never}
    print(f"\n   And MEASURED, by giving retrieval the answer on exactly those")
    print(f"   {len(eligible)} sessions and no others. Two injection positions "
          "bracket what")
    print("   any route could deliver:")
    bounds = {}
    for position, label in (("floor", "target injected at the POOL FLOOR "
                                      "(fusion_score 0, lower bound)"),
                            ("head", "target injected at the POOL HEAD "
                                     "(best fusion_score, upper bound)")):
        started = time.time()
        oracle = oracle_run(position, samples, eligible,
                            catalog_ids, categories, products)
        bounds[position] = oracle
        print(f"      {label}")
        print(f"         HR {oracle['hit_rate_at_10']:.4f}  "
              f"MRR {oracle['mrr']:.6f}  MTTC {oracle['mttc']:.3f}  "
              f"TS {oracle['recommended_technical_score']:.6f}  "
              f"(+{oracle['recommended_technical_score'] - technical_score:.4f})"
              f"   [{time.time() - started:.0f}s]", flush=True)
        print("         " + format_test(
            "vs committed", mcnemar({str(s["sample_id"]): bool(s["hit"])
                                     for s in result["sessions"]},
                                    {str(s["sample_id"]): bool(s["hit"])
                                     for s in oracle["sessions"]})))
    low = bounds["floor"]["recommended_technical_score"] - technical_score
    high = bounds["head"]["recommended_technical_score"] - technical_score
    print(f"\n   PERFECT retrieval is worth [{low:+.4f}, {high:+.4f}] TS, "
          "measured.")
    print(f"   Meanwhile {len(found) - hits} targets are IN the pool and still "
          f"lose. That\n   conversion failure is "
          f"{(len(found) - hits) / max(len(never), 1):.1f}x the size of the "
          "entire retrieval surface,\n   and every point of it is downstream "
          "of retrieval.")

    # -- 2 -------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("2. DIAGNOSIS -- what kind of failure are the misses?")
    print("=" * 76)
    print("   Dense retrieval bridges VOCABULARY MISMATCH: the shopper says")
    print("   'warm', the catalog says 'insulated'. So: do the misses share")
    print("   words with the target already?")
    print("\n   FIRST, a control on the metric itself. The evaluator builds the")
    print("   opening line as \"I'm looking for {coarse_category(target."
          "categories)}\"")
    print("   (evaluator/local_evaluator.py:154). If the target's categories")
    print("   field is counted among its terms, overlap >= 1 is an IDENTITY:")
    generated = sum(
        1 for session_id in by_session
        if coarse_category(categories.get(target_of[session_id], [])).lower()
        in opening_of[session_id].lower())
    print(f"      openings containing the target's own taxonomy string "
          f"{generated:>4}/{total}")

    variants = (
        ("full text (title+cats+feats+desc)", ("title", "categories",
                                               "features", "description")),
        ("taxonomy field REMOVED", ("title", "features", "description")),
        ("title only", ("title",)),
    )
    target_terms: dict[str, dict[str, set]] = {}
    for session_id, target in target_of.items():
        row = agent.connection.execute(
            "SELECT title, categories, features, description FROM products "
            "WHERE parent_asin = ?", (target,)).fetchone()
        columns = dict(zip(("title", "categories", "features", "description"),
                           row or ("", "", "", "")))
        target_terms[session_id] = {
            label: set(terms(" ".join(str(columns[name] or "")
                                      for name in fields)))
            for label, fields in variants
        }

    print(f"\n   {'target text counted':36}{'group':10}{'0':>5}{'1-2':>6}"
          f"{'3-4':>6}{'5+':>5}{'  share with 0':>16}")
    for label, _ in variants:
        for group_name, group in (("missed", never), ("found", sorted(found))):
            buckets: Counter = Counter()
            for session_id in group:
                shared = len(said_of[session_id] & target_terms[session_id][label])
                buckets[0 if shared == 0 else 1 if shared <= 2
                        else 2 if shared <= 4 else 3] += 1
            zero = buckets[0] / len(group) if group else 0.0
            print(f"   {label if group_name == 'missed' else '':36}"
                  f"{group_name:10}{buckets[0]:>5}{buckets[1]:>6}"
                  f"{buckets[2]:>6}{buckets[3]:>5}{zero:>16.1%}")

    inside = sum(
        1 for session_id in never
        if (said_of[session_id] & target_terms[session_id]["full text (title+cats+feats+desc)"])
        and (said_of[session_id]
             & target_terms[session_id]["full text (title+cats+feats+desc)"])
        <= set(terms(opening_of[session_id])))
    print(f"\n   missed sessions whose ENTIRE overlap sits inside the "
          f"generated opening  {inside}/{len(never)}")
    print("\n   READ THIS CORRECTLY. Under the full-text definition the 'missed'")
    print("   and 'found' rows are the same distribution, so that metric has no")
    print("   discriminative power -- it measures the simulator. With the copied")
    print("   taxonomy field removed the count of zero-overlap misses is no")
    print("   longer zero.")
    print("\n   What the data supports: PURE vocabulary-mismatch cases (shopper")
    print("   and target sharing no words at all) are rare in this set. What it")
    print("   does NOT support, and what the previous version of this tool")
    print("   claimed: that semantic retrieval therefore has no value. Sharing")
    print("   a word is not the same as being separable by that word, and a")
    print("   trained encoder's contribution is ordering among the many")
    print("   products that share it -- which this metric does not measure.")

    # -- 3 -------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("3. STARVATION -- how much did the shopper actually say?")
    print("=" * 76)
    print(f"   {'group':16}{'n':>5}{'tokens':>26}{'informative turns':>22}"
          f"{'filled slots':>22}")
    print(f"   {'':16}{'':5}{'median':>10}{'mean':>8}{'P90':>8}"
          f"{'median':>10}{'mean':>12}{'median':>10}{'mean':>12}")
    for label, group in (("target found", sorted(found)), ("target missed", never)):
        tokens = percentiles([len(said_of[s]) for s in group])
        turns = percentiles([informative_of[s] for s in group])
        slots = percentiles([slots_of[s] for s in group])
        print(f"   {label:16}{len(group):>5}{tokens.median:>10.0f}"
              f"{tokens.mean:>8.1f}{tokens.p90:>8.0f}"
              f"{turns.median:>10.0f}{turns.mean:>12.2f}"
              f"{slots.median:>10.0f}{slots.mean:>12.2f}")
    missed_scenarios = Counter(scenario_of[s] for s in never)
    all_scenarios = Counter(scenario_of[s] for s in by_session)
    print("\n   miss rate by scenario:")
    for scenario in sorted(all_scenarios):
        missed_here = missed_scenarios[scenario]
        print(f"      {scenario:18}{missed_here:>4}/{all_scenarios[scenario]:<4}"
              f"= {missed_here / all_scenarios[scenario]:.1%}")
    print("\n   The medians do NOT separate the two groups -- both sit at the")
    print("   same value -- so 'the misses are starved' rests on the MEAN and on")
    print("   the scenario mix, not on the median. Browsing is over-represented")
    print("   among the misses: those sessions open with a bare category and")
    print("   never add anything, because the agent never asks a question. The")
    print("   published '0.95 filled slots' was a mean printed as a median.")

    # -- 4 -------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("4. COMPLEMENTARITY -- does a lexical vector route ADD candidates?")
    print("=" * 76)
    print("   Standalone recall cannot answer this: a weaker retriever can")
    print("   still contribute unique candidates through the union. So each arm")
    print("   is fused with the shipped RRF and compared to the committed set.")
    print(f"\n   {'arm':28}" + "".join(f"{'@' + str(k):>9}" for k in K_VALUES)
          + f"{'recovered':>11}{'lost':>7}")
    committed_set = _recalled(records, COMMITTED_ARM, POOL_LIMIT)
    for arm in ARMS:
        arm_set = _recalled(records, arm, POOL_LIMIT)
        recovered = len(arm_set - committed_set)
        lost = len(committed_set - arm_set)
        print(f"   {arm:28}"
              + "".join(f"{recall(records, arm, k):>9.4f}" for k in K_VALUES)
              + f"{recovered:>11}{lost:>7}")
    print("   recovered / lost are vs the committed arm at "
          f"@{POOL_LIMIT}, session-level.")

    scenarios = sorted({r["scenario"] for r in records})
    print(f"\n   recall@{POOL_LIMIT} by scenario")
    print(f"   {'arm':28}" + "".join(f"{s[:12]:>14}" for s in scenarios))
    print(f"   {'':28}" + "".join(
        f"{'n=' + str(sum(1 for r in records if r['scenario'] == s)):>14}"
        for s in scenarios))
    for arm in ARMS:
        row = ""
        for scenario in scenarios:
            rows = [r for r in records if r["scenario"] == scenario]
            row += f"{recall(rows, arm, POOL_LIMIT):>14.4f}"
        print(f"   {arm:28}{row}")

    print("\n   pool shape per turn")
    print(f"   {'arm':28}{'pre-cap mean':>14}{'median':>9}{'P90':>7}{'max':>7}"
          f"{'final mean':>13}{'cap binds':>12}{'cap loss':>11}")
    for arm in ARMS:
        before = percentiles(precap[arm])
        after = percentiles(final[arm])
        binds = sum(1 for size in precap[arm] if size > POOL_LIMIT)
        dropped = percentiles([max(0, size - POOL_LIMIT) for size in precap[arm]])
        print(f"   {arm:28}{before.mean:>14.1f}{before.median:>9.0f}"
              f"{before.p90:>7.0f}{before.maximum:>7.0f}{after.mean:>13.1f}"
              f"{binds / len(precap[arm]):>11.1%}{dropped.mean:>11.1f}")
    print("   cap binds = share of turns where the pre-cap union exceeded "
          f"{POOL_LIMIT};\n   cap loss = mean unique candidates the cap "
          "discarded on a turn.")

    print(f"\n   union integrity ({UNION_ARM})")
    print(f"      route provenance preserved    "
          f"{'PASS' if bad_provenance == 0 else 'FAIL'}"
          f"  ({bad_provenance} candidates with empty or foreign route_sources)")
    print(f"      pool members carrying '{TFIDF}'   {union_with_tfidf}"
          f"/{union_members} = {union_with_tfidf / max(union_members, 1):.1%}")
    print(f"      candidates the lexical route alone surfaced, per turn  "
          f"{unique_to_tfidf / len(agent.captured):.1f}")

    print("\n   is the union's recall difference established?")
    for k in K_VALUES:
        before = {r["session_id"]: r["best_rank"][COMMITTED_ARM] is not None
                  and r["best_rank"][COMMITTED_ARM] < k for r in records}
        after = {r["session_id"]: r["best_rank"][UNION_ARM] is not None
                 and r["best_rank"][UNION_ARM] < k for r in records}
        print("      " + format_test(f"@{k}: union vs committed",
                                     mcnemar(before, after)))
    print("   One comparison at three thresholds, not three findings (D-P3).")

    # -- 5 -------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("5. FEASIBILITY -- what could ship, in this environment")
    print("=" * 76)
    print("   This section is a CONSTRAINT, not evidence about semantics. What")
    print("   a trained encoder would be worth is not measured anywhere in this")
    print("   tool; what follows is only whether one could be run and shipped.")
    print(f"\n   interpreter running this tool   {sys.version.split()[0]}")
    for name in ("numpy", "torch", "transformers", "sentence_transformers"):
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "unknown")
            print(f"   {name:31} {version}")
        except ImportError:
            print(f"   {name:31} NOT IMPORTABLE HERE")
    print("\n   Scoped precisely, because the previous version of this comment")
    print("   said 'none is installed' as a flat fact. What is actually true:")
    print("      - no encoder or weights are VENDORED in this repository, and")
    print("        docs/submission_rules.md notes the organizer 'may disable")
    print("        network access' for final scoring, so nothing may be")
    print("        downloaded at scoring time;")
    print("      - the submission is certified on the bare system interpreter")
    print("        (python3.9), which current torch builds do not support;")
    print("      - the starter package is standard-library only by design, and")
    print("        adding a ~90MB model plus a torch dependency to it is a")
    print("        submission-packaging decision, not a retrieval decision.")
    print("   A different machine having torch installed does not change any of")
    print("   those three. Neither does it make the encoder worthless.")

    print(f"\nconfig: {config_guard.describe()}")


if __name__ == "__main__":
    main()
