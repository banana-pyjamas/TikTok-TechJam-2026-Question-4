"""Phase 15 B diagnostic -- does ASKING recover candidates retrieval missed?

Phase 15 measured that clarification is worth +0.48 TS. It did not measure the
MECHANISM anyone assumed: that a question puts a constraint into the query and
the constraint pulls the target into the pool. That is a retrieval claim, and
until now nothing in this repo has checked it.

The question matters because the alternative explanation is unflattering and
fits the same score. A disclosed constraint also changes RANKING -- it fills a
slot that ``ranking.score_candidate`` scores on -- so clarification could be
worth +0.48 while recovering no candidates at all, purely by reordering a pool
that already contained the target. Phase 13 measured the target sitting in the
pool and losing on 92 of 200 sessions, so there was plenty of room for that.

Read-only. This tool changes nothing in production and the shipped run it
observes must reproduce the plain evaluator exactly.

WHAT IT REPORTS

   1  candidate recall @50 / @100 / @300, session level over eligible turns
   2  eligible-session candidate recall
   3  candidate recovery rate after an informative answer
   4  candidate recovery LATENCY, in turns
   5  candidate recovery failure rate
   6  override candidate recovery latency, 0 / 1 / 2+ turns
   7  override candidate recovery failure rate
   8  top-10 recovery latency
   9  downstream delay: top-10 recovery turn - candidate recovery turn
  10  every remaining retrieval miss, classified

SCOPE. Every turn-level figure is over SCORING-ELIGIBLE turns only, using the
evaluator's own override metadata through ``first_scoring_turn`` -- the same
correction Phase 13's gate was rewritten for. A target in the pool before the
override turn is a pool hit the evaluator would never have scored, and
counting it credits retrieval with a recovery that could not have mattered.

PROBE LIVENESS. Both spies assert they FIRED before any number derived from
them is printed. This is not defensive habit: a Phase 15 review probe patched
``starter.retrieval.retrieve`` while ``agent.py`` holds a direct reference to
that function, so the spy never ran and the tool confidently reported 31 of 31
misses as retrieval failures when the true figure was 4 (D Phase 15 review,
item 6). A probe that cannot prove it observed something must not be quoted.

Usage:  python3 -m tools.phase15_recovery      (~2 min: one evaluator run)
"""

from __future__ import annotations

import time
from collections import Counter

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter import agent as agent_module
from starter import retrieval as retrieval_module
from starter.agent import Agent
from starter.contracts import Context
from starter.state import is_non_answer
from starter.text import terms
from tools import config_guard
from tools.phase13_dense_gate import first_scoring_turn
from tools.summaries import percentiles

CATALOG = "data/catalog.jsonl"
DATASET = "data/public_set.jsonl"

# How deep to look when asking "could this target have been retrieved at all?"
# Far past POOL_LIMIT on purpose: the point is to separate "the cap cut it"
# from "the query never matched it", and only a depth the cap cannot reach
# can do that.
UNCAPPED_LIMIT = 5000


class RecoveryProbe(Agent):
    """The shipped agent, with the pool and the ranked list observed.

    Both hooks patch names in ``starter.agent``'s OWN namespace, because that
    is where ``respond`` looks them up: ``agent.py`` does ``from
    starter.retrieval import retrieve``, so patching
    ``starter.retrieval.retrieve`` binds a function nothing calls. That is the
    exact mistake this tool's docstring records, and ``fired`` below is what
    makes repeating it loud instead of silent.

    Observing only. ``respond`` is the shipped one; the hooks return their
    inputs untouched.
    """

    def __init__(self, catalog_path: str) -> None:
        super().__init__(catalog_path)
        self.records: list[dict] = []
        self.order: list[str] = []
        self.fired = Counter()
        self._pool: list[str] = []
        self._context = None

    def reset(self, session_id: str, user_profile: dict) -> None:
        super().reset(session_id, user_profile)
        self.order.append(session_id)

    def install(self) -> None:
        original_retrieve = agent_module.retrieve
        original_fuse = agent_module.fuse
        original_response = agent_module._to_response

        def watched_retrieve(connection, context, limit=None, routes=None):
            pool = original_retrieve(connection, context, limit, routes)
            self.fired["retrieve"] += 1
            self._pool = [candidate.parent_asin for candidate in pool]
            self._context = context
            return pool

        def watched_fuse(routes, limit):
            pool = original_fuse(routes, limit)
            self.fired["fuse"] += 1
            self._pool = [candidate.parent_asin for candidate in pool]
            return pool

        def watched_response(result, top_k, ask=None):
            self.fired["response"] += 1
            ranked = [candidate.parent_asin for candidate in result.ranked]
            self.records.append({"pool": list(self._pool), "ranked": ranked,
                                 "ask": ask, "context": self._context})
            return original_response(result, top_k, ask)

        agent_module.retrieve = watched_retrieve
        agent_module.fuse = watched_fuse
        agent_module._to_response = watched_response
        self._restore = (original_retrieve, original_fuse, original_response)

    def uninstall(self) -> None:
        (agent_module.retrieve, agent_module.fuse,
         agent_module._to_response) = self._restore

    def respond(self, session_id: str, user_message: str, turn: int,
                top_k: int) -> dict:
        before = len(self.records)
        payload = super().respond(session_id, user_message, turn, top_k)
        # The response hook appends exactly one record per turn. If it did
        # not, every rate below would be computed over a different
        # denominator than it claims.
        if len(self.records) != before + 1:
            raise SystemExit(
                "the response probe did not fire on a turn: "
                f"{len(self.records) - before} records for one respond(). "
                "Every number in this tool is derived from that record.")
        self.records[-1].update({
            "session_id": session_id, "turn": turn, "message": user_message,
        })
        return payload


def _position(order: list[str], target: str) -> int | None:
    try:
        return order.index(target) + 1
    except ValueError:
        return None


def _classify_miss(agent: RecoveryProbe, record: dict, target: str,
                   product: dict) -> str:
    """Why did this turn's pool not contain the target?

    Four outcomes, separated by re-running retrieval on the SAME context at a
    depth POOL_LIMIT cannot reach:

      state           nothing to search on -- the turn's query tokenized to
                      empty, so retrieval was never given a chance
      cap             the routes DID surface it at depth; POOL_LIMIT cut it
      query           the routes miss it with the query this turn built, but
                      a query made of the target's own title does find it --
                      so the product is reachable and the words were wrong
      never-retrieved not reachable even by its own title. Effectively absent
                      from the index, and the only class retrieval cannot fix
                      by asking a better question

    The order matters: each rung only runs when the one above it has been
    ruled out, so a session is never counted in two classes.
    """
    context = record.get("context")
    if context is None:
        return "unknown"
    if not terms(context.user_message or ""):
        return "state"
    deep = retrieval_module.run_routes(
        agent.connection, context, UNCAPPED_LIMIT,
        list(agent_module.DEFAULT_ROUTES))
    for rows in deep.values():
        if any(str(asin) == target for asin, _ in rows):
            return "cap"
    own_words = " ".join(terms(str(product.get("title") or ""))[:20])
    if own_words:
        rows = retrieval_module.bm25_route(
            agent.connection,
            Context(session_id="probe", turn=1, user_message=own_words),
            UNCAPPED_LIMIT)
        if any(str(asin) == target for asin, _ in rows):
            return "query"
    return "never-retrieved"


def main() -> None:
    config_guard.assert_everything()

    print("building index...", flush=True)
    started = time.time()
    agent = RecoveryProbe(CATALOG)
    samples = load_jsonl(DATASET)
    catalog_ids, categories, products = catalog_index(CATALOG)
    print(f"ready in {time.time() - started:.1f}s", flush=True)

    agent.install()
    try:
        result = evaluate(agent, samples, catalog_ids, categories, products)
    finally:
        agent.uninstall()

    # LIVENESS, before anything is derived. See the module docstring.
    if not agent.fired["response"]:
        raise SystemExit("the response probe never fired: nothing below is a "
                         "measurement of the shipped pipeline")
    if not (agent.fired["retrieve"] or agent.fired["fuse"]):
        raise SystemExit(
            "neither retrieval probe fired, so every pool is empty and this "
            "tool would report 100% retrieval failure. agent.py binds "
            "`retrieve` and `fuse` into its own namespace; patch THOSE.")
    committed = config_guard.COMMITTED_TECHNICAL_SCORE
    score = result["recommended_technical_score"]
    print(f"\nobserved run: TS {score:.6f}   probes fired: "
          f"retrieve {agent.fired['retrieve']}, fuse {agent.fired['fuse']}, "
          f"response {agent.fired['response']}")
    if abs(score - committed) > 1e-9:
        raise SystemExit(
            f"observing changed the score ({score} != {committed}); this "
            "tool is supposed to be read-only")

    target_of = {session_id: str(sample["ground_truth"]["parent_asin"])
                 for session_id, sample in zip(agent.order, samples)}
    scenario_of = {session_id: str(sample["scenario_type"])
                   for session_id, sample in zip(agent.order, samples)}
    eligible_of = {session_id: first_scoring_turn(sample, products)
                   for session_id, sample in zip(agent.order, samples)}
    hit_of = {str(s["sample_id"]): bool(s["hit"]) for s in result["sessions"]}
    hit_by_session = {session_id: hit_of[str(sample["sample_id"])]
                      for session_id, sample in zip(agent.order, samples)}

    by_session: dict[str, list[dict]] = {}
    for record in agent.records:
        session_id = record["session_id"]
        target = target_of[session_id]
        record["eligible"] = record["turn"] >= eligible_of[session_id]
        record["informative"] = (record["turn"] > 1
                                 and not is_non_answer(record["message"]))
        record["pool_position"] = _position(record["pool"], target)
        record["final_rank"] = _position(record["ranked"], target)
        record["in_top10"] = (record["final_rank"] is not None
                              and record["final_rank"] <= 10)
        by_session.setdefault(session_id, []).append(record)
    for records in by_session.values():
        records.sort(key=lambda item: item["turn"])

    sessions = sorted(by_session)
    print(f"\ncaptured {len(agent.records)} turns over {len(sessions)} "
          f"sessions; {sum(1 for r in agent.records if r['eligible'])} are "
          "scoring-eligible")

    def first_turn(records, predicate):
        for record in records:
            if record["eligible"] and predicate(record):
                return record["turn"]
        return None

    # -- 1 / 2 ---------------------------------------------------------------
    print("\n" + "=" * 74)
    print("1-2. CANDIDATE RECALL over scoring-eligible turns, session level")
    print("=" * 74)
    for depth in (50, 100, 300):
        found = sum(
            1 for session in sessions
            if first_turn(by_session[session],
                          lambda r, d=depth: r["pool_position"] is not None
                          and r["pool_position"] <= d) is not None)
        print(f"   recall@{depth:<4} {found / len(sessions):.4f}  "
              f"({found}/{len(sessions)} sessions)")
    in_pool = {session: first_turn(by_session[session],
                                   lambda r: r["pool_position"] is not None)
               for session in sessions}
    reached = sum(1 for value in in_pool.values() if value is not None)
    print(f"   eligible-session candidate recall  {reached / len(sessions):.4f}"
          f"  ({reached}/{len(sessions)})")

    # -- 3 / 4 / 5 -----------------------------------------------------------
    print("\n" + "=" * 74)
    print("3-5. RECOVERY AFTER AN INFORMATIVE ANSWER")
    print("=" * 74)
    print("   'informative' = a reply past turn 1 that is not a harness")
    print("   non-answer (state.is_non_answer). 'needed recovery' = the")
    print("   target was NOT in the pool on any eligible turn up to and")
    print("   including that answer -- the only population where a recovery")
    print("   is a thing that can happen.")
    first_informative = {
        session: first_turn(by_session[session], lambda r: r["informative"])
        for session in sessions}
    needed, recovered, latencies = [], [], []
    for session in sessions:
        answer_turn = first_informative[session]
        if answer_turn is None:
            continue
        pool_turn = in_pool[session]
        if pool_turn is not None and pool_turn < answer_turn:
            continue  # already had it; nothing to recover
        needed.append(session)
        if pool_turn is not None:
            recovered.append(session)
            latencies.append(pool_turn - answer_turn)
    print(f"\n   sessions with an informative answer            "
          f"{sum(1 for v in first_informative.values() if v is not None):>5}")
    print(f"   of those, needed a candidate recovery          {len(needed):>5}")
    print(f"   3. candidate recovery RATE                     "
          f"{len(recovered) / max(len(needed), 1):>9.4f}"
          f"  ({len(recovered)}/{len(needed)})")
    print(f"   5. candidate recovery FAILURE rate             "
          f"{(len(needed) - len(recovered)) / max(len(needed), 1):>9.4f}"
          f"  ({len(needed) - len(recovered)}/{len(needed)})")
    if latencies:
        stats = percentiles([float(value) for value in latencies])
        print(f"   4. recovery LATENCY (turns after the answer)   "
              f"mean {stats.mean:>5.2f}   median {stats.median:>4.0f}   "
              f"P90 {stats.p90:>4.0f}")
        spread = Counter(latencies)
        print("      distribution  " + "  ".join(
            f"{turns:+d}: {spread[turns]}" for turns in sorted(spread)))
        print("      A latency of 0 means the target was in the pool on the")
        print("      SAME turn the answer arrived -- the query that turn")
        print("      already carried the new constraint, which is the")
        print("      mechanism working immediately rather than not at all.")

    # -- 6 / 7 ---------------------------------------------------------------
    print("\n" + "=" * 74)
    print("6-7. OVERRIDE RECOVERY -- the case Phase 3 exists for")
    print("=" * 74)
    override_sessions = [s for s in sessions
                         if scenario_of[s] == "intent_override"]
    buckets: Counter = Counter()
    failures = 0
    for session in override_sessions:
        override_turn = eligible_of[session]
        post = first_turn(by_session[session],
                          lambda r: r["pool_position"] is not None)
        if post is None:
            failures += 1
            continue
        delta = post - override_turn
        buckets["0" if delta <= 0 else "1" if delta == 1 else "2+"] += 1
    total_override = len(override_sessions)
    print(f"   intent_override sessions                       "
          f"{total_override:>5}")
    for bucket in ("0", "1", "2+"):
        print(f"   6. recovered {bucket:>2} turns after the override      "
              f"{buckets[bucket]:>5}  "
              f"{buckets[bucket] / max(total_override, 1):>7.1%}")
    print(f"   7. override recovery FAILURE rate              "
          f"{failures / max(total_override, 1):>9.4f}  "
          f"({failures}/{total_override})")

    # -- 8 / 9 ---------------------------------------------------------------
    print("\n" + "=" * 74)
    print("8-9. TOP-10 RECOVERY, and the delay between the two")
    print("=" * 74)
    top10_turn = {session: first_turn(by_session[session],
                                      lambda r: r["in_top10"])
                  for session in sessions}
    top10_latency = [top10_turn[s] - first_informative[s]
                     for s in sessions
                     if top10_turn[s] is not None
                     and first_informative[s] is not None
                     and top10_turn[s] >= first_informative[s]]
    if top10_latency:
        stats = percentiles([float(v) for v in top10_latency])
        print(f"   8. top-10 latency after an informative answer  "
              f"mean {stats.mean:>5.2f}   median {stats.median:>4.0f}   "
              f"P90 {stats.p90:>4.0f}   (n={len(top10_latency)})")
    downstream = [top10_turn[s] - in_pool[s] for s in sessions
                  if top10_turn[s] is not None and in_pool[s] is not None]
    if downstream:
        stats = percentiles([float(v) for v in downstream])
        print(f"   9. downstream delay (top-10 turn - in-pool turn) "
              f"mean {stats.mean:>5.2f}   median {stats.median:>4.0f}   "
              f"P90 {stats.p90:>4.0f}   (n={len(downstream)})")
        spread = Counter(downstream)
        print("      distribution  " + "  ".join(
            f"{turns}: {spread[turns]}" for turns in sorted(spread)))
        print("      This is the number that says whether the residual is a")
        print("      RETRIEVAL problem or a RANKING one. A delay of 0 means")
        print("      the turn that found the target also showed it; anything")
        print("      larger is the ranker holding a candidate it already had.")

    # -- 10 ------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("10. THE REMAINING MISSES, CLASSIFIED")
    print("=" * 74)
    missed = [s for s in sessions if not hit_by_session[s]]
    never_pooled = [s for s in missed if in_pool[s] is None]
    pooled_never_top10 = [s for s in missed
                          if in_pool[s] is not None and top10_turn[s] is None]
    top10_never_hit = [s for s in missed if top10_turn[s] is not None]
    print(f"   sessions that missed                           {len(missed):>5}")
    print(f"   ... target never in the pool (RETRIEVAL)       "
          f"{len(never_pooled):>5}")
    print(f"   ... in the pool, never in the top 10 (RANKING) "
          f"{len(pooled_never_top10):>5}")
    print(f"   ... in the top 10 and still not scored         "
          f"{len(top10_never_hit):>5}")
    print("   The last row is not a bug: `evaluate` only counts a hit once")
    print("   `override_applied` is true, so a target shown before the")
    print("   override turn scores nothing however well it was ranked.")
    if never_pooled:
        print("\n   every retrieval miss, classified:")
        for session in never_pooled:
            records = [r for r in by_session[session] if r["eligible"]]
            if not records:
                print(f"     {scenario_of[session]:16} no eligible turn")
                continue
            reasons = Counter(
                _classify_miss(agent, record, target_of[session],
                               products.get(target_of[session], {}))
                for record in records)
            print(f"     {scenario_of[session]:16} "
                  f"{len(records)} eligible turns   "
                  + ", ".join(f"{k}: {v}" for k, v in reasons.most_common()))
        print("   (classification re-runs the routes at depth "
              f"{UNCAPPED_LIMIT} on the same\n    context; 'unknown' means the "
              "context was not captured for that turn.)")

    print(f"\nconfig: {config_guard.describe()}")


if __name__ == "__main__":
    main()
