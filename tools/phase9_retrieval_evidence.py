"""Phase 9 — candidate-pool evidence for the committed route set.

MEASUREMENT ONLY. Nothing here tunes anything; the route set it reports on is
``retrieval.DEFAULT_ROUTES`` exactly as committed, and the tool refuses to run
if that constant has drifted (``tools.config_guard``).

``tests/test_retrieval_recall.py`` already measures recall, but over a
SYNTHETIC turn sequence: the opening line followed by every hard constraint and
soft preference in order, as if the shopper volunteered all of them. The real
agent never sees that transcript. It asks no questions, so from turn 2 the
customer only ever says "those options are not quite right yet", and the
dialogue stops at the first hit. This tool measures the dialogue that actually
happens.

TWO PHASES

capture   run the live evaluator loop under the committed configuration, with
          an agent subclassed only to snapshot the state each turn left behind,
          and to count which route functions and which SQL queries really ran.
replay    re-run every route over those captured contexts and fuse them under
          each route set.

Holding the dialogue fixed and varying only the route set is what makes the
comparison controlled. The alternative -- re-running the loop per route set --
changes the transcript as well as the pool, so the two cannot be separated.
Recall here is therefore "would this route set have surfaced the target on the
turns the shipped agent actually reached", which is the question B is asking.

RECALL is session-level: the target is recalled at K if it appears in the top-K
of the pool on ANY turn of the session. A target the agent never had a chance
to rank cannot be ranked, and one it saw on turn 4 was available to it.

Usage:  python3 -m tools.phase9_retrieval_evidence
"""

from __future__ import annotations

import copy
import time
from collections import Counter, defaultdict

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter import retrieval
from starter.agent import Agent
from starter.catalog_meta import lookup as meta_lookup
from starter.contracts import Context
from starter.retrieval import DEFAULT_ROUTES, POOL_LIMIT, fuse
from tools import config_guard
from tools.significance import format_test, mcnemar

CATALOG = "data/catalog.jsonl"
DATASET = "data/public_set.jsonl"
K_VALUES = (50, 100, 300)

ALL_ROUTES = ("bm25", "category", "attribute")
CONFIGS: dict[str, tuple[str, ...]] = {
    "bm25 only": ("bm25",),
    "bm25+category  (committed)": ("bm25", "category"),
    "bm25+cat+attribute  (ref)": ("bm25", "category", "attribute"),
}
COMMITTED = "bm25+category  (committed)"


class CapturingAgent(Agent):
    """The shipped agent, plus a snapshot of the state each turn left behind.

    ``respond`` delegates; the capture happens afterwards, so the dialogue and
    the score are the shipped ones. The state is deep-copied because the real
    one keeps mutating for the rest of the session.
    """

    def __init__(self, catalog_path: str) -> None:
        super().__init__(catalog_path)
        self.order: list[str] = []
        self.captured: list[dict] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        super().reset(session_id, user_profile)
        self.order.append(session_id)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        response = super().respond(session_id, user_message, turn, top_k)
        self.captured.append({
            "session_id": session_id,
            "turn": turn,
            "message": user_message,
            "state": copy.deepcopy(self._states[session_id]),
        })
        return response


def _instrument_routes() -> tuple[Counter, Counter]:
    """Wrap the route table and the SQL helper with call counters.

    Counts what the committed agent path really executes -- B's question is
    whether the attribute route runs at all, and route selection is supposed to
    happen BEFORE execution rather than by discarding results afterwards.
    Route invocations and SQL executions are counted separately because a route
    can return early without querying (the category route does exactly that
    when no category has been extracted).
    """
    route_calls: Counter = Counter()
    sql_calls: Counter = Counter()
    original_run = retrieval._run

    def counting_run(connection, expression, limit):
        sql_calls["_run"] += 1
        return original_run(connection, expression, limit)

    retrieval._run = counting_run
    for name in ALL_ROUTES:
        original_route = retrieval.ROUTES[name]

        def wrapper(connection, context, limit, _name=name, _route=original_route):
            route_calls[_name] += 1
            return _route(connection, context, limit)

        retrieval.ROUTES[name] = wrapper
    return route_calls, sql_calls


def _restore_routes() -> None:
    """Put the real route table back before the replay phase measures with it."""
    retrieval._run = _ORIGINAL_RUN
    retrieval.ROUTES.update(_ORIGINAL_ROUTES)


_ORIGINAL_RUN = retrieval._run
_ORIGINAL_ROUTES = dict(retrieval.ROUTES)


def _percentiles(values: list[int]) -> tuple[float, int, int]:
    ordered = sorted(values)
    mean = sum(ordered) / len(ordered)
    median = ordered[len(ordered) // 2]
    p90 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))]
    return mean, median, p90


def _recall(records: list[dict], config: str, k: int) -> float:
    if not records:
        return 0.0
    hits = sum(1 for r in records
               if r["best_rank"][config] is not None and r["best_rank"][config] < k)
    return hits / len(records)


def main() -> None:
    config_guard.assert_all_flags_pinned(set(config_guard.COMMITTED_FLAGS))
    config_guard.assert_committed_constants()
    config_guard.restore_committed_flags()
    assert CONFIGS[COMMITTED] == tuple(DEFAULT_ROUTES), (
        "this tool's committed config must be retrieval.DEFAULT_ROUTES")

    print("building index...", flush=True)
    started = time.time()
    agent = CapturingAgent(CATALOG)
    samples = load_jsonl(DATASET)
    catalog_ids, categories, products = catalog_index(CATALOG)
    print(f"ready in {time.time() - started:.1f}s", flush=True)

    # -- capture ---------------------------------------------------------
    route_calls, sql_calls = _instrument_routes()
    started = time.time()
    result = evaluate(agent, samples, catalog_ids, categories, products)
    _restore_routes()
    turns = len(agent.captured)
    print(f"captured {turns} turns over {len(agent.order)} sessions "
          f"in {time.time() - started:.0f}s")
    print(f"score during capture: HR {result['hit_rate_at_10']:.4f} "
          f"MRR {result['mrr']:.6f} MTTC {result['mttc']:.3f} "
          f"TS {result['recommended_technical_score']:.6f}")

    target_of = {
        session_id: str(sample["ground_truth"]["parent_asin"])
        for session_id, sample in zip(agent.order, samples)
    }
    scenario_of = {
        session_id: str(sample["scenario_type"])
        for session_id, sample in zip(agent.order, samples)
    }

    # -- replay ----------------------------------------------------------
    by_session: dict[str, list[dict]] = defaultdict(list)
    for capture in agent.captured:
        by_session[capture["session_id"]].append(capture)

    records: list[dict] = []
    precap_sizes: list[int] = []
    final_sizes: list[int] = []
    duplicate_pools = 0
    bad_provenance = 0
    multi_route_candidates = 0
    thin_metadata_in_pool = 0
    pool_members = 0
    started = time.time()

    for session_id, captures in by_session.items():
        target = target_of[session_id]
        record = {
            "session_id": session_id,
            "scenario": scenario_of[session_id],
            "best_rank": {name: None for name in CONFIGS},
            "in_route": {name: False for name in ALL_ROUTES},
            "in_precap": False,
            "in_final": False,
        }
        for capture in sorted(captures, key=lambda c: c["turn"]):
            context = Context(
                session_id=session_id,
                turn=capture["turn"],
                user_message=capture["message"],
                state=capture["state"],
            )
            per_route = {
                name: _ORIGINAL_ROUTES[name](agent.connection, context, POOL_LIMIT)
                for name in ALL_ROUTES
            }
            for name, rows in per_route.items():
                if any(asin == target for asin, _ in rows):
                    record["in_route"][name] = True

            precap = {asin
                      for name in DEFAULT_ROUTES
                      for asin, _ in per_route[name]}
            precap_sizes.append(len(precap))
            record["in_precap"] = record["in_precap"] or target in precap

            for label, route_names in CONFIGS.items():
                pool = fuse({n: per_route[n] for n in route_names}, POOL_LIMIT)
                asins = [candidate.parent_asin for candidate in pool]
                if target in asins:
                    rank = asins.index(target)
                    current = record["best_rank"][label]
                    record["best_rank"][label] = rank if current is None else min(current, rank)
                if label != COMMITTED:
                    continue

                # integrity checks, committed configuration only
                final_sizes.append(len(pool))
                record["in_final"] = record["in_final"] or target in asins
                if len(set(asins)) != len(asins):
                    duplicate_pools += 1
                pool_members += len(pool)
                for candidate in pool:
                    sources = set(candidate.route_sources)
                    if not sources or not sources <= set(DEFAULT_ROUTES):
                        bad_provenance += 1
                    if len(sources) > 1:
                        multi_route_candidates += 1
                metadata = meta_lookup(agent.connection, asins)
                for asin in asins:
                    signals = metadata.get(asin, {})
                    if not signals.get("color") and not signals.get("material") \
                            and signals.get("price") is None:
                        thin_metadata_in_pool += 1
        records.append(record)

    print(f"replayed 3 routes over {turns} captured turns "
          f"in {time.time() - started:.0f}s\n")

    # -- 1. candidate recall by route set --------------------------------
    print("1. candidate recall, session-level (target in top-K on any turn)")
    print(f"   {'route set':30}" + "".join(f"{'@' + str(k):>10}" for k in K_VALUES))
    for label in CONFIGS:
        print(f"   {label:30}"
              + "".join(f"{_recall(records, label, k):>10.4f}" for k in K_VALUES))
    print("   These are LOWER than tests/test_retrieval_recall.py reports for "
          "the same\n   route sets, and both are correct. That test feeds the "
          "state every hard\n   constraint and soft preference in turn; this "
          "one feeds it the dialogue the\n   agent actually gets, in which it "
          "asks nothing and so is told nothing after\n   turn 1. The gap "
          "between the two is the value of asking a question -- which is\n"
          "   what Phase 15 is for. Retrieval is not the layer that closes it.")

    # Whether the recall difference is a real difference or two point
    # estimates. The route set's effect on the SCORE is not established in the
    # arm that ships (see tools/phase7_ablation.py); this is the other axis it
    # could be justified on, so it gets the same paired test rather than a
    # bare comparison of two numbers.
    print("\n   is the recall difference established? "
          "(McNemar exact, paired per session)")
    for k in K_VALUES:
        for label in ("bm25+category  (committed)", "bm25+cat+attribute  (ref)"):
            before = {r["session_id"]: r["best_rank"]["bm25 only"] is not None
                      and r["best_rank"]["bm25 only"] < k for r in records}
            after = {r["session_id"]: r["best_rank"][label] is not None
                     and r["best_rank"][label] < k for r in records}
            print("   " + format_test(f"@{k}: {label} vs bm25", mcnemar(before, after)))

    # -- 2. by scenario --------------------------------------------------
    print("\n2. recall by scenario, committed route set")
    print(f"   {'scenario':18}{'n':>4}" + "".join(f"{'@' + str(k):>10}" for k in K_VALUES)
          + "   (bm25-only @300)")
    for scenario in sorted({r["scenario"] for r in records}):
        rows = [r for r in records if r["scenario"] == scenario]
        print(f"   {scenario:18}{len(rows):>4}"
              + "".join(f"{_recall(rows, COMMITTED, k):>10.4f}" for k in K_VALUES)
              + f"{_recall(rows, 'bm25 only', 300):>18.4f}")

    # -- 3. per-target presence ------------------------------------------
    total = len(records)
    print(f"\n3. per-target presence (n={total} sessions, any turn)")
    for name in ALL_ROUTES:
        count = sum(1 for r in records if r["in_route"][name])
        note = "  <- not executed by the committed agent" \
            if name not in DEFAULT_ROUTES else ""
        print(f"   target in {name + ' route':22}{count:>5}{count / total:>9.4f}{note}")
    precap = sum(1 for r in records if r["in_precap"])
    final = sum(1 for r in records if r["in_final"])
    print(f"   target in {'pre-cap union':22}{precap:>5}{precap / total:>9.4f}")
    print(f"   target in {'final pool':22}{final:>5}{final / total:>9.4f}")

    # -- 4. loss stages ---------------------------------------------------
    stages: Counter = Counter()
    for record in records:
        if record["in_final"]:
            stages["in final pool"] += 1
        elif record["in_precap"]:
            stages["lost at fusion cap"] += 1
        else:
            stages["never retrieved by any route"] += 1
    print(f"\n4. candidate loss stage, committed route set (n={total})")
    for stage in ("in final pool", "lost at fusion cap", "never retrieved by any route"):
        print(f"   {stage:32}{stages[stage]:>5}{stages[stage] / total:>9.4f}")
    assert sum(stages.values()) == total, "every session must be accounted for"

    # -- 5. pool distribution ---------------------------------------------
    print("\n5. candidate pool size per turn, committed route set")
    for label, values in (("pre-cap unique", precap_sizes), ("final pool", final_sizes)):
        mean, median, p90 = _percentiles(values)
        print(f"   {label:18}mean {mean:>8.1f}   median {median:>5}   "
              f"P90 {p90:>5}   max {max(values):>5}")
    capped = sum(1 for size in precap_sizes if size > POOL_LIMIT)
    print(f"   the {POOL_LIMIT}-candidate cap binds on {capped}/{len(precap_sizes)} "
          f"turns ({capped / len(precap_sizes):.1%})")

    # -- 6. integrity ------------------------------------------------------
    print("\n6. integrity of the committed pool")
    print(f"   unique ASINs                     "
          f"{'PASS' if duplicate_pools == 0 else 'FAIL'}"
          f"  ({duplicate_pools} pools with a repeat)")
    print(f"   route provenance preserved       "
          f"{'PASS' if bad_provenance == 0 else 'FAIL'}"
          f"  ({bad_provenance} candidates with empty or foreign route_sources;"
          f" {multi_route_candidates} found by both routes)")
    print(f"   no missing-metadata hard drop    "
          f"{'PASS' if thin_metadata_in_pool > 0 else 'FAIL'}"
          f"  ({thin_metadata_in_pool}/{pool_members} pool members "
          f"({thin_metadata_in_pool / pool_members:.1%}) have no colour, no "
          f"material and no price,\n"
          f"                                          and are pooled anyway)")
    print(f"   route calls in committed path    "
          + ", ".join(f"{name}={route_calls[name]}" for name in ALL_ROUTES))
    print(f"   attribute route executions       "
          f"{'0  PASS' if route_calls['attribute'] == 0 else str(route_calls['attribute']) + '  UNEXPECTED'}")
    print(f"   SQL queries issued               {sql_calls['_run']} over {turns} turns "
          f"({sql_calls['_run'] / turns:.2f}/turn)")

    print(f"\nconfig: {config_guard.describe()}")


if __name__ == "__main__":
    main()
