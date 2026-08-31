"""Latency, token usage and cost disclosure -- the numbers, measured.

``docs/submission_rules.md`` requires "a disclosure of latency, token usage,
and estimated model cost". This tool produces it, and it exists as a TOOL
rather than as a paragraph for the reason every other number in this repo is
regenerated rather than remembered: a disclosure written by hand is stale the
first time the pipeline moves, and the pipeline has moved in every phase.

``docs/performance_disclosure.md`` is this tool's output plus its
interpretation. Re-run it and update that file in the same change.

WHAT IS AND IS NOT MEASURED HERE

Wall-clock on ONE machine, stated with the machine. Latency is not a property
of the code alone and quoting it as though it were would be the same
overclaim this repo has corrected elsewhere. What IS a property of the code,
and is the part that transfers: no network call, no model call, no API key,
no token spend, and a fixed amount of work per turn.

Usage:  python3 -m tools.disclosure       (~2 min: one full evaluator run)
"""

from __future__ import annotations

import platform
import sys
import time
import tracemalloc

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent
from tools import config_guard
from tools.summaries import percentiles

CATALOG = "data/catalog.jsonl"
DATASET = "data/public_set.jsonl"


class TimingAgent(Agent):
    """The shipped agent, timed per turn. Adds nothing to the turn but a clock.

    ``time.perf_counter`` around the real ``respond``, so the measurement
    includes everything a scoring harness would pay for: state update,
    retrieval, ranking, reranking, clarification and payload construction.
    """

    def __init__(self, catalog_path: str) -> None:
        started = time.perf_counter()
        super().__init__(catalog_path)
        self.build_seconds = time.perf_counter() - started
        self.turn_ms: list[float] = []

    def respond(self, session_id: str, user_message: str, turn: int,
                top_k: int) -> dict:
        started = time.perf_counter()
        payload = super().respond(session_id, user_message, turn, top_k)
        self.turn_ms.append((time.perf_counter() - started) * 1000.0)
        return payload


def main() -> None:
    config_guard.assert_everything()

    tracemalloc.start()
    agent = TimingAgent(CATALOG)
    built_peak = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    samples = load_jsonl(DATASET)
    catalog_ids, categories, products = catalog_index(CATALOG)

    started = time.perf_counter()
    result = evaluate(agent, samples, catalog_ids, categories, products)
    wall = time.perf_counter() - started
    peak = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    tracemalloc.stop()

    score = result["recommended_technical_score"]
    committed = config_guard.COMMITTED_TECHNICAL_SCORE
    if abs(score - committed) > 1e-9:
        raise SystemExit(
            f"timing changed the score ({score} != {committed}); this "
            "disclosure would describe a pipeline that is not the shipped one")

    print("ENVIRONMENT")
    print(f"   python                      {platform.python_version()} "
          f"({platform.python_implementation()})")
    print(f"   platform                    {platform.platform()}")
    print(f"   machine                     {platform.machine()}")
    print(f"   executable                  {sys.executable}")

    latency = percentiles(agent.turn_ms)
    print("\nLATENCY -- one full turn, end to end")
    print(f"   turns measured              {len(agent.turn_ms)}")
    print(f"   mean                        {latency.mean:.2f} ms")
    print(f"   median                      {latency.median:.2f} ms")
    print(f"   P90                         {latency.p90:.2f} ms")
    print(f"   max                         {latency.maximum:.2f} ms")
    print(f"   index build (once, startup) {agent.build_seconds:.2f} s")
    print(f"   whole 200-session run       {wall:.1f} s")

    usage = result["reported_token_usage"]
    print("\nTOKENS AND COST")
    print(f"   prompt_tokens               {usage['prompt_tokens']}")
    print(f"   completion_tokens           {usage['completion_tokens']}")
    print(f"   total_tokens                {usage['total_tokens']}")
    print("   estimated model cost        $0.00")
    print("   Not an estimate and not a rounding. The agent makes no model")
    print("   call of any kind: there is no LLM in the turn path, no API")
    print("   client in the package, and the reported usage is a literal")
    print("   {0, 0} rather than an untracked figure.")

    print("\nMEMORY")
    print(f"   peak traced, after index    {built_peak:.0f} MiB")
    print(f"   peak traced, whole run      {peak:.0f} MiB")
    print("   The catalog lives in an in-memory SQLite database, so most of")
    print("   the footprint is SQLite's and not visible to tracemalloc; the")
    print("   figures above are the Python-object half only and are a floor,")
    print("   not a total.")

    print("\nNETWORK AND CREDENTIALS")
    print("   network calls               0 -- nothing in starter/ imports a")
    print("                               network library; see the offline")
    print("                               statement in docs/method_and_limitations.md")
    print("   credentials required        none")
    print("   offline fallback            not applicable: the offline path IS")
    print("                               the only path")

    print(f"\nscore during this run: {score:.6f}")
    print(f"config: {config_guard.describe()}")


if __name__ == "__main__":
    main()
