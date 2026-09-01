"""Replay the real dialogue and keep what each turn left behind.

Every measurement tool here faces the same problem: the interesting state is
inside a turn, and the evaluator does not hand it out. Reimplementing the loop
to get at it would measure a transcript the agent never has -- the agent asks
nothing, so from turn 2 the customer only ever declines, and the dialogue stops
at the first hit.

So the loop is left alone and the agent is subclassed. ``respond`` delegates
and snapshots afterwards, which means the dialogue, the score, and the state
are the shipped ones. Observing is read-only: a tool built on this must
reproduce a plain evaluator run exactly, and the tools assert that.

Shared rather than copied per tool: two hand-maintained copies of a harness is
the drift shape this repo keeps finding (D-N2, D-P1).
"""

from __future__ import annotations

import copy

from starter.agent import Agent


class CapturingAgent(Agent):
    """The shipped agent, plus one record per turn.

    Each record is ``{"session_id", "turn", "message", "state"}``. The state is
    deep-copied because the real one keeps mutating for the rest of the
    session; a reference would leave every turn of a session pointing at its
    final state.

    ``order`` is the session ids in ``reset`` order, which is the evaluator's
    sample order -- the only way back from an opaque uuid to the sample it
    came from.
    """

    def __init__(self, catalog_path: str) -> None:
        super().__init__(catalog_path)
        self.order: list[str] = []
        self.captured: list[dict] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        super().reset(session_id, user_profile)
        self.order.append(session_id)

    def respond(self, session_id: str, user_message: str, turn: int,
                top_k: int) -> dict:
        response = super().respond(session_id, user_message, turn, top_k)
        self.captured.append({
            "session_id": session_id,
            "turn": turn,
            "message": user_message,
            "state": copy.deepcopy(self._states[session_id]),
        })
        return response

    def by_session(self) -> dict[str, list[dict]]:
        """Captured turns grouped by session, each in turn order."""
        grouped: dict[str, list[dict]] = {}
        for record in self.captured:
            grouped.setdefault(record["session_id"], []).append(record)
        for records in grouped.values():
            records.sort(key=lambda record: record["turn"])
        return grouped

    def sample_field(self, samples: list[dict], key: str) -> dict[str, str]:
        """``session_id -> str(sample[key])``, via ``reset`` order."""
        return {
            session_id: str(sample[key])
            for session_id, sample in zip(self.order, samples)
        }
