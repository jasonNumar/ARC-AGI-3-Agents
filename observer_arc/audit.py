"""Lightweight offline audit trail for ObserverArc decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .state import ActionCandidate, WorldState


@dataclass(frozen=True)
class ArcStepTrace:
    step: int
    frame_hash: str
    game_state: str
    levels_completed: int
    available_actions: list[int]
    action_key: str
    action_source: str
    final_score: float
    value_score: float
    novelty_score: float
    repetition_penalty: float
    hypotheses: list[str]
    reasoning: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArcAuditTrail:
    max_steps: int = 512
    steps: list[ArcStepTrace] = field(default_factory=list)

    def record(
        self,
        world: WorldState,
        selected: ActionCandidate,
        hypotheses: list[str],
    ) -> None:
        self.steps.append(
            ArcStepTrace(
                step=world.action_index,
                frame_hash=world.frame.frame_hash[:16],
                game_state=world.game_state,
                levels_completed=world.levels_completed,
                available_actions=list(world.available_actions),
                action_key=selected.key(),
                action_source=selected.source,
                final_score=round(selected.final_score, 5),
                value_score=round(selected.value_score, 5),
                novelty_score=round(selected.novelty_score, 5),
                repetition_penalty=round(selected.repetition_penalty, 5),
                hypotheses=hypotheses[-6:],
                reasoning=selected.explanation[:240],
            )
        )
        if len(self.steps) > self.max_steps:
            del self.steps[: len(self.steps) - self.max_steps]

    def summary(self) -> dict[str, Any]:
        if not self.steps:
            return {
                "steps": 0,
                "levels_completed": 0,
                "last_state": "unknown",
                "recent_actions": [],
            }
        return {
            "steps": len(self.steps),
            "levels_completed": self.steps[-1].levels_completed,
            "last_state": self.steps[-1].game_state,
            "recent_actions": [step.action_key for step in self.steps[-12:]],
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "steps": [step.as_dict() for step in self.steps],
        }
