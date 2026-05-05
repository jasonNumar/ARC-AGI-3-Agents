from __future__ import annotations

from typing import Any

from arcengine import FrameData, GameAction, GameState

from observer_arc import (
    LocalSimulatorPlanner,
    ObserverArcModel,
    ObserverArcPolicy,
    PlannerConfig,
)

from .agent import Agent


class ObserverArcAgent(Agent):
    """Observer-centric ARC-AGI-3 agent.

    The policy is deterministic: no network calls and no external LLM. When the
    local offline toolkit exposes a game object, a bounded simulator planner is
    used opportunistically; otherwise play falls back to frame-only scoring.
    """

    MAX_ACTIONS = ObserverArcModel.load().max_actions

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = ObserverArcModel.load()
        self.policy = ObserverArcPolicy(game_id=self.game_id, model=self.model)
        self.local_planner = LocalSimulatorPlanner(
            PlannerConfig(
                max_depth=int(getattr(self.model, "planner_depth", 48)),
                beam_width=int(getattr(self.model, "planner_beam_width", 8)),
                branch_limit=int(getattr(self.model, "planner_branch_limit", 18)),
                max_nodes=int(getattr(self.model, "planner_max_nodes", 2200)),
                max_seconds=float(getattr(self.model, "planner_max_seconds", 0.30)),
                return_best_nonterminal=bool(
                    getattr(self.model, "planner_return_best_nonterminal", True)
                ),
            )
        )

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        local_game = getattr(self.arc_env, "_game", None)
        candidate = self.local_planner.plan(local_game, latest_frame)
        if candidate is None:
            candidate = self.policy.choose(frames, latest_frame)
        else:
            self.policy.commit_external_choice(latest_frame, candidate)
        action = GameAction.from_id(candidate.action_id)
        if action.is_complex():
            action.set_data(
                {
                    "game_id": self.game_id,
                    "x": int(candidate.x or 0),
                    "y": int(candidate.y or 0),
                }
            )
        else:
            action.set_data({"game_id": self.game_id})
        action.reasoning = candidate.reasoning(self.policy.state)
        return action
