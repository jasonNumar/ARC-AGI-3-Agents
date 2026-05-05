"""Compatibility structs for older ARC-AGI-3-Agents tests.

The runtime now imports canonical action/frame types from `arcengine`. This
module keeps the scaffold's legacy tests importable without changing runtime
agent behavior.
"""

from __future__ import annotations

from typing import Any

from arcengine import ActionInput, GameAction, GameState
from pydantic import BaseModel, Field


class FrameData(BaseModel):
    game_id: str = ""
    frame: list[Any] = Field(default_factory=list)
    state: GameState = GameState.NOT_PLAYED
    score: int = Field(0, ge=0, le=254)
    win_score: int = Field(0, ge=0, le=254)
    action_input: ActionInput = Field(default_factory=ActionInput)
    guid: str | None = None
    full_reset: bool = False
    available_actions: list[int] = Field(default_factory=list)

    @property
    def levels_completed(self) -> int:
        return self.score

    @property
    def win_levels(self) -> int:
        return self.win_score

    def is_empty(self) -> bool:
        return len(self.frame) == 0


class Card(BaseModel):
    game_id: str = ""
    total_plays: int = 0
    scores: list[int] = Field(default_factory=list)
    states: list[GameState] = Field(default_factory=list)
    actions: list[int] = Field(default_factory=list)
    resets: list[int] = Field(default_factory=list)

    @property
    def started(self) -> bool:
        return self.total_plays > 0

    @property
    def idx(self) -> int:
        return self.total_plays - 1 if self.started else -1

    @property
    def score(self) -> int | None:
        return self.scores[self.idx] if self.started and self.scores else None

    @property
    def high_score(self) -> int:
        return max(self.scores) if self.scores else 0

    @property
    def state(self) -> GameState:
        return self.states[self.idx] if self.started and self.states else GameState.NOT_PLAYED

    @property
    def action_count(self) -> int:
        return self.actions[self.idx] if self.started and self.actions else 0

    @property
    def total_actions(self) -> int:
        return sum(self.actions)


class Scorecard(BaseModel):
    card_id: str = ""
    api_key: str = ""
    cards: dict[str, Card] = Field(default_factory=dict)

    @property
    def won(self) -> int:
        return sum(1 for card in self.cards.values() if card.state is GameState.WIN)

    @property
    def played(self) -> int:
        return sum(1 for card in self.cards.values() if card.started)

    @property
    def total_actions(self) -> int:
        return sum(card.total_actions for card in self.cards.values())

    def get(self, game_id: str | None = None) -> dict[str, Any]:
        if game_id is not None:
            return {game_id: self.cards[game_id].model_dump()}
        return {key: card.model_dump() for key, card in self.cards.items()}

    def get_json_for(self, game_id: str) -> dict[str, Any]:
        return {
            "won": self.won,
            "played": self.played,
            "cards": self.get(game_id),
        }


__all__ = [
    "ActionInput",
    "Card",
    "FrameData",
    "GameAction",
    "GameState",
    "Scorecard",
]
