"""Structured observer states for ARC-AGI-3 play."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class FrameComponent:
    color: int
    area: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    edge_touch: bool = False

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1


@dataclass(frozen=True)
class ClickPoint:
    x: int
    y: int
    source: str
    salience: float = 0.0
    color: int | None = None

    def key(self, bucket: int = 4) -> tuple[int, int]:
        return (self.x // bucket, self.y // bucket)


@dataclass
class FrameSignature:
    frame_hash: str
    width: int
    height: int
    background: int
    histogram: dict[int, int]
    non_background_ratio: float
    components: list[FrameComponent] = field(default_factory=list)
    salience_points: list[ClickPoint] = field(default_factory=list)

    def compact(self) -> dict[str, Any]:
        return {
            "hash": self.frame_hash[:12],
            "size": [self.width, self.height],
            "background": self.background,
            "non_bg": round(self.non_background_ratio, 4),
            "components": len(self.components),
            "points": [asdict(point) for point in self.salience_points[:8]],
        }


@dataclass
class SelfState:
    plan: list[str] = field(default_factory=list)
    novelty_budget: float = 0.62
    uncertainty: dict[str, float] = field(default_factory=dict)
    recent_commitments: list[str] = field(default_factory=list)
    suppressed_options: list[str] = field(default_factory=list)
    current_hypotheses: list[str] = field(default_factory=list)


@dataclass
class OtherState:
    expected_baselines: list[str] = field(default_factory=list)
    likely_common_actions: list[str] = field(default_factory=list)
    cliche_patterns: list[str] = field(default_factory=list)


@dataclass
class WorldState:
    game_id: str
    frame: FrameSignature
    game_state: str
    levels_completed: int
    win_levels: int
    available_actions: list[int]
    action_index: int
    score_delta: int = 0
    frame_changed: bool = False
    hard_constraints: list[str] = field(default_factory=list)


@dataclass
class MemoryState:
    visited_state_counts: dict[str, int] = field(default_factory=dict)
    recent_state_hashes: list[str] = field(default_factory=list)
    recent_action_keys: list[str] = field(default_factory=list)
    exact_success_anchors: list[str] = field(default_factory=list)
    semantic_associations: list[str] = field(default_factory=list)


@dataclass
class ActionCandidate:
    action_id: int
    x: int | None = None
    y: int | None = None
    source: str = "policy"
    base_score: float = 0.0
    novelty_score: float = 0.0
    value_score: float = 0.0
    coherence_score: float = 0.0
    grounding_score: float = 1.0
    repetition_penalty: float = 0.0
    cliche_penalty: float = 0.0
    contradiction_penalty: float = 0.0
    final_score: float = 0.0
    explanation: str = ""

    @property
    def is_complex(self) -> bool:
        return self.action_id == 6

    def key(self) -> str:
        if self.x is None or self.y is None:
            return f"A{self.action_id}"
        return f"A{self.action_id}:{self.x // 4}:{self.y // 4}"

    def reasoning(self, state: "ObserverArcState") -> dict[str, Any]:
        return {
            "policy": "observer_arc",
            "action": self.key(),
            "score": round(self.final_score, 4),
            "why": self.explanation[:240],
            "self_state": {
                "plan": state.self_state.plan[-4:],
                "hypotheses": state.self_state.current_hypotheses[-4:],
                "suppressed": state.self_state.suppressed_options[-4:],
            },
            "other_state": {
                "avoids": state.other_state.cliche_patterns[:4],
                "baselines": state.other_state.expected_baselines[:3],
            },
            "memory": {
                "recent_actions": state.memory_state.recent_action_keys[-6:],
                "anchors": state.memory_state.exact_success_anchors[-4:],
            },
        }


@dataclass
class ObserverArcState:
    self_state: SelfState = field(default_factory=SelfState)
    other_state: OtherState = field(default_factory=OtherState)
    memory_state: MemoryState = field(default_factory=MemoryState)
    last_world: WorldState | None = None
    selected_candidates: list[ActionCandidate] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
