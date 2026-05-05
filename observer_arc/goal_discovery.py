"""Lightweight ARC-native goal discovery over observed transitions."""

from __future__ import annotations

from dataclasses import dataclass, field

from .state import ActionCandidate, FrameSignature
from .state_graph import build_state_graph, transition_delta


@dataclass(frozen=True)
class GoalEstimate:
    affinity: float = 0.0
    confidence: float = 0.0
    information_gain: float = 0.0
    game_over_rate: float = 0.0
    label: str = ""


@dataclass
class GoalStats:
    count: int = 0
    total_affinity: float = 0.0
    total_information: float = 0.0
    game_over_count: int = 0
    labels: dict[str, int] = field(default_factory=dict)

    @property
    def average_affinity(self) -> float:
        return self.total_affinity / max(1, self.count)

    @property
    def average_information(self) -> float:
        return self.total_information / max(1, self.count)

    @property
    def game_over_rate(self) -> float:
        return self.game_over_count / max(1, self.count)

    @property
    def confidence(self) -> float:
        signal = min(1.0, (abs(self.average_affinity) + self.average_information) / 0.30)
        support = min(1.0, self.count / 2.0)
        safety = 1.0 - min(0.75, self.game_over_rate)
        return signal * support * safety

    def dominant_label(self) -> str:
        if not self.labels:
            return ""
        return max(self.labels.items(), key=lambda item: item[1])[0]


@dataclass
class GoalDiscoveryMemory:
    """Learn which action families tend to produce goal-relevant states."""

    max_records: int = 2048
    goals: dict[tuple[str, str], GoalStats] = field(default_factory=dict)

    def record(
        self,
        previous: FrameSignature,
        action: ActionCandidate,
        current: FrameSignature,
        reward: float,
        game_over: bool,
        previous_actions: list[int] | None = None,
        current_actions: list[int] | None = None,
    ) -> None:
        delta = transition_delta(previous, current, _action_key(action))
        label = _goal_label(delta.family, previous_actions or [], current_actions or [])
        action_change = len(set(current_actions or []) ^ set(previous_actions or []))
        affinity = (
            max(-0.45, min(1.0, reward))
            + 0.25 * delta.information_gain
            + 0.08 * min(3, action_change)
            + 0.05 * min(3, delta.relation_changes)
            - 0.50 * int(game_over)
        )
        graph = build_state_graph(previous)
        for key in self._keys(graph.profile, action):
            stats = self.goals.setdefault(key, GoalStats())
            stats.count += 1
            stats.total_affinity += affinity
            stats.total_information += delta.information_gain
            stats.game_over_count += int(game_over)
            stats.labels[label] = stats.labels.get(label, 0) + 1
        if len(self.goals) > self.max_records:
            for key in list(self.goals)[: len(self.goals) - self.max_records]:
                self.goals.pop(key, None)

    def estimate(
        self,
        signature: FrameSignature,
        action: ActionCandidate,
    ) -> GoalEstimate:
        graph = build_state_graph(signature)
        weighted_affinity = 0.0
        weighted_information = 0.0
        weighted_game_over = 0.0
        total_weight = 0.0
        best_label = ""
        best_weight = 0.0
        for priority, key in enumerate(self._keys(graph.profile, action), start=1):
            stats = self.goals.get(key)
            if stats is None:
                continue
            specificity = 1.0 / priority
            weight = stats.confidence * specificity
            weighted_affinity += stats.average_affinity * weight
            weighted_information += stats.average_information * weight
            weighted_game_over += stats.game_over_rate * weight
            total_weight += weight
            if weight > best_weight:
                best_label = stats.dominant_label()
                best_weight = weight
        if total_weight <= 0.0:
            return GoalEstimate()
        return GoalEstimate(
            affinity=max(-0.45, min(0.75, weighted_affinity / total_weight)),
            confidence=min(1.0, total_weight),
            information_gain=max(0.0, min(1.0, weighted_information / total_weight)),
            game_over_rate=max(0.0, min(1.0, weighted_game_over / total_weight)),
            label=best_label,
        )

    def hypotheses(
        self,
        signature: FrameSignature,
        action: ActionCandidate,
    ) -> list[str]:
        estimate = self.estimate(signature, action)
        if estimate.confidence <= 0.0 or estimate.affinity <= 0.04:
            return []
        label = estimate.label or "candidate subgoal"
        return [
            (
                f"{action.key()} is associated with {label} "
                f"(goal_affinity={estimate.affinity:.2f})."
            )
        ]

    def _keys(
        self,
        profile: str,
        action: ActionCandidate,
    ) -> list[tuple[str, str]]:
        broad = f"A{action.action_id}"
        exact = action.key()
        if exact == broad:
            return [(profile, broad), ("global", broad)]
        return [
            (profile, exact),
            (profile, broad),
            ("global", exact),
            ("global", broad),
        ]


def _goal_label(family: str, previous_actions: list[int], current_actions: list[int]) -> str:
    if set(current_actions) != set(previous_actions):
        return "affordance change"
    if family in {"reveal_or_create", "relation_change", "relational_move"}:
        return family.replace("_", " ")
    if family == "move":
        return "state progress"
    return "information probe"


def _action_key(action: ActionCandidate) -> str:
    if action.action_id == 6 and action.x is not None and action.y is not None:
        return f"A6:{action.x // 4}:{action.y // 4}"
    return f"A{action.action_id}"
