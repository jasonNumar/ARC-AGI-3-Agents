"""Transition motif memory for ARC-native state sequences."""

from __future__ import annotations

from dataclasses import dataclass, field

from .state import ActionCandidate, FrameSignature
from .state_graph import StateGraph, build_state_graph, transition_delta


@dataclass(frozen=True)
class MotifEstimate:
    value: float = 0.0
    information_gain: float = 0.0
    confidence: float = 0.0
    game_over_rate: float = 0.0
    family: str = ""


@dataclass
class MotifStats:
    count: int = 0
    total_value: float = 0.0
    total_information: float = 0.0
    game_over_count: int = 0
    families: dict[str, int] = field(default_factory=dict)

    @property
    def average_value(self) -> float:
        return self.total_value / max(1, self.count)

    @property
    def average_information(self) -> float:
        return self.total_information / max(1, self.count)

    @property
    def game_over_rate(self) -> float:
        return self.game_over_count / max(1, self.count)

    @property
    def confidence(self) -> float:
        signal = min(1.0, (abs(self.average_value) + self.average_information) / 0.28)
        support = min(1.0, self.count / 2.0)
        safety = 1.0 - min(0.70, self.game_over_rate)
        return signal * support * safety

    def dominant_family(self) -> str:
        if not self.families:
            return ""
        return max(self.families.items(), key=lambda item: item[1])[0]


@dataclass
class TransitionMotifMemory:
    """Store exact and abstract motifs over visible state/action transitions."""

    max_records: int = 4096
    stats: dict[tuple[str, str], MotifStats] = field(default_factory=dict)
    exact_hits: dict[str, int] = field(default_factory=dict)

    def record(
        self,
        previous: FrameSignature,
        action: ActionCandidate,
        current: FrameSignature,
        reward: float,
        game_over: bool,
    ) -> None:
        delta = transition_delta(previous, current, _action_key(action))
        value = (
            max(-0.35, min(1.0, reward))
            + 0.30 * delta.information_gain
            + 0.05 * min(3, delta.relation_changes)
        )
        for key in self._keys(delta.previous, action):
            stats = self.stats.setdefault(key, MotifStats())
            stats.count += 1
            stats.total_value += value
            stats.total_information += delta.information_gain
            stats.game_over_count += int(game_over)
            stats.families[delta.family] = stats.families.get(delta.family, 0) + 1
        exact = delta.compact_signature()
        self.exact_hits[exact] = self.exact_hits.get(exact, 0) + 1
        if len(self.stats) > self.max_records:
            for key in list(self.stats)[: len(self.stats) - self.max_records]:
                self.stats.pop(key, None)
        if len(self.exact_hits) > self.max_records:
            for key in list(self.exact_hits)[: len(self.exact_hits) - self.max_records]:
                self.exact_hits.pop(key, None)

    def estimate(
        self,
        signature: FrameSignature,
        action: ActionCandidate,
    ) -> MotifEstimate:
        graph = build_state_graph(signature)
        weighted_value = 0.0
        weighted_information = 0.0
        weighted_game_over = 0.0
        total_weight = 0.0
        best_family = ""
        best_family_weight = 0.0
        for priority, key in enumerate(self._keys(graph, action), start=1):
            stats = self.stats.get(key)
            if stats is None:
                continue
            specificity = 1.0 / priority
            weight = stats.confidence * specificity
            weighted_value += stats.average_value * weight
            weighted_information += stats.average_information * weight
            weighted_game_over += stats.game_over_rate * weight
            total_weight += weight
            if weight > best_family_weight:
                best_family = stats.dominant_family()
                best_family_weight = weight
        if total_weight <= 0.0:
            return MotifEstimate()
        return MotifEstimate(
            value=max(-0.35, min(0.65, weighted_value / total_weight)),
            information_gain=max(0.0, min(1.0, weighted_information / total_weight)),
            confidence=min(1.0, total_weight),
            game_over_rate=max(0.0, min(1.0, weighted_game_over / total_weight)),
            family=best_family,
        )

    def hypotheses(
        self,
        signature: FrameSignature,
        action: ActionCandidate,
    ) -> list[str]:
        estimate = self.estimate(signature, action)
        if estimate.confidence <= 0.0 or not estimate.family:
            return []
        return [
            (
                f"{action.key()} matches a {estimate.family} transition motif "
                f"(confidence={estimate.confidence:.2f})."
            )
        ]

    def _keys(
        self,
        graph: StateGraph,
        action: ActionCandidate,
    ) -> list[tuple[str, str]]:
        broad = f"A{action.action_id}"
        exact = action.key()
        if exact == broad:
            return [(graph.profile, broad), ("global", broad)]
        return [
            (graph.profile, exact),
            (graph.profile, broad),
            ("global", exact),
            ("global", broad),
        ]


def _action_key(action: ActionCandidate) -> str:
    if action.action_id == 6 and action.x is not None and action.y is not None:
        return f"A6:{action.x // 4}:{action.y // 4}"
    return f"A{action.action_id}"
