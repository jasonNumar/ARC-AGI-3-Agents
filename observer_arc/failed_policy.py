"""Memory for obvious-but-failed policies in ARC state sequences."""

from __future__ import annotations

from dataclasses import dataclass, field

from .state import ActionCandidate, FrameSignature
from .state_graph import build_state_graph


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class FailedPolicyEstimate:
    failure_risk: float = 0.0
    confidence: float = 0.0
    reason: str = ""


@dataclass
class FailedPolicyStats:
    count: int = 0
    no_change_count: int = 0
    game_over_count: int = 0
    total_badness: float = 0.0
    total_reward: float = 0.0

    @property
    def average_reward(self) -> float:
        return self.total_reward / max(1, self.count)

    @property
    def no_change_rate(self) -> float:
        return self.no_change_count / max(1, self.count)

    @property
    def game_over_rate(self) -> float:
        return self.game_over_count / max(1, self.count)

    @property
    def failure_risk(self) -> float:
        risk = self.total_badness / max(1, self.count)
        success_offset = 0.35 * max(0.0, self.average_reward)
        return _clamp(risk - success_offset)

    @property
    def confidence(self) -> float:
        support = min(1.0, self.count / 3.0)
        signal = min(1.0, self.failure_risk / 0.35)
        return support * signal

    def reason(self) -> str:
        if self.game_over_rate >= 0.25:
            return "game-over branch"
        if self.no_change_rate >= 0.50:
            return "no-progress loop"
        return "low-value action branch"


@dataclass
class FailedPolicyMemory:
    """Track failed baseline moves as part of the functional OtherState."""

    max_records: int = 2048
    stats: dict[tuple[str, str], FailedPolicyStats] = field(default_factory=dict)

    def record(
        self,
        previous: FrameSignature,
        action: ActionCandidate,
        current: FrameSignature,
        reward: float,
        changed: bool,
        game_over: bool,
    ) -> None:
        no_progress = not changed and reward <= 0.0
        badness = (
            0.42 * int(no_progress)
            + 0.95 * int(game_over)
            + 0.30 * max(0.0, -reward)
            - 0.28 * max(0.0, reward)
        )
        graph = build_state_graph(previous)
        for key in self._keys(graph.profile, action):
            stats = self.stats.setdefault(key, FailedPolicyStats())
            stats.count += 1
            stats.no_change_count += int(no_progress)
            stats.game_over_count += int(game_over)
            stats.total_badness += badness
            stats.total_reward += reward
        if len(self.stats) > self.max_records:
            for key in list(self.stats)[: len(self.stats) - self.max_records]:
                self.stats.pop(key, None)

    def estimate(
        self,
        signature: FrameSignature,
        action: ActionCandidate,
    ) -> FailedPolicyEstimate:
        graph = build_state_graph(signature)
        weighted_risk = 0.0
        total_weight = 0.0
        best_reason = ""
        best_weight = 0.0
        for priority, key in enumerate(self._keys(graph.profile, action), start=1):
            stats = self.stats.get(key)
            if stats is None:
                continue
            weight = stats.confidence / priority
            weighted_risk += stats.failure_risk * weight
            total_weight += weight
            if weight > best_weight:
                best_reason = stats.reason()
                best_weight = weight
        if total_weight <= 0.0:
            return FailedPolicyEstimate()
        return FailedPolicyEstimate(
            failure_risk=_clamp(weighted_risk / total_weight),
            confidence=min(1.0, total_weight),
            reason=best_reason,
        )

    def hypotheses(
        self,
        signature: FrameSignature,
        action: ActionCandidate,
    ) -> list[str]:
        estimate = self.estimate(signature, action)
        if estimate.confidence <= 0.0 or estimate.failure_risk < 0.12:
            return []
        reason = estimate.reason or "failed baseline"
        return [
            (
                f"{action.key()} resembles a {reason} "
                f"(failure_risk={estimate.failure_risk:.2f})."
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
