"""Comparator-guided action and subgoal scoring for ARC play."""

from __future__ import annotations

from dataclasses import dataclass, field


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class ActionScoreSignals:
    """Normalized observer-centric signals used for action reranking."""

    progress: float = 0.0
    information_gain: float = 0.0
    goal_affinity: float = 0.0
    consistency: float = 0.0
    empowerment: float = 0.0
    mortality_risk: float = 0.0
    repeat_penalty: float = 0.0
    budget_cost: float = 0.0

    def normalized(self) -> "ActionScoreSignals":
        return ActionScoreSignals(
            progress=_clamp(self.progress),
            information_gain=_clamp(self.information_gain),
            goal_affinity=_clamp(self.goal_affinity),
            consistency=_clamp(self.consistency),
            empowerment=_clamp(self.empowerment),
            mortality_risk=_clamp(self.mortality_risk),
            repeat_penalty=_clamp(self.repeat_penalty),
            budget_cost=_clamp(self.budget_cost),
        )

    def compact(self) -> str:
        signals = self.normalized()
        return (
            f"p={signals.progress:.2f},i={signals.information_gain:.2f},"
            f"g={signals.goal_affinity:.2f},c={signals.consistency:.2f},"
            f"e={signals.empowerment:.2f},m={signals.mortality_risk:.2f},"
            f"r={signals.repeat_penalty:.2f},b={signals.budget_cost:.2f}"
        )


@dataclass(frozen=True)
class ActionSubgoalComparator:
    """Bounded guidance layer for valuable divergence over action sequences.

    The weights mirror the project guidance: progress and information matter
    most, goal affinity and transition consistency keep exploration useful,
    and repeat/budget costs prevent blind loops.
    """

    weights: dict[str, float] = field(
        default_factory=lambda: {
            "progress": 0.35,
            "information_gain": 0.20,
            "goal_affinity": 0.20,
            "consistency": 0.15,
            "empowerment": 0.10,
            "mortality_risk": 0.12,
            "repeat_penalty": 0.07,
            "budget_cost": 0.03,
        }
    )
    min_adjustment: float = -0.18
    max_adjustment: float = 0.28

    def adjustment(self, signals: ActionScoreSignals) -> float:
        normalized = signals.normalized()
        raw = (
            self.weights["progress"] * normalized.progress
            + self.weights["information_gain"] * normalized.information_gain
            + self.weights["goal_affinity"] * normalized.goal_affinity
            + self.weights["consistency"] * normalized.consistency
            + self.weights["empowerment"] * normalized.empowerment
            - self.weights["mortality_risk"] * normalized.mortality_risk
            - self.weights["repeat_penalty"] * normalized.repeat_penalty
            - self.weights["budget_cost"] * normalized.budget_cost
        )
        return _clamp(raw, self.min_adjustment, self.max_adjustment)

    def score(self, base_score: float, signals: ActionScoreSignals) -> float:
        return base_score + self.adjustment(signals)
