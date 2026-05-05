"""Episode-local viability and empowerment memory for ARC play."""

from __future__ import annotations

from dataclasses import dataclass, field

from .state import ActionCandidate, WorldState
from .state_graph import build_state_graph, transition_delta


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class ViabilityEstimate:
    """Risk and option-preservation estimate for one candidate action."""

    mortality_risk: float = 0.0
    empowerment: float = 0.5
    empowerment_delta: float = 0.0
    confidence: float = 0.0
    reason: str = ""


@dataclass
class ViabilityStats:
    count: int = 0
    total_mortality: float = 0.0
    total_empowerment_after: float = 0.0
    total_empowerment_delta: float = 0.0
    terminal_count: int = 0
    irreversible_count: int = 0
    mobility_loss_count: int = 0
    action_collapse_count: int = 0
    no_change_count: int = 0
    total_reward: float = 0.0

    @property
    def average_mortality(self) -> float:
        reward_offset = 0.30 * max(0.0, self.total_reward / max(1, self.count))
        return _clamp((self.total_mortality / max(1, self.count)) - reward_offset)

    @property
    def average_empowerment(self) -> float:
        if self.count <= 0:
            return 0.5
        return _clamp(self.total_empowerment_after / self.count)

    @property
    def average_empowerment_delta(self) -> float:
        if self.count <= 0:
            return 0.0
        return max(-1.0, min(1.0, self.total_empowerment_delta / self.count))

    @property
    def confidence(self) -> float:
        support = min(1.0, self.count / 3.0)
        signal = min(1.0, (self.average_mortality + abs(self.average_empowerment_delta)) / 0.45)
        return support * signal

    def reason(self) -> str:
        if self.terminal_count:
            return "terminal loss"
        if self.action_collapse_count:
            return "action-option collapse"
        if self.mobility_loss_count:
            return "mobility loss"
        if self.irreversible_count:
            return "irreversible-looking object loss"
        if self.no_change_count:
            return "loop or blocked action"
        return "low viability"


@dataclass
class ViabilityModel:
    """Learn which observed transitions preserve or destroy future options."""

    max_records: int = 2048
    stats: dict[tuple[str, str], ViabilityStats] = field(default_factory=dict)

    def record(
        self,
        previous: WorldState,
        action: ActionCandidate,
        current: WorldState,
        reward: float,
        changed: bool,
        game_over: bool,
    ) -> None:
        delta = transition_delta(previous.frame, current.frame, action.key())
        before_empowerment = approximate_empowerment(previous)
        after_empowerment = approximate_empowerment(current)
        empowerment_drop = max(0.0, before_empowerment - after_empowerment)
        action_collapse = action_option_collapse(previous.available_actions, current.available_actions)
        mobility_loss = directional_option_collapse(previous.available_actions, current.available_actions)
        irreversible_loss = (
            delta.destroyed_count > delta.created_count
            and reward <= 0.0
            and not game_over
        )
        no_progress = not changed and reward <= 0.0
        unexplained_risk = 0.0
        if reward <= 0.0 and delta.family in {"remove_or_hide", "relation_change", "state_change"}:
            unexplained_risk = min(1.0, 0.35 + delta.information_gain)
        mortality = _clamp(
            1.00 * int(game_over)
            + 0.70 * int(irreversible_loss)
            + 0.50 * mobility_loss
            + 0.45 * action_collapse
            + 0.40 * empowerment_drop
            + 0.20 * unexplained_risk
            + 0.16 * int(no_progress)
        )
        graph = build_state_graph(previous.frame)
        for key in self._keys(graph.profile, action):
            stats = self.stats.setdefault(key, ViabilityStats())
            stats.count += 1
            stats.total_mortality += mortality
            stats.total_empowerment_after += after_empowerment
            stats.total_empowerment_delta += after_empowerment - before_empowerment
            stats.terminal_count += int(game_over)
            stats.irreversible_count += int(irreversible_loss)
            stats.mobility_loss_count += int(mobility_loss > 0.0)
            stats.action_collapse_count += int(action_collapse > 0.0)
            stats.no_change_count += int(no_progress)
            stats.total_reward += reward
        if len(self.stats) > self.max_records:
            for key in list(self.stats)[: len(self.stats) - self.max_records]:
                self.stats.pop(key, None)

    def estimate(
        self,
        world: WorldState,
        action: ActionCandidate,
        max_actions: int,
    ) -> ViabilityEstimate:
        graph = build_state_graph(world.frame)
        weighted_risk = 0.0
        weighted_empowerment = 0.0
        weighted_delta = 0.0
        total_weight = 0.0
        best_reason = ""
        best_weight = 0.0
        for priority, key in enumerate(self._keys(graph.profile, action), start=1):
            stats = self.stats.get(key)
            if stats is None:
                continue
            weight = stats.confidence / priority
            weighted_risk += stats.average_mortality * weight
            weighted_empowerment += stats.average_empowerment * weight
            weighted_delta += stats.average_empowerment_delta * weight
            total_weight += weight
            if weight > best_weight:
                best_reason = stats.reason()
                best_weight = weight
        current_empowerment = approximate_empowerment(world)
        timeout_pressure = _clamp(world.action_index / max(1, max_actions))
        if total_weight <= 0.0:
            return ViabilityEstimate(
                mortality_risk=0.03 * timeout_pressure,
                empowerment=current_empowerment,
                empowerment_delta=0.0,
                confidence=0.0,
                reason="unseen action",
            )
        risk = _clamp(weighted_risk / total_weight + 0.03 * timeout_pressure)
        empowerment = _clamp(weighted_empowerment / total_weight)
        return ViabilityEstimate(
            mortality_risk=risk,
            empowerment=empowerment,
            empowerment_delta=max(-1.0, min(1.0, weighted_delta / total_weight)),
            confidence=min(1.0, total_weight),
            reason=best_reason,
        )

    def hypotheses(
        self,
        world: WorldState,
        action: ActionCandidate,
        max_actions: int,
    ) -> list[str]:
        estimate = self.estimate(world, action, max_actions)
        if estimate.confidence <= 0.0:
            return []
        out: list[str] = []
        if estimate.mortality_risk >= 0.18:
            out.append(
                f"{action.key()} has episode-local viability risk "
                f"({estimate.reason}, risk={estimate.mortality_risk:.2f})."
            )
        if estimate.empowerment_delta >= 0.08:
            out.append(
                f"{action.key()} tends to preserve or expand future options "
                f"(empowerment_delta={estimate.empowerment_delta:.2f})."
            )
        return out

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


def approximate_empowerment(world: WorldState) -> float:
    actions = set(world.available_actions)
    action_diversity = len(actions) / 7.0
    directional = len(actions & {1, 2, 3, 4}) / 4.0
    click = 0.18 if 6 in actions and world.frame.salience_points else 0.0
    visible_regions = min(1.0, len(world.frame.components) / 12.0)
    frame_richness = min(1.0, world.frame.non_background_ratio * 4.0)
    terminal_penalty = 0.55 if world.game_state == "GAME_OVER" else 0.0
    return _clamp(
        0.33 * action_diversity
        + 0.24 * directional
        + click
        + 0.16 * visible_regions
        + 0.12 * frame_richness
        - terminal_penalty
    )


def action_option_collapse(previous: list[int], current: list[int]) -> float:
    before = set(previous)
    after = set(current)
    if not before:
        return 0.0
    lost = len(before - after) / max(1, len(before))
    return _clamp(lost)


def directional_option_collapse(previous: list[int], current: list[int]) -> float:
    before = set(previous) & {1, 2, 3, 4}
    after = set(current) & {1, 2, 3, 4}
    if not before:
        return 0.0
    return _clamp(len(before - after) / max(1, len(before)))
