"""Latent role inference from observed ARC state transitions.

Roles here are functional and episode-local. They are inferred from how an
action changes future possibilities: score, terminal state, object continuity,
available actions, and empowerment. The module does not use game ids, sprite
labels, hidden state, or public action traces.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .state import ActionCandidate, WorldState
from .state_graph import build_state_graph, transition_delta
from .viability import (
    action_option_collapse,
    approximate_empowerment,
    directional_option_collapse,
)


POSITIVE_ROLES = {"target", "key", "trigger", "resource", "protector", "probe"}
RISK_ROLES = {"threat", "trap", "hazard", "obstacle"}


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class LatentRoleEstimate:
    """Functional role estimate for one candidate action."""

    role: str = ""
    confidence: float = 0.0
    value: float = 0.0
    risk: float = 0.0
    information_gain: float = 0.0
    empowerment_delta: float = 0.0
    reason: str = ""


@dataclass
class LatentRoleStats:
    count: int = 0
    total_value: float = 0.0
    total_risk: float = 0.0
    total_information: float = 0.0
    total_empowerment_delta: float = 0.0
    roles: dict[str, int] = field(default_factory=dict)
    reasons: dict[str, int] = field(default_factory=dict)

    @property
    def average_value(self) -> float:
        return self.total_value / max(1, self.count)

    @property
    def average_risk(self) -> float:
        return self.total_risk / max(1, self.count)

    @property
    def average_information(self) -> float:
        return self.total_information / max(1, self.count)

    @property
    def average_empowerment_delta(self) -> float:
        return self.total_empowerment_delta / max(1, self.count)

    @property
    def confidence(self) -> float:
        support = min(1.0, self.count / 3.0)
        signal = min(
            1.0,
            (
                abs(self.average_value)
                + self.average_risk
                + self.average_information
                + abs(self.average_empowerment_delta)
            )
            / 0.42,
        )
        return support * signal

    def dominant_role(self) -> str:
        if not self.roles:
            return ""
        return max(self.roles.items(), key=lambda item: item[1])[0]

    def dominant_reason(self) -> str:
        if not self.reasons:
            return ""
        return max(self.reasons.items(), key=lambda item: item[1])[0]


@dataclass
class LatentRoleMemory:
    """Learn action roles from transition effects on future option-space."""

    max_records: int = 2048
    stats: dict[tuple[str, str], LatentRoleStats] = field(default_factory=dict)

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
        empowerment_delta = after_empowerment - before_empowerment
        previous_actions = set(previous.available_actions)
        current_actions = set(current.available_actions)
        gained_actions = len(current_actions - previous_actions)
        lost_actions = len(previous_actions - current_actions)
        action_expansion = _clamp(gained_actions / max(1, len(previous_actions)))
        action_collapse = action_option_collapse(
            previous.available_actions,
            current.available_actions,
        )
        mobility_loss = directional_option_collapse(
            previous.available_actions,
            current.available_actions,
        )
        role, reason = _infer_role(
            reward=reward,
            changed=changed,
            game_over=game_over,
            previous_levels=previous.levels_completed,
            current_levels=current.levels_completed,
            previous_game_state=previous.game_state,
            current_game_state=current.game_state,
            transition_family=delta.family,
            information_gain=delta.information_gain,
            created_count=delta.created_count,
            destroyed_count=delta.destroyed_count,
            moved_count=delta.moved_count,
            relation_changes=delta.relation_changes,
            action_expansion=action_expansion,
            action_collapse=action_collapse,
            mobility_loss=mobility_loss,
            empowerment_delta=empowerment_delta,
            lost_actions=lost_actions,
        )
        if not role:
            return

        value, risk = _role_value_and_risk(
            role=role,
            reward=reward,
            information_gain=delta.information_gain,
            action_expansion=action_expansion,
            action_collapse=action_collapse,
            mobility_loss=mobility_loss,
            empowerment_delta=empowerment_delta,
            game_over=game_over,
        )
        graph = build_state_graph(previous.frame)
        for key in self._keys(graph.profile, action):
            stats = self.stats.setdefault(key, LatentRoleStats())
            stats.count += 1
            stats.total_value += value
            stats.total_risk += risk
            stats.total_information += delta.information_gain
            stats.total_empowerment_delta += empowerment_delta
            stats.roles[role] = stats.roles.get(role, 0) + 1
            stats.reasons[reason] = stats.reasons.get(reason, 0) + 1

        if len(self.stats) > self.max_records:
            for key in list(self.stats)[: len(self.stats) - self.max_records]:
                self.stats.pop(key, None)

    def estimate(
        self,
        world: WorldState,
        action: ActionCandidate,
    ) -> LatentRoleEstimate:
        graph = build_state_graph(world.frame)
        weighted_value = 0.0
        weighted_risk = 0.0
        weighted_information = 0.0
        weighted_empowerment_delta = 0.0
        total_weight = 0.0
        role_votes: dict[str, float] = {}
        reason_votes: dict[str, float] = {}

        for priority, key in enumerate(self._keys(graph.profile, action), start=1):
            stats = self.stats.get(key)
            if stats is None:
                continue
            weight = stats.confidence / priority
            if weight <= 0.0:
                continue
            weighted_value += stats.average_value * weight
            weighted_risk += stats.average_risk * weight
            weighted_information += stats.average_information * weight
            weighted_empowerment_delta += stats.average_empowerment_delta * weight
            total_weight += weight
            role = stats.dominant_role()
            reason = stats.dominant_reason()
            if role:
                role_votes[role] = role_votes.get(role, 0.0) + weight
            if reason:
                reason_votes[reason] = reason_votes.get(reason, 0.0) + weight

        if total_weight <= 0.0:
            return LatentRoleEstimate()

        role = max(role_votes.items(), key=lambda item: item[1])[0] if role_votes else ""
        reason = max(reason_votes.items(), key=lambda item: item[1])[0] if reason_votes else ""
        return LatentRoleEstimate(
            role=role,
            confidence=min(1.0, total_weight),
            value=max(-0.45, min(0.55, weighted_value / total_weight)),
            risk=_clamp(weighted_risk / total_weight),
            information_gain=_clamp(weighted_information / total_weight),
            empowerment_delta=max(-1.0, min(1.0, weighted_empowerment_delta / total_weight)),
            reason=reason,
        )

    def hypotheses(
        self,
        world: WorldState,
        action: ActionCandidate,
    ) -> list[str]:
        estimate = self.estimate(world, action)
        if estimate.confidence <= 0.0 or not estimate.role:
            return []
        if estimate.role in RISK_ROLES and estimate.risk >= 0.12:
            return [
                (
                    f"{action.key()} behaves like a latent {estimate.role} role "
                    f"because it {estimate.reason} (risk={estimate.risk:.2f})."
                )
            ]
        if estimate.role in POSITIVE_ROLES and estimate.value >= 0.04:
            return [
                (
                    f"{action.key()} behaves like a latent {estimate.role} role "
                    f"because it {estimate.reason} (value={estimate.value:.2f})."
                )
            ]
        return []

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


def _infer_role(
    *,
    reward: float,
    changed: bool,
    game_over: bool,
    previous_levels: int,
    current_levels: int,
    previous_game_state: str,
    current_game_state: str,
    transition_family: str,
    information_gain: float,
    created_count: int,
    destroyed_count: int,
    moved_count: int,
    relation_changes: int,
    action_expansion: float,
    action_collapse: float,
    mobility_loss: float,
    empowerment_delta: float,
    lost_actions: int,
) -> tuple[str, str]:
    if game_over or current_game_state == "GAME_OVER" or reward < -0.40:
        return "threat", "ended or severely damaged the episode"
    if current_game_state == "WIN" or current_levels > previous_levels or reward > 0.55:
        return "target", "moved the episode toward completion"
    if action_collapse > 0.30 or mobility_loss > 0.30:
        return "trap", "collapsed available actions or mobility"
    if action_expansion > 0.0:
        return "key", "expanded the available action boundary"
    if transition_family == "reveal_or_create" or created_count > destroyed_count:
        return "key", "revealed or created new reachable state"
    if transition_family == "relation_change" and reward >= 0.0:
        return "trigger", "changed object relations without immediate damage"
    if transition_family == "remove_or_hide" and reward <= 0.0:
        return "hazard", "removed visible state without payoff"
    if not changed and reward <= 0.0:
        return "obstacle", "blocked progress or created a no-change branch"
    if transition_family == "relational_move" and reward >= 0.0 and empowerment_delta >= -0.05:
        return "protector", "kept a moving relation viable"
    if moved_count and empowerment_delta > 0.05:
        return "resource", "increased future options through motion"
    if information_gain > 0.08 and lost_actions == 0 and previous_game_state == "NOT_FINISHED":
        return "probe", "revealed useful transition information"
    if relation_changes and reward >= 0.0:
        return "trigger", "altered object relations"
    return "", ""


def _role_value_and_risk(
    *,
    role: str,
    reward: float,
    information_gain: float,
    action_expansion: float,
    action_collapse: float,
    mobility_loss: float,
    empowerment_delta: float,
    game_over: bool,
) -> tuple[float, float]:
    if role == "target":
        return (
            min(0.55, 0.34 + 0.22 * max(0.0, reward) + 0.12 * information_gain),
            0.02,
        )
    if role == "key":
        return (
            min(
                0.48,
                0.22
                + 0.18 * action_expansion
                + 0.12 * information_gain
                + 0.10 * max(0.0, empowerment_delta),
            ),
            0.04,
        )
    if role == "trigger":
        return (min(0.36, 0.18 + 0.16 * information_gain), 0.08)
    if role == "resource":
        return (min(0.34, 0.16 + 0.18 * max(0.0, empowerment_delta)), 0.06)
    if role == "protector":
        return (min(0.30, 0.14 + 0.10 * information_gain), 0.08)
    if role == "probe":
        return (min(0.22, 0.08 + 0.18 * information_gain), 0.05)
    if role == "threat":
        return (-0.40, min(1.0, 0.82 + 0.18 * int(game_over)))
    if role == "trap":
        return (-0.30, max(0.48, 0.62 * max(action_collapse, mobility_loss)))
    if role == "hazard":
        return (-0.22, 0.44 + 0.20 * information_gain)
    if role == "obstacle":
        return (-0.12, 0.30)
    return (0.0, 0.0)
