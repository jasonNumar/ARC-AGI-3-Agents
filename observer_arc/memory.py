"""Exact and semantic memory for ARC action selection."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .state import ActionCandidate, FrameSignature
from .vision import frame_distance


@dataclass
class TransitionStats:
    count: int = 0
    total_reward: float = 0.0
    changed_count: int = 0
    game_over_count: int = 0
    next_hashes: dict[str, int] = field(default_factory=dict)

    @property
    def average_reward(self) -> float:
        return self.total_reward / max(1, self.count)

    @property
    def change_rate(self) -> float:
        return self.changed_count / max(1, self.count)

    @property
    def game_over_rate(self) -> float:
        return self.game_over_count / max(1, self.count)

    def dominant_next_hash(self) -> str | None:
        if not self.next_hashes:
            return None
        return max(self.next_hashes.items(), key=lambda item: item[1])[0]


@dataclass
class ExactTransitionMemory:
    max_states: int = 4096
    outcomes: dict[tuple[str, str], TransitionStats] = field(default_factory=dict)
    state_action_counts: dict[tuple[str, str], int] = field(default_factory=dict)

    def record(
        self,
        state_hash: str,
        action_key: str,
        next_hash: str,
        reward: float,
        changed: bool,
        game_over: bool,
    ) -> None:
        key = (state_hash, action_key)
        stats = self.outcomes.setdefault(key, TransitionStats())
        stats.count += 1
        stats.total_reward += reward
        stats.changed_count += int(changed)
        stats.game_over_count += int(game_over)
        stats.next_hashes[next_hash] = stats.next_hashes.get(next_hash, 0) + 1
        self.state_action_counts[key] = self.state_action_counts.get(key, 0) + 1
        if len(self.outcomes) > self.max_states:
            oldest_key = next(iter(self.outcomes))
            self.outcomes.pop(oldest_key, None)
            self.state_action_counts.pop(oldest_key, None)

    def get(self, state_hash: str, action_key: str) -> TransitionStats | None:
        return self.outcomes.get((state_hash, action_key))

    def visit_count(self, state_hash: str, action_key: str) -> int:
        return self.state_action_counts.get((state_hash, action_key), 0)


@dataclass(frozen=True)
class GeneralizedActionEstimate:
    value: float = 0.0
    change_rate: float = 0.0
    game_over_rate: float = 0.0
    confidence: float = 0.0


@dataclass
class GeneralizedActionMemory:
    """Online action-effect memory over abstract frame/action schemas."""

    max_records: int = 2048
    outcomes: dict[tuple[str, str], TransitionStats] = field(default_factory=dict)

    def record(
        self,
        signature: FrameSignature,
        action: ActionCandidate,
        next_signature: FrameSignature,
        reward: float,
        changed: bool,
        game_over: bool,
    ) -> None:
        for key in self._keys(signature, action):
            stats = self.outcomes.setdefault(key, TransitionStats())
            stats.count += 1
            stats.total_reward += reward
            stats.changed_count += int(changed)
            stats.game_over_count += int(game_over)
            stats.next_hashes[next_signature.frame_hash] = (
                stats.next_hashes.get(next_signature.frame_hash, 0) + 1
            )
        if len(self.outcomes) > self.max_records:
            for key in list(self.outcomes)[: len(self.outcomes) - self.max_records]:
                self.outcomes.pop(key, None)

    def estimate(
        self, signature: FrameSignature, action: ActionCandidate
    ) -> GeneralizedActionEstimate:
        weighted_value = 0.0
        weighted_change = 0.0
        weighted_game_over = 0.0
        total_weight = 0.0
        for priority, key in enumerate(self._keys(signature, action), start=1):
            stats = self.outcomes.get(key)
            if not stats:
                continue
            confidence = min(1.0, stats.count / 4.0)
            specificity = 1.0 / priority
            weight = confidence * specificity
            weighted_value += stats.average_reward * weight
            weighted_change += stats.change_rate * weight
            weighted_game_over += stats.game_over_rate * weight
            total_weight += weight
        if total_weight <= 0.0:
            return GeneralizedActionEstimate()
        confidence = min(1.0, total_weight)
        return GeneralizedActionEstimate(
            value=weighted_value / total_weight,
            change_rate=weighted_change / total_weight,
            game_over_rate=weighted_game_over / total_weight,
            confidence=confidence,
        )

    def _keys(
        self, signature: FrameSignature, action: ActionCandidate
    ) -> list[tuple[str, str]]:
        profile = frame_profile(signature)
        schema = action_schema(action)
        broad = f"A{action.action_id}"
        if schema == broad:
            return [(profile, schema), ("global", schema)]
        return [
            (profile, schema),
            (profile, broad),
            ("global", schema),
            ("global", broad),
        ]


@dataclass(frozen=True)
class ActionInformationEstimate:
    information_gain: float = 0.0
    confidence: float = 0.0
    game_over_rate: float = 0.0


@dataclass
class ActionInformationStats:
    count: int = 0
    total_information: float = 0.0
    game_over_count: int = 0

    @property
    def average_information(self) -> float:
        return self.total_information / max(1, self.count)

    @property
    def game_over_rate(self) -> float:
        return self.game_over_count / max(1, self.count)

    @property
    def confidence(self) -> float:
        signal = min(1.0, self.average_information / 0.08)
        support = min(1.0, self.count / 2.0)
        safety = 1.0 - min(0.75, self.game_over_rate)
        return signal * support * safety


@dataclass
class ActionInformationMemory:
    """Learn which actions reveal useful observable state on similar frames."""

    max_records: int = 2048
    outcomes: dict[tuple[str, str], ActionInformationStats] = field(default_factory=dict)

    def record(
        self,
        signature: FrameSignature,
        action: ActionCandidate,
        next_signature: FrameSignature,
        game_over: bool,
    ) -> None:
        visual_delta = frame_distance(signature, next_signature)
        component_delta = abs(
            len(next_signature.components) - len(signature.components)
        ) / max(1, len(signature.components) + len(next_signature.components))
        information = visual_delta + 0.30 * min(1.0, component_delta)
        if information <= 0.004 and not game_over:
            return
        for key in self._keys(signature, action):
            stats = self.outcomes.setdefault(key, ActionInformationStats())
            stats.count += 1
            stats.total_information += information
            stats.game_over_count += int(game_over)
        if len(self.outcomes) > self.max_records:
            for key in list(self.outcomes)[: len(self.outcomes) - self.max_records]:
                self.outcomes.pop(key, None)

    def estimate(
        self, signature: FrameSignature, action: ActionCandidate
    ) -> ActionInformationEstimate:
        weighted_information = 0.0
        weighted_game_over = 0.0
        total_weight = 0.0
        for priority, key in enumerate(self._keys(signature, action), start=1):
            stats = self.outcomes.get(key)
            if stats is None:
                continue
            specificity = 1.0 / priority
            weight = stats.confidence * specificity
            weighted_information += stats.average_information * weight
            weighted_game_over += stats.game_over_rate * weight
            total_weight += weight
        if total_weight <= 0.0:
            return ActionInformationEstimate()
        return ActionInformationEstimate(
            information_gain=max(0.0, min(0.65, weighted_information / total_weight)),
            confidence=min(1.0, total_weight),
            game_over_rate=max(0.0, min(1.0, weighted_game_over / total_weight)),
        )

    def _keys(
        self, signature: FrameSignature, action: ActionCandidate
    ) -> list[tuple[str, str]]:
        schema = action_schema(action)
        broad = f"A{action.action_id}"
        profile = frame_profile(signature)
        if schema == broad:
            return [(profile, schema), ("global", schema)]
        return [
            (profile, schema),
            (profile, broad),
            ("global", schema),
            ("global", broad),
        ]


def frame_profile(signature: FrameSignature) -> str:
    density = _bucket(signature.non_background_ratio, [0.04, 0.12, 0.28, 0.50])
    components = _bucket(float(len(signature.components)), [2, 5, 12, 28])
    largest = max((component.area for component in signature.components), default=0)
    largest_bucket = _bucket(float(largest), [4, 16, 64, 192])
    return (
        f"{min(64, signature.width)}x{min(64, signature.height)}"
        f":d{density}:c{components}:l{largest_bucket}"
    )


def action_schema(action: ActionCandidate) -> str:
    if action.action_id != 6:
        return f"A{action.action_id}"
    source = action.source if action.source in {"component", "contrast-grid", "coverage-grid"} else "click"
    return f"A6:{source}:{_region(action.x, action.y)}"


def action_schema_variants(action: ActionCandidate) -> list[str]:
    schema = action_schema(action)
    broad = f"A{action.action_id}"
    if schema == broad:
        return [schema]
    return [schema, broad]


def _region(x: int | None, y: int | None) -> str:
    if x is None or y is None:
        return "unknown"
    if 24 <= x <= 40 and 24 <= y <= 40:
        return "center"
    horizontal = "w" if x < 32 else "e"
    vertical = "n" if y < 32 else "s"
    return vertical + horizontal


def _bucket(value: float, thresholds: list[float]) -> int:
    for index, threshold in enumerate(thresholds):
        if value <= threshold:
            return index
    return len(thresholds)


@dataclass(frozen=True)
class SequenceCreditEstimate:
    value: float = 0.0
    confidence: float = 0.0
    average_horizon: float = 0.0


@dataclass
class SequenceCreditStats:
    count: int = 0
    total_credit: float = 0.0
    total_horizon: float = 0.0

    @property
    def average_credit(self) -> float:
        return self.total_credit / max(1, self.count)

    @property
    def average_horizon(self) -> float:
        return self.total_horizon / max(1, self.count)

    @property
    def confidence(self) -> float:
        signal = min(1.0, abs(self.average_credit) / 0.35)
        return min(1.0, self.count / 3.0) * signal


@dataclass
class SequenceCreditMemory:
    """Assign delayed outcome credit to recent abstract state/action choices."""

    max_horizon: int = 8
    decay: float = 0.72
    max_records: int = 2048
    credits: dict[tuple[str, str], SequenceCreditStats] = field(default_factory=dict)

    def record(
        self,
        trace: list[tuple[FrameSignature, ActionCandidate]],
        terminal_reward: float,
    ) -> None:
        if not trace or abs(terminal_reward) < 0.25:
            return
        recent = trace[-self.max_horizon :]
        for horizon, (signature, action) in enumerate(reversed(recent)):
            credit = terminal_reward * (self.decay ** horizon)
            for priority, key in enumerate(self._keys(signature, action), start=1):
                stats = self.credits.setdefault(key, SequenceCreditStats())
                specificity = 1.0 / priority
                stats.count += 1
                stats.total_credit += credit * specificity
                stats.total_horizon += float(horizon + 1)
        if len(self.credits) > self.max_records:
            for key in list(self.credits)[: len(self.credits) - self.max_records]:
                self.credits.pop(key, None)

    def estimate(
        self, signature: FrameSignature, action: ActionCandidate
    ) -> SequenceCreditEstimate:
        weighted_credit = 0.0
        weighted_horizon = 0.0
        total_weight = 0.0
        for priority, key in enumerate(self._keys(signature, action), start=1):
            stats = self.credits.get(key)
            if stats is None:
                continue
            specificity = 1.0 / priority
            weight = stats.confidence * specificity
            weighted_credit += stats.average_credit * weight
            weighted_horizon += stats.average_horizon * weight
            total_weight += weight
        if total_weight <= 0.0:
            return SequenceCreditEstimate()
        return SequenceCreditEstimate(
            value=max(-0.45, min(0.55, weighted_credit / total_weight)),
            confidence=min(1.0, total_weight),
            average_horizon=weighted_horizon / total_weight,
        )

    def _keys(
        self, signature: FrameSignature, action: ActionCandidate
    ) -> list[tuple[str, str]]:
        schema = action_schema(action)
        broad = f"A{action.action_id}"
        profile = frame_profile(signature)
        if schema == broad:
            return [(profile, schema), ("global", schema)]
        return [
            (profile, schema),
            (profile, broad),
            ("global", schema),
            ("global", broad),
        ]


@dataclass(frozen=True)
class SequenceFragmentEstimate:
    value: float = 0.0
    confidence: float = 0.0
    matched_length: int = 0
    expected_schema: str = ""


@dataclass
class SequenceFragmentStats:
    count: int = 0
    total_value: float = 0.0
    total_horizon: float = 0.0
    total_length: float = 0.0

    @property
    def average_value(self) -> float:
        return self.total_value / max(1, self.count)

    @property
    def average_horizon(self) -> float:
        return self.total_horizon / max(1, self.count)

    @property
    def average_length(self) -> float:
        return self.total_length / max(1, self.count)

    @property
    def confidence(self) -> float:
        signal = min(1.0, abs(self.average_value) / 0.30)
        support = min(1.0, self.count / 2.0)
        return signal * support


@dataclass
class SequenceFragmentMemory:
    """Learn reusable next-action fragments from rewarded action sequences."""

    max_prefix: int = 3
    decay: float = 0.78
    max_records: int = 2048
    fragments: dict[tuple[str, tuple[str, ...], str], SequenceFragmentStats] = field(default_factory=dict)

    def record(
        self,
        trace: list[tuple[FrameSignature, ActionCandidate]],
        terminal_reward: float,
    ) -> None:
        if len(trace) < 2 or abs(terminal_reward) < 0.25:
            return
        schemas = [action_schema(action) for _, action in trace]
        for next_index in range(1, len(trace)):
            signature, action = trace[next_index]
            horizon = len(trace) - next_index
            value = terminal_reward * (self.decay ** max(0, horizon - 1))
            profiles = [frame_profile(signature), "global"]
            for prefix_length in range(1, min(self.max_prefix, next_index) + 1):
                prefix = tuple(schemas[next_index - prefix_length : next_index])
                for profile in profiles:
                    for next_schema in action_schema_variants(action):
                        key = (profile, prefix, next_schema)
                        stats = self.fragments.setdefault(key, SequenceFragmentStats())
                        stats.count += 1
                        stats.total_value += value
                        stats.total_horizon += float(horizon)
                        stats.total_length += float(prefix_length)
        if len(self.fragments) > self.max_records:
            for key in list(self.fragments)[: len(self.fragments) - self.max_records]:
                self.fragments.pop(key, None)

    def estimate(
        self,
        signature: FrameSignature,
        recent_schemas: list[str],
        action: ActionCandidate,
    ) -> SequenceFragmentEstimate:
        if not recent_schemas:
            return SequenceFragmentEstimate()
        weighted_value = 0.0
        total_weight = 0.0
        best_length = 0
        best_schema = ""
        profiles = [(frame_profile(signature), 1.0), ("global", 0.55)]
        variants = action_schema_variants(action)
        max_prefix = min(self.max_prefix, len(recent_schemas))
        for prefix_length in range(max_prefix, 0, -1):
            prefix = tuple(recent_schemas[-prefix_length:])
            length_weight = 1.0 + 0.35 * (prefix_length - 1)
            for profile, profile_weight in profiles:
                for schema_index, schema in enumerate(variants, start=1):
                    stats = self.fragments.get((profile, prefix, schema))
                    if stats is None:
                        continue
                    schema_weight = 1.0 / schema_index
                    weight = stats.confidence * length_weight * profile_weight * schema_weight
                    weighted_value += stats.average_value * weight
                    total_weight += weight
                    if prefix_length > best_length:
                        best_length = prefix_length
                        best_schema = schema
        if total_weight <= 0.0:
            return SequenceFragmentEstimate()
        return SequenceFragmentEstimate(
            value=max(-0.45, min(0.60, weighted_value / total_weight)),
            confidence=min(1.0, total_weight),
            matched_length=best_length,
            expected_schema=best_schema,
        )


@dataclass(frozen=True)
class TrajectoryEstimate:
    value: float = 0.0
    boundary_risk: float = 0.0
    confidence: float = 0.0
    explanation: str = ""


@dataclass
class ActionMotionStats:
    count: int = 0
    sum_dx: float = 0.0
    sum_dy: float = 0.0
    rewarded_count: int = 0

    @property
    def dx(self) -> float:
        return self.sum_dx / max(1, self.count)

    @property
    def dy(self) -> float:
        return self.sum_dy / max(1, self.count)

    @property
    def confidence(self) -> float:
        magnitude = abs(self.dx) + abs(self.dy)
        consistency = min(1.0, magnitude / 4.0)
        return min(1.0, self.count / 3.0) * consistency


@dataclass
class TrajectoryMemory:
    """Infer action objectives and constraints from sequences of states."""

    action_motion: dict[int, ActionMotionStats] = field(default_factory=dict)

    def record(
        self,
        previous: FrameSignature,
        action: ActionCandidate,
        current: FrameSignature,
        reward: float,
    ) -> None:
        before = _focus_component(previous)
        after = _matched_component(current, before)
        if before is None or after is None:
            return
        dx = after.centroid[0] - before.centroid[0]
        dy = after.centroid[1] - before.centroid[1]
        if abs(dx) + abs(dy) < 0.5:
            return
        stats = self.action_motion.setdefault(action.action_id, ActionMotionStats())
        stats.count += 1
        stats.sum_dx += dx
        stats.sum_dy += dy
        stats.rewarded_count += int(reward > 0.20)

    def estimate(
        self, signature: FrameSignature, action: ActionCandidate
    ) -> TrajectoryEstimate:
        stats = self.action_motion.get(action.action_id)
        focus = _focus_component(signature)
        if stats is None or focus is None:
            return TrajectoryEstimate()
        confidence = stats.confidence
        if confidence <= 0.0:
            return TrajectoryEstimate()
        dx = stats.dx
        dy = stats.dy
        boundary_risk = _boundary_risk(signature, focus, dx, dy) * confidence
        reward_bonus = min(0.08, 0.04 * stats.rewarded_count)
        value = max(0.0, 0.12 * confidence + reward_bonus - 0.18 * boundary_risk)
        axis = "horizontal" if abs(dx) >= abs(dy) else "vertical"
        direction = _direction(dx, dy)
        explanation = (
            f"A{action.action_id} learned {axis} motion {direction}; "
            f"boundary_risk={boundary_risk:.2f}"
        )
        return TrajectoryEstimate(value, boundary_risk, confidence, explanation)

    def hypotheses(self, signature: FrameSignature) -> list[str]:
        focus = _focus_component(signature)
        if focus is None:
            return []
        out: list[str] = []
        for action_id, stats in sorted(self.action_motion.items()):
            if stats.confidence <= 0.0:
                continue
            risk = _boundary_risk(signature, focus, stats.dx, stats.dy)
            direction = _direction(stats.dx, stats.dy)
            if risk > 0.45:
                out.append(f"A{action_id} appears constrained at {direction} boundary.")
            else:
                out.append(f"A{action_id} appears to move/cover {direction}.")
        return out[:4]


@dataclass(frozen=True)
class BoundaryEstimate:
    risk: float = 0.0
    confidence: float = 0.0
    explanation: str = ""


@dataclass
class BoundaryStats:
    count: int = 0
    sum_dx: float = 0.0
    sum_dy: float = 0.0
    total_boundary_risk: float = 0.0

    @property
    def dx(self) -> float:
        return self.sum_dx / max(1, self.count)

    @property
    def dy(self) -> float:
        return self.sum_dy / max(1, self.count)

    @property
    def average_boundary_risk(self) -> float:
        return self.total_boundary_risk / max(1, self.count)

    @property
    def confidence(self) -> float:
        motion_signal = min(1.0, (abs(self.dx) + abs(self.dy)) / 4.0)
        boundary_signal = min(1.0, 0.45 + self.average_boundary_risk)
        support = min(1.0, self.count / 2.0)
        return support * motion_signal * boundary_signal


@dataclass
class BoundaryMemory:
    """Recognize screen-edge and rendered-object boundaries for learned motion."""

    action_boundaries: dict[int, BoundaryStats] = field(default_factory=dict)

    def record(
        self,
        previous: FrameSignature,
        action: ActionCandidate,
        current: FrameSignature,
        reward: float,
    ) -> None:
        before = _focus_component(previous)
        after = _matched_component(current, before)
        if before is None or after is None:
            return
        dx = after.centroid[0] - before.centroid[0]
        dy = after.centroid[1] - before.centroid[1]
        if abs(dx) + abs(dy) < 0.5:
            return
        risk = _boundary_risk(current, after, dx, dy)
        stats = self.action_boundaries.setdefault(action.action_id, BoundaryStats())
        stats.count += 1
        stats.sum_dx += dx
        stats.sum_dy += dy
        stats.total_boundary_risk += risk

    def estimate(
        self, signature: FrameSignature, action: ActionCandidate
    ) -> BoundaryEstimate:
        stats = self.action_boundaries.get(action.action_id)
        focus = _focus_component(signature)
        if stats is None or focus is None:
            return BoundaryEstimate()
        confidence = stats.confidence
        if confidence <= 0.0:
            return BoundaryEstimate()
        risk = _boundary_risk(signature, focus, stats.dx, stats.dy)
        if risk <= 0.02:
            return BoundaryEstimate(risk=0.0, confidence=confidence)
        direction = _direction(stats.dx, stats.dy)
        return BoundaryEstimate(
            risk=max(0.0, min(1.0, risk)),
            confidence=confidence,
            explanation=(
                f"A{action.action_id} motion {direction} intersects a visible boundary; "
                f"risk={risk:.2f}"
            ),
        )

    def hypotheses(self, signature: FrameSignature) -> list[str]:
        focus = _focus_component(signature)
        if focus is None:
            return []
        out: list[str] = []
        for action_id, stats in sorted(self.action_boundaries.items()):
            if stats.confidence <= 0.0:
                continue
            risk = _boundary_risk(signature, focus, stats.dx, stats.dy)
            if risk <= 0.20:
                continue
            direction = _direction(stats.dx, stats.dy)
            out.append(
                f"A{action_id} appears blocked by a visible {direction} boundary ({risk:.2f})."
            )
        return out[:3]


@dataclass(frozen=True)
class ObjectRoleEstimate:
    value: float = 0.0
    target_alignment: float = 0.0
    confidence: float = 0.0
    explanation: str = ""


@dataclass
class ObjectRoleStats:
    count: int = 0
    sum_dx: float = 0.0
    sum_dy: float = 0.0
    sum_target_progress: float = 0.0
    toward_count: int = 0
    away_count: int = 0
    rewarded_count: int = 0

    @property
    def dx(self) -> float:
        return self.sum_dx / max(1, self.count)

    @property
    def dy(self) -> float:
        return self.sum_dy / max(1, self.count)

    @property
    def average_progress(self) -> float:
        return self.sum_target_progress / max(1, self.count)

    @property
    def confidence(self) -> float:
        motion_signal = min(1.0, (abs(self.dx) + abs(self.dy)) / 4.0)
        role_signal = min(1.0, abs(self.average_progress) / 3.0)
        return min(1.0, self.count / 2.0) * max(motion_signal, role_signal)


@dataclass
class ObjectRoleMemory:
    """Infer actor/target roles from component motion across state sequences."""

    action_roles: dict[int, ObjectRoleStats] = field(default_factory=dict)

    def record(
        self,
        previous: FrameSignature,
        action: ActionCandidate,
        current: FrameSignature,
        reward: float,
    ) -> None:
        before, after = _moving_component(previous, current)
        if before is None or after is None:
            return
        target = _nearest_target(previous, before)
        if target is None:
            return
        before_distance = _distance(before.centroid, target.centroid)
        after_distance = _distance(after.centroid, target.centroid)
        progress = before_distance - after_distance
        if abs(progress) < 0.25 and reward <= 0.0:
            return

        stats = self.action_roles.setdefault(action.action_id, ObjectRoleStats())
        stats.count += 1
        stats.sum_dx += after.centroid[0] - before.centroid[0]
        stats.sum_dy += after.centroid[1] - before.centroid[1]
        stats.sum_target_progress += progress
        stats.toward_count += int(progress > 0.25)
        stats.away_count += int(progress < -0.25)
        stats.rewarded_count += int(reward > 0.20)

    def estimate(
        self, signature: FrameSignature, action: ActionCandidate
    ) -> ObjectRoleEstimate:
        stats = self.action_roles.get(action.action_id)
        if stats is None:
            return ObjectRoleEstimate()
        confidence = stats.confidence
        if confidence <= 0.0:
            return ObjectRoleEstimate()
        toward_role = stats.toward_count >= stats.away_count
        alignment = _best_role_alignment(signature, stats.dx, stats.dy, toward_role)
        if alignment is None:
            return ObjectRoleEstimate()
        progress, distance = alignment
        signed_progress = progress if toward_role else -progress
        target_alignment = max(-1.0, min(1.0, signed_progress / max(1.0, distance)))
        reward_bonus = min(0.08, 0.04 * stats.rewarded_count)
        value = (
            0.14 * confidence
            + 0.22 * max(0.0, target_alignment) * confidence
            + reward_bonus
            - 0.10 * max(0.0, -target_alignment) * confidence
        )
        role = "toward" if toward_role else "away-from"
        explanation = (
            f"A{action.action_id} appears to move an actor {role} another object; "
            f"alignment={target_alignment:.2f}"
        )
        return ObjectRoleEstimate(
            value=max(-0.12, min(0.32, value)),
            target_alignment=target_alignment,
            confidence=confidence,
            explanation=explanation,
        )

    def hypotheses(self, signature: FrameSignature) -> list[str]:
        out: list[str] = []
        for action_id, stats in sorted(self.action_roles.items()):
            if stats.confidence <= 0.0:
                continue
            toward_role = stats.toward_count >= stats.away_count
            alignment = _best_role_alignment(signature, stats.dx, stats.dy, toward_role)
            if alignment is None:
                continue
            progress, distance = alignment
            signed_progress = progress if toward_role else -progress
            if signed_progress <= 0.1:
                continue
            relation = "toward a target" if toward_role else "away from a hazard"
            strength = signed_progress / max(1.0, distance)
            out.append(f"A{action_id} may move the controlled object {relation} ({strength:.2f}).")
        return out[:3]


def _focus_component(signature: FrameSignature):
    if not signature.components:
        return None
    return max(
        signature.components[:12],
        key=lambda component: (
            component.area,
            -component.edge_touch,
            -abs(component.centroid[0] - signature.width / 2)
            - abs(component.centroid[1] - signature.height / 2),
        ),
    )


def _matched_component(signature: FrameSignature, before):
    if before is None:
        return None
    candidates = [
        component for component in signature.components[:16]
        if component.color == before.color
    ]
    if not candidates:
        candidates = signature.components[:16]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda component: (
            abs(component.area - before.area)
            + abs(component.centroid[0] - before.centroid[0])
            + abs(component.centroid[1] - before.centroid[1])
        ),
    )


def _boundary_risk(signature: FrameSignature, component, dx: float, dy: float) -> float:
    return max(
        _screen_boundary_risk(signature, component, dx, dy),
        _visual_boundary_risk(signature, component, dx, dy),
    )


def _screen_boundary_risk(
    signature: FrameSignature, component, dx: float, dy: float
) -> float:
    width = max(1, signature.width)
    height = max(1, signature.height)
    risk = 0.0
    if dx > 0:
        risk = max(risk, max(0.0, component.bbox[2] - (width - 4)) / 4.0)
    elif dx < 0:
        risk = max(risk, max(0.0, 3 - component.bbox[0]) / 4.0)
    if dy > 0:
        risk = max(risk, max(0.0, component.bbox[3] - (height - 4)) / 4.0)
    elif dy < 0:
        risk = max(risk, max(0.0, 3 - component.bbox[1]) / 4.0)
    return min(1.0, risk)


def _visual_boundary_risk(
    signature: FrameSignature, component, dx: float, dy: float
) -> float:
    if abs(dx) + abs(dy) < 0.5:
        return 0.0
    step = max(1.0, abs(dx) + abs(dy))
    reach = max(3.0, step + 1.5)
    risk = 0.0
    for obstacle in signature.components[:24]:
        if _same_component(obstacle, component):
            continue
        if obstacle.area > component.area * 8 and obstacle.edge_touch:
            continue
        if abs(dx) >= abs(dy):
            overlap = _overlap_1d(
                component.bbox[1],
                component.bbox[3],
                obstacle.bbox[1],
                obstacle.bbox[3],
                margin=1,
            )
            if overlap <= 0:
                continue
            if dx > 0:
                gap = obstacle.bbox[0] - component.bbox[2]
            else:
                gap = component.bbox[0] - obstacle.bbox[2]
        else:
            overlap = _overlap_1d(
                component.bbox[0],
                component.bbox[2],
                obstacle.bbox[0],
                obstacle.bbox[2],
                margin=1,
            )
            if overlap <= 0:
                continue
            if dy > 0:
                gap = obstacle.bbox[1] - component.bbox[3]
            else:
                gap = component.bbox[1] - obstacle.bbox[3]
        if gap < 0 or gap > reach:
            continue
        proximity = 1.0 - gap / max(1.0, reach)
        obstacle_scale = min(1.0, obstacle.area / max(1, component.area))
        risk = max(risk, 0.45 * proximity + 0.35 * obstacle_scale + 0.20 * overlap)
    return min(1.0, risk)


def _overlap_1d(
    left_min: int,
    left_max: int,
    right_min: int,
    right_max: int,
    margin: int = 0,
) -> float:
    low = max(left_min - margin, right_min)
    high = min(left_max + margin, right_max)
    overlap = max(0, high - low + 1)
    span = max(1, min(left_max - left_min + 1, right_max - right_min + 1))
    return min(1.0, overlap / span)


def _moving_component(previous: FrameSignature, current: FrameSignature):
    best = (None, None)
    best_movement = 0.0
    for before in previous.components[:16]:
        after = _matched_component(current, before)
        if after is None:
            continue
        movement = (
            abs(after.centroid[0] - before.centroid[0])
            + abs(after.centroid[1] - before.centroid[1])
        )
        if movement > best_movement:
            best = (before, after)
            best_movement = movement
    if best_movement < 0.5:
        return (None, None)
    return best


def _nearest_target(signature: FrameSignature, actor):
    candidates = [
        component for component in signature.components[:24]
        if not _same_component(component, actor)
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda component: (
            _distance(actor.centroid, component.centroid),
            -component.area,
        ),
    )


def _same_component(left, right) -> bool:
    return left.color == right.color and left.bbox == right.bbox


def _best_role_alignment(
    signature: FrameSignature,
    dx: float,
    dy: float,
    toward_role: bool,
) -> tuple[float, float] | None:
    best: tuple[float, float] | None = None
    best_score = 0.0
    for actor in signature.components[:16]:
        target = _nearest_target(signature, actor)
        if target is None:
            continue
        distance = _distance(actor.centroid, target.centroid)
        projected = _distance(
            (actor.centroid[0] + dx, actor.centroid[1] + dy),
            target.centroid,
        )
        progress = distance - projected
        score = progress if toward_role else -progress
        if score > best_score:
            best = (progress, distance)
            best_score = score
    return best


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _direction(dx: float, dy: float) -> str:
    if abs(dx) >= abs(dy):
        return "east" if dx > 0 else "west"
    return "south" if dy > 0 else "north"


@dataclass
class FrameMemoryRecord:
    frame_hash: str
    vector: list[float]
    score: float
    action_key: str


@dataclass
class SemanticFrameMemory:
    records: list[FrameMemoryRecord] = field(default_factory=list)
    max_records: int = 2048

    def insert(self, signature: FrameSignature, score: float, action_key: str) -> None:
        self.records.append(
            FrameMemoryRecord(
                frame_hash=signature.frame_hash,
                vector=frame_vector(signature),
                score=score,
                action_key=action_key,
            )
        )
        if len(self.records) > self.max_records:
            del self.records[: len(self.records) - self.max_records]

    def retrieve_similar(self, signature: FrameSignature, k: int = 5) -> list[FrameMemoryRecord]:
        vector = frame_vector(signature)
        return sorted(self.records, key=lambda record: cosine(vector, record.vector), reverse=True)[:k]

    def retrieve_unusual_associations(self, signature: FrameSignature, k: int = 5) -> list[FrameMemoryRecord]:
        vector = frame_vector(signature)

        def score(record: FrameMemoryRecord) -> float:
            sim = cosine(vector, record.vector)
            distant_but_related = max(0.0, 0.35 - abs(sim - 0.32))
            return distant_but_related + 0.25 * max(0.0, record.score)

        return sorted(self.records, key=score, reverse=True)[:k]


def frame_vector(signature: FrameSignature) -> list[float]:
    total = max(1, signature.width * signature.height)
    bins = [0.0] * 16
    for color, count in signature.histogram.items():
        bins[color % len(bins)] += count / total
    bins.extend(
        [
            signature.non_background_ratio,
            min(1.0, len(signature.components) / 32.0),
            min(1.0, len(signature.salience_points) / 32.0),
            min(1.0, signature.width / 64.0),
            min(1.0, signature.height / 64.0),
        ]
    )
    norm = math.sqrt(sum(value * value for value in bins)) or 1.0
    return [value / norm for value in bins]


def cosine(left: list[float], right: list[float]) -> float:
    limit = min(len(left), len(right))
    if limit == 0:
        return 0.0
    return sum(left[index] * right[index] for index in range(limit))
