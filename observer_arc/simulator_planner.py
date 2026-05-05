"""Local simulator planning for ARC-AGI-3 environments.

This module is deliberately separated from the API-safe observer policy. When a
local ARC game object is available, the planner can clone it and compare action
rollouts before committing a real action. When only the remote/API wrapper is
available, callers should skip this planner and use the frame-only policy.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any

from arcengine import ActionInput, GameAction

from .state import ActionCandidate
from .vision import frame_distance, summarize_frame


@dataclass
class PlannerConfig:
    max_depth: int = 48
    beam_width: int = 8
    branch_limit: int = 18
    max_nodes: int = 2200
    max_seconds: float = 0.30
    return_best_nonterminal: bool = True


@dataclass
class PlanNode:
    game: Any
    frame: Any
    path: list[ActionCandidate]
    signatures: list[Any]
    score: float
    depth: int


@dataclass(frozen=True)
class ObjectiveProbe:
    score: float = 0.0
    reason: str = ""
    level_gain: int = 0
    win: bool = False
    action: ActionCandidate | None = None
    signature: Any | None = None


@dataclass(frozen=True)
class RoleConsequence:
    score: float = 0.0
    reason: str = "role_neutral"


@dataclass
class SubgoalActionCredit:
    attempts: int = 0
    useful: int = 0
    total_score: float = 0.0
    total_change: float = 0.0
    total_role: float = 0.0
    total_level_gain: int = 0

    @property
    def average(self) -> float:
        if self.attempts <= 0:
            return 0.0
        success_rate = self.useful / max(1, self.attempts)
        level_bonus = min(1.0, self.total_level_gain / max(1, self.attempts))
        return (
            self.total_score / self.attempts
            + 0.08 * success_rate
            + 0.20 * level_bonus
        )


class LocalSimulatorPlanner:
    """Beam-search planner over cloned local ARC game states."""

    def __init__(self, config: PlannerConfig | None = None) -> None:
        self.config = config or PlannerConfig()
        self.last_plan: list[ActionCandidate] = []
        self.queued_plan: list[ActionCandidate] = []
        self.queued_preconditions: list[Any] = []
        self.observed_frame_counts: dict[str, int] = {}
        self.frontier_frame_counts: dict[str, int] = {}
        self.subgoal_action_credit: dict[tuple[str, str], SubgoalActionCredit] = {}
        self.subgoal_general_action_credit: dict[str, SubgoalActionCredit] = {}
        self.subgoal_sequence_attempts: dict[tuple[str, tuple[str, ...]], int] = {}
        self.information_probe_streak = 0

    def plan(self, game: Any, latest_frame: Any) -> ActionCandidate | None:
        if game is None or not hasattr(game, "perform_action"):
            return None
        queued = self._queued_action(latest_frame)
        if queued is not None:
            return queued
        root_actions = self._valid_actions(game, latest_frame)
        if not root_actions:
            return None

        root_signature = summarize_frame(getattr(latest_frame, "frame", []))
        root_levels = int(getattr(latest_frame, "levels_completed", 0) or 0)
        root_state = _state_name(getattr(latest_frame, "state", "NOT_FINISHED"))
        if root_state == "WIN":
            return None
        self._record_observed(root_signature)
        frontier_pressure = self._frontier_pressure(root_signature, root_actions)

        can_return_partial = self.config.return_best_nonterminal and len(root_actions) <= 2
        depth_limit, node_limit, seconds_limit = self._limits(len(root_actions))
        started = time.perf_counter()
        if any(action.action_id == 6 for action in root_actions):
            scanned = self._single_click_probe(
                game,
                latest_frame,
                root_levels,
                root_signature,
                started,
                min(
                    self.config.max_seconds,
                    max(0.012, self.config.max_seconds * 0.10),
                ),
            )
            if scanned is not None:
                return scanned
        repeat_seconds_limit = min(seconds_limit, max(0.020, seconds_limit * 0.45))
        repeated = self._repeat_probe(
            game,
            root_actions,
            root_levels,
            root_signature,
            started,
            repeat_seconds_limit,
        )
        if repeated is not None:
            return repeated
        if self._should_probe_hidden_sequences(game, root_actions, root_levels, root_signature):
            hidden_sequence = self._hidden_sequence_probe(
                game,
                root_actions,
                root_levels,
                root_signature,
                time.perf_counter(),
                min(self.config.max_seconds, max(0.050, self.config.max_seconds * 0.35)),
            )
            if hidden_sequence is not None:
                return hidden_sequence
        sequenced = None
        sequenced = self._sequence_probe(
            game,
            root_actions,
            root_levels,
            root_signature,
            time.perf_counter(),
            min(self.config.max_seconds, 0.035),
        )
        if sequenced is not None:
            return sequenced
        information_sequence = self._information_sequence_probe(
            game,
            root_actions,
            root_signature,
            root_levels,
            time.perf_counter(),
            min(self.config.max_seconds, 0.055),
        )
        if information_sequence is not None:
            return information_sequence
        tried_frontier = False
        tried_commitment = False
        if frontier_pressure > 0.0:
            frontier_action = self._frontier_probe(
                game,
                latest_frame,
                root_actions,
                root_signature,
                root_levels,
                time.perf_counter(),
                min(
                    self.config.max_seconds,
                    max(0.090, self.config.max_seconds * 0.65),
                ),
                pressure=frontier_pressure,
            )
            tried_frontier = True
            if frontier_action is not None:
                frontier_action.explanation = (
                    f"{frontier_action.explanation}; "
                    f"frontier_pressure={frontier_pressure:.2f}"
                )
                return frontier_action
            commitment_action = self._subgoal_commitment_probe(
                game,
                latest_frame,
                root_actions,
                root_signature,
                root_levels,
                time.perf_counter(),
                min(
                    self.config.max_seconds,
                    max(0.070, self.config.max_seconds * 0.50),
                ),
                pressure=frontier_pressure,
            )
            tried_commitment = True
            if commitment_action is not None:
                commitment_action.explanation = (
                    f"{commitment_action.explanation}; "
                    f"frontier_pressure={frontier_pressure:.2f}"
                )
                return commitment_action
        informative = None
        informative = self._information_probe(
            game,
            root_actions,
            root_signature,
            time.perf_counter(),
            min(self.config.max_seconds, 0.020),
            pressure=frontier_pressure,
        )
        if informative is not None:
            return informative
        if not tried_frontier:
            frontier_action = self._frontier_probe(
                game,
                latest_frame,
                root_actions,
                root_signature,
                root_levels,
                time.perf_counter(),
                min(
                    self.config.max_seconds,
                    max(0.070, self.config.max_seconds * 0.55),
                ),
            )
            if frontier_action is not None:
                return frontier_action
        if not tried_commitment:
            commitment_action = self._subgoal_commitment_probe(
                game,
                latest_frame,
                root_actions,
                root_signature,
                root_levels,
                time.perf_counter(),
                min(
                    self.config.max_seconds,
                    max(0.055, self.config.max_seconds * 0.40),
                ),
                pressure=frontier_pressure,
            )
            if commitment_action is not None:
                return commitment_action
        if self._expired(started, seconds_limit):
            return None

        frontier = [PlanNode(copy.deepcopy(game), latest_frame, [], [], 0.0, 0)]
        seen: set[tuple[str, int, tuple[str, ...]]] = {
            (root_signature.frame_hash, 0, ())
        }
        best: PlanNode | None = None
        nodes = 0

        for depth in range(depth_limit):
            next_frontier: list[PlanNode] = []
            for node in frontier:
                if self._expired(started, seconds_limit):
                    return self._best_action(best) if can_return_partial else None
                actions = self._valid_actions(node.game, node.frame)
                for action in actions[: self.config.branch_limit]:
                    if self._expired(started, seconds_limit):
                        return self._best_action(best) if can_return_partial else None
                    nodes += 1
                    if nodes > node_limit:
                        return self._best_action(best) if can_return_partial else None
                    child_game = copy.deepcopy(node.game)
                    raw = self._perform(child_game, action)
                    if raw is None:
                        continue
                    child_sig = summarize_frame(getattr(raw, "frame", []))
                    state = _state_name(getattr(raw, "state", "NOT_FINISHED"))
                    if state == "GAME_OVER":
                        continue
                    levels = int(getattr(raw, "levels_completed", 0) or 0)
                    level_gain = levels - root_levels
                    changed = frame_distance(root_signature, child_sig)
                    child_key = _game_key(child_game, child_sig.frame_hash)
                    if child_key in seen and level_gain <= 0:
                        continue
                    seen.add(child_key)

                    candidate = _copy_candidate(action, source="local-simulator")
                    candidate.value_score = float(level_gain)
                    candidate.novelty_score = min(1.0, changed)
                    candidate.coherence_score = 1.0 / (1.0 + depth)
                    candidate.repetition_penalty = 0.0
                    candidate.cliche_penalty = 0.0
                    candidate.contradiction_penalty = 0.0

                    transition_score = (
                        500.0 * level_gain
                        + 1000.0 * (state == "WIN")
                        + 0.8 * changed
                        - 8.0 * (state == "GAME_OVER")
                        - 0.012 * len(node.path)
                    )
                    path = node.path + [candidate]
                    signatures = node.signatures + [child_sig]
                    child_score = node.score + transition_score
                    child = PlanNode(
                        child_game,
                        raw,
                        path,
                        signatures,
                        child_score,
                        depth + 1,
                    )

                    if best is None or child.score > best.score:
                        best = child
                    if level_gain > 0 or state == "WIN":
                        first = path[0]
                        first.final_score = child.score
                        first.explanation = (
                            f"local simulator found {'win' if state == 'WIN' else 'level gain'} "
                            f"at depth {depth + 1}; projected_gain={level_gain}"
                        )
                        self._activate_plan(path, signatures)
                        return first
                    next_frontier.append(child)
                    if len(next_frontier) > self.config.beam_width * 4:
                        next_frontier.sort(key=lambda item: item.score, reverse=True)
                        del next_frontier[self.config.beam_width :]
            next_frontier.sort(key=lambda item: item.score, reverse=True)
            frontier = next_frontier[: self.config.beam_width]
            if not frontier:
                break
        if can_return_partial:
            return self._best_action(best)
        return None

    def _best_action(self, best: PlanNode | None) -> ActionCandidate | None:
        if best is None or not best.path:
            return None
        if best.score <= 0.0:
            return None
        first = best.path[0]
        first.final_score = best.score
        first.explanation = (
            f"local simulator chose best nonterminal rollout; depth={best.depth}, "
            f"projected_score={best.score:.3f}"
        )
        self.last_plan = best.path
        self.queued_plan = []
        self.queued_preconditions = []
        return first

    def _repeat_probe(
        self,
        game: Any,
        actions: list[ActionCandidate],
        root_levels: int,
        root_signature: Any,
        started: float,
        seconds_limit: float,
    ) -> ActionCandidate | None:
        repeat_depth = min(self.config.max_depth, 12 if len(actions) <= 2 else 4)
        active: list[tuple[ActionCandidate, Any, list[ActionCandidate], list[Any]]] = [
            (action, copy.deepcopy(game), [], [])
            for action in actions[: self.config.branch_limit]
        ]
        for depth in range(repeat_depth):
            next_active: list[tuple[ActionCandidate, Any, list[ActionCandidate], list[Any]]] = []
            for action, child_game, path, signatures in active:
                if self._expired(started, seconds_limit):
                    return None
                candidate = _copy_candidate(action, source="local-simulator-repeat")
                raw = self._perform(child_game, candidate)
                if raw is None:
                    continue
                new_path = path + [candidate]
                child_sig = summarize_frame(getattr(raw, "frame", []))
                new_signatures = signatures + [child_sig]
                state = _state_name(getattr(raw, "state", "NOT_FINISHED"))
                levels = int(getattr(raw, "levels_completed", 0) or 0)
                level_gain = levels - root_levels
                changed = frame_distance(root_signature, child_sig)
                if level_gain > 0 or state == "WIN":
                    first = new_path[0]
                    first.value_score = float(level_gain)
                    first.novelty_score = min(1.0, changed)
                    first.coherence_score = 1.0 / (1.0 + depth)
                    first.final_score = (
                        500.0 * level_gain
                        + 1000.0 * (state == "WIN")
                        + 0.8 * changed
                    )
                    first.explanation = (
                        f"local simulator repeat probe found "
                        f"{'win' if state == 'WIN' else 'level gain'} at depth {depth + 1}; "
                        f"projected_gain={level_gain}"
                    )
                    self._activate_plan(new_path, new_signatures)
                    return first
                if state != "GAME_OVER":
                    next_active.append((action, child_game, new_path, new_signatures))
            active = next_active
            if not active:
                break
        return None

    def _should_probe_hidden_sequences(
        self,
        game: Any,
        actions: list[ActionCandidate],
        root_levels: int,
        root_signature: Any,
    ) -> bool:
        simple_actions = [
            action for action in actions
            if action.action_id != 6 and action.x is None and action.y is None
        ]
        if len(simple_actions) < 3 or len(simple_actions) != len(actions):
            return False
        observed = False
        for action in simple_actions[: self.config.branch_limit]:
            child_game = copy.deepcopy(game)
            raw = self._perform(child_game, action)
            if raw is None:
                continue
            state = _state_name(getattr(raw, "state", "NOT_FINISHED"))
            levels = int(getattr(raw, "levels_completed", 0) or 0)
            if levels > root_levels or state == "WIN":
                return False
            if state == "GAME_OVER":
                continue
            child_sig = summarize_frame(getattr(raw, "frame", []))
            if (
                child_sig.frame_hash != root_signature.frame_hash
                or frame_distance(root_signature, child_sig) > 0.001
            ):
                observed = True
                break
        return not observed

    def _hidden_sequence_probe(
        self,
        game: Any,
        actions: list[ActionCandidate],
        root_levels: int,
        root_signature: Any,
        started: float,
        seconds_limit: float,
    ) -> ActionCandidate | None:
        simple_actions = [
            action for action in actions
            if action.action_id != 6 and action.x is None and action.y is None
        ]
        if len(simple_actions) < 3 or len(simple_actions) > 5:
            return None
        if len(simple_actions) != len(actions):
            return None

        depth_limit = min(self.config.max_depth, 7 if len(simple_actions) <= 4 else 5)
        node_limit = min(self.config.max_nodes, 3200)
        frontier: list[tuple[Any, list[ActionCandidate], list[Any]]] = [
            (copy.deepcopy(game), [], [])
        ]
        nodes = 0

        for depth in range(depth_limit):
            next_frontier: list[tuple[Any, list[ActionCandidate], list[Any]]] = []
            for node_game, path, signatures in frontier:
                if self._expired(started, seconds_limit):
                    return None
                node_actions = [
                    action for action in self._valid_actions(node_game, None)
                    if action.action_id != 6 and action.x is None and action.y is None
                ][: self.config.branch_limit]
                for action in node_actions:
                    if self._expired(started, seconds_limit):
                        return None
                    nodes += 1
                    if nodes > node_limit:
                        return None
                    child_game = copy.deepcopy(node_game)
                    raw = self._perform(child_game, action)
                    if raw is None:
                        continue
                    state = _state_name(getattr(raw, "state", "NOT_FINISHED"))
                    if state == "GAME_OVER":
                        continue
                    child_sig = summarize_frame(getattr(raw, "frame", []))
                    levels = int(getattr(raw, "levels_completed", 0) or 0)
                    level_gain = levels - root_levels
                    candidate = _copy_candidate(action, source="local-simulator-hidden-sequence")
                    new_path = path + [candidate]
                    new_signatures = signatures + [child_sig]
                    if level_gain > 0 or state == "WIN":
                        changed = frame_distance(root_signature, child_sig)
                        first = new_path[0]
                        first.value_score = float(level_gain)
                        first.novelty_score = min(1.0, changed)
                        first.coherence_score = 1.0 / (1.0 + depth)
                        first.final_score = (
                            500.0 * level_gain
                            + 1000.0 * (state == "WIN")
                            + 0.8 * changed
                        )
                        first.explanation = (
                            f"local simulator hidden sequence found "
                            f"{'win' if state == 'WIN' else 'level gain'} at depth {depth + 1}; "
                            f"projected_gain={level_gain}"
                        )
                        self._activate_plan(new_path, new_signatures)
                        return first
                    next_frontier.append((child_game, new_path, new_signatures))
            if not next_frontier:
                break
            frontier = self._diverse_prefixes(
                next_frontier,
                max(1, min(node_limit, self.config.beam_width * 16)),
            )
        return None

    def _sequence_probe(
        self,
        game: Any,
        actions: list[ActionCandidate],
        root_levels: int,
        root_signature: Any,
        started: float,
        seconds_limit: float,
    ) -> ActionCandidate | None:
        simple_actions = [
            action for action in actions
            if action.action_id != 6 and action.x is None and action.y is None
        ]
        if len(simple_actions) < 3 or len(simple_actions) > 5:
            return None
        if len(simple_actions) != len(actions):
            return None

        depth_limit = min(self.config.max_depth, 8)
        node_limit = min(self.config.max_nodes, 768)
        frontier: list[tuple[Any, list[ActionCandidate], list[Any]]] = [
            (copy.deepcopy(game), [], [])
        ]
        seen: set[str] = {_game_key(game, root_signature.frame_hash)}
        nodes = 0

        for depth in range(depth_limit):
            next_frontier: list[tuple[Any, list[ActionCandidate], list[Any]]] = []
            for node_game, path, signatures in frontier:
                if self._expired(started, seconds_limit):
                    return None
                node_actions = self._valid_actions(node_game, None)
                node_simple = [
                    action for action in node_actions
                    if action.action_id != 6 and action.x is None and action.y is None
                ][: self.config.branch_limit]
                for action in node_simple:
                    if self._expired(started, seconds_limit):
                        return None
                    nodes += 1
                    if nodes > node_limit:
                        return None
                    child_game = copy.deepcopy(node_game)
                    raw = self._perform(child_game, action)
                    if raw is None:
                        continue
                    child_sig = summarize_frame(getattr(raw, "frame", []))
                    state = _state_name(getattr(raw, "state", "NOT_FINISHED"))
                    levels = int(getattr(raw, "levels_completed", 0) or 0)
                    level_gain = levels - root_levels
                    candidate = _copy_candidate(action, source="local-simulator-sequence")
                    new_path = path + [candidate]
                    new_signatures = signatures + [child_sig]
                    if level_gain > 0 or state == "WIN":
                        changed = frame_distance(root_signature, child_sig)
                        first = new_path[0]
                        first.value_score = float(level_gain)
                        first.novelty_score = min(1.0, changed)
                        first.coherence_score = 1.0 / (1.0 + depth)
                        first.final_score = (
                            500.0 * level_gain
                            + 1000.0 * (state == "WIN")
                            + 0.8 * changed
                        )
                        first.explanation = (
                            f"local simulator sequence probe found "
                            f"{'win' if state == 'WIN' else 'level gain'} at depth {depth + 1}; "
                            f"projected_gain={level_gain}"
                        )
                        self._activate_plan(new_path, new_signatures)
                        return first
                    if state == "GAME_OVER":
                        continue
                    child_key = _sequence_key(child_sig.frame_hash, new_path)
                    if child_key in seen:
                        continue
                    seen.add(child_key)
                    next_frontier.append((child_game, new_path, new_signatures))
            frontier = next_frontier[: max(1, self.config.beam_width * 8)]
            if not frontier:
                break
        return None

    def _information_sequence_probe(
        self,
        game: Any,
        actions: list[ActionCandidate],
        root_signature: Any,
        root_levels: int,
        started: float,
        seconds_limit: float,
    ) -> ActionCandidate | None:
        if len(actions) < 3:
            return None
        depth_limit = min(self.config.max_depth, 3)
        node_limit = min(self.config.max_nodes, max(96, self.config.branch_limit * 24))
        frontier: list[tuple[Any, list[ActionCandidate], list[Any], Any, float, float]] = [
            (copy.deepcopy(game), [], [], root_signature, 0.0, 0.0)
        ]
        seen: set[tuple[str, int]] = {(root_signature.frame_hash, 0)}
        best_path: list[ActionCandidate] = []
        best_signatures: list[Any] = []
        best_score = 0.0
        best_root_change = 0.0
        best_step_change = 0.0
        best_single_step_score = 0.0
        nodes = 0

        for depth in range(depth_limit):
            next_frontier: list[
                tuple[Any, list[ActionCandidate], list[Any], Any, float, float]
            ] = []
            for (
                node_game,
                path,
                signatures,
                previous_sig,
                cumulative_score,
                first_root_change,
            ) in frontier:
                if self._expired(started, seconds_limit):
                    break
                node_actions = self._valid_actions(node_game, None)
                for action in node_actions[: self.config.branch_limit]:
                    if self._expired(started, seconds_limit):
                        break
                    nodes += 1
                    if nodes > node_limit:
                        break
                    child_game = copy.deepcopy(node_game)
                    raw = self._perform(child_game, action)
                    if raw is None:
                        continue
                    state = _state_name(getattr(raw, "state", "NOT_FINISHED"))
                    if state == "GAME_OVER":
                        continue
                    child_sig = summarize_frame(getattr(raw, "frame", []))
                    levels = int(getattr(raw, "levels_completed", 0) or 0)
                    level_gain = levels - root_levels
                    candidate = _copy_candidate(action, source="local-simulator-info-sequence")
                    new_path = path + [candidate]
                    new_signatures = signatures + [child_sig]
                    if level_gain > 0 or state == "WIN":
                        first = new_path[0]
                        first.value_score = float(level_gain)
                        first.novelty_score = min(1.0, frame_distance(root_signature, child_sig))
                        first.coherence_score = 1.0 / (1.0 + depth)
                        first.final_score = 500.0 * level_gain + 1000.0 * (state == "WIN")
                        first.explanation = (
                            f"local simulator information sequence found "
                            f"{'win' if state == 'WIN' else 'level gain'} at depth {depth + 1}; "
                            f"projected_gain={level_gain}"
                        )
                        self._activate_plan(new_path, new_signatures)
                        return first

                    step_change = frame_distance(previous_sig, child_sig)
                    root_change = frame_distance(root_signature, child_sig)
                    new_first_root_change = (
                        root_change if not path else first_root_change
                    )
                    component_delta = abs(
                        len(getattr(child_sig, "components", []))
                        - len(getattr(previous_sig, "components", []))
                    ) / max(1, len(getattr(previous_sig, "components", [])))
                    step_score = step_change + 0.30 * min(1.0, component_delta)
                    new_cumulative_score = (
                        cumulative_score + step_score * (0.82 ** depth)
                    )
                    path_score = (
                        new_cumulative_score
                        + 0.45 * root_change
                        - 0.020 * len(new_path)
                    )
                    if len(new_path) == 1:
                        best_single_step_score = max(best_single_step_score, path_score)
                    if (
                        len(new_path) >= 2
                        and root_change >= 0.035
                        and new_first_root_change < 0.75 * root_change
                        and path_score > best_score
                    ):
                        best_path = new_path
                        best_signatures = new_signatures
                        best_score = path_score
                        best_root_change = root_change
                        best_step_change = step_change

                    child_key = (child_sig.frame_hash, len(new_path))
                    if child_key in seen:
                        continue
                    seen.add(child_key)
                    if step_change > 0.004 or root_change > 0.010:
                        next_frontier.append(
                            (
                                child_game,
                                new_path,
                                new_signatures,
                                child_sig,
                                new_cumulative_score,
                                new_first_root_change,
                            )
                        )
            if nodes > node_limit or self._expired(started, seconds_limit):
                break
            next_frontier.sort(key=lambda item: item[4], reverse=True)
            frontier = next_frontier[: max(1, self.config.beam_width * 4)]
            if not frontier:
                break

        if not best_path or best_score < max(0.090, best_single_step_score * 1.25):
            return None
        first = best_path[0]
        first.novelty_score = min(1.0, best_root_change)
        first.value_score = min(0.45, best_score)
        first.coherence_score = 0.62
        first.final_score = best_score
        first.explanation = (
            f"local simulator information sequence selected action; "
            f"path_len={len(best_path)}, root_change={best_root_change:.3f}, "
            f"last_step_change={best_step_change:.3f}"
        )
        self._activate_plan(best_path, best_signatures)
        return first

    def _subgoal_commitment_probe(
        self,
        game: Any,
        latest_frame: Any,
        actions: list[ActionCandidate],
        root_signature: Any,
        root_levels: int,
        started: float,
        seconds_limit: float,
        pressure: float = 0.0,
    ) -> ActionCandidate | None:
        if len(actions) < 3:
            return None

        depth_limit = min(self.config.max_depth, 5)
        node_limit = min(self.config.max_nodes, max(160, self.config.branch_limit * 48))
        root_action_ids = _available_ids(getattr(latest_frame, "available_actions", []) or [])
        if not root_action_ids:
            root_action_ids = sorted({action.action_id for action in actions})
        frontier: list[PlanNode] = [
            PlanNode(copy.deepcopy(game), latest_frame, [], [], 0.0, 0)
        ]
        seen: set[tuple[str, int, tuple[str, ...]]] = {
            (root_signature.frame_hash, 0, ())
        }
        best_path: list[ActionCandidate] = []
        best_signatures: list[Any] = []
        best_score = 0.0
        best_single_step_score = 0.0
        best_depth = 0
        best_reason = ""
        best_root_change = 0.0
        best_role_score = 0.0
        nodes = 0

        for depth in range(depth_limit):
            next_frontier: list[PlanNode] = []
            for node in frontier:
                if self._expired(started, seconds_limit):
                    break
                previous_sig = node.signatures[-1] if node.signatures else root_signature
                node_actions = self._valid_actions(
                    node.game,
                    node.frame,
                    limit=max(self.config.branch_limit, self.config.beam_width * 2),
                )
                for action in node_actions[: self.config.branch_limit]:
                    if self._expired(started, seconds_limit):
                        break
                    nodes += 1
                    if nodes > node_limit:
                        break
                    child_game = copy.deepcopy(node.game)
                    raw = self._perform(child_game, action)
                    if raw is None:
                        continue
                    state = _state_name(getattr(raw, "state", "NOT_FINISHED"))
                    if state == "GAME_OVER":
                        continue
                    child_sig = summarize_frame(getattr(raw, "frame", []))
                    levels = int(getattr(raw, "levels_completed", 0) or 0)
                    level_gain = levels - root_levels
                    candidate = _copy_candidate(
                        action,
                        source="local-simulator-subgoal-commitment",
                    )
                    path = node.path + [candidate]
                    signatures = node.signatures + [child_sig]
                    if level_gain > 0 or state == "WIN":
                        first = path[0]
                        first.value_score = float(level_gain)
                        first.novelty_score = min(
                            1.0,
                            _signature_change(root_signature, child_sig),
                        )
                        first.coherence_score = 1.0 / (1.0 + depth)
                        first.final_score = 500.0 * level_gain + 1000.0 * (state == "WIN")
                        first.explanation = (
                            f"local simulator subgoal commitment found "
                            f"{'win' if state == 'WIN' else 'level gain'}; "
                            f"path_len={len(path)}, projected_gain={level_gain}"
                        )
                        self._activate_plan(path, signatures)
                        self._record_frontier_path(signatures)
                        self._note_selection(first)
                        return first

                    child_actions = _available_ids(getattr(raw, "available_actions", []) or [])
                    objective_probe = self._objective_probe(
                        child_game,
                        raw,
                        child_sig,
                        root_levels,
                        root_action_ids,
                        limit=max(3, min(6, self.config.beam_width)),
                    )
                    score, reason = self._frontier_score(
                        root_signature,
                        previous_sig,
                        child_sig,
                        root_action_ids,
                        child_actions,
                        path,
                        objective_probe,
                    )
                    role = _role_consequence_score(
                        root_signature,
                        previous_sig,
                        child_sig,
                        objective_probe,
                    )
                    root_change = _signature_change(root_signature, child_sig)
                    step_change = _signature_change(previous_sig, child_sig)
                    cumulative = node.score + max(0.0, score) * (0.84 ** depth)
                    commitment_score = (
                        cumulative
                        + 0.22 * root_change
                        + 0.20 * role.score
                        + 0.12 * max(0.0, objective_probe.score)
                        + 0.04 * min(1.0, step_change)
                        - 0.030 * len(path)
                    )
                    if len(path) == 1:
                        best_single_step_score = max(
                            best_single_step_score,
                            commitment_score,
                        )
                    has_subgoal_evidence = (
                        root_change >= 0.012
                        or role.score >= 0.070
                        or objective_probe.score >= 0.220
                        or set(child_actions) != set(root_action_ids)
                    )
                    if (
                        len(path) >= 2
                        and has_subgoal_evidence
                        and (
                            not best_path
                            or commitment_score
                            > best_score + 0.035 * max(0, len(path) - len(best_path))
                        )
                    ):
                        best_path = path
                        best_signatures = signatures
                        best_score = commitment_score
                        best_depth = depth + 1
                        best_reason = reason
                        best_root_change = root_change
                        best_role_score = role.score

                    child_key = _sequence_key(child_sig.frame_hash, path)
                    if child_key in seen:
                        continue
                    seen.add(child_key)
                    if len(path) >= 2 and (
                        role.score >= 0.18 or objective_probe.score >= 0.25
                    ):
                        continue
                    if cumulative > -0.04 or has_subgoal_evidence:
                        next_frontier.append(
                            PlanNode(
                                child_game,
                                raw,
                                path,
                                signatures,
                                cumulative,
                                depth + 1,
                            )
                        )
            if nodes > node_limit or self._expired(started, seconds_limit):
                break
            next_frontier.sort(key=lambda item: item.score, reverse=True)
            frontier = self._diverse_plan_nodes(
                next_frontier,
                max(1, self.config.beam_width * 4),
            )
            if not frontier:
                break

        required_score = max(0.090, 0.150 - 0.050 * min(1.0, pressure))
        coverage_seconds = min(
            self.config.max_seconds,
            max(0.018, min(0.035, seconds_limit * 0.45)),
        )
        if not best_path or best_score < required_score:
            memory_action = self._subgoal_memory_probe(
                game,
                actions,
                root_signature,
                root_levels,
                time.perf_counter(),
                coverage_seconds,
            )
            if memory_action is not None:
                return memory_action
            return self._coverage_commitment_sequence(
                game,
                actions,
                root_signature,
                root_levels,
                time.perf_counter(),
                coverage_seconds,
                pressure,
            )
        if pressure < 0.50 and best_score < best_single_step_score * 1.03:
            memory_action = self._subgoal_memory_probe(
                game,
                actions,
                root_signature,
                root_levels,
                time.perf_counter(),
                coverage_seconds,
            )
            if memory_action is not None:
                return memory_action
            return self._coverage_commitment_sequence(
                game,
                actions,
                root_signature,
                root_levels,
                time.perf_counter(),
                coverage_seconds,
                pressure,
            )
        first = best_path[0]
        first.novelty_score = min(1.0, best_root_change)
        first.value_score = min(0.45, best_score)
        first.coherence_score = 0.56
        first.final_score = best_score
        first.explanation = (
            f"local simulator subgoal commitment selected sequence; "
            f"path_len={len(best_path)}, depth={best_depth}, "
            f"commitment_score={best_score:.3f}, root_change={best_root_change:.3f}, "
            f"role_score={best_role_score:.3f}, reason={best_reason}"
        )
        self._activate_plan(best_path, best_signatures)
        self._record_committed_trajectory(
            root_signature,
            best_path,
            best_signatures,
            best_score,
            best_reason,
        )
        self._record_frontier_path(best_signatures)
        self._note_selection(first)
        return first

    def _subgoal_memory_probe(
        self,
        game: Any,
        actions: list[ActionCandidate],
        root_signature: Any,
        root_levels: int,
        started: float,
        seconds_limit: float,
    ) -> ActionCandidate | None:
        if not self.subgoal_action_credit:
            return None
        best: ActionCandidate | None = None
        best_score = 0.0
        best_signature = None
        best_reason = ""
        for action in actions[: self.config.branch_limit]:
            if self._expired(started, seconds_limit):
                break
            memory_score = self._subgoal_credit_score(root_signature, action)
            if memory_score <= 0.030:
                continue
            child_game = copy.deepcopy(game)
            raw = self._perform(child_game, action)
            if raw is None:
                continue
            state = _state_name(getattr(raw, "state", "NOT_FINISHED"))
            if state == "GAME_OVER":
                continue
            child_sig = summarize_frame(getattr(raw, "frame", []))
            levels = int(getattr(raw, "levels_completed", 0) or 0)
            level_gain = levels - root_levels
            changed = frame_distance(root_signature, child_sig)
            role = _role_consequence_score(root_signature, root_signature, child_sig)
            child_actions = set(
                _available_ids(getattr(raw, "available_actions", []) or [])
            )
            root_actions = {candidate.action_id for candidate in actions}
            affordance_shift = 0.0 if child_actions == root_actions else 0.05
            score = (
                memory_score
                + 0.25 * min(1.0, changed)
                + 0.20 * role.score
                + affordance_shift
                + 500.0 * max(0, level_gain)
                + 1000.0 * (state == "WIN")
            )
            if score <= best_score:
                continue
            candidate = _copy_candidate(
                action,
                source="local-simulator-subgoal-memory",
            )
            candidate.novelty_score = min(1.0, changed)
            candidate.value_score = min(0.45, score)
            candidate.coherence_score = 0.58
            candidate.final_score = score
            candidate.explanation = (
                f"local simulator subgoal memory selected action; "
                f"memory_score={memory_score:.3f}, frame_change={changed:.3f}, "
                f"role_score={role.score:.3f}, projected_gain={level_gain}"
            )
            best = candidate
            best_score = score
            best_signature = child_sig
            best_reason = role.reason
        if best is None or best_score < 0.095:
            return None
        self._activate_plan([best], [best_signature] if best_signature is not None else [])
        if best_signature is not None:
            self._record_committed_trajectory(
                root_signature,
                [best],
                [best_signature],
                best_score,
                best_reason,
            )
            self._record_frontier_path([best_signature])
        self._note_selection(best)
        return best

    def _coverage_commitment_sequence(
        self,
        game: Any,
        actions: list[ActionCandidate],
        root_signature: Any,
        root_levels: int,
        started: float,
        seconds_limit: float,
        pressure: float,
    ) -> ActionCandidate | None:
        frame_hash = getattr(root_signature, "frame_hash", "")
        visits = self.observed_frame_counts.get(frame_hash, 0)
        if visits <= 0 and pressure <= 0.0 and self.information_probe_streak < 2:
            return None
        simple_actions = [
            action
            for action in actions
            if action.action_id != 6 and action.x is None and action.y is None
        ]
        if len(simple_actions) < 3:
            return None
        unique: dict[int, ActionCandidate] = {}
        for action in simple_actions:
            unique.setdefault(action.action_id, action)
        ordered = [unique[action_id] for action_id in sorted(unique)]
        if len(ordered) < 3:
            return None
        offset = max(0, visits - 1) % len(ordered)
        ordered = ordered[offset:] + ordered[:offset]

        width = min(3, len(ordered))
        sequences: list[list[ActionCandidate]] = []
        seen_sequences: set[tuple[str, ...]] = set()
        for base in (ordered, list(reversed(ordered))):
            for index in range(len(base)):
                sequence = [base[(index + step) % len(base)] for step in range(width)]
                key = tuple(action.key() for action in sequence)
                if key in seen_sequences:
                    continue
                seen_sequences.add(key)
                sequences.append(sequence)
                if len(sequences) >= min(12, max(4, len(ordered) * 2)):
                    break
            if len(sequences) >= min(12, max(4, len(ordered) * 2)):
                break

        best_path: list[ActionCandidate] = []
        best_signatures: list[Any] = []
        best_score = float("-inf")
        best_root_change = 0.0
        best_level_gain = 0
        best_state = "NOT_FINISHED"
        for sequence in sequences:
            if self._expired(started, seconds_limit):
                break
            child_game = copy.deepcopy(game)
            path: list[ActionCandidate] = []
            signatures: list[Any] = []
            previous_sig = root_signature
            cumulative_change = 0.0
            memory_bonus = 0.0
            best_role = 0.0
            rejected = False
            final_state = "NOT_FINISHED"
            level_gain = 0
            for action in sequence:
                if self._expired(started, seconds_limit):
                    rejected = True
                    break
                candidate = _copy_candidate(
                    action,
                    source="local-simulator-subgoal-commitment",
                )
                raw = self._perform(child_game, candidate)
                if raw is None:
                    rejected = True
                    break
                final_state = _state_name(getattr(raw, "state", "NOT_FINISHED"))
                if final_state == "GAME_OVER":
                    rejected = True
                    break
                child_sig = summarize_frame(getattr(raw, "frame", []))
                path.append(candidate)
                signatures.append(child_sig)
                cumulative_change += frame_distance(previous_sig, child_sig)
                memory_bonus += self._subgoal_credit_score(previous_sig, action)
                best_role = max(
                    best_role,
                    _role_consequence_score(root_signature, previous_sig, child_sig).score,
                )
                previous_sig = child_sig
                levels = int(getattr(raw, "levels_completed", 0) or 0)
                level_gain = levels - root_levels
                if level_gain > 0 or final_state == "WIN":
                    break
            if rejected or len(path) < 2:
                continue
            final_sig = signatures[-1]
            root_change = _signature_change(root_signature, final_sig)
            unique_state_ratio = len({sig.frame_hash for sig in signatures}) / max(
                1,
                len(signatures),
            )
            sequence_key = (frame_hash, tuple(action.key() for action in path))
            repeat_penalty = min(
                0.18,
                0.045 * self.subgoal_sequence_attempts.get(sequence_key, 0),
            )
            score = (
                500.0 * max(0, level_gain)
                + 1000.0 * (final_state == "WIN")
                + 0.18 * root_change
                + 0.10 * min(1.0, cumulative_change)
                + 0.12 * best_role
                + 0.16 * min(1.0, memory_bonus)
                + 0.05 * unique_state_ratio
                + 0.03 * min(1.0, pressure)
                - 0.008 * len(path)
                - repeat_penalty
            )
            if score > best_score:
                best_path = path
                best_signatures = signatures
                best_score = score
                best_root_change = root_change
                best_level_gain = level_gain
                best_state = final_state

        if len(best_path) < 2:
            return None
        first = best_path[0]
        first.novelty_score = min(1.0, best_root_change)
        first.value_score = min(
            0.35,
            max(0.0, float(best_level_gain)) + 0.09 + 0.03 * min(1.0, pressure),
        )
        first.coherence_score = 0.45
        first.final_score = best_score
        first.explanation = (
            f"local simulator subgoal commitment selected coverage sequence; "
            f"path_len={len(best_path)}, repeated_state_visits={visits}, "
            f"root_change={best_root_change:.3f}, pressure={pressure:.2f}, "
            f"projected_gain={best_level_gain}, final_state={best_state}"
        )
        self._activate_plan(best_path, best_signatures)
        self._record_committed_trajectory(
            root_signature,
            best_path,
            best_signatures,
            best_score,
            "coverage_commitment",
        )
        self._record_frontier_path(best_signatures)
        self._note_selection(first)
        return first

    def _frontier_probe(
        self,
        game: Any,
        latest_frame: Any,
        actions: list[ActionCandidate],
        root_signature: Any,
        root_levels: int,
        started: float,
        seconds_limit: float,
        pressure: float = 0.0,
    ) -> ActionCandidate | None:
        if len(actions) < 3:
            return None
        depth_limit = min(self.config.max_depth, 6)
        node_limit = min(self.config.max_nodes, 1800)
        root_action_ids = _available_ids(getattr(latest_frame, "available_actions", []) or [])
        if not root_action_ids:
            root_action_ids = sorted({action.action_id for action in actions})
        frontier: list[PlanNode] = [
            PlanNode(copy.deepcopy(game), latest_frame, [], [], 0.0, 0)
        ]
        seen: set[tuple[str, int, tuple[str, ...]]] = {
            (root_signature.frame_hash, 0, ())
        }
        best_path: list[ActionCandidate] = []
        best_signatures: list[Any] = []
        best_score = 0.0
        best_depth = 0
        best_reason = ""
        nodes = 0

        for depth in range(depth_limit):
            next_frontier: list[PlanNode] = []
            for node in frontier:
                if self._expired(started, seconds_limit):
                    break
                node_actions = self._valid_actions(
                    node.game,
                    node.frame,
                    limit=max(self.config.branch_limit, self.config.beam_width * 2),
                )
                for action in node_actions[: max(1, self.config.branch_limit)]:
                    if self._expired(started, seconds_limit):
                        break
                    nodes += 1
                    if nodes > node_limit:
                        break
                    child_game = copy.deepcopy(node.game)
                    raw = self._perform(child_game, action)
                    if raw is None:
                        continue
                    state = _state_name(getattr(raw, "state", "NOT_FINISHED"))
                    if state == "GAME_OVER":
                        continue
                    child_sig = summarize_frame(getattr(raw, "frame", []))
                    levels = int(getattr(raw, "levels_completed", 0) or 0)
                    level_gain = levels - root_levels
                    candidate = _copy_candidate(action, source="local-simulator-frontier")
                    path = node.path + [candidate]
                    signatures = node.signatures + [child_sig]
                    if level_gain > 0 or state == "WIN":
                        first = path[0]
                        first.value_score = float(level_gain)
                        first.novelty_score = min(1.0, frame_distance(root_signature, child_sig))
                        first.coherence_score = 1.0 / (1.0 + depth)
                        first.final_score = 500.0 * level_gain + 1000.0 * (state == "WIN")
                        first.explanation = (
                            f"local simulator frontier found "
                            f"{'win' if state == 'WIN' else 'level gain'} at depth {depth + 1}; "
                            f"projected_gain={level_gain}"
                        )
                        self._activate_plan(path, signatures)
                        self._record_frontier_path(signatures)
                        self._note_selection(first)
                        return first

                    child_actions = _available_ids(getattr(raw, "available_actions", []) or [])
                    objective_probe = self._objective_probe(
                        child_game,
                        raw,
                        child_sig,
                        root_levels,
                        root_action_ids,
                        limit=max(4, min(self.config.branch_limit, self.config.beam_width * 2)),
                    )
                    if (
                        objective_probe.action is not None
                        and (objective_probe.level_gain > 0 or objective_probe.win)
                    ):
                        future_action = _copy_candidate(
                            objective_probe.action,
                            source="local-simulator-objective-proximity",
                        )
                        future_sig = objective_probe.signature
                        path_with_future = path + [future_action]
                        signatures_with_future = (
                            signatures + [future_sig]
                            if future_sig is not None
                            else signatures
                        )
                        first = path_with_future[0]
                        first.source = "local-simulator-objective-proximity"
                        first.value_score = float(objective_probe.level_gain)
                        first.novelty_score = min(
                            1.0,
                            _signature_change(root_signature, child_sig),
                        )
                        first.coherence_score = 1.0 / (1.0 + depth)
                        first.final_score = (
                            500.0 * objective_probe.level_gain
                            + 1000.0 * int(objective_probe.win)
                            + objective_probe.score
                        )
                        first.explanation = (
                            f"local simulator objective proximity found "
                            f"{'win' if objective_probe.win else 'level gain'} beyond frontier; "
                            f"depth={depth + 2}, reason={objective_probe.reason}"
                        )
                        self._activate_plan(path_with_future, signatures_with_future)
                        self._record_frontier_path(signatures_with_future)
                        self._note_selection(first)
                        return first
                    score, reason = self._frontier_score(
                        root_signature,
                        node.signatures[-1] if node.signatures else root_signature,
                        child_sig,
                        root_action_ids,
                        child_actions,
                        path,
                        objective_probe,
                    )
                    cumulative = node.score + score * (0.88 ** depth)
                    child = PlanNode(
                        child_game,
                        raw,
                        path,
                        signatures,
                        cumulative,
                        depth + 1,
                    )
                    selection_score = score - 0.018 * len(path) + 0.035 * pressure
                    min_path_len = 1 if pressure >= 0.75 else 2
                    if (
                        score > 0.0
                        and len(path) >= min_path_len
                        and selection_score > best_score
                    ):
                        best_path = path
                        best_signatures = signatures
                        best_score = selection_score
                        best_depth = depth + 1
                        best_reason = reason

                    child_key = _sequence_key(child_sig.frame_hash, path)
                    if child_key in seen:
                        continue
                    seen.add(child_key)
                    if cumulative > -0.08:
                        next_frontier.append(child)
            if nodes > node_limit or self._expired(started, seconds_limit):
                break
            next_frontier.sort(key=lambda item: item.score, reverse=True)
            frontier = self._diverse_plan_nodes(
                next_frontier,
                max(1, self.config.beam_width * 5),
            )
            if not frontier:
                break

        required_score = max(0.055, 0.12 - 0.065 * min(1.0, pressure))
        if not best_path or best_score < required_score:
            return None
        first = best_path[0]
        first.novelty_score = min(1.0, best_score)
        first.value_score = min(0.45, best_score)
        first.coherence_score = 0.58
        first.final_score = best_score
        first.explanation = (
            f"local simulator frontier selected subgoal; depth={best_depth}, "
            f"frontier_score={best_score:.3f}, reason={best_reason}, "
            f"pressure={pressure:.2f}"
        )
        self._activate_plan(best_path, best_signatures)
        self._record_frontier_path(best_signatures)
        self._note_selection(first)
        return first

    def _frontier_score(
        self,
        root_signature: Any,
        previous_signature: Any,
        current_signature: Any,
        root_action_ids: list[int],
        current_action_ids: list[int],
        path: list[ActionCandidate],
        objective_probe: ObjectiveProbe | None = None,
    ) -> tuple[float, str]:
        objective_probe = objective_probe or ObjectiveProbe()
        root_change = _signature_change(root_signature, current_signature)
        step_change = _signature_change(previous_signature, current_signature)
        root_components = max(1, len(getattr(root_signature, "components", []) or []))
        previous_components = max(1, len(getattr(previous_signature, "components", []) or []))
        component_delta = abs(
            len(getattr(current_signature, "components", []) or [])
            - len(getattr(previous_signature, "components", []) or [])
        ) / previous_components
        salience_delta = abs(
            len(getattr(current_signature, "salience_points", []) or [])
            - len(getattr(root_signature, "salience_points", []) or [])
        ) / max(1, len(getattr(root_signature, "salience_points", []) or []))
        root_actions = set(root_action_ids)
        current_actions = set(current_action_ids)
        gained_actions = len(current_actions - root_actions)
        lost_actions = len(root_actions - current_actions)
        affordance_expansion = gained_actions / max(1, len(root_actions))
        affordance_change = len(current_actions ^ root_actions) / max(
            1,
            len(root_actions | current_actions),
        )
        option_preservation = len(current_actions) / max(1, len(root_actions | current_actions))
        novelty = 1.0 / (
            1.0
            + self.observed_frame_counts.get(current_signature.frame_hash, 0)
            + self.frontier_frame_counts.get(current_signature.frame_hash, 0)
        )
        role = _role_consequence_score(
            root_signature,
            previous_signature,
            current_signature,
            objective_probe,
        )
        depth_cost = 0.025 * len(path)
        repeat_action_cost = 0.025 * max(
            0,
            len(path) - len({candidate.key() for candidate in path}),
        )
        option_loss_cost = 0.10 * (lost_actions / max(1, len(root_actions)))
        score = (
            0.21 * root_change
            + 0.11 * step_change
            + 0.28 * min(1.0, component_delta)
            + 0.26 * min(1.0, affordance_expansion)
            + 0.16 * min(1.0, affordance_change)
            + 0.10 * min(1.0, salience_delta)
            + 0.08 * option_preservation
            + 0.12 * novelty
            + 0.42 * role.score
            + 0.30 * max(0.0, objective_probe.score)
            - depth_cost
            - repeat_action_cost
            - option_loss_cost
        )
        reasons = [
            ("root_change", root_change),
            ("step_change", step_change),
            ("component_delta", min(1.0, component_delta)),
            ("affordance_expansion", min(1.0, affordance_expansion)),
            ("affordance_change", min(1.0, affordance_change)),
            ("salience_delta", min(1.0, salience_delta)),
            ("novelty", novelty),
            (role.reason, role.score),
            (
                f"objective_{objective_probe.reason or 'future_branching'}",
                max(0.0, objective_probe.score),
            ),
        ]
        reason = max(reasons, key=lambda item: item[1])[0]
        if role.score >= 0.20 and reason in {
            "root_change",
            "step_change",
            "component_delta",
            "salience_delta",
            "novelty",
        }:
            reason = role.reason
        if (
            len(getattr(current_signature, "components", []) or []) > root_components
            and role.score < 0.20
        ):
            reason = "new_visible_components"
        return max(-1.0, min(1.0, score)), reason

    def _objective_probe(
        self,
        game: Any,
        latest_frame: Any,
        current_signature: Any,
        root_levels: int,
        root_action_ids: list[int],
        limit: int,
    ) -> ObjectiveProbe:
        """Score whether a frontier state exposes objective-relevant futures."""
        actions = self._valid_actions(game, latest_frame, limit=limit)
        if not actions:
            return ObjectiveProbe()

        root_actions = set(root_action_ids)
        current_components = max(1, len(getattr(current_signature, "components", []) or []))
        unique_future_hashes: set[str] = set()
        changes: list[float] = []
        component_deltas: list[float] = []
        affordance_expansions: list[float] = []
        role_consequences: list[RoleConsequence] = []
        nonterminal_count = 0
        game_over_count = 0

        for action in actions[:limit]:
            child_game = copy.deepcopy(game)
            raw = self._perform(child_game, action)
            if raw is None:
                continue
            state = _state_name(getattr(raw, "state", "NOT_FINISHED"))
            child_sig = summarize_frame(getattr(raw, "frame", []))
            levels = int(getattr(raw, "levels_completed", 0) or 0)
            level_gain = levels - root_levels
            candidate = _copy_candidate(action, source="local-simulator-objective")
            if level_gain > 0 or state == "WIN":
                return ObjectiveProbe(
                    score=1.0,
                    reason="future_level_gain",
                    level_gain=level_gain,
                    win=state == "WIN",
                    action=candidate,
                    signature=child_sig,
                )
            if state == "GAME_OVER":
                game_over_count += 1
                continue

            nonterminal_count += 1
            unique_future_hashes.add(child_sig.frame_hash)
            changes.append(_signature_change(current_signature, child_sig))
            component_deltas.append(
                abs(
                    len(getattr(child_sig, "components", []) or [])
                    - len(getattr(current_signature, "components", []) or [])
                )
                / current_components
            )
            role_consequences.append(
                _role_consequence_score(current_signature, current_signature, child_sig)
            )
            child_actions = set(
                _available_ids(getattr(raw, "available_actions", []) or [])
            )
            affordance_expansions.append(
                len(child_actions - root_actions) / max(1, len(root_actions))
            )

        attempted = max(1, min(limit, len(actions)))
        diversity = len(unique_future_hashes) / attempted
        max_change = max(changes, default=0.0)
        avg_change = sum(changes) / max(1, len(changes))
        max_component_delta = max(component_deltas, default=0.0)
        max_affordance_expansion = max(affordance_expansions, default=0.0)
        best_role = max(role_consequences, key=lambda item: item.score, default=RoleConsequence())
        nonterminal_rate = nonterminal_count / attempted
        game_over_rate = game_over_count / attempted
        score = (
            0.22 * diversity
            + 0.20 * max_change
            + 0.16 * avg_change
            + 0.18 * min(1.0, max_component_delta)
            + 0.16 * min(1.0, max_affordance_expansion)
            + 0.18 * best_role.score
            + 0.10 * nonterminal_rate
            - 0.12 * game_over_rate
        )
        reasons = [
            ("future_diversity", diversity),
            ("future_state_change", max_change),
            ("future_component_delta", min(1.0, max_component_delta)),
            ("future_affordance_expansion", min(1.0, max_affordance_expansion)),
            (f"future_{best_role.reason}", best_role.score),
            ("future_option_preservation", nonterminal_rate),
        ]
        reason = max(reasons, key=lambda item: item[1])[0]
        return ObjectiveProbe(
            score=max(0.0, min(1.0, score)),
            reason=reason,
        )

    def _single_click_probe(
        self,
        game: Any,
        latest_frame: Any,
        root_levels: int,
        root_signature: Any,
        started: float,
        seconds_limit: float,
    ) -> ActionCandidate | None:
        scan_limit = min(
            self.config.max_nodes,
            max(128, self.config.branch_limit * 64),
            256,
        )
        actions = self._valid_actions(game, latest_frame, limit=scan_limit)
        click_actions = [action for action in actions if action.action_id == 6]
        if len(click_actions) <= self.config.branch_limit:
            return None

        nodes = 0
        for action in click_actions:
            if self._expired(started, seconds_limit):
                return None
            nodes += 1
            if nodes > scan_limit:
                return None
            child_game = copy.deepcopy(game)
            raw = self._perform(child_game, action)
            if raw is None:
                continue
            state = _state_name(getattr(raw, "state", "NOT_FINISHED"))
            levels = int(getattr(raw, "levels_completed", 0) or 0)
            level_gain = levels - root_levels
            if level_gain <= 0 and state != "WIN":
                continue

            child_sig = summarize_frame(getattr(raw, "frame", []))
            changed = frame_distance(root_signature, child_sig)
            first = _copy_candidate(action, source="local-simulator-click-scan")
            first.value_score = float(level_gain)
            first.novelty_score = min(1.0, changed)
            first.coherence_score = 1.0
            first.final_score = (
                500.0 * level_gain
                + 1000.0 * (state == "WIN")
                + 0.8 * changed
            )
            first.explanation = (
                f"local simulator broad click scan found "
                f"{'win' if state == 'WIN' else 'level gain'}; "
                f"projected_gain={level_gain}, scanned={nodes}"
            )
            self.last_plan = [first]
            self.queued_plan = []
            self.queued_preconditions = []
            return first
        return None

    def _information_probe(
        self,
        game: Any,
        actions: list[ActionCandidate],
        root_signature: Any,
        started: float,
        seconds_limit: float,
        pressure: float = 0.0,
    ) -> ActionCandidate | None:
        if len(actions) < 3:
            return None
        best: ActionCandidate | None = None
        best_score = 0.0
        for action in actions[: self.config.branch_limit]:
            if self._expired(started, seconds_limit):
                break
            child_game = copy.deepcopy(game)
            raw = self._perform(child_game, action)
            if raw is None:
                continue
            state = _state_name(getattr(raw, "state", "NOT_FINISHED"))
            if state == "GAME_OVER":
                continue
            child_sig = summarize_frame(getattr(raw, "frame", []))
            changed = frame_distance(root_signature, child_sig)
            child_visits = self.observed_frame_counts.get(child_sig.frame_hash, 0)
            component_delta = abs(
                len(getattr(child_sig, "components", []))
                - len(getattr(root_signature, "components", []))
            ) / max(1, len(getattr(root_signature, "components", [])))
            information_score = changed + 0.35 * min(1.0, component_delta)
            if pressure > 0.0 and child_visits:
                information_score /= 1.0 + pressure * min(3.0, float(child_visits))
            if information_score <= best_score:
                continue
            candidate = _copy_candidate(action, source="local-simulator-information")
            candidate.novelty_score = min(1.0, changed)
            candidate.value_score = min(0.35, information_score)
            candidate.coherence_score = 0.55
            candidate.final_score = information_score
            candidate.explanation = (
                f"local simulator information probe selected action; "
                f"frame_change={changed:.3f}, component_delta={component_delta:.3f}"
            )
            best = candidate
            best_score = information_score
        min_score = 0.030 + 0.030 * min(1.0, pressure)
        if best is None or best_score < min_score:
            return None
        self.last_plan = [best]
        self.queued_plan = []
        self.queued_preconditions = []
        self._note_selection(best)
        return best

    @staticmethod
    def _diverse_prefixes(
        frontier: list[tuple[Any, list[ActionCandidate], list[Any]]],
        limit: int,
    ) -> list[tuple[Any, list[ActionCandidate], list[Any]]]:
        if len(frontier) <= limit:
            return frontier
        buckets: dict[int, list[tuple[Any, list[ActionCandidate], list[Any]]]] = {}
        for item in frontier:
            path = item[1]
            first = path[0].action_id if path else -1
            buckets.setdefault(first, []).append(item)
        selected: list[tuple[Any, list[ActionCandidate], list[Any]]] = []
        while len(selected) < limit and buckets:
            for first in sorted(list(buckets)):
                bucket = buckets.get(first, [])
                if not bucket:
                    buckets.pop(first, None)
                    continue
                selected.append(bucket.pop(0))
                if len(selected) >= limit:
                    break
        return selected

    @staticmethod
    def _diverse_plan_nodes(
        frontier: list[PlanNode],
        limit: int,
    ) -> list[PlanNode]:
        if len(frontier) <= limit:
            return frontier
        buckets: dict[str, list[PlanNode]] = {}
        for node in frontier:
            first = node.path[0].key() if node.path else "-"
            buckets.setdefault(first, []).append(node)
        selected: list[PlanNode] = []
        while len(selected) < limit and buckets:
            for first in sorted(list(buckets)):
                bucket = buckets.get(first, [])
                if not bucket:
                    buckets.pop(first, None)
                    continue
                selected.append(bucket.pop(0))
                if len(selected) >= limit:
                    break
        return selected

    def _record_observed(self, signature: Any) -> None:
        frame_hash = getattr(signature, "frame_hash", "")
        if not frame_hash:
            return
        self.observed_frame_counts[frame_hash] = (
            self.observed_frame_counts.get(frame_hash, 0) + 1
        )
        if len(self.observed_frame_counts) > 512:
            for key in list(self.observed_frame_counts)[
                : len(self.observed_frame_counts) - 512
            ]:
                self.observed_frame_counts.pop(key, None)

    def _record_frontier_path(self, signatures: list[Any]) -> None:
        for signature in signatures:
            frame_hash = getattr(signature, "frame_hash", "")
            if not frame_hash:
                continue
            self.frontier_frame_counts[frame_hash] = (
                self.frontier_frame_counts.get(frame_hash, 0) + 1
            )
        if len(self.frontier_frame_counts) > 1024:
            for key in list(self.frontier_frame_counts)[
                : len(self.frontier_frame_counts) - 1024
            ]:
                self.frontier_frame_counts.pop(key, None)

    def _record_committed_trajectory(
        self,
        root_signature: Any,
        path: list[ActionCandidate],
        signatures: list[Any],
        score: float,
        reason: str,
    ) -> None:
        root_hash = getattr(root_signature, "frame_hash", "")
        if not root_hash or not path or not signatures:
            return
        sequence = tuple(action.key() for action in path[: len(signatures)])
        if sequence:
            sequence_key = (root_hash, sequence)
            self.subgoal_sequence_attempts[sequence_key] = (
                self.subgoal_sequence_attempts.get(sequence_key, 0) + 1
            )
        previous = root_signature
        for index, action in enumerate(path[: len(signatures)]):
            current = signatures[index]
            previous_hash = getattr(previous, "frame_hash", "")
            if not previous_hash:
                previous = current
                continue
            change = frame_distance(previous, current)
            role = _role_consequence_score(root_signature, previous, current)
            useful = (
                change >= 0.006
                or role.score >= 0.045
                or action.value_score > 0.10
                or score > 1.0
            )
            local_score = min(
                1.0,
                0.12 * max(0.0, min(1.0, score))
                + 0.50 * min(1.0, change)
                + 0.34 * role.score
                + 0.10 * max(0.0, action.value_score)
            )
            if reason:
                local_score += 0.012
            credit_key = (previous_hash, action.key())
            credit = self.subgoal_action_credit.setdefault(
                credit_key,
                SubgoalActionCredit(),
            )
            credit.attempts += 1
            credit.useful += int(useful)
            credit.total_score += local_score
            credit.total_change += change
            credit.total_role += role.score
            if action.value_score > 0.90:
                credit.total_level_gain += int(action.value_score)
            general_key = _general_action_key(action)
            general_credit = self.subgoal_general_action_credit.setdefault(
                general_key,
                SubgoalActionCredit(),
            )
            general_credit.attempts += 1
            general_credit.useful += int(useful)
            general_credit.total_score += local_score
            general_credit.total_change += change
            general_credit.total_role += role.score
            if action.value_score > 0.90:
                general_credit.total_level_gain += int(action.value_score)
            previous = current

        if len(self.subgoal_action_credit) > 2048:
            for key in list(self.subgoal_action_credit)[
                : len(self.subgoal_action_credit) - 2048
            ]:
                self.subgoal_action_credit.pop(key, None)
        if len(self.subgoal_general_action_credit) > 128:
            for key in list(self.subgoal_general_action_credit)[
                : len(self.subgoal_general_action_credit) - 128
            ]:
                self.subgoal_general_action_credit.pop(key, None)
        if len(self.subgoal_sequence_attempts) > 2048:
            for key in list(self.subgoal_sequence_attempts)[
                : len(self.subgoal_sequence_attempts) - 2048
            ]:
                self.subgoal_sequence_attempts.pop(key, None)

    def _subgoal_credit_score(
        self,
        signature: Any,
        action: ActionCandidate,
    ) -> float:
        frame_hash = getattr(signature, "frame_hash", "")
        exact_score = 0.0
        if not frame_hash:
            exact_score = 0.0
        else:
            credit = self.subgoal_action_credit.get((frame_hash, action.key()))
            if credit is not None:
                confidence = min(1.0, credit.attempts / 3.0)
                exact_score = credit.average * confidence
        general = self.subgoal_general_action_credit.get(_general_action_key(action))
        general_score = 0.0
        if general is not None:
            general_confidence = min(1.0, general.attempts / 8.0)
            general_score = 0.45 * general.average * general_confidence
        return max(0.0, min(1.0, max(exact_score, general_score)))

    def _frontier_pressure(
        self,
        root_signature: Any,
        actions: list[ActionCandidate],
    ) -> float:
        if len(actions) < 3:
            return 0.0
        frame_hash = getattr(root_signature, "frame_hash", "")
        repeated_state = max(0, self.observed_frame_counts.get(frame_hash, 0) - 2)
        repeated_pressure = min(1.0, repeated_state / 4.0)
        streak_pressure = min(
            1.0,
            max(0, self.information_probe_streak - 2) / 3.0,
        )
        return max(repeated_pressure, streak_pressure)

    def _note_selection(self, candidate: ActionCandidate) -> None:
        source = candidate.source
        if source == "local-simulator-information":
            self.information_probe_streak += 1
        elif (
            "frontier" in source
            or "objective" in source
            or "subgoal" in source
            or "frontier" in candidate.explanation
            or "objective proximity" in candidate.explanation
            or "subgoal commitment" in candidate.explanation
        ):
            self.information_probe_streak = 0
        elif source != "local-simulator-plan":
            self.information_probe_streak = max(0, self.information_probe_streak - 1)

    def _queued_action(self, latest_frame: Any) -> ActionCandidate | None:
        state = _state_name(getattr(latest_frame, "state", "NOT_FINISHED"))
        if state in {"WIN", "GAME_OVER", "NOT_PLAYED"}:
            self._clear_queued_plan()
            return None
        while self.queued_plan:
            if not self.queued_preconditions:
                self._clear_queued_plan()
                return None
            latest_signature = summarize_frame(getattr(latest_frame, "frame", []))
            expected_signature = self.queued_preconditions.pop(0)
            if frame_distance(expected_signature, latest_signature) > 0.015:
                self._clear_queued_plan()
                return None
            candidate = self.queued_plan.pop(0)
            candidate = _copy_candidate(candidate, source="local-simulator-plan")
            candidate.explanation = "continuing validated queued local simulator rollout"
            candidate.final_score = max(candidate.final_score, 0.001)
            return candidate
        return None

    def _activate_plan(
        self,
        path: list[ActionCandidate],
        signatures: list[Any],
    ) -> None:
        self.last_plan = path
        self.queued_plan = path[1:]
        self.queued_preconditions = signatures[: max(0, len(path) - 1)]
        if not self.queued_plan:
            self.queued_preconditions = []

    def _clear_queued_plan(self) -> None:
        self.queued_plan = []
        self.queued_preconditions = []

    def _limits(self, root_action_count: int) -> tuple[int, int, float]:
        if root_action_count <= 2:
            return (
                min(self.config.max_depth, 10),
                min(self.config.max_nodes, 240),
                min(self.config.max_seconds, 0.080),
            )
        return min(self.config.max_depth, 2), min(self.config.max_nodes, 32), 0.006

    @staticmethod
    def _expired(started: float, seconds_limit: float) -> bool:
        return time.perf_counter() - started > seconds_limit

    def _valid_actions(
        self,
        game: Any,
        latest_frame: Any | None,
        limit: int | None = None,
    ) -> list[ActionCandidate]:
        raw_actions: list[Any] = []
        if hasattr(game, "_get_valid_actions"):
            try:
                raw_actions = list(game._get_valid_actions())
            except Exception:
                raw_actions = []

        frame_signature = None
        if latest_frame is not None:
            try:
                frame_signature = summarize_frame(getattr(latest_frame, "frame", []))
            except Exception:
                frame_signature = None
        candidates = [_candidate_from_action(action) for action in raw_actions]
        candidates.extend(
            _available_frame_candidates(
                latest_frame,
                frame_signature,
                max(limit or self.config.branch_limit, self.config.branch_limit),
            )
        )
        return _rank_and_dedupe(
            candidates,
            self.config.branch_limit if limit is None else limit,
            frame_signature,
        )

    @staticmethod
    def _perform(game: Any, candidate: ActionCandidate) -> Any | None:
        try:
            action = GameAction.from_id(candidate.action_id)
            data = {}
            if candidate.x is not None and candidate.y is not None:
                data = {"x": int(candidate.x), "y": int(candidate.y)}
            return game.perform_action(ActionInput(id=action, data=data), raw=True)
        except Exception:
            return None


def _candidate_from_action(action: Any) -> ActionCandidate:
    action_id = action.id.value if hasattr(action.id, "value") else int(action.id)
    data = dict(getattr(action, "data", {}) or {})
    return ActionCandidate(
        action_id=action_id,
        x=data.get("x"),
        y=data.get("y"),
        source="valid-action",
        base_score=0.0,
        final_score=0.0,
    )


def _copy_candidate(candidate: ActionCandidate, source: str) -> ActionCandidate:
    return ActionCandidate(
        action_id=candidate.action_id,
        x=candidate.x,
        y=candidate.y,
        source=source,
        base_score=candidate.base_score,
        final_score=candidate.final_score,
    )


def _general_action_key(candidate: ActionCandidate) -> str:
    if candidate.action_id == 6:
        return "A6"
    return f"A{candidate.action_id}"


def _rank_and_dedupe(
    candidates: list[ActionCandidate], limit: int, frame_signature: Any | None = None
) -> list[ActionCandidate]:
    seen: set[str] = set()
    deduped: list[ActionCandidate] = []
    for candidate in candidates:
        key = candidate.key()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)

    def rank(candidate: ActionCandidate) -> tuple[int, int, float, int, int]:
        if candidate.action_id != 6:
            return (0, 0, 0.0, candidate.action_id, 0)
        x = int(candidate.x or 0)
        y = int(candidate.y or 0)
        center_distance = abs(x - 32) + abs(y - 32)
        salience_affinity = _click_salience_affinity(x, y, frame_signature)
        grid_bias = -abs((x % 8) - 4) - abs((y % 8) - 4)
        source_priority = 0 if candidate.source == "valid-action" else 1
        return (
            1,
            source_priority,
            -120.0 * salience_affinity + center_distance + 0.2 * grid_bias,
            x,
            y,
        )

    deduped.sort(key=rank)
    if len(deduped) <= limit:
        return deduped
    simple = [candidate for candidate in deduped if candidate.action_id != 6]
    complex_actions = [candidate for candidate in deduped if candidate.action_id == 6]
    remaining = max(0, limit - len(simple))
    if remaining <= 0:
        return simple[:limit]
    if frame_signature is not None and getattr(frame_signature, "salience_points", None):
        top_count = min(remaining, max(1, remaining * 2 // 3))
        sampled = complex_actions[:top_count]
        sampled_keys = {candidate.key() for candidate in sampled}
        stride = max(1, len(complex_actions) // max(1, remaining - len(sampled)))
        for candidate in complex_actions[::stride]:
            if candidate.key() in sampled_keys:
                continue
            sampled.append(candidate)
            sampled_keys.add(candidate.key())
            if len(sampled) >= remaining:
                break
    else:
        stride = max(1, len(complex_actions) // remaining)
        sampled = complex_actions[::stride][:remaining]
    return simple + sampled


def _available_frame_candidates(
    latest_frame: Any | None,
    frame_signature: Any | None,
    limit: int,
) -> list[ActionCandidate]:
    if latest_frame is None:
        return []
    available_ids = _available_ids(getattr(latest_frame, "available_actions", []) or [])
    out: list[ActionCandidate] = []
    for action_id in available_ids:
        if action_id == 6:
            out.extend(_synthetic_click_candidates(frame_signature, max(16, limit)))
        else:
            try:
                GameAction.from_id(action_id)
            except Exception:
                continue
            out.append(ActionCandidate(action_id=action_id, source="available-action"))
    return out


def _synthetic_click_candidates(
    frame_signature: Any | None,
    limit: int,
) -> list[ActionCandidate]:
    points: list[tuple[int, int, str]] = []
    if frame_signature is not None:
        for point in getattr(frame_signature, "salience_points", []) or []:
            points.append((int(point.x), int(point.y), f"synthetic-{point.source}"))
    width = int(getattr(frame_signature, "width", 64) or 64)
    height = int(getattr(frame_signature, "height", 64) or 64)
    max_x = max(0, min(63, width - 1))
    max_y = max(0, min(63, height - 1))
    step = 4 if limit >= 128 else 8
    for y in range(0, max_y + 1, step):
        for x in range(0, max_x + 1, step):
            points.append((x, y, "synthetic-coverage"))
    for x, y in [(max_x, max_y), (max_x, 0), (0, max_y), (max_x // 2, max_y // 2)]:
        points.append((x, y, "synthetic-corner"))

    seen: set[tuple[int, int]] = set()
    out: list[ActionCandidate] = []
    for x, y, source in points:
        key = (max(0, min(63, x)), max(0, min(63, y)))
        if key in seen:
            continue
        seen.add(key)
        out.append(ActionCandidate(action_id=6, x=key[0], y=key[1], source=source))
        if len(out) >= limit:
            break
    return out


def _available_ids(raw: list[Any]) -> list[int]:
    out: list[int] = []
    for item in raw:
        if hasattr(item, "value"):
            out.append(int(item.value))
        else:
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                continue
    return sorted(set(action for action in out if 0 <= action <= 7))


def _click_salience_affinity(
    x: int, y: int, frame_signature: Any | None
) -> float:
    if frame_signature is None:
        return 0.0
    points = getattr(frame_signature, "salience_points", []) or []
    best = 0.0
    for point in points[:24]:
        distance = abs(x - int(getattr(point, "x", 0))) + abs(y - int(getattr(point, "y", 0)))
        proximity = max(0.0, 1.0 - distance / 48.0)
        salience = float(getattr(point, "salience", 0.0) or 0.0)
        best = max(best, salience * proximity)
    return best


def _signature_change(left: Any, right: Any) -> float:
    change = frame_distance(left, right)
    if getattr(left, "frame_hash", "") != getattr(right, "frame_hash", ""):
        change = max(change, 0.012)
    return min(1.0, change)


def _role_consequence_score(
    root_signature: Any,
    previous_signature: Any,
    current_signature: Any,
    objective_probe: ObjectiveProbe | None = None,
) -> RoleConsequence:
    """Estimate whether a transition creates useful object/goal-role evidence.

    This deliberately uses only rendered-frame summaries. It rewards state
    changes that look like new actionable objects, object motion, boundary
    contact changes, or near-future affordances rather than raw pixel novelty.
    """
    objective_probe = objective_probe or ObjectiveProbe()
    root_compact = _compact_components(root_signature)
    previous_compact = _compact_components(previous_signature)
    current_compact = _compact_components(current_signature)

    compact_emergence = max(0, len(current_compact) - len(root_compact)) / max(
        1,
        len(root_compact) + 1,
    )
    new_color = len(_object_colors(current_signature) - _object_colors(root_signature)) / max(
        1,
        len(_object_colors(current_signature)),
    )
    compact_motion = _compact_motion(previous_compact, current_compact, current_signature)
    boundary_delta = _boundary_interaction_delta(root_signature, current_signature)
    area_rebalance = _compact_area_rebalance(root_compact, current_compact)
    future_object_value = max(0.0, objective_probe.score)

    score = (
        0.34 * min(1.0, compact_emergence)
        + 0.18 * min(1.0, new_color)
        + 0.18 * min(1.0, compact_motion)
        + 0.16 * min(1.0, boundary_delta)
        + 0.12 * min(1.0, area_rebalance)
        + 0.18 * future_object_value
    )
    signals = [
        ("role_new_compact_object", compact_emergence),
        ("role_new_object_color", new_color),
        ("role_compact_motion", compact_motion),
        ("role_boundary_interaction", boundary_delta),
        ("role_object_area_shift", area_rebalance),
        (f"role_future_{objective_probe.reason or 'branching'}", future_object_value),
    ]
    reason = max(signals, key=lambda item: item[1])[0]
    return RoleConsequence(score=max(0.0, min(1.0, score)), reason=reason)


def _compact_components(signature: Any) -> list[Any]:
    components = list(getattr(signature, "components", []) or [])
    frame_area = max(
        1,
        int(getattr(signature, "width", 0) or 0)
        * int(getattr(signature, "height", 0) or 0),
    )
    out: list[Any] = []
    for component in components:
        area = int(getattr(component, "area", 0) or 0)
        if area <= 0:
            continue
        bbox_area = max(
            1,
            int(getattr(component, "width", 1) or 1)
            * int(getattr(component, "height", 1) or 1),
        )
        fill = area / bbox_area
        area_ratio = area / frame_area
        if fill < 0.45:
            continue
        if area_ratio > 0.18:
            continue
        if bool(getattr(component, "edge_touch", False)) and area_ratio > 0.05:
            continue
        out.append(component)
    return out


def _object_colors(signature: Any) -> set[int]:
    return {int(getattr(component, "color", -1)) for component in _compact_components(signature)}


def _compact_motion(
    previous_components: list[Any],
    current_components: list[Any],
    signature: Any,
) -> float:
    if not previous_components or not current_components:
        return 0.0
    scale = max(1.0, float(max(getattr(signature, "width", 0), getattr(signature, "height", 0))))
    best = 0.0
    for current in current_components:
        current_color = getattr(current, "color", None)
        current_area = max(1.0, float(getattr(current, "area", 1) or 1))
        current_centroid = getattr(current, "centroid", (0.0, 0.0))
        for previous in previous_components:
            if getattr(previous, "color", None) != current_color:
                continue
            previous_area = max(1.0, float(getattr(previous, "area", 1) or 1))
            ratio = min(previous_area, current_area) / max(previous_area, current_area)
            if ratio < 0.45:
                continue
            previous_centroid = getattr(previous, "centroid", (0.0, 0.0))
            distance = (
                abs(float(current_centroid[0]) - float(previous_centroid[0]))
                + abs(float(current_centroid[1]) - float(previous_centroid[1]))
            )
            best = max(best, min(1.0, distance / scale))
    return best


def _boundary_interaction_delta(root_signature: Any, current_signature: Any) -> float:
    root_count = _boundary_component_count(root_signature)
    current_count = _boundary_component_count(current_signature)
    return abs(current_count - root_count) / max(1, root_count + current_count)


def _boundary_component_count(signature: Any) -> int:
    count = 0
    for component in _compact_components(signature):
        bbox = getattr(component, "bbox", (0, 0, 0, 0))
        width = int(getattr(signature, "width", 0) or 0)
        height = int(getattr(signature, "height", 0) or 0)
        near_edge = bbox[0] <= 1 or bbox[1] <= 1 or bbox[2] >= width - 2 or bbox[3] >= height - 2
        if bool(getattr(component, "edge_touch", False)) or near_edge:
            count += 1
    return count


def _compact_area_rebalance(root_components: list[Any], current_components: list[Any]) -> float:
    root_area = sum(int(getattr(component, "area", 0) or 0) for component in root_components)
    current_area = sum(
        int(getattr(component, "area", 0) or 0) for component in current_components
    )
    return abs(current_area - root_area) / max(1, root_area + current_area)


def _game_key(game: Any, frame_hash: str) -> str:
    return frame_hash


def _sequence_key(frame_hash: str, path: list[ActionCandidate]) -> tuple[str, int, tuple[str, ...]]:
    suffix = tuple(candidate.key() for candidate in path[-4:])
    return (frame_hash, len(path), suffix)


def _state_name(state: Any) -> str:
    if hasattr(state, "value"):
        return str(state.value)
    if hasattr(state, "name"):
        return str(state.name)
    return str(state)
