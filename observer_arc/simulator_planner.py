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
    path: list[ActionCandidate]
    signatures: list[Any]
    score: float
    depth: int


class LocalSimulatorPlanner:
    """Beam-search planner over cloned local ARC game states."""

    def __init__(self, config: PlannerConfig | None = None) -> None:
        self.config = config or PlannerConfig()
        self.last_plan: list[ActionCandidate] = []
        self.queued_plan: list[ActionCandidate] = []
        self.queued_preconditions: list[Any] = []

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
        informative = None
        informative = self._information_probe(
            game,
            root_actions,
            root_signature,
            time.perf_counter(),
            min(self.config.max_seconds, 0.020),
        )
        if informative is not None:
            return informative
        if self._expired(started, seconds_limit):
            return None

        frontier = [PlanNode(copy.deepcopy(game), [], [], 0.0, 0)]
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
                actions = self._valid_actions(node.game, latest_frame if depth == 0 else None)
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
                    if state != "GAME_OVER":
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
            next_frontier: list[tuple[Any, list[ActionCandidate], list[Any], Any, float, float]] = []
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
            component_delta = abs(
                len(getattr(child_sig, "components", []))
                - len(getattr(root_signature, "components", []))
            ) / max(1, len(getattr(root_signature, "components", [])))
            information_score = changed + 0.35 * min(1.0, component_delta)
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
        if best is None or best_score < 0.030:
            return None
        self.last_plan = [best]
        self.queued_plan = []
        self.queued_preconditions = []
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
