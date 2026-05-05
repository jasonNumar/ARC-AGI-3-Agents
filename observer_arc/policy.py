"""Observer-centric action model for ARC-AGI-3."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .action_comparator import ActionScoreSignals, ActionSubgoalComparator
from .audit import ArcAuditTrail
from .failed_policy import FailedPolicyMemory
from .goal_discovery import GoalDiscoveryMemory
from .latent_roles import LatentRoleMemory
from .memory import (
    ActionInformationMemory,
    BoundaryMemory,
    ExactTransitionMemory,
    GeneralizedActionMemory,
    ObjectRoleMemory,
    SequenceCreditMemory,
    SequenceFragmentMemory,
    SemanticFrameMemory,
    TrajectoryMemory,
    action_schema,
)
from .motif_memory import TransitionMotifMemory
from .state import ActionCandidate, ObserverArcState, WorldState
from .viability import ViabilityModel
from .vision import frame_distance, summarize_frame


RESET = 0
ACTION6 = 6
TERMINAL_STATES = {"WIN"}
RESET_STATES = {"NOT_PLAYED", "GAME_OVER"}
DEFAULT_SIMPLE_ACTIONS = [1, 2, 3, 4, 5, 7]
INVERSE_ACTIONS = {1: 2, 2: 1, 3: 4, 4: 3}


@dataclass
class ObserverArcModel:
    seed: int = 1729
    max_actions: int = 1000
    simple_action_prior: dict[int, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    click_probe_limit: int = 32
    repeat_window: int = 12
    planner_depth: int = 48
    planner_beam_width: int = 8
    planner_branch_limit: int = 18
    planner_max_nodes: int = 2200
    planner_max_seconds: float = 0.30
    planner_return_best_nonterminal: bool = True

    @classmethod
    def default(cls) -> "ObserverArcModel":
        return cls(
            simple_action_prior={1: 0.58, 2: 0.58, 3: 0.56, 4: 0.56, 5: 0.46, 7: 0.22},
            weights={
                "base": 0.12,
                "novelty": 0.31,
                "value": 0.34,
                "coherence": 0.18,
                "grounding": 0.16,
                "repetition": 0.29,
                "cliche": 0.22,
                "contradiction": 0.45,
            },
        )

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ObserverArcModel":
        model = cls.default()
        config_path = Path(path) if path else Path(__file__).with_name("model_config.json")
        if not config_path.exists():
            return model
        data = json.loads(config_path.read_text(encoding="utf-8"))
        model.seed = int(data.get("seed", model.seed))
        model.max_actions = int(data.get("max_actions", model.max_actions))
        model.click_probe_limit = int(data.get("click_probe_limit", model.click_probe_limit))
        model.repeat_window = int(data.get("repeat_window", model.repeat_window))
        model.planner_depth = int(data.get("planner_depth", model.planner_depth))
        model.planner_beam_width = int(data.get("planner_beam_width", model.planner_beam_width))
        model.planner_branch_limit = int(data.get("planner_branch_limit", model.planner_branch_limit))
        model.planner_max_nodes = int(data.get("planner_max_nodes", model.planner_max_nodes))
        model.planner_max_seconds = float(data.get("planner_max_seconds", model.planner_max_seconds))
        model.planner_return_best_nonterminal = bool(
            data.get(
                "planner_return_best_nonterminal",
                model.planner_return_best_nonterminal,
            )
        )
        if "simple_action_prior" in data:
            model.simple_action_prior = {int(k): float(v) for k, v in data["simple_action_prior"].items()}
        if "weights" in data:
            model.weights.update({str(k): float(v) for k, v in data["weights"].items()})
        return model


class ObserverArcPolicy:
    """Stateful model that separates self, other, world, and memory during play."""

    def __init__(
        self,
        game_id: str,
        model: ObserverArcModel | None = None,
    ) -> None:
        self.game_id = game_id
        self.model = model or ObserverArcModel.load()
        self.state = ObserverArcState()
        self.exact_memory = ExactTransitionMemory()
        self.general_memory = GeneralizedActionMemory()
        self.action_information_memory = ActionInformationMemory()
        self.trajectory_memory = TrajectoryMemory()
        self.boundary_memory = BoundaryMemory()
        self.object_role_memory = ObjectRoleMemory()
        self.sequence_credit_memory = SequenceCreditMemory()
        self.sequence_fragment_memory = SequenceFragmentMemory()
        self.transition_motif_memory = TransitionMotifMemory()
        self.goal_discovery = GoalDiscoveryMemory()
        self.failed_policy_memory = FailedPolicyMemory()
        self.viability_model = ViabilityModel()
        self.latent_role_memory = LatentRoleMemory()
        self.action_comparator = ActionSubgoalComparator()
        self.semantic_memory = SemanticFrameMemory()
        self.pending_world: WorldState | None = None
        self.pending_action: ActionCandidate | None = None
        self.recent_transition_trace: list[tuple[Any, ActionCandidate]] = []
        self.recent_action_schemas: list[str] = []
        self.audit_trail = ArcAuditTrail()
        self.action_index = 0
        self._initialize_observer_states()

    def choose(self, frames: list[Any], latest_frame: Any) -> ActionCandidate:
        world = self._observe(latest_frame)
        self._finalize_previous(world)
        self.state.last_world = world

        if world.game_state in TERMINAL_STATES:
            return self._candidate(RESET, "terminal-cleanup", world, final_score=-1.0)
        if world.game_state in RESET_STATES:
            selected = self._candidate(RESET, "required-reset", world, final_score=1.0)
            selected.explanation = "Reset is the only coherent action from this state."
            self._commit(world, selected)
            return selected

        candidates = self._generate_candidates(world)
        if not candidates:
            action_id = self._fallback_action(world)
            selected = self._candidate(action_id, "fallback", world, final_score=0.0)
            selected.explanation = "Fallback selected because no grounded candidates were available."
        else:
            selected = max(candidates, key=lambda candidate: candidate.final_score)

        self._commit(world, selected)
        return selected

    def commit_external_choice(self, latest_frame: Any, selected: ActionCandidate) -> None:
        """Record a simulator-planner action in the observer memory pipeline."""
        world = self._observe(latest_frame)
        self._finalize_previous(world)
        self.state.last_world = world
        self._commit(world, selected)

    def _initialize_observer_states(self) -> None:
        self.state.self_state.plan = [
            "Model the current frame before acting.",
            "Probe actions that reveal dynamics, then exploit transitions that changed score or state.",
            "Prefer valuable divergence over random exploration.",
        ]
        self.state.self_state.recent_commitments = [
            "Use only observed frames, available action ids, and local transition memory.",
            "Do not claim consciousness; observer state is a functional control structure.",
        ]
        self.state.self_state.uncertainty = {
            "goal_location": 0.85,
            "action_semantics": 0.75,
            "click_target": 0.70,
        }
        self.state.other_state.expected_baselines = [
            "uniform random action sampling",
            "repeated center clicking",
            "blind clockwise movement loops",
        ]
        self.state.other_state.likely_common_actions = ["A1,A2,A3,A4 cycle", "A6 at 32,32"]
        self.state.other_state.cliche_patterns = [
            "repeat same action after no frame change",
            "alternate inverse actions without new evidence",
            "reset before game over",
            "click center before salience",
        ]

    def _observe(self, latest_frame: Any) -> WorldState:
        signature = summarize_frame(getattr(latest_frame, "frame", []))
        game_state = _state_name(getattr(latest_frame, "state", "NOT_PLAYED"))
        levels = int(getattr(latest_frame, "levels_completed", 0) or 0)
        win_levels = int(getattr(latest_frame, "win_levels", 0) or 0)
        available = _available_ids(getattr(latest_frame, "available_actions", []))
        if not available:
            available = [1, 2, 3, 4, 5, 6, 7]
        previous = self.state.last_world
        score_delta = levels - previous.levels_completed if previous else 0
        changed = bool(previous and frame_distance(previous.frame, signature) > 0.015)
        return WorldState(
            game_id=self.game_id,
            frame=signature,
            game_state=game_state,
            levels_completed=levels,
            win_levels=win_levels,
            available_actions=available,
            action_index=self.action_index,
            score_delta=score_delta,
            frame_changed=changed,
            hard_constraints=[
                "Only choose ids from the current available action set.",
                "Use ACTION6 coordinates in the 0..63 display range.",
            ],
        )

    def _finalize_previous(self, world: WorldState) -> None:
        if not self.pending_world or not self.pending_action:
            return
        reward = self._transition_reward(self.pending_world, world)
        changed = frame_distance(self.pending_world.frame, world.frame) > 0.015
        game_over = world.game_state == "GAME_OVER"
        self.exact_memory.record(
            self.pending_world.frame.frame_hash,
            self.pending_action.key(),
            world.frame.frame_hash,
            reward,
            changed,
            game_over,
        )
        self.general_memory.record(
            self.pending_world.frame,
            self.pending_action,
            world.frame,
            reward,
            changed,
            game_over,
        )
        self.action_information_memory.record(
            self.pending_world.frame,
            self.pending_action,
            world.frame,
            game_over,
        )
        self.trajectory_memory.record(
            self.pending_world.frame,
            self.pending_action,
            world.frame,
            reward,
        )
        self.boundary_memory.record(
            self.pending_world.frame,
            self.pending_action,
            world.frame,
            reward,
        )
        self.object_role_memory.record(
            self.pending_world.frame,
            self.pending_action,
            world.frame,
            reward,
        )
        self.transition_motif_memory.record(
            self.pending_world.frame,
            self.pending_action,
            world.frame,
            reward,
            game_over,
        )
        self.goal_discovery.record(
            self.pending_world.frame,
            self.pending_action,
            world.frame,
            reward,
            game_over,
            self.pending_world.available_actions,
            world.available_actions,
        )
        self.failed_policy_memory.record(
            self.pending_world.frame,
            self.pending_action,
            world.frame,
            reward,
            changed,
            game_over,
        )
        self.viability_model.record(
            self.pending_world,
            self.pending_action,
            world,
            reward,
            changed,
            game_over,
        )
        self.latent_role_memory.record(
            self.pending_world,
            self.pending_action,
            world,
            reward,
            changed,
            game_over,
        )
        self.recent_transition_trace.append(
            (self.pending_world.frame, self.pending_action)
        )
        del self.recent_transition_trace[:-12]
        self.sequence_credit_memory.record(self.recent_transition_trace, reward)
        self.sequence_fragment_memory.record(self.recent_transition_trace, reward)
        self.semantic_memory.insert(world.frame, reward, self.pending_action.key())
        if reward > 0.25:
            self.state.memory_state.exact_success_anchors.append(
                f"{self.pending_action.key()} produced reward {reward:.2f}"
            )
        self.pending_world = None
        self.pending_action = None

    def _transition_reward(self, previous: WorldState, current: WorldState) -> float:
        level_gain = max(0, current.levels_completed - previous.levels_completed)
        frame_gain = frame_distance(previous.frame, current.frame)
        game_over_penalty = 0.65 if current.game_state == "GAME_OVER" else 0.0
        win_bonus = 1.0 if current.game_state == "WIN" else 0.0
        return min(1.5, 0.85 * level_gain + 0.25 * frame_gain + win_bonus - game_over_penalty)

    def _generate_candidates(self, world: WorldState) -> list[ActionCandidate]:
        candidates: list[ActionCandidate] = []
        available = set(world.available_actions)
        for action_id in DEFAULT_SIMPLE_ACTIONS:
            if action_id in available:
                candidates.append(self._score_simple(action_id, world))
        if ACTION6 in available:
            candidates.extend(self._score_clicks(world))
        return candidates

    def _score_simple(self, action_id: int, world: WorldState) -> ActionCandidate:
        candidate = self._candidate(action_id, "simple", world)
        action_key = candidate.key()
        stats = self.exact_memory.get(world.frame.frame_hash, action_key)
        visits = self.exact_memory.visit_count(world.frame.frame_hash, action_key)
        recent = self.state.memory_state.recent_action_keys
        repeat_count = recent[-self.model.repeat_window :].count(action_key)
        inverse_loop = bool(recent and INVERSE_ACTIONS.get(action_id) == _action_id_from_key(recent[-1]))
        no_change_penalty = 0.0
        learned_value = 0.0
        if stats:
            learned_value = max(-0.4, min(1.0, stats.average_reward + 0.25 * stats.change_rate))
            no_change_penalty = 0.35 if stats.count >= 2 and stats.change_rate < 0.15 else 0.0
        generalized = self.general_memory.estimate(world.frame, candidate)
        if generalized.confidence > 0.0:
            generalized_value = max(
                -0.3,
                min(0.8, generalized.value + 0.18 * generalized.change_rate),
            )
            learned_value = max(
                learned_value,
                generalized_value * min(0.75, generalized.confidence),
            )
            if generalized.confidence > 0.5 and generalized.change_rate < 0.10:
                no_change_penalty = max(no_change_penalty, 0.18)
        information = self.action_information_memory.estimate(world.frame, candidate)
        if information.confidence > 0.0:
            uncertainty = self.state.self_state.uncertainty.get("action_semantics", 0.5)
            information_value = min(
                0.32,
                information.information_gain
                * (0.65 + 0.35 * uncertainty)
                * min(1.0, 0.55 + information.confidence),
            )
            learned_value = max(learned_value, information_value)
            if information.game_over_rate > 0.20:
                no_change_penalty = max(
                    no_change_penalty,
                    0.28 * information.game_over_rate * information.confidence,
                )
        trajectory = self.trajectory_memory.estimate(world.frame, candidate)
        if trajectory.confidence > 0.0:
            learned_value = max(learned_value, trajectory.value)
            no_change_penalty = max(no_change_penalty, 0.20 * trajectory.boundary_risk)
        boundary = self.boundary_memory.estimate(world.frame, candidate)
        if boundary.confidence > 0.0 and boundary.risk > 0.0:
            no_change_penalty = max(
                no_change_penalty,
                min(0.38, 0.46 * boundary.risk * boundary.confidence),
            )
        role = self.object_role_memory.estimate(world.frame, candidate)
        if role.confidence > 0.0:
            learned_value = max(learned_value, role.value)
            if role.value < -0.03:
                no_change_penalty = max(no_change_penalty, min(0.18, -role.value))
        sequence = self.sequence_credit_memory.estimate(world.frame, candidate)
        if sequence.confidence > 0.0:
            learned_value = max(learned_value, sequence.value)
            if sequence.value < -0.03:
                no_change_penalty = max(no_change_penalty, min(0.22, -sequence.value))
        fragment = self.sequence_fragment_memory.estimate(
            world.frame,
            self.recent_action_schemas,
            candidate,
        )
        if fragment.confidence > 0.0:
            learned_value = max(learned_value, fragment.value)
            if fragment.value < -0.03:
                no_change_penalty = max(no_change_penalty, min(0.22, -fragment.value))
        motif = self.transition_motif_memory.estimate(world.frame, candidate)
        if motif.confidence > 0.0:
            motif_value = motif.value * min(1.0, 0.55 + motif.confidence)
            learned_value = max(learned_value, motif_value)
            if motif.game_over_rate > 0.20:
                no_change_penalty = max(
                    no_change_penalty,
                    0.30 * motif.game_over_rate * motif.confidence,
                )
        goal = self.goal_discovery.estimate(world.frame, candidate)
        if goal.confidence > 0.0:
            goal_value = goal.affinity * min(1.0, 0.55 + goal.confidence)
            learned_value = max(learned_value, goal_value)
            if goal.game_over_rate > 0.20:
                no_change_penalty = max(
                    no_change_penalty,
                    0.34 * goal.game_over_rate * goal.confidence,
                )
        failed = self.failed_policy_memory.estimate(world.frame, candidate)
        if failed.confidence > 0.0:
            no_change_penalty = max(
                no_change_penalty,
                min(0.34, 0.38 * failed.failure_risk * failed.confidence),
            )
        viability = self.viability_model.estimate(world, candidate, self.model.max_actions)
        if viability.confidence > 0.0 or viability.mortality_risk > 0.0:
            viability_weight = max(0.35, viability.confidence)
            no_change_penalty = max(
                no_change_penalty,
                min(0.42, 0.48 * viability.mortality_risk * viability_weight),
            )
            if viability.mortality_risk < 0.16 and viability.empowerment_delta > 0.0:
                learned_value = max(
                    learned_value,
                    min(0.18, 0.18 * viability.empowerment_delta * viability_weight),
                )
        latent_role = self.latent_role_memory.estimate(world, candidate)
        if latent_role.confidence > 0.0:
            latent_weight = min(1.0, 0.55 + latent_role.confidence)
            if latent_role.value > 0.0:
                learned_value = max(
                    learned_value,
                    min(0.36, latent_role.value * latent_weight),
                )
            if latent_role.risk > 0.0:
                no_change_penalty = max(
                    no_change_penalty,
                    min(0.42, 0.46 * latent_role.risk * latent_weight),
                )
        continue_bonus = 0.0
        if recent and recent[-1] == action_key and stats and stats.change_rate > 0.6 and stats.average_reward >= 0:
            continue_bonus = 0.16
        novelty = 1.0 / (1.0 + visits)
        cliche = (
            0.28 * repeat_count
            + (0.22 if inverse_loop else 0.0)
            + 0.22 * failed.failure_risk * failed.confidence
            + 0.16 * viability.mortality_risk * max(0.35, viability.confidence)
            + 0.14 * latent_role.risk * max(0.35, latent_role.confidence)
        )
        repetition = min(1.0, 0.22 * repeat_count + no_change_penalty)
        coherence = 0.52 + continue_bonus - (0.18 if inverse_loop else 0.0)
        base = self.model.simple_action_prior.get(action_id, 0.30)
        candidate.base_score = base
        candidate.novelty_score = novelty
        candidate.value_score = learned_value
        candidate.coherence_score = max(0.0, min(1.0, coherence))
        candidate.repetition_penalty = max(repetition, no_change_penalty)
        candidate.cliche_penalty = min(1.0, cliche)
        candidate.contradiction_penalty = 0.0 if action_id in world.available_actions else 1.0
        candidate.final_score = self._final_score(candidate)
        comparator_signals = ActionScoreSignals(
            progress=max(0.0, learned_value),
            information_gain=max(
                information.information_gain * min(1.0, information.confidence),
                motif.information_gain * min(1.0, motif.confidence),
                goal.information_gain * min(1.0, goal.confidence),
                latent_role.information_gain * min(1.0, latent_role.confidence),
            ),
            goal_affinity=max(0.0, goal.affinity * min(1.0, goal.confidence)),
            consistency=max(
                generalized.confidence * max(0.0, generalized.change_rate),
                trajectory.confidence * max(0.0, min(1.0, 3.0 * trajectory.value)),
                role.confidence * max(0.0, 0.5 + role.value),
                sequence.confidence * max(0.0, 0.5 + sequence.value),
                fragment.confidence * max(0.0, 0.5 + fragment.value),
                motif.confidence * (1.0 if motif.family else 0.5),
                latent_role.confidence
                * max(0.0, 0.5 + latent_role.value - latent_role.risk),
            ),
            empowerment=max(0.0, viability.empowerment * max(0.35, viability.confidence)),
            mortality_risk=max(
                viability.mortality_risk * max(0.35, viability.confidence),
                latent_role.risk * max(0.35, latent_role.confidence),
            ),
            repeat_penalty=max(candidate.repetition_penalty, candidate.cliche_penalty),
            budget_cost=self._budget_cost(world, candidate),
        )
        comparator_adjustment = self._apply_comparator(candidate, comparator_signals)
        candidate.explanation = (
            f"simple action A{action_id}; novelty={novelty:.2f}, "
            f"learned_value={learned_value:.2f}, generalized={generalized.confidence:.2f}, "
            f"information={information.confidence:.2f}, "
            f"trajectory={trajectory.confidence:.2f}, boundary={boundary.risk:.2f}, "
            f"role={role.confidence:.2f}, "
            f"sequence={sequence.confidence:.2f}, fragment={fragment.confidence:.2f}, "
            f"motif={motif.confidence:.2f}, goal={goal.confidence:.2f}, "
            f"failed={failed.failure_risk:.2f}, "
            f"latent_role={latent_role.role or '-'}:{latent_role.confidence:.2f}/"
            f"{latent_role.risk:.2f}, "
            f"viability={viability.mortality_risk:.2f}, empowerment={viability.empowerment:.2f}, "
            f"repeat={candidate.repetition_penalty:.2f}, "
            f"comparator={comparator_adjustment:.2f}[{comparator_signals.compact()}]"
        )
        return candidate

    def _score_clicks(self, world: WorldState) -> list[ActionCandidate]:
        candidates: list[ActionCandidate] = []
        recent_click_keys = {
            key
            for key in self.state.memory_state.recent_action_keys[-self.model.repeat_window :]
            if key.startswith("A6:")
        }
        for point in world.frame.salience_points[: self.model.click_probe_limit]:
            candidate = self._candidate(ACTION6, point.source, world, x=point.x, y=point.y)
            action_key = candidate.key()
            stats = self.exact_memory.get(world.frame.frame_hash, action_key)
            visits = self.exact_memory.visit_count(world.frame.frame_hash, action_key)
            near_recent = action_key in recent_click_keys
            center_cliche = 0.20 if abs(point.x - 32) <= 4 and abs(point.y - 32) <= 4 and point.source == "coverage-grid" else 0.0
            learned_value = 0.0
            if stats:
                learned_value = max(-0.5, min(1.0, stats.average_reward + 0.25 * stats.change_rate))
            semantic_bonus = self._semantic_click_bonus(world, action_key)
            generalized = self.general_memory.estimate(world.frame, candidate)
            generalized_bonus = 0.0
            if generalized.confidence > 0.0:
                generalized_bonus = max(
                    -0.12,
                    min(
                        0.30,
                        (generalized.value + 0.16 * generalized.change_rate)
                        * min(0.75, generalized.confidence),
                    ),
                )
            information = self.action_information_memory.estimate(world.frame, candidate)
            information_bonus = 0.0
            if information.confidence > 0.0:
                uncertainty = self.state.self_state.uncertainty.get("click_target", 0.5)
                information_bonus = min(
                    0.26,
                    information.information_gain
                    * (0.60 + 0.40 * uncertainty)
                    * min(1.0, 0.50 + information.confidence),
                )
            sequence = self.sequence_credit_memory.estimate(world.frame, candidate)
            sequence_bonus = 0.0
            if sequence.confidence > 0.0:
                sequence_bonus = max(
                    -0.16,
                    min(0.32, sequence.value * min(0.80, sequence.confidence)),
                )
            fragment = self.sequence_fragment_memory.estimate(
                world.frame,
                self.recent_action_schemas,
                candidate,
            )
            fragment_bonus = 0.0
            if fragment.confidence > 0.0:
                fragment_bonus = max(
                    -0.16,
                    min(0.32, fragment.value * min(0.80, fragment.confidence)),
                )
            motif = self.transition_motif_memory.estimate(world.frame, candidate)
            motif_bonus = 0.0
            if motif.confidence > 0.0:
                motif_bonus = max(
                    -0.14,
                    min(0.28, motif.value * min(0.85, motif.confidence)),
                )
            goal = self.goal_discovery.estimate(world.frame, candidate)
            goal_bonus = 0.0
            if goal.confidence > 0.0:
                goal_bonus = max(
                    -0.14,
                    min(0.28, goal.affinity * min(0.85, goal.confidence)),
                )
            failed = self.failed_policy_memory.estimate(world.frame, candidate)
            viability = self.viability_model.estimate(world, candidate, self.model.max_actions)
            viability_bonus = 0.0
            if viability.mortality_risk < 0.16 and viability.empowerment_delta > 0.0:
                viability_bonus = min(
                    0.14,
                    0.16
                    * viability.empowerment_delta
                    * max(0.35, viability.confidence),
                )
            latent_role = self.latent_role_memory.estimate(world, candidate)
            latent_role_bonus = 0.0
            if latent_role.confidence > 0.0 and latent_role.value > 0.0:
                latent_role_bonus = min(
                    0.30,
                    latent_role.value * min(1.0, 0.55 + latent_role.confidence),
                )
            candidate.base_score = 0.34 + 0.28 * point.salience
            candidate.novelty_score = 1.0 / (1.0 + visits)
            candidate.value_score = (
                learned_value
                + semantic_bonus
                + generalized_bonus
                + information_bonus
                + sequence_bonus
                + fragment_bonus
                + motif_bonus
                + goal_bonus
                + viability_bonus
                + latent_role_bonus
            )
            candidate.coherence_score = 0.44 + 0.35 * point.salience
            candidate.repetition_penalty = max(
                0.42 if near_recent else 0.0,
                0.24 * information.game_over_rate * information.confidence,
                0.26 * motif.game_over_rate * motif.confidence,
                0.30 * goal.game_over_rate * goal.confidence,
                0.34 * failed.failure_risk * failed.confidence,
                0.40 * viability.mortality_risk * max(0.35, viability.confidence),
                0.42 * latent_role.risk * max(0.35, latent_role.confidence),
            )
            candidate.cliche_penalty = min(
                1.0,
                center_cliche
                + 0.22 * failed.failure_risk * failed.confidence
                + 0.14 * viability.mortality_risk * max(0.35, viability.confidence)
                + 0.12 * latent_role.risk * max(0.35, latent_role.confidence),
            )
            candidate.contradiction_penalty = 0.0
            candidate.final_score = self._final_score(candidate)
            comparator_signals = ActionScoreSignals(
                progress=max(0.0, candidate.value_score),
                information_gain=max(
                    information.information_gain * min(1.0, information.confidence),
                    motif.information_gain * min(1.0, motif.confidence),
                    goal.information_gain * min(1.0, goal.confidence),
                    latent_role.information_gain * min(1.0, latent_role.confidence),
                ),
                goal_affinity=max(0.0, goal.affinity * min(1.0, goal.confidence)),
                consistency=max(
                    point.salience,
                    generalized.confidence * max(0.0, generalized.change_rate),
                    sequence.confidence * max(0.0, 0.5 + sequence.value),
                    fragment.confidence * max(0.0, 0.5 + fragment.value),
                    motif.confidence * (1.0 if motif.family else 0.5),
                    latent_role.confidence
                    * max(0.0, 0.5 + latent_role.value - latent_role.risk),
                ),
                empowerment=max(0.0, viability.empowerment * max(0.35, viability.confidence)),
                mortality_risk=max(
                    viability.mortality_risk * max(0.35, viability.confidence),
                    latent_role.risk * max(0.35, latent_role.confidence),
                ),
                repeat_penalty=max(candidate.repetition_penalty, candidate.cliche_penalty),
                budget_cost=self._budget_cost(world, candidate),
            )
            comparator_adjustment = self._apply_comparator(candidate, comparator_signals)
            candidate.explanation = (
                f"click {point.source} at ({point.x},{point.y}); "
                f"salience={point.salience:.2f}, novelty={candidate.novelty_score:.2f}, "
                f"semantic={semantic_bonus:.2f}, generalized={generalized_bonus:.2f}, "
                f"information={information_bonus:.2f}, "
                f"sequence={sequence_bonus:.2f}, fragment={fragment_bonus:.2f}, "
                f"motif={motif_bonus:.2f}, goal={goal_bonus:.2f}, "
                f"failed={failed.failure_risk:.2f}, "
                f"latent_role={latent_role.role or '-'}:{latent_role.confidence:.2f}/"
                f"{latent_role.risk:.2f}, "
                f"viability={viability.mortality_risk:.2f}, empowerment={viability.empowerment:.2f}, "
                f"comparator={comparator_adjustment:.2f}[{comparator_signals.compact()}]"
            )
            candidates.append(candidate)
        return candidates

    def _semantic_click_bonus(self, world: WorldState, action_key: str) -> float:
        similar = self.semantic_memory.retrieve_similar(world.frame, k=6)
        if not similar:
            return 0.0
        matching = [record.score for record in similar if record.action_key == action_key]
        if matching:
            return max(-0.2, min(0.35, sum(matching) / len(matching) * 0.2))
        unusual = self.semantic_memory.retrieve_unusual_associations(world.frame, k=4)
        if any(record.action_key.startswith("A6") and record.score > 0.15 for record in unusual):
            return 0.06
        return 0.0

    def _apply_comparator(
        self,
        candidate: ActionCandidate,
        signals: ActionScoreSignals,
    ) -> float:
        adjustment = self.action_comparator.adjustment(signals)
        candidate.final_score += adjustment
        return adjustment

    def _budget_cost(self, world: WorldState, candidate: ActionCandidate) -> float:
        elapsed = world.action_index / max(1, self.model.max_actions)
        action_complexity = 0.12 if candidate.action_id == ACTION6 else 0.02
        return min(1.0, elapsed + action_complexity)

    def _final_score(self, candidate: ActionCandidate) -> float:
        weights = self.model.weights
        useful_novelty = candidate.novelty_score * max(0.0, 0.35 + candidate.value_score)
        return (
            weights["base"] * candidate.base_score
            + weights["novelty"] * self.state.self_state.novelty_budget * useful_novelty
            + weights["value"] * candidate.value_score
            + weights["coherence"] * candidate.coherence_score
            + weights["grounding"] * candidate.grounding_score
            - weights["repetition"] * candidate.repetition_penalty
            - weights["cliche"] * candidate.cliche_penalty
            - weights["contradiction"] * candidate.contradiction_penalty
        )

    def _candidate(
        self,
        action_id: int,
        source: str,
        world: WorldState,
        x: int | None = None,
        y: int | None = None,
        final_score: float | None = None,
    ) -> ActionCandidate:
        candidate = ActionCandidate(
            action_id=action_id,
            x=x,
            y=y,
            source=source,
            grounding_score=1.0 if action_id == RESET or action_id in world.available_actions else 0.0,
        )
        if final_score is not None:
            candidate.final_score = final_score
        return candidate

    def _fallback_action(self, world: WorldState) -> int:
        for action_id in [1, 4, 2, 3, 5, 7, 6]:
            if action_id in world.available_actions:
                return action_id
        return RESET

    def _commit(self, world: WorldState, selected: ActionCandidate) -> None:
        self.action_index += 1
        state_hash = world.frame.frame_hash
        memory = self.state.memory_state
        memory.visited_state_counts[state_hash] = memory.visited_state_counts.get(state_hash, 0) + 1
        memory.recent_state_hashes.append(state_hash)
        memory.recent_action_keys.append(selected.key())
        self.recent_action_schemas.append(action_schema(selected))
        del memory.recent_state_hashes[:-64]
        del memory.recent_action_keys[:-64]
        del self.recent_action_schemas[:-64]
        if selected.final_score < -0.05:
            self.state.self_state.suppressed_options.append(selected.key())
        self.state.self_state.current_hypotheses = self._hypotheses(world, selected)
        self.audit_trail.record(world, selected, self.state.self_state.current_hypotheses)
        self.state.selected_candidates.append(selected)
        del self.state.selected_candidates[:-32]
        self.pending_world = world
        self.pending_action = selected

    def _hypotheses(self, world: WorldState, selected: ActionCandidate) -> list[str]:
        hypotheses = []
        if 6 in world.available_actions:
            hypotheses.append("Visible components may be clickable controls or targets.")
        if any(action in world.available_actions for action in [1, 2, 3, 4]):
            hypotheses.append("Directional actions likely move or transform an avatar/object.")
        if selected.value_score > 0.15:
            hypotheses.append(f"Reuse transition family {selected.key()} because it previously changed the world.")
        elif selected.novelty_score > 0.7:
            hypotheses.append(f"Probe {selected.key()} to reduce action-semantics uncertainty.")
        information = self.action_information_memory.estimate(world.frame, selected)
        if information.confidence > 0.0 and information.information_gain > 0.04:
            hypotheses.append(
                f"{selected.key()} tends to reveal new observable state "
                f"({information.information_gain:.2f} information gain)."
            )
        sequence = self.sequence_credit_memory.estimate(world.frame, selected)
        if sequence.confidence > 0.0 and sequence.value > 0.05:
            hypotheses.append(
                f"{selected.key()} has delayed sequence credit over about {sequence.average_horizon:.1f} steps."
            )
        fragment = self.sequence_fragment_memory.estimate(
            world.frame,
            self.recent_action_schemas[:-1],
            selected,
        )
        if fragment.confidence > 0.0 and fragment.value > 0.05:
            hypotheses.append(
                f"{selected.key()} continues a learned action fragment of length {fragment.matched_length}."
            )
        hypotheses.extend(self.trajectory_memory.hypotheses(world.frame))
        hypotheses.extend(self.boundary_memory.hypotheses(world.frame))
        hypotheses.extend(self.object_role_memory.hypotheses(world.frame))
        hypotheses.extend(self.transition_motif_memory.hypotheses(world.frame, selected))
        hypotheses.extend(self.goal_discovery.hypotheses(world.frame, selected))
        hypotheses.extend(self.failed_policy_memory.hypotheses(world.frame, selected))
        hypotheses.extend(self.viability_model.hypotheses(world, selected, self.model.max_actions))
        hypotheses.extend(self.latent_role_memory.hypotheses(world, selected))
        return hypotheses


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


def _state_name(state: Any) -> str:
    if hasattr(state, "value"):
        return str(state.value)
    if hasattr(state, "name"):
        return str(state.name)
    return str(state)


def _action_id_from_key(key: str) -> int | None:
    if not key.startswith("A"):
        return None
    digits = []
    for char in key[1:]:
        if char.isdigit():
            digits.append(char)
        else:
            break
    if not digits:
        return None
    return int("".join(digits))
