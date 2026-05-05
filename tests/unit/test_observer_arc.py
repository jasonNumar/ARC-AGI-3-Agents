from dataclasses import dataclass

import numpy as np
from arcengine import ActionInput, FrameDataRaw, GameAction, GameState

from observer_arc.action_comparator import ActionScoreSignals, ActionSubgoalComparator
from observer_arc.failed_policy import FailedPolicyMemory
from observer_arc.latent_roles import LatentRoleMemory
from observer_arc.policy import ObserverArcPolicy
from observer_arc.simulator_planner import LocalSimulatorPlanner, PlannerConfig
from observer_arc.state import ActionCandidate
from observer_arc.state_graph import build_state_graph, transition_delta
from observer_arc.viability import ViabilityModel, approximate_empowerment
from observer_arc.vision import summarize_frame


@dataclass
class StubFrame:
    frame: list
    state: str = "NOT_FINISHED"
    levels_completed: int = 0
    win_levels: int = 1
    available_actions: list[int] = None

    def __post_init__(self):
        if self.available_actions is None:
            self.available_actions = [1, 2, 3, 4, 6]


def test_summarize_frame_finds_components_and_click_points():
    grid = [[0 for _ in range(64)] for _ in range(64)]
    for y in range(10, 16):
        for x in range(20, 27):
            grid[y][x] = 5

    summary = summarize_frame([grid])

    assert summary.background == 0
    assert summary.components
    assert summary.salience_points
    first = summary.salience_points[0]
    assert 20 <= first.x <= 26
    assert 10 <= first.y <= 15


def test_state_graph_extracts_transition_delta_motifs():
    before = [[0 for _ in range(16)] for _ in range(16)]
    after = [[0 for _ in range(16)] for _ in range(16)]
    for y in range(4, 7):
        for x in range(2, 5):
            before[y][x] = 3
        for x in range(7, 10):
            before[y][x] = 6
            after[y][x] = 6
        for x in range(4, 7):
            after[y][x] = 3

    before_sig = summarize_frame([before])
    after_sig = summarize_frame([after])
    graph = build_state_graph(before_sig)
    delta = transition_delta(before_sig, after_sig, "A4")

    assert graph.objects
    assert graph.relations
    assert delta.moved_count >= 1
    assert delta.information_gain > 0.0
    assert delta.family in {"move", "relational_move", "relation_change"}


def test_action_subgoal_comparator_rewards_useful_state_sequence_signals():
    comparator = ActionSubgoalComparator()
    useful = comparator.adjustment(
        ActionScoreSignals(
            progress=0.55,
            information_gain=0.45,
            goal_affinity=0.40,
            consistency=0.70,
            repeat_penalty=0.05,
            budget_cost=0.10,
        )
    )
    looping = comparator.adjustment(
        ActionScoreSignals(
            progress=0.02,
            information_gain=0.00,
            goal_affinity=0.00,
            consistency=0.05,
            repeat_penalty=0.95,
            budget_cost=0.60,
        )
    )

    assert useful > 0.20
    assert looping < 0.0
    assert comparator.score(0.10, ActionScoreSignals(progress=1.0)) > 0.10


def test_failed_policy_memory_marks_no_progress_branches():
    grid = [[0 for _ in range(16)] for _ in range(16)]
    grid[4][4] = 8
    signature = summarize_frame([grid])
    memory = FailedPolicyMemory()
    action = ActionCandidate(action_id=4, source="test")

    for _ in range(3):
        memory.record(
            signature,
            action,
            signature,
            reward=0.0,
            changed=False,
            game_over=False,
        )

    estimate = memory.estimate(signature, action)

    assert estimate.failure_risk > 0.30
    assert estimate.confidence > 0.50
    assert "no-progress" in memory.hypotheses(signature, action)[0]


def test_viability_model_tracks_episode_local_mortality_and_empowerment():
    before_grid = [[0 for _ in range(16)] for _ in range(16)]
    after_grid = [[0 for _ in range(16)] for _ in range(16)]
    before_grid[4][4] = 8
    after_grid[4][4] = 8
    policy = ObserverArcPolicy("stub")
    previous = policy._observe(
        StubFrame([before_grid], available_actions=[1, 2, 3, 4, 6])
    )
    current = policy._observe(
        StubFrame([after_grid], state="GAME_OVER", available_actions=[0])
    )
    model = ViabilityModel()

    for _ in range(3):
        model.record(
            previous,
            ActionCandidate(action_id=4, source="test"),
            current,
            reward=-0.6,
            changed=False,
            game_over=True,
        )

    estimate = model.estimate(
        previous,
        ActionCandidate(action_id=4, source="test"),
        max_actions=100,
    )

    assert approximate_empowerment(previous) > approximate_empowerment(current)
    assert estimate.mortality_risk > 0.55
    assert estimate.confidence > 0.50
    assert "terminal" in model.hypotheses(
        previous,
        ActionCandidate(action_id=4, source="test"),
        max_actions=100,
    )[0]


def test_latent_role_memory_infers_key_and_threat_roles():
    base_grid = [[0 for _ in range(16)] for _ in range(16)]
    key_grid = [[0 for _ in range(16)] for _ in range(16)]
    for y in range(4, 7):
        for x in range(4, 7):
            base_grid[y][x] = 5
            key_grid[y][x] = 5
        for x in range(10, 13):
            key_grid[y][x] = 8

    policy = ObserverArcPolicy("stub")
    previous = policy._observe(StubFrame([base_grid], available_actions=[1, 5]))
    key_state = policy._observe(StubFrame([key_grid], available_actions=[1, 5, 6]))
    loss_state = policy._observe(
        StubFrame([base_grid], state="GAME_OVER", available_actions=[0])
    )
    memory = LatentRoleMemory()

    for _ in range(3):
        memory.record(
            previous,
            ActionCandidate(action_id=5, source="test"),
            key_state,
            reward=0.1,
            changed=True,
            game_over=False,
        )
        memory.record(
            previous,
            ActionCandidate(action_id=4, source="test"),
            loss_state,
            reward=-0.6,
            changed=False,
            game_over=True,
        )

    key_estimate = memory.estimate(
        previous,
        ActionCandidate(action_id=5, source="test"),
    )
    threat_estimate = memory.estimate(
        previous,
        ActionCandidate(action_id=4, source="test"),
    )

    assert key_estimate.role == "key"
    assert key_estimate.value > 0.18
    assert key_estimate.confidence > 0.50
    assert threat_estimate.role == "threat"
    assert threat_estimate.risk > 0.65
    assert threat_estimate.confidence > 0.50


def test_policy_avoids_learned_viability_loss():
    before = [[0 for _ in range(24)] for _ in range(24)]
    query = [[0 for _ in range(24)] for _ in range(24)]
    for y in range(7, 10):
        for x in range(7, 10):
            before[y][x] = 5
            query[y + 3][x + 3] = 5

    policy = ObserverArcPolicy("stub")
    previous = policy._observe(StubFrame([before], available_actions=[3, 4]))
    loss_state = policy._observe(
        StubFrame([before], state="GAME_OVER", available_actions=[0])
    )
    for _ in range(3):
        policy.viability_model.record(
            previous,
            ActionCandidate(action_id=4, source="test"),
            loss_state,
            reward=-0.6,
            changed=False,
            game_over=True,
        )

    query_frame = StubFrame([query], available_actions=[3, 4])
    query_world = policy._observe(query_frame)
    risky = policy._score_simple(4, query_world)
    action = policy.choose([], query_frame)

    assert action.action_id == 3
    assert "viability=" in risky.explanation
    assert any(
        "viability risk" in hypothesis
        for hypothesis in policy.viability_model.hypotheses(
            query_world,
            risky,
            policy.model.max_actions,
        )
    )


def test_policy_uses_latent_roles_to_avoid_threat_action():
    before = [[0 for _ in range(24)] for _ in range(24)]
    query = [[0 for _ in range(24)] for _ in range(24)]
    for y in range(7, 10):
        for x in range(7, 10):
            before[y][x] = 5
            query[y + 3][x + 3] = 5

    policy = ObserverArcPolicy("stub")
    previous = policy._observe(StubFrame([before], available_actions=[3, 4]))
    loss_state = policy._observe(
        StubFrame([before], state="GAME_OVER", available_actions=[0])
    )
    for _ in range(3):
        policy.latent_role_memory.record(
            previous,
            ActionCandidate(action_id=4, source="test"),
            loss_state,
            reward=-0.6,
            changed=False,
            game_over=True,
        )

    query_frame = StubFrame([query], available_actions=[3, 4])
    query_world = policy._observe(query_frame)
    risky = policy._score_simple(4, query_world)
    action = policy.choose([], query_frame)

    assert action.action_id == 3
    assert "latent_role=threat" in risky.explanation
    assert any(
        "latent threat role" in hypothesis
        for hypothesis in policy.latent_role_memory.hypotheses(query_world, risky)
    )


def test_policy_resets_when_not_played():
    policy = ObserverArcPolicy("stub")
    action = policy.choose([], StubFrame([], state="NOT_PLAYED", available_actions=[0]))

    assert action.action_id == 0
    assert "Reset" in action.explanation


def test_policy_records_arc_audit_trace():
    grid = [[0 for _ in range(8)] for _ in range(8)]
    grid[2][2] = 4
    policy = ObserverArcPolicy("stub")

    action = policy.choose([], StubFrame([grid], available_actions=[1, 2]))
    audit = policy.audit_trail.as_dict()

    assert audit["summary"]["steps"] == 1
    assert audit["summary"]["recent_actions"] == [action.key()]
    assert audit["steps"][0]["frame_hash"]
    assert audit["steps"][0]["available_actions"] == [1, 2]


def test_policy_prefers_salient_click_when_click_available():
    grid = [[0 for _ in range(64)] for _ in range(64)]
    for y in range(40, 45):
        for x in range(6, 12):
            grid[y][x] = 9
    policy = ObserverArcPolicy("stub")

    action = policy.choose([], StubFrame([grid], available_actions=[6]))

    assert action.action_id == 6
    assert 0 <= action.x <= 63
    assert 0 <= action.y <= 63


def test_transition_memory_penalizes_repeated_noop():
    grid = [[0 for _ in range(64)] for _ in range(64)]
    grid[12][12] = 4
    policy = ObserverArcPolicy("stub")
    frame = StubFrame([grid], available_actions=[1, 2])

    first = policy.choose([], frame)
    policy.choose([], frame)
    second = policy.choose([], frame)

    assert first.key() != second.key() or second.repetition_penalty > 0


def test_policy_uses_generalized_action_effects_on_new_frame():
    train_grid = [[0 for _ in range(64)] for _ in range(64)]
    train_next = [[0 for _ in range(64)] for _ in range(64)]
    query_grid = [[0 for _ in range(64)] for _ in range(64)]
    for y in range(10, 14):
        for x in range(10, 14):
            train_grid[y][x] = 3
            train_next[y][x + 1] = 3
    for y in range(30, 34):
        for x in range(30, 34):
            query_grid[y][x] = 4

    policy = ObserverArcPolicy("stub")
    policy.general_memory.record(
        summarize_frame([train_grid]),
        ActionCandidate(action_id=2, source="test"),
        summarize_frame([train_next]),
        reward=0.8,
        changed=True,
        game_over=False,
    )

    action = policy.choose([], StubFrame([query_grid], available_actions=[1, 2]))

    assert action.action_id == 2
    assert "generalized" in action.explanation


def test_policy_prefers_learned_observable_information_gain():
    train_before = [[0 for _ in range(12)] for _ in range(12)]
    train_after = [[0 for _ in range(12)] for _ in range(12)]
    query = [[0 for _ in range(12)] for _ in range(12)]
    for y in range(1, 3):
        for x in range(1, 3):
            train_before[y][x] = 4
            train_after[y][x] = 4
            query[y + 1][x + 1] = 5
    for y in range(6, 10):
        for x in range(6, 10):
            train_after[y][x] = 7

    policy = ObserverArcPolicy("stub")
    policy.action_information_memory.record(
        summarize_frame([train_before]),
        ActionCandidate(action_id=3, source="test"),
        summarize_frame([train_after]),
        game_over=False,
    )

    action = policy.choose([], StubFrame([query], available_actions=[1, 3]))

    assert action.action_id == 3
    assert "information=" in action.explanation
    assert any(
        "observable state" in hypothesis
        for hypothesis in policy.state.self_state.current_hypotheses
    )


def test_policy_uses_transition_motifs_and_goal_affinity():
    before = [[0 for _ in range(32)] for _ in range(32)]
    after = [[0 for _ in range(32)] for _ in range(32)]
    query = [[0 for _ in range(32)] for _ in range(32)]
    for y in range(8, 12):
        for x in range(4, 8):
            before[y][x] = 4
        for x in range(12, 16):
            after[y][x] = 4
        for x in range(18, 22):
            query[y][x] = 5
    for y in range(20, 24):
        for x in range(20, 24):
            after[y][x] = 7

    policy = ObserverArcPolicy("stub")
    policy.transition_motif_memory.record(
        summarize_frame([before]),
        ActionCandidate(action_id=5, source="test"),
        summarize_frame([after]),
        reward=0.6,
        game_over=False,
    )
    policy.goal_discovery.record(
        summarize_frame([before]),
        ActionCandidate(action_id=5, source="test"),
        summarize_frame([after]),
        reward=0.6,
        game_over=False,
        previous_actions=[1, 5],
        current_actions=[1, 5, 6],
    )

    action = policy.choose([], StubFrame([query], available_actions=[1, 5]))

    assert action.action_id == 5
    assert "motif=" in action.explanation
    assert "goal=" in action.explanation
    assert "comparator=" in action.explanation
    assert any(
        "transition motif" in hypothesis or "goal_affinity" in hypothesis
        for hypothesis in policy.state.self_state.current_hypotheses
    )


def test_policy_uses_state_sequence_motion_and_boundary_constraints():
    before = [[0 for _ in range(64)] for _ in range(64)]
    after = [[0 for _ in range(64)] for _ in range(64)]
    query_open = [[0 for _ in range(64)] for _ in range(64)]
    query_boundary = [[0 for _ in range(64)] for _ in range(64)]
    for y in range(20, 24):
        for x in range(20, 24):
            before[y][x] = 5
        for x in range(25, 29):
            after[y][x] = 5
        for x in range(30, 34):
            query_open[y][x] = 5
        for x in range(60, 64):
            query_boundary[y][x] = 5

    policy = ObserverArcPolicy("stub")
    policy.trajectory_memory.record(
        summarize_frame([before]),
        ActionCandidate(action_id=4, source="test"),
        summarize_frame([after]),
        reward=0.1,
    )

    open_action = policy.choose([], StubFrame([query_open], available_actions=[3, 4]))

    assert open_action.action_id == 4
    assert "trajectory" in open_action.explanation

    boundary_policy = ObserverArcPolicy("stub")
    boundary_policy.trajectory_memory.record(
        summarize_frame([before]),
        ActionCandidate(action_id=4, source="test"),
        summarize_frame([after]),
        reward=0.1,
    )
    boundary_action = boundary_policy.choose(
        [], StubFrame([query_boundary], available_actions=[3, 4])
    )

    assert boundary_action.action_id == 3
    assert any(
        "boundary" in hypothesis
        for hypothesis in boundary_policy.state.self_state.current_hypotheses
    )


def test_policy_recognizes_visible_object_boundary():
    before = [[0 for _ in range(64)] for _ in range(64)]
    after = [[0 for _ in range(64)] for _ in range(64)]
    query_open = [[0 for _ in range(64)] for _ in range(64)]
    query_wall = [[0 for _ in range(64)] for _ in range(64)]
    for y in range(20, 24):
        for x in range(20, 24):
            before[y][x] = 5
        for x in range(25, 29):
            after[y][x] = 5
        for x in range(30, 34):
            query_open[y][x] = 5
            query_wall[y][x] = 5
        for x in range(36, 40):
            query_wall[y][x] = 9

    policy = ObserverArcPolicy("stub")
    policy.trajectory_memory.record(
        summarize_frame([before]),
        ActionCandidate(action_id=4, source="test"),
        summarize_frame([after]),
        reward=0.1,
    )
    policy.boundary_memory.record(
        summarize_frame([before]),
        ActionCandidate(action_id=4, source="test"),
        summarize_frame([after]),
        reward=0.1,
    )

    open_action = policy.choose([], StubFrame([query_open], available_actions=[3, 4]))

    assert open_action.action_id == 4

    wall_policy = ObserverArcPolicy("stub")
    wall_policy.trajectory_memory.record(
        summarize_frame([before]),
        ActionCandidate(action_id=4, source="test"),
        summarize_frame([after]),
        reward=0.1,
    )
    wall_policy.boundary_memory.record(
        summarize_frame([before]),
        ActionCandidate(action_id=4, source="test"),
        summarize_frame([after]),
        reward=0.1,
    )

    wall_action = wall_policy.choose([], StubFrame([query_wall], available_actions=[3, 4]))

    assert wall_action.action_id == 3
    assert any(
        "visible east boundary" in hypothesis
        for hypothesis in wall_policy.state.self_state.current_hypotheses
    )


def test_policy_infers_object_role_from_actor_target_sequence():
    before = [[0 for _ in range(64)] for _ in range(64)]
    after = [[0 for _ in range(64)] for _ in range(64)]
    query = [[0 for _ in range(64)] for _ in range(64)]
    for y in range(22, 26):
        for x in range(10, 14):
            before[y][x] = 5
        for x in range(16, 20):
            after[y][x] = 5
        for x in range(20, 24):
            query[y][x] = 5
        for x in range(42, 46):
            before[y][x] = 9
            after[y][x] = 9
        for x in range(50, 54):
            query[y][x] = 9

    policy = ObserverArcPolicy("stub")
    policy.object_role_memory.record(
        summarize_frame([before]),
        ActionCandidate(action_id=4, source="test"),
        summarize_frame([after]),
        reward=0.1,
    )

    action = policy.choose([], StubFrame([query], available_actions=[2, 4]))

    assert action.action_id == 4
    assert "role=" in action.explanation
    assert any(
        "target" in hypothesis
        for hypothesis in policy.state.self_state.current_hypotheses
    )


def test_policy_uses_delayed_sequence_credit_on_similar_state():
    opener = [[0 for _ in range(64)] for _ in range(64)]
    middle = [[0 for _ in range(64)] for _ in range(64)]
    query = [[0 for _ in range(64)] for _ in range(64)]
    for y in range(12, 16):
        for x in range(12, 16):
            opener[y][x] = 4
        for x in range(20, 24):
            middle[y][x] = 4
        for x in range(40, 44):
            query[y][x] = 8

    policy = ObserverArcPolicy("stub")
    policy.sequence_credit_memory.record(
        [
            (summarize_frame([opener]), ActionCandidate(action_id=5, source="test")),
            (summarize_frame([middle]), ActionCandidate(action_id=4, source="test")),
        ],
        terminal_reward=1.0,
    )

    action = policy.choose([], StubFrame([query], available_actions=[2, 5]))

    assert action.action_id == 5
    assert "sequence=" in action.explanation
    assert any(
        "delayed sequence credit" in hypothesis
        for hypothesis in policy.state.self_state.current_hypotheses
    )


def test_policy_continues_learned_action_fragment():
    opener = [[0 for _ in range(64)] for _ in range(64)]
    middle = [[0 for _ in range(64)] for _ in range(64)]
    query = [[0 for _ in range(64)] for _ in range(64)]
    for y in range(18, 22):
        for x in range(12, 16):
            opener[y][x] = 6
        for x in range(28, 32):
            middle[y][x] = 6
        for x in range(44, 48):
            query[y][x] = 7

    policy = ObserverArcPolicy("stub")
    policy.sequence_fragment_memory.record(
        [
            (summarize_frame([opener]), ActionCandidate(action_id=5, source="test")),
            (summarize_frame([middle]), ActionCandidate(action_id=4, source="test")),
        ],
        terminal_reward=1.0,
    )
    policy.recent_action_schemas.append("A5")

    action = policy.choose([], StubFrame([query], available_actions=[2, 4]))

    assert action.action_id == 4
    assert "fragment=" in action.explanation
    assert any(
        "learned action fragment" in hypothesis
        for hypothesis in policy.state.self_state.current_hypotheses
    )


class TinySearchGame:
    def __init__(self):
        self._score = 0
        self._state = GameState.NOT_FINISHED
        self._current_level_index = 0
        self.position = 0

    def _get_valid_actions(self):
        return [
            ActionInput(id=GameAction.ACTION1),
            ActionInput(id=GameAction.ACTION2),
        ]

    def perform_action(self, action_input, raw=False):
        if action_input.id == GameAction.ACTION1:
            self.position = 0
        elif action_input.id == GameAction.ACTION2:
            self.position += 1
            if self.position >= 2:
                self._score = 1
                self._state = GameState.WIN
        frame = FrameDataRaw()
        frame.state = self._state
        frame.levels_completed = self._score
        frame.win_levels = 1
        arr = np.zeros((4, 4), dtype=np.int8)
        arr[0, min(self.position, 3)] = 2
        frame.frame = [arr]
        frame.available_actions = [1, 2]
        return frame


def test_local_simulator_planner_finds_depth_two_win():
    game = TinySearchGame()
    latest = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    planner = LocalSimulatorPlanner(PlannerConfig(max_depth=3, beam_width=4, branch_limit=4))

    action = planner.plan(game, latest)

    assert action is not None
    assert action.action_id == 2
    assert "depth 2" in action.explanation


class NoChangeCounterGame:
    def __init__(self):
        self._score = 0
        self._state = GameState.NOT_FINISHED
        self._current_level_index = 0
        self._action_count = 0
        self._next_level = False
        self._full_reset = False
        self._action = ActionInput(id=GameAction.RESET)
        self._available_actions = [6]

    def _get_valid_actions(self):
        return [ActionInput(id=GameAction.ACTION6, data={"x": 4, "y": 4})]

    def perform_action(self, action_input, raw=False):
        self._action = action_input
        self._action_count += 1
        if self._action_count >= 3:
            self._score = 1
        frame = FrameDataRaw()
        frame.state = self._state
        frame.levels_completed = self._score
        frame.win_levels = 1
        frame.frame = [np.zeros((4, 4), dtype=np.int8)]
        frame.available_actions = [6]
        return frame


def test_local_simulator_planner_keeps_no_change_repeats():
    game = NoChangeCounterGame()
    latest = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    game._action_count = 0
    planner = LocalSimulatorPlanner(PlannerConfig(max_depth=4, beam_width=2, branch_limit=2))

    action = planner.plan(game, latest)

    assert action is not None
    assert action.action_id == 6
    assert "depth 3" in action.explanation


class PartialProgressGame:
    def __init__(self):
        self._score = 0
        self._state = GameState.NOT_FINISHED
        self._current_level_index = 0
        self.position = 0

    def _get_valid_actions(self):
        return [
            ActionInput(id=GameAction.ACTION1),
            ActionInput(id=GameAction.ACTION2),
        ]

    def perform_action(self, action_input, raw=False):
        if action_input.id == GameAction.ACTION2:
            self.position += 1
        frame = FrameDataRaw()
        frame.state = self._state
        frame.levels_completed = self._score
        frame.win_levels = 1
        arr = np.zeros((4, 4), dtype=np.int8)
        arr[0, 0] = 2
        if self.position:
            arr[0, 1] = 3
        frame.frame = [arr]
        frame.available_actions = [1, 2]
        return frame


def test_local_simulator_planner_returns_positive_partial_progress():
    game = PartialProgressGame()
    latest = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    planner = LocalSimulatorPlanner(
        PlannerConfig(max_depth=2, beam_width=2, branch_limit=2, return_best_nonterminal=True)
    )

    action = planner.plan(game, latest)

    assert action is not None
    assert action.action_id == 2
    assert "best nonterminal" in action.explanation


class SecondRepeatGame:
    def __init__(self):
        self._score = 0
        self._state = GameState.NOT_FINISHED
        self._current_level_index = 0
        self.first_count = 0
        self.second_count = 0

    def _get_valid_actions(self):
        return [
            ActionInput(id=GameAction.ACTION6, data={"x": 4, "y": 4}),
            ActionInput(id=GameAction.ACTION6, data={"x": 12, "y": 4}),
        ]

    def perform_action(self, action_input, raw=False):
        if action_input.data.get("x") == 4:
            self.first_count += 1
        elif action_input.data.get("x") == 12:
            self.second_count += 1
            if self.second_count >= 3:
                self._score = 1
        frame = FrameDataRaw()
        frame.state = self._state
        frame.levels_completed = self._score
        frame.win_levels = 1
        frame.frame = [np.zeros((4, 4), dtype=np.int8)]
        frame.available_actions = [6]
        return frame


def test_local_simulator_repeat_probe_round_robins_candidates():
    game = SecondRepeatGame()
    latest = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    game.first_count = 0
    game.second_count = 0
    planner = LocalSimulatorPlanner(PlannerConfig(max_depth=8, beam_width=2, branch_limit=2))

    action = planner.plan(game, latest)

    assert action is not None
    assert action.key() == "A6:3:1"
    assert "depth 3" in action.explanation


class SalientClickTargetGame:
    def __init__(self):
        self._score = 0
        self._state = GameState.NOT_FINISHED
        self._current_level_index = 0
        self.target = (4, 56)

    def _get_valid_actions(self):
        return [
            ActionInput(id=GameAction.ACTION6, data={"x": x, "y": y})
            for y in range(0, 64, 4)
            for x in range(0, 64, 4)
        ]

    def perform_action(self, action_input, raw=False):
        if action_input.id == GameAction.ACTION6:
            if (action_input.data.get("x"), action_input.data.get("y")) == self.target:
                self._score = 1
        frame = FrameDataRaw()
        frame.state = self._state
        frame.levels_completed = self._score
        frame.win_levels = 1
        arr = np.zeros((64, 64), dtype=np.int8)
        arr[55:58, 3:6] = 7
        frame.frame = [arr]
        frame.available_actions = [6]
        return frame


def test_local_simulator_ranks_valid_clicks_by_frame_salience():
    game = SalientClickTargetGame()
    latest = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    planner = LocalSimulatorPlanner(
        PlannerConfig(max_depth=2, beam_width=2, branch_limit=6)
    )

    action = planner.plan(game, latest)

    assert action is not None
    assert action.action_id == 6
    assert (action.x, action.y) == game.target


class OrderedKeyboardSequenceGame:
    def __init__(self):
        self._score = 0
        self._state = GameState.NOT_FINISHED
        self._current_level_index = 0
        self.progress = 0
        self.target = [
            GameAction.ACTION1,
            GameAction.ACTION4,
            GameAction.ACTION2,
            GameAction.ACTION3,
        ]

    def _get_valid_actions(self):
        return [
            ActionInput(id=GameAction.ACTION1),
            ActionInput(id=GameAction.ACTION2),
            ActionInput(id=GameAction.ACTION3),
            ActionInput(id=GameAction.ACTION4),
        ]

    def perform_action(self, action_input, raw=False):
        expected = self.target[self.progress]
        if action_input.id == expected:
            self.progress += 1
        else:
            self.progress = 0
        if self.progress >= len(self.target):
            self._score = 1
            self._state = GameState.WIN
        frame = FrameDataRaw()
        frame.state = self._state
        frame.levels_completed = self._score
        frame.win_levels = 1
        arr = np.zeros((4, 4), dtype=np.int8)
        arr[0, min(self.progress, 3)] = 2
        frame.frame = [arr]
        frame.available_actions = [1, 2, 3, 4]
        return frame


def test_local_simulator_sequence_probe_finds_ordered_keyboard_plan():
    game = OrderedKeyboardSequenceGame()
    latest = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    game.progress = 0
    planner = LocalSimulatorPlanner(
        PlannerConfig(max_depth=8, beam_width=4, branch_limit=4, max_nodes=1024)
    )

    first = planner.plan(game, latest)

    assert first is not None
    assert first.action_id == 1
    assert "sequence probe" in first.explanation
    assert [candidate.action_id for candidate in planner.queued_plan] == [4, 2, 3]
    assert len(planner.queued_preconditions) == 3


def test_local_simulator_continues_only_validated_queued_plan():
    game = OrderedKeyboardSequenceGame()
    latest = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    game.progress = 0
    planner = LocalSimulatorPlanner(
        PlannerConfig(max_depth=8, beam_width=4, branch_limit=4, max_nodes=1024)
    )

    first = planner.plan(game, latest)
    actual = game.perform_action(ActionInput(id=GameAction.ACTION1), raw=True)
    continued = planner.plan(game, actual)

    assert first is not None
    assert first.action_id == 1
    assert continued is not None
    assert continued.action_id == 4
    assert "validated queued" in continued.explanation


def test_local_simulator_discards_stale_queued_plan():
    game = OrderedKeyboardSequenceGame()
    latest = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    game.progress = 0
    planner = LocalSimulatorPlanner(
        PlannerConfig(max_depth=8, beam_width=4, branch_limit=4, max_nodes=1024)
    )

    first = planner.plan(game, latest)
    stale = FrameDataRaw()
    stale.state = GameState.NOT_FINISHED
    stale.levels_completed = 0
    stale.win_levels = 1
    stale_frame = np.zeros((4, 4), dtype=np.int8)
    stale_frame[3, 3] = 9
    stale.frame = [stale_frame]
    stale.available_actions = [1, 2, 3, 4]
    stale_continuation = planner._queued_action(stale)

    assert first is not None
    assert planner.queued_plan == []
    assert planner.queued_preconditions == []
    assert stale_continuation is None


class BroadClickScanGame:
    def __init__(self):
        self._score = 0
        self._state = GameState.NOT_FINISHED
        self._current_level_index = 0
        self.target = (60, 60)

    def _get_valid_actions(self):
        return [
            ActionInput(id=GameAction.ACTION6, data={"x": x, "y": y})
            for y in range(0, 64, 4)
            for x in range(0, 64, 4)
        ]

    def perform_action(self, action_input, raw=False):
        if action_input.id == GameAction.ACTION6:
            if (action_input.data.get("x"), action_input.data.get("y")) == self.target:
                self._score = 1
                self._state = GameState.WIN
        frame = FrameDataRaw()
        frame.state = self._state
        frame.levels_completed = self._score
        frame.win_levels = 1
        arr = np.zeros((64, 64), dtype=np.int8)
        frame.frame = [arr]
        frame.available_actions = [6]
        return frame


def test_local_simulator_broad_click_scan_checks_beyond_branch_limit():
    game = BroadClickScanGame()
    latest = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    planner = LocalSimulatorPlanner(
        PlannerConfig(
            max_depth=2,
            beam_width=1,
            branch_limit=4,
            max_nodes=512,
            max_seconds=1.0,
        )
    )

    action = planner.plan(game, latest)

    assert action is not None
    assert action.action_id == 6
    assert (action.x, action.y) == game.target
    assert "broad click scan" in action.explanation


class FrameAvailableClickGame:
    def __init__(self):
        self._score = 0
        self._state = GameState.NOT_FINISHED
        self._current_level_index = 0
        self.target = (20, 20)

    def _get_valid_actions(self):
        return [ActionInput(id=GameAction.ACTION1)]

    def perform_action(self, action_input, raw=False):
        if action_input.id == GameAction.ACTION6:
            if (action_input.data.get("x"), action_input.data.get("y")) == self.target:
                self._score = 1
                self._state = GameState.WIN
        frame = FrameDataRaw()
        frame.state = self._state
        frame.levels_completed = self._score
        frame.win_levels = 1
        arr = np.zeros((64, 64), dtype=np.int8)
        arr[19:22, 19:22] = 7
        frame.frame = [arr]
        frame.available_actions = [1, 6]
        return frame


def test_local_simulator_synthesizes_clicks_from_frame_available_actions():
    game = FrameAvailableClickGame()
    latest = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    planner = LocalSimulatorPlanner(
        PlannerConfig(max_depth=2, beam_width=2, branch_limit=4, max_seconds=1.0)
    )

    action = planner.plan(game, latest)

    assert action is not None
    assert action.action_id == 6
    assert (action.x, action.y) == game.target
    assert "broad click scan" in action.explanation


class InformationGainGame:
    def __init__(self):
        self._score = 0
        self._state = GameState.NOT_FINISHED
        self._current_level_index = 0
        self.revealed = False
        self.small_change = False

    def _get_valid_actions(self):
        return [
            ActionInput(id=GameAction.ACTION1),
            ActionInput(id=GameAction.ACTION2),
            ActionInput(id=GameAction.ACTION3),
            ActionInput(id=GameAction.ACTION4),
        ]

    def perform_action(self, action_input, raw=False):
        if action_input.id == GameAction.ACTION2:
            self.small_change = True
        elif action_input.id == GameAction.ACTION3:
            self.revealed = True
        elif action_input.id == GameAction.ACTION4:
            self._state = GameState.GAME_OVER
        frame = FrameDataRaw()
        frame.state = self._state
        frame.levels_completed = self._score
        frame.win_levels = 1
        arr = np.zeros((12, 12), dtype=np.int8)
        arr[0, 0] = 2
        if self.small_change:
            arr[1, 1] = 3
        if self.revealed:
            arr[6:10, 6:10] = 7
        frame.frame = [arr]
        frame.available_actions = [1, 2, 3, 4]
        return frame


def test_local_simulator_information_probe_prefers_revealing_action():
    game = InformationGainGame()
    latest = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    game.small_change = False
    game.revealed = False
    planner = LocalSimulatorPlanner(
        PlannerConfig(max_depth=2, beam_width=2, branch_limit=4, max_seconds=1.0)
    )

    action = planner.plan(game, latest)

    assert action is not None
    assert action.action_id == 3
    assert "information probe" in action.explanation


class InformationSequenceGame:
    def __init__(self):
        self._score = 0
        self._state = GameState.NOT_FINISHED
        self._current_level_index = 0
        self.primed = False
        self.decoy = False
        self.revealed = False

    def _get_valid_actions(self):
        return [
            ActionInput(id=GameAction.ACTION1),
            ActionInput(id=GameAction.ACTION2),
            ActionInput(id=GameAction.ACTION3),
            ActionInput(id=GameAction.ACTION4),
        ]

    def perform_action(self, action_input, raw=False):
        if action_input.id == GameAction.ACTION2:
            self.primed = True
        elif action_input.id == GameAction.ACTION3:
            if self.primed and not self.decoy:
                self.revealed = True
            else:
                self.decoy = True
        elif action_input.id == GameAction.ACTION4:
            self._state = GameState.GAME_OVER
        frame = FrameDataRaw()
        frame.state = self._state
        frame.levels_completed = self._score
        frame.win_levels = 1
        arr = np.zeros((12, 12), dtype=np.int8)
        arr[0, 0] = 2
        if self.primed:
            arr[1, 1] = 3
        if self.decoy:
            arr[3, 3] = 5
        if self.revealed:
            arr[6:10, 6:10] = 7
        frame.frame = [arr]
        frame.available_actions = [1, 2, 3, 4]
        return frame


def test_local_simulator_information_sequence_finds_revealing_prefix():
    game = InformationSequenceGame()
    latest = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    game.primed = False
    game.decoy = False
    game.revealed = False
    planner = LocalSimulatorPlanner(
        PlannerConfig(max_depth=3, beam_width=4, branch_limit=4, max_seconds=1.0)
    )

    action = planner.plan(game, latest)

    assert action is not None
    assert action.action_id == 2
    assert "information sequence" in action.explanation
    assert [candidate.action_id for candidate in planner.queued_plan] == [3]


class InvisibleOrderedSequenceGame:
    def __init__(self):
        self._score = 0
        self._state = GameState.NOT_FINISHED
        self._current_level_index = 0
        self.progress = 0
        self.target = [
            GameAction.ACTION3,
            GameAction.ACTION1,
            GameAction.ACTION4,
        ]

    def _get_valid_actions(self):
        return [
            ActionInput(id=GameAction.ACTION1),
            ActionInput(id=GameAction.ACTION2),
            ActionInput(id=GameAction.ACTION3),
            ActionInput(id=GameAction.ACTION4),
        ]

    def perform_action(self, action_input, raw=False):
        expected = self.target[self.progress]
        if action_input.id == expected:
            self.progress += 1
        else:
            self.progress = 0
        if self.progress >= len(self.target):
            self._score = 1
            self._state = GameState.WIN
        frame = FrameDataRaw()
        frame.state = self._state
        frame.levels_completed = self._score
        frame.win_levels = 1
        frame.frame = [np.zeros((4, 4), dtype=np.int8)]
        frame.available_actions = [1, 2, 3, 4]
        return frame


def test_local_simulator_hidden_sequence_probe_keeps_no_change_prefixes():
    game = InvisibleOrderedSequenceGame()
    latest = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    game.progress = 0
    planner = LocalSimulatorPlanner(
        PlannerConfig(max_depth=5, beam_width=4, branch_limit=4, max_nodes=1024, max_seconds=1.0)
    )

    action = planner.plan(game, latest)

    assert action is not None
    assert action.action_id == 3
    assert "hidden sequence" in action.explanation
    assert [candidate.action_id for candidate in planner.queued_plan] == [1, 4]
