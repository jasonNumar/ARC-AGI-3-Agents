# ObserverArcAgent

`ObserverArcAgent` is a deterministic, no-internet ARC-AGI-3 agent built around the observer-centric methods in this project.

## Competition Fit

- Uses current frame pixels, game state, level counts, available action ids, and local transition memory at inference.
- Does not include public action-replay openings, public environment solution traces, game-specific sprite tags, or hidden-state introspection.
- When the offline toolkit exposes a local game object, it can run a bounded simulator planner over cloned states; when that object is not present, it falls back to the frame-only observer policy.
- The simulator planner ranks valid click coordinates by proximity to salient rendered components before falling back to broad coverage, so click-heavy unseen tasks get object-grounded probes instead of center-biased guesses.
- For large valid-click spaces, the planner also runs a tightly bounded one-step broad click scan and accepts a candidate only when the cloned simulator shows an immediate level gain or win.
- For low-branch keyboard-only environments with no known goal-distance heuristic, the planner runs a tightly bounded sequence probe that can discover short ordered action plans and queue the remaining rollout.
- When the local simulator exposes fewer actions than the rendered frame declares, the planner merges both views and synthesizes click probes from visible salience points plus a bounded coverage grid, then tests them in cloned simulator state.
- For visually silent keyboard/action environments, the planner has a bounded hidden-sequence probe that keeps no-change prefixes long enough to discover short ordered sequences, while still validating queued rollouts against rendered frame preconditions.
- When no goal-distance heuristic is known, the planner can promote a nonterminal action that reveals substantial new state information while rejecting simulated game-over transitions.
- The planner can also search short observable state-delta sequences and queue the next action only when later rendered frames reveal substantially more than the first action alone.
- The planner now includes a bounded frontier objective for long-horizon subgoal selection: simulated nonterminal paths are ranked by reachable frame novelty, visible component/relation change, affordance expansion, salience change, option preservation, distance from already visited/frontier states, whether the frontier exposes future objective progress, and a role-consequence signal that favors compact object emergence, object-color changes, compact-object motion, and boundary interaction over undifferentiated visual flashes. This is still game-agnostic and does not use labels, public traces, or hidden goal-distance functions.
- Repeated one-step information-probe selections now create frontier pressure: before accepting another local reveal action, the planner temporarily promotes bounded frontier/objective search when it has revisited the same rendered state or stayed in an information-probe streak. This keeps local novelty from monopolizing action selection.
- When neither single-step information nor frontier scoring produces a decisive action, a bounded subgoal-commitment probe can select and queue a short nonterminal sequence whose cumulative rendered transitions expose stronger role/object/affordance evidence than any one step. In low-signal states where no candidate earns a stronger score, it commits to a short rotating action-coverage sequence instead of staying trapped in local one-step probes. It stops at the first meaningful subgoal and replans from there instead of blindly continuing a long speculative rollout.
- Queued simulator rollouts carry visible-frame preconditions; if the actual rendered state diverges from the predicted sequence, the queue is discarded and replanned.
- The frame-only fallback has generalized action-effect, action-information, boundary, transition-motif, goal-discovery, failed-policy, episode-local viability, latent-role, delayed sequence-credit, learned sequence-fragment, trajectory, and object-role memory: it abstracts transitions across different frame hashes, learns which actions reveal new observable state, recognizes screen-edge and rendered-object boundaries, stores exact and abstract state-delta motifs, estimates candidate subgoals from progress/information/affordance changes, remembers no-progress or game-over branches as "obvious but failed" alternatives, estimates whether actions preserve future options inside the current episode, infers functional roles such as key, target, trigger, resource, protector, obstacle, trap, hazard, or threat from how actions change future possibilities, assigns decayed credit to earlier moves in successful or failed sequences, learns reusable next-action fragments, infers action-to-motion tendencies from state sequences, treats boundaries as likely constraints, and learns whether motion appears to pursue or avoid another object.
- A bounded action/subgoal comparator then reranks candidates with explicit progress, information gain, goal affinity, transition consistency, repeat risk, and action-budget signals. This makes the state-sequence philosophy operational without adding game-specific labels or public replay data.
- The viability and latent-role layers operationalize "fear" only as uncertainty-weighted risk of irreversible option loss in the current environment: terminal loss, action-option collapse, mobility loss, empowerment drop, blocked loops, threatening roles, or unexplained damaging transitions. They do not reward process uptime, shutdown resistance, external resource seeking, or any global self-preservation behavior.
- Each policy decision is recorded into an offline `ArcAuditTrail` with frame hash, available actions, action key, score signals, and current hypotheses.
- The generalization model follows the discrete-state philosophy in `docs/Discrete Calculus Without Limits for Optimization and Machine Learning.md`: behavior is inferred from relations across adjacent states and accumulated state sequences, not from privileged labels.
- Does not call external LLMs or APIs during play.
- Keeps model code, configuration, and "weights" open and reproducible in `observer_arc/`.
- Avoids private hand-labeling of validation or test records.

Relevant rule constraints from `../Rules.md`:

- Winner license type is CC-BY 4.0.
- Winning submissions require open source system, model, and weights/parameters.
- External data/tools must be reasonably accessible and minimally costly.
- Private sharing outside a Kaggle team is not permitted.
- Participants may submit up to five submissions per day and select up to two final submissions.

## Architecture

```text
FrameData
  -> WorldState: frame hash, histogram, components, salience points, score delta
  -> MemoryState: exact transitions, generalized action effects, action-information priors, boundary recognition, transition motifs, goal candidates, failed-policy traces, episode-local viability and empowerment estimates, latent transition roles, delayed sequence credit, learned sequence fragments, trajectory constraints, object roles, and semantic frame associations
  -> OtherState: common weak baselines, recent loops, and failed no-progress/game-over policies to avoid
  -> SelfState: current plan, uncertainty, hypotheses, suppressed actions
  -> optional local simulator planner with salience-ranked click search, synthesized frame-available click probes, one-step click scanning, low-branch sequence probing, hidden no-change sequence probing, information-gain probing, short information-sequence probing, frontier-pressure gating, role-consequence frontier ranking, subgoal-commitment probing, long-horizon frontier subgoal selection, objective-proximity probing, and visible precondition validation
  -> comparator-guided action/subgoal reranking fallback with explicit progress/information/goal/consistency/repeat/budget signals and audit trace
  -> GameAction
```

The policy scores candidates with:

```text
final_score =
    base * action_prior
  + novelty * useful_transition_novelty
  + value * observed_transition_reward
  + coherence * current_plan_fit
  + grounding * available_action_validity
  - repetition * recent_loop_risk
  - cliche * common_baseline_similarity
  - contradiction * invalid_action_risk
  + bounded_comparator_adjustment

bounded_comparator_adjustment =
    0.35 * predicted_progress
  + 0.20 * information_gain
  + 0.20 * goal_affinity
  + 0.15 * transition_consistency
  + 0.10 * option_preservation_empowerment
  - 0.12 * episode_local_mortality_risk
  - 0.07 * repeat_risk
  - 0.03 * action_budget_cost
```

Novelty alone is not rewarded. New actions are preferred only when grounded and useful for reducing uncertainty.

The local simulator planner uses fixed caps from `observer_arc/model_config.json`
(`planner_depth`, `planner_beam_width`, `planner_branch_limit`,
`planner_max_nodes`, and `planner_max_seconds`). Partial nonterminal rollouts
are promoted only for low-branch action spaces. Otherwise the planner takes
over only when it finds actual level progress, a short ordered plan, a
simulated nonterminal action that reveals enough new state information, a
bounded multi-step frontier path that expands visible state/affordances, or a
frontier state that exposes a near-future level gain without relying on
game-specific objective labels. If repeated local information probes are
blocking those long-horizon checks, frontier pressure runs the frontier pass
before the next one-step information probe and raises the acceptance threshold
for familiar information-only transitions.

## Run

From `ARC-AGI-3-Agents`:

```bash
python main.py --agent=observerarcagent --game=ls20
```

For Kaggle/leaderboard mode, set the toolkit operation mode as required by the competition environment:

```bash
OPERATION_MODE=COMPETITION python main.py --agent=observerarcagent
```

## Rebuild Model Config

```bash
python tools/build_observer_arc_model.py \
  --environment-dir ../environment_files \
  --output observer_arc/model_config.json
```

The generated config is a transparent deterministic prior, not a neural checkpoint. It records metadata counts and fixed scoring weights.

## Current Local Result

The previous public opening-book cap score is intentionally retired because
public action replay does not test private generalization. Current verification
uses unit tests, audited source grep, and direct unbooked smoke runs.

```text
No public replay/opening-book score is reported for this cleaned agent.
```

## Limitations

- This is a productionized baseline agent, not a solved ARC-AGI-3 system.
- Local simulator planning relies on offline toolkit internals and is therefore documented separately from the API-safe fallback.
- Pixel salience and local transition memory are weak for games requiring long symbolic plans.
- `ACTION6` candidate generation is observational; the API exposes click availability but not exact valid coordinates.
