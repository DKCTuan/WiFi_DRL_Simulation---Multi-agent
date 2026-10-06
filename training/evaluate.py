# training/evaluate.py
"""Đánh giá policy (epsilon=0, greedy) trên môi trường cố định/generalization."""
import numpy as np

import config
from env.wifi_env import WiFiEnv
from env.scenarios import scenario_bank
from training.utils import normalize_eval_mode
from training.objectives import validate_stationary_reward


def eval_score(eval_metrics, num_agents):
    """Select checkpoints by the objective optimized during training."""
    del num_agents  # Kept for compatibility with existing callers.
    if config.REWARD_OBJECTIVE_MODE == 'throughput_temporal_jfi':
        if eval_metrics.get('objective_score') is None:
            raise KeyError('Aligned evaluation must include objective_score')
        return float(eval_metrics['objective_score'])
    if "team_reward" not in eval_metrics:
        raise KeyError("Evaluation metrics must include mean team_reward")
    return float(eval_metrics["team_reward"])


def evaluate_policy(
    agents,
    episodes=5,
    steps=None,
    seed_base=None,
    use_water_filling=False,
    action_size=config.FULL_AI_ACTION_SIZE,
    eval_mode=None,
    num_stas=None,   # None → dùng AP_LOAD_PROFILE heterogeneous; int → đồng đều K
    cooperation_enabled=False,
    scenario_specs=None,
):
    validate_stationary_reward()
    eval_mode = normalize_eval_mode(eval_mode)
    if seed_base is None:
        if eval_mode == 'fixed_test':
            seed_base = config.FIXED_TEST_SEED_BASE
        else:
            seed_base = config.FIXED_VALIDATION_SEED_BASE
    steps = config.TRAIN_STEPS_PER_EPISODE if steps is None else steps
    if episodes <= 0 or steps <= 0:
        raise ValueError("Evaluation episodes and steps must be positive")
    if scenario_specs is not None:
        if eval_mode not in ('validation','test') or not scenario_specs:
            raise ValueError('Explicit scenarios require a nonempty validation/test split')
        allowed = scenario_bank(eval_mode)
        if any(spec not in allowed for spec in scenario_specs):
            raise ValueError('Scenario does not belong to the requested evaluation split')
    saved_epsilons = {agent_id: agent.epsilon for agent_id, agent in agents.items()}
    rng_state = np.random.get_state()

    for agent in agents.values():
        agent.epsilon = 0.0

    env = WiFiEnv(
        verbose=False,
        fixed_topology=eval_mode in ('fixed', 'fixed_validation', 'fixed_test'),
        fixed_seed=(config.TRAIN_SCENARIO_SEED
                    if eval_mode in ('fixed', 'fixed_validation', 'fixed_test')
                    else None),
        mobility_enabled=config.TRAIN_MOBILITY_ENABLED,
        use_water_filling=use_water_filling,
        action_size=action_size,
        num_stas_per_ap=num_stas,
        cooperation_enabled=cooperation_enabled,
    )
    total_throughput = 0.0
    env.max_steps = steps
    total_jfi = 0.0
    total_active_aps = 0.0
    total_energy_efficiency = 0.0
    total_coordination_links = 0.0
    total_coordination_requests = 0.0
    total_coordination_utility_delta = 0.0
    total_coordination_power_reduction = 0.0
    total_team_reward = 0.0
    total_steps = 0
    episode_temporal_jfi = []
    episode_objective_scores = []
    episode_access = []
    episode_max_idle = []
    evaluated_scenarios = []
    episode_user_status = []
    bank = scenario_specs if scenario_specs is not None else (scenario_bank(eval_mode) if eval_mode in ('validation','test') else None)

    try:
        for episode_idx in range(episodes):
            np.random.seed(seed_base + episode_idx)
            spec = bank[episode_idx % len(bank)] if bank is not None else None
            reset_seed = spec['rollout_seed'] if spec is not None else seed_base+episode_idx
            states_dict, _ = env.reset(seed=reset_seed, options={'scenario':spec})
            if spec is not None:
                evaluated_scenarios.append(spec)

            for _ in range(steps):
                actions_dict = {}
                for agent_id in env.agent_ids:
                    state_input = np.array(states_dict[agent_id], dtype=np.float32)
                    actions_dict[agent_id] = agents[agent_id].act(state_input)

                next_states_dict, _, terminated, truncated, info = env.step(actions_dict)
                total_throughput += info["throughput"]
                total_jfi += info["jfi"]
                total_active_aps += info["active_ap_count"]
                total_energy_efficiency += info["energy_efficiency"]
                total_coordination_links += info.get("coordination_link_count", 0)
                total_coordination_requests += info.get(
                    "coordination_requested_link_count", 0
                )
                total_coordination_utility_delta += info.get(
                    "coordination_utility_delta", 0.0
                )
                total_coordination_power_reduction += info.get(
                    "coordination_power_reduction_w", 0.0
                )
                total_team_reward += info["team_reward"]
                total_steps += 1
                states_dict = next_states_dict

                if terminated or truncated:
                    break
            episode_temporal_jfi.append(info['jfi_time_average'])
            if info.get('user_status'):
                episode_user_status.append(info['user_status'])
            if info.get('objective_score_so_far') is not None:
                episode_objective_scores.append(float(info['objective_score_so_far']))
            episode_access.append(info['ap_access_fraction'])
            episode_max_idle.append(info['ap_max_idle_steps'])
    finally:
        for agent_id, epsilon in saved_epsilons.items():
            agents[agent_id].epsilon = epsilon
        np.random.set_state(rng_state)

    denom = max(total_steps, 1)
    return {
        "throughput": total_throughput / denom,
        "jfi": total_jfi / denom,
        "jfi_time_average": float(np.mean(episode_temporal_jfi)),
        "objective_score": (float(np.mean(episode_objective_scores))
                            if episode_objective_scores else None),
        "ap_access_fraction": np.mean(episode_access, axis=0).tolist(),
        "ap_max_idle_steps": np.max(episode_max_idle, axis=0).tolist(),
        "active_aps": total_active_aps / denom,
        "energy_efficiency": total_energy_efficiency / denom,
        "coordination_links": total_coordination_links / denom,
        "coordination_requested_links": total_coordination_requests / denom,
        "coordination_utility_delta": total_coordination_utility_delta / denom,
        "coordination_power_reduction_w": total_coordination_power_reduction / denom,
        "team_reward": total_team_reward / denom,
        "mode": eval_mode,
        "scenarios": evaluated_scenarios,
        "user_status_by_episode": episode_user_status,
        "mean_user_satisfaction": (float(np.mean([
            row['satisfaction'] for rows in episode_user_status for row in rows
        ])) if episode_user_status else None),
        "mean_users_needing_rescue": (float(np.mean([
            sum(row['needs_rescue'] for row in rows) for rows in episode_user_status
        ])) if episode_user_status else None),
    }
