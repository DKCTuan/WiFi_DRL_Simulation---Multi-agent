# training/evaluate.py
"""Đánh giá policy (epsilon=0, greedy) trên môi trường cố định/generalization."""
import numpy as np

import config
from env.wifi_env import WiFiEnv
from training.utils import normalize_eval_mode


def eval_score(eval_metrics, num_agents):
    """Điểm tổng hợp dùng để chọn checkpoint tốt nhất (không phải metric báo cáo)."""
    throughput_norm = min(eval_metrics["throughput"] / (100.0 * num_agents), 1.0)
    jfi = eval_metrics["jfi"]
    energy_norm = min(eval_metrics.get("energy_efficiency", 0.0) / 2000.0, 1.0)
    return throughput_norm + jfi + (0.5 * energy_norm)


def evaluate_policy(
    agents,
    episodes=5,
    steps=None,
    seed_base=10000,
    use_water_filling=False,
    action_size=config.FULL_AI_ACTION_SIZE,
    eval_mode=None,
    num_stas=None,   # None → dùng AP_LOAD_PROFILE heterogeneous; int → đồng đều K
    cooperation_enabled=False,
):
    eval_mode = normalize_eval_mode(eval_mode)
    steps = config.TRAIN_STEPS_PER_EPISODE if steps is None else steps
    saved_epsilons = {agent_id: agent.epsilon for agent_id, agent in agents.items()}
    rng_state = np.random.get_state()

    for agent in agents.values():
        agent.epsilon = 0.0

    env = WiFiEnv(
        verbose=False,
        fixed_topology=(eval_mode == "fixed"),
        fixed_seed=config.TRAIN_SCENARIO_SEED if eval_mode == "fixed" else None,
        mobility_enabled=config.TRAIN_MOBILITY_ENABLED,
        use_water_filling=use_water_filling,
        action_size=action_size,
        num_stas_per_ap=num_stas,
        cooperation_enabled=cooperation_enabled,
    )
    total_throughput = 0.0
    total_jfi = 0.0
    total_active_aps = 0.0
    total_energy_efficiency = 0.0
    total_coordination_links = 0.0
    total_steps = 0

    try:
        for episode_idx in range(episodes):
            np.random.seed(seed_base + episode_idx)
            states_dict, _ = env.reset(seed=seed_base + episode_idx)

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
                total_steps += 1
                states_dict = next_states_dict

                if terminated or truncated:
                    break
    finally:
        for agent_id, epsilon in saved_epsilons.items():
            agents[agent_id].epsilon = epsilon
        np.random.set_state(rng_state)

    denom = max(total_steps, 1)
    return {
        "throughput": total_throughput / denom,
        "jfi": total_jfi / denom,
        "active_aps": total_active_aps / denom,
        "energy_efficiency": total_energy_efficiency / denom,
        "coordination_links": total_coordination_links / denom,
        "mode": eval_mode,
    }
