# main.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
import config  # Import đúng chỗ

from env.wifi_env import WiFiEnv
from agent.double_dqn import DoubleDQNAgent
from agent.qmix_helper import QMixer
from agent.shared_buffer import SharedReplayBuffer
from utils.logger import ExperimentLogger
from utils.plots import plot_comparison_curve, plot_learning_curve

def build_global_state(states_dict, throughputs, tx_powers, agent_ids):
    cca_array = [states_dict[aid][0] for aid in agent_ids]
    throughput_array = [min(t / 150.0, 1.0) for t in throughputs]
    tx_power_array = [p / config.P_MAX for p in tx_powers]
    return np.array(cca_array + throughput_array + tx_power_array, dtype=np.float32)

def soft_update(target_model, source_model, tau):
    for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.mul_(1.0 - tau)
        target_param.data.add_(tau * source_param.data)

def set_global_seed(seed):
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def normalize_eval_mode(eval_mode):
    eval_mode = (eval_mode or config.EVAL_MODE).lower()
    if eval_mode not in ("fixed", "generalization"):
        raise ValueError("eval_mode must be 'fixed' or 'generalization'")
    return eval_mode

def eval_score(eval_metrics, num_agents):
    throughput_norm = min(eval_metrics["throughput"] / (100.0 * num_agents), 1.0)
    jfi = eval_metrics["jfi"]
    active_ratio = eval_metrics["active_aps"] / num_agents
    return throughput_norm + jfi + active_ratio

def evaluate_policy(
    agents,
    episodes=5,
    steps=100,
    seed_base=10000,
    use_water_filling=False,
    action_size=5,
    eval_mode=None,
):
    eval_mode = normalize_eval_mode(eval_mode)
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
    )
    total_throughput = 0.0
    total_jfi = 0.0
    total_active_aps = 0.0
    total_steps = 0

    try:
        for episode_idx in range(episodes):
            np.random.seed(seed_base + episode_idx)
            states_dict, _ = env.reset()

            for _ in range(steps):
                actions_dict = {}
                for agent_id in env.agent_ids:
                    state_input = np.array(states_dict[agent_id], dtype=np.float32)
                    actions_dict[agent_id] = agents[agent_id].act(state_input)

                next_states_dict, _, terminated, truncated, info = env.step(actions_dict)
                total_throughput += info["throughput"]
                total_jfi += info["jfi"]
                total_active_aps += info["active_ap_count"]
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
        "mode": eval_mode,
    }

def train_marl(
    experiment_name="no_water_filling",
    use_water_filling=False,
    action_size=5,
    episodes=None,
    seed=None,
    eval_mode=None,
):
    seed = config.GLOBAL_SEED if seed is None else seed
    eval_mode = normalize_eval_mode(eval_mode)
    set_global_seed(seed)
    print("=== HUẤN LUYỆN MULTI-AGENT WIFI: QMIX + INFORMATION BOTTLENECK (RESEARCH VERSION) ===")

    env = WiFiEnv(
        fixed_topology=config.FIXED_TRAIN_SCENARIO,
        fixed_seed=config.TRAIN_SCENARIO_SEED,
        mobility_enabled=config.TRAIN_MOBILITY_ENABLED,
        use_water_filling=use_water_filling,
        action_size=action_size
    )
    num_agents = env.num_agents
    latent_size = 16

    device = torch.device("cpu")
    print(f"--> Hệ thống đang sử dụng: {device.type.upper()} để huấn luyện!")
    print(f"--> Seed: {seed} | Eval mode: {eval_mode}")

    results_dir = os.path.join("results", experiment_name)
    models_dir = os.path.join(results_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    logger = ExperimentLogger(save_dir=results_dir)

    # Khởi tạo agents với observation size trong config.
    agents = {
        agent_id: DoubleDQNAgent(state_size=config.OBS_SIZE, action_size=action_size, latent_size=latent_size)
        for agent_id in env.agent_ids
    }

    for agent_id in env.agent_ids:
        agents[agent_id].encoder = agents[agent_id].encoder.to(device)
        agents[agent_id].target_encoder = agents[agent_id].target_encoder.to(device)
        agents[agent_id].q_network = agents[agent_id].q_network.to(device)
        agents[agent_id].target_network = agents[agent_id].target_network.to(device)

    shared_buffer = SharedReplayBuffer(capacity=10000, num_agents=num_agents)

    global_state_dim = num_agents * 3
    q_mixer = QMixer(num_agents=num_agents, state_dim=global_state_dim).to(device)
    target_q_mixer = QMixer(num_agents=num_agents, state_dim=global_state_dim).to(device)
    target_q_mixer.load_state_dict(q_mixer.state_dict())

    all_parameters = list(q_mixer.parameters())
    for agent_id in env.agent_ids:
        all_parameters += list(agents[agent_id].encoder.parameters())
        all_parameters += list(agents[agent_id].q_network.parameters())

    optimizer = optim.Adam(all_parameters, lr=0.00005)
    gamma = 0.99
    batch_size = 64
    episodes = config.TRAIN_EPISODES if episodes is None else episodes
    beta_ib = 0.01
    target_tau = config.TARGET_UPDATE_TAU
    eval_interval = 25
    eval_episodes = 50

    history_network_throughput = []
    history_jfi = []
    history_eval_episodes = []
    history_eval_throughput = []
    history_eval_jfi = []
    history_eval_active_aps = []
    best_eval_score = -float("inf")

    for e in range(episodes):
        states_dict, _ = env.reset()
        total_network_throughput = 0
        total_jfi = 0
        total_marl_reward = 0
        total_active_aps = 0
        steps_completed = 0

        last_throughputs = [0.0] * num_agents
        last_tx_powers = [env.aps[i]["tx_power"] for i in range(num_agents)]

        for step in range(100):
            actions_dict = {}
            for agent_id in env.agent_ids:
                state_input = np.array(states_dict[agent_id], dtype=np.float32)
                actions_dict[agent_id] = agents[agent_id].act(state_input)

            global_state = build_global_state(states_dict, last_throughputs, last_tx_powers, env.agent_ids)

            next_states_dict, rewards_dict, terminated, truncated, info = env.step(actions_dict)

            current_throughputs = info["ap_individual_throughputs"]
            current_tx_powers = [env.aps[i]["tx_power"] for i in range(num_agents)]
            next_global_state = build_global_state(next_states_dict, current_throughputs, current_tx_powers, env.agent_ids)
            done = terminated or truncated

            # Push vào shared buffer đúng chỗ, trong vòng lặp step
            states_all = [states_dict[aid] for aid in env.agent_ids]
            actions_all = [actions_dict[aid] for aid in env.agent_ids]
            next_states_all = [next_states_dict[aid] for aid in env.agent_ids]
            team_reward = info["team_reward"]
            shared_buffer.add(states_all, actions_all, team_reward, next_states_all, done, global_state, next_global_state)

            last_throughputs = current_throughputs
            last_tx_powers = current_tx_powers

            # Pha training
            if len(shared_buffer) >= batch_size:
                s_all, a_all, team_r, s_next_all, d, g_s, g_s_next = shared_buffer.sample(batch_size)

                s_all = s_all.to(device)
                a_all = a_all.to(device)
                team_r = team_r.to(device)
                s_next_all = s_next_all.to(device)
                d = d.to(device)
                g_s = g_s.to(device)
                g_s_next = g_s_next.to(device)

                batch_agent_qs = []
                batch_agent_next_qs = []
                total_kl_loss = 0.0

                for idx, agent_id in enumerate(env.agent_ids):
                    s = s_all[:, idx, :]
                    a = a_all[:, idx].unsqueeze(1)
                    s_next = s_next_all[:, idx, :]

                    mu, log_var = agents[agent_id].encoder(s)
                    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
                    total_kl_loss += kl_loss

                    q_values = agents[agent_id].q_network(mu).gather(1, a)
                    batch_agent_qs.append(q_values)

                    with torch.no_grad():
                        mu_next_online, _ = agents[agent_id].encoder(s_next)
                        best_next_a = agents[agent_id].q_network(mu_next_online).argmax(1).unsqueeze(1)
                        mu_next_target, _ = agents[agent_id].target_encoder(s_next)
                        next_q = agents[agent_id].target_network(mu_next_target).gather(1, best_next_a)
                        batch_agent_next_qs.append(next_q)

                chosen_qs = torch.cat(batch_agent_qs, dim=1)
                target_next_qs = torch.cat(batch_agent_next_qs, dim=1)

                # Tên biến đồng nhất với sample output
                q_tot_predicted = q_mixer(chosen_qs, g_s)
                with torch.no_grad():
                    q_tot_next = target_q_mixer(target_next_qs, g_s_next)
                    q_tot_target = team_r + (gamma * q_tot_next * (1 - d))

                td_loss = nn.MSELoss()(q_tot_predicted, q_tot_target.detach())
                marl_ib_loss = td_loss + (beta_ib * total_kl_loss)

                optimizer.zero_grad()
                marl_ib_loss.backward()
                torch.nn.utils.clip_grad_norm_(all_parameters, max_norm=10.0)
                optimizer.step()
                soft_update(target_q_mixer, q_mixer, target_tau)
                for agent_id in env.agent_ids:
                    soft_update(agents[agent_id].target_encoder, agents[agent_id].encoder, target_tau)
                    soft_update(agents[agent_id].target_network, agents[agent_id].q_network, target_tau)

            states_dict = next_states_dict
            total_network_throughput += info['throughput']
            total_jfi += info['jfi']
            total_marl_reward += info["team_reward"]
            total_active_aps += info["active_ap_count"]
            steps_completed += 1

            if done:
                break

        # Epsilon decay đúng indent — mỗi episode đều chạy
        for agent_id in env.agent_ids:
            if agents[agent_id].epsilon > agents[agent_id].epsilon_min:
                agents[agent_id].epsilon *= agents[agent_id].epsilon_decay

        denom = max(steps_completed, 1)
        avg_throughput = total_network_throughput / denom
        avg_jfi = total_jfi / denom
        avg_marl_reward = total_marl_reward / denom
        avg_active_aps = total_active_aps / denom

        history_network_throughput.append(avg_throughput)
        history_jfi.append(avg_jfi)

        eval_metrics = None
        if (e + 1) % eval_interval == 0:
            eval_metrics = evaluate_policy(
                agents,
                episodes=eval_episodes,
                use_water_filling=use_water_filling,
                action_size=action_size,
                eval_mode=eval_mode,
            )
            history_eval_episodes.append(e + 1)
            history_eval_throughput.append(eval_metrics["throughput"])
            history_eval_jfi.append(eval_metrics["jfi"])
            history_eval_active_aps.append(eval_metrics["active_aps"])
            current_eval_score = eval_score(eval_metrics, num_agents)
            if current_eval_score > best_eval_score:
                best_eval_score = current_eval_score
                for agent_id in env.agent_ids:
                    torch.save(agents[agent_id].encoder.to("cpu").state_dict(), os.path.join(models_dir, f"best_ib_encoder_{agent_id}.pth"))
                    torch.save(agents[agent_id].q_network.to("cpu").state_dict(), os.path.join(models_dir, f"best_ib_qmix_{agent_id}_model.pth"))
                    agents[agent_id].encoder = agents[agent_id].encoder.to(device)
                    agents[agent_id].q_network = agents[agent_id].q_network.to(device)
                torch.save(q_mixer.to("cpu").state_dict(), os.path.join(models_dir, "best_ib_qmix_mixer_model.pth"))
                q_mixer = q_mixer.to(device)

        logger.log_episode(e + 1, avg_throughput, avg_jfi, avg_marl_reward, avg_active_aps, eval_metrics)
        current_epsilon = agents["ap_0"].epsilon
        print(f"Ván {e+1:03d}/{episodes} | Tốc độ mạng: {avg_throughput:6.2f} Mbps | JFI: {avg_jfi:.3f} | AP active: {avg_active_aps:.2f}/{num_agents} | Research-Reward: {avg_marl_reward:7.2f} | Epsilon: {current_epsilon:.2f}")
        if eval_metrics:
            print(f"  Eval {eval_metrics['mode']} | Tốc độ: {eval_metrics['throughput']:6.2f} Mbps | JFI: {eval_metrics['jfi']:.3f} | AP active: {eval_metrics['active_aps']:.2f}/{num_agents} | Best score: {best_eval_score:.3f}")

    print("\n=== HUẤN LUYỆN XONG! ĐANG ĐÓNG GÓI MÔ HÌNH VÀ GỌI UTILS... ===")

    for agent_id in env.agent_ids:
        torch.save(agents[agent_id].encoder.to("cpu").state_dict(), os.path.join(models_dir, f"ib_encoder_{agent_id}.pth"))
        torch.save(agents[agent_id].q_network.to("cpu").state_dict(), os.path.join(models_dir, f"ib_qmix_{agent_id}_model.pth"))
    torch.save(q_mixer.to("cpu").state_dict(), os.path.join(models_dir, "ib_qmix_mixer_model.pth"))
    print("Hệ thống đã lưu trữ gọn gàng cấu trúc mô hình tối giản!")

    plot_learning_curve(
        history_network_throughput,
        history_jfi,
        save_dir=os.path.join(results_dir, "plots"),
        eval_episodes=history_eval_episodes,
        eval_throughput=history_eval_throughput,
        eval_jfi=history_eval_jfi,
        eval_active_aps=history_eval_active_aps,
        num_agents=num_agents
    )

    return {
        "label": f"RL + Water-Filling (action={action_size})" if use_water_filling else f"RL without Water-Filling (action={action_size})",
        "throughput": history_network_throughput,
        "jfi": history_jfi,
        "eval_episodes": history_eval_episodes,
        "eval_throughput": history_eval_throughput,
        "eval_jfi": history_eval_jfi,
        "eval_active_aps": history_eval_active_aps,
        "eval_mode": eval_mode,
        "seed": seed,
    }


def run_power_allocation_comparison():
    experiments = [
        ("no_water_filling", False, 5),
        ("water_filling", True, 3),
    ]
    histories = []

    for experiment_name, use_water_filling, action_size in experiments:
        histories.append(
            train_marl(
                experiment_name=experiment_name,
                use_water_filling=use_water_filling,
                action_size=action_size,
                episodes=config.TRAIN_EPISODES,
                seed=config.GLOBAL_SEED,
                eval_mode=config.EVAL_MODE,
            )
        )

    plot_comparison_curve(histories, save_dir=os.path.join("results", "comparison_plots"))

if __name__ == "__main__":
    run_power_allocation_comparison()
