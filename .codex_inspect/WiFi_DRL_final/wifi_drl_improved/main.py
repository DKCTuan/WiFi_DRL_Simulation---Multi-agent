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
from utils.plots import plot_learning_curve
import matplotlib.pyplot as plt

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
    )
    total_throughput = 0.0
    total_jfi = 0.0
    total_active_aps = 0.0
    total_energy_efficiency = 0.0
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
                total_energy_efficiency += info["energy_efficiency"]
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
        "mode": eval_mode,
    }

def train_marl(
    experiment_name="no_water_filling",
    use_water_filling=False,
    action_size=config.FULL_AI_ACTION_SIZE,
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
    # Latent size và epsilon decay khác nhau theo action space để so sánh công bằng
    if use_water_filling:
        latent_size = config.LATENT_SIZE_HYBRID
        epsilon_decay = config.EPSILON_DECAY_HYBRID
    else:
        latent_size = config.LATENT_SIZE_FULL_AI
        epsilon_decay = config.EPSILON_DECAY_FULL_AI

    device = torch.device("cpu")
    print(f"--> Hệ thống đang sử dụng: {device.type.upper()} để huấn luyện!")
    print(f"--> Seed: {seed} | Eval mode: {eval_mode}")

    results_dir = os.path.join("results", experiment_name)
    models_dir = os.path.join(results_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    logger = ExperimentLogger(save_dir=results_dir)

    # Khởi tạo agents với observation size trong config.
    agents = {
        agent_id: DoubleDQNAgent(
            state_size=config.OBS_SIZE,
            action_size=action_size,
            latent_size=latent_size,
            epsilon_decay=epsilon_decay,
        )
        for agent_id in env.agent_ids
    }

    for agent_id in env.agent_ids:
        agents[agent_id].encoder = agents[agent_id].encoder.to(device)
        agents[agent_id].target_encoder = agents[agent_id].target_encoder.to(device)
        agents[agent_id].q_network = agents[agent_id].q_network.to(device)
        agents[agent_id].target_network = agents[agent_id].target_network.to(device)

    shared_buffer = SharedReplayBuffer(
        capacity=config.REPLAY_BUFFER_CAPACITY,
        num_agents=num_agents
    )

    global_state_dim = num_agents * 3
    q_mixer = QMixer(num_agents=num_agents, state_dim=global_state_dim).to(device)
    target_q_mixer = QMixer(num_agents=num_agents, state_dim=global_state_dim).to(device)
    target_q_mixer.load_state_dict(q_mixer.state_dict())

    all_parameters = list(q_mixer.parameters())
    for agent_id in env.agent_ids:
        all_parameters += list(agents[agent_id].encoder.parameters())
        all_parameters += list(agents[agent_id].q_network.parameters())

    optimizer = optim.Adam(all_parameters, lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.LR_DECAY_STEP,
        gamma=config.LR_DECAY_GAMMA,
    )
    gamma = config.GAMMA
    batch_size = config.BATCH_SIZE
    episodes = config.TRAIN_EPISODES if episodes is None else episodes
    beta_ib = config.IB_BETA
    target_tau = config.TARGET_UPDATE_TAU
    eval_interval = config.EVAL_INTERVAL
    eval_episodes = config.EVAL_EPISODES

    history_network_throughput = []
    history_jfi = []
    history_eval_episodes = []
    history_eval_throughput = []
    history_eval_jfi = []
    history_eval_active_aps = []
    history_eval_energy_efficiency = []
    best_eval_score = -float("inf")

    for e in range(episodes):
        states_dict, _ = env.reset()
        total_network_throughput = 0
        total_jfi = 0
        total_marl_reward = 0
        total_active_aps = 0
        total_energy_efficiency = 0
        steps_completed = 0

        last_throughputs = [0.0] * num_agents
        last_tx_powers = [env.aps[i]["tx_power"] for i in range(num_agents)]

        for step in range(config.TRAIN_STEPS_PER_EPISODE):
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

                td_loss = nn.SmoothL1Loss()(q_tot_predicted, q_tot_target.detach())
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
            total_energy_efficiency += info["energy_efficiency"]
            steps_completed += 1

            if done:
                break

        # Epsilon decay mỗi episode
        for agent_id in env.agent_ids:
            if agents[agent_id].epsilon > agents[agent_id].epsilon_min:
                agents[agent_id].epsilon *= agents[agent_id].epsilon_decay

        # LR decay theo schedule
        scheduler.step()

        denom = max(steps_completed, 1)
        avg_throughput = total_network_throughput / denom
        avg_jfi = total_jfi / denom
        avg_marl_reward = total_marl_reward / denom
        avg_active_aps = total_active_aps / denom
        avg_energy_efficiency = total_energy_efficiency / denom

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
            history_eval_energy_efficiency.append(eval_metrics["energy_efficiency"])
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

        logger.log_episode(e + 1, avg_throughput, avg_jfi, avg_marl_reward, avg_active_aps, avg_energy_efficiency, eval_metrics)
        current_epsilon = agents["ap_0"].epsilon
        print(f"Ván {e+1:03d}/{episodes} | Tốc độ mạng: {avg_throughput:6.2f} Mbps | JFI: {avg_jfi:.3f} | AP active: {avg_active_aps:.2f}/{num_agents} | Research-Reward: {avg_marl_reward:7.2f} | Epsilon: {current_epsilon:.2f}")
        if eval_metrics:
            print(f"  Eval {eval_metrics['mode']} | Tốc độ: {eval_metrics['throughput']:6.2f} Mbps | JFI: {eval_metrics['jfi']:.3f} | AP active: {eval_metrics['active_aps']:.2f}/{num_agents} | Best score: {best_eval_score:.3f}")

    print("\n=== HUẤN LUYỆN XONG! ĐANG ĐÓNG GÓI MÔ HÌNH VÀ GỌI UTILS... ===")

    best_mixer_path = os.path.join(models_dir, "best_ib_qmix_mixer_model.pth")
    if os.path.exists(best_mixer_path):
        for agent_id in env.agent_ids:
            encoder_path = os.path.join(models_dir, f"best_ib_encoder_{agent_id}.pth")
            q_path = os.path.join(models_dir, f"best_ib_qmix_{agent_id}_model.pth")
            agents[agent_id].encoder.load_state_dict(torch.load(encoder_path, map_location=device))
            agents[agent_id].q_network.load_state_dict(torch.load(q_path, map_location=device))
        q_mixer.load_state_dict(torch.load(best_mixer_path, map_location=device))
        print(f"Restored best validation checkpoint before final save | Best score: {best_eval_score:.3f}")

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
        eval_energy_efficiency=history_eval_energy_efficiency,
        num_agents=num_agents
    )

    return {
        "label": (
            f"Hybrid-AI + Water-Filling ({config.HYBRID_ACTION_SIZE} actions)"
            if use_water_filling else
            f"Full-AI control ({config.FULL_AI_ACTION_SIZE} actions)"
        ),
        "throughput": history_network_throughput,
        "jfi": history_jfi,
        "eval_episodes": history_eval_episodes,
        "eval_throughput": history_eval_throughput,
        "eval_jfi": history_eval_jfi,
        "eval_active_aps": history_eval_active_aps,
        "eval_energy_efficiency": history_eval_energy_efficiency,
        "eval_mode": eval_mode,
        "seed": seed,
    }

def moving_average(data, window=25):
    data = np.array(data, dtype=np.float32)
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")

def aggregate_results(results_list):
    if not results_list:
        raise ValueError("results_list must not be empty")

    agg = {}
    for key in ["throughput", "jfi", "eval_throughput", "eval_jfi", "eval_active_aps", "eval_energy_efficiency"]:
        lengths = {len(r[key]) for r in results_list}
        if len(lengths) != 1:
            raise ValueError(f"All runs must have the same number of {key} points, got {sorted(lengths)}")
        arr = np.array([r[key] for r in results_list], dtype=np.float32)
        agg[key + "_mean"] = arr.mean(axis=0)
        agg[key + "_std"] = arr.std(axis=0)

    eval_episodes = results_list[0]["eval_episodes"]
    for r in results_list[1:]:
        if r["eval_episodes"] != eval_episodes:
            raise ValueError("All runs must use the same eval episodes for aggregation")

    agg["eval_episodes"] = results_list[0]["eval_episodes"]
    return agg

def best_so_far(values):
    return np.maximum.accumulate(np.asarray(values, dtype=np.float32))


def plot_comparison_multiseed(agg_full_ai, agg_hybrid, save_path="results/compare_training_multiseed.png"):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    window = config.PLOT_SMOOTHING_WINDOW
    skip = min(config.PLOT_SKIP_INITIAL_EPISODES, len(agg_full_ai["throughput_mean"]) - 1)
    series = [
        (agg_full_ai, f"Full-AI control ({config.FULL_AI_ACTION_SIZE} actions)", "tab:orange"),
        (agg_hybrid, f"Hybrid-AI + Water-Filling ({config.HYBRID_ACTION_SIZE} actions)", "tab:red"),
    ]

    for agg, label, color in series:
        for ax, key, ylabel in [
            (axes[0], "throughput", "Throughput (Mbps)"),
            (axes[1], "jfi", "JFI"),
        ]:
            mean = agg[f"{key}_mean"][skip:]
            std = agg[f"{key}_std"][skip:]
            x = np.arange(skip + 1, skip + len(mean) + 1)
            ma = moving_average(mean, window)
            x_ma = np.arange(skip + len(mean) - len(ma) + 1, skip + len(mean) + 1)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12)
            ax.plot(x_ma, ma, color=color, linewidth=2, label=f"{label} MA-{window}")
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.35)

    axes[0].set_title("Training comparison")
    axes[0].legend()
    axes[1].legend()
    axes[1].set_xlabel("Episode")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved: {save_path}")


def plot_eval_comparison_multiseed(agg_full_ai, agg_hybrid, save_path="results/compare_eval_multiseed.png"):
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    series = [
        (agg_full_ai, f"Full-AI control ({config.FULL_AI_ACTION_SIZE} actions)", "tab:orange"),
        (agg_hybrid, f"Hybrid-AI + Water-Filling ({config.HYBRID_ACTION_SIZE} actions)", "tab:red"),
    ]

    for agg, label, color in series:
        ep = np.asarray(agg["eval_episodes"])
        for ax, key, ylabel in [
            (axes[0], "eval_throughput", "Throughput (Mbps)"),
            (axes[1], "eval_jfi", "JFI"),
            (axes[2], "eval_active_aps", "Active APs"),
            (axes[3], "eval_energy_efficiency", "Energy Efficiency (Mbps/W)"),
        ]:
            ax.errorbar(
                ep,
                agg[f"{key}_mean"],
                yerr=agg[f"{key}_std"],
                marker="o",
                markersize=3,
                linewidth=1.5,
                color=color,
                label=label,
                capsize=2,
                alpha=0.9,
            )
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.35)

    axes[0].set_title("Greedy policy eval (epsilon=0)")
    axes[0].legend()
    axes[3].set_xlabel("Episode")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved: {save_path}")


def plot_publication_comparison(agg_full_ai, agg_hybrid, save_path="results/publication_policy_comparison.png"):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    series = [
        (agg_full_ai, f"Full-AI control ({config.FULL_AI_ACTION_SIZE} actions)", "tab:orange"),
        (agg_hybrid, f"Hybrid-AI + Water-Filling ({config.HYBRID_ACTION_SIZE} actions)", "tab:red"),
    ]

    for agg, label, color in series:
        ep = np.asarray(agg["eval_episodes"])
        thr = best_so_far(agg["eval_throughput_mean"])
        jfi = best_so_far(agg["eval_jfi_mean"])
        ee = best_so_far(agg["eval_energy_efficiency_mean"])
        axes[0].plot(ep, thr, color=color, linewidth=2.4, marker="o", markersize=3, label=label)
        axes[1].plot(ep, jfi, color=color, linewidth=2.4, marker="o", markersize=3, label=label)
        axes[2].plot(ep, ee, color=color, linewidth=2.4, marker="o", markersize=3, label=label)

    axes[0].set_title("Best validated greedy policy trend")
    axes[0].set_ylabel("Throughput (Mbps)")
    axes[1].set_ylabel("JFI")
    axes[2].set_ylabel("Energy Efficiency (Mbps/W)")
    axes[2].set_xlabel("Episode")
    axes[1].set_ylim(0, 1.05)
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def print_summary_table(agg_full_ai, agg_hybrid, n_last=5):
    print(f"\n=== FINAL EVAL SUMMARY ({n_last} last eval points, mean +/- std) ===")
    for name, agg in [("Full-AI control", agg_full_ai), ("Hybrid-AI + Water-Filling", agg_hybrid)]:
        thr = agg["eval_throughput_mean"][-n_last:].mean()
        thr_s = agg["eval_throughput_std"][-n_last:].mean()
        jfi = agg["eval_jfi_mean"][-n_last:].mean()
        jfi_s = agg["eval_jfi_std"][-n_last:].mean()
        aps = agg["eval_active_aps_mean"][-n_last:].mean()
        aps_s = agg["eval_active_aps_std"][-n_last:].mean()
        ee = agg["eval_energy_efficiency_mean"][-n_last:].mean()
        ee_s = agg["eval_energy_efficiency_std"][-n_last:].mean()
        print(
            f"  {name:28s} | Throughput: {thr:7.2f} +/- {thr_s:5.2f} Mbps"
            f" | JFI: {jfi:.3f} +/- {jfi_s:.3f}"
            f" | Active APs: {aps:.2f} +/- {aps_s:.2f}"
            f" | EE: {ee:.2f} +/- {ee_s:.2f} Mbps/W"
        )


if __name__ == "__main__":
    SEEDS = config.FINAL_SEEDS
    EPISODES = config.TRAIN_EPISODES

    results_full_ai = []
    results_hybrid = []

    for seed in SEEDS:
        results_full_ai.append(train_marl(
            experiment_name=f"final_full_ai_seed{seed}",
            use_water_filling=False,
            action_size=config.FULL_AI_ACTION_SIZE,
            episodes=EPISODES,
            seed=seed,
            eval_mode="fixed",
        ))
        results_hybrid.append(train_marl(
            experiment_name=f"final_hybrid_wf_seed{seed}",
            use_water_filling=True,
            action_size=config.HYBRID_ACTION_SIZE,
            episodes=EPISODES,
            seed=seed,
            eval_mode="fixed",
        ))

    agg_full_ai = aggregate_results(results_full_ai)
    agg_hybrid = aggregate_results(results_hybrid)

    plot_comparison_multiseed(agg_full_ai, agg_hybrid)
    plot_eval_comparison_multiseed(agg_full_ai, agg_hybrid)
    plot_publication_comparison(agg_full_ai, agg_hybrid)
    print_summary_table(agg_full_ai, agg_hybrid)
