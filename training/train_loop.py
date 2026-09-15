# training/train_loop.py
"""Vòng lặp huấn luyện chính: QMIX + Double DQN + VIB encoder trên buffer
dùng chung (Section 4.2-4.4 của bài báo)."""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config
from env.wifi_env import WiFiEnv
from agent.double_dqn import DoubleDQNAgent
from agent.qmix_helper import QMixer
from agent.shared_buffer import SharedReplayBuffer
from utils.logger import ExperimentLogger
from utils.plots import plot_learning_curve

from training.utils import build_global_state, soft_update, set_global_seed, normalize_eval_mode
from training.evaluate import evaluate_policy, eval_score


def train_marl(
    experiment_name="no_water_filling",
    use_water_filling=False,
    action_size=config.FULL_AI_ACTION_SIZE,
    episodes=None,
    seed=None,
    eval_mode=None,
    num_stas=None,   # None → AP_LOAD_PROFILE heterogeneous; int → đồng đều K
    use_double_dqn=True,   # False → vanilla DQN (target chọn + đánh giá) cho ablation "no_ddqn"
    use_qmix=True,         # False → VDN (Q_tot = sum Q_i), not IQL
    ib_beta=None,          # None -> clean default (0); VIB ablation uses config.VIB_BETA
    cooperation_enabled=False,
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
        action_size=action_size,
        num_stas_per_ap=num_stas,
        cooperation_enabled=cooperation_enabled,
    )
    num_agents = env.num_agents
    # Latent size và epsilon decay khác nhau theo action space để so sánh công bằng
    if use_water_filling:
        latent_size = config.LATENT_SIZE_HYBRID
        epsilon_decay = config.EPSILON_DECAY_HYBRID
    else:
        latent_size = config.LATENT_SIZE_FULL_AI
        epsilon_decay = config.EPSILON_DECAY_FULL_AI

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Hệ thống đang sử dụng: {device.type.upper()} để huấn luyện!")
    print(f"--> Seed: {seed} | Eval mode: {eval_mode}")

    results_dir = os.path.join("results", experiment_name)
    models_dir = os.path.join(results_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    logger = ExperimentLogger(save_dir=results_dir)

    # Khởi tạo agents với observation size trong config.
    agents = {
        agent_id: DoubleDQNAgent(
            state_size=env.observation_size,
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
        num_agents=num_agents,
    )

    # QMIX mixer: dùng khi use_qmix=True. Khi use_qmix=False (VDN), mixer
    # vẫn được khởi tạo để giữ code path đơn giản
    # nhưng KHÔNG tham gia optimizer và KHÔNG được dùng để tính Q_tot — thay vào
    # đó Q_tot = tổng các Q cục bộ (phép trộn cố định, không học), đúng tinh thần IQL.
    global_state_dim = num_agents * env.observation_size
    q_mixer = QMixer(num_agents=num_agents, state_dim=global_state_dim).to(device)
    target_q_mixer = QMixer(num_agents=num_agents, state_dim=global_state_dim).to(device)
    target_q_mixer.load_state_dict(q_mixer.state_dict())

    all_parameters = list(q_mixer.parameters()) if use_qmix else []
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
    beta_ib = config.IB_BETA if ib_beta is None else ib_beta
    target_tau = config.TARGET_UPDATE_TAU
    eval_interval = config.EVAL_INTERVAL
    eval_episodes = config.EVAL_EPISODES

    history_network_throughput = []
    history_jfi = []
    history_train_active_aps = []
    history_train_reward = []
    history_train_energy_efficiency = []
    history_train_epsilon = []
    history_train_coordination_links = []
    history_eval_episodes = []
    history_eval_throughput = []
    history_eval_jfi = []
    history_eval_active_aps = []
    history_eval_energy_efficiency = []
    history_eval_coordination_links = []
    best_eval_score = -float("inf")

    for e in range(episodes):
        states_dict, _ = env.reset()

        # Curriculum: ramp trọng số reward "tiết kiệm AP" từ 0 → full trong
        # REWARD_ACTIVE_AP_WARMUP_EPISODES episode đầu (0 nếu tắt curriculum).
        if config.REWARD_ACTIVE_AP_WARMUP_EPISODES > 0:
            warmup_frac = min(1.0, e / config.REWARD_ACTIVE_AP_WARMUP_EPISODES)
        else:
            warmup_frac = 1.0
        env.active_ap_weight = config.REWARD_ACTIVE_AP_WEIGHT * warmup_frac

        total_network_throughput = 0
        total_jfi = 0
        total_marl_reward = 0
        total_active_aps = 0
        total_energy_efficiency = 0
        total_coordination_links = 0
        steps_completed = 0
        optimizer_stepped = False

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
            if len(shared_buffer) >= max(batch_size, config.LEARNING_STARTS):
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

                    # Fix: trước đây Q-network luôn nhận mu (giá trị trung bình
                    # xác định), khiến hàm reparameterize() không bao giờ được
                    # gọi và "bottleneck" chỉ còn là một số hạng regularization
                    # KL cộng vào loss, không thực sự nén/nhiễu thông tin đi vào
                    # quyết định — làm cho ablation "no_vib" (beta=0) gần như
                    # không đổi gì so với baseline.
                    # Đúng theo Eq. (6) của bài báo: khi TRAIN, agent dùng latent
                    # đã sample z = mu + eps*std (reparameterization trick) để
                    # gradient của TD-loss thực sự chảy qua nhiễu ngẫu nhiên đó,
                    # khiến việc bật/tắt VIB có ảnh hưởng thật đến Q-values học
                    # được. Khi EXECUTION (act(), xem agent/double_dqn.py) vẫn
                    # dùng mu để có policy xác định/ổn định, đúng như bài báo mô
                    # tả: "During execution, the agent uses the latent mean µϕ
                    # to obtain a deterministic and stable policy."
                    # Ablation "no_vib" should remove the stochastic bottleneck,
                    # not only the KL penalty. With beta=0, train deterministically
                    # on mu so the variant is a clean w/o-IB comparison.
                    z = mu if beta_ib == 0.0 else agents[agent_id].encoder.reparameterize(mu, log_var)
                    q_values = agents[agent_id].q_network(z).gather(1, a)
                    batch_agent_qs.append(q_values)

                    with torch.no_grad():
                        mu_next_target, _ = agents[agent_id].target_encoder(s_next)
                        target_next_q_values = agents[agent_id].target_network(mu_next_target)
                        if use_double_dqn:
                            # Double DQN: online network CHỌN action tốt nhất,
                            # target network chỉ ĐÁNH GIÁ giá trị của action đó
                            # (giảm overestimation bias, theo van Hasselt et al. [8]).
                            mu_next_online, _ = agents[agent_id].encoder(s_next)
                            best_next_a = agents[agent_id].q_network(mu_next_online).argmax(1).unsqueeze(1)
                            next_q = target_next_q_values.gather(1, best_next_a)
                        else:
                            # Vanilla DQN (ablation "no_ddqn"): target network vừa
                            # chọn vừa đánh giá action tốt nhất (max), không tách
                            # vai trò select/evaluate như Double DQN.
                            next_q = target_next_q_values.max(1, keepdim=True)[0]
                        batch_agent_next_qs.append(next_q)

                chosen_qs = torch.cat(batch_agent_qs, dim=1)
                target_next_qs = torch.cat(batch_agent_next_qs, dim=1)

                if use_qmix:
                    q_tot_predicted = q_mixer(chosen_qs, g_s)
                    with torch.no_grad():
                        q_tot_next = target_q_mixer(target_next_qs, g_s_next)
                else:
                    # VDN ablation: bỏ learned mixing network
                    # đơn điệu fψ, mỗi agent hồi quy TD-loss riêng trên cùng
                    # team_reward — Q_tot chỉ là tổng cố định (không học) của các
                    # Q cục bộ, không có hypernetwork nào được huấn luyện.
                    # VDN: a fixed sum of local utilities trained with one
                    # joint TD loss. This is not Independent Q-Learning.
                    q_tot_predicted = chosen_qs.sum(dim=1, keepdim=True)
                    with torch.no_grad():
                        q_tot_next = target_next_qs.sum(dim=1, keepdim=True)

                with torch.no_grad():
                    q_tot_target = team_r + (gamma * q_tot_next * (1 - d))

                td_loss = nn.SmoothL1Loss()(q_tot_predicted, q_tot_target.detach())
                marl_ib_loss = td_loss + (beta_ib * total_kl_loss)

                optimizer.zero_grad()
                marl_ib_loss.backward()
                torch.nn.utils.clip_grad_norm_(all_parameters, max_norm=10.0)
                optimizer.step()
                optimizer_stepped = True
                if use_qmix:
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
            total_coordination_links += info.get("coordination_link_count", 0)
            steps_completed += 1
            logger.log_step(e + 1, step + 1, info)

            if done:
                break

        # Epsilon decay mỗi episode
        for agent_id in env.agent_ids:
            if agents[agent_id].epsilon > agents[agent_id].epsilon_min:
                agents[agent_id].epsilon *= agents[agent_id].epsilon_decay

        # LR decay theo schedule, only after the first optimizer update.
        if optimizer_stepped:
            scheduler.step()

        denom = max(steps_completed, 1)
        avg_throughput = total_network_throughput / denom
        avg_jfi = total_jfi / denom
        avg_marl_reward = total_marl_reward / denom
        avg_active_aps = total_active_aps / denom
        avg_energy_efficiency = total_energy_efficiency / denom
        avg_coordination_links = total_coordination_links / denom

        history_network_throughput.append(avg_throughput)
        history_jfi.append(avg_jfi)
        history_train_active_aps.append(avg_active_aps)
        history_train_reward.append(avg_marl_reward)
        history_train_energy_efficiency.append(avg_energy_efficiency)
        history_train_coordination_links.append(avg_coordination_links)
        history_train_epsilon.append(agents["ap_0"].epsilon)

        eval_metrics = None
        if (e + 1) % eval_interval == 0:
            eval_metrics = evaluate_policy(
                agents,
                episodes=eval_episodes,
                use_water_filling=use_water_filling,
                action_size=action_size,
                eval_mode=eval_mode,
                num_stas=num_stas,
                cooperation_enabled=cooperation_enabled,
            )
            history_eval_episodes.append(e + 1)
            history_eval_throughput.append(eval_metrics["throughput"])
            history_eval_jfi.append(eval_metrics["jfi"])
            history_eval_active_aps.append(eval_metrics["active_aps"])
            history_eval_energy_efficiency.append(eval_metrics["energy_efficiency"])
            history_eval_coordination_links.append(eval_metrics["coordination_links"])
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

        logger.log_episode(
            e + 1, avg_throughput, avg_jfi, avg_marl_reward, avg_active_aps,
            avg_energy_efficiency, eval_metrics, epsilon=agents["ap_0"].epsilon,
            coordination_links=avg_coordination_links,
        )
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
        num_agents=num_agents,
    )

    return {
        "label": (
            (
                f"Legacy fixed-suppression cooperation ({config.COOPERATIVE_HYBRID_ACTION_SIZE} actions)"
                if cooperation_enabled else
                f"Hybrid-AI + Water-Filling ({config.HYBRID_ACTION_SIZE} actions)"
            )
            if use_water_filling else
            f"Full-AI control ({config.FULL_AI_ACTION_SIZE} actions)"
        ),
        "throughput": history_network_throughput,
        "jfi": history_jfi,
        "train_active_aps": history_train_active_aps,
        "train_reward": history_train_reward,
        "train_energy_efficiency": history_train_energy_efficiency,
        "train_epsilon": history_train_epsilon,
        "train_coordination_links": history_train_coordination_links,
        "eval_episodes": history_eval_episodes,
        "eval_throughput": history_eval_throughput,
        "eval_jfi": history_eval_jfi,
        "eval_active_aps": history_eval_active_aps,
        "eval_energy_efficiency": history_eval_energy_efficiency,
        "eval_coordination_links": history_eval_coordination_links,
        "eval_mode": eval_mode,
        "seed": seed,
        "best_eval_score": best_eval_score,
        "models_dir": models_dir,
        "algorithm": "qmix" if use_qmix else "vdn",
        "use_double_dqn": use_double_dqn,
        "ib_beta": float(beta_ib),
        "vib_enabled": bool(beta_ib > 0.0),
        "cooperation_enabled": bool(cooperation_enabled),
        "observation_size": int(env.observation_size),
        "action_size": int(action_size),
    }
