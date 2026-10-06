# training/train_loop.py
"""Vòng lặp huấn luyện chính: QMIX + Double DQN + VIB encoder trên buffer
dùng chung (Section 4.2-4.4 của bài báo)."""
import os
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config
from env.wifi_env import WiFiEnv
from env.scenarios import training_scenario, scenario_bank
from agent.double_dqn import DoubleDQNAgent
from agent.qmix_helper import QMixer
from agent.shared_buffer import SharedReplayBuffer
from utils.logger import ExperimentLogger
from utils.plots import plot_learning_curve

from training.utils import build_global_state, soft_update, set_global_seed, normalize_eval_mode
from training.evaluate import evaluate_policy, eval_score
from training.objectives import validate_stationary_reward
from training.checkpoints import (
    CHECKPOINT_VERSION,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
    training_signature,
    write_json_atomic,
)


def _process_rss_mib():
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 ** 2)
    except Exception:
        try:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return peak / (1024 ** 2 if sys.platform == "darwin" else 1024)
        except Exception:
            return None


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
    resume=False,
    checkpoint_interval=25,
    checkpoint_callback=None,
):
    validate_stationary_reward()
    seed = config.GLOBAL_SEED if seed is None else seed
    eval_mode = normalize_eval_mode(eval_mode)
    if eval_mode in ('test', 'fixed_test'):
        raise ValueError('Test layouts cannot select checkpoints; use validation')
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

    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    checkpoint_path = os.path.join(models_dir, "last_training_checkpoint.pt")
    logger = ExperimentLogger(save_dir=results_dir, append=resume)

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
    if min(episodes, config.TRAIN_STEPS_PER_EPISODE, eval_interval, eval_episodes) <= 0:
        raise ValueError('Training and evaluation lengths must be positive')

    history_network_throughput = []
    history_jfi = []
    history_train_active_aps = []
    history_train_reward = []
    history_train_energy_efficiency = []
    history_train_epsilon = []
    history_train_coordination_links = []
    history_train_coordination_power_reduction = []
    history_eval_episodes = []
    history_eval_throughput = []
    history_eval_jfi = []
    history_eval_active_aps = []
    history_eval_energy_efficiency = []
    history_eval_team_reward = []
    history_eval_coordination_links = []
    history_eval_coordination_power_reduction = []
    history_access = {key: [] for key in (
        'train_jfi_time_average', 'train_ap_access_fraction', 'train_ap_max_idle_steps',
        'eval_jfi_time_average', 'eval_ap_access_fraction', 'eval_ap_max_idle_steps',
    )}
    history_scenarios = []
    optimizer_updates_total = 0
    best_eval_score = -float("inf")
    best_model_state = None
    start_episode = 0

    signature = training_signature(
        seed=seed,
        use_water_filling=use_water_filling,
        action_size=action_size,
        num_stas=num_stas,
        use_double_dqn=use_double_dqn,
        use_qmix=use_qmix,
        ib_beta=beta_ib,
        cooperation_enabled=cooperation_enabled,
        eval_mode=eval_mode,
    )

    if resume:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Cannot resume: checkpoint not found at {checkpoint_path}")
        saved = load_checkpoint(checkpoint_path, signature)
        for agent_id in env.agent_ids:
            for name in ("encoder", "target_encoder", "q_network", "target_network"):
                getattr(agents[agent_id], name).load_state_dict(saved["agents"][agent_id][name])
            agents[agent_id].epsilon = float(saved["agents"][agent_id]["epsilon"])
        q_mixer.load_state_dict(saved["q_mixer"])
        target_q_mixer.load_state_dict(saved["target_q_mixer"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        shared_buffer.load_state_dict(saved["replay_buffer"])
        start_episode = int(saved["episode_completed"])
        if start_episode > episodes:
            raise ValueError(
                f"Checkpoint already completed {start_episode} episodes, "
                f"but requested total is {episodes}"
            )
        history = saved["history"]
        history_scenarios = history['train_scenarios']
        optimizer_updates_total = saved['optimizer_updates_total']
        history_access = {key: history[key] for key in history_access}
        history_network_throughput = history["throughput"]
        history_jfi = history["jfi"]
        history_train_active_aps = history["train_active_aps"]
        history_train_reward = history["train_reward"]
        history_train_energy_efficiency = history["train_energy_efficiency"]
        history_train_epsilon = history["train_epsilon"]
        history_train_coordination_links = history["train_coordination_links"]
        history_train_coordination_power_reduction = history["train_coordination_power_reduction_w"]
        history_eval_episodes = history["eval_episodes"]
        history_eval_throughput = history["eval_throughput"]
        history_eval_jfi = history["eval_jfi"]
        history_eval_active_aps = history["eval_active_aps"]
        history_eval_energy_efficiency = history["eval_energy_efficiency"]
        history_eval_team_reward = history["eval_team_reward"]
        history_eval_coordination_links = history["eval_coordination_links"]
        history_eval_coordination_power_reduction = history["eval_coordination_power_reduction_w"]
        best_eval_score = float(saved["best_eval_score"])
        best_model_state = saved.get("best_model_state")
        restore_rng_state(saved["rng_state"])
        print(
            f"--> Resuming at episode {start_episode}/{episodes}; "
            f"replay={len(shared_buffer):,} transitions"
        )

    manifest_path = os.path.join(results_dir, "run_manifest.json")
    if resume and os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as stream:
            run_manifest = json.load(stream)
    else:
        run_manifest = {}
    run_manifest.setdefault("created_at_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    run_manifest.update({
        "experiment_name": experiment_name,
        "protocol_version": config.THREE_STAGE_PROTOCOL_VERSION,
        "target_episodes": int(episodes),
        "steps_per_episode": int(config.TRAIN_STEPS_PER_EPISODE),
        "scenario_banks": {split: scenario_bank(split) for split in ('train','validation','test')},
        "checkpoint_interval": int(checkpoint_interval),
        "signature": signature,
        "runtime": {
            "torch_version": str(torch.__version__),
            "device": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
        },
    })
    run_manifest.setdefault("sessions", []).append({
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "resume": bool(resume),
        "resume_from_episode": int(start_episode),
    })
    write_json_atomic(manifest_path, run_manifest)

    def write_training_checkpoint(episode_completed):
        payload = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "signature": signature,
            "episode_completed": int(episode_completed),
            "optimizer_updates_total": optimizer_updates_total,
            "best_eval_score": float(best_eval_score),
            "best_model_state": best_model_state,
            "agents": {
                agent_id: {
                    "encoder": agents[agent_id].encoder.state_dict(),
                    "target_encoder": agents[agent_id].target_encoder.state_dict(),
                    "q_network": agents[agent_id].q_network.state_dict(),
                    "target_network": agents[agent_id].target_network.state_dict(),
                    "epsilon": float(agents[agent_id].epsilon),
                }
                for agent_id in env.agent_ids
            },
            "q_mixer": q_mixer.state_dict(),
            "target_q_mixer": target_q_mixer.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "replay_buffer": shared_buffer.state_dict(),
            "rng_state": capture_rng_state(),
            "history": {
                **history_access,
                "train_scenarios": history_scenarios,
                "throughput": history_network_throughput,
                "jfi": history_jfi,
                "train_active_aps": history_train_active_aps,
                "train_reward": history_train_reward,
                "train_energy_efficiency": history_train_energy_efficiency,
                "train_epsilon": history_train_epsilon,
                "train_coordination_links": history_train_coordination_links,
                "train_coordination_power_reduction_w": history_train_coordination_power_reduction,
                "eval_episodes": history_eval_episodes,
                "eval_throughput": history_eval_throughput,
                "eval_jfi": history_eval_jfi,
                "eval_active_aps": history_eval_active_aps,
                "eval_energy_efficiency": history_eval_energy_efficiency,
                "eval_team_reward": history_eval_team_reward,
                "eval_coordination_links": history_eval_coordination_links,
                "eval_coordination_power_reduction_w": history_eval_coordination_power_reduction,
            },
            "saved_at_unix": float(time.time()),
            "runtime": {
                "torch_version": str(torch.__version__),
                "device": str(device),
                "cuda_available": bool(torch.cuda.is_available()),
            },
        }
        save_checkpoint(checkpoint_path, payload)
        if checkpoint_callback is not None:
            checkpoint_callback(checkpoint_path, episode_completed)

    for e in range(start_episode, episodes):
        episode_started = time.perf_counter()
        scenario = training_scenario(e) if config.MULTI_LAYOUT_TRAINING else None
        states_dict, _ = env.reset(seed=seed + e, options={'scenario':scenario})
        history_scenarios.append(scenario)

        # No episode-dependent coefficient: replay, eval and shield share the
        # stationary objective installed by WiFiEnv.

        total_network_throughput = 0
        total_jfi = 0
        total_marl_reward = 0
        total_active_aps = 0
        total_energy_efficiency = 0
        total_coordination_links = 0
        total_coordination_power_reduction = 0.0
        steps_completed = 0
        optimizer_stepped = False
        episode_optimizer_updates = 0
        episode_loss_total = 0.0
        episode_grad_norm_total = 0.0

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
            # Time-limit truncation still has a valid next-state value.
            shared_buffer.add(states_all, actions_all, team_reward, next_states_all, terminated, global_state, next_global_state)

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
                gradient_norm = torch.nn.utils.clip_grad_norm_(all_parameters, max_norm=10.0)
                optimizer.step()
                optimizer_updates_total += 1
                optimizer_stepped = True
                episode_optimizer_updates += 1
                episode_loss_total += float(marl_ib_loss.detach().cpu())
                episode_grad_norm_total += float(gradient_norm.detach().cpu())
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
            total_coordination_power_reduction += info.get(
                "coordination_power_reduction_w", 0.0
            )
            steps_completed += 1
            logger.log_step(e + 1, step + 1, info)

            if done:
                break

        # Epsilon decay mỗi episode
        for agent_id in env.agent_ids:
            if agents[agent_id].epsilon > agents[agent_id].epsilon_min:
                agents[agent_id].epsilon = max(
                    agents[agent_id].epsilon_min,
                    agents[agent_id].epsilon * agents[agent_id].epsilon_decay,
                )

        # LR decay theo schedule, only after the first optimizer update.
        if optimizer_stepped:
            scheduler.step()

        train_episode_seconds = time.perf_counter() - episode_started

        denom = max(steps_completed, 1)
        avg_throughput = total_network_throughput / denom
        avg_jfi = total_jfi / denom
        avg_marl_reward = total_marl_reward / denom
        avg_active_aps = total_active_aps / denom
        avg_energy_efficiency = total_energy_efficiency / denom
        avg_coordination_links = total_coordination_links / denom
        avg_coordination_power_reduction = total_coordination_power_reduction / denom

        history_network_throughput.append(avg_throughput)
        history_jfi.append(avg_jfi)
        history_access['train_jfi_time_average'].append(info['jfi_time_average'])
        history_access['train_ap_access_fraction'].append(info['ap_access_fraction'])
        history_access['train_ap_max_idle_steps'].append(info['ap_max_idle_steps'])
        history_train_active_aps.append(avg_active_aps)
        history_train_reward.append(avg_marl_reward)
        history_train_energy_efficiency.append(avg_energy_efficiency)
        history_train_coordination_links.append(avg_coordination_links)
        history_train_coordination_power_reduction.append(
            avg_coordination_power_reduction
        )
        history_train_epsilon.append(agents["ap_0"].epsilon)

        eval_metrics = None
        eval_started = time.perf_counter()
        if (e + 1) % eval_interval == 0 or e + 1 == episodes:
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
            if config.USER_DEMAND_ENABLED:
                write_json_atomic(
                    os.path.join(results_dir, 'user_service', f'episode_{e + 1:04d}.json'),
                    eval_metrics,
                )
            history_eval_throughput.append(eval_metrics["throughput"])
            history_eval_jfi.append(eval_metrics["jfi"])
            for metric in ('jfi_time_average', 'ap_access_fraction', 'ap_max_idle_steps'):
                history_access['eval_' + metric].append(eval_metrics[metric])
            history_eval_active_aps.append(eval_metrics["active_aps"])
            history_eval_energy_efficiency.append(eval_metrics["energy_efficiency"])
            history_eval_team_reward.append(eval_metrics["team_reward"])
            history_eval_coordination_links.append(eval_metrics["coordination_links"])
            history_eval_coordination_power_reduction.append(
                eval_metrics["coordination_power_reduction_w"]
            )
            current_eval_score = eval_score(eval_metrics, num_agents)
            if current_eval_score > best_eval_score:
                best_eval_score = current_eval_score
                best_model_state = {
                    "agents": {
                        agent_id: {
                            name: {
                                key: value.detach().cpu().clone()
                                for key, value in getattr(agents[agent_id], name).state_dict().items()
                            }
                            for name in ("encoder", "q_network")
                        }
                        for agent_id in env.agent_ids
                    },
                    "q_mixer": {
                        key: value.detach().cpu().clone()
                        for key, value in q_mixer.state_dict().items()
                    },
                }
                for agent_id in env.agent_ids:
                    torch.save(agents[agent_id].encoder.state_dict(), os.path.join(models_dir, f"best_ib_encoder_{agent_id}.pth"))
                    torch.save(agents[agent_id].q_network.state_dict(), os.path.join(models_dir, f"best_ib_qmix_{agent_id}_model.pth"))
                torch.save(q_mixer.state_dict(), os.path.join(models_dir, "best_ib_qmix_mixer_model.pth"))
        eval_episode_seconds = (
            time.perf_counter() - eval_started if eval_metrics is not None else 0.0
        )

        logger.log_episode(
            e + 1, avg_throughput, avg_jfi, avg_marl_reward, avg_active_aps,
            avg_energy_efficiency, eval_metrics, epsilon=agents["ap_0"].epsilon,
            coordination_links=avg_coordination_links,
            coordination_power_reduction=avg_coordination_power_reduction,
            episode_seconds=time.perf_counter() - episode_started,
            train_seconds=train_episode_seconds,
            eval_seconds=eval_episode_seconds,
            optimizer_updates=episode_optimizer_updates,
            mean_train_loss=(episode_loss_total / episode_optimizer_updates
                             if episode_optimizer_updates else None),
            mean_gradient_norm=(episode_grad_norm_total / episode_optimizer_updates
                                if episode_optimizer_updates else None),
            replay_size=len(shared_buffer),
            process_rss_mib=_process_rss_mib(),
            cuda_allocated_mib=(
                torch.cuda.memory_allocated(device) / (1024 ** 2)
                if device.type == "cuda" else 0.0
            ),
        )
        current_epsilon = agents["ap_0"].epsilon
        aligned = config.REWARD_OBJECTIVE_MODE == 'throughput_temporal_jfi'
        display_reward = (avg_marl_reward * config.TRAIN_STEPS_PER_EPISODE
                          if aligned else avg_marl_reward)
        reward_label = 'Episode score' if aligned else 'Research-Reward'
        print(f"Ván {e+1:03d}/{episodes} | Tốc độ mạng: {avg_throughput:6.2f} Mbps | JFI: {avg_jfi:.3f} | AP active: {avg_active_aps:.2f}/{num_agents} | {reward_label}: {display_reward:7.3f} | Epsilon: {current_epsilon:.2f}")
        if eval_metrics:
            print(f"  Eval {eval_metrics['mode']} | Tốc độ: {eval_metrics['throughput']:6.2f} Mbps | JFI: {eval_metrics['jfi']:.3f} | AP active: {eval_metrics['active_aps']:.2f}/{num_agents} | Current score: {current_eval_score:.3f} | Best score: {best_eval_score:.3f}")

        if (e + 1) % checkpoint_interval == 0 or e + 1 == episodes:
            write_training_checkpoint(e + 1)
            print(
                f"  Full resume checkpoint saved: episode {e + 1} "
                f"({len(shared_buffer):,} replay transitions)"
            )

    print("\n=== HUẤN LUYỆN XONG! ĐANG ĐÓNG GÓI MÔ HÌNH VÀ GỌI UTILS... ===")

    if best_model_state is not None:
        for agent_id in env.agent_ids:
            agents[agent_id].encoder.load_state_dict(best_model_state["agents"][agent_id]["encoder"])
            agents[agent_id].q_network.load_state_dict(best_model_state["agents"][agent_id]["q_network"])
        q_mixer.load_state_dict(best_model_state["q_mixer"])
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
        **history_access,
        "train_scenarios": history_scenarios,
        "optimizer_updates_total": optimizer_updates_total,
        "protocol_version": config.THREE_STAGE_PROTOCOL_VERSION,
        "access_mode": env.access_mode,
        "coordination_pipeline": env.coordination_pipeline,
        "reward_active_ap_weight": env.active_ap_weight,
        "total_power_budget_w": config.TOTAL_POWER_BUDGET_W,
        "scenario_seed": config.TRAIN_SCENARIO_SEED,
        "steps_per_episode": config.TRAIN_STEPS_PER_EPISODE,
        "eval_episodes_per_checkpoint": eval_episodes,
        "label": (
            f"Hybrid-AI + Water-Filling ({config.HYBRID_ACTION_SIZE} actions)"
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
        "train_coordination_power_reduction_w": history_train_coordination_power_reduction,
        "eval_episodes": history_eval_episodes,
        "eval_throughput": history_eval_throughput,
        "eval_jfi": history_eval_jfi,
        "eval_active_aps": history_eval_active_aps,
        "eval_energy_efficiency": history_eval_energy_efficiency,
        "eval_team_reward": history_eval_team_reward,
        "eval_coordination_links": history_eval_coordination_links,
        "eval_coordination_power_reduction_w": history_eval_coordination_power_reduction,
        "eval_mode": eval_mode,
        "seed": seed,
        "best_eval_score": best_eval_score,
        "last_checkpoint_path": checkpoint_path,
        "manifest_path": manifest_path,
        "models_dir": models_dir,
        "algorithm": "qmix" if use_qmix else "vdn",
        "use_double_dqn": use_double_dqn,
        "ib_beta": float(beta_ib),
        "vib_enabled": bool(beta_ib > 0.0),
        "cooperation_enabled": bool(cooperation_enabled),
        "observation_size": int(env.observation_size),
        "action_size": int(action_size),
    }
