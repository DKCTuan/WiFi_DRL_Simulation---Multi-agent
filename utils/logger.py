# utils/logger.py
import csv
import os

import config

class ExperimentLogger:
    def __init__(self, save_dir="results", append=False):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.filepath = os.path.join(save_dir, "training_logs.csv")
        self.step_filepath = os.path.join(save_dir, "step_logs.csv")
        
        # Resume appends to the existing trace instead of erasing prior work.
        episode_mode = 'a' if append and os.path.exists(self.filepath) else 'w'
        with open(self.filepath, mode=episode_mode, newline='', encoding='utf-8') as f:
            if episode_mode == 'a':
                pass
            else:
                writer = csv.writer(f)
                writer.writerow([
                    "Episode",
                    "Throughput_Mbps",
                    "JFI",
                    "MARL_Reward",
                    "Active_APs",
                    "Energy_Efficiency_Mbps_per_W",
                    "Epsilon",
                    "Eval_Throughput_Mbps",
                    "Eval_JFI",
                    "Eval_Active_APs",
                    "Eval_Energy_Efficiency_Mbps_per_W",
                    "Coordination_Links",
                    "Eval_Coordination_Links",
                    "Eval_Coordination_Requested_Links",
                    "Eval_Coordination_Utility_Delta",
                    "Coordination_Power_Reduction_W",
                    "Eval_Coordination_Power_Reduction_W",
                    "Eval_Team_Reward",
                    "Episode_Seconds",
                    "Train_Seconds",
                    "Eval_Seconds",
                    "Optimizer_Updates",
                    "Mean_Train_Loss",
                    "Mean_Gradient_Norm",
                    "Replay_Size",
                    "Process_RSS_MiB",
                    "CUDA_Allocated_MiB",
                    "Eval_Mode"
                ])
        step_mode = 'a' if append and os.path.exists(self.step_filepath) else 'w'
        with open(self.step_filepath, mode=step_mode, newline='', encoding='utf-8') as f:
            if step_mode != 'a':
                writer = csv.writer(f)
                writer.writerow([
                    "Episode",
                    "Step",
                    "Throughput_Mbps",
                    "JFI",
                    "Active_AP_Count",
                    "Active_AP_IDs",
                    "Coordination_Link_Count",
                    "Coordination_Requested_Link_Count",
                    "Coordination_Utility_Delta",
                    "Cooperation_Links",
                    "Coordination_Power_Reduction_W",
                    *[f"AP{ap_id}_Active" for ap_id in range(config.NUM_APS)],
                    *[f"AP{ap_id}_Throughput_Mbps" for ap_id in range(config.NUM_APS)],
                ])

    def log_episode(
        self, episode, throughput, jfi, reward, active_aps,
        energy_efficiency=None, eval_metrics=None, epsilon=None,
        coordination_links=0.0,
        coordination_power_reduction=0.0,
        episode_seconds=None,
        train_seconds=None,
        eval_seconds=None,
        optimizer_updates=None,
        mean_train_loss=None,
        mean_gradient_norm=None,
        replay_size=None,
        process_rss_mib=None,
        cuda_allocated_mib=None,
    ):
        """Lưu lại thông số của từng ván vào file CSV"""
        eval_metrics = eval_metrics or {}
        with open(self.filepath, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode,
                round(throughput, 2),
                round(jfi, 3),
                round(reward, 6) if config.REWARD_OBJECTIVE_MODE == 'throughput_temporal_jfi' else round(reward, 2),
                round(active_aps, 2),
                round(energy_efficiency, 2) if energy_efficiency is not None else "",
                round(epsilon, 4) if epsilon is not None else "",
                round(eval_metrics["throughput"], 2) if eval_metrics else "",
                round(eval_metrics["jfi"], 3) if eval_metrics else "",
                round(eval_metrics["active_aps"], 2) if eval_metrics else "",
                round(eval_metrics["energy_efficiency"], 2) if eval_metrics else "",
                round(coordination_links, 3),
                round(eval_metrics.get("coordination_links", 0.0), 3) if eval_metrics else "",
                round(eval_metrics.get("coordination_requested_links", 0.0), 3) if eval_metrics else "",
                round(eval_metrics.get("coordination_utility_delta", 0.0), 6) if eval_metrics else "",
                round(coordination_power_reduction, 6),
                round(eval_metrics.get("coordination_power_reduction_w", 0.0), 6) if eval_metrics else "",
                round(eval_metrics.get("team_reward", 0.0), 6) if eval_metrics else "",
                round(episode_seconds, 3) if episode_seconds is not None else "",
                round(train_seconds, 3) if train_seconds is not None else "",
                round(eval_seconds, 3) if eval_seconds is not None else "",
                int(optimizer_updates) if optimizer_updates is not None else "",
                round(mean_train_loss, 6) if mean_train_loss is not None else "",
                round(mean_gradient_norm, 6) if mean_gradient_norm is not None else "",
                int(replay_size) if replay_size is not None else "",
                round(process_rss_mib, 2) if process_rss_mib is not None else "",
                round(cuda_allocated_mib, 2) if cuda_allocated_mib is not None else "",
                eval_metrics.get("mode", "") if eval_metrics else ""
            ])

    def log_step(self, episode, step, info):
        """Luu active AP va JFI theo tung step de debug fairness."""
        active_ids = sorted(info.get("active_ap_ids", []))
        active_set = set(active_ids)
        ap_throughputs = info.get("ap_individual_throughputs", [None] * config.NUM_APS)
        with open(self.step_filepath, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode,
                step,
                round(info["throughput"], 4),
                round(info["jfi"], 6),
                info["active_ap_count"],
                " ".join(str(ap_id) for ap_id in active_ids),
                info.get("coordination_link_count", 0),
                info.get("coordination_requested_link_count", 0),
                round(info.get("coordination_utility_delta", 0.0), 8),
                " ".join(
                    f"{link.get('coordinating_ap', link['helper_ap'])}->{link['serving_ap']}:sta{link['sta_id']}"
                    for link in info.get("cooperation_links", [])
                ),
                round(info.get("coordination_power_reduction_w", 0.0), 6),
                *[1 if ap_id in active_set else 0 for ap_id in range(config.NUM_APS)],
                *[round(ap_throughputs[ap_id], 4) if ap_throughputs[ap_id] is not None else ""
                  for ap_id in range(config.NUM_APS)],
            ])
