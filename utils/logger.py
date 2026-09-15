# utils/logger.py
import csv
import os

import config

class ExperimentLogger:
    def __init__(self, save_dir="results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.filepath = os.path.join(save_dir, "training_logs.csv")
        self.step_filepath = os.path.join(save_dir, "step_logs.csv")
        
        # Khởi tạo file và ghi hàng tiêu đề (Header)
        with open(self.filepath, mode='w', newline='', encoding='utf-8') as f:
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
                "Eval_Mode"
            ])
        with open(self.step_filepath, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Episode",
                "Step",
                "Throughput_Mbps",
                "JFI",
                "Active_AP_Count",
                "Active_AP_IDs",
                "Coordination_Link_Count",
                "Cooperation_Links",
                *[f"AP{ap_id}_Active" for ap_id in range(config.NUM_APS)],
                *[f"AP{ap_id}_Throughput_Mbps" for ap_id in range(config.NUM_APS)],
            ])

    def log_episode(
        self, episode, throughput, jfi, reward, active_aps,
        energy_efficiency=None, eval_metrics=None, epsilon=None,
        coordination_links=0.0,
    ):
        """Lưu lại thông số của từng ván vào file CSV"""
        eval_metrics = eval_metrics or {}
        with open(self.filepath, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode,
                round(throughput, 2),
                round(jfi, 3),
                round(reward, 2),
                round(active_aps, 2),
                round(energy_efficiency, 2) if energy_efficiency is not None else "",
                round(epsilon, 4) if epsilon is not None else "",
                round(eval_metrics["throughput"], 2) if eval_metrics else "",
                round(eval_metrics["jfi"], 3) if eval_metrics else "",
                round(eval_metrics["active_aps"], 2) if eval_metrics else "",
                round(eval_metrics["energy_efficiency"], 2) if eval_metrics else "",
                round(coordination_links, 3),
                round(eval_metrics.get("coordination_links", 0.0), 3) if eval_metrics else "",
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
                " ".join(
                    f"{link['helper_ap']}->{link['serving_ap']}:sta{link['sta_id']}"
                    for link in info.get("cooperation_links", [])
                ),
                *[1 if ap_id in active_set else 0 for ap_id in range(config.NUM_APS)],
                *[round(ap_throughputs[ap_id], 4) if ap_throughputs[ap_id] is not None else ""
                  for ap_id in range(config.NUM_APS)],
            ])
