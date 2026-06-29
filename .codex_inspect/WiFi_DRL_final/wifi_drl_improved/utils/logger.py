# utils/logger.py
import csv
import os

class ExperimentLogger:
    def __init__(self, save_dir="results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.filepath = os.path.join(save_dir, "training_logs.csv")
        
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
                "Eval_Throughput_Mbps",
                "Eval_JFI",
                "Eval_Active_APs",
                "Eval_Energy_Efficiency_Mbps_per_W",
                "Eval_Mode"
            ])

    def log_episode(self, episode, throughput, jfi, reward, active_aps, energy_efficiency=None, eval_metrics=None):
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
                round(eval_metrics["throughput"], 2) if eval_metrics else "",
                round(eval_metrics["jfi"], 3) if eval_metrics else "",
                round(eval_metrics["active_aps"], 2) if eval_metrics else "",
                round(eval_metrics["energy_efficiency"], 2) if eval_metrics else "",
                eval_metrics.get("mode", "") if eval_metrics else ""
            ])
