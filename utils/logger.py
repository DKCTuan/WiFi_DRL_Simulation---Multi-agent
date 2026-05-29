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
            writer.writerow(["Episode", "Throughput_Mbps", "JFI", "MARL_Reward"])

    def log_episode(self, episode, throughput, jfi, reward):
        """Lưu lại thông số của từng ván vào file CSV"""
        with open(self.filepath, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([episode, round(throughput, 2), round(jfi, 3), round(reward, 2)])