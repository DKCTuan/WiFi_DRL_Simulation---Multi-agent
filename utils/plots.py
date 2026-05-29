# utils/plots.py
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_learning_curve(history_throughput, history_jfi, save_dir="results/plots"):
    """Hàm tiện ích giúp tách biệt logic vẽ đồ thị ra khỏi file main.py"""
    # Tự động tạo thư mục lưu trữ nếu chưa có để tránh lỗi Python
    os.makedirs(save_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    
    # 1. Vẽ đồ thị Throughput
    ax1.plot(history_throughput, color='#1f77b4', alpha=0.4, label='Thực tế ván')
    if len(history_throughput) >= 10:
        ma_thr = np.convolve(history_throughput, np.ones(10)/10, mode='valid')
        ax1.plot(range(9, len(history_throughput)), ma_thr, color='red', linestyle='-', linewidth=2, label='Xu hướng (MA-10)')
    ax1.set_title("Đường cong hội tụ của Hệ thống Phân tán Multi-Agent WiFi", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Thông lượng mạng tổng (Mbps)", fontsize=10)
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # 2. Vẽ đồ thị JFI
    ax2.plot(history_jfi, color='#2ca02c', alpha=0.4, label='Độ công bằng')
    if len(history_jfi) >= 10:
        ma_jfi = np.convolve(history_jfi, np.ones(10)/10, mode='valid')
        ax2.plot(range(9, len(history_jfi)), ma_jfi, color='darkgreen', linestyle='-', linewidth=2, label='Xu hướng (MA-10)')
    ax2.set_xlabel("Số ván huấn luyện (Episode)", fontsize=11)
    ax2.set_ylabel("Chỉ số công bằng JFI", fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "marl_learning_curve.png")
    plt.savefig(save_path, dpi=300)
    print(f"📈 Đã xuất đồ thị phân tích kép chất lượng cao vào: {save_path}")