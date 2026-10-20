"""
scripts/run_proposed_vs_baseline.py
====================================
Script thực nghiệm so sánh đối đầu trực tiếp:
1. Baseline (Bài báo gốc): Hybrid QMIX-IB + Water-Filling thông thường
2. Proposed Framework: Service-Aware + Service-Weighted WF + Mean-Field AP Cooperation
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config
from training.train_loop import train_marl
from training.utils import set_global_seed

def run_comparison(episodes=100, seed=42, output_dir="results/comparison"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"  THỰC NGHIỆM ĐỐI SÁNH: BASELINE vs PROPOSED (EPISODES={episodes}, SEED={seed})")
    print(f"{'='*70}\n")

    # 1. Chạy Baseline (Bài báo gốc)
    print(">>> [1/2] Đang huấn luyện BASELINE (Bài báo gốc)...")
    config.SERVICE_AWARE_ENABLED = False
    config.SERVICE_WEIGHTED_WF_ENABLED = False
    config.OBS_SIZE = config.BASE_OBS_SIZE

    baseline_res = train_marl(
        experiment_name="baseline_hybrid_wf",
        use_water_filling=True,
        action_size=config.HYBRID_ACTION_SIZE,
        episodes=episodes,
        seed=seed,
        eval_mode="fixed",
        cooperation_enabled=False,
    )

    # 2. Chạy Proposed Framework
    print("\n>>> [2/2] Đang huấn luyện PROPOSED FRAMEWORK (Đề xuất)...")
    config.SERVICE_AWARE_ENABLED = True
    config.SERVICE_WEIGHTED_WF_ENABLED = True
    config.OBS_SIZE = config.BASE_OBS_SIZE + config.NUM_SERVICE_CLASSES

    proposed_res = train_marl(
        experiment_name="proposed_framework",
        use_water_filling=True,
        action_size=config.HYBRID_ACTION_SIZE,
        episodes=episodes,
        seed=seed,
        eval_mode="fixed",
        cooperation_enabled=True,
    )

    # 3. Vẽ biểu đồ so sánh publication-ready
    ep_axis = np.arange(1, episodes + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Throughput
    axes[0].plot(ep_axis, baseline_res["throughput"], label="Baseline (Original Paper)", color="orange", linewidth=1.5, alpha=0.8)
    axes[0].plot(ep_axis, proposed_res["throughput"], label="Proposed (Service-Aware + WF + Coop)", color="blue", linewidth=1.5)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Network Throughput (Mbps)")
    axes[0].set_title("Training Throughput Comparison")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # Subplot 2: JFI (Fairness)
    axes[1].plot(ep_axis, baseline_res["jfi"], label="Baseline (Original Paper)", color="orange", linewidth=1.5, alpha=0.8)
    axes[1].plot(ep_axis, proposed_res["jfi"], label="Proposed (Service-Aware + WF + Coop)", color="blue", linewidth=1.5)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Jain's Fairness Index (JFI)")
    axes[1].set_title("Training Fairness (JFI) Comparison")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "baseline_vs_proposed_training.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[Xong] Đã xuất biểu đồ so sánh tại: {plot_path}")

    # In bảng tóm tắt
    b_thr = np.mean(baseline_res["throughput"][-20:])
    p_thr = np.mean(proposed_res["throughput"][-20:])
    b_jfi = np.mean(baseline_res["jfi"][-20:])
    p_jfi = np.mean(proposed_res["jfi"][-20:])

    print(f"\n{'='*60}")
    print("  KẾT QUẢ TRUNG BÌNH 20 EPISODES CUỐI:")
    print(f"  Throughput: Baseline = {b_thr:.2f} Mbps | Proposed = {p_thr:.2f} Mbps")
    print(f"  JFI:        Baseline = {b_jfi:.3f}      | Proposed = {p_jfi:.3f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    run_comparison(episodes=episodes)
