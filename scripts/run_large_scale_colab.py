"""
scripts/run_large_scale_colab.py
=================================
Script tối ưu để chạy thực nghiệm quy mô lớn trên Google Colab / GPU:
- Hỗ trợ đa seeds (seeds = 0, 1, 2)
- Hỗ trợ chọn số lượng user K (4, 6, 8, 10, 12, 14)
- Huấn luyện song song Baseline và Proposed Framework
- Tự động xuất số liệu, biểu đồ so sánh chi tiết và chỉ số QoS
"""
import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config
from training.train_loop import train_marl
from training.utils import aggregate_results, set_global_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Chạy thực nghiệm quy mô lớn trên Colab")
    parser.add_argument("--episodes", type=int, default=500, help="Số episodes huấn luyện (khuyến nghị: 500-1200)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Danh sách random seeds")
    parser.add_argument("--k", type=int, default=10, help="Số trạm STA trên mỗi AP (K)")
    parser.add_argument("--output_dir", type=str, default="results/colab_large_scale", help="Thư mục lưu kết quả")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*75}")
    print(f"  CHẠY THỰC NGHIỆM QUY MÔ LỚN: K={args.k} | EPISODES={args.episodes} | SEEDS={args.seeds}")
    print(f"{'='*75}\n")

    # Override số trạm K
    config.NUM_STAS_PER_AP = args.k
    config.AP_LOAD_PROFILE = [args.k] * config.NUM_APS

    baseline_runs = []
    proposed_runs = []

    # 1. Huấn luyện Baseline qua các seeds
    print(">>> GIAI ĐOẠN 1: HUẤN LUYỆN BASELINE (BÀI BÁO GỐC)...")
    config.SERVICE_AWARE_ENABLED = False
    config.SERVICE_WEIGHTED_WF_ENABLED = False
    config.OBS_SIZE = config.BASE_OBS_SIZE

    for seed in args.seeds:
        exp_name = f"colab_K{args.k}_baseline_seed{seed}"
        print(f"  --> Chạy: {exp_name}")
        res = train_marl(
            experiment_name=exp_name,
            use_water_filling=True,
            action_size=config.HYBRID_ACTION_SIZE,
            episodes=args.episodes,
            seed=seed,
            eval_mode="fixed",
            num_stas=args.k,
            cooperation_enabled=False,
        )
        baseline_runs.append(res)

    # 2. Huấn luyện Proposed Framework qua các seeds
    print("\n>>> GIAI ĐOẠN 2: HUẤN LUYỆN PROPOSED FRAMEWORK (ĐỀ XUẤT)...")
    config.SERVICE_AWARE_ENABLED = True
    config.SERVICE_WEIGHTED_WF_ENABLED = True
    config.OBS_SIZE = config.BASE_OBS_SIZE + config.NUM_SERVICE_CLASSES

    for seed in args.seeds:
        exp_name = f"colab_K{args.k}_proposed_seed{seed}"
        print(f"  --> Chạy: {exp_name}")
        res = train_marl(
            experiment_name=exp_name,
            use_water_filling=True,
            action_size=config.HYBRID_ACTION_SIZE,
            episodes=args.episodes,
            seed=seed,
            eval_mode="fixed",
            num_stas=args.k,
            cooperation_enabled=True,
        )
        proposed_runs.append(res)

    # 3. Tổng hợp số liệu thống kê (Mean & Std qua các seeds)
    agg_baseline = aggregate_results(baseline_runs)
    agg_proposed = aggregate_results(proposed_runs)

    ep_axis = np.arange(1, args.episodes + 1)

    # 4. Vẽ biểu đồ so sánh xuất sắc chuẩn ấn phẩm (Publication Quality)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Đồ thị Throughput
    axes[0].plot(ep_axis, agg_baseline["throughput_mean"], label="Baseline (Original Paper)", color="#E67E22", linewidth=2)
    axes[0].fill_between(
        ep_axis,
        agg_baseline["throughput_mean"] - agg_baseline["throughput_std"],
        agg_baseline["throughput_mean"] + agg_baseline["throughput_std"],
        color="#E67E22", alpha=0.15
    )
    axes[0].plot(ep_axis, agg_proposed["throughput_mean"], label="Proposed Framework", color="#2980B9", linewidth=2)
    axes[0].fill_between(
        ep_axis,
        agg_proposed["throughput_mean"] - agg_proposed["throughput_std"],
        agg_proposed["throughput_mean"] + agg_proposed["throughput_std"],
        color="#2980B9", alpha=0.15
    )
    axes[0].set_xlabel("Episode", fontsize=12)
    axes[0].set_ylabel("Network Throughput (Mbps)", fontsize=12)
    axes[0].set_title(f"Throughput Comparison (K={args.k})", fontsize=13, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=11)

    # Đồ thị Fairness JFI
    axes[1].plot(ep_axis, agg_baseline["jfi_mean"], label="Baseline (Original Paper)", color="#E67E22", linewidth=2)
    axes[1].fill_between(
        ep_axis,
        agg_baseline["jfi_mean"] - agg_baseline["jfi_std"],
        agg_baseline["jfi_mean"] + agg_baseline["jfi_std"],
        color="#E67E22", alpha=0.15
    )
    axes[1].plot(ep_axis, agg_proposed["jfi_mean"], label="Proposed Framework", color="#2980B9", linewidth=2)
    axes[1].fill_between(
        ep_axis,
        agg_proposed["jfi_mean"] - agg_proposed["jfi_std"],
        agg_proposed["jfi_mean"] + agg_proposed["jfi_std"],
        color="#2980B9", alpha=0.15
    )
    axes[1].set_xlabel("Episode", fontsize=12)
    axes[1].set_ylabel("Jain's Fairness Index (JFI)", fontsize=12)
    axes[1].set_title(f"Fairness (JFI) Comparison (K={args.k})", fontsize=13, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    chart_path = os.path.join(args.output_dir, f"comparison_K{args.k}_large_scale.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()

    # In kết quả định lượng
    b_thr_final = np.mean(agg_baseline["throughput_mean"][-50:])
    p_thr_final = np.mean(agg_proposed["throughput_mean"][-50:])
    b_jfi_final = np.mean(agg_baseline["jfi_mean"][-50:])
    p_jfi_final = np.mean(agg_proposed["jfi_mean"][-50:])

    thr_gain = ((p_thr_final - b_thr_final) / b_thr_final) * 100
    jfi_gain = ((p_jfi_final - b_jfi_final) / b_jfi_final) * 100

    print(f"\n{'='*70}")
    print(f"  TỔNG KẾT HIỆU NĂNG (TRUNG BÌNH 50 EPISODES CUỐI):")
    print(f"  • Throughput: Baseline = {b_thr_final:.2f} Mbps | Proposed = {p_thr_final:.2f} Mbps (Tăng: +{thr_gain:.2f}%)")
    print(f"  • Fairness:   Baseline = {b_jfi_final:.3f}      | Proposed = {p_jfi_final:.3f}      (Tăng: +{jfi_gain:.2f}%)")
    print(f"  • Biểu đồ đã lưu tại: {chart_path}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
