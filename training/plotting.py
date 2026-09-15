# training/plotting.py
"""Vẽ các biểu đồ so sánh multi-seed (training curve, eval curve, publication
figure) + in bảng tổng kết cuối cùng."""
import os

import numpy as np
import matplotlib.pyplot as plt

import config
from training.utils import moving_average, best_so_far


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


def print_summary_table(agg_full_ai, agg_hybrid, n_last=10):
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
