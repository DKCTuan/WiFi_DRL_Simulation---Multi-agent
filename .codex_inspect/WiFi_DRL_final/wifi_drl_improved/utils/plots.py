import os

import matplotlib.pyplot as plt
import numpy as np


def _moving_average(values, window):
    if len(values) < window:
        return None
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_learning_curve(
    history_throughput,
    history_jfi,
    save_dir="results/plots",
    eval_episodes=None,
    eval_throughput=None,
    eval_jfi=None,
    eval_active_aps=None,
    eval_energy_efficiency=None,
    num_agents=None,
):
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax1.plot(history_throughput, color="#1f77b4", alpha=0.35, label="Training episode")
    ma_thr = _moving_average(history_throughput, 25)
    if ma_thr is not None:
        ax1.plot(range(24, len(history_throughput)), ma_thr, color="red", linewidth=2, label="MA-25")
    ax1.set_title("MARL WiFi training curve", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Throughput (Mbps)", fontsize=10)
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.plot(history_jfi, color="#2ca02c", alpha=0.35, label="Training episode")
    ma_jfi = _moving_average(history_jfi, 25)
    if ma_jfi is not None:
        ax2.plot(range(24, len(history_jfi)), ma_jfi, color="darkgreen", linewidth=2, label="MA-25")
    ax2.set_xlabel("Episode", fontsize=11)
    ax2.set_ylabel("JFI", fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    train_save_path = os.path.join(save_dir, "marl_learning_curve.png")
    plt.savefig(train_save_path, dpi=300)
    plt.close(fig)
    print(f"Saved training curve to: {train_save_path}")

    if not eval_episodes:
        return

    has_energy_efficiency = eval_energy_efficiency is not None and len(eval_energy_efficiency) == len(eval_episodes)
    if has_energy_efficiency:
        fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
        ax1, ax2, ax3, ax4 = axes
    else:
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        ax1, ax2, ax3 = axes

    ax1.plot(eval_episodes, eval_throughput, color="#1f77b4", marker="o", linewidth=2, label="Eval throughput")
    ax1.set_ylabel("Throughput (Mbps)", fontsize=10)
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.plot(eval_episodes, eval_jfi, color="#2ca02c", marker="o", linewidth=2, label="Eval JFI")
    ax2.set_ylabel("JFI", fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle=":", alpha=0.6)

    ax3.plot(eval_episodes, eval_active_aps, color="#9467bd", marker="o", linewidth=2, label="Eval active APs")
    if num_agents is not None:
        ax3.set_ylim(0, num_agents + 0.5)
    ax3.set_xlabel("Episode", fontsize=11)
    ax3.set_ylabel("Active APs", fontsize=10)
    ax3.legend(loc="lower right")
    ax3.grid(True, linestyle=":", alpha=0.6)

    if has_energy_efficiency:
        ax4.plot(eval_episodes, eval_energy_efficiency, color="#d62728", marker="o", linewidth=2, label="Eval energy efficiency")
        ax4.set_xlabel("Episode", fontsize=11)
        ax4.set_ylabel("Mbps/W", fontsize=10)
        ax4.legend(loc="lower right")
        ax4.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    eval_save_path = os.path.join(save_dir, "eval_learning_curve.png")
    plt.savefig(eval_save_path, dpi=300)
    plt.close(fig)
    print(f"Saved fixed-eval curve to: {eval_save_path}")


def plot_comparison_curve(histories, save_dir="results/comparison_plots", window=25):
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    for history in histories:
        label = history["label"]
        throughput = history["throughput"]
        jfi = history["jfi"]
        episodes = range(1, len(throughput) + 1)

        ma_thr = _moving_average(throughput, window)
        ma_jfi = _moving_average(jfi, window)

        ax1.plot(episodes, throughput, alpha=0.18)
        if ma_thr is not None:
            ax1.plot(range(window, len(throughput) + 1), ma_thr, linewidth=2, label=f"{label} MA-{window}")

        ax2.plot(episodes, jfi, alpha=0.18)
        if ma_jfi is not None:
            ax2.plot(range(window, len(jfi) + 1), ma_jfi, linewidth=2, label=f"{label} MA-{window}")

    ax1.set_title("Power allocation comparison", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Throughput (Mbps)", fontsize=10)
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.set_xlabel("Episode", fontsize=11)
    ax2.set_ylabel("JFI", fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "water_filling_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved water-filling comparison to: {save_path}")
