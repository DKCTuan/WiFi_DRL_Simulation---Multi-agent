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
    print(f"Saved training curve to: {train_save_path}")

    if not eval_episodes:
        return

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

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

    plt.tight_layout()
    eval_save_path = os.path.join(save_dir, "eval_learning_curve.png")
    plt.savefig(eval_save_path, dpi=300)
    print(f"Saved fixed-eval curve to: {eval_save_path}")
