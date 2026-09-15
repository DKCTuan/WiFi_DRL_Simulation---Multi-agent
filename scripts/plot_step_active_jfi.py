import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def moving_average(values, window):
    if window <= 1 or len(values) < window:
        return values, np.arange(len(values))
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid"), np.arange(window - 1, len(values))


def main():
    parser = argparse.ArgumentParser(description="Plot step-level active APs vs JFI.")
    parser.add_argument(
        "--csv",
        default=os.path.join("results", "final_full_ai_seed0", "step_logs.csv"),
        help="Path to step_logs.csv",
    )
    parser.add_argument("--window", type=int, default=25, help="Moving-average window for JFI")
    parser.add_argument("--out", default=None, help="Output PNG path")
    args = parser.parse_args()

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {args.csv}")

    columns = rows[0].keys()
    jfi = np.array([float(row["JFI"]) for row in rows], dtype=float)
    active_count = np.array([float(row["Active_AP_Count"]) for row in rows], dtype=float)
    active_cols = [col for col in columns if col.startswith("AP") and col.endswith("_Active")]
    active_matrix = np.array(
        [[float(row[col]) for row in rows] for col in active_cols],
        dtype=float,
    )

    x = np.arange(len(rows))
    jfi_ma, jfi_x = moving_average(jfi, args.window)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.2, 1.4]},
    )

    axes[0].plot(x, jfi, color="#7aa6c2", alpha=0.25, linewidth=0.8, label="JFI per step")
    axes[0].plot(jfi_x, jfi_ma, color="#d62728", linewidth=1.8, label=f"JFI MA-{args.window}")
    axes[0].set_ylabel("JFI")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.25)

    axes[1].plot(x, active_count, color="#2ca02c", linewidth=1.0)
    axes[1].set_ylabel("Active APs")
    axes[1].set_ylim(-0.2, 6.2)
    axes[1].grid(alpha=0.25)

    axes[2].imshow(active_matrix, aspect="auto", interpolation="nearest", cmap="Greens", vmin=0, vmax=1)
    axes[2].set_yticks(np.arange(len(active_cols)))
    axes[2].set_yticklabels([col.replace("_Active", "") for col in active_cols])
    axes[2].set_xlabel("Training step")
    axes[2].set_ylabel("AP active")

    fig.suptitle("Step-level JFI vs active AP state")
    fig.tight_layout()

    out_path = args.out
    if out_path is None:
        base_dir = os.path.dirname(args.csv) or "."
        out_path = os.path.join(base_dir, "step_active_jfi.png")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
