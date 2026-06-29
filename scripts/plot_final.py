"""
scripts/plot_final.py
=====================
Vẽ 4 hình cuối từ data .txt đã collect từ v1–v4.

Cách dùng:
  1. Download tất cả file .txt từ results/data/ của 4 Kaggle version
  2. Đặt vào cùng 1 thư mục DATA_DIR (mặc định: results/data/)
  3. Chạy: python scripts/plot_final.py

Output: results/figures/
  ├── fig1_training_throughput.png
  ├── fig2_training_jfi.png
  ├── fig3_eval_throughput_vs_K.png
  └── fig4_eval_jfi_vs_K.png
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import config as _cfg

DATA_DIR    = "results/data"
FIGURES_DIR = "results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─── Helper đọc .txt ──────────────────────────────────────────────────────────
def load_txt(path):
    """Đọc file .txt → dict {col_name: np.array}."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file: {path}\n"
                                f"Hãy chắc chắn đã copy đủ data từ các Kaggle version vào {DATA_DIR}/")
    headers = None
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("# ") and headers is None:
                continue           # dòng tiêu đề đầu (mô tả)
            if line.startswith("# "):
                headers = line[2:].split("\t")
                continue
            rows.append([float(x) for x in line.split("\t")])
    data = np.array(rows)
    return {h: data[:, i] for i, h in enumerate(headers)}

def smooth(arr, w=25):
    arr = np.asarray(arr, dtype=np.float32)
    if len(arr) < w:
        return arr, np.arange(1, len(arr) + 1)
    ma = np.convolve(arr, np.ones(w) / w, mode="valid")
    x  = np.arange(w, len(arr) + 1)
    return ma, x

# ─── Style (in đen trắng phân biệt được) ─────────────────────────────────────
#  Training: 4 đường = (K=10 hybrid, K=10 full_ai, K=12 hybrid, K=12 full_ai)
TRAIN_STYLE = {
    ("K10", "Hybrid_WF"): dict(color="black",   marker="o",  ls="-",  label=f"Hybrid-AI+WF K=10 ({_cfg.HYBRID_ACTION_SIZE} act)"),
    ("K10", "Full_AI")  : dict(color="black",   marker="s",  ls="--", label=f"Full-AI K=10 ({_cfg.FULL_AI_ACTION_SIZE} act)"),
    ("K12", "Hybrid_WF"): dict(color="dimgray", marker="^",  ls="-",  label=f"Hybrid-AI+WF K=12 ({_cfg.HYBRID_ACTION_SIZE} act)"),
    ("K12", "Full_AI")  : dict(color="dimgray", marker="D",  ls="--", label=f"Full-AI K=12 ({_cfg.FULL_AI_ACTION_SIZE} act)"),
}

#  Eval: 2 đường × trục X là K
EVAL_STYLE = {
    "Hybrid_WF": dict(color="black",   marker="o",  ls="-",  label=f"Hybrid-AI+WF ({_cfg.HYBRID_ACTION_SIZE} actions)"),
    "Full_AI"  : dict(color="dimgray", marker="s",  ls="--", label=f"Full-AI ({_cfg.FULL_AI_ACTION_SIZE} actions)"),
}

K_TRAIN = [10, 12]
K_EVAL  = [4, 6, 8, 10, 12, 14]
W       = _cfg.PLOT_SMOOTHING_WINDOW

# ══════════════════════════════════════════════════════════════════════════════
# HÌNH 1 — Training Throughput
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/4] Vẽ training_throughput ...")
fig, ax = plt.subplots(figsize=(9, 5))

for k in K_TRAIN:
    for label_key in ["Hybrid_WF", "Full_AI"]:
        fname = os.path.join(DATA_DIR, f"training_K{k}_{label_key}.txt")
        d = load_txt(fname)
        ma, x = smooth(d["Throughput_mean"], W)
        st = TRAIN_STYLE[(f"K{k}", label_key)]
        ax.plot(x, ma,
                color=st["color"], marker=st["marker"], linestyle=st["ls"],
                markevery=50, markersize=6, linewidth=1.8,
                label=st["label"])

ax.set_xlabel("Episode", fontsize=12)
ax.set_ylabel("Throughput (Mbps)", fontsize=12)
ax.set_title("Training Comparison — Throughput (6 APs)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
p = os.path.join(FIGURES_DIR, "fig1_training_throughput.png")
fig.savefig(p, dpi=200)
plt.close(fig)
print(f"  → Saved: {p}")

# ══════════════════════════════════════════════════════════════════════════════
# HÌNH 2 — Training JFI
# ══════════════════════════════════════════════════════════════════════════════
print("[2/4] Vẽ training_jfi ...")
fig, ax = plt.subplots(figsize=(9, 5))

for k in K_TRAIN:
    for label_key in ["Hybrid_WF", "Full_AI"]:
        fname = os.path.join(DATA_DIR, f"training_K{k}_{label_key}.txt")
        d = load_txt(fname)
        ma, x = smooth(d["JFI_mean"], W)
        st = TRAIN_STYLE[(f"K{k}", label_key)]
        ax.plot(x, ma,
                color=st["color"], marker=st["marker"], linestyle=st["ls"],
                markevery=50, markersize=6, linewidth=1.8,
                label=st["label"])

ax.set_xlabel("Episode", fontsize=12)
ax.set_ylabel("JFI", fontsize=12)
ax.set_title("Training Comparison — Jain's Fairness Index (6 APs)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
p = os.path.join(FIGURES_DIR, "fig2_training_jfi.png")
fig.savefig(p, dpi=200)
plt.close(fig)
print(f"  → Saved: {p}")

# ══════════════════════════════════════════════════════════════════════════════
# HÌNH 3 — Eval Throughput vs K
# ══════════════════════════════════════════════════════════════════════════════
print("[3/4] Vẽ eval_throughput_vs_K ...")
fig, ax = plt.subplots(figsize=(9, 5))

for label_key in ["Hybrid_WF", "Full_AI"]:
    thrs, stds = [], []
    for k in K_EVAL:
        fname = os.path.join(DATA_DIR, f"eval_summary_K{k}_{label_key}.txt")
        d = load_txt(fname)
        thrs.append(float(d["Throughput_mean"][0]))
        stds.append(float(d["Throughput_std"][0]))
    st = EVAL_STYLE[label_key]
    ax.errorbar(K_EVAL, thrs, yerr=stds,
                color=st["color"], marker=st["marker"], linestyle=st["ls"],
                markersize=8, linewidth=1.8, capsize=4,
                label=st["label"])

ax.set_xlabel("Clients per AP (K)", fontsize=12)
ax.set_ylabel("Throughput (Mbps)", fontsize=12)
ax.set_title("Evaluation — Throughput vs K (6 APs)", fontsize=13)
ax.set_xticks(K_EVAL)
ax.legend(fontsize=10)
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
p = os.path.join(FIGURES_DIR, "fig3_eval_throughput_vs_K.png")
fig.savefig(p, dpi=200)
plt.close(fig)
print(f"  → Saved: {p}")

# ══════════════════════════════════════════════════════════════════════════════
# HÌNH 4 — Eval JFI vs K
# ══════════════════════════════════════════════════════════════════════════════
print("[4/4] Vẽ eval_jfi_vs_K ...")
fig, ax = plt.subplots(figsize=(9, 5))

for label_key in ["Hybrid_WF", "Full_AI"]:
    jfis, stds = [], []
    for k in K_EVAL:
        fname = os.path.join(DATA_DIR, f"eval_summary_K{k}_{label_key}.txt")
        d = load_txt(fname)
        jfis.append(float(d["JFI_mean"][0]))
        stds.append(float(d["JFI_std"][0]))
    st = EVAL_STYLE[label_key]
    ax.errorbar(K_EVAL, jfis, yerr=stds,
                color=st["color"], marker=st["marker"], linestyle=st["ls"],
                markersize=8, linewidth=1.8, capsize=4,
                label=st["label"])

ax.set_xlabel("Clients per AP (K)", fontsize=12)
ax.set_ylabel("JFI", fontsize=12)
ax.set_title("Evaluation — Jain's Fairness Index vs K (6 APs)", fontsize=13)
ax.set_xticks(K_EVAL)
ax.legend(fontsize=10)
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
p = os.path.join(FIGURES_DIR, "fig4_eval_jfi_vs_K.png")
fig.savefig(p, dpi=200)
plt.close(fig)
print(f"  → Saved: {p}")

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n✓ Hoàn tất! 4 hình đã lưu tại: {FIGURES_DIR}/")
