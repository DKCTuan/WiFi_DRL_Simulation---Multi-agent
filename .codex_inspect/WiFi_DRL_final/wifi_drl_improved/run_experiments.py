"""
run_experiments.py
==================
Script tự động chạy toàn bộ thí nghiệm:

  1. Training plot  : K = 10, 12  (2 mô hình × 3 seeds mỗi K)
  2. Evaluation plot: K = 4,6,8,10,12,14  (train + eval mỗi K)

Cách dùng trên Kaggle:
    python run_experiments.py

Kết quả lưu vào:
    results/
    ├── training_data/      ← .txt dùng để vẽ hình training
    ├── eval_data/          ← .txt dùng để vẽ hình evaluation
    └── figures/            ← các hình .png
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── patch config trước khi import bất kỳ module nào khác ──────────────────────
import config as _cfg

def set_k(k):
    """Override AP_LOAD_PROFILE toàn cục theo K."""
    _cfg.NUM_STAS_PER_AP = k
    _cfg.AP_LOAD_PROFILE = [k] * _cfg.NUM_APS

# ── import sau khi config đã được load ────────────────────────────────────────
from main import train_marl, aggregate_results

# ══════════════════════════════════════════════════════════════════════════════
# CẤU HÌNH THÍ NGHIỆM
# ══════════════════════════════════════════════════════════════════════════════
SEEDS            = _cfg.FINAL_SEEDS          # [0, 1, 2]
K_TRAIN          = [10, 12]                  # training plot
K_EVAL           = [4, 6, 8, 10, 12, 14]    # evaluation plot

RESULTS_DIR      = "results"
TRAIN_DATA_DIR   = os.path.join(RESULTS_DIR, "training_data")
EVAL_DATA_DIR    = os.path.join(RESULTS_DIR, "eval_data")
FIGURES_DIR      = os.path.join(RESULTS_DIR, "figures")

for d in [TRAIN_DATA_DIR, EVAL_DATA_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Style công khai (giống hình tham khảo) ────────────────────────────────────
# 4 đường → 4 marker + linestyle khác nhau để in đen trắng phân biệt được
STYLE = {
    # (K, model)       : (color,  marker, linestyle,  label_suffix)
    (10, "hybrid")     : ("black",   "o",  "-",   ""),
    (10, "full_ai")    : ("black",   "s",  "--",  ""),
    (12, "hybrid")     : ("dimgray", "^",  "-",   ""),
    (12, "full_ai")    : ("dimgray", "D",  "--",  ""),
}

EVAL_STYLE = {
    # K : (color, marker, linestyle)
    4  : ("black",   "o",  "-"),
    6  : ("black",   "s",  "--"),
    8  : ("black",   "^",  "-."),
    10 : ("black",   "D",  ":"),
    12 : ("dimgray", "v",  "-"),
    14 : ("dimgray", "P",  "--"),
}

# ══════════════════════════════════════════════════════════════════════════════
# HELPER: lưu .txt
# ══════════════════════════════════════════════════════════════════════════════
def save_txt(path, header, **arrays):
    """Lưu dict arrays ra file .txt dạng cột."""
    cols = list(arrays.items())
    n = len(cols[0][1])
    with open(path, "w") as f:
        f.write("# " + header + "\n")
        f.write("# " + "\t".join(k for k, _ in cols) + "\n")
        for i in range(n):
            row = "\t".join(f"{v[i]:.6f}" for _, v in cols)
            f.write(row + "\n")
    print(f"  [TXT] {path}")


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 1: TRAINING EXPERIMENTS  (K = 10, 12)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("BƯỚC 1: TRAINING EXPERIMENTS  (K = 10, 12)")
print("="*70)

# Lưu kết quả aggregate theo (K, model)
train_agg = {}   # key: (k, "hybrid") hoặc (k, "full_ai")

for k in K_TRAIN:
    set_k(k)
    print(f"\n── K={k} ──────────────────────────────")

    for model_key, use_wf, action_size in [
        ("hybrid",  True,  _cfg.HYBRID_ACTION_SIZE),
        ("full_ai", False, _cfg.FULL_AI_ACTION_SIZE),
    ]:
        runs = []
        for seed in SEEDS:
            exp_name = f"K{k}_{model_key}_seed{seed}"
            print(f"  Training {exp_name} ...")
            result = train_marl(
                experiment_name=exp_name,
                use_water_filling=use_wf,
                action_size=action_size,
                episodes=_cfg.TRAIN_EPISODES,
                seed=seed,
                eval_mode="fixed",
            )
            runs.append(result)

        agg = aggregate_results(runs)
        train_agg[(k, model_key)] = agg

        # Lưu .txt
        ep = np.arange(1, len(agg["throughput_mean"]) + 1)
        label = "Hybrid_WF" if model_key == "hybrid" else "Full_AI"
        save_txt(
            os.path.join(TRAIN_DATA_DIR, f"training_K{k}_{label}.txt"),
            header=f"Training K={k} {label} | seeds={SEEDS}",
            Episode=ep,
            Throughput_mean=agg["throughput_mean"],
            Throughput_std=agg["throughput_std"],
            JFI_mean=agg["jfi_mean"],
            JFI_std=agg["jfi_std"],
        )

# ── Vẽ hình training ──────────────────────────────────────────────────────────
def smooth(arr, w=25):
    arr = np.array(arr, dtype=np.float32)
    if len(arr) < w:
        return arr, np.arange(len(arr)) + 1
    ma = np.convolve(arr, np.ones(w) / w, mode="valid")
    x  = np.arange(w, len(arr) + 1)
    return ma, x

def make_label(k, model_key):
    if model_key == "hybrid":
        return f"Hybrid-AI+WF ({_cfg.HYBRID_ACTION_SIZE} actions) K={k}"
    return f"Full-AI ({_cfg.FULL_AI_ACTION_SIZE} actions) K={k}"

# Hình 1: Throughput
fig1, ax1 = plt.subplots(figsize=(9, 5))
for k in K_TRAIN:
    for model_key in ["hybrid", "full_ai"]:
        agg = train_agg[(k, model_key)]
        ma, x = smooth(agg["throughput_mean"], _cfg.PLOT_SMOOTHING_WINDOW)
        c, mk, ls, _ = STYLE[(k, model_key)]
        ax1.plot(x, ma, color=c, marker=mk, linestyle=ls,
                 markevery=50, markersize=6, linewidth=1.8,
                 label=make_label(k, model_key))

ax1.set_xlabel("Episode", fontsize=12)
ax1.set_ylabel("Throughput (Mbps)", fontsize=12)
ax1.set_title("Training — Throughput (6 APs)", fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
p = os.path.join(FIGURES_DIR, "training_throughput.png")
fig1.savefig(p, dpi=200)
plt.close(fig1)
print(f"\n[FIG] {p}")

# Hình 2: JFI
fig2, ax2 = plt.subplots(figsize=(9, 5))
for k in K_TRAIN:
    for model_key in ["hybrid", "full_ai"]:
        agg = train_agg[(k, model_key)]
        ma, x = smooth(agg["jfi_mean"], _cfg.PLOT_SMOOTHING_WINDOW)
        c, mk, ls, _ = STYLE[(k, model_key)]
        ax2.plot(x, ma, color=c, marker=mk, linestyle=ls,
                 markevery=50, markersize=6, linewidth=1.8,
                 label=make_label(k, model_key))

ax2.set_xlabel("Episode", fontsize=12)
ax2.set_ylabel("JFI", fontsize=12)
ax2.set_title("Training — Jain's Fairness Index (6 APs)", fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
p = os.path.join(FIGURES_DIR, "training_jfi.png")
fig2.savefig(p, dpi=200)
plt.close(fig2)
print(f"[FIG] {p}")


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 2: EVALUATION EXPERIMENTS  (K = 4,6,8,10,12,14)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("BƯỚC 2: EVALUATION EXPERIMENTS  (K = 4,6,8,10,12,14)")
print("="*70)

# eval_results[(k, model_key)] = dict(throughput=float, jfi=float, thr_std, jfi_std)
eval_results = {}

for k in K_EVAL:
    set_k(k)
    print(f"\n── K={k} ──────────────────────────────")

    for model_key, use_wf, action_size in [
        ("hybrid",  True,  _cfg.HYBRID_ACTION_SIZE),
        ("full_ai", False, _cfg.FULL_AI_ACTION_SIZE),
    ]:
        runs = []
        for seed in SEEDS:
            exp_name = f"eval_K{k}_{model_key}_seed{seed}"
            print(f"  Training {exp_name} ...")
            result = train_marl(
                experiment_name=exp_name,
                use_water_filling=use_wf,
                action_size=action_size,
                episodes=_cfg.TRAIN_EPISODES,
                seed=seed,
                eval_mode="fixed",
            )
            runs.append(result)

        # Lấy giá trị eval trung bình ở 5 checkpoint cuối cùng
        agg = aggregate_results(runs)
        n_last = 5
        thr_mean = float(agg["eval_throughput_mean"][-n_last:].mean())
        thr_std  = float(agg["eval_throughput_std"][-n_last:].mean())
        jfi_mean = float(agg["eval_jfi_mean"][-n_last:].mean())
        jfi_std  = float(agg["eval_jfi_std"][-n_last:].mean())

        eval_results[(k, model_key)] = dict(
            thr_mean=thr_mean, thr_std=thr_std,
            jfi_mean=jfi_mean, jfi_std=jfi_std,
        )
        print(f"    → Thr: {thr_mean:.2f}±{thr_std:.2f} Mbps | JFI: {jfi_mean:.3f}±{jfi_std:.3f}")

# Lưu eval .txt
for model_key, label in [("hybrid", "Hybrid_WF"), ("full_ai", "Full_AI")]:
    ks     = np.array(K_EVAL, dtype=float)
    thrs   = np.array([eval_results[(k, model_key)]["thr_mean"] for k in K_EVAL])
    t_stds = np.array([eval_results[(k, model_key)]["thr_std"]  for k in K_EVAL])
    jfis   = np.array([eval_results[(k, model_key)]["jfi_mean"] for k in K_EVAL])
    j_stds = np.array([eval_results[(k, model_key)]["jfi_std"]  for k in K_EVAL])
    save_txt(
        os.path.join(EVAL_DATA_DIR, f"eval_{label}.txt"),
        header=f"Evaluation vs K | {label} | seeds={SEEDS}",
        K=ks,
        Throughput_mean=thrs, Throughput_std=t_stds,
        JFI_mean=jfis, JFI_std=j_stds,
    )

# ── Vẽ hình evaluation ────────────────────────────────────────────────────────
# Style cho eval: 2 đường (hybrid vs full_ai), marker/linestyle khác nhau
EVAL_MODEL_STYLE = {
    "hybrid":  ("black",   "o", "-",  f"Hybrid-AI+WF ({_cfg.HYBRID_ACTION_SIZE} actions)"),
    "full_ai": ("dimgray", "s", "--", f"Full-AI ({_cfg.FULL_AI_ACTION_SIZE} actions)"),
}

# Hình 3: Eval Throughput vs K
fig3, ax3 = plt.subplots(figsize=(9, 5))
for model_key in ["hybrid", "full_ai"]:
    c, mk, ls, lbl = EVAL_MODEL_STYLE[model_key]
    ks   = K_EVAL
    thrs = [eval_results[(k, model_key)]["thr_mean"] for k in ks]
    stds = [eval_results[(k, model_key)]["thr_std"]  for k in ks]
    ax3.errorbar(ks, thrs, yerr=stds,
                 color=c, marker=mk, linestyle=ls,
                 markersize=8, linewidth=1.8, capsize=4,
                 label=lbl)

ax3.set_xlabel("Clients per AP (K)", fontsize=12)
ax3.set_ylabel("Throughput (Mbps)", fontsize=12)
ax3.set_title("Evaluation — Throughput vs K (6 APs)", fontsize=13)
ax3.set_xticks(K_EVAL)
ax3.legend(fontsize=10)
ax3.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
p = os.path.join(FIGURES_DIR, "eval_throughput_vs_K.png")
fig3.savefig(p, dpi=200)
plt.close(fig3)
print(f"\n[FIG] {p}")

# Hình 4: Eval JFI vs K
fig4, ax4 = plt.subplots(figsize=(9, 5))
for model_key in ["hybrid", "full_ai"]:
    c, mk, ls, lbl = EVAL_MODEL_STYLE[model_key]
    ks   = K_EVAL
    jfis = [eval_results[(k, model_key)]["jfi_mean"] for k in ks]
    stds = [eval_results[(k, model_key)]["jfi_std"]  for k in ks]
    ax4.errorbar(ks, jfis, yerr=stds,
                 color=c, marker=mk, linestyle=ls,
                 markersize=8, linewidth=1.8, capsize=4,
                 label=lbl)

ax4.set_xlabel("Clients per AP (K)", fontsize=12)
ax4.set_ylabel("JFI", fontsize=12)
ax4.set_title("Evaluation — Jain's Fairness Index vs K (6 APs)", fontsize=13)
ax4.set_xticks(K_EVAL)
ax4.legend(fontsize=10)
ax4.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
p = os.path.join(FIGURES_DIR, "eval_jfi_vs_K.png")
fig4.savefig(p, dpi=200)
plt.close(fig4)
print(f"[FIG] {p}")


# ══════════════════════════════════════════════════════════════════════════════
# TỔNG KẾT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("HOÀN TẤT!")
print(f"  Hình training : {FIGURES_DIR}/training_throughput.png")
print(f"                  {FIGURES_DIR}/training_jfi.png")
print(f"  Hình eval     : {FIGURES_DIR}/eval_throughput_vs_K.png")
print(f"                  {FIGURES_DIR}/eval_jfi_vs_K.png")
print(f"  Data .txt     : {TRAIN_DATA_DIR}/  và  {EVAL_DATA_DIR}/")
print("="*70)
