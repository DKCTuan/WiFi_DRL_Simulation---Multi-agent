"""
scripts/_base.py
----------------
Hàm dùng chung cho tất cả các script thí nghiệm.
"""
import os
import numpy as np
import sys

# Đảm bảo import được config và main từ thư mục gốc project
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config as _cfg

def set_k(k):
    """Override số user per AP toàn cục."""
    _cfg.NUM_STAS_PER_AP = k
    _cfg.AP_LOAD_PROFILE = [k] * _cfg.NUM_APS

def save_txt(path, header, **arrays):
    """Lưu các mảng numpy ra file .txt dạng cột tab-separated."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = list(arrays.items())
    n = len(cols[0][1])
    with open(path, "w") as f:
        f.write("# " + header + "\n")
        f.write("# " + "\t".join(k for k, _ in cols) + "\n")
        for i in range(n):
            row = "\t".join(f"{v[i]:.6f}" for _, v in cols)
            f.write(row + "\n")
    print(f"  [TXT saved] {path}")

def run_k(k, seeds, model_configs, data_dir, tag="training"):
    """
    Train tất cả model_configs cho một giá trị K, lưu .txt.

    model_configs: list of dict với keys:
        model_key, use_wf, action_size
    
    Trả về: dict[(k, model_key)] = agg
    """
    from main import train_marl, aggregate_results
    import config as _cfg

    set_k(k)
    results = {}

    for mc in model_configs:
        model_key  = mc["model_key"]
        use_wf     = mc["use_wf"]
        action_size = mc["action_size"]
        label      = "Hybrid_WF" if model_key == "hybrid" else "Full_AI"

        runs = []
        for seed in seeds:
            exp_name = f"{tag}_K{k}_{model_key}_seed{seed}"
            print(f"\n  >> Training: {exp_name}")
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
        results[(k, model_key)] = agg

        # Lưu training curve
        ep = np.arange(1, len(agg["throughput_mean"]) + 1)
        save_txt(
            os.path.join(data_dir, f"training_K{k}_{label}.txt"),
            header=f"Training K={k} {label} | seeds={seeds}",
            Episode=ep,
            Throughput_mean=agg["throughput_mean"],
            Throughput_std=agg["throughput_std"],
            JFI_mean=agg["jfi_mean"],
            JFI_std=agg["jfi_std"],
        )

        # Lưu eval summary (5 checkpoint cuối)
        n_last = 5
        thr_mean = float(agg["eval_throughput_mean"][-n_last:].mean())
        thr_std  = float(agg["eval_throughput_std"][-n_last:].mean())
        jfi_mean = float(agg["eval_jfi_mean"][-n_last:].mean())
        jfi_std  = float(agg["eval_jfi_std"][-n_last:].mean())
        save_txt(
            os.path.join(data_dir, f"eval_summary_K{k}_{label}.txt"),
            header=f"Eval summary K={k} {label} | seeds={seeds} | last {n_last} checkpoints",
            K=np.array([k], dtype=float),
            Throughput_mean=np.array([thr_mean]),
            Throughput_std=np.array([thr_std]),
            JFI_mean=np.array([jfi_mean]),
            JFI_std=np.array([jfi_std]),
        )
        print(f"    Eval → Thr: {thr_mean:.2f}±{thr_std:.2f} Mbps | JFI: {jfi_mean:.3f}±{jfi_std:.3f}")

    return results
