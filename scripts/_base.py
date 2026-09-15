"""
scripts/_base.py
----------------
Hàm dùng chung cho tất cả các script thí nghiệm.
"""
import os
import shutil
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

import json

def save_raw_run(data_dir, k, tag, key, seed, result):
    """
    Lưu NGUYÊN kết quả 1 lần train (1 ablation, 1 seed) ra JSON, để sau này
    gộp lại (mean/std qua seed) khi các seed được chạy RỜI RẠC trên nhiều
    máy/tài khoản khác nhau. Đây là nguồn dữ liệu "thô", đáng tin cậy nhất —
    file .txt tổng hợp trong cùng 1 lần chạy chỉ mang tính xem nhanh.
    """
    raw_dir = os.path.join(data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, f"raw_K{k}_{tag}_{key}_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(result, f)
    print(f"  [RAW saved] {path}")

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
                num_stas=k,   # truyền K xuống env để dùng đồng đều khi sweep
            )
            runs.append(result)

        agg = aggregate_results(runs)
        results[(k, model_key)] = agg

        # Lưu training curve (Throughput, JFI, Active-AP ratio — dùng chung cho Fig1/2 và Fig5)
        ep = np.arange(1, len(agg["throughput_mean"]) + 1)
        save_txt(
            os.path.join(data_dir, f"training_K{k}_{label}.txt"),
            header=f"Training K={k} {label} | seeds={seeds}",
            Episode=ep,
            Throughput_mean=agg["throughput_mean"],
            Throughput_std=agg["throughput_std"],
            JFI_mean=agg["jfi_mean"],
            JFI_std=agg["jfi_std"],
            ActiveAP_mean=agg["train_active_aps_mean"],
            ActiveAP_std=agg["train_active_aps_std"],
            Reward_mean=agg["train_reward_mean"],
            Reward_std=agg["train_reward_std"],
            EnergyEff_mean=agg["train_energy_efficiency_mean"],
            EnergyEff_std=agg["train_energy_efficiency_std"],
            Epsilon_mean=agg["train_epsilon_mean"],
            Epsilon_std=agg["train_epsilon_std"],
        )

        # Lưu eval summary (5 checkpoint cuối) — thêm Active-AP để dùng cho Fig6/Fig7 sau này
        n_last = 10   # tăng từ 5 — với EVAL_INTERVAL=50 mới, 10 checkpoint = trung bình trên 500 episode cuối,
                       # phủ đúng vùng hội tụ thật (ep ~636-960) thay vì chỉ vài checkpoint dễ bị nhiễu
        thr_mean = float(agg["eval_throughput_mean"][-n_last:].mean())
        thr_std  = float(agg["eval_throughput_std"][-n_last:].mean())
        jfi_mean = float(agg["eval_jfi_mean"][-n_last:].mean())
        jfi_std  = float(agg["eval_jfi_std"][-n_last:].mean())
        aps_mean = float(agg["eval_active_aps_mean"][-n_last:].mean())
        aps_std  = float(agg["eval_active_aps_std"][-n_last:].mean())
        save_txt(
            os.path.join(data_dir, f"eval_summary_K{k}_{label}.txt"),
            header=f"Eval summary K={k} {label} | seeds={seeds} | last {n_last} checkpoints",
            K=np.array([k], dtype=float),
            Throughput_mean=np.array([thr_mean]),
            Throughput_std=np.array([thr_std]),
            JFI_mean=np.array([jfi_mean]),
            JFI_std=np.array([jfi_std]),
            ActiveAP_mean=np.array([aps_mean]),
            ActiveAP_std=np.array([aps_std]),
        )
        print(f"    Eval → Thr: {thr_mean:.2f}±{thr_std:.2f} Mbps | JFI: {jfi_mean:.3f}±{jfi_std:.3f} | Active-AP: {aps_mean:.2f}±{aps_std:.2f}")

        # Giữ lại model của seed tốt nhất (theo best_eval_score nội bộ) vào results/best_models/
        # để sau này không phải lục qua từng seed folder khi cần checkpoint để demo/deploy.
        best_run = max(runs, key=lambda r: r["best_eval_score"])
        best_seed = best_run["seed"]
        best_score = best_run["best_eval_score"]
        best_models_src = best_run["models_dir"]
        best_models_dst = os.path.join("results", "best_models", f"K{k}_{label}")
        if os.path.isdir(best_models_dst):
            shutil.rmtree(best_models_dst)
        shutil.copytree(best_models_src, best_models_dst)
        with open(os.path.join(best_models_dst, "BEST_SEED_INFO.txt"), "w") as f:
            f.write(f"K={k}\nmodel={label}\nbest_seed={best_seed}\nbest_eval_score={best_score:.6f}\n"
                    f"all_seeds_tried={seeds}\nsource_dir={best_models_src}\n")
        print(f"    Best model (seed={best_seed}, score={best_score:.3f}) copied → {best_models_dst}/")

    return results


def save_ablation_outputs(k, tag, key, label, seeds, agg, data_dir):
    """
    Lưu training curve + eval summary .txt cho MỘT ablation (dùng chung bởi
    run_ablation_k khi train trực tiếp, và combine_ablation_results.py khi
    gộp raw JSON từ nhiều máy). Trả về (thr_mean, jfi_mean, aps_mean, ee_mean)
    của n_last checkpoint cuối, để in bảng tổng hợp.
    """
    ep = np.arange(1, len(agg["throughput_mean"]) + 1)
    training_arrays = dict(
        Episode=ep,
        Throughput_mean=agg["throughput_mean"],
        Throughput_std=agg["throughput_std"],
        JFI_mean=agg["jfi_mean"],
        JFI_std=agg["jfi_std"],
        ActiveAP_mean=agg["train_active_aps_mean"],
        ActiveAP_std=agg["train_active_aps_std"],
        Reward_mean=agg["train_reward_mean"],
        Reward_std=agg["train_reward_std"],
        EnergyEff_mean=agg["train_energy_efficiency_mean"],
        EnergyEff_std=agg["train_energy_efficiency_std"],
        Epsilon_mean=agg["train_epsilon_mean"],
        Epsilon_std=agg["train_epsilon_std"],
    )
    if "train_coordination_links_mean" in agg:
        training_arrays["CoordLinks_mean"] = agg["train_coordination_links_mean"]
        training_arrays["CoordLinks_std"] = agg["train_coordination_links_std"]
    save_txt(
        os.path.join(data_dir, f"training_K{k}_{tag}_{key}.txt"),
        header=f"Training K={k} [{tag}/{label}] | seeds={seeds}",
        **training_arrays,
    )

    n_last = 10
    thr_mean = float(agg["eval_throughput_mean"][-n_last:].mean())
    thr_std  = float(agg["eval_throughput_std"][-n_last:].mean())
    jfi_mean = float(agg["eval_jfi_mean"][-n_last:].mean())
    jfi_std  = float(agg["eval_jfi_std"][-n_last:].mean())
    aps_mean = float(agg["eval_active_aps_mean"][-n_last:].mean())
    aps_std  = float(agg["eval_active_aps_std"][-n_last:].mean())
    ee_mean  = float(agg["eval_energy_efficiency_mean"][-n_last:].mean())
    ee_std   = float(agg["eval_energy_efficiency_std"][-n_last:].mean())
    eval_arrays = dict(
        K=np.array([k], dtype=float),
        Throughput_mean=np.array([thr_mean]),
        Throughput_std=np.array([thr_std]),
        JFI_mean=np.array([jfi_mean]),
        JFI_std=np.array([jfi_std]),
        ActiveAP_mean=np.array([aps_mean]),
        ActiveAP_std=np.array([aps_std]),
        EnergyEff_mean=np.array([ee_mean]),
        EnergyEff_std=np.array([ee_std]),
    )
    if "eval_coordination_links_mean" in agg:
        coord_mean = float(agg["eval_coordination_links_mean"][-n_last:].mean())
        coord_std = float(agg["eval_coordination_links_std"][-n_last:].mean())
        eval_arrays["CoordLinks_mean"] = np.array([coord_mean])
        eval_arrays["CoordLinks_std"] = np.array([coord_std])
    save_txt(
        os.path.join(data_dir, f"eval_summary_K{k}_{tag}_{key}.txt"),
        header=f"Eval summary K={k} [{tag}/{label}] | seeds={seeds} | last {n_last} checkpoints",
        **eval_arrays,
    )
    print(f"    [{label}] Eval → Thr: {thr_mean:.2f}±{thr_std:.2f} Mbps | JFI: {jfi_mean:.3f}±{jfi_std:.3f} | "
          f"Active-AP: {aps_mean:.2f}±{aps_std:.2f} | EE: {ee_mean:.3f}±{ee_std:.3f}")
    return thr_mean, jfi_mean, aps_mean, ee_mean


def run_ablation_k(k, seeds, ablation_configs, data_dir, use_wf=False,
                    action_size=None, tag="ablation"):
    """
    Train baseline + các biến thể ablation (bỏ Double DQN / QMIX / VIB) tại
    MỘT giá trị K cố định, để so sánh "bỏ 1 phần thì thay đổi thế nào" trong
    cùng điều kiện tải (giống Bảng ablation study trong bài báo).

    ablation_configs: list of dict, mỗi dict là:
        {
            "key": "baseline" | "no_ddqn" | "no_qmix" | "with_vib" | ...,
            "label": chuỗi hiển thị,
            "use_double_dqn": bool,   # default True
            "use_qmix": bool,          # False -> VDN, default True
            "ib_beta": float hoặc None,  # None -> clean baseline beta=0
        }
    use_wf, action_size: cấu hình action-space dùng chung cho MỌI ablation
        (mặc định Full-AI, vì QMIX/DDQN/VIB là phần cốt lõi của nhánh này).
        Đổi use_wf=True + action_size=config.HYBRID_ACTION_SIZE nếu muốn
        ablation trên nhánh Hybrid + Water-Filling thay vì Full-AI.

    Trả về: dict[ablation_key] = agg
    """
    from main import train_marl, aggregate_results
    import config as _cfg

    if action_size is None:
        action_size = _cfg.HYBRID_ACTION_SIZE if use_wf else _cfg.FULL_AI_ACTION_SIZE

    set_k(k)
    results = {}

    for ac in ablation_configs:
        key = ac["key"]
        label = ac.get("label", key)
        use_ddqn = ac.get("use_double_dqn", True)
        use_qmix = ac.get("use_qmix", True)
        ib_beta = ac.get("ib_beta", None)
        ac_use_wf = ac.get("use_wf", use_wf)
        ac_action_size = ac.get("action_size", action_size)
        cooperation_enabled = ac.get("cooperation_enabled", False)

        runs = []
        for seed in seeds:
            exp_name = f"{tag}_K{k}_{key}_seed{seed}"   # tag đã có sẵn ở đây (vd: ablation vs ablation_hybrid)
            print(f"\n  >> Training [{label}]: {exp_name}")
            result = train_marl(
                experiment_name=exp_name,
                use_water_filling=ac_use_wf,
                action_size=ac_action_size,
                episodes=_cfg.TRAIN_EPISODES,
                seed=seed,
                eval_mode="fixed",
                num_stas=k,
                use_double_dqn=use_ddqn,
                use_qmix=use_qmix,
                ib_beta=ib_beta,
                cooperation_enabled=cooperation_enabled,
            )
            runs.append(result)
            save_raw_run(data_dir, k, tag, key, seed, result)   # lưu ngay, phòng khi máy này chỉ chạy 1 phần seed

        agg = aggregate_results(runs)
        results[key] = agg
        is_complete_seed_set = set(seeds) == set(_cfg.FINAL_SEEDS)
        if not is_complete_seed_set:
            print(f"  ⚠️  Lần chạy này chỉ có seed={seeds} (không đủ {_cfg.FINAL_SEEDS}) — "
                  f"chỉ lưu raw JSON riêng cho seed này để tránh nhiều process ghi đè nhau. "
                  f"Sau khi gom đủ raw JSON từ mọi máy, chạy scripts/combine_ablation_results.py "
                  f"để có file .txt chính thức (mean/std đủ 3 seed).")
        else:
            save_ablation_outputs(k, tag, key, label, seeds, agg, data_dir)

            # Copying a best model is safe only after all seeds are available.
            best_run = max(runs, key=lambda r: r["best_eval_score"])
            best_seed = best_run["seed"]
            best_score = best_run["best_eval_score"]
            best_models_src = best_run["models_dir"]
            best_models_dst = os.path.join("results", "best_models", f"K{k}_{tag}_{key}")
            if os.path.isdir(best_models_dst):
                shutil.rmtree(best_models_dst)
            shutil.copytree(best_models_src, best_models_dst)
            with open(os.path.join(best_models_dst, "BEST_SEED_INFO.txt"), "w") as f:
                f.write(f"K={k}\nbranch_tag={tag}\nablation={key}\nlabel={label}\nbest_seed={best_seed}\n"
                        f"best_eval_score={best_score:.6f}\nall_seeds_tried={seeds}\nsource_dir={best_models_src}\n")
            print(f"    Best model (seed={best_seed}, score={best_score:.3f}) copied → {best_models_dst}/")

    # Bảng tổng hợp gọn để so baseline vs từng ablation ngay trên console
    print(f"\n{'='*70}\n  TỔNG HỢP ABLATION K={k}\n{'='*70}")
    print(f"  {'Ablation':<12}{'Throughput':>14}{'JFI':>10}{'Active-AP':>12}{'EnergyEff':>12}")
    for ac in ablation_configs:
        key = ac["key"]
        agg = results[key]
        n_last = 10
        thr = float(agg["eval_throughput_mean"][-n_last:].mean())
        jfi = float(agg["eval_jfi_mean"][-n_last:].mean())
        aps = float(agg["eval_active_aps_mean"][-n_last:].mean())
        ee  = float(agg["eval_energy_efficiency_mean"][-n_last:].mean())
        print(f"  {key:<12}{thr:>11.2f} Mbps{jfi:>10.3f}{aps:>12.2f}{ee:>12.3f}")

    return results
