"""
scripts/eval_K4_K6.py
=====================
Load model đã train, eval lại K=4 và K=6 với nhiều episode hơn.

Cách dùng trên Kaggle:
  1. Upload results_K4.zip và results_K6.zip lên Kaggle Dataset (vd: "models-k4-k6")
  2. Giải nén vào /kaggle/working/project/results/
  3. Chạy: python scripts/eval_K4_K6.py

Output: cập nhật eval_summary_K4_*.txt và eval_summary_K6_*.txt
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import numpy as np
import config as _cfg
from scripts._base import set_k, save_txt
from agent.double_dqn import DoubleDQNAgent
from agent.qmix_helper import QMixer
from main import evaluate_policy

# ── Cấu hình ────────────────────────────────────────────────────────────────
SEEDS       = _cfg.FINAL_SEEDS        # [0, 1, 2]
K_LIST      = [4, 6]
EVAL_EP     = 100                     # tăng từ 30 lên 100 theo yêu cầu thầy
DATA_DIR    = "results/data"
RESULTS_DIR = "results"

os.makedirs(DATA_DIR, exist_ok=True)

MODEL_CONFIGS = [
    dict(model_key="hybrid",  use_wf=True,  action_size=_cfg.HYBRID_ACTION_SIZE,
         latent_size=_cfg.LATENT_SIZE_HYBRID,  epsilon_decay=_cfg.EPSILON_DECAY_HYBRID),
    dict(model_key="full_ai", use_wf=False, action_size=_cfg.FULL_AI_ACTION_SIZE,
         latent_size=_cfg.LATENT_SIZE_FULL_AI, epsilon_decay=_cfg.EPSILON_DECAY_FULL_AI),
]

def load_agents(models_dir, action_size, latent_size, epsilon_decay, agent_ids):
    """Load best model từ thư mục models/"""
    device = torch.device("cpu")
    agents = {}
    for agent_id in agent_ids:
        agent = DoubleDQNAgent(
            state_size=_cfg.OBS_SIZE,
            action_size=action_size,
            latent_size=latent_size,
            epsilon_decay=epsilon_decay,
        )
        # Ưu tiên best model, fallback sang final model
        enc_path = os.path.join(models_dir, f"best_ib_encoder_{agent_id}.pth")
        q_path   = os.path.join(models_dir, f"best_ib_qmix_{agent_id}_model.pth")
        if not os.path.exists(enc_path):
            enc_path = os.path.join(models_dir, f"ib_encoder_{agent_id}.pth")
            q_path   = os.path.join(models_dir, f"ib_qmix_{agent_id}_model.pth")

        agent.encoder.load_state_dict(torch.load(enc_path, map_location=device))
        agent.q_network.load_state_dict(torch.load(q_path, map_location=device))
        agent.epsilon = 0.0   # greedy hoàn toàn
        agents[agent_id] = agent

    return agents

# ── Main eval loop ───────────────────────────────────────────────────────────
for k in K_LIST:
    set_k(k)
    print(f"\n{'='*60}")
    print(f"  EVAL K={k} | EVAL_EPISODES={EVAL_EP}")
    print(f"{'='*60}")

    # Agent IDs
    agent_ids = [f"ap_{i}" for i in range(_cfg.NUM_APS)]

    for mc in MODEL_CONFIGS:
        model_key   = mc["model_key"]
        use_wf      = mc["use_wf"]
        action_size = mc["action_size"]
        latent_size = mc["latent_size"]
        epsilon_decay = mc["epsilon_decay"]
        label       = "Hybrid_WF" if model_key == "hybrid" else "Full_AI"

        thr_list, jfi_list = [], []

        for seed in SEEDS:
            exp_name  = f"v3_K{k}_{model_key}_seed{seed}"
            models_dir = os.path.join(RESULTS_DIR, exp_name, "models")

            if not os.path.exists(models_dir):
                print(f"  ⚠ Không tìm thấy: {models_dir} — bỏ qua")
                continue

            print(f"\n  >> Load model: {exp_name}")
            agents = load_agents(models_dir, action_size, latent_size, epsilon_decay, agent_ids)

            metrics = evaluate_policy(
                agents,
                episodes=EVAL_EP,
                use_water_filling=use_wf,
                action_size=action_size,
                eval_mode="fixed",
            )
            thr_list.append(metrics["throughput"])
            jfi_list.append(metrics["jfi"])
            print(f"    Thr: {metrics['throughput']:.2f} Mbps | JFI: {metrics['jfi']:.3f}")

        if not thr_list:
            print(f"  ❌ Không có data cho K={k} {label}")
            continue

        thr_mean = float(np.mean(thr_list))
        thr_std  = float(np.std(thr_list))
        jfi_mean = float(np.mean(jfi_list))
        jfi_std  = float(np.std(jfi_list))

        print(f"\n  ✅ K={k} {label} | Thr: {thr_mean:.2f}±{thr_std:.2f} | JFI: {jfi_mean:.3f}±{jfi_std:.3f}")

        # Ghi đè file eval_summary cũ
        save_txt(
            os.path.join(DATA_DIR, f"eval_summary_K{k}_{label}.txt"),
            header=f"Eval summary K={k} {label} | seeds={SEEDS} | eval_ep={EVAL_EP} | last 5 checkpoints",
            K=np.array([k], dtype=float),
            Throughput_mean=np.array([thr_mean]),
            Throughput_std=np.array([thr_std]),
            JFI_mean=np.array([jfi_mean]),
            JFI_std=np.array([jfi_std]),
        )

print(f"\n✓ Xong! File eval_summary K=4 và K=6 đã được cập nhật tại: {DATA_DIR}/")
print("  Gom với data K=8,10,12,14 cũ rồi chạy plot_final.py để vẽ lại hình!")
