"""
scripts/run_v3_K6.py  —  Kaggle Version 3.3
======================================================
K = 6  |  Hybrid + Full-AI  |  3 seeds
~3-4 giờ trên Kaggle GPU

Chạy từ thư mục gốc project:
    python scripts/run_v3_K6.py
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts._base import run_k
import config as _cfg

SEEDS    = _cfg.FINAL_SEEDS
DATA_DIR = "results/data"
K        = 6

MODEL_CONFIGS = [
    dict(model_key="hybrid",  use_wf=True,  action_size=_cfg.HYBRID_ACTION_SIZE),
    dict(model_key="full_ai", use_wf=False, action_size=_cfg.FULL_AI_ACTION_SIZE),
]

print(f"\n{'='*60}")
print(f"  K=6 | seeds={SEEDS}")
print(f"{'='*60}")

run_k(K, SEEDS, MODEL_CONFIGS, DATA_DIR, tag="v3")

print(f"\n✓ Xong! Data lưu tại: {DATA_DIR}/")
