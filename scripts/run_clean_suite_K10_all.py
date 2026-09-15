"""Run the complete K=10 clean-baseline validation suite on one machine.

This is the Kaggle entry point: it runs all four configurations for seeds
0, 1 and 2, first for Full-AI and then for Hybrid + Water-Filling.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config
from scripts._base import run_ablation_k


K = 10
DATA_DIR = "results/data"
SEEDS = config.FINAL_SEEDS

CONFIGURATIONS = [
    dict(key="baseline", label="Clean baseline (no VIB)",
         use_double_dqn=True, use_qmix=True, ib_beta=0.0),
    dict(key="no_ddqn", label="No Double DQN",
         use_double_dqn=False, use_qmix=True, ib_beta=0.0),
    dict(key="no_qmix", label="VDN (no learned mixer)",
         use_double_dqn=True, use_qmix=False, ib_beta=0.0),
    dict(key="with_vib", label="With VIB",
         use_double_dqn=True, use_qmix=True, ib_beta=config.VIB_BETA),
]


print(f"\n{'=' * 70}\nCLEAN VALIDATION SUITE | K={K} | seeds={SEEDS}\n{'=' * 70}")

run_ablation_k(
    K, SEEDS, CONFIGURATIONS, DATA_DIR,
    use_wf=False,
    action_size=config.FULL_AI_ACTION_SIZE,
    tag="ablation_clean",
)
run_ablation_k(
    K, SEEDS, CONFIGURATIONS, DATA_DIR,
    use_wf=True,
    action_size=config.HYBRID_ACTION_SIZE,
    tag="ablation_clean_hybrid",
)

print("\nComplete. Aggregated summaries are in results/data/.")
