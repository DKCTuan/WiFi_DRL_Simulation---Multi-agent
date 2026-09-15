# main.py
"""
Entry point mỏng: chỉ orchestration cho lần chạy "final" (3 seeds x 2 model).
Toàn bộ logic thật nằm trong:
  - training/train_loop.py : vòng lặp huấn luyện (train_marl)
  - training/evaluate.py   : đánh giá greedy policy (evaluate_policy)
  - training/utils.py      : hàm tiện ích (aggregate_results, seeding, ...)
  - training/plotting.py   : vẽ biểu đồ so sánh + bảng tổng kết

`train_marl` và `aggregate_results` được re-export ở đây để giữ tương thích
ngược với scripts/_base.py (`from main import train_marl, aggregate_results`).
"""
import config

from training.train_loop import train_marl
from training.evaluate import evaluate_policy, eval_score
from training.utils import aggregate_results, moving_average, best_so_far
from training.plotting import (
    plot_comparison_multiseed,
    plot_eval_comparison_multiseed,
    plot_publication_comparison,
    print_summary_table,
)

__all__ = [
    "train_marl",
    "evaluate_policy",
    "eval_score",
    "aggregate_results",
    "moving_average",
    "best_so_far",
    "plot_comparison_multiseed",
    "plot_eval_comparison_multiseed",
    "plot_publication_comparison",
    "print_summary_table",
]


if __name__ == "__main__":
    SEEDS = config.FINAL_SEEDS
    EPISODES = config.TRAIN_EPISODES

    results_full_ai = []
    results_hybrid = []

    for seed in SEEDS:
        results_full_ai.append(train_marl(
            experiment_name=f"final_full_ai_seed{seed}",
            use_water_filling=False,
            action_size=config.FULL_AI_ACTION_SIZE,
            episodes=EPISODES,
            seed=seed,
            eval_mode="fixed",
        ))
        results_hybrid.append(train_marl(
            experiment_name=f"final_hybrid_wf_seed{seed}",
            use_water_filling=True,
            action_size=config.HYBRID_ACTION_SIZE,
            episodes=EPISODES,
            seed=seed,
            eval_mode="fixed",
        ))

    agg_full_ai = aggregate_results(results_full_ai)
    agg_hybrid = aggregate_results(results_hybrid)

    plot_comparison_multiseed(agg_full_ai, agg_hybrid)
    plot_eval_comparison_multiseed(agg_full_ai, agg_hybrid)
    plot_publication_comparison(agg_full_ai, agg_hybrid)
    print_summary_table(agg_full_ai, agg_hybrid)
