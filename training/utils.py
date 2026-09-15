# training/utils.py
"""Các hàm tiện ích nhỏ dùng chung cho vòng lặp train/eval, tách ra khỏi
main.py để file chính không bị phình to."""
import os
import random

import numpy as np
import torch

import config


def build_global_state(states_dict, throughputs, tx_powers, agent_ids):
    """Global state cho QMIX mixer: nối CCA index, throughput, tx_power đã
    chuẩn hóa của mọi AP (Section 4.3)."""
    # Use every local feature in centralized training. The old state retained
    # only CCA/throughput/power and discarded SINR, load and active state.
    # Keep the legacy arguments so existing callers remain compatible.
    del throughputs, tx_powers
    return np.concatenate(
        [np.asarray(states_dict[aid], dtype=np.float32) for aid in agent_ids]
    ).astype(np.float32, copy=False)


def soft_update(target_model, source_model, tau):
    """Polyak averaging cho target network."""
    for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.mul_(1.0 - tau)
        target_param.data.add_(tau * source_param.data)


def set_global_seed(seed):
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def normalize_eval_mode(eval_mode):
    eval_mode = (eval_mode or config.EVAL_MODE).lower()
    if eval_mode not in ("fixed", "generalization"):
        raise ValueError("eval_mode must be 'fixed' or 'generalization'")
    return eval_mode


def moving_average(data, window=25):
    data = np.array(data, dtype=np.float32)
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


def aggregate_results(results_list):
    """Gộp kết quả nhiều seed: trung bình + std cho từng chuỗi số liệu."""
    if not results_list:
        raise ValueError("results_list must not be empty")

    agg = {}
    for key in [
        "throughput", "jfi", "train_active_aps", "train_reward",
        "train_energy_efficiency", "train_epsilon",
        "eval_throughput", "eval_jfi", "eval_active_aps", "eval_energy_efficiency",
    ]:
        lengths = {len(r[key]) for r in results_list}
        if len(lengths) != 1:
            raise ValueError(f"All runs must have the same number of {key} points, got {sorted(lengths)}")
        arr = np.array([r[key] for r in results_list], dtype=np.float32)
        agg[key + "_mean"] = arr.mean(axis=0)
        agg[key + "_std"] = arr.std(axis=0)

    # New experiment metrics are optional so legacy raw JSON files can still
    # be combined without migration.
    for key in ["train_coordination_links", "eval_coordination_links"]:
        if not all(key in result for result in results_list):
            continue
        lengths = {len(result[key]) for result in results_list}
        if len(lengths) != 1:
            raise ValueError(
                f"All runs must have the same number of {key} points, got {sorted(lengths)}"
            )
        arr = np.array([result[key] for result in results_list], dtype=np.float32)
        agg[key + "_mean"] = arr.mean(axis=0)
        agg[key + "_std"] = arr.std(axis=0)

    eval_episodes = results_list[0]["eval_episodes"]
    for r in results_list[1:]:
        if r["eval_episodes"] != eval_episodes:
            raise ValueError("All runs must use the same eval episodes for aggregation")

    agg["eval_episodes"] = results_list[0]["eval_episodes"]
    return agg


def best_so_far(values):
    return np.maximum.accumulate(np.asarray(values, dtype=np.float32))
