"""Auditable non-neural scheduling baselines for :class:`CoSREnv`."""
from collections import defaultdict

import numpy as np


def _valid_non_idle_actions(env, mask):
    return [idx for idx in np.flatnonzero(mask) if idx != 0]


def select_single_oldest(env, mask, rng=None):
    """Co-TDMA baseline: schedule the single STA with greatest HoL delay."""
    candidates = [
        action for action in _valid_non_idle_actions(env, mask)
        if len(env.group_for_action(action)) == 1
    ]
    if not candidates:
        return 0
    return max(
        candidates,
        key=lambda action: (
            env._hol_delay(env.group_for_action(action)[0][1]),
            env.queues[env.group_for_action(action)[0][1]],
            -action,
        ),
    )


def select_group_mnp(env, mask, rng=None):
    """Maximum-number-of-packets scheduler over feasible Co-SR groups."""
    candidates = _valid_non_idle_actions(env, mask)
    if not candidates:
        return 0
    return max(
        candidates,
        key=lambda action: (
            sum(env.queues[sta_id] for _, sta_id in env.group_for_action(action)),
            len(env.group_for_action(action)),
            -action,
        ),
    )


def select_group_oldest(env, mask, rng=None):
    """Prioritize the group containing the globally oldest queued packet."""
    candidates = _valid_non_idle_actions(env, mask)
    if not candidates:
        return 0
    return max(
        candidates,
        key=lambda action: (
            max(env._hol_delay(sta_id) for _, sta_id in env.group_for_action(action)),
            sum(env.queues[sta_id] for _, sta_id in env.group_for_action(action)),
            len(env.group_for_action(action)),
            -action,
        ),
    )


def select_random_valid(env, mask, rng):
    candidates = _valid_non_idle_actions(env, mask)
    return int(rng.choice(candidates)) if candidates else 0


class HierarchicalUCBPolicy:
    """Factorized two-level UCB for Co-SR scheduling.

    Level one learns the value of an AP set.  Level two learns one value per
    AP--STA link, not one value per joint AP--STA group.  This retains the
    hierarchy from the Co-SR MAB formulation and avoids expanding the second
    level to every STA-pair combination.
    """

    def __init__(self, env, exploration=1.0):
        self.exploration = float(exploration)
        self.total_updates = 0
        self.ap_set_counts = defaultdict(int)
        self.ap_set_values = defaultdict(float)
        self.link_counts = defaultdict(int)
        self.link_values = defaultdict(float)

    @staticmethod
    def _ap_set(env, action):
        return tuple(ap_id for ap_id, _ in env.group_for_action(action))

    def _ucb(self, value, count, log_t):
        if count == 0:
            return float("inf")
        return value + self.exploration * np.sqrt(log_t / count)

    def select(self, env, mask, rng=None):
        valid = _valid_non_idle_actions(env, mask)
        if not valid:
            return 0
        by_ap_set = defaultdict(list)
        for action in valid:
            by_ap_set[self._ap_set(env, action)].append(action)

        log_t = np.log(max(self.total_updates, 1) + 1.0)
        unvisited_sets = [key for key in by_ap_set if self.ap_set_counts[key] == 0]
        if unvisited_sets:
            # Prefer larger feasible groups while every first-level arm is new.
            chosen_set = max(unvisited_sets, key=lambda key: (len(key), tuple(-x for x in key)))
        else:
            chosen_set = max(
                by_ap_set,
                key=lambda key: self._ucb(
                    self.ap_set_values[key], self.ap_set_counts[key], log_t
                ),
            )

        actions = by_ap_set[chosen_set]
        return max(
            actions,
            key=lambda action: (
                sum(
                    self._ucb(
                        self.link_values[(ap_id, sta_id)],
                        self.link_counts[(ap_id, sta_id)],
                        log_t,
                    )
                    for ap_id, sta_id in env.group_for_action(action)
                ),
                sum(env.queues[sta_id] for _, sta_id in env.group_for_action(action)),
                -action,
            ),
        )

    def update(self, env, action, info):
        if action == 0:
            return
        ap_set = self._ap_set(env, action)
        self.total_updates += 1
        self.ap_set_counts[ap_set] += 1
        n_set = self.ap_set_counts[ap_set]
        set_reward = float(info["throughput"])
        self.ap_set_values[ap_set] += (set_reward - self.ap_set_values[ap_set]) / n_set

        # Per-link effective PHY rate is the second-level bandit reward.  It
        # stays meaningful even when traffic queues are transiently empty.
        for ap_id, sta_id in env.group_for_action(action):
            link = (ap_id, sta_id)
            link_reward = float(info["rates_mbps"].get(sta_id, 0.0))
            self.link_counts[link] += 1
            n_link = self.link_counts[link]
            self.link_values[link] += (link_reward - self.link_values[link]) / n_link


BASELINE_SELECTORS = {
    "random": select_random_valid,
    "single_oldest": select_single_oldest,
    "group_mnp": select_group_mnp,
    "group_oldest": select_group_oldest,
}
