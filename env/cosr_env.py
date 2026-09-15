"""Packet-level, TXOP-based environment for Wi-Fi 8 coordinated spatial reuse.

This environment intentionally lives beside ``WiFiEnv``.  It does not change
the legacy Hybrid-WF/QMIX experiments, which remain reproducible baselines.
Actions are centralized indices into AP-STA groups; this gives us a physically
auditable baseline before adapting the multi-agent learner.
"""
from itertools import combinations

import gymnasium as gym
from gymnasium import spaces
import numpy as np

import config
from env.cosr_phy import (
    allocate_group_power,
    channel_gain,
    group_phy_rates,
    is_compatible_group,
)
from env.mcs import sinr_linear_to_db, validate_mcs_table
from env.mobility import update_mobility
from env.topology import setup_aps, setup_stas


class CoSREnv(gym.Env):
    """Centralized Co-SR scheduler with packet queues and dynamic action masks."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_stas_per_ap=None,
        fixed_topology=True,
        fixed_seed=config.TRAIN_SCENARIO_SEED,
        mobility_enabled=False,
        power_mode="full",
        max_group_aps=config.COSR_MAX_GROUP_APS,
        max_steps=None,
        arrival_rate_per_sta=config.COSR_MEAN_ARRIVALS_PER_STA,
    ):
        super().__init__()
        validate_mcs_table()
        if max_group_aps not in (1, 2):
            raise ValueError("The validated Co-SR baseline currently supports group sizes 1 or 2")
        if power_mode not in ("full", "pf_pair"):
            raise ValueError("power_mode must be 'full' or 'pf_pair'")
        if arrival_rate_per_sta < 0:
            raise ValueError("arrival_rate_per_sta must be non-negative")

        self.fixed_topology = bool(fixed_topology)
        self.fixed_seed = fixed_seed
        self.mobility_enabled = bool(mobility_enabled)
        self.power_mode = power_mode
        self.max_group_aps = int(max_group_aps)
        self.max_steps = config.TRAIN_STEPS_PER_EPISODE if max_steps is None else int(max_steps)
        self.arrival_rate_per_sta = float(arrival_rate_per_sta)
        self.ap_load_profile = (
            [int(num_stas_per_ap)] * config.NUM_APS
            if num_stas_per_ap is not None
            else list(config.AP_LOAD_PROFILE)
        )

        self.aps = []
        self.stas = []
        self.stas_by_id = {}
        self.sta_ids_by_ap = {}
        self.groups = []
        self._physical_mask = None
        self.current_step = 0
        self.queues = []
        self.arrival_steps = []
        self.delivered_packets = []
        self.total_arrivals = 0
        self.total_drops = 0
        self.total_throughput_mbps = 0.0
        self.total_energy_efficiency = 0.0
        self.total_group_size = 0.0
        self.served_delays = []
        self._measurement_start = None

        # Topology size, and therefore action/observation dimensions, is known
        # from the load profile even before the first reset.
        self.num_stas = int(sum(self.ap_load_profile))
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3 * self.num_stas,), dtype=np.float32
        )
        candidate_count = self.num_stas
        if self.max_group_aps >= 2:
            candidate_count += sum(
                self.ap_load_profile[i] * self.ap_load_profile[j]
                for i, j in combinations(range(config.NUM_APS), 2)
            )
        # Action zero is an explicit idle action.
        self.action_space = spaces.Discrete(1 + candidate_count)

    def _setup_topology(self, seed):
        self.aps = setup_aps()
        if self.fixed_topology:
            topology_seed = self.fixed_seed if self.fixed_seed is not None else seed
        else:
            topology_seed = int(self.np_random.integers(0, 2**31 - 1))
        if topology_seed is None:
            topology_seed = config.GLOBAL_SEED
        self.stas = setup_stas(self.aps, self.ap_load_profile, int(topology_seed))
        self.stas_by_id = {sta["id"]: sta for sta in self.stas}
        self.sta_ids_by_ap = {
            ap_id: [sta["id"] for sta in self.stas if sta["ap_id"] == ap_id]
            for ap_id in range(config.NUM_APS)
        }

    def _build_candidate_groups(self):
        groups = []
        for ap_id in range(config.NUM_APS):
            groups.extend(((ap_id, sta_id),) for sta_id in self.sta_ids_by_ap[ap_id])
        if self.max_group_aps >= 2:
            for ap_i, ap_j in combinations(range(config.NUM_APS), 2):
                for sta_i in self.sta_ids_by_ap[ap_i]:
                    for sta_j in self.sta_ids_by_ap[ap_j]:
                        groups.append(((ap_i, sta_i), (ap_j, sta_j)))
        self.groups = groups
        if len(groups) + 1 != self.action_space.n:
            raise RuntimeError("Co-SR action-space size does not match generated groups")
        self._physical_mask = np.asarray(
            [is_compatible_group(self.aps, self.stas_by_id, group) for group in groups],
            dtype=bool,
        )

    def _add_arrivals(self, initial=False):
        mean = (
            config.COSR_INITIAL_MEAN_PACKETS_PER_STA
            if initial
            else self.arrival_rate_per_sta
        )
        arrivals = self.np_random.poisson(mean, size=self.num_stas)
        for sta_id, count in enumerate(arrivals):
            count = int(count)
            self.total_arrivals += count
            room = config.COSR_MAX_QUEUE_PACKETS - self.queues[sta_id]
            admitted = min(count, max(room, 0))
            self.queues[sta_id] += admitted
            self.arrival_steps[sta_id].extend([self.current_step] * admitted)
            self.total_drops += count - admitted

    def _hol_delay(self, sta_id):
        if not self.arrival_steps[sta_id]:
            return 0
        return max(0, self.current_step - self.arrival_steps[sta_id][0])

    def _observation(self):
        values = []
        max_queue = max(config.COSR_MAX_QUEUE_PACKETS, 1)
        max_delay = max(config.COSR_MAX_HOL_DELAY_TXOPS, 1)
        max_power = {ap_id: config.P_MAX for ap_id in range(config.NUM_APS)}
        for sta in self.stas:
            ap_id = sta["ap_id"]
            direct_sinr = max_power[ap_id] * channel_gain(self.aps[ap_id], sta) / config.NOISE_POWER
            sinr_norm = np.clip((sinr_linear_to_db(direct_sinr) + 10.0) / 50.0, 0.0, 1.0)
            values.extend((
                min(self.queues[sta["id"]] / max_queue, 1.0),
                min(self._hol_delay(sta["id"]) / max_delay, 1.0),
                float(sinr_norm),
            ))
        return np.asarray(values, dtype=np.float32)

    def action_masks(self):
        """Mask offline PHY-incompatible and online empty-queue groups."""
        if self.mobility_enabled:
            physical = np.asarray(
                [is_compatible_group(self.aps, self.stas_by_id, group) for group in self.groups],
                dtype=bool,
            )
        else:
            physical = self._physical_mask
        queue_valid = np.asarray(
            [all(self.queues[sta_id] > 0 for _, sta_id in group) for group in self.groups],
            dtype=bool,
        )
        # Idle remains valid, including when no data can be scheduled.
        return np.concatenate((np.asarray([True]), physical & queue_valid))

    def group_for_action(self, action):
        action = int(action)
        if action == 0:
            return tuple()
        if action < 0 or action >= self.action_space.n:
            raise IndexError(f"Co-SR action {action} outside [0, {self.action_space.n})")
        return self.groups[action - 1]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self._setup_topology(seed)
        self._build_candidate_groups()
        self.queues = [0] * self.num_stas
        self.arrival_steps = [[] for _ in range(self.num_stas)]
        self.delivered_packets = [0] * self.num_stas
        self.total_arrivals = 0
        self.total_drops = 0
        self.total_throughput_mbps = 0.0
        self.total_energy_efficiency = 0.0
        self.total_group_size = 0.0
        self.served_delays = []
        self._measurement_start = None
        self._add_arrivals(initial=True)
        mask = self.action_masks()
        return self._observation(), {
            "action_mask": mask,
            "valid_action_count": int(mask.sum()),
            "candidate_group_count": len(self.groups),
        }

    def begin_measurement_window(self):
        """Mark the end of warm-up so reported metrics use steady-state data."""
        self._measurement_start = {
            "step": self.current_step,
            "queue_packets": int(sum(self.queues)),
            "arrivals": int(self.total_arrivals),
            "drops": int(self.total_drops),
            "delivered": np.asarray(self.delivered_packets, dtype=np.int64).copy(),
            "delay_index": len(self.served_delays),
            "total_throughput_mbps": float(self.total_throughput_mbps),
            "total_energy_efficiency": float(self.total_energy_efficiency),
            "total_group_size": float(self.total_group_size),
        }

    def measurement_metrics(self):
        """Return metrics accumulated since :meth:`begin_measurement_window`."""
        if self._measurement_start is None:
            raise RuntimeError("Call begin_measurement_window() before measurement_metrics()")
        start = self._measurement_start
        steps = max(self.current_step - start["step"], 1)
        delivered = np.asarray(self.delivered_packets, dtype=np.int64) - start["delivered"]
        if delivered.sum() > 0:
            jfi = float(delivered.sum() ** 2 / (len(delivered) * np.square(delivered).sum()))
        else:
            jfi = 0.0
        window_delays = self.served_delays[start["delay_index"]:]
        arrivals = self.total_arrivals - start["arrivals"]
        drops = self.total_drops - start["drops"]
        queue_end = int(sum(self.queues))
        # Each step adds its delivered-packet rate to the cumulative counter.
        # Dividing its window delta by TXOP count is therefore identical to
        # delivered_packets * packet_bits / (steps * TXOP_duration * 1e6).
        throughput_mbps = (
            self.total_throughput_mbps - start["total_throughput_mbps"]
        ) / steps
        return {
            "steps": steps,
            "jfi": jfi,
            "mean_delay_txops": float(np.mean(window_delays)) if window_delays else 0.0,
            "p95_delay_txops": float(np.percentile(window_delays, 95)) if window_delays else 0.0,
            "p95_hol_txops": float(np.percentile(
                [self._hol_delay(sta_id) for sta_id in range(self.num_stas)], 95
            )),
            "drop_rate": drops / max(arrivals, 1),
            "offered_load_mbps": (
                arrivals * config.COSR_PACKET_SIZE_BITS
                / (steps * config.COSR_TXOP_DURATION_S * 1e6)
            ),
            "throughput_mbps": float(throughput_mbps),
            "mean_group_size": (
                (self.total_group_size - start["total_group_size"]) / steps
            ),
            "energy_efficiency_mbps_per_w": (
                (self.total_energy_efficiency - start["total_energy_efficiency"]) / steps
            ),
            "queue_start_packets": start["queue_packets"],
            "queue_end_packets": queue_end,
            "queue_growth_packets_per_txop": (queue_end - start["queue_packets"]) / steps,
        }

    def step(self, action):
        mask = self.action_masks()
        requested_action = int(action)
        valid = 0 <= requested_action < self.action_space.n and bool(mask[requested_action])
        executed_action = requested_action if valid else 0
        group = self.group_for_action(executed_action)

        rates = {}
        powers = {}
        served_by_sta = {}
        if group:
            powers = allocate_group_power(
                self.aps, self.stas_by_id, group, mode=self.power_mode
            )
            rates = group_phy_rates(self.aps, self.stas_by_id, group, powers)
            payload_duration = max(
                config.COSR_TXOP_DURATION_S
                - config.COSR_FIXED_MAC_OVERHEAD_S
                - config.COSR_COORDINATION_OVERHEAD_S_PER_EXTRA_AP * (len(group) - 1),
                0.0,
            )
            for _, sta_id in group:
                capacity_packets = int(
                    rates[sta_id] * 1e6 * payload_duration
                    // config.COSR_PACKET_SIZE_BITS
                )
                served = min(self.queues[sta_id], capacity_packets)
                served_by_sta[sta_id] = served
                for _ in range(served):
                    arrival_step = self.arrival_steps[sta_id].pop(0)
                    self.served_delays.append(self.current_step - arrival_step + 1)
                self.queues[sta_id] -= served
                self.delivered_packets[sta_id] += served

        served_packets = int(sum(served_by_sta.values()))
        throughput_mbps = (
            served_packets * config.COSR_PACKET_SIZE_BITS
            / config.COSR_TXOP_DURATION_S / 1e6
        )
        total_tx_power = float(sum(powers.values()))
        power_consumption = total_tx_power + len(group) * config.AP_CIRCUIT_POWER_W
        energy_efficiency = throughput_mbps / max(power_consumption, 1e-12)

        self.current_step += 1
        if self.mobility_enabled:
            update_mobility(self.stas, self.aps)
        self._add_arrivals(initial=False)

        delivered = np.asarray(self.delivered_packets, dtype=float)
        if delivered.sum() > 0.0:
            jfi = float(delivered.sum() ** 2 / (len(delivered) * np.square(delivered).sum()))
        else:
            jfi = 0.0
        hol_delays = np.asarray([self._hol_delay(i) for i in range(self.num_stas)], dtype=float)
        p95_hol = float(np.percentile(hol_delays, 95)) if hol_delays.size else 0.0
        mean_served_delay = float(np.mean(self.served_delays)) if self.served_delays else 0.0
        p95_served_delay = float(np.percentile(self.served_delays, 95)) if self.served_delays else 0.0

        max_group_rate = len(group) * max(config.COSR_MCS_RATES_MBPS) if group else max(config.COSR_MCS_RATES_MBPS)
        throughput_reward = min(throughput_mbps / max_group_rate, 1.0)
        delay_penalty = min(p95_hol / config.COSR_MAX_HOL_DELAY_TXOPS, 1.0)
        # Queue occupancy is directly observable and reacts immediately to a
        # poor scheduling decision, unlike tail delay which is delayed and
        # sparse.  Including it makes the RL objective consistent with the
        # reported latency/queue metrics without changing any baseline policy.
        queue_fraction = sum(self.queues) / max(self.num_stas * config.COSR_MAX_QUEUE_PACKETS, 1)
        delay_cost = 0.25 * delay_penalty
        queue_cost = 0.35 * queue_fraction
        reward = float(np.clip(
            throughput_reward - delay_cost - queue_cost,
            -1.0, 1.0,
        ))

        self.total_throughput_mbps += throughput_mbps
        self.total_energy_efficiency += energy_efficiency
        self.total_group_size += len(group)

        terminated = False
        truncated = self.current_step >= self.max_steps
        next_mask = self.action_masks()
        info = {
            "requested_action": requested_action,
            "executed_action": executed_action,
            "invalid_action": not valid,
            "scheduled_group": group,
            "scheduled_group_size": len(group),
            "rates_mbps": rates,
            "powers_w": powers,
            "served_packets": served_packets,
            "served_packets_by_sta": served_by_sta,
            "throughput": float(throughput_mbps),
            "jfi": jfi,
            "queue_packets": int(sum(self.queues)),
            "mean_served_delay_txops": mean_served_delay,
            "p95_served_delay_txops": p95_served_delay,
            "p95_hol_delay_txops": p95_hol,
            "packet_drop_rate": self.total_drops / max(self.total_arrivals, 1),
            "total_tx_power": total_tx_power,
            "energy_efficiency": float(energy_efficiency),
            "reward_terms": {
                "throughput": float(throughput_reward),
                "delay_cost": float(delay_cost),
                "queue_cost": float(queue_cost),
                "total": reward,
            },
            "action_mask": next_mask,
            "valid_action_count": int(next_mask.sum()),
        }
        return self._observation(), reward, terminated, truncated, info
