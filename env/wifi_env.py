import copy
import sys
import os

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Thêm đường dẫn để import được config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from env.topology import setup_aps, setup_stas
from env.user_demand import initialize_users, update_users
from env.scenarios import ap_positions, place_edge_stas
from env.mobility import update_mobility
from env.active_set import get_active_aps
from env.water_filling import (
    apply_water_filling,
    enforce_power_budget,
)
from env.metrics import (
    compute_network_metrics,
    update_jfi_ema,
)
from env.observation import build_observation
from training.objectives import (
    reward_components as build_reward_components,
    throughput_temporal_episode_score,
    throughput_temporal_step_reward,
)


class WiFiEnv(gym.Env):
    """
    Môi trường Multi-Agent WiFi (research version).

    File này chỉ còn phần ORCHESTRATION (gym API: __init__/reset/step) và
    lắp ráp lời gọi tới các module vật lý/thuật toán nằm trong env/*.py:
      - topology.py      : sinh vị trí AP/STA          (Section 2)
      - channel.py        : path-loss + channel gain     (Section 2.1, eq 1)
      - mobility.py       : cập nhật vị trí STA          (Section 2.2)
      - active_set.py     : fresh random-order slotted carrier sensing (V5)
      - water_filling.py  : phân bổ công suất             (Section 4.5, eq 11-12)
      - metrics.py         : rate/throughput/JFI + JFI-EMA (Section 2.3, eq 3-4)
      - observation.py     : vector quan sát 7 chiều       (Section 4.1)

    Việc tách file này chủ yếu để dễ debug: mỗi công thức trong bài báo có
    đúng 1 file tương ứng, thay vì gộp hết vào 1 file 500+ dòng.
    """

    # 4 STA đầu mỗi AP luôn sinh với seed cố định → vị trí nhất quán qua mọi K
    BASE_STAS_PER_AP = 4
    BASE_STAS_SEED = 9999

    def __init__(
        self,
        verbose=True,
        fixed_topology=False,
        fixed_seed=None,
        mobility_enabled=True,
        use_water_filling=None,
        action_size=config.FULL_AI_ACTION_SIZE,
        num_stas_per_ap=None,   # None → dùng config.AP_LOAD_PROFILE (heterogeneous)
        cooperation_enabled=False,
        access_mode=None,
        ap_positions=None,
    ):
        super().__init__()
        if verbose:
            print("Đang khởi tạo Môi trường Multi-Agent WiFi (Nâng cấp chuẩn Research: Smooth Mobility + VIB)...")
        self.fixed_topology = fixed_topology
        self.fixed_seed = fixed_seed
        self.mobility_enabled = mobility_enabled
        self.use_water_filling = config.USE_WATER_FILLING_BASELINE if use_water_filling is None else use_water_filling
        self.action_size = action_size
        self.reward_objective_mode = config.REWARD_OBJECTIVE_MODE
        if self.reward_objective_mode not in {'legacy', 'throughput_temporal_jfi'}:
            raise ValueError('Unknown reward objective mode')
        if cooperation_enabled:
            raise ValueError('Cooperation is excluded from the core branch')
        self.cooperation_enabled = False
        self.coordination_pipeline = 'disabled'
        self.access_mode = config.CCA_ACCESS_MODE if access_mode is None else access_mode
        if self.access_mode not in {'slot_contention', 'legacy_ewma'}:
            raise ValueError(f'Unknown CCA access mode: {self.access_mode}')
        self.ap_positions = copy.deepcopy(ap_positions)
        self.observation_size = config.OBS_SIZE
        if config.USER_DEMAND_ENABLED:
            self.observation_size += config.USER_DEMAND_OBS_SIZE
        # Nếu truyền num_stas_per_ap (int) thì override đồng đều cho tất cả AP,
        # ngược lại dùng AP_LOAD_PROFILE không đồng đều từ config
        if num_stas_per_ap is not None:
            self.ap_load_profile = [num_stas_per_ap] * config.NUM_APS
        else:
            self.ap_load_profile = list(config.AP_LOAD_PROFILE)
        self._fixed_aps = None
        self._fixed_stas = None

        self.num_agents = config.NUM_APS
        self.agent_ids = [f"ap_{i}" for i in range(self.num_agents)]

        # Stationary reward: training and validation both use this same weight.
        self.active_ap_weight = config.REWARD_ACTIVE_AP_WEIGHT

        self.action_space = spaces.Dict({
            agent_id: spaces.Discrete(self.action_size)
            for agent_id in self.agent_ids
        })

        self.observation_space = spaces.Dict({
            agent_id: spaces.Box(low=0, high=1, shape=(self.observation_size,), dtype=np.float32)
            for agent_id in self.agent_ids
        })

        self.max_steps = config.TRAIN_STEPS_PER_EPISODE
        self.current_step = 0

        self.aps = []
        self.stas = []
        self._cca_interference_ewma_dbm = [None] * self.num_agents
        self._cca_active_state = [True] * self.num_agents
        self._jfi_reward_ema = None  # trạng thái EMA cho reward (xem metrics.update_jfi_ema)

    def _build_obs(self, agent_idx, local_throughput, active_aps):
        return build_observation(
            agent_idx, self.aps, self.stas, local_throughput, active_aps,
            self.ap_load_profile,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self._cca_interference_ewma_dbm = [None] * self.num_agents
        self._cca_active_state = [True] * self.num_agents
        self._jfi_reward_ema = None
        # Separate streams keep contention independent of exploration and
        # mobility, and make paired controls consume exactly one order/step.
        access_seed = int(seed) if seed is not None else int(self.np_random.integers(0, 2**63))
        self._access_rng = np.random.default_rng(np.random.SeedSequence([access_seed, 0xCCA]))
        self._ap_access_counts = np.zeros(self.num_agents, dtype=np.int64)
        self._ap_idle_steps = np.zeros(self.num_agents, dtype=np.int64)
        self._ap_max_idle_steps = np.zeros(self.num_agents, dtype=np.int64)
        self.scenario = copy.deepcopy((options or {}).get('scenario'))
        if self.scenario is not None:
            self.aps = setup_aps(ap_positions(self.scenario))
            self.stas = setup_stas(self.aps, self.ap_load_profile, self.scenario['layout_seed'])
            place_edge_stas(self.scenario, self.aps, self.stas)
        elif self.fixed_topology:
            if self._fixed_aps is None or self._fixed_stas is None:
                rng_state = np.random.get_state()
                if self.fixed_seed is not None:
                    np.random.seed(self.fixed_seed)
                self.aps = setup_aps(self.ap_positions)
                topology_seed = (
                    self.fixed_seed
                    if self.fixed_seed is not None
                    else self.__class__.BASE_STAS_SEED
                )
                self.stas = setup_stas(self.aps, self.ap_load_profile, topology_seed)
                self._fixed_aps = copy.deepcopy(self.aps)
                self._fixed_stas = copy.deepcopy(self.stas)
                np.random.set_state(rng_state)
            self.aps = copy.deepcopy(self._fixed_aps)
            self.stas = copy.deepcopy(self._fixed_stas)
        else:
            self.aps = setup_aps(self.ap_positions)
            # A non-fixed environment must actually generate a new STA
            # topology. reset(seed=...) makes this deterministic for eval.
            topology_seed = int(self.np_random.integers(0, 2**31 - 1))
            self.stas = setup_stas(self.aps, self.ap_load_profile, topology_seed)

        if config.USER_DEMAND_ENABLED:
            initialize_users(self.stas)
        self._sta_rate_sum = np.zeros(len(self.stas), dtype=float)
        self._previous_temporal_jfi = 0.0
        self._throughput_norm_sum = 0.0
        obs_dict = {}
        for i, agent_id in enumerate(self.agent_ids):
            obs_dict[agent_id] = self._build_obs(i, 0.0, active_aps=[])  # Truyền list rỗng lúc reset

        return obs_dict, {}

    def step(self, action_dict):
        self.current_step += 1
        # CCA senses the previous slot, before applying this slot's power action.
        sensing_aps = [dict(ap) for ap in self.aps]

        # A. CẬP NHẬT CCA CỦA CÁC AP TỪ LỆNH AI
        for i, agent_id in enumerate(self.agent_ids):
            action = int(action_dict[agent_id])

            # Direct action decoding.
            # Hybrid-WF: action selects CCA; Water-Filling allocates power.
            # Full-AI: action selects one CCA-power pair.
            if self.use_water_filling:
                cca_idx = int(np.clip(action, 0, len(config.CCA_THRESHOLDS) - 1))
                self.aps[i]['cca_threshold'] = config.CCA_THRESHOLDS[cca_idx]
                continue

            num_power_levels = len(config.TX_POWER_LEVELS)
            action = int(np.clip(action, 0, config.FULL_AI_ACTION_SIZE - 1))
            cca_idx = action // num_power_levels
            pwr_idx = action % num_power_levels
            self.aps[i]['cca_threshold'] = config.CCA_THRESHOLDS[cca_idx]
            self.aps[i]['tx_power'] = config.TX_POWER_LEVELS[pwr_idx]
            continue

        # B. MÔ HÌNH MOBILITY MƯỢT (VELOCITY VECTOR)
        if self.mobility_enabled:
            update_mobility(self.stas, self.aps, rng=self.np_random)

        # C. CẬP NHẬT TRẠNG THÁI MẠNG VÀ PHÂN BỔ CÔNG SUẤT
        priority_order = self._access_rng.permutation(self.num_agents).tolist()
        sensed_active = get_active_aps(
            [
                dict(ap, cca_threshold=self.aps[ap['id']]['cca_threshold'])
                for ap in sensing_aps
            ],
            self._cca_interference_ewma_dbm,
            self._cca_active_state,
            config.CCA_EWMA_ALPHA,
            config.CCA_HYSTERESIS_DB,
            access_mode=self.access_mode,
            priority_order=priority_order,
        )
        active_aps = [self.aps[ap['id']] for ap in sensed_active]
        if not self.use_water_filling:
            enforce_power_budget(active_aps)
        if self.use_water_filling:
            apply_water_filling(self.aps, self.stas, active_aps)
            # Do not call get_active_aps() twice in one environment step. It
            # mutates EWMA/hysteresis state, which previously made Hybrid CCA
            # dynamics run twice as fast as Full-AI. The active set is chosen
            # at slot start; WF allocates power inside that fixed set.

        rate_metrics = compute_network_metrics(
            self.aps, self.stas, active_aps, config.BANDWIDTH
        )
        # Retain zero-valued legacy log fields, without a coordination engine.
        cooperation_links, cooperation_details = {}, []
        coordination_power_reduction = coordination_utility_delta = 0.0
        coordination_requested_link_count = signaling_requests = 0
        request_load = [0] * self.num_agents

        total_throughput = rate_metrics["throughput_mbps"]
        jain_index = rate_metrics["jfi"]
        ap_throughputs = rate_metrics["ap_throughputs_mbps"]
        self._sta_rate_sum += np.asarray(rate_metrics['sta_throughputs_mbps'])
        user_status = (update_users(self.stas, self._sta_rate_sum, self.current_step)
                       if config.USER_DEMAND_ENABLED else [])
        squares = float(np.square(self._sta_rate_sum).sum())
        jfi_time_average = (float(self._sta_rate_sum.sum()) ** 2
                            / (len(self.stas) * squares)) if squares else 0.0
        active_ids = {ap['id'] for ap in active_aps}
        for ap_id in range(self.num_agents):
            self._ap_access_counts[ap_id] += int(ap_id in active_ids)
            self._ap_idle_steps[ap_id] = 0 if ap_id in active_ids else self._ap_idle_steps[ap_id] + 1
        self._ap_max_idle_steps = np.maximum(self._ap_max_idle_steps, self._ap_idle_steps)
        active_ratio = len(active_aps) / self.num_agents
        total_tx_power = max(sum(ap['tx_power'] for ap in active_aps), 1e-9)
        # Sleeping APs do not pay active circuit power. This keeps the energy
        # metric consistent with the reward for using fewer active APs.
        total_power_consumption = total_tx_power + (len(active_aps) * config.AP_CIRCUIT_POWER_W)
        energy_efficiency = total_throughput / total_power_consumption

        obs_dict = {}
        for i, agent_id in enumerate(self.agent_ids):
            obs_dict[agent_id] = self._build_obs(i, ap_throughputs[i], active_aps)

        # D. TÍNH TOÁN HÀM THƯỞNG CÂN BẰNG
        # Fix JFI dao động: dùng EMA của jain_index làm tín hiệu công bằng
        # cho reward (nếu bật), thay vì jain_index tức thời — xem config.py.
        # info["jfi"] báo cáo VẪN LÀ jain_index gốc, không đổi.
        if config.REWARD_JFI_SMOOTHING_ENABLED:
            self._jfi_reward_ema = update_jfi_ema(
                self._jfi_reward_ema, jain_index, config.REWARD_JFI_EMA_ALPHA
            )
            jfi_for_reward = self._jfi_reward_ema
        else:
            jfi_for_reward = jain_index

        if self.reward_objective_mode == 'throughput_temporal_jfi':
            throughput_norm = min(max(total_throughput /
                                      config.REWARD_TOTAL_THROUGHPUT_REF_MBPS, 0.0), 1.0)
            self._throughput_norm_sum += throughput_norm
            aligned_step_reward = throughput_temporal_step_reward(
                throughput_norm, jfi_time_average, self._previous_temporal_jfi,
                self.max_steps,
            )
            self._previous_temporal_jfi = jfi_time_average
            agent_rewards = [aligned_step_reward] * self.num_agents
            reward_components = {
                'total_thr_norm': throughput_norm,
                'jfi': float(jain_index),
                'temporal_jfi': float(jfi_time_average),
                'aligned_step_reward': float(aligned_step_reward),
            }
        else:
            agent_rewards, reward_components = build_reward_components(
                total_throughput,
                jfi_for_reward,
                ap_throughputs,
                active_ratio,
                active_ap_weight=self.active_ap_weight,
            )
        rewards_dict = dict(zip(self.agent_ids, agent_rewards))
        team_reward = float(np.mean(list(rewards_dict.values())))

        # This objective is genuinely finite-horizon. Its terminal temporal
        # JFI must not bootstrap into a non-existent next episode.
        terminated = (self.reward_objective_mode == 'throughput_temporal_jfi'
                      and self.current_step >= self.max_steps)
        truncated = (self.reward_objective_mode != 'throughput_temporal_jfi'
                     and self.current_step >= self.max_steps)

        info = {
            "throughput": total_throughput,
            "jfi": jain_index,                 # metric gốc, KHÔNG bị làm mượt — dùng cho Fig 1-4/eval
            "jfi_time_average": jfi_time_average,
            "sta_throughputs_mbps": rate_metrics['sta_throughputs_mbps'],
            "sta_capacity_mbps": rate_metrics['sta_capacity_mbps'],
            "user_status": user_status,
            "users_needing_rescue": sum(row['needs_rescue'] for row in user_status),
            "ap_access_fraction": (self._ap_access_counts / self.current_step).tolist(),
            "ap_max_idle_steps": self._ap_max_idle_steps.tolist(),
            "access_mode": self.access_mode,
            "contention_order": priority_order,
            "jfi_reward_ema": jfi_for_reward,   # giá trị thực sự dùng để tính reward (debug/theo dõi)
            "ap_individual_throughputs": ap_throughputs,
            "active_ap_count": len(active_aps),
            "active_ap_ids": [ap["id"] for ap in active_aps],
            "active_ratio": active_ratio,
            "cooperation_enabled": self.cooperation_enabled,
            "coordination_pipeline": self.coordination_pipeline,
            "coordination_link_count": sum(len(v) for v in cooperation_links.values()),
            "coordinated_sta_count": len(cooperation_links),
            "cooperation_links": cooperation_details,
            "coordination_power_reduction_w": coordination_power_reduction,
            "coordination_requested_link_count": coordination_requested_link_count,
            "coordination_signaling_requests": signaling_requests,
            "coordination_request_load": request_load,
            "coordination_utility_delta": coordination_utility_delta,
            "total_tx_power": total_tx_power,
            "total_power_consumption": total_power_consumption,
            "energy_efficiency": energy_efficiency,
            "team_reward": team_reward,
            "reward_objective_mode": self.reward_objective_mode,
            "objective_score_so_far": (
                throughput_temporal_episode_score(
                    self._throughput_norm_sum / self.current_step,
                    jfi_time_average,
                ) if self.reward_objective_mode == 'throughput_temporal_jfi'
                else None
            ),
            "reward_components": dict(
                reward_components,
                jfi_reward_ema=float(jfi_for_reward),
            ),
        }

        return obs_dict, rewards_dict, terminated, truncated, info
