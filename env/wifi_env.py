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
from env.mobility import update_mobility
from env.active_set import get_active_aps
from env.water_filling import apply_water_filling
from env.cooperation import build_cooperation_links, decode_cooperative_action
from env.metrics import (
    compute_individual_ap_throughput,
    compute_network_throughput_and_jfi,
    update_jfi_ema,
)
from env.observation import build_observation


class WiFiEnv(gym.Env):
    """
    Môi trường Multi-Agent WiFi (research version).

    File này chỉ còn phần ORCHESTRATION (gym API: __init__/reset/step) và
    lắp ráp lời gọi tới các module vật lý/thuật toán nằm trong env/*.py:
      - topology.py      : sinh vị trí AP/STA          (Section 2)
      - channel.py        : path-loss + channel gain     (Section 2.1, eq 1)
      - mobility.py       : cập nhật vị trí STA          (Section 2.2)
      - active_set.py     : CCA active set + EWMA/hysteresis (eq 2 + smoothing)
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
    ):
        super().__init__()
        if verbose:
            print("Đang khởi tạo Môi trường Multi-Agent WiFi (Nâng cấp chuẩn Research: Smooth Mobility + VIB)...")
        self.fixed_topology = fixed_topology
        self.fixed_seed = fixed_seed
        self.mobility_enabled = mobility_enabled
        self.use_water_filling = config.USE_WATER_FILLING_BASELINE if use_water_filling is None else use_water_filling
        self.action_size = action_size
        self.cooperation_enabled = bool(cooperation_enabled)
        if self.cooperation_enabled and not self.use_water_filling:
            raise ValueError("Multi-AP cooperation currently requires Hybrid water-filling mode")
        if self.cooperation_enabled and self.action_size != config.COOPERATIVE_HYBRID_ACTION_SIZE:
            raise ValueError(
                "Cooperative Hybrid mode requires action_size="
                f"{config.COOPERATIVE_HYBRID_ACTION_SIZE}, got {self.action_size}"
            )
        self.observation_size = (
            config.COOPERATIVE_OBS_SIZE if self.cooperation_enabled else config.OBS_SIZE
        )
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

        # Trọng số reward cho việc tiết kiệm AP — để mặc định = config, nhưng có thể
        # bị main.py ghi đè mỗi episode để làm curriculum (ramp 0 → full) tránh việc
        # agent học "tắt AP" trước khi học tốt throughput/fairness (gây lệch tải ở K thấp).
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
            self.ap_load_profile, cooperation_enabled=self.cooperation_enabled,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self._cca_interference_ewma_dbm = [None] * self.num_agents
        self._cca_active_state = [True] * self.num_agents
        self._jfi_reward_ema = None
        if self.fixed_topology:
            if self._fixed_aps is None or self._fixed_stas is None:
                rng_state = np.random.get_state()
                if self.fixed_seed is not None:
                    np.random.seed(self.fixed_seed)
                self.aps = setup_aps()
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
            self.aps = setup_aps()
            # A non-fixed environment must actually generate a new STA
            # topology. reset(seed=...) makes this deterministic for eval.
            topology_seed = int(self.np_random.integers(0, 2**31 - 1))
            self.stas = setup_stas(self.aps, self.ap_load_profile, topology_seed)

        obs_dict = {}
        for i, agent_id in enumerate(self.agent_ids):
            obs_dict[agent_id] = self._build_obs(i, 0.0, active_aps=[])  # Truyền list rỗng lúc reset

        return obs_dict, {}

    def step(self, action_dict):
        self.current_step += 1
        target_requests = {}

        # A. CẬP NHẬT CCA CỦA CÁC AP TỪ LỆNH AI
        for i, agent_id in enumerate(self.agent_ids):
            action = int(action_dict[agent_id])

            # Direct action decoding.
            # Hybrid-WF: action selects CCA; Water-Filling allocates power.
            # Full-AI: action selects one CCA-power pair.
            if self.cooperation_enabled:
                cca_idx, target_offset = decode_cooperative_action(action)
                self.aps[i]['cca_threshold'] = config.CCA_THRESHOLDS[cca_idx]
                target_requests[i] = (
                    None if target_offset is None
                    else (i + target_offset) % self.num_agents
                )
                continue

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
            update_mobility(self.stas, self.aps)

        # C. CẬP NHẬT TRẠNG THÁI MẠNG VÀ PHÂN BỔ CÔNG SUẤT
        active_aps = get_active_aps(
            self.aps,
            self._cca_interference_ewma_dbm,
            self._cca_active_state,
            config.CCA_EWMA_ALPHA,
            config.CCA_HYSTERESIS_DB,
        )
        if self.use_water_filling:
            apply_water_filling(self.aps, self.stas, active_aps)
            # Do not call get_active_aps() twice in one environment step. It
            # mutates EWMA/hysteresis state, which previously made Hybrid CCA
            # dynamics run twice as fast as Full-AI. The active set is chosen
            # at slot start; WF allocates power inside that fixed set.

        cooperation_links, cooperation_details = ({}, [])
        if self.cooperation_enabled:
            cooperation_links, cooperation_details = build_cooperation_links(
                self.aps, self.stas, active_aps, target_requests
            )

        total_throughput, jain_index = compute_network_throughput_and_jfi(
            self.aps, self.stas, active_aps, config.BANDWIDTH, cooperation_links
        )
        ap_throughputs = compute_individual_ap_throughput(
            self.aps, self.stas, active_aps, config.BANDWIDTH, cooperation_links
        )
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

        rewards_dict = {}
        total_thr_norm = float(np.clip(total_throughput / config.REWARD_TOTAL_THROUGHPUT_REF_MBPS, 0, 1))

        for i, agent_id in enumerate(self.agent_ids):
            local_thr = ap_throughputs[i]
            local_thr_norm = float(np.clip(local_thr / config.LOCAL_THROUGHPUT_REF_MBPS, 0, 1))

            reward = (
                config.REWARD_GLOBAL_WEIGHT * total_thr_norm +
                config.REWARD_LOCAL_WEIGHT * local_thr_norm +
                config.REWARD_FAIRNESS_WEIGHT * jfi_for_reward +
                self.active_ap_weight * (1.0 - active_ratio)
            )
            rewards_dict[agent_id] = float(np.clip(reward, 0, 1))

        mean_local_thr_norm = float(np.mean([
            np.clip(thr / config.LOCAL_THROUGHPUT_REF_MBPS, 0, 1)
            for thr in ap_throughputs
        ]))
        team_reward = float(np.mean(list(rewards_dict.values())))

        terminated = False
        truncated = bool(self.current_step >= self.max_steps)

        info = {
            "throughput": total_throughput,
            "jfi": jain_index,                 # metric gốc, KHÔNG bị làm mượt — dùng cho Fig 1-4/eval
            "jfi_reward_ema": jfi_for_reward,   # giá trị thực sự dùng để tính reward (debug/theo dõi)
            "ap_individual_throughputs": ap_throughputs,
            "active_ap_count": len(active_aps),
            "active_ap_ids": [ap["id"] for ap in active_aps],
            "active_ratio": active_ratio,
            "cooperation_enabled": self.cooperation_enabled,
            "coordination_link_count": sum(len(v) for v in cooperation_links.values()),
            "coordinated_sta_count": len(cooperation_links),
            "cooperation_links": cooperation_details,
            "total_tx_power": total_tx_power,
            "total_power_consumption": total_power_consumption,
            "energy_efficiency": energy_efficiency,
            "team_reward": team_reward,
            "reward_components": {
                "total_thr_norm": total_thr_norm,
                "mean_local_thr_norm": mean_local_thr_norm,
                "jfi": jain_index,
                "jfi_reward_ema": jfi_for_reward,
                "active_ratio": active_ratio,
            },
        }

        return obs_dict, rewards_dict, terminated, truncated, info
