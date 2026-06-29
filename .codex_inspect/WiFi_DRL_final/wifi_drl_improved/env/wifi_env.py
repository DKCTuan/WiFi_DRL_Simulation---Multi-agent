import numpy as np
import sys
import os
import copy
import gymnasium as gym
from gymnasium import spaces

# Thêm đường dẫn để import được config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class WiFiEnv(gym.Env):
    def __init__(
        self,
        verbose=True,
        fixed_topology=False,
        fixed_seed=None,
        mobility_enabled=True,
        use_water_filling=None,
        action_size=config.FULL_AI_ACTION_SIZE
    ):
        super().__init__()
        if verbose:
            print("Đang khởi tạo Môi trường Multi-Agent WiFi (Nâng cấp chuẩn Research: Smooth Mobility + VIB)...")
        self.fixed_topology = fixed_topology
        self.fixed_seed = fixed_seed
        self.mobility_enabled = mobility_enabled
        self.use_water_filling = config.USE_WATER_FILLING_BASELINE if use_water_filling is None else use_water_filling
        self.action_size = action_size
        self._fixed_aps = None
        self._fixed_stas = None

        self.num_agents = config.NUM_APS
        self.agent_ids = [f"ap_{i}" for i in range(self.num_agents)]

        self.action_space = spaces.Dict({
            agent_id: spaces.Discrete(self.action_size)
            for agent_id in self.agent_ids
        })

        self.observation_space = spaces.Dict({
            agent_id: spaces.Box(low=0, high=1, shape=(config.OBS_SIZE,), dtype=np.float32)
            for agent_id in self.agent_ids
        })

        self.max_steps = config.TRAIN_STEPS_PER_EPISODE
        self.current_step = 0

        self.aps = []
        self.stas = []

    def _build_obs(self, agent_idx, local_throughput, active_aps):
        ap = self.aps[agent_idx]
        bss_stas = [s for s in self.stas if s['ap_id'] == agent_idx]
        active_ap_ids = {ap['id'] for ap in active_aps}

        # SINR trung bình của các STA thuộc AP này
        sinr_list = []
        for sta in bss_stas:
            dist = self._calculate_distance(ap, sta)
            gain = self._calculate_channel_gain(dist, sta['shadowing_db'])
            signal = ap['tx_power'] * gain
            interf = sum(
                o['tx_power'] * self._calculate_channel_gain(self._calculate_distance(o, sta), sta['shadowing_db'])
                for o in active_aps if o['id'] != agent_idx
            )
            sinr_list.append(signal / (interf + config.NOISE_POWER))

        avg_sinr = float(np.mean(sinr_list)) if sinr_list else 0.0
        avg_sinr_db = 10 * np.log10(avg_sinr + 1e-10)

        cca_idx = config.CCA_THRESHOLDS.index(ap['cca_threshold'])

        return np.array([
            cca_idx / (len(config.CCA_THRESHOLDS) - 1),          # CCA norm [0,1]
            min(local_throughput / 150.0, 1.0),                   # Throughput norm
            ap['tx_power'] / config.P_MAX,                        # TX power norm
            np.clip((avg_sinr_db + 10) / 50, 0, 1),              # SINR norm
            len(bss_stas) / (max(config.AP_LOAD_PROFILE) * 1.5), # Load norm
            float(ap['id'] in active_ap_ids),                    # AP này có đang active không
            len(active_ap_ids) / self.num_agents,                 # Tỷ lệ AP active toàn mạng
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        if self.fixed_topology:
            if self._fixed_aps is None or self._fixed_stas is None:
                rng_state = np.random.get_state()
                if self.fixed_seed is not None:
                    np.random.seed(self.fixed_seed)
                self.aps = self._setup_aps()
                self.stas = self._setup_stas()
                self._fixed_aps = copy.deepcopy(self.aps)
                self._fixed_stas = copy.deepcopy(self.stas)
                np.random.set_state(rng_state)
            self.aps = copy.deepcopy(self._fixed_aps)
            self.stas = copy.deepcopy(self._fixed_stas)
        else:
            self.aps = self._setup_aps()
            self.stas = self._setup_stas()

        obs_dict = {}
        for i, agent_id in enumerate(self.agent_ids):
            obs_dict[agent_id] = self._build_obs(i, 0.0, active_aps=[])  # ✅ Truyền list rỗng lúc reset

        return obs_dict, {}

    def step(self, action_dict):
        self.current_step += 1

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
            for sta in self.stas:
                associated_ap = self.aps[sta['ap_id']]

                sta['x'] += sta['vx']
                sta['y'] += sta['vy']

                sta['vx'] += np.random.uniform(-0.02, 0.02)
                sta['vy'] += np.random.uniform(-0.02, 0.02)

                speed = np.sqrt(sta['vx']**2 + sta['vy']**2)
                max_speed = 0.3
                if speed > max_speed:
                    sta['vx'] *= max_speed / speed
                    sta['vy'] *= max_speed / speed

                dist_to_ap = np.sqrt((sta['x'] - associated_ap['x'])**2 + (sta['y'] - associated_ap['y'])**2)
                if dist_to_ap > 22.0:
                    sta['vx'] *= -1
                    sta['vy'] *= -1

        # CẬP NHẬT TRẠNG THÁI MẠNG VÀ PHÂN BỔ CÔNG SUẤT
        active_aps = self._get_active_aps()
        if self.use_water_filling:
            self._apply_water_filling(active_aps)
            active_aps = self._get_active_aps()
        total_throughput, jain_index = self._compute_throughput_from(active_aps)
        ap_throughputs = self._calculate_individual_ap_throughput(active_aps)
        active_ratio = len(active_aps) / self.num_agents
        total_tx_power = max(sum(ap['tx_power'] for ap in active_aps), 1e-9)
        total_power_consumption = total_tx_power + (config.NUM_APS * config.AP_CIRCUIT_POWER_W)
        energy_efficiency = total_throughput / total_power_consumption
        obs_dict = {}
        for i, agent_id in enumerate(self.agent_ids):
            obs_dict[agent_id] = self._build_obs(i, ap_throughputs[i], active_aps)

        # C. TÍNH TOÁN HÀM THƯỞNG CÂN BẰNG
        rewards_dict = {}
        total_thr_norm = float(np.clip(total_throughput / config.REWARD_TOTAL_THROUGHPUT_REF_MBPS, 0, 1))

        for i, agent_id in enumerate(self.agent_ids):
            local_thr = ap_throughputs[i]
            local_thr_norm = float(np.clip(local_thr / config.LOCAL_THROUGHPUT_REF_MBPS, 0, 1))

            reward = (
                config.REWARD_GLOBAL_WEIGHT * total_thr_norm +
                config.REWARD_LOCAL_WEIGHT * local_thr_norm +
                config.REWARD_FAIRNESS_WEIGHT * jain_index
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
            "jfi": jain_index,
            "ap_individual_throughputs": ap_throughputs,
            "active_ap_count": len(active_aps),
            "active_ratio": active_ratio,
            "total_tx_power": total_tx_power,
            "total_power_consumption": total_power_consumption,
            "energy_efficiency": energy_efficiency,
            "team_reward": team_reward,
            "reward_components": {
                "total_thr_norm": total_thr_norm,
                "mean_local_thr_norm": mean_local_thr_norm,
                "jfi": jain_index,
                "active_ratio": active_ratio,
            },
        }

        return obs_dict, rewards_dict, terminated, truncated, info

    # --- CÁC HÀM TÍNH TOÁN VẬT LÝ VÀ THUẬT TOÁN NỀN THÔ ---
    def _setup_aps(self):
        aps = []
        side = config.AREA_SIZE
        for i in range(config.NUM_APS):
            # Xếp AP đều theo vòng tròn quanh tâm vùng phủ sóng
            angle = 2 * np.pi * i / config.NUM_APS
            radius = side * 0.35  # 35% kích thước vùng
            x = side / 2 + radius * np.cos(angle)
            y = side / 2 + radius * np.sin(angle)
            aps.append({
                'id': i, 'x': x, 'y': y,
                'cca_threshold': config.DEFAULT_CCA,
                'tx_power': config.TX_POWER_LEVELS[2]  # Mức công suất mặc định = 0.05W
            })
        return aps

    def _setup_stas(self):
        stas = []
        sta_id = 0
        for ap in self.aps:
            sta_count = config.AP_LOAD_PROFILE[ap['id']]
            radius_max = config.AP_STA_RADIUS_MAX[ap['id']]
            shadow_bias = config.AP_SHADOW_BIAS_DB[ap['id']]
            for _ in range(sta_count):
                radius = np.random.uniform(1, radius_max)
                angle = np.random.uniform(0, 2 * np.pi)
                stas.append({
                    'id': sta_id, 'ap_id': ap['id'],
                    'x': ap['x'] + radius * np.cos(angle),
                    'y': ap['y'] + radius * np.sin(angle),
                    # Station-level shadowing approximation; per-link shadowing would change the channel model.
                    'shadowing_db': np.random.normal(shadow_bias, config.SHADOW_STD),
                    'vx': np.random.uniform(-0.1, 0.1),
                    'vy': np.random.uniform(-0.1, 0.1)
                })
                sta_id += 1
        return stas

    def _calculate_distance(self, node1, node2):
        return np.sqrt((node1['x'] - node2['x'])**2 + (node1['y'] - node2['y'])**2)

    def _calculate_channel_gain(self, distance, shadowing_db=0):
        d = max(distance, 0.1)
        term1 = 40.05
        term2 = 20 * np.log10(config.FREQ_C / 2.4)
        term3 = 20 * np.log10(min(d, config.D_BP))
        term4 = 35 * np.log10(d / config.D_BP) if d > config.D_BP else 0

        path_loss_db = term1 + term2 + term3 + term4 + config.L_W + shadowing_db
        return 10 ** (-path_loss_db / 10)

    def _get_active_aps(self):
        active_aps = []
        for ap in self.aps:
            interference_received = 0
            for other_ap in self.aps:
                if other_ap['id'] != ap['id']:
                    dist = self._calculate_distance(ap, other_ap)
                    gain = self._calculate_channel_gain(dist, shadowing_db=0)
                    interference_received += other_ap['tx_power'] * gain

            # Đổi sang dBm để so sánh đúng với CCA threshold
            if interference_received > 0:
                interference_dbm = 10 * np.log10(interference_received / 1e-3)
            else:
                interference_dbm = -150.0

            # AP active khi nhiễu nhận được THẤP HƠN ngưỡng CCA
            if interference_dbm < ap['cca_threshold']:
                active_aps.append(ap)

        # FIX: Nếu không AP nào active, chọn AP có CCA threshold cao nhất (dễ active hơn)
        if len(active_aps) == 0:
            active_aps.append(max(self.aps, key=lambda a: a['cca_threshold']))

        return active_aps

    def _calculate_individual_ap_throughput(self, active_aps):
        throughputs = [0.0] * self.num_agents
        for ap in active_aps:
            bss_stas = [sta for sta in self.stas if sta['ap_id'] == ap['id']]
            if len(bss_stas) == 0: continue
            user_bandwidth = config.BANDWIDTH / len(bss_stas)

            ap_thr_bps = 0
            for sta in bss_stas:
                dist_signal = self._calculate_distance(ap, sta)
                gain_signal = self._calculate_channel_gain(dist_signal, sta['shadowing_db'])
                signal_power = ap['tx_power'] * gain_signal

                interference_power = 0
                for other_ap in active_aps:
                    if other_ap['id'] != ap['id']:
                        dist_interf = self._calculate_distance(other_ap, sta)
                        gain_interf = self._calculate_channel_gain(dist_interf, sta['shadowing_db'])
                        interference_power += other_ap['tx_power'] * gain_interf

                sinr = signal_power / (interference_power + config.NOISE_POWER)
                ap_thr_bps += user_bandwidth * np.log2(1 + sinr)
            throughputs[ap['id']] = ap_thr_bps / 1e6
        return throughputs

    def _compute_throughput_from(self, active_aps):
        total_throughput_bps = 0
        sta_throughputs = []
        active_ap_ids = {ap['id'] for ap in active_aps}

        for ap in self.aps:
            bss_stas = [sta for sta in self.stas if sta['ap_id'] == ap['id']]
            if len(bss_stas) == 0:
                continue

            if ap['id'] not in active_ap_ids:
                sta_throughputs.extend([0.0] * len(bss_stas))
                continue

            user_bandwidth = config.BANDWIDTH / len(bss_stas)

            for sta in bss_stas:
                dist_signal = self._calculate_distance(ap, sta)
                gain_signal = self._calculate_channel_gain(dist_signal, sta['shadowing_db'])
                signal_power = ap['tx_power'] * gain_signal

                interference_power = 0
                for other_ap in active_aps:
                    if other_ap['id'] != ap['id']:
                        dist_interf = self._calculate_distance(other_ap, sta)
                        gain_interf = self._calculate_channel_gain(dist_interf, sta['shadowing_db'])
                        interference_power += other_ap['tx_power'] * gain_interf

                sinr = signal_power / (interference_power + config.NOISE_POWER)
                throughput = user_bandwidth * np.log2(1 + sinr)
                sta_throughputs.append(throughput)
                total_throughput_bps += throughput

        total_throughput_mbps = total_throughput_bps / 1e6

        if len(sta_throughputs) > 0 and sum(sta_throughputs) > 0:
            sum_thr = sum(sta_throughputs)
            sum_sq = sum(x**2 for x in sta_throughputs)
            jain_index = (sum_thr**2) / (len(sta_throughputs) * sum_sq)
        else:
            jain_index = 0.0

        return total_throughput_mbps, jain_index

    def _apply_water_filling(self, active_aps):
        if not active_aps:
            return

        channel_costs = []
        for ap in active_aps:
            bss_stas = [sta for sta in self.stas if sta['ap_id'] == ap['id']]
            if not bss_stas:
                channel_costs.append(float("inf"))
                continue

            gains = [
                self._calculate_channel_gain(
                    self._calculate_distance(ap, sta),
                    sta['shadowing_db']
                )
                for sta in bss_stas
            ]
            avg_gain = max(float(np.mean(gains)), 1e-18)

            interference = 0
            for other_ap in active_aps:
                if other_ap['id'] != ap['id']:
                    link_losses = [
                        self._calculate_channel_gain(
                            self._calculate_distance(other_ap, sta),
                            sta['shadowing_db']
                        )
                        for sta in bss_stas
                    ]
                    interference += other_ap['tx_power'] * float(np.mean(link_losses))

            channel_costs.append((interference + config.NOISE_POWER) / avg_gain)

        mu_low = 0.0
        finite_costs = [cost for cost in channel_costs if np.isfinite(cost)]
        if not finite_costs:
            return

        budget = min(config.TOTAL_POWER_BUDGET_W, len(active_aps) * config.P_MAX)
        mu_high = config.P_MAX + max(finite_costs)
        mu_optimal = 0.0

        for _ in range(30):
            mu_mid = (mu_low + mu_high) / 2
            total_allocated = 0
            for cost in channel_costs:
                power = 0.0 if not np.isfinite(cost) else max(mu_mid - cost, 0)
                power = min(power, config.P_MAX)
                total_allocated += power

            if total_allocated > budget:
                mu_high = mu_mid
            else:
                mu_low = mu_mid
                mu_optimal = mu_mid

        for i, ap in enumerate(active_aps):
            if not np.isfinite(channel_costs[i]):
                optimal_power = 0.0
            else:
                optimal_power = max(mu_optimal - channel_costs[i], 0)
            optimal_power = min(optimal_power, config.P_MAX)
            self.aps[ap['id']]['tx_power'] = max(optimal_power, config.TX_POWER_LEVELS[0])
