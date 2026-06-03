# env/wifi_env.py
import numpy as np
import sys
import os
import gymnasium as gym
from gymnasium import spaces

# Thêm đường dẫn để import được config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class WiFiEnv(gym.Env):
    def __init__(self):
        super().__init__()
        print("Đang khởi tạo Môi trường Multi-Agent WiFi (Nâng cấp chuẩn Research: Smooth Mobility + VIB)...")
        
        self.num_agents = config.NUM_APS 
        self.agent_ids = [f"ap_{i}" for i in range(self.num_agents)]
        
        self.action_space = spaces.Dict({
            agent_id: spaces.Discrete(3) for agent_id in self.agent_ids
        })
        
        num_cca_levels = len(config.CCA_THRESHOLDS)
        self.observation_space = spaces.Dict({
            agent_id: spaces.Box(low=0, high=num_cca_levels-1, shape=(1,), dtype=np.float32)
            for agent_id in self.agent_ids
        })
        
        self.max_steps = 100
        self.current_step = 0
        
        self.aps = []
        self.stas = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.aps = self._setup_aps()   
        self.stas = self._setup_stas() # Khởi tạo vị trí + Shadowing cố định + Vector vận tốc mượt
        
        obs_dict = {}
        for agent_id in self.agent_ids:
            obs_dict[agent_id] = np.array([0.0], dtype=np.float32)
            
        return obs_dict, {}

    def step(self, action_dict):
        self.current_step += 1
        
        # A. CẬP NHẬT CCA CỦA CÁC AP TỪ LỆNH AI
        obs_dict = {}
        for i, agent_id in enumerate(self.agent_ids):
            action = action_dict[agent_id]
            current_idx = config.CCA_THRESHOLDS.index(self.aps[i]['cca_threshold'])
            
            if action == 0:
                new_idx = max(0, current_idx - 1)
            elif action == 2:
                new_idx = min(len(config.CCA_THRESHOLDS) - 1, current_idx + 1)
            else:
                new_idx = current_idx
                
            self.aps[i]['cca_threshold'] = config.CCA_THRESHOLDS[new_idx]
            obs_dict[agent_id] = np.array([float(new_idx)], dtype=np.float32)

        # 🌟 CẢI TIẾN 2: Mô hình Mobility mượt (Velocity Vector), loại bỏ Teleport giật cục
        for sta in self.stas:
            associated_ap = self.aps[sta['ap_id']]
            
            # Cập nhật vị trí mượt theo vận tốc
            sta['x'] += sta['vx']
            sta['y'] += sta['vy']
            
            # Đổi hướng nhẹ ngẫu nhiên (chuyển động quán tính của con người)
            sta['vx'] += np.random.uniform(-0.02, 0.02)
            sta['vy'] += np.random.uniform(-0.02, 0.02)
            
            # Giới hạn tốc độ max_speed = 0.3 m/step
            speed = np.sqrt(sta['vx']**2 + sta['vy']**2)
            max_speed = 0.3
            if speed > max_speed:
                sta['vx'] *= max_speed / speed
                sta['vy'] *= max_speed / speed
                
            # Kiểm tra vùng phủ sóng, nếu chạy ra rìa > 22m thì dội ngược vector vận tốc lại
            dist_to_ap = np.sqrt((sta['x'] - associated_ap['x'])**2 + (sta['y'] - associated_ap['y'])**2)
            if dist_to_ap > 22.0:
                sta['vx'] *= -1
                sta['vy'] *= -1

        # B1. Lọc AP active theo xác suất CSMA/CA mềm (Contention Probability)
        active_aps = self._get_active_aps()
        
        # B2. Chạy thuật toán Water-Filling có màng lọc Smoothing 0.8 / 0.2 tránh sốc công suất
        self._apply_water_filling(active_aps)

        # B3. Tính toán thông lượng mạng tổng
        total_throughput, jain_index = self.calculate_network_throughput()
        
        # C. TÍNH TOÁN HÀM THƯỞNG CÂN BẰNG REWARD VARIANCE (Chuẩn mục số 4)
        rewards_dict = {}
        ap_throughputs = self._calculate_individual_ap_throughput(active_aps)
        
        for i, agent_id in enumerate(self.agent_ids):
            local_thr = ap_throughputs[i]
            interference_penalty = 0
            if self.aps[i] in active_aps:
                for other_ap in active_aps:
                    if other_ap['id'] != i:
                        dist = self._calculate_distance(self.aps[i], other_ap)
                        # Truyền kèm theo shadowing_db mặc định của môi trường nền = 0 cho liên AP
                        gain = self._calculate_channel_gain(dist, shadowing_db=0)
                        interference_penalty += self.aps[i]['tx_power'] * gain
            
            # Cấu hình Reward mới: Giảm biên độ sốc thông lượng, đẩy cấu trúc JFI tuyến tính hóa (*100)
            rewards_dict[agent_id] = (0.3 * local_thr) + (0.4 * total_throughput) + (0.3 * (jain_index * 100)) - (1e3 * interference_penalty)

        terminated = False
        truncated = bool(self.current_step >= self.max_steps)
        
        info = {
            "throughput": total_throughput, 
            "jfi": jain_index,
            "ap_individual_throughputs": ap_throughputs
        }
        
        return obs_dict, rewards_dict, terminated, truncated, info
    
    # --- CÁC HÀM TÍNH TOÁN VẬT LÝ VÀ THUẬT TOÁN NỀN ---
    def _setup_aps(self):
        side = config.AREA_SIZE
        ap_coords = [(0, 0), (side, 0), (side/2, side * np.sqrt(3)/2)]
        aps = []
        for i, coord in enumerate(ap_coords):
            aps.append({
                'id': i, 'x': coord[0], 'y': coord[1],
                'cca_threshold': config.DEFAULT_CCA, 'tx_power': config.P_MAX
            })
        return aps

    def _setup_stas(self):
        stas = []
        sta_id = 0
        for ap in self.aps:
            for _ in range(config.NUM_STAS_PER_AP):
                radius = np.random.uniform(1, 20)
                angle = np.random.uniform(0, 2 * np.pi)
                stas.append({
                    'id': sta_id, 'ap_id': ap['id'], 
                    'x': ap['x'] + radius * np.cos(angle), 'y': ap['y'] + radius * np.sin(angle),
                    # 🌟 CẢI TIẾN 1: Găm cấu trúc Shadowing DB cố định theo từng STA đơn lẻ khi khởi tạo
                    'shadowing_db': np.random.normal(0, config.SHADOW_STD),
                    # 🌟 CẢI TIẾN 2: Vector vận tốc ban đầu cho di chuyển mượt
                    'vx': np.random.uniform(-0.1, 0.1),
                    'vy': np.random.uniform(-0.1, 0.1)
                })
                sta_id += 1
        return stas

    def _calculate_distance(self, node1, node2):
        return np.sqrt((node1['x'] - node2['x'])**2 + (node1['y'] - node2['y'])**2)

    # 🌟 CẢI TIẾN 1: Nhận tham số găm shadowing_db cố định từ cấu trúc STA truyền vào
    def _calculate_channel_gain(self, distance, shadowing_db=0):
        d = max(distance, 0.1)
        term1 = 40.05
        term2 = 20 * np.log10(config.FREQ_C / 2.4)
        term3 = 20 * np.log10(min(d, config.D_BP))
        term4 = 35 * np.log10(d / config.D_BP) if d > config.D_BP else 0
        
        # Không dùng lệnh random ngẫu nhiên liên tục ở đây nữa!
        path_loss_db = term1 + term2 + term3 + term4 + config.L_W + shadowing_db
        return 10 ** (-path_loss_db / 10)

    # 🌟 CẢI TIẾN 5: Chuyển đổi cơ chế ON/OFF thô lậu sang Contention Probability mềm mại của CSMA/CA
    def _get_active_aps(self):
        active_aps = []
        for ap in self.aps:
            interference_received = 0
            for other_ap in self.aps:
                if other_ap['id'] != ap['id']:
                    dist = self._calculate_distance(ap, other_ap)
                    gain = self._calculate_channel_gain(dist, shadowing_db=0)
                    interference_received += other_ap['tx_power'] * gain
            
            cca_watt = 10 ** (ap['cca_threshold'] / 10) * 1e-3
            
            # Tính toán xác suất va chạm cạnh tranh kênh truyền theo hàm mũ mềm
            activity_prob = np.exp(-interference_received / (cca_watt * 10)) # Scale nhẹ để giữ độ nhạy
            if np.random.rand() < activity_prob:
                active_aps.append(ap)
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
                # Kèm cặp đúng mã định danh Shadowing cố định của STA
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

    def calculate_network_throughput(self):
        active_aps = self._get_active_aps()
        total_throughput_bps = 0
        sta_throughputs = [] 
        
        for ap in active_aps:
            bss_stas = [sta for sta in self.stas if sta['ap_id'] == ap['id']]
            if len(bss_stas) == 0: continue
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
            sum_throughput = sum(sta_throughputs)
            sum_sq_throughput = sum([x**2 for x in sta_throughputs])
            jain_index = (sum_throughput**2) / (len(sta_throughputs) * sum_sq_throughput)
        else:
            jain_index = 0.0

        return total_throughput_mbps, jain_index

    # 🌟 CẢI TIẾN 3: Thuật toán Water-Filling có bộ lọc Smoothing công suất tránh giật cục bộ
    def _apply_water_filling(self, active_aps):
        if not active_aps:
            return
        noise_levels = []
        for ap in active_aps:
            interference = 0
            for other_ap in active_aps:
                if other_ap['id'] != ap['id']:
                    dist = self._calculate_distance(other_ap, ap)
                    gain = self._calculate_channel_gain(dist, shadowing_db=0)
                    interference += config.P_MAX * gain
            
            total_noise_penalty = interference + config.NOISE_POWER
            noise_levels.append(total_noise_penalty)

        mu_low = 0.0
        mu_high = 10.0
        mu_optimal = 0.0
        
        for _ in range(30):
            mu_mid = (mu_low + mu_high) / 2
            total_allocated = 0
            for noise in noise_levels:
                power = max(mu_mid - noise, 0)
                power = min(power, config.P_MAX)
                total_allocated += power
                
            if total_allocated > (config.P_MAX * self.num_agents):
                mu_high = mu_mid
            else:
                mu_low = mu_mid
                mu_optimal = mu_mid

        for i, ap in enumerate(active_aps):
            optimal_power = max(mu_optimal - noise_levels[i], 0)
            optimal_power = min(optimal_power, config.P_MAX)
            
            # Áp bộ lọc trượt quán tính công suất: Giữ 80% cấu hình cũ, chỉ đổi 20% theo kịch bản mới
            old_power = self.aps[ap['id']]['tx_power']
            smoothed_power = (0.8 * old_power) + (0.2 * optimal_power)
            self.aps[ap['id']]['tx_power'] = smoothed_power