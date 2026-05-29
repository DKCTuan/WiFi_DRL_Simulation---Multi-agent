# env/wifi_env.py
import numpy as np
import sys
import os

# Thêm đường dẫn để import được config từ thư mục gốc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class WiFiEnv:
    def __init__(self):
        print("Đang khởi tạo Môi trường WiFi 6 (OBSS/PD)...")
        self.aps = self._setup_aps()
        self.stas = self._setup_stas()
        
    def _setup_aps(self):
        """Khởi tạo 3 AP tạo thành tam giác đều cạnh 50m"""
        side = config.AREA_SIZE
        ap_coords = [
            (0, 0),                                # AP 0
            (side, 0),                             # AP 1
            (side/2, side * np.sqrt(3)/2)          # AP 2
        ]
        
        aps = []
        for i, coord in enumerate(ap_coords):
            ap = {
                'id': i,
                'x': coord[0],
                'y': coord[1],
                'cca_threshold': config.DEFAULT_CCA,
                'tx_power': config.P_MAX
            }
            aps.append(ap)
        return aps

    def _setup_stas(self):
        """Rải ngẫu nhiên người dùng (STA) xung quanh mỗi AP"""
        stas = []
        sta_id = 0
        
        # Duyệt qua từng AP, mỗi AP sẽ "đẻ" ra K người dùng
        for ap in self.aps:
            for _ in range(config.NUM_STAS_PER_AP):
                # Rải ngẫu nhiên trong bán kính 20m quanh AP
                radius = np.random.uniform(1, 20)
                angle = np.random.uniform(0, 2 * np.pi)
                
                sta_x = ap['x'] + radius * np.cos(angle)
                sta_y = ap['y'] + radius * np.sin(angle)
                
                sta = {
                    'id': sta_id,
                    'ap_id': ap['id'], # Ghi nhớ STA này thuộc về AP nào
                    'x': sta_x,
                    'y': sta_y
                }
                stas.append(sta)
                sta_id += 1
                
        print(f"Đã rải xong {len(stas)} người dùng (STA) vào bản đồ.")
        return stas
    def _calculate_distance(self, node1, node2):
        """Tính khoảng cách vật lý giữa 2 điểm (AP và STA hoặc AP và AP)"""
        return np.sqrt((node1['x'] - node2['x'])**2 + (node1['y'] - node2['y'])**2)

    def _calculate_channel_gain(self, distance):
        """Tính độ suy hao sóng (Path Loss) theo công thức của bài báo"""
        d = max(distance, 0.1) # Tránh lỗi chia cho 0
        term1 = 40.05
        term2 = 20 * np.log10(config.FREQ_C / 2.4)
        term3 = 20 * np.log10(min(d, config.D_BP))
        term4 = 35 * np.log10(d / config.D_BP) if d > config.D_BP else 0
        
        shadowing = np.random.normal(0, config.SHADOW_STD)
        path_loss_db = term1 + term2 + term3 + term4 + config.L_W + shadowing
        
        # Đổi từ dB sang dạng tuyến tính (Linear)
        return 10 ** (-path_loss_db / 10)

    def calculate_network_throughput(self):
        """Tính tổng thông lượng (tốc độ mạng) của toàn hệ thống (Công thức 3 & 4)"""
        total_throughput_mbps = 0
        
        # Đi đếm tốc độ của từng người dùng một
        for sta in self.stas:
            serving_ap = self.aps[sta['ap_id']]
            
            # 1. Tính tín hiệu mong muốn (Sóng từ AP của mình)
            dist_signal = self._calculate_distance(serving_ap, sta)
            gain_signal = self._calculate_channel_gain(dist_signal)
            signal_power = serving_ap['tx_power'] * gain_signal
            
            # 2. Tính Nhiễu (Sóng rác từ các AP hàng xóm)
            interference_power = 0
            for other_ap in self.aps:
                if other_ap['id'] != serving_ap['id']:
                    dist_interf = self._calculate_distance(other_ap, sta)
                    gain_interf = self._calculate_channel_gain(dist_interf)
                    interference_power += other_ap['tx_power'] * gain_interf
                    
            # 3. Tính SINR và Tốc độ Shannon
            sinr = signal_power / (interference_power + config.NOISE_POWER)
            throughput = config.BANDWIDTH * np.log2(1 + sinr) # Đơn vị: bps
            
            total_throughput_mbps += (throughput / 1e6) # Đổi sang Mbps
            
        return total_throughput_mbps
    
    def _get_active_aps(self):
        """Kiểm tra AP nào được phép phát sóng (Vượt qua vòng gửi xe CCA)"""
        active_aps = []
        for ap in self.aps:
            interference_received = 0
            # Đo xem các AP khác dội nhiễu vào AP này bao nhiêu
            for other_ap in self.aps:
                if other_ap['id'] != ap['id']:
                    dist = self._calculate_distance(ap, other_ap)
                    gain = self._calculate_channel_gain(dist)
                    interference_received += other_ap['tx_power'] * gain
                    
            # Đổi CCA từ dBm sang Watt để so sánh
            cca_watt = 10 ** (ap['cca_threshold'] / 10) * 1e-3
            
            # Nếu nhiễu nhận được NHỎ HƠN ngưỡng CCA -> Được phép phát sóng
            if interference_received < cca_watt:
                active_aps.append(ap)
                
        return active_aps

    def calculate_network_throughput(self):
        """Tính thông lượng dựa trên chia sẻ băng thông và cơ chế CCA"""
        # BƯỚC 1: Tìm xem AP nào đang được phát sóng
        active_aps = self._get_active_aps()
        
        total_throughput_mbps = 0
        
        # BƯỚC 2: Duyệt qua các AP đang hoạt động để tính tốc độ
        for ap in active_aps:
            # Tìm những người dùng thuộc về AP này
            bss_stas = [sta for sta in self.stas if sta['ap_id'] == ap['id']]
            if len(bss_stas) == 0: continue
            
            # Chia đều băng thông cho số người dùng
            user_bandwidth = config.BANDWIDTH / len(bss_stas) 
            bss_throughput = 0
            
            for sta in bss_stas:
                # Tính Tín hiệu
                dist_signal = self._calculate_distance(ap, sta)
                gain_signal = self._calculate_channel_gain(dist_signal)
                signal_power = ap['tx_power'] * gain_signal
                
                # Tính Nhiễu (CHỈ tính nhiễu từ các AP ĐANG PHÁT SÓNG)
                interference_power = 0
                for other_ap in active_aps:
                    if other_ap['id'] != ap['id']:
                        dist_interf = self._calculate_distance(other_ap, sta)
                        gain_interf = self._calculate_channel_gain(dist_interf)
                        interference_power += other_ap['tx_power'] * gain_interf
                        
                # Tính Tốc độ (Shannon)
                sinr = signal_power / (interference_power + config.NOISE_POWER)
                throughput = user_bandwidth * np.log2(1 + sinr)
                bss_throughput += throughput
                
            total_throughput_mbps += (bss_throughput / 1e6)
            
        return total_throughput_mbps
if __name__ == "__main__":
    # 1. Tạo môi trường
    env = WiFiEnv()
    
    # 2. Chạy thử 5 lần để xem tốc độ mạng thay đổi ra sao (do nhiễu ngẫu nhiên Shadowing)
    print("\n--- BẮT ĐẦU ĐO KIỂM THÔNG LƯỢNG MẠNG ---")
    sum_throughput = 0
    num_tests = 5
    
    for i in range(num_tests):
        throughput = env.calculate_network_throughput()
        sum_throughput += throughput
        print(f"Lần đo {i+1}: Tổng thông lượng toàn mạng = {throughput:.2f} Mbps")
        
    print(f"\n=> TỐC ĐỘ TRUNG BÌNH: {sum_throughput/num_tests:.2f} Mbps")