import unittest
import numpy as np

import config
from env.topology import setup_aps, setup_stas
from env.observation import build_observation
from env.metrics import compute_qos_metrics
from env.wifi_env import WiFiEnv

class TestServiceAware(unittest.TestCase):
    def test_service_classes_assigned(self):
        aps = setup_aps()
        stas = setup_stas(aps, [10] * config.NUM_APS, base_seed=42)
        classes = [s['service_class'] for s in stas]
        for c in config.SERVICE_CLASS_NAMES:
            self.assertIn(c, classes)

    def test_observation_dimensions(self):
        env = WiFiEnv(verbose=False, fixed_topology=True, fixed_seed=42)
        obs, _ = env.reset(seed=42)
        for agent_id, o in obs.items():
            self.assertEqual(len(o), config.OBS_SIZE)
            self.assertEqual(len(o), 11)  # 7 base + 4 service classes
            # Verify the service proportions sum to 1.0
            self.assertAlmostEqual(float(np.sum(o[7:11])), 1.0, places=4)

    def test_qos_metrics_calculation(self):
        stas = [
            {'id': 0, 'service_class': 'video'},  # req: 5.0 Mbps
            {'id': 1, 'service_class': 'voice'},  # req: 0.5 Mbps
            {'id': 2, 'service_class': 'web'},    # req: 1.0 Mbps
            {'id': 3, 'service_class': 'iot'},    # req: 0.05 Mbps
        ]
        rates = [6.0, 0.4, 2.0, 0.1]
        # video meets 6.0 >= 5.0 (satisfied)
        # voice fails 0.4 < 0.5 (violation = 0.1)
        # web meets 2.0 >= 1.0 (satisfied)
        # iot meets 0.1 >= 0.05 (satisfied)
        # Satisfaction rate = 3 / 4 = 75%
        sat_rate, violation, class_sat, w_thr = compute_qos_metrics(stas, rates)
        self.assertEqual(sat_rate, 0.75)
        self.assertAlmostEqual(violation, 0.1 / 4.0)
        self.assertEqual(class_sat['video'], 1.0)
        self.assertEqual(class_sat['voice'], 0.0)
        self.assertEqual(class_sat['web'], 1.0)
        self.assertEqual(class_sat['iot'], 1.0)

if __name__ == "__main__":
    unittest.main()
