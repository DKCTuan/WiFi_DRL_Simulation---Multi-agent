import unittest
import numpy as np

import config
from env.topology import setup_aps
from env.water_filling import apply_water_filling, _channel_cost

class TestWeightedWaterFilling(unittest.TestCase):
    def test_weighted_wf_budget_and_bounds(self):
        aps = setup_aps()
        # Create STAs with diverse service classes
        stas = []
        sta_id = 0
        for ap in aps:
            for s_idx, s_class in enumerate(config.SERVICE_CLASS_NAMES):
                stas.append({
                    'id': sta_id,
                    'ap_id': ap['id'],
                    'x': ap['x'] + (s_idx + 1) * 2.0,
                    'y': ap['y'] + (s_idx + 1) * 2.0,
                    'shadowing_db': 0.0,
                    'service_class': s_class
                })
                sta_id += 1

        active_aps = list(aps)
        apply_water_filling(aps, stas, active_aps)

        powers = np.asarray([ap["tx_power"] for ap in active_aps])
        self.assertLessEqual(float(powers.sum()), config.TOTAL_POWER_BUDGET_W + 1e-6)
        self.assertTrue(np.all(powers >= config.TX_POWER_LEVELS[0] - 1e-12))
        self.assertTrue(np.all(powers <= config.P_MAX + 1e-12))

    def test_priority_allocation_bias(self):
        aps = setup_aps()
        # AP 0 serves only high priority Video STAs
        # AP 1 serves only low priority IoT STAs
        stas = [
            {'id': 0, 'ap_id': 0, 'x': aps[0]['x'] + 5.0, 'y': aps[0]['y'], 'shadowing_db': 0.0, 'service_class': 'video'},
            {'id': 1, 'ap_id': 1, 'x': aps[1]['x'] + 5.0, 'y': aps[1]['y'], 'shadowing_db': 0.0, 'service_class': 'iot'},
        ]
        active_aps = [aps[0], aps[1]]
        apply_water_filling(aps, stas, active_aps)
        # AP 0 with video users should receive >= power than AP 1 with iot users
        self.assertGreaterEqual(aps[0]['tx_power'], aps[1]['tx_power'])

if __name__ == "__main__":
    unittest.main()
