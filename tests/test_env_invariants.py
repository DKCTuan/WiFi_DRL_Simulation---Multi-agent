import unittest
from unittest.mock import patch
import numpy as np
import config
from env.active_set import get_active_aps
from env.metrics import _sinr, compute_network_metrics
from env.topology import setup_aps, setup_stas
from env.water_filling import apply_water_filling
from env.wifi_env import WiFiEnv
from training.utils import build_global_state


class CoreInvariantTests(unittest.TestCase):
    def test_hybrid_updates_cca_once_per_step(self):
        env = WiFiEnv(
            verbose=False,
            fixed_topology=True,
            fixed_seed=7,
            mobility_enabled=False,
            use_water_filling=True,
            action_size=config.HYBRID_ACTION_SIZE,
        )
        env.reset(seed=7)
        actions = {agent_id: 0 for agent_id in env.agent_ids}

        from env import wifi_env as wifi_env_module

        with patch.object(
            wifi_env_module,
            "get_active_aps",
            wraps=wifi_env_module.get_active_aps,
        ) as mocked:
            env.step(actions)
        self.assertEqual(mocked.call_count, 1)


    def test_water_filling_respects_total_budget_and_bounds(self):
        aps = setup_aps()
        stas = setup_stas(aps, [4] * config.NUM_APS, base_seed=11)
        active_aps = list(aps)
        apply_water_filling(aps, stas, active_aps)

        powers = np.asarray([ap["tx_power"] for ap in active_aps])
        self.assertLessEqual(float(powers.sum()), config.TOTAL_POWER_BUDGET_W + 1e-6)
        self.assertTrue(np.all(powers >= config.TX_POWER_LEVELS[0] - 1e-12))
        self.assertTrue(np.all(powers <= config.P_MAX + 1e-12))


    def test_inactive_ap_does_not_create_cca_interference(self):
        aps = setup_aps()
        # Only AP 0 transmitted in the previous slot.
        active_state = [True] + [False] * (config.NUM_APS - 1)
        ewma_state = [None] * config.NUM_APS
        get_active_aps(
            aps,
            ewma_state,
            active_state,
            ewma_alpha=1.0,
            hysteresis_db=0.0,
        )
        # AP 0 cannot sense interference from inactive neighbors.
        self.assertAlmostEqual(ewma_state[0], -150.0)


    def test_generalization_seed_changes_sta_topology(self):
        env = WiFiEnv(verbose=False, fixed_topology=False, mobility_enabled=False)
        env.reset(seed=100)
        xy_100 = np.asarray([(sta["x"], sta["y"]) for sta in env.stas])
        env.reset(seed=100)
        xy_100_repeat = np.asarray([(sta["x"], sta["y"]) for sta in env.stas])
        env.reset(seed=101)
        xy_101 = np.asarray([(sta["x"], sta["y"]) for sta in env.stas])

        np.testing.assert_allclose(xy_100, xy_100_repeat)
        self.assertFalse(np.allclose(xy_100, xy_101))


    def test_qmix_global_state_uses_all_observations(self):
        env = WiFiEnv(verbose=False, fixed_topology=True, fixed_seed=3)
        observations, _ = env.reset(seed=3)
        state = build_global_state(
            observations,
            [0.0] * env.num_agents,
            [0.0] * env.num_agents,
            env.agent_ids,
        )
        self.assertEqual(state.shape, (env.num_agents * config.OBS_SIZE,))


    def test_vectorized_network_metrics_match_scalar_rate_calculation(self):
        aps = setup_aps()
        stas = setup_stas(aps, [3] * config.NUM_APS, base_seed=31)
        active_aps = [aps[index] for index in (0, 1, 3, 4)]
        measured = compute_network_metrics(
            aps, stas, active_aps, config.BANDWIDTH
        )
        active_ids = {ap["id"] for ap in active_aps}
        counts = np.bincount([sta["ap_id"] for sta in stas], minlength=len(aps))
        airtime = [1.0] * len(aps)
        scalar_rates = []
        scalar_ap_rates = [0.0] * len(aps)
        for sta in stas:
            ap_id = sta["ap_id"]
            rate = 0.0
            if ap_id in active_ids:
                rate = (
                    config.BANDWIDTH * airtime[ap_id] / counts[ap_id]
                    * np.log2(1.0 + _sinr(
                        aps[ap_id], sta, active_aps, config.NOISE_POWER
                    ))
                )
            scalar_rates.append(rate)
            scalar_ap_rates[ap_id] += rate
        scalar_total = sum(scalar_rates)
        scalar_jfi = scalar_total ** 2 / (len(stas) * sum(r ** 2 for r in scalar_rates))
        self.assertAlmostEqual(measured["throughput_mbps"], scalar_total / 1e6, places=9)
        self.assertAlmostEqual(measured["jfi"], scalar_jfi, places=12)
        np.testing.assert_allclose(
            measured["ap_throughputs_mbps"], np.asarray(scalar_ap_rates) / 1e6,
            rtol=1e-12, atol=1e-12,
        )
