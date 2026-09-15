"""Fast regression tests for environment/model invariants (no training required)."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

import config
from env.active_set import get_active_aps
from env.cooperation import build_cooperation_links, decode_cooperative_action
from env.cosr_baselines import HierarchicalUCBPolicy
from env.cosr_env import CoSREnv
from env.cosr_phy import allocate_group_power, group_phy_rates, is_compatible_group, standalone_rate
from env.mcs import sinr_to_mcs_rate_mbps
from env.metrics import _sinr
from env.topology import setup_aps, setup_stas
from env.water_filling import apply_water_filling
from env.wifi_env import WiFiEnv
from training.utils import build_global_state
from training.cosr_dqn import imitation_scale_for_episode, masked_argmax


class EnvironmentInvariantTests(unittest.TestCase):
    def test_masked_dqn_never_selects_invalid_high_value_action(self):
        values = torch.tensor([[1.0, 99.0, 2.0], [3.0, 1.0, 9.0]])
        masks = torch.tensor([[True, False, True], [True, True, False]])
        self.assertEqual(masked_argmax(values, masks).tolist(), [2, 0])

    def test_imitation_floor_prevents_teacher_loss_from_vanishing(self):
        self.assertEqual(
            imitation_scale_for_episode(
                episode=225, episodes=300, imitation_weight=0.30, imitation_floor=0.05,
            ),
            0.05,
        )
        self.assertEqual(
            imitation_scale_for_episode(
                episode=225, episodes=300, imitation_weight=0.30, imitation_floor=0.0,
            ),
            0.0,
        )


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

    def test_cooperative_action_and_observation_dimensions(self):
        env = WiFiEnv(
            verbose=False,
            fixed_topology=True,
            fixed_seed=5,
            mobility_enabled=False,
            use_water_filling=True,
            action_size=config.COOPERATIVE_HYBRID_ACTION_SIZE,
            cooperation_enabled=True,
            num_stas_per_ap=4,
        )
        observations, _ = env.reset(seed=5)
        self.assertEqual(
            observations["ap_0"].shape, (config.COOPERATIVE_OBS_SIZE,)
        )
        self.assertTrue(np.all(observations["ap_0"] >= 0.0))
        self.assertTrue(np.all(observations["ap_0"] <= 1.0))

        # Mode index 0 means no coordination for every AP.
        cca_idx = config.CCA_THRESHOLDS.index(config.DEFAULT_CCA)
        no_coop_action = cca_idx * len(config.COOPERATION_TARGET_OFFSETS)
        actions = {agent_id: no_coop_action for agent_id in env.agent_ids}
        next_observations, _, _, _, info = env.step(actions)
        self.assertEqual(info["coordination_link_count"], 0)
        self.assertEqual(
            next_observations["ap_0"].shape, (config.COOPERATIVE_OBS_SIZE,)
        )

    def test_user_centric_link_selects_sta_and_suppresses_interference(self):
        aps = setup_aps()
        stas = setup_stas(aps, [4] * config.NUM_APS, base_seed=17)
        links, details = build_cooperation_links(
            aps, stas, aps, target_requests={0: 1}
        )
        self.assertEqual(len(details), 1)
        self.assertEqual(details[0]["helper_ap"], 0)
        self.assertEqual(details[0]["serving_ap"], 1)
        sta = next(sta for sta in stas if sta["id"] == details[0]["sta_id"])
        serving_ap = aps[sta["ap_id"]]
        sinr_without = _sinr(serving_ap, sta, aps, config.NOISE_POWER, {})
        sinr_with = _sinr(serving_ap, sta, aps, config.NOISE_POWER, links)
        self.assertGreater(sinr_with, sinr_without)

    def test_cooperative_action_decoding(self):
        num_modes = len(config.COOPERATION_TARGET_OFFSETS)
        cca_idx, offset = decode_cooperative_action(2 * num_modes + 2)
        self.assertEqual(cca_idx, 2)
        self.assertEqual(offset, 1)

    def test_cosr_mcs_is_discrete_and_monotonic(self):
        sinrs = [10 ** (db / 10) for db in (-20, -5, 0, 10, 20, 40)]
        rates = [sinr_to_mcs_rate_mbps(value) for value in sinrs]
        self.assertEqual(rates[0], 0.0)
        self.assertTrue(all(b >= a for a, b in zip(rates, rates[1:])))
        self.assertLessEqual(len(set(rates)), len(config.COSR_MCS_RATES_MBPS) + 1)

    def test_cosr_action_mask_combines_phy_and_queue_constraints(self):
        env = CoSREnv(num_stas_per_ap=2, fixed_seed=19, max_steps=2)
        observation, info = env.reset(seed=19)
        self.assertEqual(observation.shape, (3 * 2 * config.NUM_APS,))
        self.assertEqual(info["action_mask"].shape, (env.action_space.n,))
        self.assertTrue(info["action_mask"][0])

        env.queues = [0] * env.num_stas
        empty_mask = env.action_masks()
        self.assertEqual(int(empty_mask.sum()), 1)
        _, _, _, _, step_info = env.step(1)
        self.assertTrue(step_info["invalid_action"])
        self.assertEqual(step_info["executed_action"], 0)

    def test_cosr_pf_pair_stays_on_pareto_edge_and_keeps_group_feasible(self):
        aps = [
            {"id": 0, "x": 0.0, "y": 0.0},
            {"id": 1, "x": 10.0, "y": 0.0},
        ]
        stas = {
            0: {"id": 0, "ap_id": 0, "x": 1.0, "y": 0.0, "shadowing_db": 0.0},
            1: {"id": 1, "ap_id": 1, "x": 9.0, "y": 0.0, "shadowing_db": 0.0},
        }
        group = ((0, 0), (1, 1))
        self.assertTrue(is_compatible_group(aps, stas, group))
        powers = allocate_group_power(aps, stas, group, mode="pf_pair")
        self.assertTrue(any(abs(value - config.P_MAX) < 1e-12 for value in powers.values()))
        rates = group_phy_rates(aps, stas, group, powers)
        for ap_id, sta_id in group:
            self.assertGreaterEqual(
                len(group) * rates[sta_id] + config.COSR_COMPATIBILITY_TOLERANCE,
                standalone_rate(aps, stas[sta_id], ap_id),
            )

    def test_cosr_measurement_window_excludes_warmup(self):
        env = CoSREnv(
            num_stas_per_ap=2,
            fixed_seed=23,
            arrival_rate_per_sta=0.30,
            max_steps=4,
        )
        _, info = env.reset(seed=23)
        first_action = next(index for index in np.flatnonzero(info["action_mask"]) if index != 0)
        env.step(first_action)
        env.begin_measurement_window()
        for _ in range(3):
            mask = env.action_masks()
            action = next(index for index in np.flatnonzero(mask) if index != 0)
            env.step(action)
        metrics = env.measurement_metrics()
        self.assertEqual(metrics["steps"], 3)
        self.assertGreater(metrics["offered_load_mbps"], 0.0)
        self.assertIn("queue_growth_packets_per_txop", metrics)
        self.assertIn("throughput_mbps", metrics)
        self.assertIn("mean_group_size", metrics)
        self.assertIn("energy_efficiency_mbps_per_w", metrics)

    def test_cosr_fixed_topology_ignores_traffic_reset_seed(self):
        env = CoSREnv(num_stas_per_ap=2, fixed_topology=True, fixed_seed=2026)
        env.reset(seed=101)
        xy_101 = np.asarray([(sta["x"], sta["y"]) for sta in env.stas])
        env.reset(seed=102)
        xy_102 = np.asarray([(sta["x"], sta["y"]) for sta in env.stas])
        np.testing.assert_allclose(xy_101, xy_102)

    def test_factorized_ucb_updates_links_not_joint_actions(self):
        env = CoSREnv(num_stas_per_ap=2, fixed_seed=29, max_steps=2)
        _, info = env.reset(seed=29)
        policy = HierarchicalUCBPolicy(env)
        action = policy.select(env, info["action_mask"])
        _, _, _, _, step_info = env.step(action)
        policy.update(env, step_info["executed_action"], step_info)
        if action != 0:
            self.assertLessEqual(len(policy.link_counts), len(env.group_for_action(action)))
        self.assertFalse(hasattr(policy, "action_counts"))


if __name__ == "__main__":
    unittest.main()
