"""Regression coverage for fresh-slot access and a stationary objective."""
import os
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import config
from env.active_set import get_active_aps, get_slot_active_aps
from env.topology import setup_aps
from env.wifi_env import WiFiEnv
from training.evaluate import evaluate_policy
from training.objectives import team_utility, validate_stationary_reward


class V5Tests(unittest.TestCase):
    def make_env(self, **kwargs):
        return WiFiEnv(verbose=False, fixed_topology=True, fixed_seed=123,
                       mobility_enabled=False, use_water_filling=True,
                       action_size=config.HYBRID_ACTION_SIZE, num_stas_per_ap=2,
                       **kwargs)

    def test_every_slot_releases_previous_winner(self):
        aps = setup_aps()
        # All pairs strongly sense one another: only the first contender wins.
        for ap in aps:
            ap.update(x=50 + ap['id'] * .1, y=50, cca_threshold=-82)
        sensed, active = [None]*6, [True]*6
        for first in range(6):
            order = list(range(first, 6)) + list(range(first))
            got = get_slot_active_aps(aps, order, sensed, active)
            self.assertEqual([ap['id'] for ap in got], [first])

    def test_access_is_equivariant_under_ap_id_permutation(self):
        aps = setup_aps()
        for ap in aps:
            ap['cca_threshold'] = -78
        order = [4, 1, 5, 0, 2, 3]
        mapping = [2, 4, 0, 5, 1, 3]
        renamed = [dict(ap, id=mapping[ap['id']]) for ap in aps]
        actual = get_slot_active_aps(aps, order, [None]*6, [False]*6)
        permuted = get_slot_active_aps(renamed, [mapping[i] for i in order], [None]*6, [False]*6)
        self.assertEqual({mapping[a['id']] for a in actual}, {a['id'] for a in permuted})

    def test_order_validation(self):
        with self.assertRaises(ValueError):
            get_slot_active_aps(setup_aps(), [0]*6, [None]*6, [False]*6)

    def test_low_cca_no_permanent_ap0_lockin(self):
        env = self.make_env()
        env.reset(seed=100)
        for _ in range(600):
            *_, info = env.step({aid: 0 for aid in env.agent_ids})
        self.assertTrue(np.all(np.asarray(info['ap_access_fraction']) > .08))
        self.assertLess(max(info['ap_max_idle_steps']), 100)
        self.assertGreater(info['jfi_time_average'], .5)

    def test_contention_and_mobility_streams_are_action_independent(self):
        a, b = self.make_env(), self.make_env()
        a.mobility_enabled = b.mobility_enabled = True
        a.reset(seed=17)
        b.reset(seed=17)
        for _ in range(20):
            *_, ia = a.step({aid: 0 for aid in a.agent_ids})
            np.random.random(50)
            *_, ib = b.step({aid: 6 for aid in b.agent_ids})
            self.assertEqual(ia['contention_order'], ib['contention_order'])
        np.testing.assert_array_equal([(s['x'], s['y']) for s in a.stas],
                                      [(s['x'], s['y']) for s in b.stas])

    def test_reset_replays_contention_and_clears_temporal_metrics(self):
        env = self.make_env()
        orders = []
        for _ in range(2):
            env.reset(seed=77)
            self.assertEqual(env._ap_access_counts.sum(), 0)
            self.assertEqual(env._sta_rate_sum.sum(), 0)
            trace = []
            for _ in range(8):
                *_, info = env.step({aid: 0 for aid in env.agent_ids})
                trace.append(info['contention_order'])
            orders.append(trace)
        self.assertEqual(*orders)

    def test_temporal_fairness_computed_from_rate_sums(self):
        env = self.make_env()
        env.reset(seed=3)
        sums = np.zeros(len(env.stas))
        for _ in range(30):
            *_, info = env.step({aid: 0 for aid in env.agent_ids})
            sums += info['sta_throughputs_mbps']
        expected = sums.sum()**2 / (len(sums) * np.square(sums).sum())
        self.assertAlmostEqual(info['jfi_time_average'], expected)

    def test_custom_geometry_is_preserved_and_validated(self):
        xy = [[10+i*12, 20+(i%2)*35] for i in range(6)]
        env = self.make_env(ap_positions=xy)
        xy[0][0] = 99  # Constructor owns its copy.
        env.reset(seed=3)
        self.assertEqual(env.aps[0]['x'], 10)
        self.assertEqual(env.aps[5]['y'], 55)
        with self.assertRaises(ValueError):
            setup_aps([[1, 2]])

    def test_reward_matches_reported_team_objective(self):
        for episode in (0, 25, 150, 500):
            env = self.make_env()
            env.reset(seed=episode)
            _, rewards, _, _, info = env.step({aid: 0 for aid in env.agent_ids})
            expected = team_utility(info['throughput'], info['jfi'],
                                    info['ap_individual_throughputs'], info['active_ratio'])
            self.assertAlmostEqual(info['team_reward'], expected)
            self.assertAlmostEqual(np.mean(list(rewards.values())), expected)
            self.assertEqual(info['reward_components']['active_ap_weight'], .1)

    def test_validation_matches_identical_manual_rollout(self):
        class FixedAgent:
            epsilon = .2
            def act(self, state):
                return 0
        agents = {f'ap_{i}': FixedAgent() for i in range(6)}
        with patch.object(config, 'TRAIN_MOBILITY_ENABLED', False):
            result = evaluate_policy(agents, episodes=1, steps=25, seed_base=77,
                                     use_water_filling=True, action_size=7,
                                     eval_mode='fixed', num_stas=2)
        env = WiFiEnv(verbose=False, fixed_topology=True, fixed_seed=config.TRAIN_SCENARIO_SEED,
                      mobility_enabled=False, use_water_filling=True, action_size=7,
                      num_stas_per_ap=2)
        env.reset(seed=77)
        values = []
        for _ in range(25):
            *_, info = env.step({aid: 0 for aid in env.agent_ids})
            values.append(info['team_reward'])
        self.assertAlmostEqual(result['team_reward'], np.mean(values))
        self.assertAlmostEqual(result['jfi_time_average'], info['jfi_time_average'])
        self.assertTrue(all(a.epsilon == .2 for a in agents.values()))

    def test_legacy_reward_switches_fail_early(self):
        with patch.object(config, 'REWARD_ACTIVE_AP_WARMUP_EPISODES', 150):
            with self.assertRaisesRegex(ValueError, 'stationary'):
                validate_stationary_reward()
        with patch.object(config, 'REWARD_JFI_SMOOTHING_ENABLED', True):
            with self.assertRaisesRegex(ValueError, 'shield'):
                validate_stationary_reward()

    def test_source_hash_independent_of_directory_enumeration_order(self):
        from training import checkpoints
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            (root/'training').mkdir()
            (root/'env').mkdir()
            (root/'training'/'a.py').write_text('a=1', encoding='utf-8')
            (root/'env'/'b.py').write_text('b=2', encoding='utf-8')
            listing = list(os.walk(root))
            with patch.object(checkpoints, '__file__', str(root/'training'/'checkpoints.py')):
                with patch.object(checkpoints.os, 'walk', return_value=iter(listing)):
                    first = checkpoints.project_source_sha256()
                with patch.object(checkpoints.os, 'walk', return_value=iter(reversed(listing))):
                    second = checkpoints.project_source_sha256()
            self.assertEqual(first, second)


if __name__ == '__main__':
    unittest.main()
