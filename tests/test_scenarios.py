import unittest
import numpy as np
import config
from env.scenarios import scenario_bank, training_scenario, FAMILIES
from env.wifi_env import WiFiEnv
from training.train_loop import train_marl
from training.evaluate import evaluate_policy


class ScenarioTests(unittest.TestCase):
    def test_splits_and_bss_streams_are_disjoint(self):
        sets = [{spec['layout_seed']+i for spec in scenario_bank(split) for i in range(6)}
                for split in ('train','validation','test')]
        for i in range(3):
            for j in range(i):
                self.assertFalse(sets[i] & sets[j])
        for split, streams in zip(('train','validation','test'), sets):
            self.assertEqual(len(streams), len(scenario_bank(split))*6)


    def test_training_cycles_all_families_and_never_uses_holdout(self):
        bank = scenario_bank('train')
        self.assertEqual({training_scenario(i)['family'] for i in range(6)}, set(FAMILIES))
        self.assertEqual([training_scenario(i) for i in range(len(bank))], bank)
        self.assertEqual(training_scenario(len(bank)), bank[0])


    def test_reset_changes_ap_geometry_and_reproduces_each_spec(self):
        env = WiFiEnv(verbose=False, fixed_topology=True, fixed_seed=7)
        first,second = scenario_bank('train')[1:3]
        env.reset(seed=0, options={'scenario':first})
        original = [(a['x'],a['y']) for a in env.aps]
        stas = [(s['x'],s['y']) for s in env.stas]
        env.reset(seed=0, options={'scenario':second})
        self.assertFalse(np.allclose(original,[(a['x'],a['y']) for a in env.aps]))
        env.reset(seed=0, options={'scenario':first})
        np.testing.assert_array_equal(stas,[(s['x'],s['y']) for s in env.stas])


    def test_train_cannot_select_checkpoint_on_test(self):
        with self.assertRaisesRegex(ValueError, 'Test layouts'):
            train_marl(eval_mode='test', episodes=1)
        with self.assertRaisesRegex(ValueError, 'Test layouts'):
            train_marl(eval_mode='fixed_test', episodes=1)


    def test_evaluation_rejects_wrong_split_before_mutating_agents(self):
        with self.assertRaisesRegex(ValueError, 'requested evaluation split'):
            evaluate_policy({}, eval_mode='validation', scenario_specs=[scenario_bank('test')[0]])
