"""Integration checks for both baseline learners and exact episode resume."""
from contextlib import ExitStack, redirect_stdout
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import config
from env.wifi_env import WiFiEnv
from training.train_loop import train_marl


class CoreTrainingTests(unittest.TestCase):
    def test_cooperative_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'excluded'):
            WiFiEnv(verbose=False, cooperation_enabled=True)

    def test_both_arms_update_and_resume_exactly(self):
        original_cwd = Path.cwd()
        original_threads = torch.get_num_threads()
        settings = dict(TRAIN_STEPS_PER_EPISODE=4, LEARNING_STARTS=2,
                        BATCH_SIZE=2, REPLAY_BUFFER_CAPACITY=32,
                        EVAL_INTERVAL=1, EVAL_EPISODES=1,
                        MULTI_LAYOUT_TRAINING=False, TRAIN_MOBILITY_ENABLED=False)
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
                for key, value in settings.items():
                    stack.enter_context(patch.object(config, key, value))
                os.chdir(folder)
                for wf in (False, True):
                    arm = 'ai_wf' if wf else 'ai_only'
                    kwargs = dict(use_water_filling=wf, seed=3, num_stas=2,
                                  action_size=config.HYBRID_ACTION_SIZE if wf else config.FULL_AI_ACTION_SIZE,
                                  eval_mode='fixed_validation', checkpoint_interval=1)
                    with redirect_stdout(io.StringIO()):
                        train_marl(experiment_name=arm, episodes=1, **kwargs)
                        resumed = train_marl(experiment_name=arm, episodes=3, resume=True, **kwargs)
                        complete = train_marl(experiment_name=arm + '_fresh', episodes=3, **kwargs)
                    self.assertGreater(resumed['optimizer_updates_total'], 0)
                    self.assertEqual(resumed['optimizer_updates_total'], complete['optimizer_updates_total'])
                    self.assertEqual(resumed['throughput'], complete['throughput'])
                    a = torch.load(resumed['last_checkpoint_path'], weights_only=True)
                    b = torch.load(complete['last_checkpoint_path'], weights_only=True)
                    self.assertEqual(a['episode_completed'], 3)
                    self.assertEqual(len(a['replay_buffer']['states']), 12)
                    self.assertTrue(torch.equal(a['replay_buffer']['states'], b['replay_buffer']['states']))
                    for aid in a['agents']:
                        for component in ('encoder', 'target_encoder', 'q_network', 'target_network'):
                            for name, value in a['agents'][aid][component].items():
                                self.assertTrue(torch.equal(value, b['agents'][aid][component][name]))
                    for name, value in a['q_mixer'].items():
                        self.assertTrue(torch.equal(value, b['q_mixer'][name]))
                os.chdir(original_cwd)
        finally:
            os.chdir(original_cwd)
            torch.set_num_threads(original_threads)
