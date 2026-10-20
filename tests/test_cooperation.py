import unittest
import torch
import numpy as np

import config
from env.mean_field import get_circular_neighbor_indices, compute_mean_field_z
from agent.double_dqn import DoubleDQNAgent
from training.train_loop import train_marl

class TestMeanFieldCooperation(unittest.TestCase):
    def test_circular_neighbors(self):
        # 6 APs: 0, 1, 2, 3, 4, 5
        self.assertEqual(get_circular_neighbor_indices(0, 6), [5, 1])
        self.assertEqual(get_circular_neighbor_indices(5, 6), [4, 0])
        self.assertEqual(get_circular_neighbor_indices(2, 6), [1, 3])

    def test_compute_mean_field_z_tensor(self):
        # batch=2, num_agents=6, latent_size=4
        all_zs = torch.randn(2, 6, 4)
        mf_z = compute_mean_field_z(all_zs, 6)
        self.assertEqual(mf_z.shape, (2, 6, 4))
        # For agent 0, neighbors are 5 and 1
        expected_0 = (all_zs[:, 5, :] + all_zs[:, 1, :]) / 2.0
        torch.testing.assert_close(mf_z[:, 0, :], expected_0)

    def test_cooperative_agent_act(self):
        agent = DoubleDQNAgent(state_size=config.OBS_SIZE, latent_size=16, cooperation_enabled=True)
        agent.epsilon = 0.0
        state = np.zeros(config.OBS_SIZE, dtype=np.float32)
        neighbor_z = np.zeros(16, dtype=np.float32)
        action = agent.act(state, neighbor_z=neighbor_z)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, agent.action_size)

    def test_train_marl_with_cooperation(self):
        result = train_marl(
            experiment_name="test_coop_run",
            use_water_filling=True,
            action_size=config.HYBRID_ACTION_SIZE,
            episodes=2,
            seed=42,
            eval_mode="fixed",
            cooperation_enabled=True,
        )
        self.assertIn("throughput", result)
        self.assertEqual(len(result["throughput"]), 2)

if __name__ == "__main__":
    unittest.main()
