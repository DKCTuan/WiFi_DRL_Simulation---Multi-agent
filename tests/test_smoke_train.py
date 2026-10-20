import unittest
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config
from training.train_loop import train_marl

class SmokeTrainTest(unittest.TestCase):
    def test_train_marl_3_episodes(self):
        result = train_marl(
            experiment_name="smoke_test_run",
            use_water_filling=True,
            action_size=config.HYBRID_ACTION_SIZE,
            episodes=3,
            seed=42,
            eval_mode="fixed",
        )
        self.assertIn("throughput", result)
        self.assertIn("jfi", result)
        self.assertEqual(len(result["throughput"]), 3)
        print("Smoke train test passed 100% successfully!")

if __name__ == "__main__":
    unittest.main()
