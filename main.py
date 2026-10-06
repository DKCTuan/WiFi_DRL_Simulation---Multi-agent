"""Core entry point. Default is a short runtime check, not long training."""
import argparse
import json
import numpy as np
import config
from agent.double_dqn import DoubleDQNAgent
from env.wifi_env import WiFiEnv
from training.train_loop import train_marl
from training.utils import set_global_seed


def smoke(steps, seed):
    rows = []
    for wf in (False, True):
        set_global_seed(seed)
        size = config.HYBRID_ACTION_SIZE if wf else config.FULL_AI_ACTION_SIZE
        env = WiFiEnv(verbose=False, fixed_topology=True, fixed_seed=2026,
                      mobility_enabled=False, use_water_filling=wf,
                      action_size=size, num_stas_per_ap=10)
        obs, _ = env.reset(seed=seed)
        agents = {aid: DoubleDQNAgent(env.observation_size, size, config.LATENT_SIZE)
                  for aid in env.agent_ids}
        for agent in agents.values():
            agent.epsilon = 0.0
        for _ in range(steps):
            obs, _, _, _, info = env.step({aid: agents[aid].act(obs[aid])
                                         for aid in env.agent_ids})
            assert np.isfinite(info['throughput'])
            assert info['total_tx_power'] <= config.TOTAL_POWER_BUDGET_W + 1e-6
        rows.append(dict(arm='ai_wf' if wf else 'ai_only', slots=steps,
                         observation_size=env.observation_size, action_size=size))
        env.close()
    print(json.dumps(dict(status='passed', untrained_runtime_check=True, arms=rows)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('smoke', 'train'), default='smoke')
    parser.add_argument('--arm', choices=('ai_only', 'ai_wf'), default='ai_only')
    parser.add_argument('--steps', type=int, default=12, help='Slots for smoke mode')
    parser.add_argument('--episodes', type=int, default=25)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--experiment-name', default=None)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.steps <= 0 or args.episodes <= 0:
        parser.error('steps and episodes must be positive')
    if args.mode == 'smoke':
        if args.resume:
            parser.error('--resume requires --mode train')
        smoke(args.steps, args.seed)
    else:
        wf = args.arm == 'ai_wf'
        train_marl(experiment_name=args.experiment_name or f'{args.arm}_seed{args.seed}',
                   use_water_filling=wf,
                   action_size=config.HYBRID_ACTION_SIZE if wf else config.FULL_AI_ACTION_SIZE,
                   episodes=args.episodes, seed=args.seed, resume=args.resume)


if __name__ == '__main__':
    main()
