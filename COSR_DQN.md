# Central masked DQN with expert warm-start

This learner is a central MAPC scheduler for `CoSREnv`, not a replacement for
the legacy decentralised QMIX implementation. At every TXOP it scores every
feasible AP--STA group using shared group features, then selects the highest
scoring group. The action mask removes PHY-incompatible groups and groups with
empty queues.

## Stabilisation design

`Group-Oldest + PF` is the transparent reference policy. Before TD learning,
the shared scorer is trained by masked behaviour cloning on actions from
`Group-Oldest`.  During the first 75% of DQN training, a decaying imitation
loss and optional floor prevent catastrophic forgetting.  PF remains the deterministic power
allocator, so RL learns scheduling rather than a second, confounded power
control problem.

The default is fixed topology.  This is the first curriculum stage: verify the
scheduler can equal the expert on a reproducible scenario before using
`--random-topology` for generalisation.  The run seed controls DQN/replay/
exploration randomness; `--scenario-seed` independently controls the fixed
topology, while reset seeds vary traffic arrivals.

## Kaggle command

```python
import sys
!{sys.executable} scripts/run_cosr_dqn.py --episodes 300 --steps 300 --warmup-steps 150 --replay-warmup 2000 --batch-size 128 --eval-interval 25 --eval-episodes 20 --arrival-rate 0.45 --fixed-topology --scenario-seed 2026 --expert-steps 6000 --expert-pretrain-updates 500 --imitation-weight 0.30 --imitation-floor 0.05 --lr-decay-after-episode 150 --lr-decay-gamma 0.50 --seed 0 --output-dir results_data/cosr_seed0_fixed2026_floor005_lrdecay150
```

The imitation term decays linearly for the first 75% of training and then
remains at `--imitation-floor`; use `--imitation-floor 0` to reproduce the
earlier no-floor setting.  The optional LR schedule above changes the learning
rate from `3e-4` to `1.5e-4` after episode 150.  JSON output records reward
components, imitation scale, learning rate, scenario seed, and evaluation
metrics.

Run seed 0 first.  Only after a stable seed-0 configuration is identified and
compared with `Group-Oldest + PF` on matching traffic traces should seeds 1
and 2 be launched.  For the generalisation stage, add `--random-topology`; do
not mix fixed and random results in one aggregate.
