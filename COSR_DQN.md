# Central masked DQN with expert warm-start

This learner is a central MAPC scheduler for `CoSREnv`, not a replacement for
the legacy decentralised QMIX implementation. At every TXOP it scores every
feasible AP--STA group using shared group features, then selects the highest
scoring group. The action mask removes PHY-incompatible groups and groups with
empty queues.

## Stabilisation design

`Group-Oldest + PF` is the transparent reference policy. Before TD learning,
the shared scorer is trained by masked behaviour cloning on actions from
`Group-Oldest`.  During the first 60% of DQN training, a decaying imitation
loss prevents catastrophic forgetting.  PF remains the deterministic power
allocator, so RL learns scheduling rather than a second, confounded power
control problem.

The default is fixed topology.  This is the first curriculum stage: verify the
scheduler can equal the expert on a reproducible scenario before using
`--random-topology` for generalisation.

## Kaggle command

```python
import sys
!{sys.executable} scripts/run_cosr_dqn.py --episodes 300 --steps 300 --warmup-steps 150 --replay-warmup 2000 --batch-size 128 --eval-interval 25 --eval-episodes 3 --arrival-rate 0.45 --seed 0 --output-dir results_data/cosr_dqn_seed0
```

Run seed 0 first.  Only if its final evaluation is near or better than the
`Group-Oldest + PF` reference should seeds 1 and 2 be launched.  For the
generalisation stage, add `--random-topology`; do not mix fixed and random
results in one aggregate.
