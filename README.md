# WiFi DRL Simulation — Multi-agent Co-SR

This repository studies a reproducible, packet-level simulation of
centralized **Cooperative Spatial Reuse (Co-SR)** scheduling for a multi-AP
Wi-Fi network.  The research question is whether a learned central scheduler
can choose compatible concurrent AP--STA transmissions with lower tail delay
than a transparent `Group-Oldest + PF` heuristic, while retaining throughput
and fairness.

## What the current model does

At each 5 ms TXOP, the scheduler selects either one AP--STA link or a
PHY-compatible pair from two APs.  Pairwise proportional-fair (PF) power
allocation is fixed, so the DQN learns the scheduling decision rather than a
confounded power-control policy.  The simulator includes finite packet queues,
Poisson traffic, HoL delay, drops, discrete MCS rates, action masks, warm-up
exclusion, Jain fairness, throughput, group size, and energy efficiency.

This is a Co-SR simulation abstraction, **not** a full Wi-Fi 8 standards
implementation or true Co-BF: it does not yet model CSI, antenna arrays,
beamforming/precoding, or zero forcing.

## Current experimental status

The fixed-topology evaluation protocol separates three sources of randomness:
the DQN run seed, the fixed topology scenario seed, and traffic seeds.  This
prevents the run seed from silently changing the scenario during comparison.

With `run_seed=0`, `scenario_seed=2026`, and fixed topology, the best
structured DQN checkpoint occurred at episode 100:

| Metric | Best DQN checkpoint |
| --- | ---: |
| P95 delay | 33.00 TXOP |
| Mean delay | 16.33 TXOP |
| Jain fairness index | 0.9904 |
| Throughput | 64.88 Mbps |
| Mean group size | 1.82 |

This is close to the earlier `Group-Oldest + PF` reference (about 31.33 TXOP
P95), but it is not yet evidence that DQN beats the baseline.  The policy
drifts after roughly episode 175 and increasingly schedules single links.  An
imitation-loss floor of 0.05 reduced, but did not eliminate, this degradation.

## Reproduce the current learning-rate ablation

```bash
python scripts/run_cosr_dqn.py \
  --episodes 300 --steps 300 --warmup-steps 150 \
  --replay-warmup 2000 --batch-size 128 \
  --eval-interval 25 --eval-episodes 20 \
  --arrival-rate 0.45 --fixed-topology \
  --scenario-seed 2026 --seed 0 \
  --expert-steps 6000 --expert-pretrain-updates 500 \
  --imitation-weight 0.30 --imitation-floor 0.05 \
  --lr-decay-after-episode 150 --lr-decay-gamma 0.50 \
  --output-dir results_data/cosr_seed0_fixed2026_floor005_lrdecay150
```

The run writes the best/last model state dictionaries, full optimizer
checkpoints, and a JSON training/evaluation history to the output directory.

## Next experiments

1. Run the learning-rate-decay ablation above to test whether late TD updates
   cause the learned Co-SR policy to drift.
2. Run a replay-buffer ablation (`50,000` versus `100,000` transitions) with
   the same seed and without LR decay.  The 50,000-transition buffer becomes
   full near episode 167, close to the observed degradation point.
3. Evaluate the selected configuration against `Group-Oldest + PF` using the
   same fixed scenario and traffic traces.
4. Only after a stable seed-0 configuration is identified, run seeds 0, 1,
   and 2 and report mean and variability.

See [COSR_BASELINE.md](COSR_BASELINE.md) for environmental assumptions and
baseline commands, and [COSR_DQN.md](COSR_DQN.md) for learner details.
