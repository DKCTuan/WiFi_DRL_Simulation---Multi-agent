# Literature-aligned Co-SR baseline

The new Co-SR path is intentionally separate from `env/wifi_env.py`. Existing
Hybrid-WF, Full-AI, QMIX, VDN, and historical Co-BF-like experiments are not
modified, so old results remain reproducible.

## Model boundary

- One environment step is one 5 ms MAPC TXOP.
- Fixed MAC and per-extra-AP coordination airtime are deducted before payload
  capacity is calculated.
- One AP transmits to at most one associated STA in a TXOP.
- Valid groups contain one or two AP-STA links.
- A concurrent group is admitted only when `|G| * R_CoSR >= R_single` for every
  receiver.
- PHY rates use an explicit discrete MCS table rather than Shannon capacity.
- The online mask rejects groups containing an empty queue.
- Traffic, finite queues, HoL delay, packet drops, JFI, throughput, and energy
  efficiency are reported.
- Warm-up is excluded from reported metrics, and queue growth per TXOP is
  recorded to identify overload.
- `pf_pair` searches the two Pareto-boundary power edges where at least one AP
  transmits at maximum power.

The MCS thresholds in `config.py` are an explicit simulation abstraction, not
a claim of bit-exact IEEE conformance. Replace them with the exact PHY table
used in the final paper before producing publication results.

## Fast validation

```bash
python -m unittest discover -s tests -p "test_*.py"
python scripts/run_cosr_baselines.py --seeds 0 --arrival-rates 0.45 --episodes 1 --warmup-steps 10 --steps 20
```

## Three-seed load sweep

```bash
python scripts/run_cosr_baselines.py --seeds 0 1 2 --episodes 3 --warmup-steps 150 --steps 300 --num-stas 10 --arrival-rates 0.30 0.35 0.40 0.45
```

Results are written to `results_data/cosr_baselines/`. This run compares
single-link oldest-packet scheduling, two Co-SR group schedulers, pairwise PF
power allocation, and a factorized two-level UCB scheduler.

Interpret each load point using `queue_growth_packets_per_txop`: near zero
means the system is stable; a persistent positive value means a saturation
scenario, not a steady-state latency result. Repeat the selected load with
`--topology-mode random` after the controlled fixed-topology comparison.

Do not compare the resulting Mbps directly with the legacy saturated Shannon
model: the Co-SR environment is traffic-limited and packet-based. Compare
policies within this environment using throughput, p95 delay, drop rate, JFI,
and energy efficiency together.
