# Extension A (learning-rate tuning) — results

Selection rule (pre-registered in Milestone 1): for each depth, the initial
learning rate with the **lowest evaluation-mode training error at the last
iteration**. Validation error is reported but not used for selection.

## Grid

| Depth | Initial lr | Train 45k (%) | Val 5k (%) | Test 10k (%) | Status |
|---|---|---|---|---|---|
| plain-20 | 0.01 | 1.013 | 11.36 | 12.26 | finished |
| plain-20 | 0.03 | 0.453 | 9.96 | 10.79 | finished |
| **plain-20** | **0.1 (selected)** | **0.429** | 9.14 | 10.15 | finished |
| plain-20 | 0.3 | 0.431 | 8.82 | 8.82 | finished |
| plain-20 | 0.5 | 0.640 | 9.34 | 9.59 | finished |
| plain-56 | 0.01 | 3.911 | 15.92 | 16.94 | finished |
| **plain-56** | **0.03 (selected)** | **1.938** | 15.10 | 15.61 | finished |
| plain-56 | 0.1 | 2.809 | 13.96 | 14.58 | finished |
| plain-56 | 0.3 | --- | --- | --- | **diverged at iteration 23** |
| plain-56 | 0.5 | --- | --- | --- | **diverged at iteration 14** |

Both divergences reproduce exactly when the run is repeated on its own (same
iteration counts), so they are genuine numeric explosions and not an artefact
of running several jobs at once. plain-20 trains normally at both 0.3 and 0.5.
A diverged run is recorded and not replaced, as the protocol requires.

## Hypothesis H1

> The initial learning rate selected by the tuning criterion is lower for
> plain-56 than for plain-20.

Confirmed: plain-20 selects **0.1**, plain-56 selects **0.03**, a factor of 3.3
lower. The evidence is stronger than the selection alone — plain-56 cannot be
trained at all at 0.3 or 0.5, while plain-20 converges at both.

## Two qualifications to state in the report

1. **The plain-20 selection is nominal.** 0.1 (0.429%) and 0.3 (0.431%) differ
   by 0.002 percentage points, which is far below any meaningful resolution,
   and the seed-to-seed spread of plain-20 at lr 0.1 is 0.11 points on the
   training error. The rule selects 0.1, but the two rates should be described
   as tied. On test error, 0.3 is in fact the better of the two (8.82% against
   10.15%), and better than the paper's own 0.1 setting.
2. **Selection on training error and test performance disagree for plain-56.**
   The selected rate (0.03, training error 1.94%) has a *higher* test error
   (15.61%) than the unselected 0.1 (training error 2.81%, test error 14.58%).
   The pre-registered rule is followed as written, but the report must say that
   for the deeper plain network, the rate that fits the training set best is not
   the rate that generalises best.

## Seeds at the selected rates

`plain-20` at lr 0.1 already has three seeds, because that rate is also the
replication's seed 0/1/2 configuration. `plain-56` at lr 0.03 was completed with
seeds 1 and 2 to match.

| Network (selected lr) | Seed | Train 45k (%) | Val 5k (%) | Test 10k (%) |
|---|---|---|---|---|
| plain-20 (0.1) | 0 | 0.43 | 9.14 | 10.15 |
| plain-20 (0.1) | 1 | 0.38 | 9.42 | 9.55 |
| plain-20 (0.1) | 2 | 0.48 | 9.04 | 9.92 |
| plain-20 (0.1) | **mean** | **0.43** | 9.20 | **9.87** |
| plain-20 (0.1) | **spread** | 0.11 | 0.38 | 0.60 |
| plain-56 (0.03) | 0 | 1.94 | 15.10 | 15.61 |
| plain-56 (0.03) | 1 | 1.37 | 13.44 | 13.95 |
| plain-56 (0.03) | 2 | 1.89 | 12.54 | 13.07 |
| plain-56 (0.03) | **mean** | **1.73** | 13.69 | **14.21** |
| plain-56 (0.03) | **spread** | 0.57 | 2.56 | 2.54 |

## Hypothesis H2

> The degradation persists after each network is tuned, and the difference
> exceeds the spread across seeds.

Confirmed at both rates. At the selected rates the deep network is worse by
1.30 points on training error against a larger seed spread of 0.57, by 2.12
points on the running training error against 0.68, and by 4.34 points on test
error against 2.54. Tuning each depth to its own best rate therefore does not
remove the degradation.

Worth recording: the seed spread at the selected rate is much wider for
plain-56 (2.54 points on test) than for plain-20 (0.60), which is itself part of
the picture -- the deeper plain network is not only worse on average but much
less reproducible.

## Operational notes

- Keep concurrency at 3 or below. Eight parallel processes crashed two runs on
  13 November (`CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED` and
  `CUDA error: an illegal memory access`).
- Running two 56-layer jobs at once gives about 5% more aggregate throughput,
  so 56-layer runs are best run one at a time; 20-layer jobs leave the GPU half
  idle and do benefit from parallelism.
- `src/train.py` writes `last.pth` and `results.json` only when the 64k
  schedule ends, so an interrupted run cannot be resumed and has to be
  repeated.
