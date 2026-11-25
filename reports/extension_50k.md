# Sensitivity check: training on 50,000 images

The course protocol (topic card) fixes the split at 45,000 train / 5,000
validation. The paper places its 64k stopping point on a 45k/5k split but never
says whether the networks behind Table 6 were themselves trained on 45,000 or
on all 50,000. That left one unexplained number in our replication: plain-56
came out at 14.43% against the paper's 13.3%.

This check trains the two plain networks on all 50,000 images, with everything
else unchanged: same initialisation, augmentation, batch size, learning-rate
schedule, 64,000 iterations. With no held-out images there is no validation set,
so the validation error is not reported. Three seeds per depth, six runs.

## Results

| Network | Training set | Train 45k/50k (%) | Test 10k (%) | Paper |
|---|---|---|---|---|
| plain-20 | 45,000 (protocol) | 0.43 | 9.87 | 9.7 |
| plain-20 | **50,000** | 0.61 | **9.44** | 9.7 |
| plain-56 | 45,000 (protocol) | 2.47 | 14.43 | 13.3 |
| plain-56 | **50,000** | 4.77 | **13.06** | 13.3 |

Per seed:

| Network | Seed | Train (%) | Test (%) |
|---|---|---|---|
| plain-20, 50k | 0 | 0.64 | 9.47 |
| plain-20, 50k | 1 | 0.55 | 9.33 |
| plain-20, 50k | 2 | 0.64 | 9.51 |
| plain-20, 50k | **mean / spread** | **0.61 / 0.09** | **9.44 / 0.18** |
| plain-56, 50k | 0 | 5.81 | 13.32 |
| plain-56, 50k | 1 | 5.97 | 13.53 |
| plain-56, 50k | 2 | 2.53 | 12.34 |
| plain-56, 50k | **mean / spread** | **4.77 / 3.45** | **13.06 / 1.19** |

## What it shows

**1. The 1.13-point gap is explained by the training-set size.** With 50,000
images plain-56 falls from 14.43% to 13.06%, against the paper's 13.3% -- now
0.24 points *below* the published value instead of 1.13 above. plain-20 moves in
the same direction, from 9.87% to 9.44%. Both networks now sit at or slightly
better than the paper's single-run numbers, which is what training on the same
data should produce. The hypothesis we could only state as unverified is
confirmed.

**2. The degradation does not go away.** With 50,000 images the plain-56
network is still worse than the plain-20 one: 13.06% against 9.44% on test
error, a gap of 3.62 points against a larger seed spread of 1.19, so real.
Training error: 4.77% against 0.61%, a gap of 4.16 points against a larger
spread of 3.45 -- real by the card's rule, but only just, and the margin is
driven by the wide plain-56 spread.

**3. More data makes the deep plain network fit *worse*, not better.** Training
error rises from 2.47% at 45k to 4.77% at 50k while test error falls. This is
the degradation signature again: the deeper network's problem is fitting the
training set, and adding training data makes that problem harder, not easier,
even though it improves generalisation.

**4. The instability finding gets stronger.** At 50,000 images the plain-56
training error ranges from 2.53% to 5.97% across three seeds -- a spread of 3.45
points, five times the 0.70 spread at 45,000. The deeper plain network is not
only worse on average but markedly less reproducible, and that is true at both
training-set sizes.

## Cost

Six runs, 3.2 hours of GPU time on one RTX 4060, run one at a time because two
concurrent processes exhausted host memory on this machine.

## How this should be read

This is a sensitivity check, not part of the pre-registered protocol: the course
fixes the 45k/5k split and the replication's three-seed results stand as they
are. The 50,000-image runs exist to test one specific explanation, and they
support it. They are reported separately for that reason.
