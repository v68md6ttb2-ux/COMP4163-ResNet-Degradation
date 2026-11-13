| Network | Seed | Train 45k (%) | Val 5k (%) | Test 10k (%) | Minutes | Status |
|---|---|---|---|---|---|---|
| plain-20 | 0 | 0.43 | 9.14 | 10.15 | 15.4 | iter 64000 |
| plain-20 | 1 | 0.38 | 9.42 | 9.55 | 31.3 | iter 64000 |
| plain-20 | 2 | 0.48 | 9.04 | 9.92 | 31.1 | iter 64000 |
| plain-20 | **mean** | 0.43 | 9.20 | 9.87 | | |
| plain-20 | **spread** | 0.11 | 0.38 | 0.60 | | |
| plain-56 | 0 | 2.81 | 13.96 | 14.58 | 47.6 | iter 64000 |
| plain-56 | 1 | 2.11 | 14.22 | 14.37 | 74.3 | iter 64000 |
| plain-56 | 2 | 2.48 | 13.54 | 14.35 | 73.1 | iter 64000 |
| plain-56 | **mean** | 2.47 | 13.91 | 14.43 | | |
| plain-56 | **spread** | 0.70 | 0.68 | 0.23 | | |
| ResNet-20 | 0 | 0.22 | 7.96 | 8.71 | 17.6 | iter 64000 |
| ResNet-20 | 1 | 0.19 | 8.34 | 8.83 | 34.5 | iter 64000 |
| ResNet-20 | 2 | 0.24 | 8.26 | 8.49 | 34.5 | iter 64000 |
| ResNet-20 | **mean** | 0.21 | 8.19 | 8.68 | | |
| ResNet-20 | **spread** | 0.05 | 0.38 | 0.34 | | |
| ResNet-56 | 0 | 0.02 | 6.84 | 7.49 | 81.7 | iter 64000 |
| ResNet-56 | 1 | 0.01 | 6.84 | 7.41 | 79.6 | iter 64000 |
| ResNet-56 | 2 | 0.01 | 7.04 | 7.19 | 49.2 | iter 64000 |
| ResNet-56 | **mean** | 0.01 | 6.91 | 7.36 | | |
| ResNet-56 | **spread** | 0.01 | 0.20 | 0.30 | | |

- plain-56 minus plain-20: train_err 2.47 - 0.43 = +2.04 pp; larger spread 0.70; real
- plain-56 minus plain-20: running_train_err 3.94 - 0.85 = +3.10 pp; larger spread 0.71; real
- ResNet-20 minus ResNet-56: test_err 8.68 - 7.36 = +1.31 pp; larger spread 0.34; real (paper: 8.75 - 6.97 = 1.78, one run each)
