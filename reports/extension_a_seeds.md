| Network | Seed | Train 45k (%) | Val 5k (%) | Test 10k (%) | Minutes | Status |
|---|---|---|---|---|---|---|
| plain-20 | 0 | 0.43 | 9.14 | 10.15 | 15.4 | iter 64000 |
| plain-20 | 1 | 0.38 | 9.42 | 9.55 | 31.3 | iter 64000 |
| plain-20 | 2 | 0.48 | 9.04 | 9.92 | 31.1 | iter 64000 |
| plain-20 | **mean** | 0.43 | 9.20 | 9.87 | | |
| plain-20 | **spread** | 0.11 | 0.38 | 0.60 | | |
| plain-56 | 0 | 1.94 | 15.10 | 15.61 | 126.2 | iter 64000 |
| plain-56 | 1 | 1.37 | 13.44 | 13.95 | 39.9 | iter 64000 |
| plain-56 | 2 | 1.89 | 12.54 | 13.07 | 41.5 | iter 64000 |
| plain-56 | **mean** | 1.73 | 13.69 | 14.21 | | |
| plain-56 | **spread** | 0.57 | 2.56 | 2.54 | | |

- plain-56 minus plain-20: train_err 1.73 - 0.43 = +1.30 pp; larger spread 0.57; real
- plain-56 minus plain-20: running_train_err 2.97 - 0.85 = +2.12 pp; larger spread 0.68; real
