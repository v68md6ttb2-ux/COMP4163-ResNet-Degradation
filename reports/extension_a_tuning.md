| Depth | Initial lr | Train 45k (%) | Val 5k (%) | Test 10k (%) | Minutes | Status |
|---|---|---|---|---|---|---|
| 20 | 0.01 | 1.01 | 11.36 | 12.26 | 57.3 | iter 64000 |
| 20 | 0.03 | 0.45 | 9.96 | 10.79 | 62.8 | iter 64000 |
| 20 | 0.1 **selected** | 0.43 | 9.14 | 10.15 | 15.4 | iter 64000 |
| 20 | 0.3 | 0.43 | 8.82 | 8.82 | 63.9 | iter 64000 |
| 20 | 0.5 | 0.64 | 9.34 | 9.59 | 15.4 | iter 64000 |
| 56 | 0.01 | 3.91 | 15.92 | 16.94 | 126.9 | iter 64000 |
| 56 | 0.03 **selected** | 1.94 | 15.10 | 15.61 | 126.2 | iter 64000 |
| 56 | 0.1 | 2.81 | 13.96 | 14.58 | 47.6 | iter 64000 |
| 56 | 0.3 | 89.98 | 90.14 | 90.00 | 0.4 | diverged at 23 |
| 56 | 0.5 | 89.98 | 90.14 | 90.00 | 0.4 | diverged at 14 |

plain-20: selected initial rate 0.1 (training error 0.43% at iteration 64000).

plain-56: selected initial rate 0.03 (training error 1.94% at iteration 64000).
