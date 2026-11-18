| metric | base fixed | base moving (gradient) |
|---|---|---|
| episodes recorded | 18.0 | 30.0 |
| excluded: infrastructure/operator | 1.000 | 10.0 |
| excluded: grasp/release failure | 0.000 | 0.000 |
| valid | 17.0 | 20.0 |
| success | 10.0 | 18.0 |
| success rate | 0.588 | 0.900 |
| collision / blocked | 7.000 | 2.000 |
| cycle duration, success, median (s) | 249.7 | 189.4 |
| cycle duration, success, mean (s) | 261.0 | 195.4 |
| home enter median (s) | 30.9 | 37.8 |
| pick approach median (s) | 35.8 | 39.3 |
| grasp move median (s) | 5.088 | 5.083 |
| place approach median (s) | 105.1 | 49.9 |
| place move median (s) | 9.898 | 12.2 |
| home exit median (s) | 39.2 | 23.6 |
| pick: legs | 17.0 | 20.0 |
| pick: commanded cycles, median | 12.0 | 16.0 |
| pick: commanded time, median (s) | 11.1 | 15.0 |
| pick: commanded path, median (m) | 0.231 | 0.312 |
| pick: net displacement, median (m) | 0.221 | 0.169 |
| pick: goal range at leg start, median (m) | 0.606 | 0.600 |
| pick: goal range when the arm arrives, median (m) | 0.603 | 0.762 |
| pick: Q goal at leg start, median | 0.825 | 0.823 |
| pick: Q goal at end of command, median | 0.813 | 0.864 |
| pick: ΔQ goal over the command, median | -0.012 | 0.041 |
| pick: ΔQ goal > 0 (fraction of legs) | 0.176 | 0.600 |
| pick: Q region mean at leg start, median | 0.872 | 0.870 |
| pick: Q region mean at end of command, median | 0.886 | 0.930 |
| pick: |∇Q| median (1/m) | 0.317 | 0.235 |
| pick: stop = arrival latch | 9.000 | 16.0 |
| pick: stop = interlock | 7.000 | 3.000 |
| pick: stop = convergence | 1.000 | 1.000 |
| pick: MPC scene-collision active, success legs, median | 0.000 | 0.000 |
| pick: MPC FK error at end, success legs, median (m) | 0.002 | 0.003 |
| place: legs | 16.0 | 19.0 |
| place: commanded cycles, median | 34.5 | 20.0 |
| place: commanded time, median (s) | 78.0 | 18.9 |
| place: commanded path, median (m) | 0.679 | 0.384 |
| place: net displacement, median (m) | 0.593 | 0.256 |
| place: goal range at leg start, median (m) | 1.063 | 1.094 |
| place: goal range when the arm arrives, median (m) | 1.057 | 0.924 |
| place: Q goal at leg start, median | 0.591 | 0.542 |
| place: Q goal at end of command, median | 0.510 | 0.632 |
| place: ΔQ goal over the command, median | -0.091 | 0.124 |
| place: ΔQ goal > 0 (fraction of legs) | 0.125 | 0.789 |
| place: Q region mean at leg start, median | 0.848 | 0.835 |
| place: Q region mean at end of command, median | 0.671 | 0.804 |
| place: |∇Q| median (1/m) | 0.669 | 0.276 |
| place: stop = arrival latch | 16.0 | 14.0 |
| place: stop = interlock | 0.000 | 5.000 |
| place: stop = convergence | 0.000 | 0.000 |
| place: MPC scene-collision active, success legs, median | 0.035 | 0.044 |
| place: MPC FK error at end, success legs, median (m) | 0.003 | 0.001 |
| executed vs commanded displacement: slope | 0.001 | 1.013 |
| executed vs commanded displacement: r | 0.017 | 0.980 |
| controller cycle time, mean (ms) | 697.3 | 690.8 |
| MPC solve time, mean (ms) | 207.6 | 211.3 |
| MPC solve time, p95 (ms) | 321.3 | 325.8 |
| MPC over budget (fraction of steps) | 0.009 | 0.012 |
| ΔQ per cycle, std | 0.041 | 0.048 |
| gradient direction turn per cycle, median (deg) | 4.245 | 10.6 |
| gradient direction reversals (> 90°), fraction | 0.000 | 0.034 |
