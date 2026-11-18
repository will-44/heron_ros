| metric | base moving (exp 1) | base fixed (exp 2) |
|---|---|---|
| episodes recorded | 35.0 | 30.0 |
| excluded: infrastructure/operator | 3.000 | 2.000 |
| excluded: grasp/release failure | 7.000 | 5.000 |
| valid | 25.0 | 23.0 |
| success | 23.0 | 18.0 |
| success rate | 0.920 | 0.783 |
| bad-grasp cancel | 7.000 | 0.000 |
| grasp/release failed | 0.000 | 5.000 |
| collision | 2.000 | 5.000 |
| cycle duration, success, median (s) | 177.8 | 214.4 |
| cycle duration, success, mean (s) | 183.9 | 224.2 |
| home enter median (s) | 37.9 | 37.8 |
| pick approach median (s) | 41.1 | 41.8 |
| grasp move median (s) | 5.095 | 5.090 |
| place approach median (s) | 42.3 | 71.4 |
| place move median (s) | 12.4 | 13.0 |
| home exit median (s) | 22.3 | 21.8 |
| pick: legs | 25.0 | 23.0 |
| pick: commanded cycles, median | 21.0 | 19.0 |
| pick: commanded time, median (s) | 20.1 | 17.9 |
| pick: commanded path, median (m) | 0.413 | 0.362 |
| pick: Q goal at leg start, median | 0.791 | 0.789 |
| pick: Q goal at end of command, median | 0.836 | 0.790 |
| pick: ΔQ goal over the command, median | 0.041 | -0.001 |
| pick: ΔQ goal > 0 (fraction of legs) | 0.720 | 0.435 |
| pick: Q region mean at leg start, median | 0.858 | 0.860 |
| pick: Q region mean at end of command, median | 0.852 | 0.810 |
| pick: |∇Q| median (1/m) | 0.199 | 0.224 |
| pick: stop = arrival latch | 18.0 | 19.0 |
| pick: stop = interlock | 5.000 | 4.000 |
| pick: stop = convergence | 2.000 | 0.000 |
| pick: MPC scene-collision active, success legs, median | 0.013 | 0.000 |
| pick: MPC FK error at end, success legs, median (m) | 0.001 | 0.001 |
| place: legs | 23.0 | 23.0 |
| place: commanded cycles, median | 21.0 | 24.0 |
| place: commanded time, median (s) | 20.9 | 29.1 |
| place: commanded path, median (m) | 0.405 | 0.448 |
| place: Q goal at leg start, median | 0.722 | 0.770 |
| place: Q goal at end of command, median | 0.773 | 0.697 |
| place: ΔQ goal over the command, median | 0.016 | -0.068 |
| place: ΔQ goal > 0 (fraction of legs) | 0.652 | 0.174 |
| place: Q region mean at leg start, median | 0.794 | 0.788 |
| place: Q region mean at end of command, median | 0.832 | 0.789 |
| place: |∇Q| median (1/m) | 0.193 | 0.202 |
| place: stop = arrival latch | 16.0 | 17.0 |
| place: stop = interlock | 7.000 | 4.000 |
| place: stop = convergence | 0.000 | 2.000 |
| place: MPC scene-collision active, success legs, median | 0.027 | 0.051 |
| place: MPC FK error at end, success legs, median (m) | 0.002 | 0.001 |
| controller cycle time, mean (ms) | 701.2 | 696.7 |
| MPC solve time, mean (ms) | 207.9 | 203.8 |
| MPC solve time, p95 (ms) | 323.0 | 315.7 |
| MPC over budget (fraction of steps) | 0.011 | 0.011 |
| ΔQ per cycle, std | 0.060 | 0.053 |
| gradient direction turn per cycle, median (deg) | 10.2 | 8.964 |
| gradient direction reversals (> 90°), fraction | 0.059 | 0.035 |
