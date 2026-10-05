# M6b: expected-F0.5 decision (alternative to the m6 thresholds)

Validation fold 0: 100,000 S1, split by hashed id into half 1 (50,000, fits calibration and c) and half 2 (50,000, reported). One-owner rule applied first on raw p (over all fold-0 S1s, as in m6). Candidates after one-owner: 2,251,023 (removed 744,440).

## Calibration (isotonic, half 1)

230 thresholds. Reliability on half 2 (mean label vs mean p / calibrated q per raw-p bin):

| raw p bin | pairs | mean label | mean raw p | mean q |
|---|---|---|---|---|
| (-0.001, 0.01] | 937,780 | 0.0003 | 0.0001 | 0.0002 |
| (0.01, 0.05] | 13,499 | 0.0360 | 0.0236 | 0.0358 |
| (0.05, 0.2] | 8,627 | 0.1422 | 0.1071 | 0.1431 |
| (0.2, 0.4] | 4,287 | 0.3224 | 0.2901 | 0.3270 |
| (0.4, 0.6] | 2,915 | 0.4988 | 0.4983 | 0.5143 |
| (0.6, 0.8] | 3,206 | 0.6716 | 0.7052 | 0.7003 |
| (0.8, 0.95] | 5,878 | 0.8702 | 0.8949 | 0.8713 |
| (0.95, 0.99] | 10,083 | 0.9726 | 0.9768 | 0.9680 |
| (0.99, 1.0] | 138,564 | 0.9991 | 0.9992 | 0.9989 |

Sum over half 2: labels 160,294, raw p 159,899, q 160,340.

## c = expected number of never-retrieved true matches (tuned on half 1)

| c | plug-in half 1 | Monte Carlo half 1 |
|---|---|---|
| 0.0 | 0.94940 | 0.94959 |
| 0.05 | 0.94944 | 0.94962 |
| 0.1 | 0.94945 | 0.94961 |
| 0.15 | 0.94943 | 0.94962 |
| 0.2 | 0.94952 | 0.94967 |
| 0.25 | 0.94949 | 0.94968 |
| 0.3 | 0.94947 | 0.94970 |
| 0.35 | 0.94951 | 0.94969 |
| 0.4 | 0.94951 | 0.94971 |
| 0.5 | 0.94950 | 0.94972 |
| 0.6 | 0.94953 | 0.94972 |
| 0.75 | 0.94968 | 0.94989 |
| 1.0 | 0.94982 | 0.94979 |
| 1.5 | 0.94981 | 0.94987 |

## Half 2 macro F0.5 (never used for fitting anything except the production thresholds)

| decision | half-2 macro F0.5 |
|---|---|
| current rule (decision.json, as in production) | 0.95081 |
| current rule re-tuned on half 1 (g=0.59, t=0.01, r=0.725) | 0.95034 |
| expected F0.5, plug-in (c=1.0) | 0.94966 |
| expected F0.5, Monte Carlo 200 draws (c=0.75) | 0.94969 |
| (reference) oracle: perfect picking among the candidates | 0.97207 |

plug-in: empty lists 0.0628, mean list size 3.132

Monte Carlo: empty lists 0.0628, mean list size 3.125

current rule: empty lists 0.0661, mean list size 3.119

Paired difference best - current on half 2: -0.00112 (std. error 0.00025); entities better 322, worse 374.

**Best alternative: mc (0.94969) vs current rule (0.95081) -> does NOT beat the current rule.**

