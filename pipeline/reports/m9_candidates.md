# M9: candidate generation v3 - variants on fold-0 validation (100k S1, full train pool)

Recall ceiling = share of true (S1, match) pairs among the candidates; oracle F0.5 = macro F0.5 if a perfect model picked among them. **Reverse**: each of the 10.3M train pool records searches all 2.2M train S1s (same keys, idf with df over S1, cap 1000; same country via the keys) and keeps its top-30 list; a variant adds (S1, record) for the record's top-k, or for ranks 1-30 whose reverse score is >= X x the record's rank-1 score. **Second hop**: seeds = each S1's top-3 candidates by OUT-OF-FOLD p (fold 0: the fold-1 model; fold 1: the cross-fit fold-0 model); pool->pool key search (tight key types 2,6,7,8,10,11, top 20 per seed), kept if token_set_ratio on name_core AND on the address >= T; <= 20 added per S1.

Generation cost on the train split (all 2.2M S1 / 10.3M pool records): reverse pass + regrouping 76 min, peak RAM 2.2 GB (one pass serves every reverse variant); second hop 4 min for the 592,279 seed records of the 200k sample S1s (all 2.2M S1: ~10x the seeds), peak RAM 2.8 GB. Forward blocking alone: ~40 min for 2.2M S1.

RAM fit: LightGBM on the fold-1 training pairs (estimate: 66 float32 features + bins + ids) must stay < 3.5 GB, and a 48k-S1 inference batch must stay < 3.0M pairs (the m7 batches used 2.9M pairs at <= 2.7 GB).

| variant | recall ceiling | oracle F0.5 | mean cands/S1 | max cands/S1 | total pairs (fold 0) | extra runtime | peak RAM | training pairs (fold 1) | est. LightGBM RAM | pairs per 48k batch | fits |
|---|---|---|---|---|---|---|---|---|---|---|---|
| forward K=30 (current) | 0.9261 | 0.9721 | 30.0 | 30 | 2,995,463 | 0 min | 0.0 GB | 2,995,491 | 1.5 GB | 1.44M | yes |
| forward K=30 (current) + reverse top-3 | 0.9591 | 0.9859 | 34.6 | 2,670 | 3,459,505 | 76 min | 2.2 GB | 3,460,188 | 1.7 GB | 1.66M | yes |
| forward K=30 (current) + reverse top-5 | 0.9626 | 0.9871 | 40.6 | 3,400 | 4,059,026 | 76 min | 2.2 GB | 4,062,860 | 1.9 GB | 1.95M | yes |
| forward K=30 (current) + reverse top-10 | 0.9673 | 0.9889 | 59.1 | 4,770 | 5,911,269 | 76 min | 2.2 GB | 5,944,506 | 2.5 GB | 2.84M | yes |
| forward K=30 (current) + reverse ratio >= 0.9 x best (ranks 1-30) | 0.9653 | 0.9880 | 34.0 | 1,109 | 3,400,548 | 76 min | 2.2 GB | 3,401,610 | 1.7 GB | 1.63M | yes |
| forward K=30 (current) + reverse ratio >= 0.8 x best (ranks 1-30) | 0.9675 | 0.9888 | 35.0 | 1,142 | 3,496,707 | 76 min | 2.2 GB | 3,499,157 | 1.7 GB | 1.68M | yes |
| forward K=30 (current) + reverse ratio >= 0.7 x best (ranks 1-30) | 0.9695 | 0.9896 | 37.3 | 1,304 | 3,727,324 | 76 min | 2.2 GB | 3,733,492 | 1.8 GB | 1.79M | yes |
| forward K=30 (current) + second hop (sim >= 90) | 0.9311 | 0.9735 | 30.1 | 41 | 3,005,655 | 4 min | 2.8 GB | 3,005,887 | 1.5 GB | 1.44M | yes |
| forward K=30 (current) + second hop (sim >= 85) | 0.9330 | 0.9741 | 30.1 | 41 | 3,010,863 | 4 min | 2.8 GB | 3,011,186 | 1.5 GB | 1.45M | yes |
| forward K=30 (current) + second hop (sim >= 80) | 0.9348 | 0.9746 | 30.2 | 43 | 3,017,256 | 4 min | 2.8 GB | 3,017,499 | 1.5 GB | 1.45M | yes |
| forward K=30 (current) + reverse top-3 + second hop (sim >= 90) | 0.9615 | 0.9865 | 34.7 | 2,670 | 3,468,071 | 80 min | 2.8 GB | 3,468,866 | 1.7 GB | 1.66M | yes |
| forward K=30 (current) + reverse top-3 + second hop (sim >= 85) | 0.9624 | 0.9868 | 34.7 | 2,670 | 3,472,535 | 80 min | 2.8 GB | 3,473,442 | 1.7 GB | 1.67M | yes |
| forward K=30 (current) + reverse top-3 + second hop (sim >= 80) | 0.9633 | 0.9870 | 34.8 | 2,670 | 3,478,177 | 80 min | 2.8 GB | 3,479,006 | 1.7 GB | 1.67M | yes |
| forward K=30 (current) + reverse top-5 + second hop (sim >= 90) | 0.9648 | 0.9877 | 40.7 | 3,400 | 4,067,144 | 80 min | 2.8 GB | 4,071,090 | 1.9 GB | 1.95M | yes |
| forward K=30 (current) + reverse top-5 + second hop (sim >= 85) | 0.9656 | 0.9879 | 40.7 | 3,400 | 4,071,401 | 80 min | 2.8 GB | 4,075,441 | 1.9 GB | 1.95M | yes |
| forward K=30 (current) + reverse top-5 + second hop (sim >= 80) | 0.9664 | 0.9881 | 40.8 | 3,400 | 4,076,778 | 80 min | 2.8 GB | 4,080,758 | 1.9 GB | 1.96M | yes |
| forward K=30 (current) + reverse top-10 + second hop (sim >= 90) | 0.9690 | 0.9894 | 59.2 | 4,770 | 5,918,633 | 80 min | 2.8 GB | 5,951,920 | 2.6 GB | 2.84M | yes |
| forward K=30 (current) + reverse top-10 + second hop (sim >= 85) | 0.9697 | 0.9896 | 59.2 | 4,770 | 5,922,578 | 80 min | 2.8 GB | 5,955,912 | 2.6 GB | 2.84M | yes |
| forward K=30 (current) + reverse top-10 + second hop (sim >= 80) | 0.9704 | 0.9898 | 59.3 | 4,770 | 5,927,545 | 80 min | 2.8 GB | 5,960,787 | 2.6 GB | 2.85M | yes |
| forward K=30 (current) + reverse ratio >= 0.9 x best (ranks 1-30) + second hop (sim >= 90) | 0.9677 | 0.9886 | 34.1 | 1,109 | 3,409,481 | 80 min | 2.8 GB | 3,410,654 | 1.7 GB | 1.64M | yes |
| forward K=30 (current) + reverse ratio >= 0.9 x best (ranks 1-30) + second hop (sim >= 85) | 0.9687 | 0.9889 | 34.1 | 1,109 | 3,414,137 | 80 min | 2.8 GB | 3,415,379 | 1.7 GB | 1.64M | yes |
| forward K=30 (current) + reverse ratio >= 0.9 x best (ranks 1-30) + second hop (sim >= 80) | 0.9696 | 0.9891 | 34.2 | 1,109 | 3,419,983 | 80 min | 2.8 GB | 3,421,125 | 1.7 GB | 1.64M | yes |
| forward K=30 (current) + reverse ratio >= 0.8 x best (ranks 1-30) + second hop (sim >= 90) | 0.9696 | 0.9894 | 35.1 | 1,142 | 3,505,413 | 80 min | 2.8 GB | 3,507,969 | 1.7 GB | 1.68M | yes |
| forward K=30 (current) + reverse ratio >= 0.8 x best (ranks 1-30) + second hop (sim >= 85) | 0.9705 | 0.9896 | 35.1 | 1,142 | 3,509,968 | 80 min | 2.8 GB | 3,512,606 | 1.7 GB | 1.68M | yes |
| forward K=30 (current) + reverse ratio >= 0.8 x best (ranks 1-30) + second hop (sim >= 80) | 0.9713 | 0.9899 | 35.2 | 1,142 | 3,515,710 | 80 min | 2.8 GB | 3,518,256 | 1.7 GB | 1.69M | yes |
| forward K=30 (current) + reverse ratio >= 0.7 x best (ranks 1-30) + second hop (sim >= 90) | 0.9714 | 0.9902 | 37.4 | 1,304 | 3,735,809 | 80 min | 2.8 GB | 3,742,092 | 1.8 GB | 1.79M | yes |
| forward K=30 (current) + reverse ratio >= 0.7 x best (ranks 1-30) + second hop (sim >= 85) | 0.9722 | 0.9904 | 37.4 | 1,304 | 3,740,230 | 80 min | 2.8 GB | 3,746,636 | 1.8 GB | 1.80M | yes |
| forward K=30 (current) + reverse ratio >= 0.7 x best (ranks 1-30) + second hop (sim >= 80) | 0.9729 | 0.9905 | 37.5 | 1,304 | 3,745,838 | 80 min | 2.8 GB | 3,752,152 | 1.8 GB | 1.80M | yes |

**Chosen: forward K=30 (current) + reverse ratio >= 0.7 x best (ranks 1-30) + second hop (sim >= 80)** - the highest recall ceiling among the variants that fit (recall 0.9729 vs 0.9261 for forward only; oracle 0.9905 vs 0.9721). -> models/v3/cand_cfg.json


## Decision step and fold-0 macro F0.5 (100k validation S1s)

*fold-0 only*: the one-owner rule only among the 100k fold-0 S1s (how 0.9503 was measured). *full competition*: the one-owner rule over ALL 2.2M train S1s, each with an out-of-fold p (fold-1 sample: the model trained on fold 0; everyone else: the model trained on the fold-1 sample), as on the test set. Thresholds (gate g, absolute t, relative r) are grid-searched on the fold-0 rows; 'half 2' columns: thresholds tuned on half 1 (or on all of fold 0) and scored on half 2 only.

| pipeline | competition | F0.5 (tuned on fold 0) | g / t / r | half 2, fold-0-tuned | half 2, half-1-tuned |
|---|---|---|---|---|---|
| old | fold-0 only | **0.95030** | 0.67 / 0.01 / 0.725 | 0.95081 | 0.95034 |
| old | full competition | **0.95190** | 0.59 / 0.01 / 0.7 | 0.95220 | 0.95177 |
| v3 | fold-0 only | **0.96689** | 0.68 / 0.01 / 0.725 | 0.96726 | 0.96707 |
| v3 | full competition | **0.96769** | 0.68 / 0.01 / 0.7 | 0.96811 | 0.96810 |

Expected-F0.5 decision (m6b) on v3 / full competition, half 2: plug-in (c=1.5): 0.96752, Monte Carlo (c=1.5): 0.96774 (rule with fold-0 thresholds on half 2: 0.96811) -> the rule stays.

Consistency checks (p of the full runs vs the stored validation predictions): {'old': {'rows': 2995463, 'ref_rows': 2995463, 'matched': 2995463, 'max_abs_p_diff': 0.0}, 'v3': {'rows': 3745838, 'ref_rows': 3745838, 'matched': 3745838, 'max_abs_p_diff': 0.0}}

**Gate for writing output_v3: v3 full competition 0.96769 vs old full competition 0.95190 -> WRITE.**


## Test output check (C:/amlc/submission/output_v3)

rows 1,732,544; rows whose matches are NOT a subset of candidates: 0; candidates 72,169,455 (mean 41.66, max 1707); matches 5,745,731 (mean 3.316, max 28)


## Test vs fold-0 per country (v3, full competition, same decision)

| set | country | S1 | mean matches | empty | best p 0.00-0.10 | best p 0.10-0.30 | best p 0.30-0.50 | best p 0.50-0.68 | best p 0.68-0.80 | best p 0.80-0.90 | best p 0.90-0.99 | best p 0.99-1.00 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fold 0 | India | 40,270 | 3.251 | 0.0603 | 0.0444 | 0.0079 | 0.0046 | 0.0030 | 0.0024 | 0.0040 | 0.0256 | 0.9080 |
| fold 0 | US | 59,730 | 3.254 | 0.0596 | 0.0403 | 0.0110 | 0.0053 | 0.0026 | 0.0016 | 0.0023 | 0.0168 | 0.9201 |
| test | France | 259,452 | 3.605 | 0.0437 | 0.0196 | 0.0105 | 0.0063 | 0.0046 | 0.0033 | 0.0040 | 0.0156 | 0.9362 |
| test | India | 809,986 | 3.251 | 0.0602 | 0.0415 | 0.0097 | 0.0049 | 0.0039 | 0.0030 | 0.0044 | 0.0272 | 0.9055 |
| test | US | 663,106 | 3.284 | 0.0589 | 0.0376 | 0.0124 | 0.0056 | 0.0032 | 0.0018 | 0.0026 | 0.0151 | 0.9217 |
