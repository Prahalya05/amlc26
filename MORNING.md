# Report - v3 candidate generation (run 2026-09-25 22:40 -> 2026-09-26 20:45)

## Summary
- **Priority 1 is finished; the new output beats the old one and validates.** v3 adds reverse blocking plus a second
  hop, and the model was retrained with two new reverse features. Fold-0 macro F0.5 under full competition (the
  one-owner rule over all 2.2M train S1s, as on test) is **0.9677 vs 0.9519** for the old pipeline (+0.0158).
- **File to upload:** `C:\amlc\submission\output_v3\matching_results.tsv` (96.5 MB).
  **Official validator: PASS**, also with `--check-ids`. Candidates: `C:\amlc\submission\output_v3\candidate_pairs.tsv`
  (952.5 MB).
- Nothing was submitted. `output\`, `output_v1\`, the old pipeline files, data and models are untouched.
- **Earlier upload failure:** the portal previously marked the upload of `output\matching_results.tsv` (92.9 MB)
  as "Failed" with an unknown cause. The new file is a similar size, so check the portal message if it fails again.

## What finished / failed / was skipped
| step | status | runtime | peak RAM |
|---|---|---|---|
| reverse pass, train (top-30 S1s for each of 10.3M records) | done - killed once overnight by Claude Code (low SYSTEM memory), resumed 08:46 | 74 min compute | 2.3 GB |
| regroup reverse lists (train / test) | done | 2 min / 2 min | 1.8 GB |
| cross-fit model (old features, trained on fold 0) | done, fold-1 out-of-fold AUC 0.99973 | 2.4 h (LightGBM Dataset.subset() was ~60x slower; fixed for later trainings) | 1.5 GB |
| second hop, fold samples / all train / test | done (1st sample run hit the RAM guard, 150k seeds per chunk fixed it) | 4 / 45 / 44 min | 2.9-3.1 GB |
| variant evaluation (28 variants, fold 0) | done | 3.5 min | 1.6 GB |
| build v3 features + train models A (fold 1) / B (fold 0) | done | 5.5 + 8.3 min | 3.85 GB / 2.3 GB |
| old pipeline over all 2.2M train S1s (full competition) | done | 1.9 h | 2.7 GB |
| v3 pipeline over all 2.2M train S1s (full competition) | done | 1.8 h | 3.2 GB |
| decision (old vs v3, both competition modes, rule vs expected F0.5) | done | 8 min | 1.4 GB |
| reverse pass test / second hop test / test inference | done (one inference call ended silently; rerun fine) | 42 min / 44 min / 2.2 h | 2.3 / 3.1 / 3.7 GB |
| write output_v3 + own checks + official validator | **PASS** | 3.5 min + 2 min | 1.4 GB |
| P2 per-country test vs fold 0 | done | 1 min | - |
| P3 round-2 key ideas | **skipped** - conditional on finishing before 06:00; ask to run it now | - | - |

## Fold-0 results (100k validation S1s)
**Recall ceiling** = share of true pairs among the candidates. **Oracle** = F0.5 with perfect picking among them.

| candidates | recall ceiling | oracle F0.5 | cands/S1 | F0.5 fold-0-only | F0.5 full competition |
|---|---|---|---|---|---|
| old: forward K=30 | 0.9261 | 0.9721 | 30.0 | 0.9503 | **0.9519** |
| + reverse top-3 / top-5 / top-10 | 0.9591 / 0.9626 / 0.9673 | 0.9859 / 0.9871 / 0.9889 | 34.6 / 40.6 / 59.1 | - | - |
| + reverse ratio >= 0.9 / 0.8 / 0.7 x best | 0.9653 / 0.9675 / 0.9695 | 0.9880 / 0.9888 / 0.9896 | 34.0 / 35.0 / 37.3 | - | - |
| + second hop sim >= 90 / 85 / 80 | 0.9311 / 0.9330 / 0.9348 | 0.9735 / 0.9741 / 0.9746 | ~30.1 | - | - |
| **v3 = reverse ratio >= 0.7 + second hop >= 80 (chosen)** | **0.9729** | **0.9905** | 37.5 | **0.9669** | **0.9677** |

- **Only one v3 variant got an F0.5.** Each variant's F0.5 needs its own full train pass (~2 h), so the variant was
  chosen by recall ceiling (all 28 fit the RAM rules; full table in `pipeline\reports\m9_candidates.md`). F0.5 was then
  measured for that variant against the old pipeline.
- **Fair half-split (thresholds tuned on half 1, scored on half 2, full competition):** old 0.9518, v3 0.9681.
- **Decision step:** the current rule (gate 0.68, relative 0.7) beats expected F0.5 on half 2
  (plug-in 0.9675, Monte Carlo 0.9677, rule 0.9681), so the rule is used.
- **New features:** `rev_rank` takes 58% of the model's gain and `rev_gap` 4.7%.

## Test output stats (output_v3)
- **Size:** 72.2M candidates (mean 41.7 per S1, max 1,707) and 5,745,731 matches (mean 3.32; old 3.16).
  Empty rows: 5.7% (old 6.5%).

| set | country | S1 | mean matches | empty | best p < 0.1 | best p >= 0.99 |
|---|---|---|---|---|---|---|
| fold 0 | India / US | 40k / 60k | 3.25 / 3.25 | 6.0% / 6.0% | 4.4% / 4.0% | 90.8% / 92.0% |
| test | India / US | 810k / 663k | 3.25 / 3.28 | 6.0% / 5.9% | 4.2% / 3.8% | 90.6% / 92.2% |
| test | France (unseen) | 259k | 3.61 | 4.4% | 2.0% | 93.6% |

- India and US on test behave like validation.
- France gets more matches and fewer empty lists. With no French labels, it's unknown whether that is real or
  overconfidence.

## Runtimes (for planning the next re-blocking)
- **Reverse pass:** ~2-7 min per 500k pool records, depending on the laptop's speed (it changed up to 4x during the day).
  About 40-75 min per split.
- **Full pipeline over all train S1s:** old ~1.9 h, v3 ~1.8 h.
- **Test inference (v3):** ~1.5-2.2 h.
- **The laptop is the bottleneck.** Speed swung between fast and slow phases (CPU at 100%, 1.3 GHz shown).
  Keeping it plugged in, in a performance power mode, with other programs closed, should help.

Everything, including how to rerun each stage (`python m9_cands.py STAGE`, resumable with `--budget SECONDS`),
is in `C:\amlc\pipeline\PROGRESS.md` and `C:\amlc\pipeline\reports\m9_candidates.md`.
