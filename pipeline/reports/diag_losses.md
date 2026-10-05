# Diagnosis: where the validation F0.5 goes (fold 0, 100k S1, K=30, cap 1000)

Diagnosis only: no pipeline code, data or output was changed. Script: `diag_losses.py`.

## 1. Blocking misses (true pairs that are not among the 30 candidates)

Validation true pairs 346,197; missed 25,578 (7.4%); S1 entities with >= 1 miss: 19,117 of 100,000.

Check: re-running the blocking score for the true pairs reproduces the pipeline's rank for 100.0% of 53,375 found pairs.

Causes: (a) the pair shares no blocking key at all; (b) it shares keys, but every shared key is carried by > 1000 pool records, so the cap removes them; (c) it shares an allowed key (so it is scored) but >= 30 pool records score higher.

| cause | missed pairs | share |
|---|---|---|
| a_no_shared_key | 30 | 0.0012 |
| b_only_capped_keys | 5,450 | 0.2131 |
| c_ranked_below_K | 20,098 | 0.7858 |

By country:

| country | a_no_shared_key | b_only_capped_keys | c_ranked_below_K | total missed | true pairs | miss rate |
|---|---|---|---|---|---|---|
| India | 1.0000 | 2,383 | 10,130 | 12,514 | 139,581 | 0.0897 |
| US | 29 | 3,067 | 9,968 | 13,064 | 206,616 | 0.0632 |

By source:

| source | a_no_shared_key | b_only_capped_keys | c_ranked_below_K | total missed | true pairs | miss rate |
|---|---|---|---|---|---|---|
| S2 | 15 | 2,825 | 9,488 | 12,328 | 167,434 | 0.0736 |
| S3 | 15 | 2,625 | 10,610 | 13,250 | 178,763 | 0.0741 |

By cross-script (name):

| cross-script (name) | a_no_shared_key | b_only_capped_keys | c_ranked_below_K | total missed | true pairs | miss rate |
|---|---|---|---|---|---|---|
| False | 30 | 4,042 | 15,442 | 19,514 | 321,166 | 0.0608 |
| True | 0.0000 | 1,408 | 4,656 | 6,064 | 25,031 | 0.2423 |

**(a) no shared key** - string-level overlap anyway (the key rules did not produce a common key): share a name_core word 46.7%, an address number 0.0%, an address word (>= 3 letters) 0.0%; none of the three 53.3%; cross-script 0.0%.

**(b) only capped keys** - smallest df among the shared keys: 1001-2000: 1,901, 2001-5000: 1,402, 5001-20000: 1,332, >20000: 815; median number of shared (capped) keys 4.

**(c) ranked below K** - forward rank of the true record (1 = best): 31-50: 2,683 (13.3%), 51-100: 5,297 (26.4%), 101-1000: 10,484 (52.2%), >1000: 1,634 (8.1%); recomputed rank <= 30 (tie/rounding disagreements): 0; median records retrieved for these S1s 1,175.


**Reverse search** (query = the missed S2/S3 record, index = all 2,206,821 train S1, same key types, weights log(N/df) x type weight with df counted over S1, same cap 1000). Where does the TRUE S1 rank? (shares of pairs)

| pairs | rank 1 | 2-5 | 6-30 | 31-100 | >100 | not retrieved | n |
|---|---|---|---|---|---|---|---|
| a_no_shared_key | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 30 |
| b_only_capped_keys | 0.0615 | 0.0426 | 0.0483 | 0.0506 | 0.3561 | 0.4409 | 5,450 |
| c_ranked_below_K | 0.4171 | 0.1838 | 0.2004 | 0.1294 | 0.0693 | 0.0000 | 20,098 |
| all missed | 0.3408 | 0.1535 | 0.1677 | 0.1125 | 0.1303 | 0.0951 | 25,578 |
| control: 20k found pairs | 0.9792 | 0.0183 | 0.0022 | 0.0003 | 0.0000 | 0.0000 | 20,000 |

Missed pairs whose true S1 is in the reverse top 30: 16,935 (66.2%); top 5: 49.4%.


**Close match to another candidate of the same S1** (token_set_ratio >= 90 on name_core AND on the normalised address; only same-script comparisons can pass):

| cause | close to any candidate | exactly equal (name_core + address) | close to a TRUE-match candidate | ... that we also predicted | close to a non-match candidate |
|---|---|---|---|---|---|
| a_no_shared_key | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| b_only_capped_keys | 0.1396 | 0.0000 | 0.1279 | 0.1220 | 0.0308 |
| c_ranked_below_K | 0.2111 | 0.0225 | 0.1966 | 0.1894 | 0.0449 |
| all missed | 0.1956 | 0.0177 | 0.1818 | 0.1748 | 0.0418 |

**20 random missed pairs** (raw strings; fwd = forward rank of the true record, rev = reverse rank of the true S1, '-' = not retrieved; close = close to a candidate of the same S1)

| # | country | src | cause | fwd | rev | close | S1 name / address | missed S2/S3 name / address |
|---|---|---|---|---|---|---|---|---|
| 1 | US | S2 | c | 1584 | 2 | n | Silver Payment Dynamics Clinic / 40 Warren Avenue, Unit 2, Woburn, MA | Silver-Payment /  |
| 2 | US | S2 | b | - | 65 | n | Value Bay Black LLC / 1614 Leclaire St, Davenport, IA | Value 8ay Black /  |
| 3 | US | S2 | b | - | 1 | n | Nelson Aberdeen LLC / 92 Redwood Avenue, Ventura, CA | Nelson LLC Center / REDWOOD AVENUE, VENTURA, CA |
| 4 | US | S2 | c | 221 | 1 | y | Gulf Chile LLC / 7 Fenner Street, Billerica, MA | Gulf  LLC Services / www.gulfllc.com / FENNER ST, BILLERICA, MA |
| 5 | US | S3 | b | - | - | n | Chiropractic Modern Center LLC / 5324 Advance Mills Road, Albemarle County, VA | Chiropractic Mne Center LLC /  |
| 6 | US | S2 | c | 53 | 1 | n | Chen, Salli, Esq. / 37 Taft Street, Unit 1, Revere, MA | Chen, Salli, / TAFT STREET, REVERE, MA |
| 7 | India | S3 | c | 486 | 61 | n | Prime Agro Private Limited / A-3/1, Ground Floor, Acharya Niketan, Mayur Vihar Ph-1, Near Neelgiri Restaurant, East Delhi, Delhi | Primeagro.Com / A-3/1, East Delhi, दिल्ली |
| 8 | India | S2 | c | 77 | 1 | n | Premier Enterprises Private Limited / Nai Abadi Udgir, Latur, Udgir, Udgir, Latur, Maharashtra | प्रीमियर एंटरप्राइजेज प्राइवेट लिमिटेड / NAI ABADI UDGIR, LATUR, UDGIR, Maharashtra |
| 9 | US | S2 | c | 40 | 1 | n | Towers Made LLC / 370 Allen Road, Town Of Brookfield, WI | Towers Meba LLC / Allen Road, TOWN OF BROOKFIELD, WI |
| 10 | India | S3 | c | 83 | 2 | n | New Delhi Creative Pvt Ltd / R - 78 Greater Kailash - I, New Delhi, South Delhi, Delhi | Wexarc / R - 78 Greater Kailash - I, New Delhi, South Delhi, DL |
| 11 | India | S3 | c | 167 | 40 | n | Guru Management / 1966 2Nd Sector, Hsr Layout, Bangalore, Karnataka | GURU Management Enterprises /  |
| 12 | US | S2 | c | 773 | 2 | y | Pediatric Care Associates Inc / KY, 281 Battery Court, Fort Mitchell | Pediatric Care / 81 BATTERY CT, FORT MITCHELL, KY |
| 13 | US | S2 | b | - | 64 | n | Little Deli / 1801 Snyder Avenue, Dundalk, MD | Little  Dgmi / 1688 SNYDER AVE, DUNDALK, MD |
| 14 | US | S3 | c | 90 | 1 | n | Heart Guild LLC / 2315 Needham Drive, Covington, KY | heartguild.com / 2315 Needam Dr, Covington, Kentucky |
| 15 | US | S2 | c | 590 | 1 | n | Viix Ubs, LLC / 3334 Cherry Blossom Drive, Fl 1, Olympia, WA | Viix Uiebgb, LLC / WA, CHERRY BLOSSOM DR, OLYMPIA |
| 16 | India | S2 | b | - | - | n | Future Media / #30/54-1, 7Th Main 4Th Block, Jayanagar, Bengaluru, Bangalore, Karnataka | ಫ್ಯೂಚರ್ ಮೀಡಿಯಾ / NO 944/6 30592, BENGALURU, BANGALORE, Karnataka |
| 17 | India | S2 | c | 69 | 7 | n | Galaxy Management Private Limited / 401, Svr East Corner, 12Th Cross, Ananth Nagar, Bangalore South, Bangalore, Karnataka | ಗ್ಯಾಲಕ್ಸಿ ಮ್ಯಾನೇಜ್‌ಮೆಂಟ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್ / PLOT D/108 401, BANGALORE SOUTH, BANGALORE, Karnataka |
| 18 | India | S3 | c | 88 | 1 | n | Global Trading Private Limited / Unit No. 2, 5Th Office Floor, Building No. 9, K Raheja It Park, Mindspace Cyberabad Su, Rvey No.64, Madhapur, Hyderabad, Telangana | Global Trading Private / Unit No. 2, Madhapur, Hyderabad, TG |
| 19 | US | S2 | c | 697 | 106 | n | Surgical Clinic / NY, 3024 Cooley Road, Canandaigua | Surgical Clinic LP /  |
| 20 | India | S2 | c | 59 | 6 | n | Om Management Private Limited / D/29, S/F Kalkaji, Central Delhi, New Delhi, South Delhi, Delhi | ॐ मैनेजमेंट प्राइवेट लिमिटेड / HN 814 D/29, S/F KALKAJI, CENTRAL DELHI, NEW DELHI, SOUTH DELHI, Delhi |

## 2. Decision losses

Decision reproduced: macro F0.5 = 0.95030 (g=0.67, r=0.725, one-owner). True pairs that were candidates: 320,619 -> predicted 308,562, below threshold 11,853, removed by the one-owner rule 204.

Entities whose true matches were ALL candidates: 80,883 (incl. 5,555 singletons). Of these, 10,585 score < 1; they lose 1,700.1 entity-points = 0.0170 macro F0.5.

| part | entities | entity-points lost | macro F0.5 lost |
|---|---|---|---|
| singletons we predicted something for | 238 | 238 | 0.0024 |
| non-singletons predicted empty | 231 | 231 | 0.0023 |
| false positives in non-empty lists | 2,259 | 513.3432 | 0.0051 |
| misses (true candidates not picked) in non-empty lists | 8,135 | 717.7774 | 0.0072 |
| total | 10,585 | 1700.1206 | 0.0170 |

Non-singletons predicted empty: 231; best p (after one-owner) below the gate 0.67: 231; all candidates taken by the one-owner rule: 0. Singletons predicted non-empty: 238 (mean 1.08 predicted ids). FP/FN split inside non-empty lists = 2-player Shapley split of (1 - F).


**False positives** (3,136 predicted pairs that are wrong). 'owner' = the S1 the record truly belongs to; shares are of all FPs in the row; p_wrong = our p for the wrong S1.

| true owner of the wrong record | FPs | owner has it as candidate | median p_owner | p_owner >= 0.5 | p_owner > p_wrong | median p_wrong |
|---|---|---|---|---|---|---|
| owner is a fold-1 TRAINING S1 (p in-sample) | 137 | 0.9270 | 0.9691 | 0.8978 | 0.5839 | 0.8748 |
| owner is a validation S1 (competed in one-owner) | 33 | 0.8182 | 0.7080 | 0.6364 | 0.0000 | 0.9041 |
| owner is another train S1 (fresh run) | 1,091 | 0.9514 | 0.9589 | 0.8368 | 0.5628 | 0.8701 |
| record belongs to no S1 | 1,875 | 0.0000 | nan | 0.0000 | 0.0000 | 0.9061 |
| all FPs | 3,136 | 0.3801 | 0.9529 | 0.3371 | 0.2213 | 0.8914 |

FPs sitting in a singleton's list: 8.2%. Out-of-sample FPs (owner not in fold 1) that the owner would claim with a higher p (the one-owner rule removes them once all S1s compete, as on the test set): 614.

Rough what-if (only removing those FPs, ignoring that full competition can also take TRUE matches away): macro F0.5 0.9503 -> 0.9520.


## 3. Model features

The saved model uses **64 features** (models/lgb_model.txt, max_feature_idx=63; the '72 features' in PROGRESS.md was a miscount). Country is a model feature: **no**.

1. `score`, 2. `nkeys`, 3. `rank`, 4. `name_ratio`, 5. `name_tset`, 6. `name_tsort`, 7. `name_partial`, 8. `core_ratio`, 9. `core_tset`, 10. `core_jw`, 11. `core_eq`, 12. `roman_ratio`, 13. `roman_tset`, 14. `skel_ratio`, 15. `skel_tset`, 16. `addr_ratio`, 17. `addr_tset`, 18. `addr_partial`, 19. `core_shared`, 20. `core_jacc`, 21. `core_idf_cos`, 22. `ntok_core1`, 23. `ntok_core2`, 24. `skel_jacc`, 25. `addr_word_shared`, 26. `addr_word_jacc`, 27. `addr_idf_cos`, 28. `num_n1`, 29. `num_n2`, 30. `num_shared`, 31. `num_jacc`, 32. `num_miss1`, 33. `num_miss2`, 34. `num_first_eq`, 35. `num_first_in2`, 36. `len_core1`, 37. `len_core2`, 38. `len_addr1`, 39. `len_addr2`, 40. `nonlatin1`, 41. `nonlatin2`, 42. `cross_script`, 43. `msrc3`, 44. `combo`, 45. `n_cands`, 46. `score_gap`, 47. `score_rk`, 48. `name_tset_gap`, 49. `name_tset_rk`, 50. `core_jw_gap`, 51. `core_jw_rk`, 52. `skel_ratio_gap`, 53. `skel_ratio_rk`, 54. `addr_tset_gap`, 55. `addr_tset_rk`, 56. `num_jacc_gap`, 57. `num_jacc_rk`, 58. `core_idf_cos_gap`, 59. `core_idf_cos_rk`, 60. `addr_idf_cos_gap`, 61. `addr_idf_cos_rk`, 62. `combo_gap`, 63. `combo_rk`, 64. `n_strong`

**How the unseen country France is handled:** country appears in exactly one place, the blocking keys (keys.py / keys2.py): the country string is hashed and XOR-ed into every key, so a France S1 can only share keys with France pool records. Key df / IDF weights and the cap are computed per key over the test pool, which contains the France records, so France keys get their own frequencies. No feature, threshold or model split looks at the country value, so there is no one-hot and no 'unknown category' path. What is NOT covered: the model and the thresholds were fitted on US + India only, so their calibration on French names/addresses is unmeasured (no French labels exist). The normalisation has partial French support: accents stripped, legal words sarl/sas/sci/eurl/snc removed, avenue/boulevard/saint abbreviated; 'rue', 'chemin', 'allee', 'cedex' etc. are left as plain words.


## Where the loss goes (1.000 -> oracle -> actual), in macro F0.5 points

| step | loss | F0.5 after |
|---|---|---|
| **Perfect** | | **1.0000** |
| blocking (a) no shared key | -0.0000 | 1.0000 |
| blocking (b) only keys with df > 1000 (cap) | -0.0061 | 0.9939 |
| blocking (c) scored but rank > 30 | -0.0218 | 0.9721 |
| **= oracle at K=30 (perfect picking among the candidates)** | | **0.9721** |
| decision, fully covered: singletons predicted non-empty | -0.0024 | 0.9697 |
| decision, fully covered: non-singletons predicted empty | -0.0023 | 0.9674 |
| decision, fully covered: false positives in lists | -0.0051 | 0.9623 |
| decision, fully covered: true candidates not picked | -0.0072 | 0.9551 |
| decision, entities with blocking misses: predicted empty | -0.0014 | 0.9537 |
| decision, entities with blocking misses: false positives | -0.0010 | 0.9527 |
| decision, entities with blocking misses: true candidates not picked | -0.0024 | 0.9503 |
| **= validation macro F0.5** | | **0.9503** |
