# M12: v5d = v3 + dense top-1 additions for the records v3 leaves unassigned

Train S1s excluding part 1a (2,096,150); open train records with a dense list: 2,412,273 after removing part-1a S1s / records. Additions correct: 0.1173 overall.

v3 alone: half 1 0.96729, half 2 0.96733

Tuned on half 1: {'c1': 9.0, 'margin': 9.0, 'c1_empty': np.float64(0.58), 'margin_empty': 0.07, 'only_S1s_with_v3_matches': False}
**Half 2: v5d 0.96758 vs v3 0.96733 (+0.00025)**; additions on half 2: 5,547, correct 0.8635, of them empty-address 1.000.

| c1 >= | margin >= | half 2 F0.5 |
|---|---|---|
| 0.8 | 9.0 | 0.96758 |
| 0.85 | 9.0 | 0.96758 |
| 0.9 | 9.0 | 0.96758 |
| 0.93 | 9.0 | 0.96758 |
| 0.95 | 9.0 | 0.96758 |
| 0.97 | 9.0 | 0.96758 |

(173s)


## Test output -> C:\amlc\submission\output_v5d

v3 matches 5,745,731 + additions 17,311 (empty-address 1.000; by country {'US': 8254, 'India': 7829, 'France': 1228}) = 5,763,042 matches, mean 3.326 per S1, empty rows 0.0570; candidates 72,171,769. (119s)


## output_v5f: stricter empty gate for France only -> C:\amlc\submission\output_v5f

India+US empty share 0.0594; France empty share 0.0436 -> 0.0600 with gate g_France = 0.982 (others keep 0.68). France S1 rows emptied: 4,266 of 259,452. (86s)


## output_v5g: fewer matches per France row -> C:\amlc\submission\output_v5g

India+US mean matches per non-empty row 3.483; France 3.817 -> 3.481 with t_France = 0.96 (each row keeps its best). France matches dropped: 81,782. (139s)


## output_v5h: France gate 0.99 / pair cut 0.99 from output_v5d

France: empty 0.0647, matches per non-empty row 3.246, matches 787,694 (was 936,646). (129s)

