# Rule 6: Indic romanisation check (train)

Pairs: 20,000 true cross-script pairs (S1 Latin, match non-Latin), 19,998 random non-matching pairs (pool names shuffled), 20,000 hard non-matching pairs (non-Latin blocking candidates of validation S1s).

Similarity = rapidfuzz score 0-100 between the S1 core name and the transformed pool core name.

| feature | set | mean | p10 | p25 | p50 | p75 | p90 |
|---|---|---|---|---|---|---|---|
| roman_tset | TRUE | 55.0 | 37 | 44 | 53 | 63 | 76 |
| roman_tset | random | 30.8 | 23 | 26 | 30 | 35 | 39 |
| roman_tset | hard | 31.5 | 22 | 26 | 30 | 35 | 40 |
| roman_ratio | TRUE | 53.9 | 39 | 45 | 52 | 61 | 71 |
| roman_ratio | random | 30.6 | 22 | 26 | 30 | 35 | 39 |
| roman_ratio | hard | 30.7 | 21 | 26 | 30 | 35 | 39 |
| skel_ratio | TRUE | 93.3 | 78 | 90 | 100 | 100 | 100 |
| skel_ratio | random | 41.4 | 22 | 33 | 43 | 50 | 59 |
| skel_ratio | hard | 41.7 | 22 | 33 | 43 | 50 | 59 |
| raw_ratio | TRUE | 7.9 | 4 | 4 | 5 | 6 | 9 |
| raw_ratio | random | 6.0 | 4 | 4 | 5 | 6 | 8 |
| raw_ratio | hard | 7.0 | 4 | 4 | 5 | 7 | 9 |

| feature | AUC true vs random | AUC true vs hard |
|---|---|---|
| roman_tset | 0.9404 | 0.9256 |
| roman_ratio | 0.9586 | 0.9479 |
| skel_ratio | 0.9810 | 0.9745 |
| raw_ratio | 0.5190 | 0.4971 |

Histogram of roman_tset (share of pairs per 10-point bin):

| bin | TRUE | random | hard |
|---|---|---|---|
| 0-10 | 0.000 | 0.001 | 0.004 |
| 10-20 | 0.002 | 0.043 | 0.059 |
| 20-30 | 0.028 | 0.418 | 0.408 |
| 30-40 | 0.111 | 0.447 | 0.429 |
| 40-50 | 0.246 | 0.080 | 0.071 |
| 50-60 | 0.294 | 0.008 | 0.007 |
| 60-70 | 0.165 | 0.002 | 0.003 |
| 70-80 | 0.075 | 0.001 | 0.002 |
| 80-90 | 0.039 | 0.000 | 0.004 |
| 90-100 | 0.040 | 0.000 | 0.014 |

## 30 true pairs: S1 core name | pool core name -> romanised (score)

- `white finance` | `व्हाइट फाइनेंस प्रा लि` -> `vhait phainens pra li` (47)
- `raj infotech` | `ರಾಜ್ ಇನ್ ಫೋಟೆಕ್` -> `raj in photek` (72)
- `fortune infra` | `ఫార్చూన్ ఇన్ ఫ్రా ప్రైవేట్ లిమిటెడ్` -> `pharchun in phra praivet limited` (27)
- `real consultancy` | `रियल कंसल्टेंसी प्रा लि` -> `riyal kansaltensi pra li` (45)
- `real infotech` | `real इंफोटेक प्राइवेट लिमिटेड` -> `real inphotek praivet limited` (48)
- `my technology` | `മൈ technology പ്രൈവറ്റ് ലിമിറ്റഡ്` -> `mai technology praivatt limittad` (87)
- `shiva business` | `শিব বিজনেস প্রাইভেট লিমিটেড` -> `shib bijanes praibhet limited` (42)
- `all care` | `ऑल care` -> `ol care` (80)
- `silver shiva impex` | `सिल्वर शिवा इम्पेक्स प्राइवेट लिमिटेड` -> `silvar shiva impeks praivet limited` (60)
- `one enterprises` | `वन एंटरप्राइजेज लिमिटेड` -> `van entarapraijej limited` (50)
- `shivam future tech` | `ಶಿವಂ ಫ್ಯೂಚರ್ ಟೆಕ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್` -> `shivan phyuchar tek praivet limited` (49)
- `eastern healthcare` | `ઈસ્ટર્ન હેલ્થકેર પ્રા લિ` -> `istarn helthaker pra li` (44)
- `new business` | `নিউ বিজনেস` -> `niu bijanes` (61)
- `royal exports` | `రాయల్ ఎక్స్ పోర్ట్స్ ప్రైవేట్ లిమిటెడ్` -> `rayal eks ports praivet limited` (50)
- `jain industries` | `जैन इंडस्ट्रीज प्रा लि` -> `jain indastrij pra li` (67)
- `south agro` | `ಸೌತ್ ಅಗ್ರೋ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್` -> `saut agro praivet limited` (57)
- `first impex` | `ఫస్ట్ ఇంపెక్స్ లిమిటెడ్` -> `phast inpeks limited` (39)
- `my projects` | `માય પ્રોજેક્ટ્સ પ્રાઇવેટ લિમિટેડ` -> `may projekts praivet limited` (51)
- `all vision foods` | `ऑल vision फूड्स` -> `ol vision phuds` (71)
- `ss care` | `एसएस केयर प्राइवेट लिमिटेड` -> `esaes keyar praivet limited` (24)
- `bharat management` | `भारत मैनेजमेंट प्राइवेट लिमिटेड` -> `bharat mainejament praivet limited` (59)
- `guru media` | `गुरु मीडिया प्राइवेट लिमिटेड` -> `guru midiya praivet limited` (57)
- `silver investments` | `सिल्वर इन्वेस्टमेंट्स` -> `silvar investaments` (92)
- `alpha enterprises` | `अल्फा एंटरप्राइजेज प्रा लि` -> `alpha entarapraijej pra li` (65)
- `all agro` | `অল অ্যাগ্রো` -> `al ayagro` (47)
- `life hospitality` | `लाइफ हॉस्पिटैलिटी` -> `laiph hospitailiti` (76)
- `white consulting` | `व्हाइट consulting एलएलपी` -> `vhait consulting elaelapi` (77)
- `lakshmi hospitality` | `लक्ष्मी हॉस्पिटैलिटी प्रा लि` -> `lakshmi hospitailiti pra li` (78)
- `city arihant investments` | `সিটি অরিহন্ত ইনভেস্টমেন্টস প্রাইভেট লিমিটেড` -> `siti arihant inabhestamentas praibhet limited` (52)
- `krishna consultants` | `कृष्ण कंसल्टेंट्स प्राइवेट लिमिटेड` -> `krishn kansaltents praivet limited` (57)
