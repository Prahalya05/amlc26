# M2 EDA report (train unless stated)

S1 entities: 2,206,821; true pairs: 7,638,365
**Singleton fraction = predict-nothing macro F0.5 = 0.0558**

## Matches per S1 (10 = 10+)

| n_match | S1 count | share |
|---|---|---|
| 0 | 123,247 | 0.0558 |
| 1 | 119,157 | 0.0540 |
| 2 | 375,212 | 0.1700 |
| 3 | 530,841 | 0.2405 |
| 4 | 484,115 | 0.2194 |
| 5 | 321,957 | 0.1459 |
| 6 | 164,868 | 0.0747 |
| 7 | 63,968 | 0.0290 |
| 8 | 18,680 | 0.0085 |
| 9 | 4,205 | 0.0019 |
| 10 | 571 | 0.0003 |

mean 3.461, mean among non-singletons 3.666, max 11

## Source of matched records

| msrc | proportion |
|---|---|
| S3 | 0.5164 |
| S2 | 0.4836 |

Per non-singleton S1: mean #S2 matches 1.773, mean #S3 matches 1.893; share with >=1 S2 0.9211, share with >=1 S3 0.9314

Distribution of #S2 matches per non-singleton S1: {0: 164498, 1: 789108, 2: 652779, 3: 333957, 4: 119078, 5: 24154}
Distribution of #S3 matches per non-singleton S1: {0: 143029, 1: 716417, 2: 668375, 3: 372443, 4: 145116, 5: 35378, 6: 2816}

## Is each S2/S3 id matched to at most one S1?

S2/S3 ids appearing under >1 S1: 0 (of 7,638,365 matched ids)

## Country

| country | S1 | singleton_rate | mean_matches | S2 | S3 |
|---|---|---|---|---|---|
| US | 1323633 | 0.0558 | 3.4591 | 3016817 | 3170056 |
| India | 883188 | 0.0559 | 3.4645 | 2017799 | 2115547 |

True pairs whose S2/S3 country equals the S1 country: 1.000000
Share of S2/S3 pool records that are a true match of some train S1: 0.7401 (S2 0.7336, S3 0.7463)

## Scripts

| file | name Devanagari | name non-Latin | addr Devanagari | addr non-Latin |
|---|---|---|---|---|
| train S1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| train S2 | 0.0535 | 0.0942 | 0.0551 | 0.0948 |
| train S3 | 0.0299 | 0.0527 | 0.0522 | 0.0900 |
| test S1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
|  countries: {'India': 809986, 'US': 663106, 'France': 259452} | | | | |
| test S2 | 0.0632 | 0.1118 | 0.0651 | 0.1127 |
|  countries: {'India': 2312565, 'US': 1871330, 'France': 703378} | | | | |
| test S3 | 0.0356 | 0.0631 | 0.0628 | 0.1083 |
|  countries: {'India': 2405000, 'US': 1945701, 'France': 731615} | | | | |

True pairs where the two NAMES are in different scripts (one non-Latin, other Latin): 0.0722; S1 name non-Latin: 0.0000; match name non-Latin: 0.0722
Cross-script pairs by country: {'US': 0.0, 'India': 0.1802}
Non-Latin match names by source: {2: 0.0933, 3: 0.0524}

## String agreement of true pairs (300k sample)

Latin-Latin pairs: 0.9266 of sample
| statistic | all pairs | by country |
|---|---|---|
| name_n exactly equal | 0.2668 | {'India': 0.1903, 'US': 0.318} |
| name_core exactly equal | 0.5134 | {'India': 0.4455, 'US': 0.5587} |
| share >=1 name_core token | 0.8494 | {'India': 0.7534, 'US': 0.9136} |
| share >=1 name_n token | 0.8544 | {'India': 0.7587, 'US': 0.9184} |
| share >=1 address number | 0.8259 | {'India': 0.8401, 'US': 0.8165} |
| S1 has no address number | 0.0347 | {'India': 0.0866, 'US': 0.0} |
| match has no address number | 0.1216 | {'India': 0.1223, 'US': 0.121} |
| match address empty | 0.0444 | {'India': 0.0401, 'US': 0.0473} |
| first address number equal | 0.7252 | {'India': 0.6577, 'US': 0.7704} |
| share >=1 non-number addr token | 0.9542 | {'India': 0.9588, 'US': 0.9511} |
| share >=1 name token OR >=1 addr number | 0.9795 | {'India': 0.9679, 'US': 0.9873} |

## 40 random true pairs

- `roanna s combs pa` / `749 cascada st el paso county tx`  <->  `roanna s combs pa lp` / `749 cascada st el paso county texas` (US, S3)
- `shiv trading private limited` / `plot no 2a 1st fl kh no 294 kehar singh state saidulajab village l ane no 2 new delhi s w delhi delhi`  <->  `smt shiv trading private ltd` / `` (India, S3)
- `artis and good tech` / `22318 irma ln surprise az`  <->  `artis and good tech` / `` (US, S3)
- `hill best dynamic llc` / `16296 310th ave isle mn`  <->  `hill best dynamic` / `` (US, S2)
- `one constructions private limited` / `hingoli plot no 11 a hingoli maharashtra anand ngr balsond`  <->  `oneconstructionscom` / `plot no 11 a hingoli mh` (India, S3)
- `silver innovative management` / `rajkot c o nijamudin kadivar gujarat wankaner`  <->  `સિલ્વર ઇનોવેટિવ મેનેજમેન્ટ` / `211 c o nijamudin kadivar wankaner rajkot ગુજરાત` (India, S3)
- `brigham and lander cleaning service inc` / `9 treetop ct burlington ma`  <->  `brigham and lander cleaning service inc` / `burlington 9 treetop ct massachusetts` (US, S3)
- `tacoma foot and ankle midwest medicine inc` / `wa 2901 15th st tacoma`  <->  `tacoma foot ankle midwest medicine inc` / `2901 fifteenth st po box 7751 tacoma wa` (US, S2)
- `mpm systems llp` / `maharashtra pune the lake district sn 15 16 p yewalewadi nr bapdev ghat fl a3 1104 pune`  <->  `mpm systems llp` / `fl a3 1104 the lake district sn 15 16 p yewalewadi nr bapdev ghat pune maharashtra` (India, S2)
- `modern ramaco llc` / `1878 1100 rd milford il`  <->  `modern raho llc` / `1878 1100 rhad milford il` (US, S2)
- `gujarat north exports` / `innov8 times sq unt no 2 4th fl a wing time sq bldg andheri kurla rd marol nilkanth park andheri e mumbai maharashtra`  <->  `गुजरात नॉर्थ एक्सपोर्ट्स` / `innov8 times sq unt no 2 mumbai maharashtra` (India, S2)
- `finkle empire solid llc` / `1595 bateman st fl 1 galesburg il`  <->  `finkle llc solid empire` / `1595 bateman st galesburg il` (US, S2)
- `exim business private limited` / `j 114 ashok vihar phase 1 montfort new delhi delhi`  <->  `shri exim private limited services` / `j 114 ashok vihar phase 1 montfort new delhi dl` (India, S3)
- `bandana trading private limited` / `1st fl room no 1 kolkata howrah 315 behari mondal rd kolkata w bengal`  <->  `bandana trading private ltd` / `h no 315 behari mondal rd 1st fl room no 1 kolkata kolkata w bengal` (India, S2)
- `rapid supreme hotel inc` / `endicott 401 hayes ave unit ste 204 ny`  <->  `rapid sreue hotel inc` / `00401 hayes ave endicott ny` (US, S2)
- `guru services private limited` / `no 1973 a trichy rd coimbatore s coimbatore tamil nadu`  <->  `குரு சர்வீசஸ் பிரைவேட் லிமிடெட்` / `no 1973 a trichy rd coimbatore tamil nadu` (India, S2)
- `environmental league` / `762 sunny river rd unit 1513 taylorsville city ut`  <->  `environmental league` / `62 sunny river rd taylorsville city ut` (US, S2)
- `royal logistics technologies` / `9625 david taylor dr nc charlotte unit 113`  <->  `royal logistics technologies inc` / `` (US, S2)
- `ritu brothers private limited` / `103 kamalapur e dum dum kolkata howrah w bengal`  <->  `ritu brothers private` / `10 kolkata howrah wb` (India, S3)
- `solutions best consulting group` / `2155 tapkir galli ahmednagar ahmednagar maharashtra`  <->  `solutions best consulting group` / `2155 tapkir galli ahmednagar ahmednagar mh` (India, S3)
- `cherish baker manufacturing llc` / `3645 old hanover rd westminster md`  <->  `cherish baker manufacturing llc` / `westminster 3645 old hanover rd md` (US, S2)
- `global himgiri engineers private limited` / `8 44 sec 3 rajendra ngr ghaziabad uttar pradesh`  <->  `global himgiri engineers private limited` / `8 44 sec 3 rajendra ngr ghaziabad उत्तर प्रदेश` (India, S2)
- `sf classic ngl llc` / `109 yorktown rd amherst ny`  <->  `sf classic ngl llc` / `ny 109 yorktown rd amherst` (US, S2)
- `department of buildings` / `30 brush hill rd clinton ct`  <->  `department of 8uilnaigs` / `30 brush hill rd clinton connecticut` (US, S3)
- `raj balaji enterprises pvt ltd` / `29 786 a nr sn college bus stop kannur kerala`  <->  `രാജ് ബാലാജി എന്റർപ്രൈസസ് പ്രൈവറ്റ് ലിമിറ്റഡ്` / `29 786 a nr sn college bus stop kannur കേരളം` (India, S2)
- `gernert garcia and lee loan group` / `935 burlington ave unit 302 downers grove il`  <->  `gernert garcia and lee loan` / `downers grove il 935 burlington ave` (US, S2)
- `pediatric dentistry royal specialists` / `mt 213 mn st three forks`  <->  `pediatric dentistry royal` / `213 mn st three forks montana` (US, S3)
- `verrora detailing pc` / `809 carnes st gainesville tx`  <->  `verroradetailing` / `809 carnes st texas gainesville cpd` (US, S3)
- `dream solutions private limited` / `d no 14 5 356 and 357 baidiwadi begum bazar hyderabad telangana`  <->  `dream solltutcions private limited` / `14 5 356 and 357 baidiwadi begum bazar hyderabad tg` (India, S3)
- `deegan food` / `907 dashwood ct mishawaka in`  <->  `deegan 5ervice` / `907 c dashwood ct mshawaka indiana` (US, S3)
- `venitas clear family practice` / `3155 jason pl w valley city ut`  <->  `venitas clear family` / `003155 jason pl w valley city utah` (US, S3)
- `bombay media` / `20 2003 anjaneyam nedumangad thiruvananthapuram trivandrum kerala`  <->  `ബോംബെ media` / `20 2003 thiruvananthapuram trivandrum kerala` (India, S2)
- `new labs pvt ltd` / `398 kirti kunj 14th rd khar w mumbai mumbai city maharashtra`  <->  `new pvt labs ltd` / `no 831 398 mumbai mumbai city mh` (India, S3)
- `pediatric dentistry specialists of sistersville lp` / `120 virginia st sistersville wv`  <->  `pediatric dentistry specialists of sistersville` / `sistersville 20 virginia st w virginia` (US, S3)
- `vivekanand financiers ltd` / `d 1203 shypram southbopal safal parisar rd daskroi ahmedabad gujarat`  <->  `umbradelta` / `hn 987 d 1203 shypram southbopal safal parisar rd ahmedabad daskroi gujarat` (India, S2)
- `palace india broadcast care` / `block no 4 fl 1 plot 47a vidya prasad laxmibai kelkar marg sion mumbai mumbai city maharashtra`  <->  `palace india care 5ervice` / `block no 6 4 fl 1 plot 47a vidya prasad laxmibai kelkar marg sion mumbai mumbai city mh` (India, S3)
- `american sumitomo llc` / `2050 gerrard ave indianapolis in`  <->  `american llc partners` / `2050 gerrard ave indiana indianapolis` (US, S3)
- `chiropractic coastal associates llc` / `wa spokane 520 kiernan ave`  <->  `chiropractic coastal associates llc` / `washington 520 kiernan ave spokane citty` (US, S3)
- `abbott and helmer financial` / `5307 windsor ave phoenix az`  <->  `abbott financial helmer and` / `5307 windsor ave phoenix arizona` (US, S3)
- `nln courier` / `126 157 1e 403 princeton apts manipal county rd bangalore s bangalore karnataka`  <->  `nln conrurm` / `0126 157 1e 403 princeon apts manipal county rd bangalore ka` (India, S3)
