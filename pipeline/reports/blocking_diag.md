# Blocking diagnostics (10k validation S1, cap 2000)

true pairs 34,694; missed at k=20: 0.1134; not in top-100: 0.0699

Among pairs missed at k=20:
- share no key at all: 0.0015
- share keys, but all above cap (df>2000): 0.2237
- have >=1 allowed shared key (ranked too low): 0.7747

Shared key types among missed pairs (share of missed pairs having that type shared):
- type 1: 0.5487 (median df 13792)
- type 2: 0.2222 (median df 160)
- type 3: 0.7249 (median df 34630)
- type 4: 0.8403 (median df 24934)
- type 5: 0.5228 (median df 14355)
- type 6: 0.2255 (median df 70)
- type 7: 0.1042 (median df 94)

Shared key types among FOUND pairs (rank<20):
- type 1: 0.8846 (median df 6814)
- type 2: 0.7237 (median df 14)
- type 3: 0.8965 (median df 26334)
- type 4: 0.9611 (median df 7360)
- type 5: 0.8140 (median df 5726)
- type 6: 0.7279 (median df 7)
- type 7: 0.5634 (median df 5)

## 40 missed pairs (rank, #shared keys, #allowed)

- rank nan sh 6/1: `jai finance limited` / `39a ashoka ave sainik farms delhi s delhi delhi` <-> `जय फाइनेंस लिमिटेड` / `h no 39 ashoka ave sainik farms new delhi delhi dl`
- rank nan sh 3/1: `asc services private limited` / `no 48 yash classic enclave hennur crs chellakere kalyanagar post bangalore karnataka` <-> `private asc 5ervices limited` / `no 48 e of ngef layout ಕರ್ನಾಟಕ`
- rank nan sh 6/0: `eastern balaji enterprises private limited` / `44 agrasen ngr girwa udaipur rajasthan` <-> `ईस्टर्न बालाजी एंटरप्राइजेज प्राइवेट लिमिटेड` / `472 44 girwa udaipur राजस्थान`
- rank nan sh 3/1: `davis and rivas atlanticus inc` / `9066 pine dr rochester in` <-> `davisrivasatlanticuscom` / `9066 a pine dr rochester indiana`
- rank nan sh 5/1: `united network` / `505 goodrich cove leander tx` <-> `united netwhork` / `505b goodrich cove leander texas`
- rank 26.0 sh 2/1: `vega` / `fayetteville 4216 pleasantburg dr nc` <-> `xyloevo` / `421 pleasantburg dr fayetteville nc`
- rank nan sh 2/1: `metropolitan supply brands llc` / `576 31st st northwood nd` <-> `wexirionyx` / `31st st northwood n dakota`
- rank nan sh 6/1: `gujarat lotus food pvt ltd` / `9 anthony bagan ln kolkata calcutta w bengal` <-> `গুজরাট লোটাস ফুড প্রাইভেট লিমিটেড` / `calcutta wb 9 anthiny bagan ln kolkata region`
- rank 58.0 sh 3/1: `star hospitality pvt ltd` / `javahar pole anand gujarat` <-> `starhospitalitycom` / `javahar pole anand n a ગુજરાત`
- rank 25.0 sh 5/2: `pink research private limited` / `4 1208a 1st fl royaltower opp arya vaidya sala factory complex kottakkal malappuram malapuram kerala` <-> `pinkresearch` / `4 1208a malapuram malappuram kl`
- rank 86.0 sh 7/1: `laxmi international private limited` / `nadia b 12 98 kalyani w bengal central park e` <-> `dr laxmiinternationalcom` / `w bengal b 12 98 central park e kalyani`
- rank nan sh 2/0: `webers safe cafe` / `15314 carrollton blvd unit c isle of wight county va` <-> `weberssafecafecom` / `5314 carrollton blvd carrolllton va`
- rank nan sh 2/0: `dk and sons private limited` / `no 63 katha no 6 agara tataguni post kengeri bangalore s bangalore karnataka` <-> `nexgild` / `no 063 bangalore s bangalore ka`
- rank nan sh 5/0: `bespoke planet private limited` / `shop no 8 2nd fl 10795 rexine bazar nabi karim jhandewalan rd new delhi central delhi delhi` <-> `pbplanetcom` / `shop no 8 new delhi central delhi delhi`
- rank nan sh 5/1: `pediatric dentistry care llc` / `1055 sugar maple dr crystal lake il` <-> `pediatricdentistrycarecom` / `1055 1059 sugar maple dr crystal lake il`
- rank 96.0 sh 11/2: `smart constructions private limited` / `doctor centre premises chs 3r fl offfice no 301 135 august kranti marg mumbai mumbai city maharashtra` <-> `smart constructions limited services` / `doctor centre premises chs mumbai mumbai city mh`
- rank 29.0 sh 4/1: `east analytics inc` / `oakland me 40 high st` <-> `eastanalyticscom` / `40 high st oakland maine`
- rank 36.0 sh 4/2: `ramkripa clinic` / `mysore no d 12 gr fl 06th mn rd jaylakshmipuram karnataka` <-> `ramkripa center` / `no d 12 mysuru mysore ಕರ್ನಾಟಕ`
- rank nan sh 8/1: `dental empire clinic` / `revere ma 400 ocean ave unit 435` <-> `dental empire` / `ocean ave po box 2154 revere massachusetts`
- rank 74.0 sh 10/2: `surya systems private limited` / `158 dumdum rd bagjola ghosh para kolkata kolkata howrah w bengal` <-> `সূর্য সিস্টেমস প্রাইভেট লিমিটেড` / `158 calcutta dumdum rd bagjola ghosh para kolkata w bengal howrah`
- rank nan sh 2/1: `highland association llc` / `5157 scarecrow ct columbia md` <-> `associationhighlandcom` / `157 scarecrow ct columbia md`
- rank nan sh 4/0: `delta architecture inc` / `900 fourth st delaware city de` <-> `delta inc center` / `900d 4th st delaware city delaware`
- rank nan sh 3/0: `karie h tulloch od dds pc` / `468 valley dr unit 101 naperville il` <-> `karieh` / `46 valley dr naperville township il`
- rank nan sh 5/0: `new delhi developers ltd` / `plot no 121 2nd fl blk 1 whs kirti ngr ind area new delhi w delhi delhi` <-> `new delhi devel0pers ltd` / ``
- rank 59.0 sh 8/2: `jain impex limited` / `shop no 10 4 102 ist fl plot 13 sec 10 odean mlu plaza dwarka delhi s w delhi delhi` <-> `जैन इम्पेक्स लिमिटेड` / `no 10 4 102 n a delhi s w delhi delhi`
- rank nan sh 8/1: `cure deli` / `717 5th ave huntington wv` <-> `cure dloai` / `717 5th ave huntington wv`
- rank nan sh 4/0: `jai industries private limited` / `4 devyani apartments iti rd aundh pune maharashtra` <-> `जय इंडस्ट्रीज प्राइवेट लिमिटेड` / `block a 607 4 pune maharashtra`
- rank nan sh 4/1: `life consultancy` / `c o shailendra kumar gram bhuhar ward haiderg lucknow uttar pradesh` <-> `life consu1tancy` / `c o shailendra kumar lucknow up`
- rank nan sh 6/1: `jupiter charitable trust corporation` / `no 29 1 a crs 2nd mn cambridge layout extension ulsoor bangalore karnataka` <-> `m s jupiter trust corporation center` / `h no 29 bangalore ka`
- rank nan sh 5/1: `seven media private limited` / `sambhal uttar pradesh moradabad 736 deepa sarai` <-> `सेवन मीडिया प्राइवेट लिमिटेड` / `g 736 moradabad uttar pradesh`
- rank 52.0 sh 4/1: `koontz utilities inc` / `544 pearl st brownsville borough pa` <-> `koontz inc center` / `54 pearl st pa brownsville`
- rank nan sh 5/1: `literacy network holdings` / `31 clark dr barboursville wv` <-> `literacy network network` / ``
- rank nan sh 5/1: `custom solana inc` / `8926 redmond woodinville rd redmond wa` <-> `custom slaona inc` / `redmond woodinville rd redmond city washington`
- rank nan sh 4/1: `gulf initiative llc` / `2136 churchill dr owensboro ky` <-> `gulf llc services` / `churchill dr owensboro townhip ky`
- rank 46.0 sh 9/2: `interstate preparatory academy` / `9105 state hwy 66 unit apt 2205 sapulpa ok` <-> `interstate preparatory academy inc` / ``
- rank nan sh 8/1: `family care inc` / `20 dartmouth st winchester ma` <-> `family care inc` / `dartmouth st winchester massachusetts`
- rank 28.0 sh 5/3: `allie coombes mortgage llc` / `21 pack castle rd wayland ky` <-> `fayedova` / `21 pack castle rd wayland ky`
- rank 70.0 sh 6/1: `wells and raymond corp` / `17607 foothills dr sun city az` <-> `wells and ramnd corp` / `foothills dr sun city cdp az`
- rank 25.0 sh 10/4: `wilson signature reit llc` / `3851 cobble ridge dr unit 5301 w jordan city ut` <-> `wilson sigagnue reit llc` / `03851 cobble ridge dr po box 5766 w jordan city utah`
- rank nan sh 2/0: `precision cantor` / `249 troublesome creek rd campbell county va` <-> `precision partners partners enterprises` / ``
