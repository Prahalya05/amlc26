# M6 decision layer

validation S1 entities: 100,000; predict-nothing floor F0.5 = 0.0556

recall ceiling of the scored candidates: 0.9261

- **plain**: best macro F0.5 = 0.95018 at gate g=0.67, abs t=0.01, rel r=0.725
  - top-1 only if p>=0.5: 0.68217; all p>=0.5: 0.94774
- **one_owner**: best macro F0.5 = 0.95030 at gate g=0.67, abs t=0.01, rel r=0.725
  - top-1 only if p>=0.5: 0.68221; all p>=0.5: 0.94795

chosen: one_owner -> {'f05': np.float64(0.9503001984233616), 'g': 0.67, 't': 0.01, 'r': 0.725, 'one_owner': True}

