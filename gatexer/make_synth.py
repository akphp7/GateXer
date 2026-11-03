"""SMOKE-TEST data only (synthetic). Not evidence of anything about real grids."""
import numpy as np, pandas as pd
rng = np.random.default_rng(0)
n = 24 * 365 * 3
t = pd.date_range("2021-01-01", periods=n, freq="h")
h = t.hour.values; doy = t.dayofyear.values; dow = t.dayofweek.values
def ar(n, phi, s):
    x = np.zeros(n); e = rng.normal(0, s, n)
    for i in range(1, n): x[i] = phi * x[i-1] + e[i]
    return x
cloud = np.clip(0.5 + ar(n, 0.97, 0.06), 0, 1)
ws = np.clip(7 + ar(n, 0.98, 0.5) + 2*np.sin(2*np.pi*doy/365), 0, None)
temp = 28 + 6*np.sin(2*np.pi*(doy-100)/365) + 4*np.sin(2*np.pi*(h-14)/24) + ar(n, 0.98, 0.3)
clear = np.clip(np.sin(np.pi*(h-6)/12), 0, None) * (0.8 + 0.2*np.sin(2*np.pi*(doy-80)/365))
solar = 100*clear*(1-0.8*cloud) + rng.normal(0, 1.5, n)*(clear > 0)
wind = 100*np.clip((ws/12)**3, 0, 1) + rng.normal(0, 3, n)
load = 300 + 40*np.sin(2*np.pi*(h-19)/24) - 25*(dow >= 5) + 6*np.clip(temp-28, 0, None) + ar(n, 0.9, 4)
df = pd.DataFrame(dict(date=t, load=load, solar=np.clip(solar, 0, None), wind=np.clip(wind, 0, None),
                       temp=temp + rng.normal(0, 0.8, n), cloud=cloud + rng.normal(0, 0.05, n), wspeed=ws + rng.normal(0, 0.6, n)))
df.to_csv("synth.csv", index=False); print(df.shape)
