"""OPSD time series (60 min) + OPSD weather (MERRA-2: temperature, direct and diffuse radiation) -> one CSV per country:
date(UTC), load, solar, wind, temp, rad_dir, rad_dif.  Usage: python prep_opsd.py <datasets_dir> <out_dir>"""
import sys, os, json, zipfile, io, pandas as pd, numpy as np
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
COUNTRIES = ["DE", "ES", "FR", "DK", "PT", "IT", "AT", "BE", "GR"]   # NL, PL, HU dropped: large gaps / too short
zt = zipfile.ZipFile(os.path.join(src, "opsd", "opsd-time_series-2020-10-06.zip"))
zw = zipfile.ZipFile(os.path.join(src, "opsd", "opsd-weather_data-2020-09-16.zip"))
tn = [n for n in zt.namelist() if n.endswith("time_series_60min_singleindex.csv")][0]
wn = [n for n in zw.namelist() if n.endswith("weather_data.csv")][0]
hdr = pd.read_csv(zt.open(tn), nrows=1).columns
want = ["utc_timestamp"] + [c for cc in COUNTRIES for c in (f"{cc}_load_actual_entsoe_transparency", f"{cc}_solar_generation_actual",
        f"{cc}_wind_generation_actual", f"{cc}_wind_onshore_generation_actual", f"{cc}_wind_offshore_generation_actual") if c in hdr]
ts = pd.read_csv(zt.open(tn), usecols=want, parse_dates=["utc_timestamp"]).set_index("utc_timestamp")
wh = pd.read_csv(zw.open(wn), nrows=1).columns
wcols = ["utc_timestamp"] + [c for cc in COUNTRIES for c in (f"{cc}_temperature", f"{cc}_radiation_direct_horizontal", f"{cc}_radiation_diffuse_horizontal") if c in wh]
w = pd.read_csv(zw.open(wn), usecols=wcols, parse_dates=["utc_timestamp"]).set_index("utc_timestamp")
w = w[w.index >= "2014-12-01"]
report = {}
for cc in COUNTRIES:
    L, S = f"{cc}_load_actual_entsoe_transparency", f"{cc}_solar_generation_actual"
    Wt, Won, Wof = f"{cc}_wind_generation_actual", f"{cc}_wind_onshore_generation_actual", f"{cc}_wind_offshore_generation_actual"
    if L not in ts or S not in ts: report[cc] = "missing load or solar"; continue
    if Wt in ts and ts[Wt].notna().mean() > 0.5: wind = ts[Wt]
    elif Won in ts: wind = ts[Won].add(ts[Wof], fill_value=0) if Wof in ts else ts[Won]
    else: report[cc] = "missing wind"; continue
    df = pd.DataFrame({"load": ts[L], "solar": ts[S], "wind": wind})
    for k, n in (("temp", "temperature"), ("rad_dir", "radiation_direct_horizontal"), ("rad_dif", "radiation_diffuse_horizontal")):
        df[k] = w[f"{cc}_{n}"] if f"{cc}_{n}" in w else np.nan
    df = df[df.index >= "2015-01-01"]
    wv = df["temp"].notna(); df = df.loc[wv.idxmax():wv[::-1].idxmax()]      # keep only the period where weather exists (ends ~2019-12)
    ok = df[["load", "solar", "wind"]].notna().all(axis=1)
    if ok.sum() == 0: report[cc] = "no overlapping rows"; continue
    df = df.loc[ok.idxmax():ok[::-1].idxmax()]                   # trim to first/last complete row
    miss = df.isna().mean().round(4).to_dict()
    df = df.interpolate(limit=6, limit_area="inside")
    left = int(df.isna().any(axis=1).sum()); df.index.name = "date"
    df.to_csv(os.path.join(out, f"opsd_{cc}.csv"))
    report[cc] = dict(rows=len(df), start=str(df.index.min()), end=str(df.index.max()), missing_before_fill=miss, rows_still_missing=left,
                      solar_zero_share=round(float((df.solar.fillna(0) <= 0).mean()), 3), mean=df.mean().round(1).to_dict(), max=df.max().round(1).to_dict(), min=df.min().round(1).to_dict())
    print(cc, report[cc], flush=True)
json.dump(report, open(os.path.join(out, "opsd_report.json"), "w"), indent=1)
