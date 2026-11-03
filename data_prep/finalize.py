"""Final cleaning for every prepared CSV: physical limits, outlier removal, gap filling (no NaN left), and a quality report.
Writes <name>.csv into <out_dir> (final/) and a summary json.  Usage: python finalize.py <prepared_dir> <final_dir>"""
import sys, os, glob, json, pandas as pd, numpy as np
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
rep = {}
for f in sorted(glob.glob(os.path.join(src, "*.csv"))):
    name = os.path.basename(f)[:-4]
    if name in ("eia_NY", "opsd_NL", "opsd_PL", "opsd_HU"): continue   # no solar data / too short / large gaps
    df = pd.read_csv(f, parse_dates=["date"]).set_index("date").sort_index()
    n0 = int(df[["load", "solar", "wind"]].isna().any(axis=1).sum()); fixes = {}
    for c in ["solar", "wind"]:
        fixes[f"{c}_negative"] = int((df[c] < 0).sum()); df[c] = df[c].clip(lower=0)
    med = df["load"].median()
    bad = (df["load"] < 0.3 * med) | (df["load"] > 2.5 * med)
    fixes["load_outliers"] = int(bad.sum()); df.loc[bad, "load"] = np.nan
    for c in ["wind", "solar"]:                                       # absurd spikes vs robust scale
        q = df[c].quantile(0.999) if df[c].notna().any() else np.nan
        sp = df[c] > 1.5 * q if q == q else pd.Series(False, index=df.index)
        fixes[f"{c}_spikes"] = int(sp.sum()); df.loc[sp, c] = np.nan
    df = df.interpolate(limit=6, limit_area="inside")
    for lag in (24 * 7, 24 * 14):                                     # long gaps: same hour of previous week(s), then ffill/bfill
        df = df.fillna(df.shift(lag))
    df = df.ffill(limit=48).bfill(limit=48)
    left = int(df.isna().any(axis=1).sum())
    if left: df = df.dropna()
    df.index.name = "date"; df.to_csv(os.path.join(out, name + ".csv"))
    rep[name] = dict(rows=len(df), start=str(df.index.min()), end=str(df.index.max()), initially_missing_rows=n0, fixes=fixes, dropped_rows=left,
                     columns=list(df.columns), solar_zero_share=round(float((df.solar <= 0).mean()), 3),
                     corr_load_solar=round(float(df.load.corr(df.solar)), 3), corr_load_wind=round(float(df.load.corr(df.wind)), 3), corr_solar_wind=round(float(df.solar.corr(df.wind)), 3))
    print(name, rep[name], flush=True)
json.dump(rep, open(os.path.join(out, "quality_report.json"), "w"), indent=1)
