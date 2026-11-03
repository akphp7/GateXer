"""EIA-930 BA/region xlsx -> hourly CSV [date(UTC), load, solar, wind]. Solar/wind by source exist from 2018-07.
Usage: python prep_eia.py <datasets_dir> <out_dir>"""
import sys, os, json, openpyxl, pandas as pd, numpy as np
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
FILES = {"ERCO": "ERCO.xlsx", "CISO": "CISO.xlsx", "SWPP": "SWPP.xlsx", "MISO": "MISO.xlsx", "NY": "Region_NY.xlsx"}
START = pd.Timestamp("2018-07-01")
report = {}
for name, fn in FILES.items():
    p = os.path.join(src, "eia930", fn)
    if not os.path.exists(p): continue
    if os.path.exists(os.path.join(out, f'eia_{name}.csv')) and '--force' not in sys.argv: print('skip', name, flush=True); continue
    wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
    ws = wb["Published Hourly Data"]; it = ws.iter_rows(values_only=True)
    hdr = list(next(it)); ix = {h: i for i, h in enumerate(hdr)}
    pick = lambda *c: next((ix[x] for x in c if x in ix), None)
    i_t, i_d = pick("UTC time"), pick("Adjusted demand", "Demand")
    i_d2 = pick("Demand")
    i_s, i_sb, i_w, i_wb = pick("NG: SUN"), pick("NG: SNB"), pick("NG: WND"), pick("NG: WNB")
    rows = []
    for r in it:
        if r[i_t] is None: continue
        g = lambda i: (r[i] if (i is not None and isinstance(r[i], (int, float))) else np.nan)
        d = g(i_d); d = g(i_d2) if np.isnan(d) else d
        s = np.nansum([g(i_s), g(i_sb)]) if not (np.isnan(g(i_s)) and np.isnan(g(i_sb))) else np.nan
        w = np.nansum([g(i_w), g(i_wb)]) if not (np.isnan(g(i_w)) and np.isnan(g(i_wb))) else np.nan
        rows.append((r[i_t], d, s, w))
    df = pd.DataFrame(rows, columns=["date", "load", "solar", "wind"])
    df["date"] = pd.to_datetime(df["date"]); df = df[df["date"] >= START].drop_duplicates("date").sort_values("date")
    full = pd.date_range(df["date"].min(), df["date"].max(), freq="h")
    df = df.set_index("date").reindex(full); df.index.name = "date"
    miss = df.isna().mean().round(4).to_dict()
    df = df.interpolate(limit=6, limit_area="inside")           # fill gaps of up to 6 h only
    left = int(df.isna().any(axis=1).sum())
    df.to_csv(os.path.join(out, f"eia_{name}.csv"))
    report[name] = dict(rows=len(df), start=str(df.index.min()), end=str(df.index.max()), missing_before_fill=miss,
                        rows_still_missing=left, solar_zero_share=round(float((df.solar.fillna(0) <= 0).mean()), 3),
                        mean=df.mean().round(1).to_dict(), max=df.max().round(1).to_dict(), min=df.min().round(1).to_dict())
    print(name, report[name], flush=True)
json.dump(report, open(os.path.join(out, "eia_report.json"), "w"), indent=1)
