"""Grid-India (all-India SCADA) monthly xlsx -> hourly CSV [date(IST), load, solar, wind].
2021-09..2023-12: SCADA at 5-min or 10-s with tag columns -> hourly mean. 2024-01..2025-06: hourly sheet 'Report'.
Usage: python prep_grid_india.py <datasets_dir> <out_dir>"""
import sys, os, re, json, glob, zipfile, tempfile, openpyxl, pandas as pd, numpy as np
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
zf = glob.glob(os.path.join(src, "grid_india", "*.zip"))[0]
tmp = tempfile.mkdtemp(); zipfile.ZipFile(zf).extractall(tmp)
files = glob.glob(os.path.join(tmp, "*", "*.xlsx"))
MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"], 1)}
def key(f):
    b = os.path.basename(f).lower()
    y = int(re.search(r"(20\d\d)", b).group(1)); m = next((v for k, v in MONTHS.items() if b.startswith(k[:3])), 0)
    return (y, m)
parts = []
cache = os.path.join(out, '_gi_cache'); os.makedirs(cache, exist_ok=True)
for f in sorted(files, key=key):
    b = os.path.basename(f)
    cf = os.path.join(cache, b + '.csv')
    if os.path.exists(cf):
        h = pd.read_csv(cf, parse_dates=['date']).set_index('date'); parts.append(h); print('cached', b, flush=True); continue
    wb = openpyxl.load_workbook(f, read_only=True, data_only=True)
    if "Report" in wb.sheetnames:                                   # hourly 2024-25 sheet
        ws = wb["Report"]; it = ws.iter_rows(values_only=True); hdr = next(it)
        d = pd.DataFrame(list(it), columns=hdr).dropna(subset=["Timestamp"])
        d["date"] = pd.to_datetime(d["Timestamp"], dayfirst=True, errors="coerce")
        d = d.rename(columns={"Demand (MW)": "load", "Wind (MW)": "wind", "Solar (MW)": "solar"})[["date", "load", "solar", "wind"]]
    else:
        ws = wb["Sheet1"]; it = ws.iter_rows(values_only=True); h1 = next(it); h2 = next(it)
        find = lambda name: next(i for i, v in enumerate(h2) if isinstance(v, str) and v.strip().upper() == name)
        iw, isol, idm = find("ALL_INDIA_WIND|P"), find("ALL_IND_SOLAR|P"), find("NLDC_DEMAND|P")
        num = lambda x: x if isinstance(x, (int, float)) else np.nan
        rows = [(r[0], num(r[idm]), num(r[isol]), num(r[iw])) for r in it if r[0] is not None]
        d = pd.DataFrame(rows, columns=["date", "load", "solar", "wind"]); d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"]).set_index("date")
    for c in ["load", "solar", "wind"]: d[c] = pd.to_numeric(d[c], errors="coerce")
    h = d.resample("h").mean(); h.to_csv(cf); parts.append(h); print(b, len(d), "->", len(h), "hourly", flush=True)
df = pd.concat(parts).sort_index(); df = df[~df.index.duplicated(keep="first")]
df["solar"] = df["solar"].clip(lower=0)
full = pd.date_range(df.index.min(), df.index.max(), freq="h"); df = df.reindex(full); df.index.name = "date"
miss = df.isna().mean().round(4).to_dict(); df = df.interpolate(limit=6, limit_area="inside"); left = int(df.isna().any(axis=1).sum())
df.to_csv(os.path.join(out, "grid_india.csv"))
rep = dict(rows=len(df), start=str(df.index.min()), end=str(df.index.max()), missing_before_fill=miss, rows_still_missing=left,
           mean=df.mean().round(1).to_dict(), max=df.max().round(1).to_dict(), min=df.min().round(1).to_dict(),
           note="timestamps as given by Grid-India (IST assumed)")
json.dump(rep, open(os.path.join(out, "grid_india_report.json"), "w"), indent=1); print(rep)
