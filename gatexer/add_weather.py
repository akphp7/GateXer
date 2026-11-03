"""Download hourly ERA5-based weather (Open-Meteo archive API, free, no key) for representative points of each grid, average the points,
and merge into the final CSVs -> writes data/final_w/<name>.csv with extra columns temp, ws100, ghi, cloud.
Only EIA and Grid-India files are processed (OPSD already has its own weather).
Usage (needs internet):
    pip install requests pandas
    python add_weather.py --final ../data/final --out ../data/final_w --cache ../data/weather
Re-running reuses the cache; use --allow_partial to continue when a point cannot be downloaded."""
import argparse, os, time, json, requests, pandas as pd
POINTS = {  # representative load/solar/wind areas (equal weights; an approximation, stated in the paper)
    "eia_ERCO": [(29.76, -95.37), (32.78, -96.80), (29.42, -98.49), (33.58, -101.86), (31.99, -102.08)],
    "eia_CISO": [(34.05, -118.24), (37.77, -122.42), (36.74, -119.79), (35.37, -119.02), (34.90, -117.02)],
    "eia_SWPP": [(35.47, -97.52), (37.69, -97.34), (35.22, -101.83), (41.26, -95.93), (39.10, -94.58)],
    "eia_MISO": [(41.88, -87.63), (44.98, -93.27), (41.59, -93.62), (39.77, -86.16), (42.33, -83.05), (29.95, -90.07)],
    "grid_india": [(26.24, 73.02), (23.25, 69.67), (11.02, 76.96), (12.97, 77.59), (28.61, 77.21), (19.08, 72.88), (13.08, 80.27), (22.57, 88.36), (14.68, 77.60)],
}
TZ = {"grid_india": "Asia/Kolkata"}          # Grid-India timestamps are IST; EIA 'UTC time' is UTC
VARS = {"temperature_2m": "temp", "wind_speed_100m": "ws100", "shortwave_radiation": "ghi", "cloud_cover": "cloud"}
def fetch(lat, lon, start, end, tz, retries=4):
    p = dict(latitude=lat, longitude=lon, start_date=start, end_date=end, hourly=",".join(VARS), timezone=tz, wind_speed_unit="ms")
    reason = ""
    for k in range(retries):
        try:
            r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=p, timeout=180)
            if r.status_code == 200 and r.text.strip():
                j = r.json()
                if "hourly" in j:
                    h = j["hourly"]
                    return pd.DataFrame({"date": pd.to_datetime(h["time"]), **{VARS[v]: h[v] for v in VARS}}).set_index("date")
                reason = str(j.get("reason", j))[:120]
            else:
                reason = f"status {r.status_code}, body {r.text[:80]!r}"
        except Exception as e:                                  # empty body, timeout, connection reset ...
            reason = repr(e)[:120]
        wait = 30 * (k + 1)
        print(f"  retry {k + 1}/{retries} in {wait}s ({reason})", flush=True); time.sleep(wait)
    raise RuntimeError(f"weather download failed for {lat},{lon}: {reason}")
ap = argparse.ArgumentParser(); ap.add_argument("--final", required=True); ap.add_argument("--out", required=True); ap.add_argument("--cache", required=True)
ap.add_argument("--allow_partial", action="store_true", help="if some points cannot be downloaded, use the points already cached (needs >= --min_points)")
ap.add_argument("--min_points", type=int, default=4)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); os.makedirs(a.cache, exist_ok=True)
for name, pts in POINTS.items():
    f = os.path.join(a.final, name + ".csv")
    if not os.path.exists(f): continue
    df = pd.read_csv(f, parse_dates=["date"]).set_index("date")
    start = (df.index.min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d"); end = min(df.index.max(), pd.Timestamp.today() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    parts = []
    for i, (la, lo) in enumerate(pts):
        c = os.path.join(a.cache, f"{name}_{i}.csv")
        if os.path.exists(c): parts.append(pd.read_csv(c, parse_dates=["date"]).set_index("date")); continue
        try:
            w = fetch(la, lo, start, end, TZ.get(name, "UTC")); w.to_csv(c); parts.append(w); print(name, i, la, lo, len(w), flush=True); time.sleep(8)
        except RuntimeError as e:
            if not a.allow_partial: raise
            print("  skipping point", la, lo, "->", str(e)[:100], flush=True)
    if len(parts) < (a.min_points if a.allow_partial else len(pts)): print('not enough weather points for', name, len(parts)); continue
    print(name, 'uses', len(parts), 'of', len(pts), 'weather points', flush=True)
    wm = sum(parts) / len(parts)
    out = df.join(wm, how="inner").dropna()
    out.to_csv(os.path.join(a.out, name + ".csv")); print("wrote", name, len(out), "of", len(df), "rows", out.index.min(), out.index.max())
