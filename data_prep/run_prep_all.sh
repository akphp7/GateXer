#!/usr/bin/env bash
# Sequential, resumable data preparation.
# Expects the raw files under $DATA_DIR (default: ./data), laid out as described in docs/DATA.md:
#   data/opsd/  data/eia930/  data/grid_india/
# Writes cleaned hourly CSVs to $DATA_DIR/final (one file per grid) and a quality_report.json.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
D="${DATA_DIR:-$ROOT/data}"
P="$ROOT/data_prep"
mkdir -p "$D/prepared" "$D/final"
python "$P/prep_opsd.py"       "$D" "$D/prepared"
python "$P/prep_eia.py"        "$D" "$D/prepared"
python "$P/prep_grid_india.py" "$D" "$D/prepared"
python "$P/finalize.py"        "$D/prepared" "$D/final"
echo "done: cleaned files are in $D/final"
