# Reproducing the experiments

Everything below runs from the repository root. A GPU is strongly recommended for the real runs (we used a free Kaggle T4; one training run takes about 100 s). On CPU a full-size run takes hours.

## 0. Install and test

```bash
pip install -r requirements.txt pytest
pytest -q tests
```

## 1. Data

Download the raw files as described in [`DATA.md`](DATA.md), then:

```bash
bash data_prep/run_prep_all.sh                    # writes data/final/*.csv
python gatexer/add_weather.py --final data/final --out data/final_w --cache data/weather   # U.S. grids + India (internet needed, rate-limited; re-run to resume)
```

## 2. Quick check (CPU, minutes)

```bash
python gatexer/run_experiments.py --stage main --data_dir data/final --out runs_smoke \
    --datasets opsd_DE --presets gx,patchtst --seeds 0 --epochs 2 --max_train_windows 2000
python gatexer/aggregate.py runs_smoke
```

## 3. Full pipeline

Runs are resumable: finished runs are skipped, so you can restart after a session reset.

```bash
W="--data_dir data/final --weather_dir data/final_w --bs 128"

# tuning on validation (Germany + ERCOT) and choice of lookback/width
python gatexer/run_experiments.py --stage tune $W --out runs
python gatexer/select_best.py runs

# main comparison, 3 seeds, all grids
python gatexer/run_experiments.py --stage main $W --out runs --seeds 0 1 2 \
    --datasets opsd_DE,opsd_ES,eia_ERCO,eia_CISO,opsd_FR,opsd_DK,opsd_PT,opsd_AT,opsd_GR,eia_SWPP,eia_MISO,grid_india

# yesterday-correction protocol (re-use the tuned widths)
mkdir -p runs/snb/tune && cp runs/tune/best.json runs/snb/tune/
python gatexer/run_experiments.py --stage main $W --out runs/snb --snb 1 --seeds 0 1 2 \
    --datasets eia_ERCO,eia_CISO,opsd_DE,opsd_ES,grid_india --presets gx,ci,patchtst,itransformer,timexer

# ablations: no sharing, always-on, equal weights, no trend split, sources-only, weather-only
mkdir -p runs/ablate4/tune && cp runs/tune/best.json runs/ablate4/tune/
python gatexer/run_experiments.py --stage ablate $W --out runs/ablate4 --seeds 0 1 2 \
    --datasets eia_ERCO,eia_CISO,opsd_DE --presets gx_nosh,gx_always,gx_gateonly,gx_nodec,gx_srconly,gx_exonly
```

Each run writes `runs/<stage>/<grid>/<model>_s<seed>.json` (metrics, gate values, parameter count, seconds) and an `.npz` with test predictions. `--dry` prints the commands without running them.

## 4. Tables, statistics and figures

```bash
python analysis/agg.py runs                       # collects all run JSONs into results/results_all.csv
GATEXER_RUNS=runs python analysis/boot.py eia_ERCO,eia_CISO,opsd_DE 0     # paired bootstrap, plain protocol
GATEXER_RUNS=runs python analysis/boot.py eia_ERCO,eia_CISO,opsd_DE 1     # corrected protocol
python analysis/make_tables.py                    # Markdown tables from the bootstrap files
cd figures/src && python resfigs.py               # results figures into ./res
```

The shipped `results/` folder already contains the per-run table and bootstrap outputs used in the paper, so `make_tables.py` and the figure scripts run without re-training. The bootstrap scripts need the per-run prediction files (`.npz`), which are large and not included.

## Notes

* Different GPU types and library versions give slightly different numbers; the seed fixes initialisation but not all GPU non-determinism.
* The data files change over time (EIA appends new hours); we cropped to the ranges stated in `DATA.md`, but the end date of the EIA series depends on when you download.
