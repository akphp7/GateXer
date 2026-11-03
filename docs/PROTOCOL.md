# Experimental protocol

The protocol is designed so that a gain cannot come from a leaked test set or from handicapped baselines.

| Item | Setting |
|---|---|
| Task | 24 h ahead, hourly; lookback 168 h |
| Split | chronological 70 / 10 / 20 (train / validation / test) |
| Scaling | mean and std from training data only |
| Metric | nMAE = MAE / mean actual, per source, then averaged |
| Model selection | on validation only; test read once per run |
| Tuning grid (same for all trained models) | lookback {168, 336} x width {64, 128}, chosen on validation of Germany and ERCOT (40 runs) |
| Chosen | lookback 168; width 128 for GateXer, the no-sharing control and TimeXer; 64 for PatchTST and iTransformer |
| Seeds | 0, 1, 2; results reported as single-seed means and 3-seed ensembles |
| Statistics | paired bootstrap over test windows, 95% intervals, averaged over the 24 phases of the day to respect autocorrelation |

## Baselines and their inputs

All baselines use the official Time-Series-Library code (vendored in `gatexer/tslib_vendor`), with the inputs they were designed for (checked in the library code).

| Model | Inputs |
|---|---|
| PatchTST | targets + weather, no calendar |
| iTransformer | channels + time marks |
| TimeXer | one model per target; other sources + weather + time marks as exogenous variables |
| DLinear | targets only |
| Seasonal naive | none (value 24 h earlier) |
| `ci` (own control) | GateXer with CrossMix switched off: no sharing with other sources and no weather |

## Two protocols

* **Plain:** every model predicts the 24 h values directly (main comparison).
* **Corrected (`--snb 1`):** models predict a correction on yesterday's values (see `METHOD.md`). Only corrected models are compared with each other.

## Ablations

* `gx_nosh` / `ci`: no sharing. `gx_srconly`: gated sharing between sources only. `gx_exonly`: gated weather mixing only. `gx_always`: mixing always on. `gx_gateonly`: gated, equal loss weights (no FAMO). `gx_nodec`: no trend split.
* For ERCOT, CAISO and Germany the no-sharing control is `gx_nosh` (same recipe). For SWPP, MISO and Portugal it is `ci`, which also lacks FAMO and the trend split, so those rows mix several differences.

## Scope of the study

Eleven grids (4 U.S. balancing authorities, 7 European countries) plus one stress case (all-India). About 557 training runs in total across tuning, main comparison, corrected protocol and ablations.
