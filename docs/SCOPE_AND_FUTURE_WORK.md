# Scope and future work

This page states what the study covers and where it can be extended, so readers can interpret the results correctly.

## What the study covers
* 11 grids (4 U.S. balancing authorities and 7 European countries), 24-hour-ahead hourly forecasts of load, solar and wind, lookback 168 h, point forecasts.
* Plain protocol (models predict values directly): GateXer has the lowest error on all 11 grids.
* Corrected protocol (every model predicts a correction on yesterday's values): GateXer and TimeXer are tied (differences 0.1% to 1.4%, intervals include zero) on the five grids tested, and GateXer stays ahead of iTransformer and PatchTST on four of five.
* All-India stress case: the solar level there tripled over the data and a seasonal-naive rule beat all trained models on solar. GateXer shows no advantage there (-0.1%, interval [-4.0, 3.5]), so India is reported separately and is not part of the 11-grid average.

## Interpretation notes
* **Weather inputs are observed values** for the target hours, the usual setting in this literature, applied identically to every model. Operational use would feed weather forecasts instead.
* **Attribution of gains:** the clean sharing ablation is on ERCOT, CAISO and Germany (control = same model without sharing). On SWPP, MISO and Portugal the control (`ci`) also lacks FAMO and the trend split.
* **Selection:** lookback and width were chosen on validation data of two grids (Germany and ERCOT) with the same grid for every model, then used unchanged on the other grids.
* Wind gains are smaller than solar and load gains (mean 2.3% against 8.6% and 6.2%).

## Future work
1. Replace observed weather by archived weather forecasts.
2. Probabilistic forecasts (prediction intervals) and longer horizons.
3. Evaluate directly on net load (load minus solar minus wind).
4. Stronger handling of distribution shift (growing solar capacity) and more shifted datasets.
5. Broader hyper-parameter search for all models, and comparison with pretrained time-series foundation models.
