# Limitations

We list these first so that no reader has to find them.

1. **Corrected protocol is a tie.** The headline win (lowest error on all 11 grids) is under the plain protocol. When every model also uses the yesterday-correction, GateXer and TimeXer are statistically tied on the five grids tested (differences 0.1 to 1.4%, intervals include zero). GateXer stays ahead of iTransformer and PatchTST on four of the five.
2. **Observed weather, not forecast weather.** Weather inputs are historical observations for the target hours. Real operation uses forecasts with errors, so absolute accuracy here is optimistic for deployment. All models receive the same inputs. Using archived forecasts is future work.
3. **Wind gains are small** (mean 2.3% against 8.6% for solar and 6.2% for load) and flat or slightly negative on France and Austria.
4. **All-India stress case:** no advantage (-0.1%, interval [-4.0, 3.5]). The solar level there tripled over the data, and a seasonal-naive rule beat the trained models on solar in the plain protocol. Strong distribution shift is not solved by this model.
5. **Trend split is neutral** in the ablation; it is kept but not claimed as a contribution.
6. **Ablation control differs by grid.** For SWPP, MISO and Portugal the no-sharing control (`ci`) also lacks FAMO and the trend split, so the attribution to sharing alone is clean only for ERCOT, CAISO and Germany.
7. **Scope.** Point forecasts only, one horizon (24 h), lookback 168 h, 11 grids. Pretrained time-series foundation models are not part of the comparison (a Chronos-2 stage exists in the code as an exploratory option and is not reported).
8. **Hyperparameters were tuned on two grids** (Germany and ERCOT) and applied to the rest.
