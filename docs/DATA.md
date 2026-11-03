# Data

Raw datasets are **not** included in this repository (size, and third-party licences). All of them are public. Put the raw files as shown and run `data_prep/run_prep_all.sh`.

| Source | What | Used for | Licence / terms |
|---|---|---|---|
| Open Power System Data (OPSD), `time_series` 2020-10-06 (60 min) | hourly load, solar and wind for European countries | DE, ES, FR, DK, PT, AT, GR | CC BY 4.0 |
| OPSD `weather_data` 2020-09-16 | MERRA-2 temperature, direct and diffuse radiation | weather for the European grids | CC BY 4.0 |
| U.S. EIA-930 hourly grid monitor (Excel downloads per balancing authority) | hourly demand, solar and wind generation | ERCOT, CAISO, SWPP, MISO | U.S. government data, public |
| Grid-India (Grid Controller of India) monthly all-India demand, solar and wind files | hourly series | India stress case | check the publisher's terms |
| Open-Meteo historical weather API (ERA5-based) | temperature, wind speed at 100 m, shortwave radiation (GHI), cloud cover | weather for the U.S. grids and India | CC BY 4.0, attribution required |

Links: <https://data.open-power-system-data.org/>, <https://www.eia.gov/electricity/gridmonitor/>, <https://open-meteo.com/en/docs/historical-weather-api>.

## Expected folder layout

```
data/
  opsd/        opsd-time_series-2020-10-06.zip   opsd-weather_data-2020-09-16.zip
  eia930/      ERCO.xlsx  CISO.xlsx  SWPP.xlsx  MISO.xlsx   (hourly "Published Hourly Data" sheet)
  grid_india/  <zip with the monthly .xlsx files>
```

## Processing

`data_prep/run_prep_all.sh` converts everything to one hourly CSV per grid (`date, load, solar, wind`, plus weather for OPSD), applies physical limits (negative solar/wind set to zero), removes load outliers (outside 0.3x to 2.5x the median), fills gaps, and writes `data/final/quality_report.json` with the number of fixes per grid.

## Weather

* **European grids (OPSD):** the weather columns supplied with the OPSD package (temperature, direct and diffuse radiation). OPSD weather ends in December 2019, so the OPSD series are 2015 to 2019.
* **U.S. grids and India:** `gatexer/add_weather.py` downloads Open-Meteo hourly values and **averages 5 to 6 points per grid (9 for India)**, chosen in representative load, solar and wind areas. The exact coordinates are in [`weather_locations.txt`](weather_locations.txt). Timestamps: EIA in UTC, Grid-India in IST.
* **Observed, not forecast:** the weather values are historical observations for the same hours as the target. In real operation only forecasts are available, which carry errors. This is a standard simplification, applied identically to all models; see `LIMITATIONS.md`.

## Time ranges

EIA series are cropped to start 2021-07-01 because solar and wind capacity grew by large factors since 2018. Grid-India covers 2021-09 to 2025-06.
