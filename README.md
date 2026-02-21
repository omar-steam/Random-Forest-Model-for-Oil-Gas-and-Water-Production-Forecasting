# Random-Forest-Model-for-Oil-Gas-and-Water-Production-Forecasting
A machine learning pipeline that uses a Random Forest Regressor to predict and analyze oil, gas, and water production rates from well test data.

It also includes SHAP-based explainability to identify the key drivers behind production trends.

---

## Features

- **Automatic column detection** — pattern-matches column names across varying CSV formats
- **Multi-target prediction** — simultaneously forecasts liquid rate (BLPD), oil (BOPD), gas (MCF), water (BWPD), and GOR
- **Lag feature engineering** — creates time-lagged inputs for time series modelling
- **SHAP explainability** — identifies the top factors contributing to production changes
- **Decline analysis** — highlights the most impactful features over a recent window
- **Optimization suggestions** — maps important features to actionable operational recommendations
- **Actual vs. predicted visualization** — time series plots comparing model output to real production data

---

## Requirements

Install dependencies with:

```bash
pip install pandas numpy matplotlib scikit-learn shap
```

---

## Input Data

The script expects three CSV files in the working directory:

| File | Description |
|------|-------------|
| `Platform MC_Low Pressure System Monitoring clean.csv` | Platform monitoring data |
| `cleaned_data (1).csv` | General cleaned production data |
| `Well_Test_Cleaned (1).csv` | Well test data (primary modelling input) |

The well test file is the primary source for model training and must contain columns mappable to: `well_string`, `liquid`, `oil`, `gas`, `water`, and/or `gor`. The column detection logic uses fuzzy keyword matching, so exact column names are not required.

---

## How It Works

1. **Load & clean data** — reads the well test CSV, detects and renames columns, fills missing targets with 0, and assigns a synthetic daily date index starting from `2000-01-01`.

2. **Feature engineering** — renames targets to standardized names and creates one lag feature per target (previous day's value).

3. **Train/test split** — uses a 70/30 chronological split.

4. **Model training** — fits a `RandomForestRegressor` (100 trees, max depth 5) on lag features to predict all targets simultaneously.

5. **Evaluation** — reports Mean Absolute Error (MAE) and plots the top 10 most important features.

6. **SHAP analysis** — uses `TreeExplainer` to compute feature attributions and summarizes the top contributors for the liquid rate target.

7. **Decline analysis** — averages SHAP magnitudes over the most recent 30 days to identify which features are currently driving production changes.

8. **Optimization suggestions** — maps top features to plain-language operational actions.

9. **Visualization** — generates a 3-panel time series chart of actual vs. predicted oil, gas, and water production over the test period.

---

## Usage

```bash
python randomforestmodelforoilgasandwater.py
```

Ensure all three CSV files are in the same directory as the script before running.

---

## Output

- Printed column mappings and data shape summaries
- Model MAE score
- Feature importance bar chart
- SHAP summary plot
- Top decline factors (last 30 days)
- Optimization suggestions
- Actual vs. predicted production time series chart (oil, gas, water)

---

## Configuration

Key parameters to adjust in the script:

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `n_estimators` | `RandomForestRegressor` | `100` | Number of trees |
| `max_depth` | `RandomForestRegressor` | `5` | Maximum tree depth |
| `split` | Train/test split | `0.7` | Fraction of data used for training |
| `days` | `identify_decline_reasons` | `30` | Lookback window for decline analysis |
| `start` | `pd.date_range` | `2000-01-01` | Synthetic start date for date index |

---

## Notes

- The date index is synthetic and generated sequentially from `2000-01-01`. If your data includes real timestamps, the loading function can be modified to parse them directly.
- The `well_string` column (well identifier) is excluded from model features as it is non-numeric.
- If fewer than 10 samples are found after cleaning, the script raises a `ValueError`.
- SHAP values are computed for all targets but only the first target (`liquid_blpd`) is plotted by default.
