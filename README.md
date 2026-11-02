# PrepCast AI

A working restaurant demand forecasting demo, based on the attached Stitch Precision Culinary Telemetry designs. Dark, rectilinear UI with orange operational accents; overview, CSV ingestion, forecasting, inventory, and model evaluation screens.

## Run

Python 3.11+:

```sh
pip install -r requirements.txt
python app.py
```

Open http://localhost:8080. Production: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2`.

## Demo data

`data/synthetic_sales.csv` contains 112 days × 8 dishes (896 rows), generated deterministically with seed 42. Includes weekday/weekend patterns, mild trend, random demand noise, over-preparation, and capacity-limited sales. All data is synthetic. Download the same CSV in the upload screen or `/api/demo.csv`. Demo forecast dates follow the final observation, rather than silently assuming current-day data.

CSV schema: `date,dish,sales,prepared,price`. Required: ISO date, dish, non-negative sales. Optional: prepared (defaults to sales), INR price (defaults to zero). One row per dish/date, minimum 14 distinct dates, max 100 dishes, 20,000 rows, 2 MB. Include zero-sales days; missing dates are not filled. Missing prepared values mean no recorded waste, not proven zero waste.

## Forecast methodology

Ridge Regression is the default operational model. The forecast screen also offers Gradient Boosting and the original weekday/recent-mean ensemble. ML fitting uses at most the latest 366 observations per dish to bound request cost. Models use weekday indicators, elapsed days and lag/rolling sales features. StandardScaler is fit inside the Ridge training pipeline. Gradient Boosting uses a fixed seed (42), 40 shallow trees and no holdout tuning. Histories below 28 observations fall back explicitly to the statistical ensemble.

Both ML models are implemented with scikit-learn. For the next 1/7/30 days, each model is fit on the available history and forecasts recursively using its own prior predictions. Lag values refer to observed rows, not calendar-day imputation. Missing dates remain absent.

Model Metrics compares all three models on the same final 14 observations per eligible dish. At least 28 earlier training observations are required (42 total). Each model fits once on the training prefix, then predicts the whole holdout recursively without seeing its actual sales. A trailing-seven-training-observation mean is frozen across that same block as the baseline. The UI reports calculated MAE, signed relative improvement, prediction counts and a downloadable JSON report including per-dish train/test dates. Negative improvement means worse MAE. The holdout is not used to tune or choose the model. This block evaluation differs from the earlier rolling one-step scores, so those numbers are not directly comparable.

API: `POST /api/forecast` accepts `model=ridge|gradient_boosting|ensemble`, default `ridge`. `POST /api/model-evaluation` accepts the same optional CSV file and returns reproducible comparisons. The separate what-if sandbox retains its original statistical ensemble. It does not change the operational model or source records.

Approximate 90% residual intervals are not calibrated and widen with horizon. Suggested prep adds half a residual standard deviation. No live weather, holiday, event, or censored-demand correction is connected. Real sales during stockouts underestimate true demand. Synthetic-data evaluations do not prove restaurant accuracy, cost savings, waste reduction or a formal TRL. The 112-day / 896-row demo remains labelled synthetic; no six-month restaurant dataset or field pilot is fabricated.

Prep edits persist across views until recalculation; export CSV or print the plan. Illustrative recipes map the eight demo dishes to ingredients. Unmapped imported dishes are explicitly listed. On-hand stock is editable and stored in browser localStorage; recipe values and default stock are synthetic. Milk is litres, other ingredients kilograms.

## Privacy and persistence

Public demo, no login or database. Uploaded records stay in browser memory and are sent to the server for request-scoped computation only; no upload files or records are persisted server-side. Refresh resets to synthetic data. Do not use this deployment as a production data store. Authentication, durable records, recipe management, and pilot validation are future work.

## Checks

```sh
pip install pytest
pytest -q
```

Render: Python web service, build `pip install -r requirements.txt`, start `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2`, free plan, health endpoint `/health`. No API keys required.

## Interactive what-if workspace

The demand forecast screen includes a sandbox with three selectable synthetic restaurant histories: Neighborhood Kitchen, Office District, and Weekend Bistro. Multiple sources can be overlaid; they are forecast independently and never silently pooled. Upload additional CSVs (daily totals across dishes), up to six sources and 366 dates per source. Bundled sources use the most recent 28 days; uploaded sources use all their observations for calculation and show the last 28 on the graph.

Select an editing source, drag its historical square handles vertically, use arrow keys on a focused handle, or type exact daily sales in the historical table. Recalculation is debounced; outdated responses are discarded. Compare original and edited forecasts and total portion changes for 1/7/30 days. Reset restores the active source. Original CSV files remain unchanged. Changes live only in the current tab. This workspace is separate from the operational run-list. Forecast ranges are approximate, not calibrated. Source differences and user edits are illustrative associations, not causal effects.

API: `GET /api/lab/sources`, `POST /api/lab/upload` (multipart CSV), `POST /api/lab/scenario` (JSON rows, date-keyed edits, horizon). Inputs reject duplicates, non-finite/negative values, unknown edit dates, and unsupported horizons.

## Restaurant analytics

Analytics uses the operational demo or the currently uploaded sales CSV. Includes observed daily sales/revenue/waste trends, weekday averages and observation counts, dish rankings by sales or revenue, waste contributions, dish mix, coefficient of variation, zero-sales rows, and sold-all-prep review signals. Recent changes compare per-observed-day averages in the last 14 calendar days with the prior 14, anchored to the dataset's end date. Missing dates are not filled; prices and prepared quantities may be incomplete. Revenue is not profit. Sold-all-prep is not proof of a stockout. Scenario sandbox changes do not alter operational analytics.


## Reproduce evaluation

```sh
python evaluate_models.py > evaluation.json
python evaluate_models.py path/to/sales.csv > uploaded-evaluation.json
```

The JSON identifies the data source, eligible per-dish train/test dates, baseline definition, MAE and signed improvement. Uploaded-file provenance is not verified. Do not transfer synthetic results into restaurant-validation claims.
