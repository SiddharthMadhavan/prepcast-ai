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

Transparent statistical ensemble, not a trained neural model: 75% mean of last six matching weekdays plus 25% mean of last fourteen observations. Rolling holdout evaluation uses up to fourteen observations per dish and only earlier history. Compare MAE against a trailing-seven-observation mean. Approximate 90% residual intervals are not calibrated; widen with horizon. Suggested prep adds half a residual standard deviation. No weather, holiday, event, or censored-demand correction yet. Real sales during stockouts underestimate true demand. This synthetic-data demo does not prove real-world accuracy or savings.

Prep edits persist across views until recalculation; export CSV or print the plan. Illustrative recipes map the eight demo dishes to ingredients. Unmapped imported dishes are explicitly listed. On-hand stock is editable and stored in browser localStorage; recipe values and default stock are synthetic. Milk is litres, other ingredients kilograms.

## Privacy and persistence

Public demo, no login or database. Uploaded records stay in browser memory and are sent to the server for request-scoped computation only; no upload files or records are persisted server-side. Refresh resets to synthetic data. Do not use this deployment as a production data store. Authentication, durable records, recipe management, and pilot validation are future work.

## Checks

```sh
pip install pytest
pytest -q
```

Render: Python web service, build `pip install -r requirements.txt`, start `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2`, free plan, health endpoint `/health`. No API keys required.
