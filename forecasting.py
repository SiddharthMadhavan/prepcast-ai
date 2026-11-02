"""Reproducible calendar/lag models and chronological, recursive holdout scoring.

No weather or restaurant outcome data is inferred. The final holdout block is
never used for fitting, feature construction or model selection.
"""
import math
import statistics
from datetime import date

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODELS = {
    'ensemble': 'Weekday + recent mean ensemble',
    'ridge': 'Ridge Regression',
    'gradient_boosting': 'Gradient Boosting',
}
MIN_HISTORY = 28
HOLDOUT = 14


def ensemble(history, target):
    matching = [r['sales'] for r in history
                if date.fromisoformat(r['date']).weekday() == target.weekday()][-6:]
    recent = [r['sales'] for r in history[-14:]]
    return .75 * statistics.mean(matching or recent) + .25 * statistics.mean(recent)


def features(history, target, origin):
    """Only prior observations. Lags are observation lags, not imputed dates."""
    values = [r['sales'] for r in history[-14:]]
    weekday = target.weekday()
    matching = [r['sales'] for r in history
                if date.fromisoformat(r['date']).weekday() == weekday][-6:]
    return ([float(weekday == i) for i in range(7)] +
            [(target - origin).days, values[-1], values[-7], values[-14],
             statistics.mean(values[-7:]), statistics.mean(values[-14:]),
             statistics.mean(matching or values[-14:])])


def predictor(history, model):
    if model not in MODELS:
        raise ValueError('Model must be ensemble, ridge or gradient_boosting.')
    if model == 'ensemble' or len(history) < MIN_HISTORY:
        return ensemble, 'ensemble'
    # Bound per-request fitting work even for the maximum permitted CSV size.
    history = history[-366:]
    origin = date.fromisoformat(history[0]['date'])
    x = [features(history[:i], date.fromisoformat(history[i]['date']), origin)
         for i in range(14, len(history))]
    y = [r['sales'] for r in history[14:]]
    estimator = (make_pipeline(StandardScaler(), Ridge(alpha=10.0))
                 if model == 'ridge' else
                 GradientBoostingRegressor(n_estimators=40, max_depth=2,
                                           min_samples_leaf=3, random_state=42))
    estimator.fit(x, y)

    def predict(prior, target):
        return float(estimator.predict([features(prior, target, origin)])[0])
    return predict, model


def forecast_path(history, targets, model):
    """Fit once and recursively append predictions, never holdout actuals."""
    history = sorted(history, key=lambda r: r['date'])
    predict, used = predictor(history, model)
    prior = [dict(r) for r in history]
    result = []
    for target in targets:
        value = predict(prior, target)
        if not math.isfinite(value):
            raise ValueError('Model produced a non-finite forecast.')
        value = min(1e7, max(0.0, value))
        result.append(value)
        prior.append(dict(date=target.isoformat(), sales=value))
    return result, used


def evaluate_dish(history, model):
    history = sorted(history, key=lambda r: r['date'])
    if len(history) < MIN_HISTORY + HOLDOUT:
        return [], [], None
    train, test = history[:-HOLDOUT], history[-HOLDOUT:]
    targets = [date.fromisoformat(r['date']) for r in test]
    predictions, _ = forecast_path(train, targets, model)
    # Freeze this simple baseline at the forecast origin for the entire block.
    baseline = statistics.mean(r['sales'] for r in train[-7:])
    residuals = [r['sales'] - p for r, p in zip(test, predictions)]
    baseline_errors = [abs(r['sales'] - baseline) for r in test]
    window = dict(trainEnd=train[-1]['date'], testStart=test[0]['date'],
                  testEnd=test[-1]['date'], trainRows=len(train), testRows=len(test))
    return residuals, baseline_errors, window


def model_evaluation(rows):
    dishes = sorted({r['dish'] for r in rows})
    models, windows = [], []
    for model, name in MODELS.items():
        errors, baseline_errors = [], []
        for dish in dishes:
            h = sorted([r for r in rows if r['dish'] == dish], key=lambda r: r['date'])
            residual, baseline, window = evaluate_dish(h, model)
            errors.extend(abs(r) for r in residual)
            baseline_errors.extend(baseline)
            if model == 'ensemble' and window:
                windows.append(dict(dish=dish, **window))
        mae = statistics.mean(errors) if errors else None
        base = statistics.mean(baseline_errors) if baseline_errors else None
        models.append(dict(id=model, name=name, mae=round(mae, 4) if mae is not None else None,
                           baselineMae=round(base, 4) if base is not None else None,
                           improvementPct=round((base-mae)/base*100, 2) if base else None,
                           validationPoints=len(errors),
                           status='evaluated' if errors else 'insufficient_history'))
    return dict(models=models, windows=windows, rows=len(rows), dishes=len(dishes),
                observedDays=len({r['date'] for r in rows}),
                method='Final 14 observations per dish; models fit only on earlier observations; recursive predictions',
                baseline='Mean of final 7 training observations, frozen across the holdout',
                minimumObservationsPerDish=MIN_HISTORY+HOLDOUT,
                maximumMLTrainingObservations=366,
                limitations=['Calendar and lag features only; weather/events not connected.',
                             'Sales can understate demand during stockouts.',
                             'No measured waste reduction, restaurant validation or formal TRL assessment.'])
