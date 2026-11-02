import io
from datetime import date, timedelta

import pytest

from app import app, DEMO, analyze
from forecasting import MODELS, evaluate_dish, forecast_path, model_evaluation


def test_models_are_reproducible_on_the_same_holdout():
    first = model_evaluation(DEMO)
    assert first == model_evaluation(DEMO)
    assert {m['id'] for m in first['models']} == set(MODELS)
    assert {m['validationPoints'] for m in first['models']} == {112}
    assert len({m['baselineMae'] for m in first['models']}) == 1
    for window in first['windows']:
        assert window['trainEnd'] < window['testStart'] <= window['testEnd']
        assert window['trainRows'] == 98 and window['testRows'] == 14


@pytest.mark.parametrize('model', list(MODELS))
def test_holdout_actuals_do_not_influence_any_holdout_prediction(model):
    history = [r for r in DEMO if r['dish'] == 'Masala Dosa']
    residuals, _, _ = evaluate_dish(history, model)
    changed = [dict(r, sales=r['sales'] + 10000) if i >= len(history)-14 else dict(r)
               for i, r in enumerate(history)]
    changed_residuals, _, _ = evaluate_dish(changed, model)
    before = [r['sales']-e for r, e in zip(history[-14:], residuals)]
    after = [r['sales']-e for r, e in zip(changed[-14:], changed_residuals)]
    assert before == pytest.approx(after)


def test_short_histories_never_claim_ml_validation():
    history = [r for r in DEMO if r['dish'] == 'Masala Dosa'][:14]
    report = model_evaluation(history)
    assert all(m['mae'] is None and m['validationPoints'] == 0 for m in report['models'])
    targets = [date.fromisoformat(history[-1]['date'])+timedelta(days=1)]
    for model in ('ridge', 'gradient_boosting'):
        predictions, used = forecast_path(history, targets, model)
        assert used == 'ensemble' and predictions[0] >= 0
        assert 'fallback' in analyze(history, 1, model)['model']


def test_zero_baseline_does_not_invent_an_improvement():
    history = [dict(date=(date(2026, 1, 1)+timedelta(days=i)).isoformat(),
                    dish='Zero', sales=0, prepared=0, price=0) for i in range(42)]
    for model in model_evaluation(history)['models']:
        assert model['mae'] == 0 and model['improvementPct'] is None


def test_forecast_model_selection_and_validation_report_use_uploaded_data():
    client = app.test_client()
    for model in MODELS:
        response = client.post('/api/forecast', data={'model': model})
        assert response.status_code == 200 and response.json['modelId'] == model
        assert all(r['modelId'] == model for r in response.json['forecasts'])
    assert client.post('/api/forecast', data={'model': 'invented'}).status_code == 400
    csv = ('date,dish,sales\n'+'\n'.join(
        f'{(date(2026,1,1)+timedelta(days=i)).isoformat()},Custom,10' for i in range(42)))
    report = client.post('/api/model-evaluation', data={'file': (io.BytesIO(csv.encode()), 'custom.csv')})
    assert report.status_code == 200
    assert not report.json['synthetic'] and report.json['rows'] == 42
    assert {m['validationPoints'] for m in report.json['models']} == {14}
    assert app.test_client().post('/api/model-evaluation').json['synthetic']
