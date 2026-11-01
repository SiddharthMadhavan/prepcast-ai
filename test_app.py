import io
import pytest
from app import app,parse,DEMO,analyze
@pytest.fixture
def client(): return app.test_client()
def test_demo(client):
    d=client.post('/api/forecast').json
    assert d['synthetic'] and d['summary']['rows']==896
    assert len(d['forecasts'])==56
    assert all(0<=r['lower']<=r['predicted']<=r['upper'] for r in d['forecasts'])
def test_upload_roundtrip(client):
    csv=client.get('/api/demo.csv').data
    d=client.post('/api/forecast',data={'file':(io.BytesIO(csv),'sales.csv'),'horizon':'30'})
    assert d.status_code==200 and not d.json['synthetic']
    assert len(d.json['forecasts'])==240
def test_invalid(client):
    for value in ['0','no','900']:
        assert client.post('/api/forecast',data={'horizon':value}).status_code==400
    for csv in [b'date,dish,sales\n2026-01-01,a,NaN',b'hello',b'date,dish,sales\n2026-01-01,a,-1']:
        assert client.post('/api/forecast',data={'file':(io.BytesIO(csv),'bad.csv')}).status_code==400
def test_oversize(client):
    assert client.post('/api/forecast',data={'file':(io.BytesIO(b'x'*2100000),'big.csv')}).status_code==413
def test_holdout_does_not_read_future():
    d=analyze(DEMO,7)
    assert d['summary']['validationPoints']==112
    assert d['summary']['mae']>0 and d['summary']['baselineMae']>0
def test_pages(client):
    assert client.get('/').status_code==200
    assert client.get('/health').json=={'status':'ok'}
def test_scenario_edit_changes_prediction_and_preserves_source(client):
    sources=client.get('/api/lab/sources').json['sources']
    assert len(sources)==3 and sources[0]['rows']!=sources[1]['rows']
    rows=sources[0]['rows']; day=rows[-1]['date']
    before=client.post('/api/lab/scenario',json={'rows':rows,'horizon':7}).json
    after=client.post('/api/lab/scenario',json={'rows':rows,'edits':{day:2000},'horizon':7}).json
    assert before['baseline']==after['baseline']
    assert sum(x['predicted'] for x in after['forecast'])>sum(x['predicted'] for x in before['forecast'])
    assert client.get('/api/lab/sources').json['sources']==sources

def test_scenario_validation(client):
    rows=client.get('/api/lab/sources').json['sources'][0]['rows']
    for payload in [{'rows':rows,'edits':{rows[-1]['date']:-1}},{'rows':rows,'edits':{'2099-01-01':10}},{'rows':rows,'horizon':99},{'rows':rows[:2]},{'rows':rows,'edits':{rows[-1]['date']:'bad'}}]:
        assert client.post('/api/lab/scenario',json=payload).status_code==400

def test_lab_upload(client):
    csv=client.get('/api/demo.csv').data
    result=client.post('/api/lab/upload',data={'file':(io.BytesIO(csv),'demo.csv')})
    assert result.status_code==200 and len(result.json['rows'])==112
    assert not result.json['synthetic']
def test_analytics_totals_and_weekday_means(client):
    from app import DEMO
    d=client.post('/api/analytics').json
    assert d['observedDays']==112 and len(d['dishes'])==8
    assert d['totalSales']==sum(r['sales'] for r in DEMO)
    assert d['totalRevenue']==sum(r['sales']*r['price'] for r in DEMO)
    assert d['recordedWaste']==sum(r['prepared']-r['sales'] for r in DEMO)
    assert all(x['days']==16 for x in d['weekdays'])
    assert d['recent']['days']==14 and d['previous']['days']==14
    assert abs(sum(r['sales'] for r in d['dishes'])-d['totalSales'])<.001

def test_analytics_upload_uses_active_data(client):
    content='date,dish,sales\n'+'\n'.join(f'2026-10-{i:02d},Dish,10' for i in range(1,15))
    d=client.post('/api/analytics',data={'file':(io.BytesIO(content.encode()),'custom.csv')}).json
    assert d['totalSales']==140 and not d['synthetic'] and not d['hasPrices']
    assert d['previous']['dailySales'] is None
