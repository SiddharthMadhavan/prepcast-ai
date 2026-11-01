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
