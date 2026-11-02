"""PrepCast demo: stateless forecasts; uploaded records never leave the request."""
import csv, io, math, random, statistics
from datetime import date, timedelta
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from forecasting import MODELS, forecast_path, evaluate_dish, model_evaluation
app = Flask(__name__, static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024
MENU = [('Masala Dosa',95,90),('Idli Set',65,120),('Chicken Biryani',210,75),('Veg Meals',140,105),('Paneer Butter Masala',190,48),('Filter Coffee',35,160),('Ghee Roast',110,64),('Lemon Rice',85,58)]
def synthetic():
    rng=random.Random(42); rows=[]
    for i in range(112):
        day=date(2026,7,13)+timedelta(days=i)
        for dish,price,base in MENU:
            demand=max(0,round(base*(1.25 if day.weekday()>=5 else 1)*(1+i*.0015)+rng.gauss(0,base*.12)))
            prepared=max(0,round(demand+rng.gauss(8,10)))
            sold=min(prepared,demand)
            rows.append(dict(date=day.isoformat(),dish=dish,sales=sold,prepared=prepared,price=price))
    return rows
DEMO=synthetic()
def parse(text):
    reader=csv.DictReader(io.StringIO(text.lstrip('\ufeff')))
    if not {'date','dish','sales'}.issubset(reader.fieldnames or []): raise ValueError('Required columns: date, dish, sales. Optional: prepared, price.')
    out=[]; seen=set()
    for n,r in enumerate(reader,2):
        if n>20001: raise ValueError('Maximum 20,000 rows.')
        try:
            d=date.fromisoformat(r['date']).isoformat(); dish=r['dish'].strip(); sales=float(r['sales']); prepared=float(r.get('prepared') or sales); price=float(r.get('price') or 0)
            if not dish or len(dish)>100 or not all(math.isfinite(x) and 0<=x<=1e7 for x in [sales,prepared,price]) or prepared<sales: raise ValueError()
            if (d,dish) in seen: raise ValueError()
            seen.add((d,dish)); out.append(dict(date=d,dish=dish,sales=sales,prepared=prepared,price=price))
        except (ValueError,TypeError,KeyError): raise ValueError(f'Row {n}: invalid date, quantity, price, or duplicate dish/date. Prepared must be at least sales.')
    if not out: raise ValueError('CSV has no sales records.')
    if len({r['dish'] for r in out})>100: raise ValueError('Maximum 100 dishes.')
    if len({r['date'] for r in out})<14: raise ValueError('Provide at least 14 distinct dates.')
    return sorted(out,key=lambda r:r['date'])
def predict(history,target):
    matching=[r['sales'] for r in history if date.fromisoformat(r['date']).weekday()==target.weekday()][-6:]
    recent=[r['sales'] for r in history[-14:]]
    return .75*statistics.mean(matching or recent)+.25*statistics.mean(recent)
def analyze(rows,horizon,model='ensemble'):
    if model not in MODELS: raise ValueError('Invalid forecasting model.')
    last=date.fromisoformat(max(r['date'] for r in rows)); dishes=sorted({r['dish'] for r in rows}); forecasts=[]; errors=[]; naive_errors=[]
    for dish in dishes:
        h=sorted([r for r in rows if r['dish']==dish],key=lambda r:r['date']); residual=[]
        residual, baseline, _ = evaluate_dish(h,model)
        errors.extend(abs(r) for r in residual); naive_errors.extend(baseline)
        spread=statistics.stdev(residual) if len(residual)>1 else max(5,statistics.mean(r['sales'] for r in h)*.25)
        targets=[last+timedelta(days=i) for i in range(1,horizon+1)]
        values,used=forecast_path(h,targets,model)
        for offset,(target,value) in enumerate(zip(targets,values),1):
            p=max(0,round(value)); width=math.ceil(1.64*spread*math.sqrt(1+offset/14))
            forecasts.append(dict(date=target.isoformat(),dish=dish,predicted=p,lower=max(0,p-width),upper=p+width,prep=math.ceil(p+spread*.5),price=h[-1]['price'],modelId=used))
    daily={}
    for r in rows:
        v=daily.setdefault(r['date'],dict(date=r['date'],sales=0,waste=0)); v['sales']+=r['sales']; v['waste']+=r['prepared']-r['sales']
    total=sum(r['sales'] for r in rows); waste=sum(r['prepared']-r['sales'] for r in rows)
    return dict(forecasts=forecasts,history=sorted(daily.values(),key=lambda r:r['date'])[-28:],summary=dict(rows=len(rows),dishes=len(dishes),start=min(r['date'] for r in rows),end=last.isoformat(),sales=total,waste=waste,wasteRate=round(waste/(total+waste)*100,1) if total+waste else 0,mae=round(statistics.mean(errors),2) if errors else None,baselineMae=round(statistics.mean(naive_errors),2) if naive_errors else None,validationPoints=len(errors)),model=MODELS[model]+(' (ensemble fallback for short dish histories)' if any(r['modelId']!=model for r in forecasts) else ''),modelId=model,validationMethod='Final 14 observations per dish; recursive holdout; minimum 42 observations per dish',interval='Approximate 90% residual range; not calibrated')
@app.get('/')
def index(): return send_from_directory('static','index.html')
@app.get('/health')
def health(): return jsonify(status='ok')
@app.get('/api/demo.csv')
def demo_csv():
    buf=io.StringIO(); w=csv.DictWriter(buf,fieldnames=['date','dish','sales','prepared','price']); w.writeheader(); w.writerows(DEMO)
    return app.response_class(buf.getvalue(),mimetype='text/csv',headers={'Content-Disposition':'attachment; filename=prepcast_synthetic_sales.csv'})
@app.post('/api/forecast')
def forecast():
    try:
        horizon=int(request.form.get('horizon','7'))
        if horizon not in (1,7,30): raise ValueError('Horizon must be 1, 7 or 30 days.')
        upload=request.files.get('file'); rows=parse(upload.read().decode('utf-8')) if upload else DEMO
        model=request.form.get('model','ridge')
        return jsonify(**analyze(rows,horizon,model),synthetic=not bool(upload))
    except (ValueError,UnicodeDecodeError) as e: return jsonify(error=str(e)),400
@app.post('/api/model-evaluation')
def model_evaluation_route():
    try:
        upload=request.files.get('file'); rows=parse(upload.read().decode('utf-8')) if upload else DEMO
        return jsonify(**model_evaluation(rows),synthetic=not bool(upload))
    except (ValueError,UnicodeDecodeError) as e: return jsonify(error=str(e)),400
@app.errorhandler(413)
def oversized(e): return jsonify(error='Maximum upload size is 2 MB.'),413
def analytics(rows):
    daily={}; dishes={}; weekday={i:[] for i in range(7)}
    for r in rows:
        waste=r['prepared']-r['sales']; revenue=r['sales']*r['price']
        day=daily.setdefault(r['date'],dict(date=r['date'],sales=0,revenue=0,waste=0,prepared=0))
        dish=dishes.setdefault(r['dish'],dict(dish=r['dish'],sales=0,revenue=0,waste=0,prepared=0,observations=0,zeroDays=0,soldAllPrepared=0,values=[]))
        for entry in (day,dish):
            for key,value in [('sales',r['sales']),('revenue',revenue),('waste',waste),('prepared',r['prepared'])]: entry[key]+=value
        dish['observations']+=1;dish['zeroDays']+=r['sales']==0;dish['soldAllPrepared']+=r['prepared']>0 and r['sales']==r['prepared'];dish['values'].append(r['sales'])
    ordered=sorted(daily.values(),key=lambda x:x['date']); last=date.fromisoformat(ordered[-1]['date'])
    recent=[r for r in ordered if date.fromisoformat(r['date'])>last-timedelta(days=14)]
    previous=[r for r in ordered if last-timedelta(days=28)<date.fromisoformat(r['date'])<=last-timedelta(days=14)]
    def period(items):
        return dict(days=len(items),dailySales=round(statistics.mean(r['sales'] for r in items),1) if items else None,dailyRevenue=round(statistics.mean(r['revenue'] for r in items),2) if items else None)
    for r in ordered: weekday[date.fromisoformat(r['date']).weekday()].append(r['sales'])
    for dish in dishes.values():
        values=dish.pop('values');mean=statistics.mean(values)
        dish['dailyMean']=round(mean,1);dish['cv']=round(statistics.stdev(values)/mean*100,1) if len(values)>1 and mean else 0
        dish['wasteRate']=round(dish['waste']/dish['prepared']*100,1) if dish['prepared'] else 0
    return dict(daily=ordered,dishes=sorted(dishes.values(),key=lambda x:x['sales'],reverse=True),weekdays=[dict(day=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][i],mean=round(statistics.mean(v),1) if v else None,days=len(v)) for i,v in weekday.items()],recent=period(recent),previous=period(previous),recordedWaste=sum(r['waste'] for r in ordered),totalRevenue=sum(r['revenue'] for r in ordered),totalSales=sum(r['sales'] for r in ordered),observedDays=len(ordered),start=ordered[0]['date'],end=ordered[-1]['date'])
@app.post('/api/analytics')
def analytics_route():
    try:
        upload=request.files.get('file');rows=parse(upload.read().decode('utf-8')) if upload else DEMO
        return jsonify(**analytics(rows),synthetic=not bool(upload),hasPrices=all(r['price']>0 for r in rows))
    except (ValueError,UnicodeDecodeError) as e:return jsonify(error=str(e)),400
# Scenario datasets are independent synthetic restaurants, not live POS integrations.
LAB_SOURCES = [dict(id='neighborhood',name='Neighborhood Kitchen',description='Steady weekday trade / weekend lift'),dict(id='office',name='Office District',description='Weekday lunch traffic / quieter weekends'),dict(id='weekend',name='Weekend Bistro',description='Strong weekend peaks / higher variability')]
def lab_rows(source):
    rng=random.Random(17); result=[]
    for row in analyze(DEMO,1)['history']:
        d=date.fromisoformat(row['date']); value=row['sales']
        if source=='office': value=round(value*(.48 if d.weekday()>=5 else 1.18))
        if source=='weekend': value=round(value*(1.55 if d.weekday()>=5 else .7)*rng.uniform(.88,1.12))
        result.append(dict(date=row['date'],sales=value,dish='All dishes',prepared=value,price=0))
    return result
@app.get('/api/lab/sources')
def lab_sources():
    return jsonify(sources=[dict(**s,rows=lab_rows(s['id']),synthetic=True) for s in LAB_SOURCES])
@app.post('/api/lab/upload')
def lab_upload():
    try:
        f=request.files.get('file')
        if not f: raise ValueError('Choose a CSV file.')
        rows=parse(f.read().decode('utf-8')); daily={}
        for r in rows: daily[r['date']]=daily.get(r['date'],0)+r['sales']
        return jsonify(rows=[dict(date=d,sales=v) for d,v in sorted(daily.items())],synthetic=False)
    except (ValueError,UnicodeDecodeError) as e: return jsonify(error=str(e)),400
@app.post('/api/lab/scenario')
def lab_scenario():
    try:
        body=request.get_json(silent=True) or {}
        if not isinstance(body,dict): raise ValueError('Expected a JSON object.')
        horizon=body.get('horizon',7)
        if type(horizon) is not int or horizon not in (1,7,30): raise ValueError('Invalid horizon.')
        supplied=body.get('rows'); edits=body.get('edits',{})
        if not isinstance(supplied,list) or not 14<=len(supplied)<=366 or not isinstance(edits,dict): raise ValueError('Provide 14–366 daily observations.')
        original=[]; seen=set()
        for r in supplied:
            day=date.fromisoformat(r['date']).isoformat(); value=r['sales']
            if day in seen or type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1000000: raise ValueError('Invalid or duplicate daily observation.')
            seen.add(day); original.append(dict(date=day,sales=value,dish='All dishes',prepared=value,price=0))
        if set(edits)-seen: raise ValueError('Edit dates must exist in history.')
        changed=[]
        for r in original:
            value=edits.get(r['date'],r['sales'])
            if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1000000: raise ValueError('Sales must be between 0 and 1,000,000.')
            changed.append(dict(r,sales=value,prepared=value))
        original.sort(key=lambda r:r['date']);changed.sort(key=lambda r:r['date'])
        base=analyze(original,horizon); scenario=analyze(changed,horizon)
        return jsonify(history=changed,baseline=base['forecasts'],forecast=scenario['forecasts'],model=scenario['model'],interval=scenario['interval'])
    except (ValueError,TypeError,KeyError,OverflowError) as e: return jsonify(error=str(e)),400
if __name__=='__main__': app.run(host='0.0.0.0',port=8080)

