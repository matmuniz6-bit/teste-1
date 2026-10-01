import csv, io, json, statistics, urllib.request, zipfile
from datetime import datetime, timezone, timedelta
from bisect import bisect_left

START = datetime(2026,9,1,tzinfo=timezone.utc)
END = datetime(2026,9,29,tzinfo=timezone.utc)
BASE = "https://data.binance.vision/data/spot/daily/klines"

def dates(a,b):
    d=a
    while d<=b:
        yield d
        d += timedelta(days=1)

def norm_ts(v):
    x=int(float(v))
    if x > 10**15:
        x//=1000
    return x

def get_zip_rows(symbol, interval, day):
    ds=day.strftime("%Y-%m-%d")
    url=f"{BASE}/{symbol}/{interval}/{symbol}-{interval}-{ds}.zip"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data=r.read()
    except Exception as e:
        print("FETCH_FAIL", symbol, interval, ds, type(e).__name__, str(e)[:120], flush=True)
        return []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            name=z.namelist()[0]
            txt=io.TextIOWrapper(z.open(name), encoding="utf-8")
            return [row for row in csv.reader(txt) if row]
    except Exception as e:
        print("ZIP_FAIL", symbol, interval, ds, type(e).__name__, str(e)[:120], flush=True)
        return []

mins=[]
for d in dates(START, END):
    rows=get_zip_rows("SOLUSDT","1m",d)
    for r in rows:
        try:
            mins.append((norm_ts(r[0]), float(r[4])))
        except:
            pass
mins.sort()
print("SOL_MINUTES", len(mins), flush=True)

rets=[None]*len(mins)
for i in range(1,len(mins)):
    rets[i]=mins[i][1]/mins[i-1][1]-1

events=[]
last_t=None
for i in range(61,len(mins)):
    cur=rets[i]
    hist=[x for x in rets[i-60:i] if x is not None]
    if len(hist)<60:
        continue
    mu=sum(hist)/len(hist)
    sd=statistics.pstdev(hist)
    if sd<=0:
        continue
    z=(cur-mu)/sd
    if cur>=0.002 and z>=5:
        t=mins[i][0]
        if last_t is None or t-last_t>=30*60*1000:
            events.append({"minute_open_ms":t,"sol_ret_1m":cur,"z":z})
            last_t=t

print("EVENTS",len(events), "EVENTS_PER_DAY",len(events)/29, flush=True)

cache={}
def one_sec_series(symbol, day):
    key=(symbol,day.strftime("%Y-%m-%d"))
    if key in cache:
        return cache[key]
    rows=get_zip_rows(symbol,"1s",day)
    arr=[]
    for r in rows:
        try:
            arr.append((norm_ts(r[0]), float(r[1])))
        except:
            pass
    arr.sort()
    cache[key]=arr
    print("ONE_SEC",symbol,key[1],len(arr), flush=True)
    return arr

def px_at_or_after(symbol, ts_ms):
    day=datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc)
    arr=one_sec_series(symbol, datetime(day.year,day.month,day.day,tzinfo=timezone.utc))
    if not arr:
        return None
    times=[x[0] for x in arr]
    j=bisect_left(times, ts_ms)
    if j>=len(arr) or arr[j][0]-ts_ms>1500:
        return None
    return arr[j][1]

details=[]
for ev in events:
    entry=ev["minute_open_ms"]+60_000
    exit_=entry+40_000
    sol0=px_at_or_after("SOLUSDT",entry)
    sol1=px_at_or_after("SOLUSDT",exit_)
    row={**ev,
         "event_utc":datetime.fromtimestamp(ev["minute_open_ms"]/1000,tz=timezone.utc).isoformat(),
         "entry_utc":datetime.fromtimestamp(entry/1000,tz=timezone.utc).isoformat()}
    if sol0 is None or sol1 is None:
        row["error"]="missing SOL 1s"
        details.append(row)
        continue
    for token in ("BOMEUSDT","BONKUSDT"):
        p0=px_at_or_after(token,entry)
        p1=px_at_or_after(token,exit_)
        if p0 is None or p1 is None:
            row[token+"_bps"]=None
        else:
            rel0=p0/sol0
            rel1=p1/sol1
            row[token+"_bps"]=(rel1/rel0-1)*10000
    details.append(row)

def stats(key):
    vals=[r[key] for r in details if r.get(key) is not None]
    if not vals:
        return None
    return {
        "n":len(vals),
        "mean_gross_bps":sum(vals)/len(vals),
        "median_gross_bps":statistics.median(vals),
        "win_rate_gross_pct":100*sum(v>0 for v in vals)/len(vals),
        "mean_net_0_86bps":sum(vals)/len(vals)-0.86,
        "win_rate_net_0_86pct":100*sum(v>0.86 for v in vals)/len(vals),
        "mean_net_3bps":sum(vals)/len(vals)-3.0,
        "win_rate_net_3pct":100*sum(v>3.0 for v in vals)/len(vals),
        "min_bps":min(vals),
        "max_bps":max(vals)
    }

bome=stats("BOMEUSDT_bps")
bonk=stats("BONKUSDT_bps")
comparable=[r for r in details if r.get("BOMEUSDT_bps") is not None and r.get("BONKUSDT_bps") is not None]
comp={
    "n":len(comparable),
    "bome_beats_bonk":sum(r["BOMEUSDT_bps"]>r["BONKUSDT_bps"] for r in comparable),
    "bonk_beats_bome":sum(r["BONKUSDT_bps"]>r["BOMEUSDT_bps"] for r in comparable),
    "bome_positive_bonk_nonpositive":sum(r["BOMEUSDT_bps"]>0 and r["BONKUSDT_bps"]<=0 for r in comparable),
    "bonk_positive_bome_nonpositive":sum(r["BONKUSDT_bps"]>0 and r["BOMEUSDT_bps"]<=0 for r in comparable),
    "either_positive":sum(max(r["BOMEUSDT_bps"],r["BONKUSDT_bps"])>0 for r in comparable),
    "both_positive":sum(r["BOMEUSDT_bps"]>0 and r["BONKUSDT_bps"]>0 for r in comparable)
}
if comparable:
    best=[max(r["BOMEUSDT_bps"],r["BONKUSDT_bps"]) for r in comparable]
    comp["oracle_best_mean_gross_bps"]=sum(best)/len(best)
    comp["oracle_best_win_rate_pct"]=100*sum(v>0 for v in best)/len(best)

summary={
    "period":"2026-09-01..2026-09-29 UTC",
    "lookback_minutes":60,
    "threshold_return_pct":0.20,
    "z_threshold":5,
    "cooldown_minutes":30,
    "hold_seconds":40,
    "event_count":len(events),
    "events_per_day":len(events)/29,
    "BOME":bome,
    "BONK_same_events":bonk,
    "comparison":comp,
    "details":details,
    "proxy_note":"Binance spot 1s opens; token/SOL synthesized as tokenUSDT/SOLUSDT. No historical DEX fills/slippage."
}
print("RESULT_JSON_START", flush=True)
print(json.dumps(summary,indent=2), flush=True)
print("RESULT_JSON_END", flush=True)
