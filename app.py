import os,sys,math
from datetime import datetime,timezone
from typing import List,Literal
import numpy as np
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
imports={}
for n in ("tradeexecutor","eth_defi","demeter"):
    try:
        m=__import__(n); imports[n]={"ok":True,"path":getattr(m,"__file__",None)}
    except Exception as e: imports[n]={"ok":False,"error":f"{type(e).__name__}: {e}"}
api=FastAPI(title="DeFi Simulator",version="0.1.0")
class P(BaseModel):
    timestamp:str
    price:float=Field(gt=0)
class R(BaseModel):
    initial_capital:float=Field(gt=0)
    prices:List[P]
    strategy:Literal["buy_and_hold","periodic_rebalance"]="buy_and_hold"
    fee_bps:float=Field(default=0,ge=0)
    slippage_bps:float=Field(default=0,ge=0)
    gas_cost_per_trade:float=Field(default=0,ge=0)
    rebalance_every:int=Field(default=24,ge=1)
def ts(s):
    try:
        d=datetime.fromisoformat(s.replace("Z","+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except: raise HTTPException(400,f"Invalid timestamp: {s}")
def cf(r): return (r.fee_bps+r.slippage_bps)/10000.0
def buy(c,p,r):
    g=r.gas_cost_per_trade
    if c<=g: raise HTTPException(400,"Gas cost >= available cash")
    f=cf(r); u=(c-g)/(p*(1+f)); return u,0.0,g+u*p*f
def sell(u,p,r):
    f=cf(r); gross=u*p; cost=gross*f+r.gas_cost_per_trade; c=gross-cost
    if c<0: raise HTTPException(400,"Trading costs exceed proceeds")
    return 0.0,c,cost
@api.get("/health")
def health(): return {"status":"ok","python":sys.version.split()[0],"imports":imports}
@api.get("/capabilities")
def caps():
    return {"strategies":["buy_and_hold","periodic_rebalance"],
            "metrics":["final_value","total_return_pct","max_drawdown_pct","annualized_volatility_pct","sharpe_ratio","trade_count","total_costs","equity_curve"],
            "defi_engines":imports,
            "protocol_native":{"uniswap_v3":False,"aave_v3":False}}
@api.post("/backtest")
def backtest(r:R):
    if len(r.prices)<2: raise HTTPException(400,"Need at least 2 observations")
    raw=[(ts(x.timestamp),float(x.price)) for x in r.prices]
    orig=[x[0] for x in raw]; raw.sort(key=lambda x:x[0]); sorted_input=orig!=[x[0] for x in raw]
    t=[x[0] for x in raw]; p=np.array([x[1] for x in raw],float)
    if not np.isfinite(p).all() or (p<=0).any(): raise HTTPException(400,"Prices must be finite and > 0")
    ds=[(t[i]-t[i-1]).total_seconds() for i in range(1,len(t))]
    med=float(np.median(ds))
    if med<=0: raise HTTPException(400,"Timestamps must increase")
    opy=(365*24*3600)/med
    cash=float(r.initial_capital); units=0.0; trades=0; costs=0.0; curve=[]
    for i,(tt,pp) in enumerate(zip(t,p)):
        if r.strategy=="buy_and_hold":
            if i==0: units,cash,c=buy(cash,pp,r); costs+=c; trades+=1
            if i==len(p)-1: units,cash,c=sell(units,pp,r); costs+=c; trades+=1
        else:
            if i==0: units,cash,c=buy(cash,pp,r); costs+=c; trades+=1
            elif i<len(p)-1 and i%r.rebalance_every==0:
                units,cash,c=sell(units,pp,r); costs+=c; trades+=1
                units,cash,c=buy(cash,pp,r); costs+=c; trades+=1
            if i==len(p)-1: units,cash,c=sell(units,pp,r); costs+=c; trades+=1
        eq=cash+units*pp; curve.append({"timestamp":tt.isoformat(),"equity":round(float(eq),8)})
    e=np.array([x["equity"] for x in curve],float); final=float(e[-1]); tr=final/r.initial_capital-1
    dd=e/np.maximum.accumulate(e)-1; mdd=float(dd.min())
    rets=e[1:]/e[:-1]-1
    vol=float(np.std(rets,ddof=1)*math.sqrt(opy)) if len(rets)>1 else 0.0
    mean=float(np.mean(rets)*opy) if len(rets) else 0.0; sharpe=mean/vol if vol>0 else 0.0
    return {"final_value":round(final,8),"total_return_pct":round(tr*100,6),
            "max_drawdown_pct":round(mdd*100,6),"annualized_volatility_pct":round(vol*100,6),
            "sharpe_ratio":round(sharpe,6),"trade_count":trades,"total_costs":round(costs,8),
            "equity_curve":curve,
            "assumptions":{"sequential_no_lookahead":True,"sorted_input":sorted_input,
            "median_interval_seconds":med,"observations_per_year":opy,"risk_free_rate":0.0,
            "fee_bps":r.fee_bps,"slippage_bps":r.slippage_bps,"gas_cost_per_trade":r.gas_cost_per_trade,
            "rebalance_every":r.rebalance_every}}
