"""每日 Top-3 最强信号组合回测"""

import sys, os, time, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
logging.basicConfig(level=logging.WARNING)

import numpy as np
from collections import defaultdict
from strategy.n_pattern import NPatternParams, find_n_signals
from strategy.backtest import get_limit_pct
from mootdx.quotes import Quotes

params = NPatternParams(stop_loss_pct=0.05)
SLIP = 0.001; COMM = 0.00025; TAX = 0.001
MAX_PCT = 0.2; MAX_POS = 3; MIN_STR = 65; WAIT = 5; CASH0 = 1_000_000

def scan_signals(ohlcv):
    if len(ohlcv) < 120: return []
    ohlcv = ohlcv.sort_values('date').reset_index(drop=True)
    o=ohlcv['open'].values; h=ohlcv['high'].values; l=ohlcv['low'].values
    c=ohlcv['close'].values; v=ohlcv['volume'].values
    out = []
    for i in range(120, len(o)):
        if i%5: continue
        w = slice(max(0,i-500), i)
        try: sigs = find_n_signals(o[w],h[w],l[w],c[w],v[w], params)
        except: continue
        if not sigs: continue
        best = max(sigs, key=lambda s: s['strength'])
        if best['strength'] >= MIN_STR:
            out.append((i, dict(best)))
    return out

def portfolio_sim(all_data, all_sigs):
    # dates [:10]
    date_set = set()
    for df in all_data.values():
        for d in df['date']: date_set.add(str(d)[:10])
    dates = sorted(date_set)

    arrs = {}; dmaps = {}
    for cd, df in all_data.items():
        df = df.sort_values('date').reset_index(drop=True)
        arrs[cd] = {k: df[k].values for k in ['open','high','low','close','volume','date']}
        dmaps[cd] = {str(d)[:10]: i for i,d in enumerate(df['date'])}

    d2sig = defaultdict(list)
    for cd, sl in all_sigs.items():
        if cd not in arrs: continue
        for bi, sig in sl:
            sig['code']=cd; sig['bi']=bi
            sig['dd']=str(arrs[cd]['date'][bi])[:10]
            d2sig[sig['dd']].append(sig)

    cash= CASH0; pos=[]; trades=[]; eq=[]; pend=[]

    for date in dates:
        for sig in d2sig.get(date,[]):
            pend.append({'sig':sig,'w':0,'ok':False,'cc':0})

        # exits
        dead = []
        for pi,p in enumerate(pos):
            bi = dmaps[p['code']].get(date,-1)
            if bi<0: continue
            p['hd']+=1; arr=arrs[p['code']]
            cl=arr['close'][bi]; hi=arr['high'][bi]; lo=arr['low'][bi]
            go=False; xp=0; xr=""
            if p['stop']>0 and (cl<=p['stop']):
                xp=p['stop']*(1-SLIP); xr="stop_loss"; go=True
            elif p['tgt']>0 and hi>=p['tgt']:
                xp=p['tgt']*(1-SLIP); xr="take_profit"; go=True
            elif p['hd']>=30:
                xp=cl*(1-SLIP); xr="force_exit"; go=True
            if go:
                sv=p['sh']*xp*(1-COMM-TAX)
                profit=sv-p['sh']*p['ep']*(1+COMM)
                cash+=sv
                trades.append({'code':p['code'],'ed':p['ed'],'xd':date,'ep':p['ep'],
                    'xp':xp,'sh':p['sh'],'profit':profit,'pp':(xp/p['ep']-1)*100,
                    'str':p['str'],'xr':xr})
                dead.append(pi)
        for pi in reversed(dead): pos.pop(pi)

        # entries (sorted by strength)
        pend.sort(key=lambda x:x['sig']['strength'], reverse=True)
        newp=[]
        for po in pend[:]:
            if po['ok']:
                sig=po['sig']; cd=sig['code']
                if len(pos)+len(newp)>=MAX_POS: pend.remove(po); continue
                bp=po['cc']
                ms=int(cash*MAX_PCT/bp); sh=max(100,ms//100*100)
                cost=sh*bp*(1+COMM)
                if cost>cash: sh=int(cash*0.99/bp)//100*100; cost=sh*bp*(1+COMM)
                if sh<100: pend.remove(po); continue
                cash-=cost
                sd=(sig['entry_price']-sig['stop_loss'])/sig['entry_price']
                newp.append({'code':cd,'sh':sh,'ep':bp,'ed':date,
                    'stop':round(bp*(1-sd),2),'tgt':sig['target_price'],
                    'str':sig['strength'],'hd':0})
                pend.remove(po); continue

            sig=po['sig']; cd=sig['code']
            if cd not in arrs: po['w']+=1
            else:
                bi=dmaps[cd].get(date,-1)
                if bi<0: po['w']+=1
                else:
                    arr=arrs[cd]; lim=sig['entry_price']
                    lo=arr['low'][bi]; cl=arr['close'][bi]; opn=arr['open'][bi]; vol=arr['volume'][bi]
                    pc=arr['close'][bi-1] if bi>0 else cl
                    lp=get_limit_pct(cd)
                    if lim>pc*(1+lp)*1.001 or lim<pc*(1-lp)*0.999: pend.remove(po)
                    elif lo<=lim:
                        if cl<lim: pend.remove(po)
                        elif bi>0 and vol>arr['volume'][bi-1]*1.5 and (min(opn,cl)-lo)/max(cl,0.01)<0.005: pend.remove(po)
                        elif bi>=20 and vol>np.mean(arr['volume'][bi-20:bi])*1.2: pend.remove(po)
                        else: po['ok']=True; po['cc']=cl
                    else: po['w']+=1
            if po.get('w',0)>WAIT: pend.remove(po)
        pos.extend(newp)

        tv=cash
        for p in pos:
            bi=dmaps[p['code']].get(date,-1)
            if bi>=0: tv+=p['sh']*arrs[p['code']]['close'][bi]
        eq.append(tv)

    # final force-exit
    last=dates[-1]
    for p in pos:
        lc=arrs[p['code']]['close'][-1]; xp=lc*(1-SLIP)
        sv=p['sh']*xp*(1-COMM-TAX); profit=sv-p['sh']*p['ep']*(1+COMM)
        cash+=sv
        trades.append({'code':p['code'],'ed':p['ed'],'xd':last,'ep':p['ep'],
            'xp':xp,'sh':p['sh'],'profit':profit,'pp':(xp/p['ep']-1)*100,
            'str':p['str'],'xr':'force_exit'})
    return trades, eq

# ====== main ======
print("获取主板列表...")
import akshare as ak
info=ak.stock_info_a_code_name()
m=info[info['code'].str.match(r'^(60\d{4}|00[0-4]\d{3})$')].copy()
m=m[~m['name'].str.contains('ST',na=False)]
univ=list(zip(m['code'],m['name']))
print(f"主板 {len(univ)} 只")

client=Quotes.factory(market='std', timeout=10)

print("\nPass 1: 预扫描...")
sdata={}; sigs={}; sc=0; t0=time.time()
for idx,(code,name) in enumerate(univ):
    try:
        df=client.bars(symbol=code,frequency=9,start=0,offset=500)
        if df is None or len(df)<150: continue
        df['date']=df.index.astype(str); df=df.sort_values('date').reset_index(drop=True)
        sdata[code]=df; sl=scan_signals(df)
        if sl: sigs[code]=sl; sc+=len(sl)
        if (idx+1)%500==0: print(f"  {idx+1}/{len(univ)}, {sc} signals ({time.time()-t0:.0f}s)")
    except: pass
print(f"Pass 1 done: {len(sdata)} stocks, {sc} signals ({time.time()-t0:.0f}s)")

print("\nPass 2: 组合模拟...")
t0=time.time()
trades,equity=portfolio_sim(sdata,sigs)
print(f"Done ({time.time()-t0:.0f}s)")

print("\n"+"="*90)
print("每日 Top-3 最强信号组合回测")
print("="*90)
if trades:
    wins=[t for t in trades if t['profit']>0]; losses=[t for t in trades if t['profit']<=0]
    wr=len(wins)/len(trades)*100
    tp=sum(t['profit'] for t in wins); tl=abs(sum(t['profit'] for t in losses))
    pf=tp/tl if tl>0 else 999
    ap=np.mean([t['pp'] for t in wins]) if wins else 0
    al=np.mean([t['pp'] for t in losses]) if losses else 0
    tr=sum(t['profit'] for t in trades)
    ex={}
    for t in trades: ex[t['xr']]=ex.get(t['xr'],0)+1
    st=[t['str'] for t in trades]
    print(f"交易: {len(trades)} | 胜率: {wr:.1f}% | 总利润: {tr:,.0f} | 盈亏比: {pf:.2f}")
    print(f"均盈: {ap:.1f}% | 均损: {al:.1f}% | 信号: {min(st):.0f}-{max(st):.0f}")
    print(f"出场: {ex}")
    if equity:
        fe=equity[-1]; ret=(fe/CASH0-1)*100; peak=CASH0; dd=0
        for e in equity:
            if e>peak: peak=e
            d=(peak-e)/peak*100
            if d>dd: dd=d
        print(f"期末: {fe:,.0f} ({ret:.1f}%) | 最大回撤: {dd:.1f}%")
    print("\n强度分层:")
    for lb,lo,hi in [("强≥110",110,999),("中90-109",90,109),("弱<90",0,89)]:
        tier=[t for t in trades if lo<=t['str']<hi]
        if tier:
            tw=len([t for t in tier if t['profit']>0])/len(tier)*100
            ta=np.mean([t['pp'] for t in tier])
            print(f"  {lb}: {len(tier)}笔 胜率{tw:.1f}% 均收益{ta:.1f}%")
    print("\n出场分层:")
    for r in ["take_profit","stop_loss","force_exit"]:
        tier=[t for t in trades if t['xr']==r]
        if tier:
            tw=len([t for t in tier if t['profit']>0])/len(tier)*100
            ta=np.mean([t['pp'] for t in tier])
            print(f"  {r}: {len(tier)}笔 胜率{tw:.1f}% 均收益{ta:.1f}%")
    print(f"\n全量对比: Top-3={len(trades)}笔/{wr:.1f}%/{tr:,.0f} vs 全量=1023笔/37.1%/5,477,458")
else:
    print("无交易")
