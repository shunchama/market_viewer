"""市場ごとの値上がり・値下がり銘柄数(騰落数)ページ

東証プライム / スタンダード / グロースの全上場銘柄について、選んだ期間で
値上がり・値下がり・変わらずが何銘柄あるかを集計して1枚のHTMLにする。
HTML上で期間(週数)・終了日を切り替えられる(再計算はブラウザ内)。

  python breadth_flow.py                      # breadth.html を出力(約3,900銘柄を取得するため数分〜数十分)
  python breadth_flow.py --demo               # 合成データで動作確認(ネット不要)
  python breadth_flow.py --list-file data_j.xls   # JPXの上場銘柄一覧を手元のファイルから読む
  python breadth_flow.py --limit 300          # 動作確認用に銘柄数を絞る

データ: 銘柄一覧と市場区分=JPX『東証上場銘柄一覧』/ 株価=Yahoo Finance(yfinance)
"""
import argparse
import datetime as dt
import io
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from market_flow import asof_html, nav_html, notify

JPX_LIST_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
MARKETS = (("prime", "プライム", "プライム"), ("standard", "スタンダード", "スタンダード"),
           ("growth", "グロース", "グロース"))
WEEKS = 26
WINDOWS = [1] + [5 * k for k in range(1, WEEKS + 1)]


# ───────────────────────── 銘柄一覧 ─────────────────────────
def _code(x):
    if isinstance(x, float) and x.is_integer():
        x = int(x)
    return str(x).strip()


def parse_jpx_list(raw):
    """JPXの上場銘柄一覧(xls/xlsx)から {市場key: [Yahooティッカー]} を作る(内国株式のみ)。"""
    try:
        df = pd.read_excel(io.BytesIO(raw))
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xlrd"])
        df = pd.read_excel(io.BytesIO(raw))
    df.columns = [str(c).strip() for c in df.columns]
    code_col = next(c for c in df.columns if "コード" in c and "業種" not in c and "規模" not in c)
    seg_col = next(c for c in df.columns if "市場" in c)
    out = {k: [] for k, _, _ in MARKETS}
    for code, seg in zip(df[code_col], df[seg_col]):
        seg = str(seg)
        if "内国株式" not in seg:  # ETF・REIT・外国株式・PRO Market などは除く
            continue
        for key, _, prefix in MARKETS:
            if seg.startswith(prefix):
                out[key].append(_code(code) + ".T")
    return out


def jpx_list_date(raw):
    """銘柄一覧ファイル内の『日付』列(例: 20261125)。読めなければ None。"""
    try:
        df = pd.read_excel(io.BytesIO(raw))
        col = next(c for c in df.columns if "日付" in str(c))
        return pd.to_datetime(str(int(df[col].dropna().iloc[0])), format="%Y%m%d")
    except Exception:  # noqa: BLE001
        return None


def load_universe(list_file=None, demo=False, limit=None):
    """({市場: [ティッカー]}, 一覧ファイルの日付) を返す。"""
    if demo:
        return {"prime": [f"P{i:04d}.T" for i in range(180)], "standard": [f"S{i:04d}.T" for i in range(120)],
                "growth": [f"G{i:04d}.T" for i in range(80)]}, pd.Timestamp.today().normalize()
    if list_file:
        raw = open(list_file, "rb").read()
    else:
        import urllib.request
        req = urllib.request.Request(JPX_LIST_URL, headers={"User-Agent": "Mozilla/5.0 market-flow"})
        raw = urllib.request.urlopen(req, timeout=60).read()
    uni = parse_jpx_list(raw)
    if limit:
        uni = {k: v[: max(1, limit // 3)] for k, v in uni.items()}
    return uni, jpx_list_date(raw)


# ───────────────────────── 株価取得 ─────────────────────────
def fetch_closes(tickers, chunk=250, max_minutes=40):
    """終値(調整済み)を銘柄ごとに取得。時間切れ・失敗した銘柄は除外し、取れた分だけ返す。"""
    import yfinance as yf
    t0 = time.time()
    frames = []

    def grab(part, threads):
        df = yf.download(part, period="1y", auto_adjust=True, progress=False, threads=threads)["Close"]
        return df.to_frame(part[0]) if isinstance(df, pd.Series) else df

    for i in range(0, len(tickers), chunk):
        if time.time() - t0 > max_minutes * 60:
            print(f"[warn] 時間切れのため {i}/{len(tickers)} 銘柄で打ち切り")
            break
        part = tickers[i:i + chunk]
        try:
            frames.append(grab(part, True))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 取得失敗(chunk {i}): {type(e).__name__}: {e}")
        print(f"[info] {min(i + chunk, len(tickers))}/{len(tickers)} 銘柄 ({int(time.time() - t0)}秒)")
    if not frames:
        return pd.DataFrame()
    close = pd.concat(frames, axis=1)
    close = close.loc[:, ~close.columns.duplicated()]
    # 並列取得で失敗した銘柄(キャッシュDBのロックなど)を1回だけ直列でやり直す
    miss = [t for t in tickers[: len(tickers)] if t not in close.columns or close[t].isna().all()]
    if miss and time.time() - t0 < max_minutes * 60:
        print(f"[info] 取得できなかった {len(miss)} 銘柄を再試行")
        for i in range(0, len(miss), 100):
            if time.time() - t0 > max_minutes * 60:
                break
            try:
                retry = grab(miss[i:i + 100], False)
                for c in retry.columns:
                    close[c] = retry[c].reindex(close.index)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] 再試行失敗: {type(e).__name__}")
    return close.dropna(how="all").sort_index()


def demo_closes(universe):
    rng = np.random.default_rng(5)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=252)
    market = rng.normal(0.0002, 0.008, len(idx))
    cols = {}
    for k, tickers in universe.items():
        tilt = {"prime": 0.0, "standard": -0.0004, "growth": 0.0008}[k]
        for t in tickers:
            r = market + tilt + rng.normal(0, 0.014, len(idx))
            cols[t] = 100 * np.exp(np.cumsum(r)) + 0.0
    return pd.DataFrame(cols, index=idx).round(1)


# ───────────────────────── 集計 ─────────────────────────
def breadth(close, universe):
    """windows ごと・市場ごとの値上がり/値下がり/変わらず銘柄数(日付ごと)。"""
    groups = {k: [t for t in v if t in close.columns and close[t].notna().sum() > 30] for k, v in universe.items()}
    groups["all"] = [t for k in universe for t in groups[k]]
    filled = close.ffill()
    counts = {}
    for w in WINDOWS:
        prev = filled.shift(w)
        valid = close.notna() & prev.notna()
        up, down, flat = (close > prev) & valid, (close < prev) & valid, (close == prev) & valid
        counts[w] = {mk: {"up": up[cols].sum(axis=1).astype(int).tolist(),
                          "down": down[cols].sum(axis=1).astype(int).tolist(),
                          "flat": flat[cols].sum(axis=1).astype(int).tolist()}
                     for mk, cols in groups.items()}
    return counts, {mk: len(v) for mk, v in groups.items()}


def sources_html(asof, generated, demo, cover):
    demo_note = ("<p class='warn'>※ DEMO: 合成の乱数データです。実際の市場データではありません。</p>" if demo else "")
    rows = "".join(f"<tr><td>{n}</td><td>{cover.get('total', {}).get(k, '-')}</td><td>{cover.get(k, '-')}</td></tr>"
                   for k, n, _ in MARKETS)
    return f"""
<details class="src" open><summary>出典・計算方法・注意事項</summary>{demo_note}
<h4>データの出典</h4>
<ul>
<li><b>銘柄一覧・市場区分</b>: 日本取引所グループ(JPX)『東証上場銘柄一覧』。プライム・スタンダード・グロースの内国株式のみ(ETF・REIT・外国株式・PRO Marketは除く)。</li>
<li><b>株価</b>: Yahoo Finance(<code>yfinance</code> 経由)の日足終値(分割・配当調整済み)、直近約1年。無料データのため欠損・遅延があり、取得できなかった銘柄は集計から除外します。</li>
<li><b>最新データ日</b>: {asof} ／ <b>ページ生成</b>: {generated}</li>
</ul>
<h4>集計できた銘柄数</h4>
<table class="mini"><tr><th>市場</th><th>一覧の銘柄数</th><th>集計対象</th></tr>{rows}</table>
<h4>計算方法</h4>
<ul>
<li><b>値上がり/値下がり/変わらず</b>: 終了日の終値と、選んだ期間だけ前の営業日の終値を比べた銘柄数。期間が「1日」なら前日比です。どちらかの価格が無い銘柄(売買不成立・上場前など)は数えません。</li>
<li><b>値上がり比率</b> = 値上がり ÷ (値上がり + 値下がり)。</li>
<li><b>騰落レシオ(25日)</b> = 過去25営業日の日次の値上がり銘柄数の合計 ÷ 値下がり銘柄数の合計 × 100。一般に120以上で過熱、70以下で売られすぎとされますが、目安にすぎません。</li>
<li><b>ADライン</b> = 日次の(値上がり − 値下がり)銘柄数の累積。</li>
<li><b>注意</b>: 現在上場している銘柄で過去を計算するため、期間中に上場廃止・合併した銘柄は含まれません(生存者バイアス)。取引所が公表する騰落銘柄数とは、対象銘柄や価格の調整方法の違いで数字がずれます。</li>
</ul>
<p class="warn">本ページは情報提供のみを目的とし、投資助言ではありません。</p>
</details>"""


PAGE = r"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Market Breadth</title>
__PLOTLY__
<style>
body{font-family:-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif;max-width:1040px;margin:auto;padding:0 12px 60px;color:#222}
h2{margin:18px 0 4px}.sub{color:#666;font-size:13px;margin:0 0 10px}
.ctl{position:sticky;top:0;z-index:5;background:#fffffff2;backdrop-filter:blur(4px);padding:10px 0;border-bottom:1px solid #ddd;display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.ctl button,.ctl input{font:inherit;padding:5px 10px;border:1px solid #bbb;border-radius:6px;background:#fff;cursor:pointer}
.ctl button.on{background:#264653;color:#fff;border-color:#264653}
.ctl label{font-size:12px;color:#666}.ctl input[type=number]{width:56px}
.sum{font-size:15px;margin:12px 0 2px}.note{color:#777;font-size:12px;margin:0 0 6px}
.chart{width:100%}.dt{font-size:12px;color:#5a6b7a;margin:2px 0 0}hr.sep{border:0;border-top:2px solid #cfd6df;margin:30px 0 14px}.stale{color:#b5503a;font-weight:600}.tw{overflow-x:auto}table.h,table.mini{border-collapse:collapse;font-size:13px;width:100%}
table.h th,table.h td,table.mini th,table.mini td{padding:6px 8px;border-bottom:1px solid #eee;text-align:left}
table.h th,table.mini th{background:#e8edf3}table.h td.r,table.h th.r{text-align:right}
.up{color:#1a9850;font-weight:600}.dn{color:#c0392b;font-weight:600}.hot{background:#fde2dd}.cold{background:#dbe9f3}
.src{margin-top:34px;font-size:13px;line-height:1.7;background:#f7f8fa;padding:10px 16px;border-radius:8px}
.src summary{cursor:pointer;font-weight:600}.src h4{margin:12px 0 2px}.warn{color:#b5503a}
code{background:#eceff3;padding:1px 4px;border-radius:3px}
@media (max-width:640px){
 body{padding:0 8px 40px}h2{font-size:20px}
 .ctl{position:static;gap:5px}.ctl button,.ctl input{padding:4px 8px;font-size:13px}
 .sum{font-size:14px}table.h{font-size:12px}table.h th,table.h td{padding:5px 5px}.src{padding:8px 12px;font-size:12.5px}
}
</style></head><body>
__NAV__
<h2>市場ごとの値上がり・値下がり <small id="mk"></small></h2>
<p class="sub" id="asof"></p>
__ASOF__
<div class="ctl">
 <label>期間</label><span id="presets"></span>
 <input type="number" id="wk" min="1" max="26" title="任意の週数"><label>週</label>
 <label>終了日</label><button id="prev">◀ 前週</button><input type="date" id="end"><button id="next">次週 ▶</button><button id="latest">最新</button>
</div>
<p class="sum" id="sum"></p>
<hr class="sep">
<p class="note" id="n1"></p><div id="c1" class="chart"></div>
<hr class="sep">
<p class="note">② 騰落レシオは日次データから25営業日で計算(期間の選択には依存しません)。120以上=過熱、70以下=売られすぎが一般的な目安です。</p><div class="tw" id="tb"></div>
<hr class="sep">
<p class="note" id="n2"></p><div id="c2" class="chart"></div>
<hr class="sep">
<p class="note">④ 日次の(値上がり−値下がり)の累積。右肩上がりなら上昇が広い銘柄に及んでいる状態、指数だけ上がってADラインが下がるときは一部の銘柄頼みとされます。</p><div id="c3" class="chart"></div>
__SOURCES__
<script type="application/json" id="data">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent),META=__META__;
const N=D.dates.length,MIN_END=Math.min(30,N-1),MK=D.markets,$=id=>document.getElementById(id),st={end:N-1,weeks:META.weeks};
const M=()=>window.innerWidth<640,CFG={displayModeBar:false,responsive:true};
const FONT='-apple-system,Hiragino Sans,Noto Sans JP,sans-serif';
const L=()=>M()?{font:{family:FONT,size:11},margin:{l:78,r:10,t:44,b:44}}:{font:{family:FONT},margin:{l:110,r:30,t:50,b:50}};
const T=s=>({text:s,font:{size:M()?14:17}});
const dt=(id,html)=>{let el=document.getElementById('dt_'+id);if(!el){el=document.createElement('p');el.className='dt';el.id='dt_'+id;document.getElementById(id).insertAdjacentElement('beforebegin',el)}el.innerHTML='📅 '+html};
const COLS={up:'#2a9d8f',flat:'#b8bec6',down:'#e76f51'},LINE=['#264653','#2a9d8f','#e9c46a','#e76f51'];
const win=e=>5*Math.max(1,Math.min(st.weeks,Math.floor(e/5)));
const c=(w,mk,e)=>({up:D.counts[w][mk].up[e],down:D.counts[w][mk].down[e],flat:D.counts[w][mk].flat[e]});
const ratio=o=>(o.up+o.down)?o.up/(o.up+o.down)*100:null;
function render(){
 const e=st.end,w=win(e),rng=`${D.dates[Math.max(0,e-w)]} → ${D.dates[e]}(${w}営業日)`;
 $('asof').textContent=`最終更新 ${META.generated} ／ 最新データ ${D.dates[N-1]} ／ 表示中の終了日 ${D.dates[e]}`;
 document.querySelectorAll('#presets button').forEach(b=>b.classList.toggle('on',+b.dataset.w===Math.round(w/5)));
 $('wk').value=st.weeks;$('end').value=D.dates[e];
 const all=c(w,'all',e),r=ratio(all);
 $('sum').innerHTML=`全市場で <b style="color:#1a9850">${all.up.toLocaleString()}銘柄が値上がり</b>、<b style="color:#c0392b">${all.down.toLocaleString()}銘柄が値下がり</b>(変わらず ${all.flat.toLocaleString()}) ／ 値上がり比率 <b>${r==null?'-':r.toFixed(1)+'%'}</b>(${rng})`;

 // ① stacked share
 const rows=MK.map(m=>({m,o:c(w,m.key,e)})),tot=o=>o.up+o.down+o.flat||1;
 const tr=['up','flat','down'].map(k=>({type:'bar',orientation:'h',name:{up:'値上がり',flat:'変わらず',down:'値下がり'}[k],
  y:rows.map(x=>x.m.name),x:rows.map(x=>x.o[k]/tot(x.o)*100),marker:{color:COLS[k]},
  text:rows.map(x=>x.o[k].toLocaleString()),textposition:'inside',insidetextanchor:'middle',textfont:{color:'#fff',size:M()?10:12},
  customdata:rows.map(x=>[x.o[k].toLocaleString(),(x.o[k]/tot(x.o)*100).toFixed(1)]),
  hovertemplate:'%{y} '+{up:'値上がり',flat:'変わらず',down:'値下がり'}[k]+' %{customdata[0]}銘柄(%{customdata[1]}%)<extra></extra>'}));
 Plotly.react('c1',tr,{...L(),title:T('① 値上がり・値下がり銘柄数(構成比)'),barmode:'stack',xaxis:{range:[0,100],ticksuffix:'%'},
  yaxis:{autorange:'reversed'},height:M()?300:320,legend:{orientation:'h',y:-0.18,traceorder:'normal'}},CFG);
 $('n1').textContent=`① ${rng}。終了日の終値を、${w}営業日前の終値と比べた銘柄数(棒の中の数字=銘柄数)。`;

 // ④ table
 const lr=m=>{let u=0,d=0;for(let i=Math.max(0,e-24);i<=e;i++){u+=D.counts[1][m].up[i];d+=D.counts[1][m].down[i]}return d?u/d*100:null};
 $('tb').innerHTML='<table class="h"><tr><th>市場</th><th class="r">対象銘柄</th><th class="r">値上がり</th><th class="r">値下がり</th><th class="r">変わらず</th><th class="r">値上がり比率</th><th class="r">騰落レシオ(25日)</th></tr>'+
  rows.map(x=>{const rt=ratio(x.o),l=lr(x.m.key),cl=l==null?'':(l>=120?'hot':(l<=70?'cold':''));
   return`<tr><td>${x.m.name}</td><td class="r">${x.m.n.toLocaleString()}</td><td class="r up">${x.o.up.toLocaleString()}</td><td class="r dn">${x.o.down.toLocaleString()}</td><td class="r">${x.o.flat.toLocaleString()}</td><td class="r">${rt==null?'-':rt.toFixed(1)+'%'}</td><td class="r ${cl}">${l==null?'-':l.toFixed(1)}</td></tr>`}).join('')+'</table>';

 // ② ratio trend
 const s0=Math.max(w,e-119),xs=D.dates.slice(s0,e+1);
 const t2=MK.map((m,i)=>({type:'scatter',mode:'lines',name:m.name,x:xs,y:xs.map((_,k)=>ratio(c(w,m.key,s0+k))),line:{color:LINE[i%4],width:m.key==='all'?3:2},
  hovertemplate:m.name+' %{y:.1f}%<extra></extra>'}));
 Plotly.react('c2',t2,{...L(),title:T(`③ 値上がり比率の推移(各日の${w}営業日比)`),yaxis:{title:{text:'値上がり比率 (%)'},range:[0,100]},height:M()?380:420,margin:M()?{l:44,r:8,t:44,b:50}:{l:60,r:20,t:50,b:50},
  shapes:[{type:'line',xref:'paper',x0:0,x1:1,yref:'y',y0:50,y1:50,line:{dash:'dot',color:'#888'}}],legend:{orientation:'h',y:-0.2}},CFG);
 $('n2').textContent=`③ 50%より上なら値上がり銘柄のほうが多い状態。直近約120営業日。`;

 // ③ AD line
 const a0=Math.max(1,e-119),xa=D.dates.slice(a0,e+1);
 const t3=MK.map((m,i)=>{let s=0;return{type:'scatter',mode:'lines',name:m.name,x:xa,y:xa.map((_,k)=>{s+=D.counts[1][m.key].up[a0+k]-D.counts[1][m.key].down[a0+k];return s}),
  line:{color:LINE[i%4],width:m.key==='all'?3:2},hovertemplate:m.name+' %{y:,}<extra></extra>'}});
 Plotly.react('c3',t3,{...L(),title:T('④ ADライン(値上がり−値下がりの累積)'),yaxis:{title:{text:'累積(銘柄数)'}},height:M()?380:420,margin:M()?{l:52,r:8,t:44,b:50}:{l:70,r:20,t:50,b:50},legend:{orientation:'h',y:-0.2}},CFG);
 const end=D.dates[e],old=D.dates[Math.max(0,e-w)],lag=(e===N-1&&D.stale_n)?` ／ <span class="stale">${D.stale_n.toLocaleString()}銘柄は最新日の値なし(除外)</span>`:'';
 dt('c1',`終値の基準日 <b>${end}</b>(比較先 ${old})${lag} ／ 銘柄一覧 ${D.list_date||'不明'}`);
 dt('tb',`終値の基準日 <b>${end}</b>(騰落レシオは ${D.dates[Math.max(0,e-24)]} 〜 ${end} の25営業日)`);
 dt('c2',`最新 <b>${end}</b>(各日の${w}営業日比。${D.dates[s0]} 〜 ${end})`);
 dt('c3',`最新 <b>${end}</b>(日次の累積。${D.dates[a0]} 〜 ${end})`);
}
$('mk').textContent=META.label;
[[1,'1週'],[2,'2週'],[4,'4週'],[13,'13週']].forEach(([w,l])=>{const b=document.createElement('button');b.textContent=l;b.dataset.w=w;
 b.onclick=()=>{st.weeks=w;render()};$('presets').appendChild(b)});
$('wk').onchange=ev=>{st.weeks=Math.max(1,Math.min(26,+ev.target.value||1));render()};
$('end').min=D.dates[MIN_END];$('end').max=D.dates[N-1];
$('end').onchange=ev=>{let i=N-1;while(i>MIN_END&&D.dates[i]>ev.target.value)i--;st.end=i;render()};
$('prev').onclick=()=>{st.end=Math.max(MIN_END,st.end-5);render()};
$('next').onclick=()=>{st.end=Math.min(N-1,st.end+5);render()};
$('latest').onclick=()=>{st.end=N-1;render()};
let wasM=M();addEventListener('resize',()=>{if(M()!==wasM){wasM=M();render()}});
render();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", type=int, default=5, help="初期表示の集計期間(営業日。5=1週)")
    ap.add_argument("--demo", action="store_true", help="合成データで実行")
    ap.add_argument("--list-file", help="JPXの上場銘柄一覧(data_j.xls)を手元のファイルから読む")
    ap.add_argument("--limit", type=int, help="動作確認用: 取得する銘柄数を絞る")
    ap.add_argument("--max-minutes", type=int, default=40, help="株価取得の時間上限(分)")
    ap.add_argument("--nav", action="store_true", help="ページ間ナビを表示(公開用)")
    ap.add_argument("--notify", action="store_true", help="要約を Slack/Discord Webhook に送る")
    ap.add_argument("--offline", action="store_true", help="plotly.js をHTMLに埋め込む(CDN不要)")
    ap.add_argument("-o", "--output", default="breadth.html")
    a = ap.parse_args()

    universe, list_date = load_universe(a.list_file, a.demo, a.limit)
    total = {k: len(v) for k, v in universe.items()}
    print(f"[info] 銘柄一覧: {total}")
    tickers = [t for v in universe.values() for t in v]
    close = demo_closes(universe) if a.demo else fetch_closes(tickers, max_minutes=a.max_minutes)
    if close.empty or len(close) < 40:
        sys.exit("株価を十分に取得できませんでした")
    counts, used = breadth(close, universe)
    print(f"[info] 集計対象: {used}")

    asof = close.index[-1].date()
    generated = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M JST")
    names = {"all": "全市場", **{k: n for k, n, _ in MARKETS}}
    payload = {"dates": [d.strftime("%Y-%m-%d") for d in close.index],
               "markets": [{"key": k, "name": names[k], "n": used[k]} for k in ("all", "prime", "standard", "growth")],
               "counts": counts}
    meta = {"weeks": max(1, round(a.window / 5)), "generated": generated, "label": "(DEMO)" if a.demo else ""}
    plotly_tag = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>'
    if a.offline:
        from plotly.offline import get_plotlyjs
        plotly_tag = f"<script>{get_plotlyjs()}</script>"
    cover = {**used, "total": total}
    used_t = [t for v in universe.values() for t in v if t in close.columns]
    lv = close[used_t].apply(lambda x: x.last_valid_index())
    stale_n = int((lv < close.index[-1]).sum())
    payload["stale_n"] = stale_n
    payload["list_date"] = None if list_date is None else pd.Timestamp(list_date).strftime("%Y-%m-%d")
    rows = [("株価(上場全銘柄)", close.index[-1], f"{len(used_t):,}銘柄を集計。うち{stale_n:,}銘柄は最新日の値なし(売買不成立・取得漏れなど)"),
            ("銘柄一覧(JPX)", list_date, "上場銘柄一覧ファイルの日付" if list_date is not None else "日付を読み取れず")]
    html = (PAGE.replace("__PLOTLY__", plotly_tag)
            .replace("__ASOF__", asof_html(rows, close.index[-1]))
            .replace("__NAV__", nav_html("breadth") if a.nav else "")
            .replace("__SOURCES__", sources_html(asof, generated, a.demo, cover))
            .replace("__META__", json.dumps(meta, ensure_ascii=False))
            .replace("__DATA__", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")))
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    with open(a.output, "w", encoding="utf-8") as fh:
        fh.write(html)

    w = max(1, min(a.window, len(close) - 1))
    w = w if w in WINDOWS else 5
    lines = []
    for k in ("prime", "standard", "growth"):
        c = {x: counts[w][k][x][-1] for x in ("up", "down", "flat")}
        lines.append(f"{names[k]}: 値上がり{c['up']:,} / 値下がり{c['down']:,}")
    summary = f"[騰落数 {asof} 直近{w}営業日]\n" + "\n".join(lines)
    if a.notify:
        notify(summary)
    print(f"{summary}\n出力: {a.output}")


if __name__ == "__main__":
    main()
