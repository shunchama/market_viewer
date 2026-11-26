"""為替(クロス円)・日米金利差の可視化ページ

円がどの通貨に対して強く/弱くなっているか、日米の金利差がどう動いているかを1枚のHTMLに出力する。
HTML上で期間(週数)・終了日を切り替えられる(再計算はブラウザ内)。

  python fx_flow.py                 # fx.html を出力
  python fx_flow.py --demo          # 合成データで動作確認(ネット不要)
  python fx_flow.py --offline       # plotly.js を埋め込む(CDN不要)
  python fx_flow.py --nav           # 株式ページへのナビを付ける(公開用)

データ: 為替=Yahoo Finance(yfinance) / 米国金利=FRED(日次) / 日本金利=財務省 国債金利情報(日次)
"""
import argparse
import datetime as dt
import io
import json
import os
import sys

import numpy as np
import pandas as pd

from market_flow import lst, load_prices, nav_html, notify

FX_PAIRS = {
    "USDJPY=X": "米ドル", "EURJPY=X": "ユーロ", "GBPJPY=X": "英ポンド", "AUDJPY=X": "豪ドル",
    "NZDJPY=X": "NZドル", "CADJPY=X": "カナダドル", "CHFJPY=X": "スイスフラン",
}
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
MOF_URL = "https://www.mof.go.jp/jgbs/reference/interest_rate/data/jgbcm_all.csv"
RATE_NAMES = {"US2": "米国債2年", "US10": "米国債10年", "JP2": "日本国債2年", "JP10": "日本国債10年"}


def _http(url, timeout=40):
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 market-flow"})
    return urllib.request.urlopen(req, timeout=timeout).read()


def parse_fred(raw):
    """FRED の fredgraph.csv(日付, 値。欠損は '.')を Series に。"""
    df = pd.read_csv(io.BytesIO(raw))
    d = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    v = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    return pd.Series(v.values, index=d).dropna().sort_index()


def parse_mof(raw, cols=("2年", "10年")):
    """財務省『国債金利情報』CSV(Shift_JIS、'基準日' を含む行がヘッダ)から指定年限を取り出す。"""
    text = raw.decode("cp932", errors="replace")
    lines = text.splitlines()
    hdr = next(i for i, ln in enumerate(lines) if "基準日" in ln)
    df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])))
    df.columns = [str(c).strip() for c in df.columns]
    d = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    out = {}
    for c in cols:
        if c in df.columns:
            out[c] = pd.Series(pd.to_numeric(df[c], errors="coerce").values, index=d).dropna().sort_index()
    return out


def load_fx(demo=False):
    """クロス円(外貨/円)の日足終値。為替の営業日をそのまま index にする。"""
    if demo:
        rng = np.random.default_rng(23)
        idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=260)
        base = dict(zip(FX_PAIRS, (150, 163, 190, 98, 90, 108, 170)))
        yen = np.cumsum(rng.normal(0.0004, 0.004, len(idx)))  # 全通貨に共通する円の動き
        return pd.DataFrame({t: base[t] * np.exp(yen + np.cumsum(rng.normal(0, 0.003, len(idx))))
                             for t in FX_PAIRS}, index=idx)
    close, _ = load_prices(list(FX_PAIRS))
    return close.dropna(how="all").ffill()


def load_rates(index, demo=False):
    """日米の2年・10年金利(%)。US=FRED(日次), JP=財務省(日次)。取れなかった系列は NaN 列。"""
    cols = list(RATE_NAMES)
    if demo:
        rng = np.random.default_rng(31)
        lvl = {"US2": 3.8, "US10": 4.2, "JP2": 0.9, "JP10": 1.6}
        return pd.DataFrame({k: lvl[k] + np.cumsum(rng.normal(0, 0.02, len(index))) for k in cols}, index=index)
    out = {}
    for key, sid in (("US2", "DGS2"), ("US10", "DGS10")):
        try:
            out[key] = parse_fred(_http(FRED_URL.format(sid)))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] FRED {sid} 取得失敗: {type(e).__name__}: {e}")
    try:
        jp = parse_mof(_http(MOF_URL))
        out["JP2"], out["JP10"] = jp.get("2年"), jp.get("10年")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 財務省 国債金利 取得失敗: {type(e).__name__}: {e}")
    res = pd.DataFrame(np.nan, index=index, columns=cols)
    for k, ser in out.items():
        if ser is not None and len(ser):
            res[k] = ser.reindex(index.union(ser.index)).ffill(limit=7).reindex(index)
    return res


def sources_html(asof, generated, demo):
    demo_note = ("<p class='warn'>※ DEMO: 合成の乱数データです。実際の市場データではありません。</p>" if demo else "")
    return f"""
<details class="src" open><summary>出典・計算方法・注意事項</summary>{demo_note}
<h4>データの出典</h4>
<ul>
<li><b>為替(クロス円)</b>: Yahoo Finance(Pythonライブラリ <code>yfinance</code> 経由、例: <code>USDJPY=X</code>)の日足終値、直近約1年。無料データのため遅延・欠損・訂正があり得ます。為替は24時間取引のため、日付の区切りは株式市場と数時間ずれます。</li>
<li><b>米国債利回り(2年・10年)</b>: FRED(セントルイス連銀、系列 DGS2 / DGS10、日次)。</li>
<li><b>日本国債利回り(2年・10年)</b>: 財務省『国債金利情報』(日次CSV)。</li>
<li>ユーロ・豪ドルなど他通貨の金利差は、日次で取れる無料の出典が無いため未対応です(米ドルとの差のみ)。</li>
<li><b>最新データ日</b>: {asof} ／ <b>ページ生成</b>: {generated}</li>
</ul>
<h4>計算方法</h4>
<ul>
<li><b>変化率</b> = 終了日のレート ÷ 比較日のレート − 1。外貨/円が上昇(プラス)=円安、下落(マイナス)=円高。1日=前営業日、1週=5営業日、1ヶ月=21営業日、3ヶ月=63営業日。</li>
<li><b>円全体(平均)</b> = 7通貨の変化率の単純平均。</li>
<li><b>日米金利差</b> = 米国債利回り − 日本国債利回り(%ポイント)。bp = 0.01%ポイント。</li>
<li><b>相関</b> = 直近60営業日の、ドル円の日次変化と日米10年金利差の日次変化のピアソン相関。</li>
</ul>
<p class="warn">本ページは情報提供のみを目的とし、投資助言ではありません。</p>
</details>"""


PAGE = r"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FX &amp; Rates</title>
__PLOTLY__
<style>
body{font-family:-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif;max-width:1040px;margin:auto;padding:0 12px 60px;color:#222}
h2{margin:18px 0 4px}.sub{color:#666;font-size:13px;margin:0 0 10px}
.ctl{position:sticky;top:0;z-index:5;background:#fffffff2;backdrop-filter:blur(4px);padding:10px 0;border-bottom:1px solid #ddd;display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.ctl button,.ctl input{font:inherit;padding:5px 10px;border:1px solid #bbb;border-radius:6px;background:#fff;cursor:pointer}
.ctl button.on{background:#264653;color:#fff;border-color:#264653}
.ctl label{font-size:12px;color:#666}.ctl input[type=number]{width:56px}
.sum{font-size:15px;margin:12px 0 2px}.note{color:#777;font-size:12px;margin:0 0 6px}
.chart{width:100%}table.h,table.mini{border-collapse:collapse;font-size:13px;width:100%}
table.h th,table.h td{padding:6px 8px;border-bottom:1px solid #eee;text-align:left}
table.h th{background:#e8edf3}.up{background:#fde8d8}.dn{background:#dbe9f3}
.src{margin-top:34px;font-size:13px;line-height:1.7;background:#f7f8fa;padding:10px 16px;border-radius:8px}
.src summary{cursor:pointer;font-weight:600}.src h4{margin:12px 0 2px}.warn{color:#b5503a}
code{background:#eceff3;padding:1px 4px;border-radius:3px}
</style></head><body>
__NAV__
<h2>為替・金利 <small id="mk"></small></h2>
<p class="sub" id="asof"></p>
<div class="ctl">
 <label>期間</label><span id="presets"></span>
 <input type="number" id="wk" min="1" max="26" title="任意の週数"><label>週</label>
 <label>終了日</label><button id="prev">◀ 前週</button><input type="date" id="end"><button id="next">次週 ▶</button><button id="latest">最新</button>
</div>
<p class="sum" id="sum"></p>
<div id="c1" class="chart"></div><p class="note">① 外貨/円の変化率。プラス(橙)=円安、マイナス(青)=円高。マスにマウスを乗せるとレートと変動幅(円)を表示します。</p>
<div id="c2" class="chart"></div><p class="note" id="n2"></p>
<div id="c3" class="chart"></div><p class="note">③ 全通貨が同じ向きなら「円」そのものの動き、通貨ごとに違えば相手通貨側の事情が大きいことを示します。</p>
<div id="ratebox">
 <div id="c4" class="chart"></div><p class="note" id="n4"></p>
 <h3 style="font-weight:400">⑤ 日米の金利</h3><div id="c5"></div><p class="note">⑤ 水準は終了日の利回り(%)。変化は期間・1日の変化(bp)。</p>
</div>
__SOURCES__
<script type="application/json" id="data">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent),META=__META__;
const N=D.dates.length,MIN_END=Math.min(70,N-1),F=D.fx.close,NM=D.fx.names,R=D.rates,P=Object.keys(F);
const st={end:N-1,weeks:META.weeks},$=id=>document.getElementById(id);
const pct=(a,b)=>(a==null||b==null||b===0)?null:(a/b-1)*100;
const fmt=(x,d=1)=>x==null?'-':(x>=0?'+':'')+x.toFixed(d)+'%';
const yen=v=>(v>=0?'+':'')+v.toFixed(2)+'円';
const LAY={font:{family:'-apple-system,Hiragino Sans,Noto Sans JP,sans-serif'},margin:{l:140,r:30,t:50,b:50}};
const COL=[[0,'#3f7fae'],[0.5,'#f7f7f7'],[1,'#f08a3c']];
function corr(x,y){const n=x.length;if(n<10)return null;const mx=x.reduce((a,b)=>a+b)/n,my=y.reduce((a,b)=>a+b)/n;
 let sxy=0,sx=0,sy=0;for(let i=0;i<n;i++){sxy+=(x[i]-mx)*(y[i]-my);sx+=(x[i]-mx)**2;sy+=(y[i]-my)**2}return(sx&&sy)?sxy/Math.sqrt(sx*sy):null}

function render(){
 const e=st.end,w=Math.min(st.weeks*5,e),a=e-w,rng=`${D.dates[a]} → ${D.dates[e]}(${w}営業日)`;
 $('asof').textContent=`最終更新 ${META.generated} ／ 最新データ ${D.dates[N-1]} ／ 表示中の終了日 ${D.dates[e]}`;
 document.querySelectorAll('#presets button').forEach(b=>b.classList.toggle('on',+b.dataset.w===st.weeks));
 $('wk').value=st.weeks;$('end').value=D.dates[e];
 const chg=P.filter(t=>F[t][e]!=null&&F[t][a]!=null).map(t=>({t,name:NM[t],r:pct(F[t][e],F[t][a]),d:F[t][e]-F[t][a],last:F[t][e]})).sort((x,y)=>y.r-x.r);
 if(!chg.length){$('sum').textContent='この期間の為替データがありません';return}
 const avg=chg.reduce((s,x)=>s+x.r,0)/chg.length,lo=chg[chg.length-1];
 $('sum').innerHTML=`円は主要${chg.length}通貨に対し平均 <b>${Math.abs(avg).toFixed(2)}% ${avg>=0?'円安':'円高'}</b>(${rng})　最も円安: <b>${chg[0].name}</b> ${fmt(chg[0].r,2)}　${lo.r<0?'最も円高':'円安が最小'}: <b>${lo.name}</b> ${fmt(lo.r,2)}`;

 // ① heatmap
 const PN=[1,5,21,63],cols=['1日','1週','1ヶ月','3ヶ月'],z=chg.map(c=>PN.map(n=>pct(F[c.t][e],F[c.t][e-n])));
 const lim=Math.max(1,...z.flat().filter(v=>v!=null).map(Math.abs));
 const hv=chg.map((c,i)=>PN.map((n,k)=>{const y=F[c.t][e-n],x=F[c.t][e];return`<b>${c.name}/円</b>  ${cols[k]}: ${fmt(z[i][k],2)}<br>${x==null?'-':x.toFixed(2)}円(${(x==null||y==null)?'-':yen(x-y)})`}));
 Plotly.react('c1',[{type:'heatmap',z,x:cols,y:chg.map(c=>c.name+'/円'),zmid:0,zmin:-lim,zmax:lim,colorscale:COL,
  text:z.map(r=>r.map(v=>v==null?'':v.toFixed(2)+'%')),texttemplate:'%{text}',hovertext:hv,hovertemplate:'%{hovertext}<extra></extra>'}],
  {...LAY,title:{text:'① クロス円の変化率(橙=円安 / 青=円高)'},height:100+40*chg.length,yaxis:{autorange:'reversed'}},{displayModeBar:false});

 // ② bars (selected window)
 const rv=[...chg].reverse();
 Plotly.react('c2',[{type:'bar',orientation:'h',x:rv.map(x=>x.r),y:rv.map(x=>x.name+'/円'),marker:{color:rv.map(x=>x.r>=0?'#f08a3c':'#3f7fae')},
  customdata:rv.map(x=>[x.last.toFixed(2),yen(x.d)]),hovertemplate:'%{y}<br>%{x:.2f}%<br>%{customdata[0]}円(%{customdata[1]})<extra></extra>'}],
  {...LAY,title:{text:'② 選択期間の変化率(右=円安 / 左=円高)'},xaxis:{title:{text:'変化率 (%)'}},height:100+36*chg.length},{displayModeBar:false});
 $('n2').textContent=`② ${rng}。外貨/円が上昇=円安、下落=円高。`;

 // ③ strength lines
 const xs=D.dates.slice(a,e+1),ok=chg.map(c=>c.t),base=t=>F[t][a];
 const tr=ok.map(t=>({type:'scatter',mode:'lines',name:NM[t],x:xs,y:xs.map((_,k)=>{const v=F[t][a+k];return(v==null||!base(t))?null:v/base(t)*100}),
  hovertemplate:NM[t]+' %{y:.2f}<extra></extra>'}));
 tr.push({type:'scatter',mode:'lines',name:'円全体(平均)',line:{color:'#111',width:4,dash:'dash'},x:xs,
  y:xs.map((_,k)=>{let s=0,n=0;ok.forEach(t=>{const v=F[t][a+k];if(v!=null&&base(t)){s+=v/base(t)*100;n++}});return n?s/n:null})});
 Plotly.react('c3',tr,{...LAY,title:{text:'③ 円の強弱(期間開始=100・上=円安)'},height:460,margin:{l:60,r:20,t:50,b:50}},{displayModeBar:false});

 // ④⑤ rates
 const U=F['USDJPY=X'],rb=$('ratebox');
 if(!(R&&R.US10&&R.JP10&&U)){rb.style.display='none';return}
 rb.style.display='';
 const sp=(x,y)=>D.dates.map((_,i)=>(x[i]==null||y[i]==null)?null:x[i]-y[i]),s10=sp(R.US10,R.JP10),s2=(R.US2&&R.JP2)?sp(R.US2,R.JP2):null;
 const s0=Math.max(0,e-249),X=D.dates.slice(s0,e+1);
 const t4=[{type:'scatter',mode:'lines',name:'ドル円(左軸)',x:X,y:U.slice(s0,e+1),line:{color:'#111'},yaxis:'y',connectgaps:true},
  {type:'scatter',mode:'lines',name:'日米10年金利差(右軸)',x:X,y:s10.slice(s0,e+1),line:{color:'#e76f51'},yaxis:'y2',connectgaps:true}];
 if(s2)t4.push({type:'scatter',mode:'lines',name:'日米2年金利差(右軸)',x:X,y:s2.slice(s0,e+1),line:{color:'#2a9d8f',dash:'dot'},yaxis:'y2',connectgaps:true});
 Plotly.react('c4',t4,{...LAY,title:{text:'④ 日米金利差とドル円(直近約1年)'},height:480,margin:{l:60,r:60,t:50,b:70},
  yaxis:{title:{text:'ドル円(円)'}},yaxis2:{title:{text:'金利差 (%pt)'},overlaying:'y',side:'right',showgrid:false,tickformat:'.2f'},legend:{orientation:'h',y:-0.15}},{displayModeBar:false});
 const d10=(s10[e]!=null&&s10[a]!=null)?(s10[e]-s10[a])*100:null,xa=[],ya=[];
 for(let i=Math.max(1,e-59);i<=e;i++)if(s10[i]!=null&&s10[i-1]!=null&&U[i]!=null&&U[i-1]!=null){xa.push(s10[i]-s10[i-1]);ya.push(U[i]-U[i-1])}
 const r=corr(xa,ya);
 $('n4').textContent=`④ 日米10年金利差 ${s10[e]==null?'-':s10[e].toFixed(2)+'%pt'}(${rng}で${d10==null?'-':(d10>=0?'+':'')+d10.toFixed(0)}bp)。金利差の拡大はドル高・円安要因とされますが、常に連動するわけではありません。直近60日の日次変化の相関: ${r==null?'-':r.toFixed(2)}`;
 const bp=(s,i,j)=>(s&&s[i]!=null&&s[j]!=null)?(s[i]-s[j])*100:null,bt=v=>v==null?'-':(v>=0?'+':'')+v.toFixed(0)+'bp',
  cls=v=>v==null?'':(v>=0?'up':'dn'),rows=[];
 Object.keys(R).forEach(k=>rows.push([META.rn[k]||k,R[k][e],bp(R[k],e,a),bp(R[k],e,e-1)]));
 if(s2)rows.push(['日米2年金利差',s2[e],bp(s2,e,a),bp(s2,e,e-1)]);
 rows.push(['日米10年金利差',s10[e],bp(s10,e,a),bp(s10,e,e-1)]);
 $('c5').innerHTML='<table class="h"><tr><th>項目</th><th>水準(%)</th><th>期間の変化</th><th>1日の変化</th></tr>'+
  rows.map(r=>`<tr><td>${r[0]}</td><td>${r[1]==null?'-':r[1].toFixed(3)}</td><td class="${cls(r[2])}">${bt(r[2])}</td><td class="${cls(r[3])}">${bt(r[3])}</td></tr>`).join('')+'</table>';
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
render();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", type=int, default=5, help="初期表示の集計期間(営業日。5=1週)")
    ap.add_argument("--demo", action="store_true", help="合成データで実行")
    ap.add_argument("--nav", action="store_true", help="ページ間ナビ(株式/為替)を表示(公開用)")
    ap.add_argument("--notify", action="store_true", help="要約を Slack/Discord Webhook に送る")
    ap.add_argument("--offline", action="store_true", help="plotly.js をHTMLに埋め込む(CDN不要)")
    ap.add_argument("-o", "--output", default="fx.html")
    a = ap.parse_args()

    fx = load_fx(demo=a.demo)
    fx = fx[[t for t in FX_PAIRS if t in fx.columns and fx[t].notna().any()]]
    if fx.empty or len(fx) < 80:
        sys.exit("為替データを十分に取得できませんでした")
    rates = load_rates(fx.index, demo=a.demo)

    asof = fx.index[-1].date()
    generated = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M JST")
    payload = {
        "dates": [d.strftime("%Y-%m-%d") for d in fx.index],
        "fx": {"names": {t: FX_PAIRS[t] for t in fx.columns}, "close": {t: lst(fx[t]) for t in fx.columns}},
        "rates": {k: lst(rates[k]) for k in rates.columns if rates[k].notna().any()},
    }
    meta = {"weeks": max(1, round(a.window / 5)), "generated": generated, "rn": RATE_NAMES,
            "label": "(DEMO)" if a.demo else ""}
    plotly_tag = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>'
    if a.offline:
        from plotly.offline import get_plotlyjs
        plotly_tag = f"<script>{get_plotlyjs()}</script>"
    data_json = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    html = (PAGE.replace("__PLOTLY__", plotly_tag)
            .replace("__NAV__", nav_html("fx") if a.nav else "")
            .replace("__SOURCES__", sources_html(asof, generated, a.demo))
            .replace("__META__", json.dumps(meta, ensure_ascii=False))
            .replace("__DATA__", data_json))
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    with open(a.output, "w", encoding="utf-8") as fh:
        fh.write(html)

    ch = (fx.iloc[-1] / fx.iloc[-1 - a.window] - 1) * 100
    avg = ch.mean()
    line = f"円: 主要{len(ch)}通貨に対し平均{abs(avg):.2f}%{'円安' if avg >= 0 else '円高'}(直近{a.window}営業日)"
    if rates[["US10", "JP10"]].notna().all(axis=1).any():
        line += f" / 日米10年金利差 {(rates['US10'] - rates['JP10']).dropna().iloc[-1]:.2f}%pt"
    else:
        print("[warn] 日米金利が取得できなかったため、金利差のセクションは非表示になります")
    if a.notify:
        notify(f"[為替 {asof}]\n{line}")
    print(f"{line}\n出力: {a.output}")


if __name__ == "__main__":
    main()
