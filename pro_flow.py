"""プロ指標ページ: 機関投資家・クオンツが見る指標を無料データでまとめる

  A. セクターの強弱   12-1か月モメンタム順位 / Zスコア / 50日・200日線(業種単位のブレッドス)
  B. ボラティリティ   VIXの期間構造(VIX/VIX3M) / VIX − 実現ボラティリティ
  C. 金利・クレジット 米10年−2年、ハイイールド・スプレッド、米実質金利、ドル指数、銅/金比率
  D. ポジション       CFTC COT(円ほか主要通貨の投機筋の買い越し・売り越し)

  python pro_flow.py                 # pro.html を出力
  python pro_flow.py --demo          # 合成データで動作確認(ネット不要)
  python pro_flow.py --nav           # 他ページへのナビを付ける(公開用)

データ: 株価・指数=Yahoo Finance / 米国債=米財務省 / HYスプレッド=FRED / COT=CFTC。
取得できない系列は、そのグラフだけ非表示になる(ページ全体は壊れない)。
"""
import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd

import market_flow as mf
from fx_flow import FRED_URL, _clean, _http, parse_fred
from market_flow import MARKETS, asof_html, date_str, last_span, lst, nav_html

MACRO = {"^VIX": "VIX", "^VIX3M": "VIX3M(3か月)", "^VIX9D": "VIX9D(9日)", "^GSPC": "S&P500", "^N225": "日経平均",
         "DX-Y.NYB": "ドル指数", "HG=F": "銅先物", "GC=F": "金先物"}
TREASURY_TMPL = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
                 "daily-treasury-rates.csv/{y}/all?type={kind}&field_tdr_date_value={y}&page&_format=csv")
COT_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"   # Legacy - Futures Only
COT_CODES = {"097741": ("JPY", "円"), "099741": ("EUR", "ユーロ"), "096742": ("GBP", "英ポンド"),
             "232741": ("AUD", "豪ドル"), "090741": ("CAD", "カナダドル"), "092741": ("CHF", "スイスフラン")}


# ───────────────────────── 取得 ─────────────────────────
def parse_treasury_cols(raw, want):
    """米財務省CSV(Date, '2 Yr' など)から want={キー: 列名(大文字小文字・空白は無視)} を Series で取り出す。"""
    import io
    df = pd.read_csv(io.BytesIO(raw))
    norm = {str(c).strip().upper().replace("  ", " "): c for c in df.columns}
    d = pd.to_datetime(df["Date"], errors="coerce")
    out = {}
    for key, col in want.items():
        c = norm.get(col.upper())
        if c is not None:
            out[key] = _clean(pd.Series(pd.to_numeric(df[c], errors="coerce").values, index=d))
    return out


def fetch_treasury(kind, want, years=3):
    year = dt.date.today().year
    parts = [parse_treasury_cols(_http(TREASURY_TMPL.format(y=y, kind=kind), timeout=30, tries=2), want)
             for y in range(year - years + 1, year + 1)]
    return {k: _clean(pd.concat([p[k] for p in parts if k in p])) for k in want if any(k in p for p in parts)}


def fetch_cot():
    """CFTC COT(Legacy, 先物のみ)。通貨ごとに {date, net, oi}(投機筋=非商業の買い−売り)。"""
    import urllib.parse
    codes = ",".join(f"'{c}'" for c in COT_CODES)
    q = urllib.parse.urlencode({"$where": f"cftc_contract_market_code in ({codes})",
                                "$order": "report_date_as_yyyy_mm_dd DESC", "$limit": "1200"})
    rows = json.loads(_http(f"{COT_URL}?{q}", timeout=40, tries=2))
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    df["d"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"].str[:10], errors="coerce")
    for c in ("noncomm_positions_long_all", "noncomm_positions_short_all", "open_interest_all"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    out = {}
    for code, (key, name) in COT_CODES.items():
        g = df[df["cftc_contract_market_code"] == code].dropna(subset=["d"]).sort_values("d").tail(110)
        if len(g):
            out[key] = {"name": name, "d": [x.strftime("%Y-%m-%d") for x in g["d"]],
                        "net": (g["noncomm_positions_long_all"] - g["noncomm_positions_short_all"]).astype(float).tolist(),
                        "oi": g["open_interest_all"].astype(float).tolist()}
    return out or None


def safe(label, fn, notes):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        print(f"[warn] {label} 取得失敗: {type(e).__name__}: {e}")
        notes.append(f"{label}: 取得できず")
        return None


def demo_macro():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=520)
    n = len(idx)

    def walk(base, sd):
        return base * np.exp(np.cumsum(rng.normal(0, sd, n)))

    def mean_rev(base, sd):
        x = np.empty(n)
        x[0] = base
        for i in range(1, n):
            x[i] = x[i - 1] + 0.05 * (base - x[i - 1]) + rng.normal(0, sd)
        return np.clip(x, 9, None)

    vix = mean_rev(18, 1.1)
    df = pd.DataFrame({"^VIX": vix, "^VIX3M": vix * 0.07 + 0.93 * mean_rev(19.5, 0.6), "^VIX9D": vix * 0.8 + mean_rev(4, 0.5),
                       "^GSPC": walk(5800, 0.009), "^N225": walk(38000, 0.011), "DX-Y.NYB": walk(104, 0.003),
                       "HG=F": walk(4.2, 0.012), "GC=F": walk(2600, 0.008)}, index=idx).round(3)
    mf.LAST_VALID.update({t: idx[-1] for t in df.columns})
    return df


def demo_rates(idx):
    rng = np.random.default_rng(8)
    n = len(idx)
    mk = lambda base, sd: pd.Series(base + np.cumsum(rng.normal(0, sd, n)), index=idx)
    return {"T10Y2Y": mk(0.4, 0.02), "HY": mk(3.2, 0.03).clip(lower=2.2), "REAL10": mk(1.9, 0.02)}


def demo_cot():
    rng = np.random.default_rng(12)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=104, freq="W-TUE")
    out = {}
    for code, (key, name) in COT_CODES.items():
        net = np.cumsum(rng.normal(0, 6000, len(dates))) + rng.normal(-20000 if key == "JPY" else 5000, 8000)
        out[key] = {"name": name, "d": [d.strftime("%Y-%m-%d") for d in dates], "net": net.round().tolist(),
                    "oi": (200000 + rng.normal(0, 8000, len(dates))).round().tolist()}
    return out


# ───────────────────────── HTML ─────────────────────────
def sources_html(asof, generated, demo, notes):
    demo_note = "<p class='warn'>※ DEMO: 合成の乱数データです。実際の市場データではありません。</p>" if demo else ""
    miss = ("<p class='warn'>取得できなかった項目: " + " / ".join(notes) + "</p>") if notes else ""
    return f"""
<details class="src" open><summary>出典・計算方法・注意事項</summary>{demo_note}{miss}
<h4>データの出典</h4>
<ul>
<li><b>株価・指数・先物</b>: Yahoo Finance(<code>yfinance</code> 経由)。セクターは NEXT FUNDS TOPIX-17 ETF(1617〜1633)と Select Sector SPDR、VIX系は <code>^VIX</code> <code>^VIX3M</code> <code>^VIX9D</code>、ドル指数 <code>DX-Y.NYB</code>、銅 <code>HG=F</code>、金 <code>GC=F</code>。無料データのため遅延・欠損があります。</li>
<li><b>米国債(2年・10年・実質10年)</b>: 米財務省『Daily Treasury Par Yield Curve / Real Yield Curve Rates』(日次CSV)。</li>
<li><b>ハイイールド・スプレッド</b>: FRED(セントルイス連銀)系列 <code>BAMLH0A0HYM2</code>(ICE BofA 米国ハイイールド債 OAS)。</li>
<li><b>COT</b>: 米商品先物取引委員会(CFTC)Commitments of Traders(Legacy・先物のみ)。基準日は火曜日、公表は通常その週の金曜日で、<b>3日ほど遅れて</b>出ます。</li>
<li><b>最新データ日</b>: {asof} ／ <b>ページ生成</b>: {generated}</li>
</ul>
<h4>計算方法</h4>
<ul>
<li><b>12-1か月モメンタム</b> = 21営業日前の価格 ÷ 約252営業日前の価格 − 1(直近1か月を除く)。直近1か月は反転しやすいため除くのが慣例です。</li>
<li><b>勢い/ボラ</b> = 12-1か月モメンタム ÷ 直近60日の年率ボラティリティ。値動きの荒さを考慮した強さです。</li>
<li><b>Zスコア</b> = (終値 − 60日平均) ÷ 60日標準偏差。+2超は短期的な買われすぎ、−2未満は売られすぎの目安です。</li>
<li><b>VIX/VIX3M</b> = 1未満は平常(期先が高い=コンタンゴ)、1超は期近が高い(バックワーデーション)で、市場のストレスが高いとされます。</li>
<li><b>実現ボラティリティ</b> = 直近20営業日の日次騰落率の標準偏差 × √252。VIX − 実現ボラ が大きいほど、市場が将来の変動を割高に織り込んでいます。</li>
<li><b>COTの買い越し</b> = 非商業(投機筋)の買い建玉 − 売り建玉。「円」は円先物のポジションで、<b>買い越し=円高方向</b>の賭けです。他通貨は通貨先物(対ドル)の買い越し=その通貨高方向です。</li>
</ul>
<p class="warn">指標は単独で将来を予測するものではありません。売買コストを考慮した検証が必要で、本ページは情報提供のみを目的とし、投資助言ではありません。</p>
</details>"""


PAGE = r"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pro Indicators</title>
__PLOTLY__
<style>
body{font-family:-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif;max-width:1040px;margin:auto;padding:0 12px 60px;color:#222}
h2{margin:18px 0 4px}h3.sec{margin:34px 0 0;padding:6px 10px;background:#264653;color:#fff;border-radius:6px;font-weight:500;font-size:16px}
.sub{color:#666;font-size:13px;margin:0 0 10px}
.ctl{position:sticky;top:0;z-index:5;background:#fffffff2;backdrop-filter:blur(4px);padding:10px 0;border-bottom:1px solid #ddd;display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.ctl button{font:inherit;padding:5px 10px;border:1px solid #bbb;border-radius:6px;background:#fff;cursor:pointer}
.ctl button.on{background:#264653;color:#fff;border-color:#264653}.ctl label{font-size:12px;color:#666}
.sum{font-size:14px;line-height:1.8;margin:12px 0 2px;background:#f4f7fa;padding:8px 12px;border-radius:8px}
.note{color:#777;font-size:12px;margin:0 0 6px}.chart{width:100%}.dt{font-size:12px;color:#5a6b7a;margin:2px 0 0}
hr.sep{border:0;border-top:2px solid #cfd6df;margin:30px 0 14px}
.tw{overflow-x:auto}table.h,table.mini{border-collapse:collapse;font-size:13px;width:100%}
table.h th,table.h td{padding:6px 8px;border-bottom:1px solid #eee;text-align:right;white-space:nowrap}
table.h th:nth-child(-n+2),table.h td:nth-child(-n+2){text-align:left}table.h th{background:#e8edf3}
.up{background:#d8f3dc}.dn{background:#fde2dd}.yes{color:#1a9850;font-weight:600}.no{color:#c0392b;font-weight:600}
.src{margin-top:34px;font-size:13px;line-height:1.7;background:#f7f8fa;padding:10px 16px;border-radius:8px}
.src summary{cursor:pointer;font-weight:600}.src h4{margin:12px 0 2px}.warn{color:#b5503a}
code{background:#eceff3;padding:1px 4px;border-radius:3px}
@media (max-width:640px){
 body{padding:0 8px 40px}h2{font-size:20px}.ctl{position:static;gap:5px}.ctl button{padding:4px 8px;font-size:13px}
 .sum{font-size:13px}table.h{font-size:12px}table.h th,table.h td{padding:5px 5px}.src{padding:8px 12px;font-size:12.5px}
}
</style></head><body>
__NAV__
<h2>プロ指標 <small id="mk"></small></h2>
<p class="sub" id="asof"></p>
__ASOF__
<div class="ctl"><label>時系列の表示期間</label><span id="mon"></span><label>　セクター</label><span id="mkt"></span></div>
<p class="sum" id="sum"></p>

<h3 class="sec">A. セクターの強弱(モメンタム・ブレッドス)</h3>
<hr class="sep"><p class="note" id="nA0"></p><div id="a0"></div>
<hr class="sep"><p class="note">① 12-1か月モメンタムの順位。過去1年の上昇率から直近1か月を除いたもので、機関投資家が「強いセクターを買い、弱いセクターを売る」ときの基本の物差しです。</p><div id="a1" class="chart"></div>
<hr class="sep"><p class="note">② セクター別の指標一覧。勢い/ボラは値動きの荒さを考慮した強さ、Zスコアは60日平均からの乖離(±2が目安)、50日線・200日線は価格が上にあるか(○)下か(×)です。</p><div class="tw" id="a2"></div>
<hr class="sep"><p class="note">③ Zスコア(60日平均からの乖離)。右に大きいほど短期的に買われすぎ、左に大きいほど売られすぎで、平均に戻りやすい(平均回帰)とみる指標です。</p><div id="a3" class="chart"></div>

<h3 class="sec">B. ボラティリティ</h3>
<hr class="sep"><p class="note">④ VIX(米国の恐怖指数)とその期先(3か月・9日)。点線は VIX/VIX3M(右軸)で、1を上回ると期近が高い状態(ストレス)です。</p><div id="b1" class="chart"></div>
<hr class="sep"><p class="note">⑤ VIX と、実際の値動きの大きさ(実現ボラティリティ・20日)の比較。VIXが実現ボラを大きく上回る=市場が将来の変動を割高に織り込んでいる状態です。日経平均には同等の指数をここでは使わず、実現ボラのみ表示します。</p><div id="b2" class="chart"></div>

<h3 class="sec">C. 金利・クレジット・ドル</h3>
<hr class="sep"><p class="note">⑥ 米国の10年−2年金利差。マイナス(逆イールド)は景気後退の先行指標とされます。</p><div id="c1" class="chart"></div>
<hr class="sep"><p class="note">⑦ 米国ハイイールド債のスプレッド(国債との利回り差)。上昇は信用不安・リスクオフのサインです。</p><div id="c2" class="chart"></div>
<hr class="sep"><p class="note">⑧ 米国10年の実質金利(物価上昇を除いた金利)。上昇は株式のバリュエーションや金には逆風とされます。</p><div id="c3" class="chart"></div>
<hr class="sep"><p class="note">⑨ ドル指数(主要通貨に対するドルの強さ)。</p><div id="c4" class="chart"></div>
<hr class="sep"><p class="note">⑩ 銅/金比率(銅先物 ÷ 金先物 × 1000)。景気に敏感な銅と安全資産の金の比で、上昇は景気拡大・長期金利上昇と連動しやすいとされます。</p><div id="c5" class="chart"></div>

<h3 class="sec">D. ポジション(COT)</h3>
<div id="cotbox">
<hr class="sep"><p class="note">⑪ 円先物の投機筋(非商業)の買い越し・売り越し(建玉数)。買い越し=円高方向、売り越し=円安方向のポジションで、偏りが極端なときは反転(ポジションの巻き戻し)に注意が必要とされます。</p><div id="d1" class="chart"></div>
<hr class="sep"><p class="note">⑫ 主要通貨の投機筋ポジション(建玉全体に対する割合)。右=その通貨の買い越し。</p><div id="d2" class="chart"></div>
</div>
__SOURCES__
<script type="application/json" id="data">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent),META=__META__,REF=D.ref,$=id=>document.getElementById(id);
const M=()=>window.innerWidth<640,CFG={displayModeBar:false,responsive:true};
const FONT='-apple-system,Hiragino Sans,Noto Sans JP,sans-serif';
const L=()=>M()?{font:{family:FONT,size:11},margin:{l:96,r:10,t:44,b:44}}:{font:{family:FONT},margin:{l:140,r:30,t:50,b:50}};
const T=s=>({text:s,font:{size:M()?14:17}});
const st={months:6,mkt:'jp'};
const dt=(id,html)=>{let el=$('dt_'+id);if(!el){el=document.createElement('p');el.className='dt';el.id='dt_'+id;$(id).insertAdjacentElement('beforebegin',el)}el.innerHTML='📅 '+html};
const stale=d=>(new Date(REF)-new Date(d))/864e5>3;
const dtl=(id,items)=>dt(id,items.map(([l,d])=>d?`${l} ${stale(d)?`<span class="stale">${d}</span>`:`<b>${d}</b>`}`:`${l} <span class="stale">取得できず</span>`).join(' ／ '));
const cut=()=>{const d=new Date(REF);d.setMonth(d.getMonth()-st.months);return d.toISOString().slice(0,10)};
const sl=(d,v)=>{const c=cut(),x=[],y=[];for(let i=0;i<d.length;i++)if(d[i]>=c&&v[i]!=null){x.push(d[i]);y.push(v[i])}return{x,y}};
const lastOf=(d,v)=>{for(let i=d.length-1;i>=0;i--)if(v[i]!=null)return{date:d[i],val:v[i],i};return null};
const mean=a=>a.reduce((s,x)=>s+x,0)/a.length,sd=a=>{const m=mean(a);return Math.sqrt(a.reduce((s,x)=>s+(x-m)**2,0)/a.length)};
const pctile=(a,v)=>{const b=a.filter(x=>x!=null);return b.length?b.filter(x=>x<=v).length/b.length*100:null};
const fm=(x,d=1)=>x==null?'-':(x>=0?'+':'')+x.toFixed(d)+'%';
const tr=(name,x,y,o={})=>({type:'scatter',mode:'lines',name,x,y,connectgaps:true,hovertemplate:name+' %{y:.2f}<extra></extra>',...o});
const LN=(extra={})=>({...L(),height:M()?300:340,margin:M()?{l:44,r:extra.y2?44:8,t:44,b:54}:{l:60,r:extra.y2?60:20,t:50,b:54},legend:{orientation:'h',y:-0.22,font:{size:M()?10:12}}});
const hide=(id,msg)=>{Plotly.purge(id);$(id).innerHTML=`<p class="note" style="padding:14px 0">${msg}</p>`};
const MC=D.macro,mc=t=>MC.close[t];
function rvol(c,k=20){const o=new Array(c.length).fill(null);for(let i=k;i<c.length;i++){const r=[];for(let j=i-k+1;j<=i;j++)if(c[j]!=null&&c[j-1]!=null)r.push(c[j]/c[j-1]-1);if(r.length>=k*0.8)o[i]=sd(r)*Math.sqrt(252)*100}return o}

function secMetrics(G){
 const n=G.dates.length,Lx=n-1,res=[];
 for(const t of Object.keys(G.names)){const c=G.close[t];if(!c||c[Lx]==null)continue;
  const at=k=>Lx-k>=0?c[Lx-k]:null,ret=k=>at(k)?(c[Lx]/at(k)-1)*100:null,K=Math.min(252,Lx);
  const m121=(at(21)&&at(K))?(at(21)/at(K)-1)*100:null;
  const win=k=>{const a=[];for(let i=Math.max(0,Lx-k+1);i<=Lx;i++)if(c[i]!=null)a.push(c[i]);return a};
  const w60=win(60),w50=win(50),w200=win(200),rets=[];
  for(let i=Math.max(1,Lx-59);i<=Lx;i++)if(c[i]!=null&&c[i-1]!=null)rets.push(c[i]/c[i-1]-1);
  const vol=rets.length>10?sd(rets)*Math.sqrt(252)*100:null;
  res.push({t,name:G.names[t],m121,r6:ret(126),r3:ret(63),r1:ret(21),vol,ra:(m121!=null&&vol)?m121/vol:null,
   z:(w60.length>=30&&sd(w60))?(c[Lx]-mean(w60))/sd(w60):null,a50:w50.length>=50?c[Lx]>mean(w50):null,a200:w200.length>=200?c[Lx]>mean(w200):null,K})}
 return res}

function renderA(){
 const G=D.groups[st.mkt],rows=secMetrics(G),dts=Object.keys(G.names).map(t=>D.last[t]).filter(Boolean).sort(),lastD=dts.length?dts[dts.length-1]:null;
 const ok=rows.filter(r=>r.m121!=null);
 const n200=rows.filter(r=>r.a200===true).length,d200=rows.filter(r=>r.a200!=null).length,n50=rows.filter(r=>r.a50===true).length,d50=rows.filter(r=>r.a50!=null).length,up1=rows.filter(r=>r.r1>0).length;
 $('nA0').textContent='業種単位のブレッドス(市場の広がり)。全銘柄の代わりにセクターETFで「上昇がどれだけの業種に広がっているか」を見ます。';
 $('a0').innerHTML=`<p class="sum" style="margin:0">${st.mkt==='jp'?'日本(TOPIX-17業種)':'米国(11セクター)'}: <b>${d200?n200+'/'+d200:'-'}</b> が200日線の上 ／ <b>${d50?n50+'/'+d50:'-'}</b> が50日線の上 ／ 過去1か月で上昇 <b>${up1}/${rows.length}</b>業種</p>`;
 dtl('a0',[['セクターETF終値',lastD]]);
 if(!ok.length){['a1','a3'].forEach(i=>hide(i,'この図に必要な履歴(約1年)が不足しています。'));$('a2').innerHTML='';return}
 const srt=[...ok].sort((a,b)=>a.m121-b.m121);
 Plotly.react('a1',[{type:'bar',orientation:'h',x:srt.map(r=>r.m121),y:srt.map(r=>r.name),marker:{color:srt.map(r=>r.m121>=0?'#2a9d8f':'#e76f51')},
  customdata:srt.map(r=>[r.K,fm(r.r6),fm(r.r3),fm(r.r1)]),hovertemplate:'%{y}<br>12-1か月 %{x:.1f}%(基準 %{customdata[0]}営業日前)<br>6か月 %{customdata[1]} / 3か月 %{customdata[2]} / 1か月 %{customdata[3]}<extra></extra>'}],
  {...L(),title:T(`① 12-1か月モメンタム(${st.mkt==='jp'?'日本':'米国'})`),xaxis:{title:{text:'騰落率 (%)'}},height:80+28*srt.length},CFG);
 dtl('a1',[['終値',lastD],['基準',ok[0].K<252?`履歴${ok[0].K}営業日(12か月に満たない)`:'約12か月前']]);
 const col=v=>v==null?'':(v>=0?'up':'dn'),mk=v=>v===true?'<span class="yes">○</span>':(v===false?'<span class="no">×</span>':'-');
 $('a2').innerHTML='<table class="h"><tr><th>順位</th><th>セクター</th><th>12-1M</th><th>6M</th><th>3M</th><th>1M</th><th>年率ボラ</th><th>勢い/ボラ</th><th>Zスコア</th><th>50日線</th><th>200日線</th></tr>'+
  [...ok].sort((a,b)=>b.m121-a.m121).map((r,i)=>`<tr><td>${i+1}</td><td>${r.name}</td><td class="${col(r.m121)}">${fm(r.m121)}</td><td class="${col(r.r6)}">${fm(r.r6)}</td><td class="${col(r.r3)}">${fm(r.r3)}</td><td class="${col(r.r1)}">${fm(r.r1)}</td><td>${r.vol==null?'-':r.vol.toFixed(1)+'%'}</td><td>${r.ra==null?'-':r.ra.toFixed(2)}</td><td class="${r.z==null?'':(r.z>=2?'dn':(r.z<=-2?'up':''))}">${r.z==null?'-':r.z.toFixed(2)}</td><td>${mk(r.a50)}</td><td>${mk(r.a200)}</td></tr>`).join('')+'</table>';
 dt('a2',`セクターETF終値 <b>${lastD||'-'}</b>`);
 const zs=rows.filter(r=>r.z!=null).sort((a,b)=>a.z-b.z);
 Plotly.react('a3',[{type:'bar',orientation:'h',x:zs.map(r=>r.z),y:zs.map(r=>r.name),marker:{color:zs.map(r=>Math.abs(r.z)>=2?'#e9a23b':(r.z>=0?'#2a9d8f':'#e76f51'))},
  hovertemplate:'%{y} Zスコア %{x:.2f}<extra></extra>'}],{...L(),title:T('③ Zスコア(60日平均からの乖離)'),xaxis:{title:{text:'標準偏差'}},height:80+28*zs.length,
  shapes:[2,-2].map(v=>({type:'line',x0:v,x1:v,yref:'paper',y0:0,y1:1,line:{dash:'dot',color:'#999'}}))},CFG);
 dt('a3',`セクターETF終値 <b>${lastD||'-'}</b>(橙=±2超)`);
}

function renderB(){
 const dts=MC.dates,v=mc('^VIX'),v3=mc('^VIX3M'),v9=mc('^VIX9D');
 if(!v){hide('b1','VIXを取得できませんでした。');hide('b2','VIXを取得できませんでした。');return}
 const a=sl(dts,v),t=[tr('VIX',a.x,a.y,{line:{color:'#264653',width:2.5}})];
 if(v3){const b=sl(dts,v3);t.push(tr('VIX3M',b.x,b.y,{line:{color:'#2a9d8f'}}))}
 if(v9){const b=sl(dts,v9);t.push(tr('VIX9D',b.x,b.y,{line:{color:'#e9a23b'}}))}
 let ratio=null;if(v3){ratio=dts.map((_,i)=>(v[i]!=null&&v3[i])?v[i]/v3[i]:null);const r=sl(dts,ratio);t.push(tr('VIX/VIX3M(右軸)',r.x,r.y,{yaxis:'y2',line:{color:'#e76f51',dash:'dot'},hovertemplate:'VIX/VIX3M %{y:.2f}<extra></extra>'}))}
 Plotly.react('b1',t,{...LN({y2:!!v3}),title:T('④ VIXの期間構造'),yaxis:{title:{text:'VIX'}},yaxis2:{overlaying:'y',side:'right',showgrid:false,tickformat:'.2f',range:[0.6,1.4]},
  shapes:v3?[{type:'line',xref:'paper',x0:0,x1:1,yref:'y2',y0:1,y1:1,line:{dash:'dash',color:'#e76f51'}}]:[]},CFG);
 const lv=lastOf(dts,v),lr=ratio?lastOf(dts,ratio):null;
 dtl('b1',[['VIX',lv&&lv.date],['VIX3M',v3?lastOf(dts,v3).date:null]]);
 const g=mc('^GSPC'),nk=mc('^N225'),t2=[tr('VIX',a.x,a.y,{line:{color:'#264653',width:2.5}})];
 if(g){const r=rvol(g),s=sl(dts,r);t2.push(tr('S&P500 実現ボラ(20日)',s.x,s.y,{line:{color:'#e76f51'}}))}
 if(nk){const r=rvol(nk),s=sl(dts,r);t2.push(tr('日経平均 実現ボラ(20日)',s.x,s.y,{line:{color:'#2a9d8f'}}))}
 Plotly.react('b2',t2,{...LN(),title:T('⑤ VIXと実現ボラティリティ'),yaxis:{title:{text:'年率 (%)'}}},CFG);
 dtl('b2',[['VIX',lv&&lv.date],['S&P500',g?lastOf(dts,g).date:null],['日経平均',nk?lastOf(dts,nk).date:null]]);
 return{vix:lv,ratio:lr,rv:g?lastOf(dts,rvol(g)):null}
}

function renderC(){
 const one=(id,title,s,note,lay={})=>{
  if(!s||!s.v.some(x=>x!=null)){hide(id,note+'(取得できませんでした)');return null}
  const a=sl(s.d,s.v);Plotly.react(id,[tr(title,a.x,a.y,{line:{color:'#264653',width:2.2},fill:lay.fill?'tozeroy':undefined,fillcolor:'rgba(38,70,83,0.08)'})],
   {...LN(),title:T(title),showlegend:false,height:M()?260:300,shapes:lay.zero?[{type:'line',xref:'paper',x0:0,x1:1,y0:0,y1:0,line:{dash:'dash',color:'#999'}}]:[]},CFG);
  const l=lastOf(s.d,s.v);dtl(id,[[s.src||'',l&&l.date]]);return l};
 const S=D.series,res={};
 res.curve=one('c1','⑥ 米10年−2年金利差(%pt)',S.T10Y2Y,'米財務省',{zero:true});
 res.hy=one('c2','⑦ 米ハイイールド・スプレッド(%)',S.HY,'FRED');
 res.real=one('c3','⑧ 米10年実質金利(%)',S.REAL10,'米財務省',{zero:true});
 const dts=MC.dates,dx=mc('DX-Y.NYB'),cu=mc('HG=F'),au=mc('GC=F');
 res.dx=one('c4','⑨ ドル指数',dx?{d:dts,v:dx,src:'ドル指数'}:null,'Yahoo Finance');
 res.cg=one('c5','⑩ 銅/金比率(×1000)',(cu&&au)?{d:dts,v:dts.map((_,i)=>(cu[i]!=null&&au[i])?cu[i]/au[i]*1000:null),src:'銅・金先物'}:null,'Yahoo Finance');
 return res}

function renderD(){
 const C=D.cot;if(!C||!C.JPY){$('cotbox').style.display='none';return}
 $('cotbox').style.display='';
 const j=C.JPY,c=cut(),mo=Math.max(st.months,12),cc=(()=>{const d=new Date(REF);d.setMonth(d.getMonth()-mo);return d.toISOString().slice(0,10)})();
 const x=[],y=[];j.d.forEach((d,i)=>{if(d>=cc){x.push(d);y.push(j.net[i])}});
 Plotly.react('d1',[{type:'bar',x,y,marker:{color:y.map(v=>v>=0?'#2a9d8f':'#e76f51')},hovertemplate:'%{x}<br>%{y:,.0f}枚<extra></extra>'}],
  {...LN(),title:T('⑪ 円先物の投機筋ポジション(買い−売り)'),showlegend:false,yaxis:{title:{text:'建玉(枚)'}}},CFG);
 const lastI=j.d.length-1,p=pctile(j.net,j.net[lastI]);
 dt('d1',`COT基準日 <b>${j.d[lastI]}</b>(公表は通常その週の金曜) ／ 直近 <b>${j.net[lastI]>=0?'買い越し':'売り越し'} ${Math.abs(j.net[lastI]).toLocaleString()}枚</b>(約2年のうち ${p==null?'-':p.toFixed(0)}パーセンタイル)`);
 const ks=Object.keys(C),vals=ks.map(k=>{const i=C[k].d.length-1;return{k,name:C[k].name,v:C[k].oi[i]?C[k].net[i]/C[k].oi[i]*100:null,d:C[k].d[i]}}).filter(o=>o.v!=null).sort((a,b)=>a.v-b.v);
 Plotly.react('d2',[{type:'bar',orientation:'h',x:vals.map(o=>o.v),y:vals.map(o=>o.name),marker:{color:vals.map(o=>o.v>=0?'#2a9d8f':'#e76f51')},hovertemplate:'%{y} %{x:.1f}%<extra></extra>'}],
  {...L(),title:T('⑫ 主要通貨の投機筋ポジション(建玉比)'),xaxis:{title:{text:'買い越し − 売り越し(建玉に対する %)'}},height:80+40*vals.length},CFG);
 dt('d2',`COT基準日 <b>${vals.length?vals[vals.length-1].d:'-'}</b>`);
 return{net:j.net[lastI],p,date:j.d[lastI]}}

function render(){
 $('asof').textContent=`最終更新 ${META.generated} ／ 最新データ ${REF}`;
 document.querySelectorAll('#mon button').forEach(b=>b.classList.toggle('on',+b.dataset.m===st.months));
 document.querySelectorAll('#mkt button').forEach(b=>b.classList.toggle('on',b.dataset.k===st.mkt));
 renderA();const b=renderB()||{},c=renderC(),d=renderD(),parts=[];
 if(b.vix)parts.push(`VIX <b>${b.vix.val.toFixed(1)}</b>${b.ratio?`(期先比 ${b.ratio.val.toFixed(2)}・${b.ratio.val>1?'<span class="no">期近が高い=ストレス</span>':'平常'})`:''}`);
 if(c.curve)parts.push(`米10年−2年 <b>${c.curve.val>=0?'+':''}${c.curve.val.toFixed(2)}%pt</b>${c.curve.val<0?'(逆イールド)':''}`);
 if(c.hy)parts.push(`HYスプレッド <b>${c.hy.val.toFixed(2)}%</b>`);
 if(c.real)parts.push(`米10年実質金利 <b>${c.real.val.toFixed(2)}%</b>`);
 if(d)parts.push(`円の投機筋 <b>${d.net>=0?'買い越し(円高方向)':'売り越し(円安方向)'} ${Math.abs(d.net).toLocaleString()}枚</b>(2年で${d.p==null?'-':d.p.toFixed(0)}%位・${d.date})`);
 $('sum').innerHTML=parts.length?parts.join(' ／ '):'取得できた指標がありません';
}
$('mk').textContent=META.label;
[[3,'3か月'],[6,'6か月'],[12,'1年'],[24,'2年']].forEach(([m,l])=>{const x=document.createElement('button');x.textContent=l;x.dataset.m=m;x.onclick=()=>{st.months=m;render()};$('mon').appendChild(x)});
[['jp','日本'],['us','米国']].forEach(([k,l])=>{const x=document.createElement('button');x.textContent=l;x.dataset.k=k;x.onclick=()=>{st.mkt=k;render()};$('mkt').appendChild(x)});
let wasM=M();addEventListener('resize',()=>{if(M()!==wasM){wasM=M();render()}});
render();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="合成データで実行")
    ap.add_argument("--nav", action="store_true", help="ページ間ナビを表示(公開用)")
    ap.add_argument("--offline", action="store_true", help="plotly.js をHTMLに埋め込む(CDN不要)")
    ap.add_argument("-o", "--output", default="pro.html")
    a = ap.parse_args()
    notes = []

    groups, rows = {}, []
    for key in ("jp", "us"):
        cfg = MARKETS[key]
        tick = [cfg["benchmark"][0], *cfg["sectors"]]
        close, _ = mf.load_prices(tick, demo=a.demo, period="2y")
        groups[key] = {"dates": [d.strftime("%Y-%m-%d") for d in close.index], "names": dict(cfg["sectors"]),
                       "close": {t: lst(close[t]) for t in tick if t in close.columns}}
        d, n = last_span(list(cfg["sectors"]))
        rows.append((f"セクターETF({'日本' if key == 'jp' else '米国'})", d, n))

    if a.demo:
        macro = demo_macro()
    else:
        macro, _ = mf.load_prices(list(MACRO), period="2y")
    mdates = [d.strftime("%Y-%m-%d") for d in macro.index]
    mclose = {t: lst(macro[t]) for t in MACRO if t in macro.columns and macro[t].notna().any()}
    d, n = last_span([t for t in MACRO if t in macro.columns])
    rows.append(("VIX・指数・ドル指数・銅/金", d, n))

    series = {}
    if a.demo:
        for k, s in demo_rates(macro.index).items():
            series[k] = s
        src = {"T10Y2Y": "合成データ", "HY": "合成データ", "REAL10": "合成データ"}
    else:
        src = {}
        nom = safe("米財務省(名目)", lambda: fetch_treasury("daily_treasury_yield_curve", {"T2": "2 YR", "T10": "10 YR"}), notes)
        if nom and "T2" in nom and "T10" in nom:
            series["T10Y2Y"] = (nom["T10"] - nom["T2"]).dropna()
            src["T10Y2Y"] = "米財務省"
        real = safe("米財務省(実質)", lambda: fetch_treasury("daily_treasury_real_yield_curve", {"R10": "10 YR"}), notes)
        if real and "R10" in real:
            series["REAL10"] = real["R10"]
            src["REAL10"] = "米財務省"
        hy = safe("ハイイールド・スプレッド(FRED)", lambda: parse_fred(_http(FRED_URL.format("BAMLH0A0HYM2"), timeout=20, tries=1)), notes)
        if hy is not None and len(hy):
            series["HY"] = hy
            src["HY"] = "FRED"
    sp = {}
    for k, s in series.items():
        s = _clean(s).tail(560)
        sp[k] = {"d": [x.strftime("%Y-%m-%d") for x in s.index], "v": [round(float(v), 4) for v in s.values], "src": src.get(k, "")}
        rows.append(({"T10Y2Y": "米10年−2年", "HY": "ハイイールド・スプレッド", "REAL10": "米10年実質金利"}[k], s.index[-1], f"出典: {src.get(k, '')}"))
    for k, nm in (("T10Y2Y", "米10年−2年"), ("REAL10", "米10年実質金利"), ("HY", "ハイイールド・スプレッド")):
        if k not in series:
            rows.append((nm, None, "取得できず(この図は非表示)"))

    cot = demo_cot() if a.demo else safe("COT(CFTC)", fetch_cot, notes)
    if cot:
        rows.append(("COT(投機筋ポジション)", pd.Timestamp(cot["JPY"]["d"][-1]) if "JPY" in cot else None,
                     "基準日は火曜、公表は通常その週の金曜"))
    else:
        rows.append(("COT(投機筋ポジション)", None, "取得できず(この図は非表示)"))

    all_dates = [mdates[-1]] + [g["dates"][-1] for g in groups.values()] + [v["d"][-1] for v in sp.values()]
    ref = max(all_dates)
    last = {t: date_str(mf.LAST_VALID.get(t)) for t in
            [*MACRO, *[t for g in groups.values() for t in g["close"]]]}
    payload = {"ref": ref, "groups": groups, "macro": {"dates": mdates, "close": mclose}, "series": sp, "cot": cot, "last": last}

    generated = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M JST")
    plotly_tag = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>'
    if a.offline:
        from plotly.offline import get_plotlyjs
        plotly_tag = f"<script>{get_plotlyjs()}</script>"
    meta = {"generated": generated, "label": "(DEMO)" if a.demo else ""}
    html = (PAGE.replace("__PLOTLY__", plotly_tag)
            .replace("__NAV__", nav_html("pro") if a.nav else "")
            .replace("__ASOF__", asof_html(rows, ref))
            .replace("__SOURCES__", sources_html(ref, generated, a.demo, notes))
            .replace("__META__", json.dumps(meta, ensure_ascii=False))
            .replace("__DATA__", json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")))
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    with open(a.output, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"取得できなかった項目: {notes or 'なし'}\n出力: {a.output}")


if __name__ == "__main__":
    main()
