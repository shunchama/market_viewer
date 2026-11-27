"""セクター資金フロー可視化ツール

どのセクターから資金が抜け、どのセクターに入っているかを1枚のHTMLに出力する。
HTML上で期間(週数)・終了日・フローの種類を切り替えられる(再計算はブラウザ内)。

  python market_flow.py                 # 日本(TOPIX-17 ETF)
  python market_flow.py --market us     # 米国(SPDRセクターETF)
  python market_flow.py --demo          # 合成データで動作確認(ネット不要)
  python market_flow.py --offline       # plotly.js を埋め込む(CDN不要)

フローの種類:
  実フロー = ETF発行口数の増減 x 終値(設定・解約による資金流出入の推定)
  近似     = 値上がり率 x 売買代金(口数データが無いときの代替)
口数の履歴は data/shares_history.csv に蓄積する(Yahoo Financeから取得。
欠けている分は手動でCSVに追記してもよい: date,ticker,shares)。
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys

for _pkg in ("yfinance", "pandas", "numpy"):
    try:
        __import__(_pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", _pkg])

import numpy as np
import pandas as pd

MARKETS = {
    "jp": {
        "benchmark": ("1306.T", "TOPIX(NEXT FUNDS TOPIX連動型ETF 1306)"),
        "source": "NEXT FUNDS TOPIX-17シリーズ(野村アセットマネジメント)のETF 1617〜1633 を各業種の代理として使用",
        "sectors": {
            "1617.T": "食品", "1618.T": "エネルギー資源", "1619.T": "建設・資材",
            "1620.T": "素材・化学", "1621.T": "医薬品", "1622.T": "自動車・輸送機",
            "1623.T": "鉄鋼・非鉄", "1624.T": "機械", "1625.T": "電機・精密",
            "1626.T": "情報通信・サービス", "1627.T": "電力・ガス", "1628.T": "運輸・物流",
            "1629.T": "商社・卸売", "1630.T": "小売", "1631.T": "銀行",
            "1632.T": "金融(除く銀行)", "1633.T": "不動産",
        },
    },
    "us": {
        "benchmark": ("SPY", "S&P500(SPDR S&P 500 ETF)"),
        "source": "Select Sector SPDR ETF(State Street)を各セクターの代理として使用",
        "sectors": {
            "XLK": "情報技術", "XLF": "金融", "XLV": "ヘルスケア", "XLY": "一般消費財",
            "XLP": "生活必需品", "XLE": "エネルギー", "XLI": "資本財", "XLB": "素材",
            "XLU": "公益", "XLRE": "不動産", "XLC": "通信",
        },
    },
}

# 個別銘柄 -> セクター名(上の sectors の値と一致させる)。足りない銘柄は追加してください。
STOCK_SECTOR = {
    "4755.T": "情報通信・サービス", "9432.T": "情報通信・サービス",
    "7201.T": "自動車・輸送機", "7012.T": "自動車・輸送機", "7014.T": "自動車・輸送機",
    "9501.T": "電力・ガス", "7011.T": "機械", "7013.T": "機械",
}


# ヒートマップのホバーに出す代表銘柄(各セクターの時価総額上位から手動選定。ETFの公式構成銘柄ではない)
REPS = {
    "jp": {
        "1617.T": [("2914.T", "JT"), ("2502.T", "アサヒ"), ("2802.T", "味の素")],
        "1618.T": [("1605.T", "INPEX"), ("5020.T", "ENEOS"), ("5019.T", "出光興産")],
        "1619.T": [("1925.T", "大和ハウス"), ("1812.T", "鹿島"), ("1802.T", "大林組")],
        "1620.T": [("4063.T", "信越化学"), ("3407.T", "旭化成"), ("4452.T", "花王")],
        "1621.T": [("4502.T", "武田薬品"), ("4568.T", "第一三共"), ("4519.T", "中外製薬")],
        "1622.T": [("7203.T", "トヨタ"), ("7267.T", "ホンダ"), ("6902.T", "デンソー")],
        "1623.T": [("5401.T", "日本製鉄"), ("5411.T", "JFE"), ("5713.T", "住友金属鉱山")],
        "1624.T": [("6301.T", "コマツ"), ("6367.T", "ダイキン"), ("7011.T", "三菱重工")],
        "1625.T": [("6758.T", "ソニーG"), ("8035.T", "東京エレクトロン"), ("6501.T", "日立")],
        "1626.T": [("9984.T", "ソフトバンクG"), ("9432.T", "NTT"), ("9433.T", "KDDI")],
        "1627.T": [("9501.T", "東京電力"), ("9503.T", "関西電力"), ("9531.T", "東京ガス")],
        "1628.T": [("9022.T", "JR東海"), ("9020.T", "JR東日本"), ("9101.T", "日本郵船")],
        "1629.T": [("8058.T", "三菱商事"), ("8001.T", "伊藤忠"), ("8031.T", "三井物産")],
        "1630.T": [("9983.T", "ファーストリテイリング"), ("3382.T", "セブン&アイ"), ("8267.T", "イオン")],
        "1631.T": [("8306.T", "三菱UFJ"), ("8316.T", "三井住友FG"), ("8411.T", "みずほFG")],
        "1632.T": [("8766.T", "東京海上"), ("8591.T", "オリックス"), ("8604.T", "野村HD")],
        "1633.T": [("8801.T", "三井不動産"), ("8802.T", "三菱地所"), ("8830.T", "住友不動産")],
    },
    "us": {
        "XLK": [("AAPL", "Apple"), ("MSFT", "Microsoft"), ("NVDA", "NVIDIA")],
        "XLF": [("BRK-B", "Berkshire"), ("JPM", "JPMorgan"), ("V", "Visa")],
        "XLV": [("LLY", "Eli Lilly"), ("UNH", "UnitedHealth"), ("JNJ", "J&J")],
        "XLY": [("AMZN", "Amazon"), ("TSLA", "Tesla"), ("HD", "Home Depot")],
        "XLP": [("PG", "P&G"), ("COST", "Costco"), ("WMT", "Walmart")],
        "XLE": [("XOM", "Exxon"), ("CVX", "Chevron"), ("COP", "ConocoPhillips")],
        "XLI": [("GE", "GE Aerospace"), ("CAT", "Caterpillar"), ("RTX", "RTX")],
        "XLB": [("LIN", "Linde"), ("SHW", "Sherwin-Williams"), ("FCX", "Freeport")],
        "XLU": [("NEE", "NextEra"), ("SO", "Southern"), ("DUK", "Duke")],
        "XLRE": [("PLD", "Prologis"), ("AMT", "American Tower"), ("EQIX", "Equinix")],
        "XLC": [("META", "Meta"), ("GOOGL", "Alphabet"), ("NFLX", "Netflix")],
    },
}


LAST_VALID = {}   # ティッカー -> 価格が最後に取得できた日(ffill 前の実データの日付)
SHARES_LAST = {}  # ティッカー -> 口数の最終観測日


def date_str(x):
    return None if x is None or pd.isna(x) else pd.Timestamp(x).strftime("%Y-%m-%d")


def asof_html(rows, ref):
    """データごとの基準日の表。rows = [(名前, 日付 or None, 備考)]。ref(ページの最新日)より古いものを強調する。"""
    ref = pd.Timestamp(ref).normalize()
    trs, short = [], []
    for label, d, note in rows:
        if d is None or pd.isna(d):
            trs.append(f"<tr><td>{label}</td><td class='stale'>データなし</td><td>{note}</td></tr>")
            short.append(f"{label.split('(')[0]} なし")
            continue
        d = pd.Timestamp(d).normalize()
        lag = (ref - d).days
        late = f" <span class='stale'>({lag}日前)</span>" if lag > 0 else ""
        cls = " class='stale'" if lag > 3 else ""
        trs.append(f"<tr><td>{label}</td><td{cls}>{d:%Y-%m-%d}{late}</td><td>{note}</td></tr>")
        short.append(f"{label.split('(')[0]} {d:%m/%d}")
    return ("<style>.asof{font-size:13px;margin:4px 0 8px}.asof summary{cursor:pointer;color:#555}"
            ".asof table{border-collapse:collapse;margin-top:6px;width:100%}.asof td,.asof th{padding:4px 8px;"
            "border-bottom:1px solid #eee;text-align:left}.asof th{background:#eef1f5}.stale{color:#b5503a;font-weight:600}</style>"
            f"<details class='asof'><summary>データ基準日: {' ／ '.join(short)}(タップで詳細)</summary>"
            "<table><tr><th>データ</th><th>最新日</th><th>備考</th></tr>" + "".join(trs) + "</table></details>")


def last_span(tickers):
    """複数ティッカーの最終取得日の (最新, 備考)。遅れている銘柄があれば備考に挙げる。"""
    ds = {t: LAST_VALID.get(t) for t in tickers if LAST_VALID.get(t) is not None}
    if not ds:
        return None, ""
    mx = max(ds.values())
    late = [t for t, d in ds.items() if pd.Timestamp(d).normalize() < pd.Timestamp(mx).normalize()]
    note = f"全{len(ds)}本が同日" if not late else f"{len(late)}本が遅れ: " + ", ".join(
        f"{t}({pd.Timestamp(ds[t]):%m/%d})" for t in late[:4]) + ("…" if len(late) > 4 else "")
    return mx, note


def nav_html(active):
    """ページ間ナビ(index.html=日本株, us.html=米国株, fx.html=為替)。公開フォルダ内の相対リンク。"""
    items = (("index.html", "株式(日本)", "jp"), ("us.html", "株式(米国)", "us"), ("fx.html", "為替", "fx"), ("analysis.html", "分析", "analysis"))
    links = "".join(f'<a href="{h}"{" class=on" if k == active else ""}>{n}</a>' for h, n, k in items)
    return ("<style>.nav{display:flex;gap:6px;padding-top:14px}.nav a{padding:5px 14px;border:1px solid #bbb;"
            "border-radius:999px;text-decoration:none;color:#264653;font-size:14px}"
            ".nav a.on{background:#264653;color:#fff;border-color:#264653}</style>"
            f'<nav class="nav">{links}</nav>')


def notify(text):
    """SLACK_WEBHOOK_URL または DISCORD_WEBHOOK_URL に要約を送る(未設定なら何もしない)。"""
    import urllib.request
    for env, key in (("SLACK_WEBHOOK_URL", "text"), ("DISCORD_WEBHOOK_URL", "content")):
        url = os.environ.get(env)
        if url:
            req = urllib.request.Request(url, json.dumps({key: text}).encode(),
                                         {"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=15)
            print(f"通知送信: {env}")


# ───────────────────────── データ取得 ─────────────────────────
def load_prices(tickers, demo=False, period="1y"):
    """(close, volume) の DataFrame を返す。"""
    if demo:
        rng = np.random.default_rng(7)
        idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=260)
        n = len(tickers)
        market = rng.normal(0.0003, 0.008, len(idx))
        drift = rng.normal(0, 0.0012, n)
        drift[: n // 3] += 0.0015  # 一部のセクターに資金が集まる
        drift[-n // 3:] -= 0.0015  # 一部から抜ける
        ret = market[:, None] + drift + rng.normal(0, 0.007, (len(idx), n))
        close = pd.DataFrame(100 * np.exp(ret.cumsum(0)), index=idx, columns=tickers)
        vol = pd.DataFrame(rng.lognormal(12, 0.3, (len(idx), n)) * (1 + 3 * np.abs(ret)),
                           index=idx, columns=tickers)
        close.attrs["drift"] = dict(zip(tickers, drift))
        LAST_VALID.update({t: idx[-1] for t in tickers})
        return close, vol
    import time
    import yfinance as yf
    # threads=False: 並列取得だとキャッシュDBが "database is locked" になることがある
    df = yf.download(list(tickers), period=period, auto_adjust=True, progress=False, threads=False)
    close = df["Close"].copy()
    vol = df["Volume"].copy()
    # 取得に失敗した銘柄は1本ずつリトライ
    for t in tickers:
        for attempt in range(3):
            if t in close.columns and close[t].notna().any():
                break
            time.sleep(1 + attempt)
            one = yf.download(t, period=period, auto_adjust=True, progress=False, threads=False)
            if not one.empty:
                c, v = one["Close"], one["Volume"]
                close[t] = (c.iloc[:, 0] if hasattr(c, "columns") else c).reindex(close.index)
                vol[t] = (v.iloc[:, 0] if hasattr(v, "columns") else v).reindex(close.index)
    for t in close.columns:  # 補完(ffill)する前に、実際に値があった最終日を記録
        LAST_VALID[t] = close[t].last_valid_index()
    close = close.dropna(how="all").ffill()
    vol = vol.reindex(close.index).fillna(0)
    missing = [t for t in tickers if t not in close.columns or close[t].isna().all()]
    if missing:
        print(f"[warn] データ取得失敗: {missing}")
    return close, vol


def load_shares(tickers, index, store_path, demo=False, close=None):
    """ETF発行口数(日次)を DataFrame で返す。取れない/変化の無い銘柄は列ごと NaN。

    取得元: Yahoo Finance (yfinance の get_shares_full と、実行日のスナップショット)。
    履歴は store_path のCSV(date,ticker,shares)に蓄積し、次回以降も使う。
    """
    out = pd.DataFrame(np.nan, index=index, columns=list(tickers))
    if demo:
        rng = np.random.default_rng(11)
        drift = close.attrs.get("drift", {})
        for k, t in enumerate(tickers):
            if k % 6 == 5:  # 一部は口数データ無し(カバレッジ不足の再現)
                continue
            step = rng.normal(0, 0.002, len(index)) + drift.get(t, 0) * 0.5
            out[t] = 1e6 * np.exp(step.cumsum())
            SHARES_LAST[t] = index[-1]
        return out

    import yfinance as yf
    store = pd.DataFrame(columns=["date", "ticker", "shares"])
    if os.path.exists(store_path):
        store = pd.read_csv(store_path, dtype={"ticker": str})
    new_rows = []
    today = pd.Timestamp.today().normalize()
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            s = tk.get_shares_full(start=(index[0] - pd.Timedelta(days=7)).strftime("%Y-%m-%d"))
            if s is not None and len(s):
                s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
                s = s[~s.index.duplicated(keep="last")]
                new_rows += [(d.strftime("%Y-%m-%d"), t, float(v)) for d, v in s.items()]
            else:  # 履歴が無ければ今日のスナップショットだけ記録
                snap = tk.info.get("sharesOutstanding") or tk.info.get("impliedSharesOutstanding")
                if snap:
                    new_rows.append((today.strftime("%Y-%m-%d"), t, float(snap)))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 口数取得失敗 {t}: {type(e).__name__}")
    if new_rows:
        store = pd.concat([store, pd.DataFrame(new_rows, columns=store.columns)])
    store = (store.drop_duplicates(["date", "ticker"], keep="last")
             .sort_values(["ticker", "date"]))
    if len(store):
        os.makedirs(os.path.dirname(os.path.abspath(store_path)), exist_ok=True)
        store.to_csv(store_path, index=False)
    store["date"] = pd.to_datetime(store["date"])
    for t in tickers:
        s = store[store["ticker"] == t].set_index("date")["shares"].astype(float)
        if s.nunique() > 1:  # 一度も変化していない口数は「データ無し」とみなす
            out[t] = s.reindex(index.union(s.index)).ffill().reindex(index)
            SHARES_LAST[t] = s.index.max()
    return out


# ───────────────────────── 集計(通知・検証用) ─────────────────────────
def flows(close, vol, shares, sectors, window):
    """各セクターのフロー。proxy=値上がり率x売買代金, actual=口数増減x終値。単位: 通貨"""
    tick = [t for t in sectors if t in close.columns]
    proxy = (close[tick].pct_change() * close[tick] * vol[tick]).tail(window).sum()
    actual = (shares[tick].diff() * close[tick]).tail(window).sum(min_count=1)
    actual = actual.where(shares[tick].notna().any())
    return pd.DataFrame({"name": [sectors[t] for t in tick], "proxy": proxy, "actual": actual})


def summary(fl):
    mode = "actual" if fl["actual"].notna().mean() >= 0.7 else "proxy"
    col = fl[mode].dropna().sort_values()
    names = fl["name"]
    return mode, [names[t] for t in col.index[::-1][:3]], [names[t] for t in col.index[:3]]


# ───────────────────────── HTML ─────────────────────────
def lst(series):
    return [None if pd.isna(x) else round(float(x), 4) for x in series]


def build_payload(close, vol, shares, sectors, bench, groups, reps):
    cols = [t for t in [bench, *sectors] if t in close.columns]
    holdings = []
    for g, tickers in groups.items():
        for t in tickers:
            if t in close.columns and close[t].notna().sum() > 64:
                cols.append(t)
                holdings.append({"g": g, "t": t, "sector": STOCK_SECTOR.get(t, "未分類")})
    cols = list(dict.fromkeys(cols))
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in close.index],
        "bench": bench,
        "sectors": [{"t": t, "name": n} for t, n in sectors.items() if t in close.columns],
        "holdings": holdings,
        "close": {t: lst(close[t]) for t in cols},
        "vol": {t: lst(vol[t]) for t in cols},
        "reps": {st: [{"t": t, "name": n} for t, n in lst_ if t in close.columns]
                 for st, lst_ in reps.items()},
        "last": {t: date_str(LAST_VALID.get(t)) for t in
                 dict.fromkeys(cols + [t for lst_ in reps.values() for t, _ in lst_])},
        "slast": {t: date_str(SHARES_LAST.get(t)) for t in sectors if SHARES_LAST.get(t) is not None},
        "rclose": {t: lst(close[t]) for lst_ in reps.values() for t, _ in lst_ if t in close.columns},
        "shares": {t: (lst(shares[t]) if t in shares.columns and shares[t].notna().any() else None)
                   for t in sectors if t in close.columns},
    }


def sources_html(cfg, bench_name, asof, generated, demo, sectors):
    rows = "".join(f"<tr><td>{t}</td><td>{n}</td></tr>" for t, n in sectors.items())
    demo_note = ("<p class='warn'>※ DEMO: 合成の乱数データです。実際の市場データではありません。</p>"
                 if demo else "")
    return f"""
<details class="src" open><summary>出典・計算方法・注意事項</summary>{demo_note}
<h4>データの出典</h4>
<ul>
<li><b>株価・売買高</b>: Yahoo Finance(Pythonライブラリ <code>yfinance</code> 経由)。日足、分割・配当調整済みの終値、直近約1年。無料データのため遅延・欠損・訂正があり得ます。</li>
<li><b>セクター</b>: {cfg['source']}。</li>
<li><b>代表銘柄</b>(ヒートマップのホバー表示): 各セクターの時価総額上位から手動で3銘柄ずつ選定(<code>market_flow.py</code> の <code>REPS</code>)。ETFの公式な構成銘柄・比率ではありません。株価はYahoo Finance。</li>
<li><b>ベンチマーク</b>: {bench_name}。</li>
<li><b>ETF発行口数</b>: Yahoo Finance(<code>get_shares_full</code> と、実行日ごとのスナップショット)。リポジトリの <code>data/shares_history.csv</code> に蓄積。ETFの口数がYahooで未提供・更新が遅い場合は「近似」に自動で切り替わります。公式値(運用会社・取引所の公表)で補うときはこのCSVに <code>date,ticker,shares</code> で追記してください。</li>
<li><b>最新データ日</b>: {asof} ／ <b>ページ生成</b>: {generated}</li>
</ul>
<h4>計算方法</h4>
<ul>
<li><b>実フロー</b> = Σ(各日の発行口数の増減 × その日の終値)。設定(口数増)=流入、解約(口数減)=流出の推定。基準価額ではなく終値で評価するため誤差があります。</li>
<li><b>近似フロー</b> = Σ(各日の値上がり率 × 売買代金)、売買代金 = 終値×出来高。値動きと売買の勢いからの推定で、実際の資金流出入ではありません。</li>
<li><b>フロー%</b> = 各セクターのフロー ÷ 全セクターの|フロー|合計。</li>
<li><b>サンキー図</b> = 流出額を流入セクターの大きさで按分した模式図。「AからBへ移った」という事実を示すものではありません。</li>
<li><b>RRG</b> = 相対強度(セクター÷ベンチマーク、60日平均=100)とその10日モメンタム。</li>
<li><b>保有銘柄のセクター</b>は <code>market_flow.py</code> の <code>STOCK_SECTOR</code> で手動指定しています。</li>
</ul>
<details><summary>対象ETF一覧</summary><table class="mini"><tr><th>コード</th><th>セクター</th></tr>{rows}</table></details>
<p class="warn">本ページは情報提供のみを目的とし、投資助言ではありません。</p>
</details>"""


PAGE = r"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sector Flow</title>
__PLOTLY__
<style>
body{font-family:-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif;max-width:1040px;margin:auto;padding:0 12px 60px;color:#222}
h2{margin:18px 0 4px}.sub{color:#666;font-size:13px;margin:0 0 10px}
.ctl{position:sticky;top:0;z-index:5;background:#fffffff2;backdrop-filter:blur(4px);padding:10px 0;border-bottom:1px solid #ddd;display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.ctl button,.ctl select,.ctl input{font:inherit;padding:5px 10px;border:1px solid #bbb;border-radius:6px;background:#fff;cursor:pointer}
.ctl button.on{background:#264653;color:#fff;border-color:#264653}
.ctl label{font-size:12px;color:#666}.ctl input[type=number]{width:56px}
.sum{font-size:15px;margin:12px 0 2px}.note{color:#777;font-size:12px;margin:0 0 6px}
.chart{width:100%}.dt{font-size:12px;color:#5a6b7a;margin:2px 0 0}hr.sep{border:0;border-top:2px solid #cfd6df;margin:30px 0 14px}table.h,table.mini{border-collapse:collapse;font-size:13px;width:100%}
table.h th,table.h td,table.mini th,table.mini td{padding:6px 8px;border-bottom:1px solid #eee;text-align:left}
table.h th,table.mini th{background:#e8edf3}.up{background:#d8f3dc}.dn{background:#fde2dd}
.src{margin-top:34px;font-size:13px;line-height:1.7;background:#f7f8fa;padding:10px 16px;border-radius:8px}
.src summary{cursor:pointer;font-weight:600}.src h4{margin:12px 0 2px}.warn{color:#b5503a}
code{background:#eceff3;padding:1px 4px;border-radius:3px}

.tw{overflow-x:auto}
@media (max-width:640px){
 body{padding:0 8px 40px}h2{font-size:20px}
 .ctl{position:static;gap:5px}.ctl button,.ctl select,.ctl input{padding:4px 8px;font-size:13px}
 .sum{font-size:14px}table.h{font-size:12px}table.h th,table.h td{padding:5px 6px}
 table.h td:nth-child(3){min-width:84px}.src{padding:8px 12px;font-size:12.5px}
}
</style></head><body>
__NAV__
<h2>セクター資金フロー <small id="mk"></small></h2>
<p class="sub" id="asof"></p>
__ASOF__
<div class="ctl">
 <label>期間</label>
 <span id="presets"></span>
 <input type="number" id="wk" min="1" max="26" title="任意の週数"><label>週</label>
 <label>終了日</label><button id="prev">◀ 前週</button><input type="date" id="end"><button id="next">次週 ▶</button><button id="latest">最新</button>
 <label>フロー</label>
 <select id="mode"><option value="auto">自動</option><option value="actual">実フロー(口数×価格)</option><option value="proxy">近似(値動き×売買代金)</option></select>
</div>
<p class="sum" id="sum"></p><p class="note" id="modenote"></p>
<hr class="sep">
<p class="note">① 騰落率は終了日基準(1日=前日比、1週=5営業日、1ヶ月=21営業日、3ヶ月=63営業日)。マスにマウスを乗せると、代表銘柄の変動率と変動幅(円/ドル)を表示します。代表銘柄は各セクターの時価総額上位から手動で選んだもので、ETFの公式な構成銘柄ではありません。</p><div id="c1" class="chart"></div>
<hr class="sep">
<p class="note" id="n2"></p><div id="c2" class="chart"></div>
<hr class="sep">
<p class="note">③ 流出額を流入セクターの大きさで按分した模式図です。実際の資金の行き先を示すものではありません。</p><div id="c3" class="chart"></div>
<hr class="sep">
<p class="note">④ 大きい点が終了日、線は直近8営業日の軌跡。時計回り(Improving→Leading→Weakening→Lagging)に循環するとされます。</p><div id="c4" class="chart"></div>
<hr class="sep">
<div id="hold" style="display:none"><h3 style="font-weight:400">⑤ 保有・監視銘柄</h3><div id="c5"></div></div>
<hr class="sep">
<p class="note" id="n6"></p><div id="c6" class="chart"></div>
__SOURCES__
<script type="application/json" id="data">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
const META=__META__;
const N=D.dates.length, MIN_END=Math.min(80,N-1), secs=D.sectors, C=D.close, V=D.vol, S=D.shares, B=D.bench;
const st={end:N-1,weeks:META.weeks,mode:'auto'};
const $=id=>document.getElementById(id);
const pct=(a,b)=>(a==null||b==null||b===0)?null:(a/b-1)*100;
const fmt=(x,d=1)=>x==null?'-':(x>=0?'+':'')+x.toFixed(d)+'%';
const oku=x=>(x/1e8).toFixed(1)+(META.cur==='JPY'?'億円':'億ドル');
const M=()=>window.innerWidth<640,CFG={displayModeBar:false,responsive:true};
const FONT='-apple-system,Hiragino Sans,Noto Sans JP,sans-serif';
const L=()=>M()?{font:{family:FONT,size:11},margin:{l:112,r:10,t:44,b:44}}:{font:{family:FONT},margin:{l:140,r:30,t:50,b:50}};
const T=s=>({text:s,font:{size:M()?14:17}});
const dt=(id,html)=>{let el=document.getElementById('dt_'+id);if(!el){el=document.createElement('p');el.className='dt';el.id='dt_'+id;document.getElementById(id).insertAdjacentElement('beforebegin',el)}el.innerHTML='📅 '+html};
const chip=(ts,end)=>{const LV=D.last||{},ds=ts.map(t=>LV[t]).filter(Boolean);if(!ds.length)return '不明';
 const mx=ds.reduce((a,b)=>a>b?a:b),ref=mx<end?mx:end,late=ts.filter(t=>LV[t]&&LV[t]<ref);
 return `<b>${ref}</b>`+(late.length?` <span class="stale">(${late.length}本は${[...new Set(late.map(t=>LV[t].slice(5)))].join('・')}が最新)</span>`:'')};

function proxyFlow(t,a,b){let s=0;for(let i=a+1;i<=b;i++){const c=C[t][i],p=C[t][i-1],v=V[t][i];if(c==null||p==null||v==null)continue;s+=(c/p-1)*c*v}return s}
function actualFlow(t,a,b){const s=S[t];if(!s)return null;let x=0,n=0;for(let i=a+1;i<=b;i++){const c=C[t][i];if(s[i]==null||s[i-1]==null||c==null)continue;x+=(s[i]-s[i-1])*c;n++}return n?x:null}
function dayFlow(t,i,mode){const c=C[t][i];if(c==null)return null;
 if(mode==='actual'){const s=S[t];return(s&&s[i]!=null&&s[i-1]!=null)?(s[i]-s[i-1])*c:null}
 const p=C[t][i-1],v=V[t][i];return(p==null||v==null)?null:(c/p-1)*c*v}
function effMode(a,e){if(st.mode!=='auto')return st.mode;
 const n=secs.filter(x=>actualFlow(x.t,a,e)!==null).length;return n>=secs.length*0.7?'actual':'proxy'}
const flowOf=(t,a,e,m)=>m==='actual'?actualFlow(t,a,e):proxyFlow(t,a,e);

function metrics(){
 const e=st.end,w=Math.min(st.weeks*5,e),a=e-w,mode=effMode(a,e);
 const rows=secs.map(x=>{const t=x.t,c=C[t];
  let tw=0,tl=0,nw=0,nl=0;for(let i=Math.max(1,e-59);i<=e;i++){const q=c[i]*V[t][i];if(c[i]==null)continue;tl+=q;nl++;if(i>e-w){tw+=q;nw++}}
  return{t,name:x.name,r1:pct(c[e],c[e-1]),r5:pct(c[e],c[e-5]),r21:pct(c[e],c[e-21]),r63:pct(c[e],c[e-63]),
   flow:flowOf(t,a,e,mode),tr:(nw&&nl&&tl)?(tw/nw)/(tl/nl):null}});
 const ok=rows.filter(r=>r.flow!=null),tot=ok.reduce((s,r)=>s+Math.abs(r.flow),0)||1;
 rows.forEach(r=>r.pct=r.flow==null?null:r.flow/tot*100);
 return{rows,mode,a,e,w,missing:rows.length-ok.length};
}

function render(){
 const m=metrics(),{rows,mode,a,e,w}=m,unit=mode==='actual'?'実フロー(口数増減×終値)':'近似フロー(値上がり率×売買代金)';
 const rng=`${D.dates[a]} → ${D.dates[e]}(${w}営業日)`;
 $('asof').textContent=`最終更新 ${META.generated} ／ 最新データ ${D.dates[N-1]} ／ 表示中の終了日 ${D.dates[e]}`;
 const srt=rows.filter(r=>r.pct!=null).sort((x,y)=>y.pct-x.pct);
 $('sum').innerHTML=srt.length?`流入上位: <b>${srt.slice(0,3).map(r=>r.name).join(' / ')}</b>　流出上位: <b>${srt.slice(-3).reverse().map(r=>r.name).join(' / ')}</b>`:'フローを計算できるデータがありません';
 $('modenote').textContent=`集計期間 ${rng}｜${unit}`+(m.missing?`｜口数データが無い${m.missing}業種は除外`:'')+(mode==='proxy'&&st.mode==='auto'?'｜口数データが不足しているため近似を使用中':'');
 document.querySelectorAll('#presets button').forEach(b=>b.classList.toggle('on',+b.dataset.w===st.weeks));
 $('wk').value=st.weeks;$('end').value=D.dates[e];$('mode').value=st.mode;

 // ① heatmap
 const h=[...rows].sort((x,y)=>(y.r5??-1e9)-(x.r5??-1e9)),cols=['1日','1週','1ヶ月','3ヶ月'];
 const z=h.map(r=>[r.r1,r.r5,r.r21,r.r63]),lim=Math.max(1,...z.flat().filter(v=>v!=null).map(Math.abs));
 const PN=[1,5,21,63],money=v=>META.cur==='JPY'?(v>=0?'+':'-')+Math.abs(Math.round(v)).toLocaleString()+'円':(v>=0?'+':'-')+'$'+Math.abs(v).toFixed(2);
 const hv=h.map((row,i)=>PN.map((n,k)=>{const lines=(D.reps[row.t]||[]).map(r=>{const c=D.rclose[r.t];if(!c)return null;const x=c[e],y=c[e-n];
   return(x==null||y==null||!y)?null:`${r.name}  ${fmt((x/y-1)*100,2)}(${money(x-y)})`}).filter(Boolean);
  return`<b>${row.name}</b>  ${cols[k]}: ${fmt(z[i][k],2)}`+(lines.length?'<br>代表銘柄:<br>'+lines.join('<br>'):'')}));
 Plotly.react('c1',[{type:'heatmap',z,x:cols,y:h.map(r=>r.name),zmid:0,zmin:-lim,zmax:lim,showscale:!M(),textfont:{size:M()?9:11},colorscale:[[0,'#b2182b'],[0.25,'#f4a582'],[0.5,'#ffffbf'],[0.75,'#a6d96a'],[1,'#1a9850']],
  text:z.map(rw=>rw.map(v=>v==null?'':v.toFixed(1)+'%')),texttemplate:'%{text}',hovertext:hv,hovertemplate:'%{hovertext}<extra></extra>'}],
  {...L(),title:T('① セクター別騰落率'),height:80+28*h.length,yaxis:{autorange:'reversed'}},CFG);

 // ② flow bars
 const f=[...srt].reverse();
 Plotly.react('c2',[{type:'bar',orientation:'h',x:f.map(r=>r.pct),y:f.map(r=>r.name),
  marker:{color:f.map(r=>r.pct>=0?'#2a9d8f':'#e76f51')},
  customdata:f.map(r=>[oku(r.flow),r.tr==null?'-':r.tr.toFixed(2)]),
  hovertemplate:'%{y}<br>フロー %{x:.1f}%(%{customdata[0]})<br>売買代金 平常比 x%{customdata[1]}<extra></extra>'}],
  {...L(),title:T('② 資金フロー(右=流入 / 左=流出)'),xaxis:{title:{text:M()?'フロー比率 (%)':'全セクターの|フロー|合計に対する比率 (%)'}},height:80+28*f.length},CFG);
 $('n2').textContent=`② ${unit}。${rng}。`+(mode==='actual'?'設定・解約の推定のため、現物拠出など実際の資金移動と差が出ます。':'実際のファンド資金流出入ではありません。');

 // ③ sankey
 const outs=srt.filter(r=>r.flow<0),ins=srt.filter(r=>r.flow>0);
 if(outs.length&&ins.length){const tin=ins.reduce((s,r)=>s+r.flow,0),src=[],dst=[],val=[];
  outs.forEach((o,i)=>ins.forEach((p,j)=>{src.push(i);dst.push(outs.length+j);val.push(Math.abs(o.flow)*p.flow/tin)}));
  Plotly.react('c3',[{type:'sankey',node:{label:[...outs.map(r=>r.name+'(流出)'),...ins.map(r=>r.name+'(流入)')],
   color:[...outs.map(()=>'#e76f51'),...ins.map(()=>'#2a9d8f')],pad:12},
   link:{source:src,target:dst,value:val,color:val.map(()=> 'rgba(231,111,81,0.25)')}}],
   {...L(),title:T('③ 資金の移動イメージ(流出 → 流入)'),height:M()?760:520,margin:{l:6,r:6,t:44,b:10}},CFG);
 }else Plotly.react('c3',[],{...L(),title:T('③ 資金の移動イメージ(流出または流入のセクターがありません)'),height:120});

 // ④ RRG
 const rs=(t,i)=>C[t][i]/C[B][i],rsn=(t,i)=>{if(i<59)return null;let s=0;for(let k=i-59;k<=i;k++)s+=rs(t,k);return rs(t,i)/(s/60)*100},
  mom=(t,i)=>{const p=rsn(t,i),q=rsn(t,i-10);return(p==null||q==null)?null:p/q*100};
 const tr=secs.map(x=>{const xs=[],ys=[];for(let i=e-7;i<=e;i++){const p=rsn(x.t,i),q=mom(x.t,i);if(p!=null&&q!=null){xs.push(p);ys.push(q)}}
  return{type:'scatter',mode:'lines+markers',name:x.name,x:xs,y:ys,marker:{size:xs.map((_,k)=>k===xs.length-1?11:4)},
   hovertemplate:x.name+'<br>相対強度 %{x:.1f}<br>モメンタム %{y:.1f}<extra></extra>'}});
 const q=(x,y,t)=>({xref:'paper',yref:'paper',x,y,text:M()?t.split(' ')[0]:t,showarrow:false,xanchor:x?'right':'left',yanchor:y?'top':'bottom',font:{size:M()?10:13,color:'gray'}});
 Plotly.react('c4',tr,{...L(),title:T('④ セクターローテーション(RRG)'),height:M()?660:620,margin:M()?{l:44,r:8,t:44,b:50}:{l:60,r:20,t:50,b:50},legend:M()?{orientation:'h',y:-0.2,font:{size:10}}:{},
  xaxis:{title:{text:'ベンチマーク比 相対強度'}},yaxis:{title:{text:'相対強度のモメンタム'}},
  shapes:[{type:'line',xref:'paper',x0:0,x1:1,yref:'y',y0:100,y1:100,line:{dash:'dot'}},{type:'line',yref:'paper',y0:0,y1:1,xref:'x',x0:100,x1:100,line:{dash:'dot'}}],
  annotations:[q(1,1,'Leading 先導'),q(0,1,'Improving 改善'),q(0,0,'Lagging 出遅れ'),q(1,0,'Weakening 失速')]},CFG);

 // ⑤ holdings
 if(D.holdings.length){$('hold').style.display='';
  const cls=v=>v==null?'':(v>=0?'up':'dn'),byName=Object.fromEntries(rows.map(r=>[r.name,r]));
  $('c5').innerHTML='<div class="tw"><table class="h"><tr><th>区分</th><th>銘柄</th><th>セクター(フロー%)</th><th>1日</th><th>1週</th><th>1ヶ月</th></tr>'+
   D.holdings.map(x=>{const c=C[x.t],r=byName[x.sector],sec=r?`${x.sector} (${fmt(r.pct)})`:x.sector,
    p=[pct(c[e],c[e-1]),pct(c[e],c[e-5]),pct(c[e],c[e-21])];
    return`<tr><td>${x.g}</td><td>${x.t}</td><td>${sec}</td>`+p.map(v=>`<td class="${cls(v)}">${fmt(v,2)}</td>`).join('')+'</tr>'}).join('')+'</table></div>'}

 // ⑥ cumulative
 const s0=Math.max(1,e-59),ser=secs.map(x=>{let c=0;const ys=[];for(let i=s0;i<=e;i++){const d=dayFlow(x.t,i,mode);c+=d||0;ys.push(c)}
  const any=(()=>{for(let i=s0;i<=e;i++)if(dayFlow(x.t,i,mode)!=null)return true;return false})();return{x,ys,any}}).filter(o=>o.any);
 let sc=0;ser.forEach(o=>{for(let i=s0;i<=e;i++)sc+=Math.abs(dayFlow(o.x.t,i,mode)||0)});sc=sc||1;
 ser.sort((p,q)=>q.ys[q.ys.length-1]-p.ys[p.ys.length-1]);
 Plotly.react('c6',ser.map(o=>({type:'scatter',mode:'lines',name:o.x.name,x:D.dates.slice(s0,e+1),y:o.ys.map(v=>v/sc*100),
  hovertemplate:o.x.name+' %{y:.1f}%<extra></extra>'})),
  {...L(),title:T('⑥ 累積フロー推移(直近60営業日)'),yaxis:{title:{text:M()?'累積フロー (%)':'累積フロー (全体の|フロー|に対する %)'}},height:M()?600:520,margin:M()?{l:44,r:8,t:44,b:50}:{l:60,r:20,t:50,b:50},legend:M()?{orientation:'h',y:-0.2,font:{size:10}}:{}},CFG);
 $('n6').textContent=`⑥ ${unit}。上昇が続くセクターは資金流入が継続しています。`;
 const end=D.dates[e],secT=secs.map(x=>x.t),repT=Object.keys(D.rclose||{}),SL=Object.values(D.slast||{}).filter(Boolean).sort();
 const flowTxt=mode==='actual'?`ETF口数 <b>${SL.length?SL[SL.length-1]:'-'}</b>(最終観測日)・ETF終値 ${chip(secT,end)}`:`ETF終値・売買代金 ${chip(secT,end)}`;
 dt('c1',`ETF終値 ${chip(secT,end)} ／ 代表銘柄 ${chip(repT,end)}`);dt('c2',flowTxt);dt('c3',flowTxt);
 dt('c4',`ETF終値(ベンチマーク含む) ${chip([...secT,B],end)}`);
 if(D.holdings.length)dt('c5',`保有・監視銘柄の終値 ${chip(D.holdings.map(x=>x.t),end)}`);
 dt('c6',flowTxt);
}

// ── controls ──
$('mk').textContent=META.label;
[[1,'1週'],[2,'2週'],[4,'4週'],[13,'13週']].forEach(([w,l])=>{const b=document.createElement('button');b.textContent=l;b.dataset.w=w;
 b.onclick=()=>{st.weeks=w;render()};$('presets').appendChild(b)});
$('wk').onchange=ev=>{st.weeks=Math.max(1,Math.min(26,+ev.target.value||1));render()};
$('mode').onchange=ev=>{st.mode=ev.target.value;render()};
$('end').min=D.dates[MIN_END];$('end').max=D.dates[N-1];
$('end').onchange=ev=>{let i=D.dates.length-1;while(i>MIN_END&&D.dates[i]>ev.target.value)i--;st.end=i;render()};
$('prev').onclick=()=>{st.end=Math.max(MIN_END,st.end-5);render()};
$('next').onclick=()=>{st.end=Math.min(N-1,st.end+5);render()};
$('latest').onclick=()=>{st.end=N-1;render()};
let wasM=M();addEventListener('resize',()=>{if(M()!==wasM){wasM=M();render()}});
render();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market", choices=MARKETS, default="jp")
    ap.add_argument("--window", type=int, default=5, help="初期表示の集計期間(営業日。5=1週)")
    ap.add_argument("--demo", action="store_true", help="合成データで実行")
    ap.add_argument("--no-holdings", action="store_true", help="保有・監視銘柄の表を出さない")
    ap.add_argument("--notify", action="store_true", help="要約を Slack/Discord Webhook に送る")
    ap.add_argument("--offline", action="store_true", help="plotly.js をHTMLに埋め込む(CDN不要)")
    ap.add_argument("--nav", action="store_true", help="ページ間ナビ(株式/為替)を表示(公開用)")
    ap.add_argument("--shares-store", default="data/shares_history.csv", help="ETF口数の履歴CSV")
    ap.add_argument("-o", "--output", default="market_flow.html")
    a = ap.parse_args()

    cfg = MARKETS[a.market]
    bench, bench_name = cfg["benchmark"]
    sectors = cfg["sectors"]
    groups = {}
    if not a.no_holdings:
        try:
            import get_stock_price as gsp
            groups = {"保有": gsp.owning_list, "監視": gsp.check_list}
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 保有銘柄を読み込めません: {e}")
    extra = [t for g in groups.values() for t in g]
    reps = REPS[a.market]
    rep_tickers = [t for lst_ in reps.values() for t, _ in lst_]
    close, vol = load_prices(list(dict.fromkeys([bench, *sectors, *extra, *rep_tickers])), demo=a.demo)
    if bench not in close.columns or close[bench].isna().all():
        sys.exit("ベンチマークの価格を取得できませんでした")
    shares = load_shares([t for t in sectors if t in close.columns], close.index,
                         a.shares_store, demo=a.demo, close=close)

    fl = flows(close, vol, shares, sectors, a.window)
    mode, top_in, top_out = summary(fl)
    asof = close.index[-1].date()
    generated = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M JST")

    payload = build_payload(close, vol, shares, sectors, bench, groups, reps)
    n_sh = sum(v is not None for v in payload["shares"].values())
    sec_t = [t for t in sectors if t in close.columns]
    d_etf, n_etf = last_span(sec_t)
    d_rep, n_rep = last_span([t for lst_ in reps.values() for t, _ in lst_])
    sh_d = max(SHARES_LAST.values()) if SHARES_LAST else None
    rows = [("株価・出来高(セクターETF)", d_etf, n_etf),
            (f"ベンチマーク({bench})", LAST_VALID.get(bench), ""),
            ("代表銘柄(ホバー表示)", d_rep, n_rep),
            ("ETF発行口数", sh_d, f"口数が取れたのは {n_sh}/{len(sec_t)} 業種(最終観測日の最新)" if sh_d is not None
             else "口数データなし → 近似フローで表示")]
    if groups:
        d_h, n_h = last_span(extra)
        rows.append(("保有・監視銘柄", d_h, n_h))
    meta = {"weeks": max(1, round(a.window / 5)), "cur": "JPY" if a.market == "jp" else "USD",
            "label": f"({a.market.upper()}{' / DEMO' if a.demo else ''})",
            "generated": generated}
    plotly_tag = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>'
    if a.offline:
        from plotly.offline import get_plotlyjs
        plotly_tag = f"<script>{get_plotlyjs()}</script>"
    data_json = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    html = (PAGE.replace("__PLOTLY__", plotly_tag)
            .replace("__ASOF__", asof_html(rows, d_etf or asof))
            .replace("__NAV__", nav_html(a.market) if a.nav else "")
            .replace("__SOURCES__", sources_html(cfg, bench_name, asof, generated, a.demo, sectors))
            .replace("__META__", json.dumps(meta, ensure_ascii=False))
            .replace("__DATA__", data_json))
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    with open(a.output, "w", encoding="utf-8") as fh:
        fh.write(html)

    label = "実フロー" if mode == "actual" else "近似フロー"
    if a.notify:
        notify(f"[セクター資金フロー {a.market.upper()} {asof} / {label}]\n"
               f"流入: {' / '.join(top_in)}\n流出: {' / '.join(top_out)}")
    print(f"口数データ: {n_sh}/{len(payload['shares'])}業種 → {label}")
    print(f"流入上位: {top_in}\n流出上位: {top_out}\n出力: {a.output}")


if __name__ == "__main__":
    main()
