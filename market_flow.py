"""セクター資金フロー可視化ツール

どのセクターから資金が抜け、どのセクターに入っているかを1枚のHTMLに出力する。

  python market_flow.py                 # 日本(TOPIX-17 ETF)
  python market_flow.py --market us     # 米国(SPDRセクターETF)
  python market_flow.py --window 10     # フロー集計期間(営業日)
  python market_flow.py --demo          # 合成データで動作確認(ネット不要)

注意: 「資金フロー」は ETF の 値上がり率 x 売買代金 から作った近似値で、
ファンドの実際の流出入額ではありません。
"""
import argparse
import subprocess
import sys

for _pkg in ("yfinance", "pandas", "numpy", "plotly"):
    try:
        __import__(_pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", _pkg])

import numpy as np
import pandas as pd
import plotly.graph_objects as go

MARKETS = {
    "jp": {
        "benchmark": ("1306.T", "TOPIX"),
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
        "benchmark": ("SPY", "S&P500"),
        "sectors": {
            "XLK": "情報技術", "XLF": "金融", "XLV": "ヘルスケア", "XLY": "一般消費財",
            "XLP": "生活必需品", "XLE": "エネルギー", "XLI": "資本財", "XLB": "素材",
            "XLU": "公益", "XLRE": "不動産", "XLC": "通信",
        },
    },
}


def load_prices(tickers, demo=False):
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
        return close, vol
    import yfinance as yf
    df = yf.download(list(tickers), period="1y", auto_adjust=True, progress=False)
    close = df["Close"].dropna(how="all").ffill()
    vol = df["Volume"].reindex(close.index).fillna(0)
    missing = [t for t in tickers if t not in close.columns or close[t].isna().all()]
    if missing:
        print(f"[warn] データ取得失敗: {missing}")
    return close, vol


def compute(close, vol, sectors, bench, window):
    ret = close.pct_change()
    turnover = close * vol
    out = pd.DataFrame(index=[t for t in sectors if t in close.columns])
    for label, n in (("1日", 1), ("1週", 5), ("1ヶ月", 21), ("3ヶ月", 63)):
        out[label] = (close.iloc[-1] / close.iloc[-1 - n] - 1) * 100
    # フロー近似 = 値上がり率 x 売買代金 の期間合計 → 全セクター合計に対する比率(%)
    signed = (ret * turnover).tail(window).sum()
    out["flow"] = signed[out.index]
    out["flow_pct"] = out["flow"] / out["flow"].abs().sum() * 100
    # 売買代金の増減(直近 window 日平均 / 過去60日平均)
    out["turnover_ratio"] = (turnover.tail(window).mean() / turnover.tail(60).mean())[out.index]
    out["name"] = [sectors[t] for t in out.index]
    return out


def rrg(close, sectors, bench, tail=8):
    """相対ローテーショングラフ: x=相対強度(100中心) y=そのモメンタム。"""
    rs = close[list(sectors)].div(close[bench], axis=0)
    rs = rs / rs.rolling(60).mean() * 100          # 相対強度
    mom = rs / rs.shift(10) * 100                  # 相対強度のモメンタム
    return rs.tail(tail), mom.tail(tail)


def fig_heatmap(m):
    cols = ["1日", "1週", "1ヶ月", "3ヶ月"]
    d = m.sort_values("1週", ascending=False)
    z = d[cols].values
    lim = np.nanmax(np.abs(z))
    f = go.Figure(go.Heatmap(
        z=z, x=cols, y=d["name"], zmid=0, zmin=-lim, zmax=lim,
        colorscale="RdYlGn", text=np.round(z, 1), texttemplate="%{text}%",
        hovertemplate="%{y} %{x}: %{z:.2f}%<extra></extra>"))
    f.update_yaxes(autorange="reversed")
    f.update_layout(title="① セクター別騰落率(上ほど直近1週間が強い)", height=60 + 30 * len(d))
    return f


def fig_flow(m, window):
    d = m.sort_values("flow_pct")
    f = go.Figure(go.Bar(
        x=d["flow_pct"], y=d["name"], orientation="h",
        marker_color=np.where(d["flow_pct"] >= 0, "#2a9d8f", "#e76f51"),
        customdata=d["turnover_ratio"],
        hovertemplate="%{y}<br>フロー %{x:.1f}%<br>売買代金 平常比 x%{customdata:.2f}<extra></extra>"))
    f.update_layout(title=f"② 資金フロー近似(直近{window}営業日・右=流入 / 左=流出)",
                    xaxis_title="全セクターの絶対フロー合計に対する比率 (%)",
                    height=60 + 30 * len(d))
    return f


def fig_sankey(m, window):
    out = m[m["flow"] < 0]
    inn = m[m["flow"] > 0]
    if out.empty or inn.empty:
        return None
    labels = [f"{n}(流出)" for n in out["name"]] + [f"{n}(流入)" for n in inn["name"]]
    src, dst, val = [], [], []
    total_in = inn["flow"].sum()
    for i, (_, o) in enumerate(out.iterrows()):
        for j, (_, p) in enumerate(inn.iterrows()):
            src.append(i)
            dst.append(len(out) + j)
            val.append(abs(o["flow"]) * p["flow"] / total_in)
    colors = ["#e76f51"] * len(out) + ["#2a9d8f"] * len(inn)
    f = go.Figure(go.Sankey(
        node=dict(label=labels, color=colors, pad=12),
        link=dict(source=src, target=dst, value=val,
                  color=["rgba(231,111,81,0.25)"] * len(val))))
    f.update_layout(title=f"③ 資金の移動イメージ(流出セクター → 流入セクター・直近{window}営業日)",
                    height=520)
    return f


def fig_rrg(close, sectors, bench):
    rs, mom = rrg(close, [t for t in sectors if t in close.columns], bench)
    f = go.Figure()
    for t in rs.columns:
        f.add_trace(go.Scatter(x=rs[t], y=mom[t], mode="lines+markers", name=sectors[t],
                               marker=dict(size=[4] * (len(rs) - 1) + [11]),
                               hovertemplate=sectors[t] + "<br>相対強度 %{x:.1f}<br>モメンタム %{y:.1f}<extra></extra>"))
    f.add_hline(y=100, line_dash="dot")
    f.add_vline(x=100, line_dash="dot")
    for x, y, txt in ((1, 1, "Leading 先導"), (0, 1, "Improving 改善"),
                      (0, 0, "Lagging 出遅れ"), (1, 0, "Weakening 失速")):
        f.add_annotation(xref="paper", yref="paper", x=x, y=y, text=txt, showarrow=False,
                         xanchor="right" if x else "left", yanchor="top" if y else "bottom",
                         font=dict(size=13, color="gray"))
    f.update_layout(title="④ セクターローテーション(RRG)— 大きい点が最新、反時計回りに循環",
                    xaxis_title="ベンチマーク比 相対強度", yaxis_title="相対強度のモメンタム",
                    height=620)
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market", choices=MARKETS, default="jp")
    ap.add_argument("--window", type=int, default=5, help="フロー集計期間(営業日)")
    ap.add_argument("--demo", action="store_true", help="合成データで実行")
    ap.add_argument("-o", "--output", default="market_flow.html")
    a = ap.parse_args()

    cfg = MARKETS[a.market]
    bench, bench_name = cfg["benchmark"]
    sectors = cfg["sectors"]
    close, vol = load_prices([bench, *sectors], demo=a.demo)
    if bench not in close.columns or close[bench].isna().all():
        sys.exit("ベンチマークの価格を取得できませんでした")

    m = compute(close, vol, sectors, bench, a.window)
    asof = close.index[-1].date()
    figs = [fig_heatmap(m), fig_flow(m, a.window), fig_sankey(m, a.window),
            fig_rrg(close, sectors, bench)]

    top_in = m.sort_values("flow_pct").tail(3)["name"][::-1].tolist()
    top_out = m.sort_values("flow_pct").head(3)["name"].tolist()
    head = (f"<h2>セクター資金フロー ({a.market.upper()} / {asof}{' / DEMO' if a.demo else ''})</h2>"
            f"<p>流入上位: <b>{' / '.join(top_in)}</b>　流出上位: <b>{' / '.join(top_out)}</b></p>"
            f"<p style='color:#888;font-size:12px'>フローは 値上がり率x売買代金 による近似で、実際のファンド資金流出入ではありません。"
            f"ベンチマーク: {bench_name}</p>")
    body = "".join(f.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False)
                   for i, f in enumerate(x for x in figs if x is not None))
    with open(a.output, "w", encoding="utf-8") as fh:
        fh.write(f"<html><head><meta charset='utf-8'><title>Sector Flow</title></head>"
                 f"<body style='font-family:sans-serif;max-width:1000px;margin:auto'>{head}{body}</body></html>")
    print(f"流入上位: {top_in}\n流出上位: {top_out}\n出力: {a.output}")


if __name__ == "__main__":
    main()
