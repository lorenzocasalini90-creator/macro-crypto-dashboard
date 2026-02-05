import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime

# =========================
# Page + Theme
# =========================
st.set_page_config(page_title="Macro Crypto Radar", layout="wide")

TITLE = "Macro Crypto Radar"
SUBTITLE = "Regime dashboard: liquidità → real rates → risk → conferme (macro-first)."

st.markdown(
    """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2.2rem; max-width: 1220px; }
h1 { letter-spacing: -0.02em; margin-bottom: 0.1rem; }
.small-muted { color: rgba(49,51,63,0.65); font-size: 0.95rem; margin-top: -0.15rem; line-height: 1.25rem; }
p, li { line-height: 1.35rem; }
.hr { height: 1px; background: rgba(49,51,63,0.12); margin: 18px 0 18px 0; }

.section-title { font-size: 1.55rem; font-weight: 780; letter-spacing: -0.02em; margin: 0.2rem 0 0.2rem 0; }
.section-desc { color: rgba(49,51,63,0.70); font-size: 0.98rem; margin: 0 0 0.8rem 0; }

.card {
  background: #ffffff;
  border: 1px solid rgba(49,51,63,0.10);
  border-radius: 16px;
  padding: 14px 14px 10px 14px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.card-header { display:flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
.card-title { font-weight: 740; font-size: 0.98rem; color: rgba(49,51,63,0.90); }
.card-sub { color: rgba(49,51,63,0.60); font-size: 0.86rem; margin-top: -0.2rem; }
.metric-row { display:flex; gap: 10px; align-items: baseline; margin-top: 2px; }
.metric-big { font-size: 1.85rem; font-weight: 820; }
.metric-small { color: rgba(49,51,63,0.60); font-size: 0.9rem; }

.pill {
  display:inline-block; padding: 4px 10px; border-radius: 999px;
  border: 1px solid rgba(49,51,63,0.14);
  font-size: 0.82rem; color: rgba(49,51,63,0.80);
  background: rgba(49,51,63,0.03);
}
.pill-on { background: rgba(16, 185, 129, 0.08); border-color: rgba(16, 185, 129, 0.25); color: rgba(16, 120, 86, 1); }
.pill-off { background: rgba(245, 158, 11, 0.08); border-color: rgba(245, 158, 11, 0.25); color: rgba(146, 64, 14, 1); }
.pill-na { background: rgba(148, 163, 184, 0.10); border-color: rgba(148, 163, 184, 0.25); color: rgba(71, 85, 105, 1); }
summary { font-weight: 650; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(f"# {TITLE}")
st.markdown(f"<div class='small-muted'>{SUBTITLE}</div>", unsafe_allow_html=True)

st.markdown(
    """
Questa dashboard serve a **identificare cambi di regime** (risk-on / neutral / risk-off) per crypto e asset risk.

Framework:
- **Liquidità in USD** (carburante)
- **Costo reale del denaro** (real yield: spesso filtro #1)
- **Risk sentiment** (USD + stress)
- **Conferme crypto** (dopo la macro)
"""
)

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# =========================
# Helpers
# =========================
def safe_series(x):
    if x is None:
        return pd.Series(dtype=float)
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 0:
            return pd.Series(dtype=float)
        return x.iloc[:, 0]
    return x

def safe_last(s):
    s = safe_series(s).dropna()
    return None if len(s) == 0 else float(s.iloc[-1])

def last_date(s):
    s = safe_series(s).dropna()
    return None if len(s) == 0 else s.index.max().date().isoformat()

def fmt(x, digits=2, suffix=""):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}{suffix}"

def ma(s: pd.Series, n: int):
    return s.rolling(n).mean()

def bp_change_n(s: pd.Series, n: int):
    s2 = s.dropna()
    if len(s2) < n + 2:
        return pd.Series(dtype=float)
    return (s2 - s2.shift(n)) * 100.0

def normalize_series(s: pd.Series, mode: str):
    s = s.dropna()
    if len(s) == 0:
        return s
    if mode == "Raw":
        return s
    if mode == "Index 100":
        base = float(s.iloc[0])
        return s if base == 0 else (s / base) * 100.0
    if mode == "Z-score":
        w = min(252, len(s))
        mu = s.rolling(w).mean()
        sd = s.rolling(w).std()
        return ((s - mu) / sd).dropna()
    return s

def slice_lookback(s: pd.Series, lookback: str):
    if s is None or len(s) == 0:
        return s
    end = s.index.max()
    if lookback == "1M":
        start = end - pd.DateOffset(months=1)
    elif lookback == "6M":
        start = end - pd.DateOffset(months=6)
    elif lookback == "1Y":
        start = end - pd.DateOffset(years=1)
    elif lookback == "2Y":
        start = end - pd.DateOffset(years=2)
    elif lookback == "5Y":
        start = end - pd.DateOffset(years=5)
    else:
        start = s.index.min()
    return s.loc[s.index >= start]

def coerce_flag(x):
    if x is None:
        return None
    if isinstance(x, pd.Series):
        x = x.dropna()
        if len(x) == 0:
            return None
        x = x.iloc[-1]
    try:
        if isinstance(x, float) and np.isnan(x):
            return None
    except Exception:
        pass
    if x is True:
        return True
    if x is False:
        return False
    if isinstance(x, (int, float, np.integer, np.floating)):
        return bool(x)
    try:
        return bool(x)
    except Exception:
        return None

def pill(flag):
    f = coerce_flag(flag)
    if f is None:
        return "<span class='pill pill-na'>n/a</span>"
    return "<span class='pill pill-on'>ON</span>" if f else "<span class='pill pill-off'>OFF</span>"

def mean_flags(flags) -> float:
    vals = []
    for f in flags:
        f = coerce_flag(f)
        if f is True:
            vals.append(1.0)
        elif f is False:
            vals.append(0.0)
    return float(np.mean(vals)) if vals else 0.0

def section(label: str, desc: str):
    st.markdown(f"<div class='section-title'>{label}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-desc'>{desc}</div>", unsafe_allow_html=True)

def metric_card(title: str, value: str, subtitle: str = "", trend: str = ""):
    trend_html = f"<span class='metric-small'>{trend}</span>" if trend else ""
    st.markdown(
        f"""
<div class="card">
  <div class="card-title">{title}</div>
  <div class="metric-row">
    <div class="metric-big">{value}</div>
    {trend_html}
  </div>
  <div class="card-sub">{subtitle}</div>
</div>
""",
        unsafe_allow_html=True,
    )

def chart_card(title: str, series: pd.Series, badge: str = "", kind: str = "line"):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="card-header">
  <div class="card-title">{title}</div>
  <div><span class="pill">{badge}</span></div>
</div>
""",
        unsafe_allow_html=True,
    )
    if series is None or len(series.dropna()) == 0:
        st.info("Dati non disponibili.")
    else:
        st.plotly_chart(px.area(series) if kind == "area" else px.line(series), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

def trend_hint(s, days: int = 20, unit: str = ""):
    s = safe_series(s).dropna()
    if len(s) < days + 2:
        return ""
    a = float(s.iloc[-1]); b = float(s.iloc[-days])
    delta = a - b
    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
    return f"{arrow} {delta:+.2f}{unit} (≈{days}g)" if unit else f"{arrow} {delta:+.2f} (≈{days}g)"

def delta_abs(s: pd.Series, days: int):
    s = safe_series(s).dropna()
    if len(s) < days + 2:
        return None
    return float(s.iloc[-1]) - float(s.iloc[-days])

def delta_pct(s: pd.Series, days: int):
    s = safe_series(s).dropna()
    if len(s) < days + 2:
        return None
    a = float(s.iloc[-1]); b = float(s.iloc[-days])
    if b == 0:
        return None
    return (a / b - 1.0) * 100.0

def arrow_from(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "→"
    return "↑" if x > 0 else ("↓" if x < 0 else "→")

def compute_rs(btc_close: pd.Series, nasdaq_close: pd.Series) -> pd.Series:
    btc_close = safe_series(btc_close).dropna()
    nasdaq_close = safe_series(nasdaq_close).dropna()
    if len(btc_close) == 0 or len(nasdaq_close) == 0:
        return pd.Series(dtype=float)
    nasdaq_on_btc = nasdaq_close.reindex(btc_close.index).ffill()
    rs = (btc_close / nasdaq_on_btc).dropna()
    rs.name = "BTC/IXIC"
    return rs

def summarize_horizon(series: pd.Series, horizon: str, mode: str = "abs"):
    s = safe_series(series).dropna()
    if len(s) < 10:
        return {"ok": False}
    s_h = slice_lookback(s, horizon).dropna()
    if len(s_h) < 10:
        return {"ok": False}
    last = float(s_h.iloc[-1])
    first = float(s_h.iloc[0])
    if mode == "pct":
        ch = None if first == 0 else (last / first - 1.0) * 100.0
    else:
        ch = last - first
    lo = float(s_h.min()); hi = float(s_h.max())
    pos = None if (hi - lo) == 0 else (last - lo) / (hi - lo)
    return {"ok": True, "last": last, "change": ch, "lo": lo, "hi": hi, "pos": pos, "n": len(s_h)}

def fmt_pos(pos):
    return "n/a" if pos is None else f"{pos*100:.0f}pctl(range)"

# =========================
# Data sources
# =========================
def fred_series(series_id: str, api_key: str) -> pd.Series:
    r = requests.get(
        FRED_BASE,
        params={"series_id": series_id, "api_key": api_key, "file_type": "json"},
        timeout=30,
    )
    r.raise_for_status()
    obs = r.json().get("observations", [])
    s = pd.Series({o["date"]: (np.nan if o["value"] == "." else float(o["value"])) for o in obs}, name=series_id)
    s.index = pd.to_datetime(s.index)
    return s.sort_index().dropna()

@st.cache_data(ttl=60 * 60)
def load_macro(fred_key: str):
    walcl = fred_series("WALCL", fred_key)
    dfii10 = fred_series("DFII10", fred_key)
    rrp = fred_series("RRPONTSYD", fred_key)
    t10y2y = fred_series("T10Y2Y", fred_key)
    return walcl, dfii10, rrp, t10y2y

@st.cache_data(ttl=60 * 30)
def yf_close_try(tickers, period="5y"):
    for t in tickers:
        for _ in range(2):
            try:
                df = yf.download(t, period=period, interval="1d", auto_adjust=True, progress=False, threads=False)
                if isinstance(df, pd.DataFrame) and "Close" in df:
                    s = df["Close"].dropna()
                    if len(s) > 5:
                        s.name = t
                        return s, t
            except Exception:
                pass
            time.sleep(1)
    return pd.Series(dtype=float, name=str(tickers[0])), None

# =========================
# Sidebar controls
# =========================
with st.sidebar:
    st.header("Controlli")
    lookback = st.selectbox("Lookback", ["1M", "6M", "1Y", "2Y", "5Y", "Max"], index=1)
    view_mode = st.selectbox("Vista", ["Raw", "Index 100", "Z-score"], index=0)

    st.divider()
    st.subheader("Soglie semafori")
    real_yield_thr = st.slider("Real Yield risk-on (%, sotto è meglio)", 0.0, 4.0, 1.75, 0.05)
    vix_thr = st.slider("VIX risk-on (sotto è meglio)", 10.0, 40.0, 20.0, 0.5)
    rrp_trend_days = st.slider("RRP trend (giorni, MA)", 5, 60, 20, 1)
    dxy_ma_days = st.slider("DXY filtro MA (giorni)", 50, 300, 200, 10)
    rs_days = st.slider("BTC vs Nasdaq RS (giorni)", 10, 90, 30, 1)

# =========================
# Secrets
# =========================
fred_key = st.secrets.get("FRED_API_KEY", "")
if not fred_key:
    st.error("Manca FRED_API_KEY nei Secrets di Streamlit Cloud.")
    st.stop()

# =========================
# Load data
# =========================
walcl, dfii10, rrp, t10y2y = load_macro(fred_key)

dxy, dxy_used = yf_close_try(["DX-Y.NYB", "^DXY"], period="5y")
vix, vix_used = yf_close_try(["^VIX"], period="5y")
ixic, ixic_used = yf_close_try(["^IXIC"], period="5y")
btc, btc_used = yf_close_try(["BTC-USD"], period="5y")

warn = []
if dxy_used is None: warn.append("DXY non disponibile (yfinance).")
if vix_used is None: warn.append("VIX non disponibile (yfinance).")
if ixic_used is None: warn.append("Nasdaq (^IXIC) non disponibile (yfinance) → RS non calcolabile.")
if btc_used is None: warn.append("BTC-USD non disponibile (yfinance).")
if warn:
    st.warning(" ".join(warn))

rs_series = compute_rs(btc, ixic)

# =========================
# View transforms
# =========================
walcl_v = normalize_series(slice_lookback(walcl, lookback), view_mode)
dfii10_v = normalize_series(slice_lookback(dfii10, lookback), "Raw")
rrp_v = normalize_series(slice_lookback(rrp, lookback), view_mode)
t10y2y_v = normalize_series(slice_lookback(t10y2y, lookback), "Raw")
dxy_v = normalize_series(slice_lookback(dxy, lookback), view_mode)
vix_v = normalize_series(slice_lookback(vix, lookback), "Raw")
btc_v = normalize_series(slice_lookback(btc, lookback), view_mode)
rs_v = normalize_series(slice_lookback(rs_series, lookback), view_mode if view_mode != "Raw" else "Raw")

# =========================
# KPI
# =========================
ry_last = safe_last(dfii10)
rrp_last = safe_last(rrp)
dxy_last = safe_last(dxy)
vix_last = safe_last(vix)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    metric_card("Real Yield 10Y", fmt(ry_last, 2, "%"), "Più basso = più spazio per risk-on", trend_hint(dfii10, 60, "%"))
with kpi2:
    metric_card("RRP", "n/a" if rrp_last is None else f"{rrp_last:,.0f}", "In calo = liquidità che rientra", trend_hint(rrp, 20, ""))
with kpi3:
    metric_card("DXY", fmt(dxy_last, 2), "USD forte spesso frena asset risk", trend_hint(dxy, 20, ""))
with kpi4:
    metric_card("VIX", fmt(vix_last, 2), "Basso = calma / risk-on", trend_hint(vix, 20, ""))

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# =========================
# What changed (7d / 30d / 1M)
# =========================
def what_changed_table():
    rows = []
    walcl_last = safe_last(walcl)
    if walcl_last is not None:
        last_b = walcl_last / 1000.0
        d7 = delta_abs(walcl, 7); d30 = delta_abs(walcl, 30); d1m = delta_abs(walcl, 21)
        rows.append(("WALCL", f"{last_b:,.0f}B",
                    "n/a" if d7 is None else f"{arrow_from(d7)} {d7/1000.0:+,.0f}B",
                    "n/a" if d30 is None else f"{arrow_from(d30)} {d30/1000.0:+,.0f}B",
                    "n/a" if d1m is None else f"{arrow_from(d1m)} {d1m/1000.0:+,.0f}B"))
    if rrp_last is not None:
        d7 = delta_abs(rrp, 7); d30 = delta_abs(rrp, 30); d1m = delta_abs(rrp, 21)
        rows.append(("RRP", f"{rrp_last:,.0f}",
                    "n/a" if d7 is None else f"{arrow_from(d7)} {d7:+,.0f}",
                    "n/a" if d30 is None else f"{arrow_from(d30)} {d30:+,.0f}",
                    "n/a" if d1m is None else f"{arrow_from(d1m)} {d1m:+,.0f}"))
    if ry_last is not None:
        d7 = delta_abs(dfii10, 7); d30 = delta_abs(dfii10, 30); d1m = delta_abs(dfii10, 21)
        rows.append(("Real Yield 10Y", f"{ry_last:.2f}%",
                    "n/a" if d7 is None else f"{arrow_from(d7)} {d7*100:+.0f} bps",
                    "n/a" if d30 is None else f"{arrow_from(d30)} {d30*100:+.0f} bps",
                    "n/a" if d1m is None else f"{arrow_from(d1m)} {d1m*100:+.0f} bps"))
    if dxy_last is not None:
        d7 = delta_abs(dxy, 7); d30 = delta_abs(dxy, 30); d1m = delta_abs(dxy, 21)
        rows.append(("DXY", f"{dxy_last:.2f}",
                    "n/a" if d7 is None else f"{arrow_from(d7)} {d7:+.2f}",
                    "n/a" if d30 is None else f"{arrow_from(d30)} {d30:+.2f}",
                    "n/a" if d1m is None else f"{arrow_from(d1m)} {d1m:+.2f}"))
    if vix_last is not None:
        d7 = delta_abs(vix, 7); d30 = delta_abs(vix, 30); d1m = delta_abs(vix, 21)
        rows.append(("VIX", f"{vix_last:.2f}",
                    "n/a" if d7 is None else f"{arrow_from(d7)} {d7:+.2f}",
                    "n/a" if d30 is None else f"{arrow_from(d30)} {d30:+.2f}",
                    "n/a" if d1m is None else f"{arrow_from(d1m)} {d1m:+.2f}"))
    btc_last = safe_last(btc)
    if btc_last is not None:
        p7 = delta_pct(btc, 7); p30 = delta_pct(btc, 30); p1m = delta_pct(btc, 21)
        rows.append(("BTC", f"{btc_last:,.0f}",
                    "n/a" if p7 is None else f"{arrow_from(p7)} {p7:+.1f}%",
                    "n/a" if p30 is None else f"{arrow_from(p30)} {p30:+.1f}%",
                    "n/a" if p1m is None else f"{arrow_from(p1m)} {p1m:+.1f}%"))
    rs_last = safe_last(rs_series)
    if rs_last is not None:
        p7 = delta_pct(rs_series, 7); p30 = delta_pct(rs_series, 30); p1m = delta_pct(rs_series, 21)
        rows.append(("BTC/Nasdaq RS", f"{rs_last:.4f}",
                    "n/a" if p7 is None else f"{arrow_from(p7)} {p7:+.2f}%",
                    "n/a" if p30 is None else f"{arrow_from(p30)} {p30:+.2f}%",
                    "n/a" if p1m is None else f"{arrow_from(p1m)} {p1m:+.2f}%"))

    return pd.DataFrame(rows, columns=["Metric", "Latest", "Δ 7d", "Δ 30d", "Δ 1M"])

section("What changed", "Snapshot driver principali: variazioni 7 giorni, 30 giorni e 1 mese (proxy ~21 trading days).")
st.markdown("<div class='card'>", unsafe_allow_html=True)
df_wc = what_changed_table()
if len(df_wc):
    st.dataframe(df_wc, use_container_width=True, hide_index=True)
else:
    st.info("Dati insufficienti.")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# =========================
# Regime Score
# =========================
walcl_ok = None
w = safe_series(walcl).dropna()
if len(w) > 10:
    back = min(60, len(w) - 1)
    walcl_ok = (w.iloc[-1] - w.iloc[-back]) >= 0

rrp_trending_down = None
r = safe_series(rrp).dropna()
if len(r) > rrp_trend_days + 5:
    rrp_ma = ma(r, rrp_trend_days).dropna()
    if len(rrp_ma) > 2:
        rrp_trending_down = (rrp_ma.diff().dropna().iloc[-1] < 0)

ry_ok_level = None if ry_last is None else (ry_last < real_yield_thr)
ry_change_60d_bps_series = bp_change_n(safe_series(dfii10).dropna(), 60).dropna()
ry_drop_60d_bps = None if len(ry_change_60d_bps_series) == 0 else float(ry_change_60d_bps_series.iloc[-1])
ry_drop_fast = None if ry_drop_60d_bps is None else (ry_drop_60d_bps <= -50.0)

dxy_below_ma = None
d = safe_series(dxy).dropna()
if len(d) > dxy_ma_days + 10:
    dxy_below_ma = (d.iloc[-1] < ma(d, dxy_ma_days).iloc[-1])

vix_ok = None if vix_last is None else (vix_last < vix_thr)

btc_outperform = None
if len(rs_series.dropna()) > rs_days + 10:
    rs_mom = rs_series.dropna().pct_change(rs_days).dropna()
    if len(rs_mom) > 0:
        btc_outperform = float(rs_mom.iloc[-1]) > 0

liquidity_score = mean_flags([walcl_ok, rrp_trending_down])
realrates_score = mean_flags([ry_ok_level, ry_drop_fast])
risk_score = mean_flags([dxy_below_ma, vix_ok])
crypto_score = mean_flags([btc_outperform])

WGT = {"liq": 0.30, "rr": 0.30, "risk": 0.25, "cr": 0.15}
regime_score = 100.0 * (
    WGT["liq"] * liquidity_score
    + WGT["rr"] * realrates_score
    + WGT["risk"] * risk_score
    + WGT["cr"] * crypto_score
)

if regime_score >= 70:
    regime_label = "RISK-ON ✅"
elif regime_score >= 40:
    regime_label = "NEUTRAL ⚖️"
else:
    regime_label = "RISK-OFF ⚠️"

section("Stato (Regime)", "È contesto, non timing: serve per calibrare size e rischio.")
s1, s2 = st.columns([1.1, 2.2])
with s1:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(regime_score),
            number={"suffix": "/100"},
            title={"text": "Regime Score"},
            gauge={"axis": {"range": [0, 100]}},
        )
    )
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

with s2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"### {regime_label}")
    st.markdown(
        "\n".join(
            [
                f"- WALCL trend: {pill(walcl_ok)}",
                f"- RRP trend: {pill(rrp_trending_down)}",
                f"- Real yield < soglia: {pill(ry_ok_level)}",
                f"- Real yield drop (60g): {pill(ry_drop_fast)}",
                f"- DXY sotto MA{dxy_ma_days}: {pill(dxy_below_ma)}",
                f"- VIX < soglia: {pill(vix_ok)}",
                f"- BTC sovraperforma Nasdaq: {pill(btc_outperform)}",
            ]
        ),
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# =========================
# Reference levels (used in expanders + payload)
# =========================
REF = {
    "WALCL": {
        "metric": "Totale attivo della FED (proxy QE/QT).",
        "good_bad": "Bene: trend stabile/in salita (QT rallenta/finisce). Meno bene: trend in calo persistente (QT).",
        "notes": "Guarda il trend (settimane), non il singolo dato."
    },
    "RRP": {
        "metric": "Reverse Repo: cash parcheggiato alla FED.",
        "good_bad": "Bene: in calo sostenuto. Meno bene: stabile alto o in risalita.",
        "notes": "È liquidità 'in panchina' che può rientrare."
    },
    "DFII10": {
        "metric": "10Y Real Yield (TIPS): rendimento risk-free reale.",
        "good_bad": "Bene: <~1.5–2% e/o in discesa. Meno bene: >~2% e/o in salita rapida.",
        "notes": "Conta molto il delta (es. -50bps in 2 mesi)."
    },
    "T10Y2Y": {
        "metric": "Spread 10Y–2Y: forma della curva e stance di policy.",
        "good_bad": "Warning: inversione profonda. Più costruttivo: dis-inversione da tagli (non crisi).",
        "notes": "È contesto macro, non timing diretto."
    },
    "DXY": {
        "metric": "Dollar Index: forza USD (filtro condizioni finanziarie).",
        "good_bad": "Bene per risk: USD debole / sotto trend. Meno bene: USD forte/in accelerazione.",
        "notes": "Trend > livello."
    },
    "VIX": {
        "metric": "Volatilità implicita equity USA (stress).",
        "good_bad": "Risk-on: <~15; neutro: 15–20; risk-off: >~25.",
        "notes": "Crypto raramente riparte con VIX alto."
    },
    "BTC": {
        "metric": "Prezzo BTC spot (conferma).",
        "good_bad": "Bene: sale con macro migliorativa. Meno bene: rally con macro avversa (fragile).",
        "notes": "Non è driver primario nel framework."
    },
    "RS": {
        "metric": "BTC/Nasdaq: forza relativa (leadership).",
        "good_bad": "Bene: RS in salita (BTC sovraperforma). Meno bene: RS in calo.",
        "notes": "Molto utile come conferma di regime."
    }
}

# =========================
# Charts sections (PRIMARY VIEW FIRST)
# =========================
section("1) Liquidità", "Carburante del sistema: WALCL e RRP guidano il contesto di liquidità.")
c1, c2 = st.columns(2)
with c1:
    with st.expander("📘 WALCL — definizione e livelli guida", expanded=False):
        st.write(REF["WALCL"]["metric"])
        st.write(REF["WALCL"]["good_bad"])
        st.caption(REF["WALCL"]["notes"])
    chart_card(f"Fed Balance Sheet (WALCL) — {view_mode}", walcl_v, badge="Liquidity", kind="line")
with c2:
    with st.expander("📘 RRP — definizione e livelli guida", expanded=False):
        st.write(REF["RRP"]["metric"])
        st.write(REF["RRP"]["good_bad"])
        st.caption(REF["RRP"]["notes"])
    chart_card(f"Reverse Repo (RRP) — {view_mode}", rrp_v, badge="Liquidity", kind="area")

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

section("2) Costo reale del denaro", "Filtro #1: il real yield spesso decide quanto spazio c’è per crypto e growth.")
c3, c4 = st.columns(2)
with c3:
    with st.expander("📘 Real Yield — definizione e livelli guida", expanded=False):
        st.write(REF["DFII10"]["metric"])
        st.write(REF["DFII10"]["good_bad"])
        st.caption(REF["DFII10"]["notes"])
    chart_card("10Y Real Yield (DFII10) — Raw", dfii10_v, badge="Real Rates", kind="line")
with c4:
    with st.expander("📘 Yield Curve — definizione e livelli guida", expanded=False):
        st.write(REF["T10Y2Y"]["metric"])
        st.write(REF["T10Y2Y"]["good_bad"])
        st.caption(REF["T10Y2Y"]["notes"])
    chart_card("Yield Curve 10Y–2Y (T10Y2Y) — Raw", t10y2y_v, badge="Rates", kind="line")

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

section("3) Risk sentiment", "Filtri: USD e stress possono bloccare o liberare il risk-on.")
c5, c6 = st.columns(2)
with c5:
    with st.expander("📘 DXY — definizione e livelli guida", expanded=False):
        st.write(REF["DXY"]["metric"])
        st.write(REF["DXY"]["good_bad"])
        st.caption(REF["DXY"]["notes"])
    chart_card(f"DXY — {view_mode}", dxy_v, badge="Risk Filter", kind="line")
with c6:
    with st.expander("📘 VIX — definizione e livelli guida", expanded=False):
        st.write(REF["VIX"]["metric"])
        st.write(REF["VIX"]["good_bad"])
        st.caption(REF["VIX"]["notes"])
    chart_card("VIX — Raw", vix_v, badge="Risk Filter", kind="line")

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

section("4) Crypto confirmation", "Conferme: BTC e forza relativa vs Nasdaq indicano leadership e qualità del risk-on.")
c7, c8 = st.columns(2)
with c7:
    with st.expander("📘 BTC — definizione e livelli guida", expanded=False):
        st.write(REF["BTC"]["metric"])
        st.write(REF["BTC"]["good_bad"])
        st.caption(REF["BTC"]["notes"])
    chart_card(f"BTC — {view_mode}", btc_v, badge="Confirmation", kind="line")
with c8:
    with st.expander("📘 BTC/Nasdaq RS — definizione e livelli guida", expanded=False):
        st.write(REF["RS"]["metric"])
        st.write(REF["RS"]["good_bad"])
        st.caption(REF["RS"]["notes"])
    chart_card(f"BTC / Nasdaq (Relative Strength) — {view_mode if view_mode != 'Raw' else 'Raw'}", rs_v, badge="Confirmation", kind="line")

st.caption("Tip: Usa ‘1M’ per il brevissimo, e 6M/1Y/5Y per regime e contesto.")

# =========================
# REPORT (OPTIONAL) — AT THE BOTTOM
# =========================
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
section("Report (opzionale)", "Genera un payload copiabile: lo incolli in ChatGPT per un report AI operativo (senza API).")

def metric_pack(name, series, mode="abs"):
    out = {"name": name}
    out["as_of"] = last_date(series)
    last = safe_last(series)
    out["last"] = None if last is None else float(last)
    for hz in ["1M", "6M", "1Y", "5Y"]:
        out[hz] = summarize_horizon(series, hz, mode=mode)
    return out

def build_payload_text():
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    payload = {
        "generated": now,
        "regime": {"score": float(regime_score), "label": regime_label},
        "signals": {
            "walcl_trend_ok": coerce_flag(walcl_ok),
            "rrp_trending_down": coerce_flag(rrp_trending_down),
            "real_yield_below_thr": coerce_flag(ry_ok_level),
            "real_yield_drop_fast": coerce_flag(ry_drop_fast),
            "dxy_below_ma": coerce_flag(dxy_below_ma),
            "vix_below_thr": coerce_flag(vix_ok),
            "btc_outperforms_nasdaq": coerce_flag(btc_outperform),
        },
        "what_changed": df_wc.to_dict(orient="records") if len(df_wc) else [],
        "reference_levels": REF,
        "metrics": {
            "WALCL": metric_pack("WALCL", walcl, mode="abs"),
            "RRP": metric_pack("RRP", rrp, mode="abs"),
            "DFII10": metric_pack("RealYield10Y", dfii10, mode="abs"),
            "T10Y2Y": metric_pack("YieldCurve10Y2Y", t10y2y, mode="abs"),
            "DXY": metric_pack("DXY", dxy, mode="abs"),
            "VIX": metric_pack("VIX", vix, mode="abs"),
            "BTC": metric_pack("BTC", btc, mode="pct"),
            "RS": metric_pack("BTC_NASDAQ_RS", rs_series, mode="pct"),
        },
        "notes": {
            "method": "RS uses Nasdaq forward-filled on BTC calendar; Δ1M in what_changed uses ~21 trading days proxy."
        }
    }

    def line_metric(key, label, unit="", pct=False):
        m = payload["metrics"][key]
        last = m["last"]
        last_s = "n/a" if last is None else (f"{last:.2f}{unit}" if unit else f"{last:,.2f}")

        def hz(h):
            ss = m[h]
            if not ss.get("ok"):
                return "n/a"
            ch = ss["change"]
            pos = ss.get("pos")
            if ch is None or (isinstance(ch, float) and np.isnan(ch)):
                return "n/a"
            if pct:
                return f"{arrow_from(ch)} {ch:+.1f}% | {fmt_pos(pos)}"
            return f"{arrow_from(ch)} {ch:+.2f}{unit} | {fmt_pos(pos)}"

        return (
            f"- **{label}** (last={last_s})\n"
            f"  - 5Y: {hz('5Y')}\n"
            f"  - 1Y: {hz('1Y')}\n"
            f"  - 6M: {hz('6M')}\n"
            f"  - 1M: {hz('1M')}"
        )

    txt = []
    txt.append(f"## Macro Crypto Radar — Payload\nGenerated: {now}\n")
    txt.append(f"### Regime\n- score: {payload['regime']['score']:.1f}/100\n- label: {payload['regime']['label']}\n")
    txt.append("### Signals (ON/OFF)\n" + "\n".join([f"- {k}: {payload['signals'][k]}" for k in payload["signals"]]) + "\n")

    txt.append("### What changed (7d/30d/1M)\n")
    if payload["what_changed"]:
        for row in payload["what_changed"]:
            txt.append(f"- {row['Metric']}: last={row['Latest']} | 7d={row['Δ 7d']} | 30d={row['Δ 30d']} | 1M={row['Δ 1M']}")
    else:
        txt.append("- n/a")

    txt.append("\n### Metrics summary (5Y/1Y/6M/1M)\n")
    txt.append(line_metric("WALCL", "WALCL (Fed balance sheet)", unit="", pct=False))
    txt.append(line_metric("RRP", "RRP (Reverse Repo)", unit="", pct=False))
    txt.append(line_metric("DFII10", "10Y Real Yield", unit="%", pct=False))
    txt.append(line_metric("T10Y2Y", "Yield Curve 10Y–2Y", unit="", pct=False))
    txt.append(line_metric("DXY", "DXY", unit="", pct=False))
    txt.append(line_metric("VIX", "VIX", unit="", pct=False))
    txt.append(line_metric("BTC", "BTC", unit="", pct=True))
    txt.append(line_metric("RS", "BTC/Nasdaq RS", unit="", pct=True))

    txt.append("\n### Reference levels (definitions + good/bad)\n")
    for k, v in REF.items():
        txt.append(f"- **{k}**: {v['metric']} | {v['good_bad']} | Note: {v['notes']}")

    txt.append("\n### Notes\n- " + payload["notes"]["method"])
    return "\n".join(txt)

PROMPT = """Sei un macro strategist. Usa SOLO i dati nel payload qui sopra. Non inventare numeri.
Scrivi un report in italiano, sobrio e operativo (no hype), con questa struttura:

1) Executive summary (max 6 righe): cosa dice il regime e cosa è cambiato nel brevissimo (1M + what changed).
2) Lettura per blocchi: Liquidità / Real rates / Risk sentiment / Crypto confirmation
   - per ogni blocco: lungo(5Y), medio(1Y), breve(6M), brevissimo(1M)
   - evidenzia: direzione, velocità del cambiamento e coerenza tra indicatori.
3) “Cosa sta cambiando adesso e perché” (focus 1M + ultimi 7/30 giorni).
4) Implicazioni operative:
   - stance: prudente / base / aggressivo
   - sizing e risk management coerenti col regime
   - cosa deve cambiare per aumentare o ridurre rischio.
5) Trigger da monitorare (3–5) per la prossima finestra (1–4 settimane).
"""

with st.expander("1) Genera & copia il payload (apri/chiudi)", expanded=False):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    colA, colB = st.columns([1, 2])
    with colA:
        gen = st.button("Generate payload", type="primary")
    with colB:
        st.caption("Workflow: Generate payload → copia → incolla in ChatGPT + prompt (sezione sotto).")

    if gen:
        st.session_state["payload_text"] = build_payload_text()

    payload_text = st.session_state.get("payload_text", "")
    if payload_text:
        st.text_area("Payload (copiami)", payload_text, height=340)
        st.download_button("Download payload (.txt)", payload_text, file_name="macro_crypto_payload.txt", mime="text/plain")
    else:
        st.info("Premi “Generate payload” per creare il testo copiabile.")
    st.markdown("</div>", unsafe_allow_html=True)

with st.expander("2) Prompt per ChatGPT (apri/chiudi)", expanded=False):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.text_area("Prompt (copiami)", PROMPT, height=260)
    st.markdown("</div>", unsafe_allow_html=True)
