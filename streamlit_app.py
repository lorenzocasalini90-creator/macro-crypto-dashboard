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

details { border-radius: 14px; }
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
- **Costo reale del denaro** (real yield: spesso il filtro #1)
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
    if len(s) == 0:
        return None
    return float(s.iloc[-1])

def last_date(s):
    s = safe_series(s).dropna()
    if len(s) == 0:
        return None
    return s.index.max().date().isoformat()

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
    return (s2 - s2.shift(n)) * 100.0  # 1pp=100bps

def normalize_series(s: pd.Series, mode: str):
    s = s.dropna()
    if len(s) == 0:
        return s
    if mode == "Raw":
        return s
    if mode == "Index 100":
        base = float(s.iloc[0])
        if base == 0:
            return s
        return (s / base) * 100.0
    if mode == "Z-score":
        w = min(252, len(s))
        mu = s.rolling(w).mean()
        sd = s.rolling(w).std()
        z = (s - mu) / sd
        return z.dropna()
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

def cond_turns_true_dates(cond):
    if cond is None:
        return []
    if isinstance(cond, pd.DataFrame):
        if cond.shape[1] == 0:
            return []
        cond = cond.iloc[:, 0]
    c = cond.dropna().astype(bool)
    if len(c) < 2:
        return []
    turned = (c.astype(int).diff() == 1).fillna(False)
    return list(c.index[turned.to_numpy(dtype=bool)])

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
        if kind == "area":
            st.plotly_chart(px.area(series), use_container_width=True)
        else:
            st.plotly_chart(px.line(series), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

def trend_hint(s, days: int = 20, unit: str = ""):
    s = safe_series(s).dropna()
    if len(s) < days + 2:
        return ""
    try:
        a = float(s.iloc[-1]); b = float(s.iloc[-days])
    except Exception:
        return ""
    delta = a - b
    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
    if unit:
        return f"{arrow} {delta:+.2f}{unit} (≈{days}g)"
    return f"{arrow} {delta:+.2f} (≈{days}g)"

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
    """
    horizon: '1M','6M','1Y','5Y'
    mode: abs or pct
    """
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
    pos = None if (hi - lo) == 0 else (last - lo) / (hi - lo)  # 0..1
    return {"ok": True, "last": last, "change": ch, "lo": lo, "hi": hi, "pos": pos, "n": len(s_h)}

def fmt_pos(pos):
    if pos is None:
        return "n/a"
    return f"{pos*100:.0f}pctl (within range)"


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

    st.divider()
    st.subheader("Trigger recenti")
    ry_drop_bps = st.slider("Real Yield drop (bps) in 60g", 10, 150, 50, 5)
    trigger_window_days = st.slider("Finestra trigger (giorni)", 7, 90, 30, 1)


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

# =========================
# What changed (7d / 30d / 1M)
# =========================
def what_changed_table():
    rows = []

    walcl_last = safe_last(walcl)
    if walcl_last is not None:
        last_b = walcl_last / 1000.0
        d7 = delta_abs(walcl, 7); d30 = delta_abs(walcl, 30)
        d1m = delta_abs(walcl, 21)
        rows.append((
            "WALCL (Fed balance sheet)",
            f"{last_b:,.0f}B",
            "n/a" if d7 is None else f"{arrow_from(d7)} {d7/1000.0:+,.0f}B",
            "n/a" if d30 is None else f"{arrow_from(d30)} {d30/1000.0:+,.0f}B",
            "n/a" if d1m is None else f"{arrow_from(d1m)} {d1m/1000.0:+,.0f}B",
        ))

    if rrp_last is not None:
        d7 = delta_abs(rrp, 7); d30 = delta_abs(rrp, 30); d1m = delta_abs(rrp, 21)
        rows.append((
            "RRP",
            f"{rrp_last:,.0f}",
            "n/a" if d7 is None else f"{arrow_from(d7)} {d7:+,.0f}",
            "n/a" if d30 is None else f"{arrow_from(d30)} {d30:+,.0f}",
            "n/a" if d1m is None else f"{arrow_from(d1m)} {d1m:+,.0f}",
        ))

    if ry_last is not None:
        d7 = delta_abs(dfii10, 7); d30 = delta_abs(dfii10, 30); d1m = delta_abs(dfii10, 21)
        rows.append((
            "10Y Real Yield",
            f"{ry_last:.2f}%",
            "n/a" if d7 is None else f"{arrow_from(d7)} {d7*100:+.0f} bps",
            "n/a" if d30 is None else f"{arrow_from(d30)} {d30*100:+.0f} bps",
            "n/a" if d1m is None else f"{arrow_from(d1m)} {d1m*100:+.0f} bps",
        ))

    if dxy_last is not None:
        d7 = delta_abs(dxy, 7); d30 = delta_abs(dxy, 30); d1m = delta_abs(dxy, 21)
        rows.append((
            "DXY",
            f"{dxy_last:.2f}",
            "n/a" if d7 is None else f"{arrow_from(d7)} {d7:+.2f}",
            "n/a" if d30 is None else f"{arrow_from(d30)} {d30:+.2f}",
            "n/a" if d1m is None else f"{arrow_from(d1m)} {d1m:+.2f}",
        ))

    if vix_last is not None:
        d7 = delta_abs(vix, 7); d30 = delta_abs(vix, 30); d1m = delta_abs(vix, 21)
        rows.append((
            "VIX",
            f"{vix_last:.2f}",
            "n/a" if d7 is None else f"{arrow_from(d7)} {d7:+.2f}",
            "n/a" if d30 is None else f"{arrow_from(d30)} {d30:+.2f}",
            "n/a" if d1m is None else f"{arrow_from(d1m)} {d1m:+.2f}",
        ))

    btc_last = safe_last(btc)
    if btc_last is not None:
        p7 = delta_pct(btc, 7); p30 = delta_pct(btc, 30); p1m = delta_pct(btc, 21)
        rows.append((
            "BTC",
            f"{btc_last:,.0f}",
            "n/a" if p7 is None else f"{arrow_from(p7)} {p7:+.1f}%",
            "n/a" if p30 is None else f"{arrow_from(p30)} {p30:+.1f}%",
            "n/a" if p1m is None else f"{arrow_from(p1m)} {p1m:+.1f}%",
        ))

    rs_last = safe_last(rs_series)
    if rs_last is not None:
        p7 = delta_pct(rs_series, 7); p30 = delta_pct(rs_series, 30); p1m = delta_pct(rs_series, 21)
        rows.append((
            "BTC/Nasdaq RS",
            f"{rs_last:.4f}",
            "n/a" if p7 is None else f"{arrow_from(p7)} {p7:+.2f}%",
            "n/a" if p30 is None else f"{arrow_from(p30)} {p30:+.2f}%",
            "n/a" if p1m is None else f"{arrow_from(p1m)} {p1m:+.2f}%",
        ))

    return pd.DataFrame(rows, columns=["Metric", "Latest", "Δ 7d", "Δ 30d", "Δ 1M"])

section("What changed", "Snapshot driver principali: variazioni 7 giorni, 30 giorni e 1 mese.")
st.markdown("<div class='card'>", unsafe_allow_html=True)
df_wc = what_changed_table()
if len(df_wc) == 0:
    st.info("Dati insufficienti per calcolare le variazioni.")
else:
    st.dataframe(df_wc, use_container_width=True, hide_index=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# =========================
# Reference levels + explainers (with snapshot)
# =========================
def metric_snapshot(name: str, series: pd.Series, unit: str, mode: str):
    # mode: 'abs' or 'pct' for change; unit used in formatting last
    s1m = summarize_horizon(series, "1M", mode=mode)
    s6m = summarize_horizon(series, "6M", mode=mode)
    s1y = summarize_horizon(series, "1Y", mode=mode)

    last = safe_last(series)
    if last is None:
        return f"**Snapshot attuale:** n/a"

    if unit == "%":
        last_str = f"{last:.2f}%"
    else:
        last_str = f"{last:,.2f}" if abs(last) < 10000 else f"{last:,.0f}"

    def ch_str(ss):
        if not ss.get("ok"):
            return "n/a"
        ch = ss["change"]
        if ch is None or (isinstance(ch, float) and np.isnan(ch)):
            return "n/a"
        if mode == "pct":
            return f"{arrow_from(ch)} {ch:+.1f}% | {fmt_pos(ss['pos'])}"
        return f"{arrow_from(ch)} {ch:+.2f} | {fmt_pos(ss['pos'])}"

    return (
        f"**Snapshot attuale ({name}):** {last_str}\n\n"
        f"- **1M:** {ch_str(s1m)}\n"
        f"- **6M:** {ch_str(s6m)}\n"
        f"- **1Y:** {ch_str(s1y)}"
    )

# Reference content (pragmatic)
REF = {
    "WALCL": {
        "what": "Totale attivo della FED (proxy QE/QT). È una misura della direzione della liquidità lato banca centrale.",
        "refs": "- **Costruttivo:** trend stabile/in salita (QT rallenta / finisce)\n- **Negativo:** trend in calo persistente (QT)",
        "bi": "- **Se sale:** tende ad allentare condizioni finanziarie\n- **Se scende:** tende a stringerle",
    },
    "RRP": {
        "what": "Reverse Repo: cash parcheggiato alla FED. È liquidità “in panchina”.",
        "refs": "- **Costruttivo:** RRP che scende in modo sostenuto\n- **Meno bene:** RRP stabile alto o che risale",
        "bi": "- **Se scende:** potenziale supporto a asset risk\n- **Se sale:** più cash fuori dal circuito risk",
    },
    "DFII10": {
        "what": "10Y Real Yield (TIPS). È il rendimento risk-free *reale*.",
        "refs": "- **Bullish crypto (regola pratica):** < ~1.5–2.0%\n- **Bearish:** > ~2% e/o in salita rapida\n- **Trigger forte:** -50 bps in ~2 mesi",
        "bi": "- **Se sale:** crypto/growth competono con risk-free reale\n- **Se scende:** “sollievo” condizioni finanziarie",
    },
    "T10Y2Y": {
        "what": "Yield curve 10Y–2Y (spread). Descrive stance di policy e aspettative crescita.",
        "refs": "- **Warning:** inversione profonda e persistente\n- **Più costruttivo:** dis-inversione guidata da tagli (non crash)",
        "bi": "- **Più invertita:** condizioni più restrittive\n- **Si normalizza:** spesso più favorevole ai risk asset (se non per crisi)",
    },
    "DXY": {
        "what": "Dollar Index (forza USD). È un filtro di condizioni finanziarie globali.",
        "refs": "- **Costruttivo per risk:** DXY debole / sotto trend (es. sotto MA lunga)\n- **Negativo:** DXY forte/in accelerazione",
        "bi": "- **Se sale:** tende a stringere global liquidity\n- **Se scende:** più ossigeno per risk/EM/crypto",
    },
    "VIX": {
        "what": "Volatilità implicita equity USA (stress).",
        "refs": "- **Risk-on:** < ~15\n- **Neutro:** 15–20\n- **Risk-off serio:** > ~25",
        "bi": "- **Se sale:** deleveraging, correlazioni aumentano\n- **Se scende:** contesto più favorevole ai risk asset",
    },
    "BTC": {
        "what": "Prezzo BTC spot. In questo framework è una **conferma**, non il driver.",
        "refs": "- **Costruttivo:** BTC regge e/o rompe mentre macro migliora\n- **Meno bene:** BTC underperforma mentre macro è negativa",
        "bi": "- **Se sale con macro ok:** conferma risk-on\n- **Se sale con macro no:** spesso rally fragile",
    },
    "RS": {
        "what": "BTC/Nasdaq (forza relativa). Misura leadership di BTC vs equity growth.",
        "refs": "- **Costruttivo:** RS in salita per settimane\n- **Warning:** RS in calo (BTC underperforma growth)",
        "bi": "- **RS ↑:** domanda ‘vera’ e leadership\n- **RS ↓:** spesso non è ancora il momento",
    },
}

def expander_metric(key: str, title: str, series: pd.Series, unit: str, mode: str):
    with st.expander(f"📘 {title} — definizione, livelli guida, lettura (apri/chiudi)", expanded=False):
        st.markdown(f"**Che metrica è:** {REF[key]['what']}")
        st.markdown("**Valori di riferimento (pratici):**\n" + REF[key]["refs"])
        st.markdown("**Interpretazione bidirezionale:**\n" + REF[key]["bi"])
        st.markdown("---")
        st.markdown(metric_snapshot(title, series, unit=unit, mode=mode))

# =========================
# Report generator
# =========================
def make_report():
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    def block_line(name, series, unit, mode):
        s5 = summarize_horizon(series, "5Y", mode=mode)
        s1 = summarize_horizon(series, "1Y", mode=mode)
        s6 = summarize_horizon(series, "6M", mode=mode)
        sM = summarize_horizon(series, "1M", mode=mode)
        last = safe_last(series)

        def f_last():
            if last is None:
                return "n/a"
            if unit == "%":
                return f"{last:.2f}%"
            if abs(last) < 10000:
                return f"{last:.2f}"
            return f"{last:,.0f}"

        def f_ch(ss):
            if not ss.get("ok"):
                return "n/a"
            ch = ss["change"]
            if ch is None or (isinstance(ch, float) and np.isnan(ch)):
                return "n/a"
            if mode == "pct":
                return f"{arrow_from(ch)} {ch:+.1f}% ({fmt_pos(ss['pos'])})"
            return f"{arrow_from(ch)} {ch:+.2f} ({fmt_pos(ss['pos'])})"

        return (
            f"- **{name}** | last: **{f_last()}**\n"
            f"  - 5Y: {f_ch(s5)}\n"
            f"  - 1Y: {f_ch(s1)}\n"
            f"  - 6M: {f_ch(s6)}\n"
            f"  - 1M: {f_ch(sM)}"
        )

    md = []
    md.append(f"# Macro Crypto Radar — Report\n")
    md.append(f"_Generated: {now}_\n")
    md.append("## Executive summary\n")
    md.append(
        "Questo report riassume le evidenze su 4 blocchi (Liquidità, Real rates, Risk sentiment, Conferme) "
        "su **4 orizzonti**: **5Y / 1Y / 6M / 1M**. "
        "L’obiettivo non è fare timing perfetto, ma leggere se il contesto sta migliorando o peggiorando.\n"
    )

    md.append("## 1) Liquidità\n")
    md.append(block_line("WALCL (Fed balance sheet)", walcl, unit="", mode="abs"))
    md.append(block_line("RRP (Reverse Repo)", rrp, unit="", mode="abs"))
    md.append("\n**Lettura rapida:** WALCL stabile/in salita + RRP in calo è tipicamente più costruttivo per risk-on.\n")

    md.append("\n## 2) Costo reale del denaro\n")
    md.append(block_line("10Y Real Yield (DFII10)", dfii10, unit="%", mode="abs"))
    md.append(block_line("Yield Curve 10Y–2Y (T10Y2Y)", t10y2y, unit="", mode="abs"))
    md.append("\n**Lettura rapida:** real yield in discesa (specie 1M/6M) spesso apre la finestra risk-on; in salita è vento contrario.\n")

    md.append("\n## 3) Risk sentiment\n")
    md.append(block_line("DXY", dxy, unit="", mode="abs"))
    md.append(block_line("VIX", vix, unit="", mode="abs"))
    md.append("\n**Lettura rapida:** DXY debole + VIX contenuto supportano risk-on; USD forte o stress alto tendono a bloccare i rally.\n")

    md.append("\n## 4) Crypto confirmation\n")
    md.append(block_line("BTC", btc, unit="", mode="pct"))
    md.append(block_line("BTC/Nasdaq RS", rs_series, unit="", mode="pct"))
    md.append("\n**Lettura rapida:** RS in salita su 1M/6M è una conferma di leadership; RS in calo suggerisce prudenza.\n")

    return "\n".join(md)

section("Report generator", "Genera un report qualitativo su 5Y / 1Y / 6M / 1M + una sintesi per blocchi.")
with st.markdown("<div class='card'>", unsafe_allow_html=True):
    pass
st.markdown("<div class='card'>", unsafe_allow_html=True)
btn = st.button("Generate report", type="primary")
if btn:
    report_md = make_report()
    st.markdown(report_md)
    st.download_button("Download report (.md)", report_md, file_name="macro_crypto_report.md", mime="text/markdown")
else:
    st.caption("Premi “Generate report” per produrre la sintesi testuale. (Nessun costo extra: tutto local in Streamlit.)")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# =========================
# Regime score
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
ry_drop_fast = None if ry_drop_60d_bps is None else (ry_drop_60d_bps <= -float(ry_drop_bps))

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
                f"- Liquidità (WALCL trend): {pill(walcl_ok)}",
                f"- Liquidità (RRP trend): {pill(rrp_trending_down)}",
                f"- Real yield < soglia: {pill(ry_ok_level)}",
                f"- Real yield drop (60g): {pill(ry_drop_fast)}",
                f"- DXY sotto MA{dxy_ma_days}: {pill(dxy_below_ma)}",
                f"- VIX < {vix_thr:.1f}: {pill(vix_ok)}",
                f"- BTC sovraperforma Nasdaq ({rs_days}g): {pill(btc_outperform)}",
            ]
        ),
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# =========================
# Sections (expander ABOVE charts)
# =========================
section(
    "1) Liquidità",
    "Carburante del sistema. WALCL stabile/in salita e RRP in calo tendono a rendere più sostenibile il risk-on; l’opposto è vento contrario."
)
c1, c2 = st.columns(2)
with c1:
    expander_metric("WALCL", "WALCL", walcl, unit="", mode="abs")
    chart_card(f"Fed Balance Sheet (WALCL) — {view_mode}", walcl_v, badge="Liquidity", kind="line")
with c2:
    expander_metric("RRP", "RRP", rrp, unit="", mode="abs")
    chart_card(f"Reverse Repo (RRP) — {view_mode}", rrp_v, badge="Liquidity", kind="area")

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

section(
    "2) Costo reale del denaro",
    "Filtro chiave. Real yield in salita rende il risk-free reale competitivo; in discesa (soprattutto su 1M/6M) spesso apre la finestra risk-on."
)
c3, c4 = st.columns(2)
with c3:
    expander_metric("DFII10", "10Y Real Yield", dfii10, unit="%", mode="abs")
    chart_card("10Y Real Yield (DFII10) — Raw", dfii10_v, badge="Real Rates", kind="line")
with c4:
    expander_metric("T10Y2Y", "Yield Curve 10Y–2Y", t10y2y, unit="", mode="abs")
    chart_card("Yield Curve 10Y–2Y (T10Y2Y) — Raw", t10y2y_v, badge="Rates", kind="line")

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

section(
    "3) Risk sentiment",
    "Filtri. DXY forte o VIX alto spesso bloccano i rally; DXY debole e VIX contenuto rendono il contesto più respirabile."
)
c5, c6 = st.columns(2)
with c5:
    expander_metric("DXY", "DXY", dxy, unit="", mode="abs")
    chart_card(f"DXY — {view_mode}", dxy_v, badge="Risk Filter", kind="line")
with c6:
    expander_metric("VIX", "VIX", vix, unit="", mode="abs")
    chart_card("VIX — Raw", vix_v, badge="Risk Filter", kind="line")

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

section(
    "4) Crypto confirmation",
    "Conferme. BTC e soprattutto RS (BTC/Nasdaq) indicano leadership: RS in salita su più settimane è un segnale forte."
)
c7, c8 = st.columns(2)
with c7:
    expander_metric("BTC", "BTC", btc, unit="", mode="pct")
    chart_card(f"BTC — {view_mode}", btc_v, badge="Confirmation", kind="line")
with c8:
    expander_metric("RS", "BTC/Nasdaq RS", rs_series, unit="", mode="pct")
    chart_card(f"BTC / Nasdaq (Relative Strength) — {view_mode if view_mode != 'Raw' else 'Raw'}", rs_v, badge="Confirmation", kind="line")

st.caption("Tip: aggiungi ‘1M’ per il brevissimo; ‘6M/1Y/5Y’ per regime e contesto.")
