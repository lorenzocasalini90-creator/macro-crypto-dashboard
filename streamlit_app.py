import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

# =========================
# Page + Theme
# =========================
st.set_page_config(page_title="Macro Crypto Radar", layout="wide")

TITLE = "Macro Crypto Radar"
SUBTITLE = "Un radar macro per capire quando cambia il regime (liquidità → real rates → risk → conferme)."

# ---------- CSS (premium, sober) ----------
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
Questa dashboard serve a **identificare cambi di regime** (risk-on / risk-off) per crypto e asset risk.
Monitoriamo 3 forze:

1) **Liquidità in USD** (carburante del sistema)  
2) **Costo reale del denaro** (real yield: quando sale è spesso “kryptonite”)  
3) **Risk appetite sistemico** (stress o calma sui mercati)

Gli indicatori crypto sono usati come **conferma**, non come driver principale.
"""
)


# =========================
# Helpers
# =========================
def safe_last(s: pd.Series):
    if s is None:
        return None
    if isinstance(s, pd.DataFrame):
        if s.shape[1] == 0:
            return None
        s = s.iloc[:, 0]
    s2 = s.dropna()
    if len(s2) == 0:
        return None
    return float(s2.iloc[-1])


def last_date(s: pd.Series):
    if s is None:
        return None
    if isinstance(s, pd.DataFrame):
        if s.shape[1] == 0:
            return None
        s = s.iloc[:, 0]
    s2 = s.dropna()
    if len(s2) == 0:
        return None
    return s2.index.max().date().isoformat()


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
    if lookback == "6M":
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
    mask = turned.to_numpy(dtype=bool)
    return list(c.index[mask])


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
    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals))


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
    """
    Robust: accetta Series o DataFrame; restituisce freccia + delta.
    """
    if s is None:
        return ""
    if isinstance(s, pd.DataFrame):
        if s.shape[1] == 0:
            return ""
        s = s.iloc[:, 0]
    s2 = s.dropna()
    if len(s2) < days + 2:
        return ""
    try:
        a = float(s2.iloc[-1])
        b = float(s2.iloc[-days])
    except Exception:
        return ""
    delta = a - b
    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
    if unit:
        return f"{arrow} {delta:+.2f}{unit} (≈{days}g)"
    return f"{arrow} {delta:+.2f} (≈{days}g)"


# =========================
# Detailed explanations
# =========================
DETAIL = {
    "STATE": """
### Come leggere “Stato” e “Regime Score”
- **Regime Score (0–100)** = media pesata di 4 blocchi:
  - **Liquidità** (WALCL, RRP)  
  - **Real rates** (real yield: livello + discesa rapida)  
  - **Risk sentiment** (DXY, VIX)  
  - **Conferma crypto** (BTC/Nasdaq)

**Cosa significa “RISK-OFF” qui**
- Non è un “sell signal” automatico.
- Vuol dire che **le condizioni macro/risk non sono (ancora) favorevoli**: i rally sono più fragili e il rischio di drawdown aumenta.

**Cosa dovrebbe cambiare per passare verso risk-on**
- Real yield che scende (soprattutto velocemente)  
- DXY che perde trend  
- VIX sotto soglia e stabile  
- BTC che sovraperforma Nasdaq (conferma)
""",
    "WALCL": "**WALCL:** trend ↑/stabile = più favorevole; trend ↓ = QT (vento contrario).",
    "RRP": "**RRP:** in calo = liquidità che rientra; in aumento/stabile alto = meno ossigeno.",
    "DFII10": "**Real yield:** se sale = kryptonite; se scende rapidamente = sollievo per risk-on.",
    "T10Y2Y": "**Yield curve:** inversione profonda = warning; dis-inversione per tagli = costruttiva.",
    "DXY": "**DXY:** USD forte = freno; USD debole = aiuto (filtro).",
    "VIX": "**VIX:** basso = risk-on; alto = stress / risk-off.",
    "BTC": "**BTC:** conferma; utile quando macro già ok.",
    "RS": "**BTC/Nasdaq:** forza relativa; RS ↑ = leadership di BTC.",
}


def expl_expander(key: str, title: str = "📌 Spiegazione (apri/chiudi)"):
    with st.expander(title, expanded=False):
        st.markdown(DETAIL.get(key, ""))


# =========================
# Data sources
# =========================
def fred_series(series_id: str, api_key: str) -> pd.Series:
    r = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
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
    lookback = st.selectbox("Lookback", ["6M", "1Y", "2Y", "5Y", "Max"], index=2)
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

dxy, _ = yf_close_try(["DX-Y.NYB", "^DXY"], period="5y")
vix, _ = yf_close_try(["^VIX"], period="5y")
ixic, _ = yf_close_try(["^IXIC"], period="5y")
btc, _ = yf_close_try(["BTC-USD"], period="5y")

# =========================
# View transforms for charts
# =========================
walcl_v = normalize_series(slice_lookback(walcl, lookback), view_mode)
dfii10_v = normalize_series(slice_lookback(dfii10, lookback), "Raw")
rrp_v = normalize_series(slice_lookback(rrp, lookback), view_mode)
t10y2y_v = normalize_series(slice_lookback(t10y2y, lookback), "Raw")
dxy_v = normalize_series(slice_lookback(dxy, lookback), view_mode)
vix_v = normalize_series(slice_lookback(vix, lookback), "Raw")
btc_v = normalize_series(slice_lookback(btc, lookback), view_mode)

# =========================
# KPI premium
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
# Stato / Score
# =========================
walcl_ok = None
w = walcl.dropna()
if len(w) > 10:
    back = min(60, len(w) - 1)
    walcl_ok = (w.iloc[-1] - w.iloc[-back]) >= 0

rrp_trending_down = None
r = rrp.dropna()
if len(r) > rrp_trend_days + 5:
    rrp_ma = ma(r, rrp_trend_days).dropna()
    if len(rrp_ma) > 2:
        rrp_trending_down = (rrp_ma.diff().dropna().iloc[-1] < 0)

ry_ok_level = None if ry_last is None else (ry_last < real_yield_thr)
ry_change_60d_bps_series = bp_change_n(dfii10.dropna(), 60).dropna()
ry_drop_60d_bps = None if len(ry_change_60d_bps_series) == 0 else float(ry_change_60d_bps_series.iloc[-1])
ry_drop_fast = None if ry_drop_60d_bps is None else (ry_drop_60d_bps <= -float(ry_drop_bps))

dxy_below_ma = None
d = dxy.dropna()
if len(d) > dxy_ma_days + 10:
    dxy_below_ma = (d.iloc[-1] < ma(d, dxy_ma_days).iloc[-1])

vix_ok = None if vix_last is None else (vix_last < vix_thr)

btc_outperform = None
ratio = pd.Series(dtype=float)
b = btc.dropna()
# note: ixic not transformed for ratio, only raw
ixic_raw = ixic.dropna() if isinstance(ixic, pd.Series) else pd.Series(dtype=float)
if len(b) > rs_days + 10 and len(ixic_raw) > rs_days + 10:
    ratio = (b / ixic_raw).dropna()
    rs = ratio.pct_change(rs_days).dropna()
    if len(rs) > 0:
        btc_outperform = (float(rs.iloc[-1]) > 0)

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

section("Stato (Regime)", "Che regime stiamo osservando: è contesto, non timing. Serve per calibrare size e rischio.")
expl_expander("STATE", "📘 Come leggere Stato/Score (apri/chiudi)")

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
                f"- Real yield < soglia: {pill(ry_ok_level)}",
                f"- Real yield drop (60g): {pill(ry_drop_fast)}",
                f"- DXY sotto MA{dxy_ma_days}: {pill(dxy_below_ma)}",
                f"- VIX < {vix_thr:.1f}: {pill(vix_ok)}",
                f"- BTC RS ({rs_days}g): {pill(btc_outperform)}",
            ]
        ),
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# =========================
# Sections with expander ABOVE charts
# =========================
section(
    "1) Liquidità",
    "Carburante del sistema. WALCL in salita/stabile + RRP in calo = contesto più favorevole. "
    "QT persistente e RRP che non scende = vento contrario."
)
c1, c2 = st.columns(2)
with c1:
    expl_expander("WALCL", "📌 WALCL: guida (apri/chiudi)")
    chart_card(f"Fed Balance Sheet (WALCL) — {view_mode}", walcl_v, badge="Liquidity", kind="line")
with c2:
    expl_expander("RRP", "📌 RRP: guida (apri/chiudi)")
    chart_card(f"Reverse Repo (RRP) — {view_mode}", rrp_v, badge="Liquidity", kind="area")

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

section(
    "2) Costo reale del denaro",
    "Filtro chiave. Real yield in salita = kryptonite. Real yield in discesa (soprattutto rapida) = spazio per risk-on."
)
c3, c4 = st.columns(2)
with c3:
    expl_expander("DFII10", "📌 Real yield: guida (apri/chiudi)")
    chart_card("10Y Real Yield (DFII10) — Raw", dfii10_v, badge="Real Rates", kind="line")
with c4:
    expl_expander("T10Y2Y", "📌 Yield curve: guida (apri/chiudi)")
    chart_card("Yield Curve 10Y–2Y (T10Y2Y) — Raw", t10y2y_v, badge="Rates", kind="line")

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

section(
    "3) Risk sentiment",
    "Filtri. USD forte (DXY) o stress alto (VIX) spesso bloccano il risk-on. "
    "DXY in indebolimento + VIX basso = contesto più respirabile."
)
c5, c6 = st.columns(2)
with c5:
    expl_expander("DXY", "📌 DXY: guida (apri/chiudi)")
    chart_card(f"DXY — {view_mode}", dxy_v, badge="Risk Filter", kind="line")
with c6:
    expl_expander("VIX", "📌 VIX: guida (apri/chiudi)")
    chart_card("VIX — Raw", vix_v, badge="Risk Filter", kind="line")

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

section(
    "4) Crypto confirmation",
    "Conferme. BTC che sovraperforma Nasdaq segnala leadership. Underperformance rispetto a equity growth spesso indica rally fragile."
)
c7, c8 = st.columns(2)
with c7:
    expl_expander("BTC", "📌 BTC: guida (apri/chiudi)")
    chart_card(f"BTC — {view_mode}", btc_v, badge="Confirmation", kind="line")
with c8:
    expl_expander("RS", "📌 BTC/Nasdaq: guida (apri/chiudi)")
    if len(b) > 0 and len(ixic_raw) > 0:
        ratio_chart = slice_lookback((b / ixic_raw).dropna(), lookback)
        ratio_chart = normalize_series(ratio_chart, view_mode if view_mode != "Raw" else "Raw")
    else:
        ratio_chart = pd.Series(dtype=float)
    chart_card(f"BTC / Nasdaq (Relative Strength) — {view_mode}", ratio_chart, badge="Confirmation", kind="line")

st.caption("Tip: Index 100 per confrontare trend; Z-score per vedere quanto un indicatore è estremo vs storico.")
