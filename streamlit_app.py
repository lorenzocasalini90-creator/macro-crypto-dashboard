import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

# =========================
# Page
# =========================
st.set_page_config(page_title="Macro/Crypto Regime Dashboard", layout="wide")
st.title("Macro → Risk → Crypto: Regime Dashboard")

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


# =========================
# Helpers
# =========================
def safe_last(s: pd.Series):
    if s is None:
        return None
    s2 = s.dropna()
    if len(s2) == 0:
        return None
    return float(s2.iloc[-1])


def last_date(s: pd.Series):
    if s is None:
        return None
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
    # assumes series in percent points; 1.00 = 1%
    s2 = s.dropna()
    if len(s2) < n + 2:
        return pd.Series(dtype=float)
    return (s2 - s2.shift(n)) * 100.0  # percent points -> bps (1pp=100bps)


def normalize_series(s: pd.Series, mode: str):
    """
    mode: "Raw", "Index 100", "Z-score"
    """
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


def cond_turns_true_dates(cond: pd.Series):
    """
    cond: boolean series indexed by date.
    Returns dates where it switches False->True
    """
    if cond is None or len(cond) == 0:
        return []
    c = cond.dropna().astype(bool)
    if len(c) < 2:
        return []
    turned = (c.astype(int).diff() == 1)
    return list(c.index[turned.fillna(False)])


def mean_flags(flags) -> float:
    """
    flags: list di True/False/None
    - True  -> 1
    - False -> 0
    - None  -> ignorato
    Se sono tutti None -> 0
    """
    vals = []
    for f in flags:
        if f is True:
            vals.append(1.0)
        elif f is False:
            vals.append(0.0)
        else:
            continue
    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals))


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
    s = pd.Series(
        {o["date"]: (np.nan if o["value"] == "." else float(o["value"])) for o in obs},
        name=series_id,
    )
    s.index = pd.to_datetime(s.index)
    return s.sort_index().dropna()


@st.cache_data(ttl=60 * 60)
def load_macro(fred_key: str):
    walcl = fred_series("WALCL", fred_key)       # Fed balance sheet
    dfii10 = fred_series("DFII10", fred_key)     # 10Y real yield (TIPS)
    rrp = fred_series("RRPONTSYD", fred_key)     # Reverse repo
    t10y2y = fred_series("T10Y2Y", fred_key)     # 10Y-2Y spread
    return walcl, dfii10, rrp, t10y2y


@st.cache_data(ttl=60 * 30)
def yf_close_try(tickers, period="5y"):
    """
    tickers: lista di ticker in ordine di preferenza
    ritorna (serie, ticker_usato) oppure (serie vuota, None)
    """
    for t in tickers:
        for _ in range(2):  # retry
            try:
                df = yf.download(
                    t,
                    period=period,
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
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
    real_yield_thr = st.slider("Real Yield: soglia risk-on (%, sotto è meglio)", 0.0, 4.0, 1.75, 0.05)
    vix_thr = st.slider("VIX: soglia risk-on (sotto è meglio)", 10.0, 40.0, 20.0, 0.5)
    rrp_trend_days = st.slider("RRP trend: finestra (giorni, MA)", 5, 60, 20, 1)
    dxy_ma_days = st.slider("DXY filtro: MA (giorni)", 50, 300, 200, 10)
    rs_days = st.slider("BTC vs Nasdaq: lookback forza relativa (giorni)", 10, 90, 30, 1)

    st.divider()
    st.subheader("Alert / Trigger")
    ry_drop_bps = st.slider("Trigger: Real Yield drop (bps) in 60g", 10, 150, 50, 5)
    trigger_window_days = st.slider("Finestra 'Trigger recenti' (giorni)", 7, 90, 30, 1)


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

warnings = []
if dxy_used is None:
    warnings.append("DXY non disponibile da Yahoo (yfinance) in questo momento.")
if vix_used is None:
    warnings.append("VIX non disponibile da Yahoo (yfinance) in questo momento.")
if ixic_used is None:
    warnings.append("Nasdaq (^IXIC) non disponibile da Yahoo (yfinance) in questo momento.")
if btc_used is None:
    warnings.append("BTC-USD non disponibile da Yahoo (yfinance) in questo momento.")
if warnings:
    st.warning(" ".join(warnings))


# =========================
# View transforms for charts
# =========================
walcl_v = normalize_series(slice_lookback(walcl, lookback), view_mode)
dfii10_v = normalize_series(slice_lookback(dfii10, lookback), "Raw")  # yields raw
rrp_v = normalize_series(slice_lookback(rrp, lookback), view_mode)
t10y2y_v = normalize_series(slice_lookback(t10y2y, lookback), "Raw")  # spread raw
dxy_v = normalize_series(slice_lookback(dxy, lookback), view_mode)
vix_v = normalize_series(slice_lookback(vix, lookback), "Raw")        # VIX raw
ixic_v = normalize_series(slice_lookback(ixic, lookback), view_mode)
btc_v = normalize_series(slice_lookback(btc, lookback), view_mode)


# =========================
# KPI row (raw)
# =========================
k1, k2, k3, k4 = st.columns(4)
k1.metric("Real Yield 10Y (DFII10)", fmt(safe_last(dfii10), 2, "%"))
last_rrp_val = safe_last(rrp)
k2.metric("RRP (latest)", "n/a" if last_rrp_val is None else f"{last_rrp_val:,.0f}")
k3.metric("DXY", fmt(safe_last(dxy), 2))
k4.metric("VIX", fmt(safe_last(vix), 2))


# =========================
# Explanations
# =========================
EXPL = {
    "WALCL": (
        "**Fed Balance Sheet (WALCL)**\n\n"
        "- **Cosa misura:** dimensione del bilancio FED.\n"
        "- **Come leggerlo:** trend ↑ = più liquidità/meno drenaggio; trend ↓ = QT.\n"
        "- **Implicazioni:** spesso costruttivo per asset risk quando smette di scendere o risale.\n"
    ),
    "RRP": (
        "**Reverse Repo (RRP)**\n\n"
        "- **Cosa misura:** liquidità parcheggiata (cash che non va su asset risk).\n"
        "- **Come leggerlo:** RRP che scende = potenziale liquidità che rientra nei mercati.\n"
        "- **Implicazioni:** calo rapido è spesso positivo per risk-on.\n"
    ),
    "DFII10": (
        "**10Y Real Yield (DFII10)**\n\n"
        "- **Cosa misura:** rendimento reale risk-free (TIPS).\n"
        "- **Come leggerlo:** real yield ↑ = competizione per asset risk; ↓ rapido = sollievo.\n"
        "- **Implicazioni:** crypto tende a soffrire con real yield alti/in salita.\n"
    ),
    "T10Y2Y": (
        "**Yield Curve 10Y–2Y (T10Y2Y)**\n\n"
        "- **Cosa misura:** forma della curva (stress/aspettative su crescita e policy).\n"
        "- **Come leggerlo:** inversione profonda = warning; dis-inversione per tagli = spesso costruttiva.\n"
    ),
    "DXY": (
        "**DXY (Dollar Index)**\n\n"
        "- **Cosa misura:** forza del dollaro vs basket.\n"
        "- **Come leggerlo:** trend ↑ spesso frena asset risk; trend ↓ tende ad aiutare.\n"
        "- **Implicazioni:** ottimo filtro: dollaro in accelerazione = risk-on più difficile.\n"
    ),
    "VIX": (
        "**VIX**\n\n"
        "- **Cosa misura:** stress/percezione del rischio.\n"
        "- **Come leggerlo:** basso = risk-on; alto = risk-off.\n"
        "- **Implicazioni:** crypto raramente riparte con VIX elevato.\n"
    ),
    "BTC": (
        "**BTC (prezzo)**\n\n"
        "- **Cosa misura:** prezzo spot.\n"
        "- **Come leggerlo:** da solo non basta; diventa più utile come conferma quando la macro è favorevole.\n"
    ),
    "RS": (
        "**BTC / Nasdaq (Forza relativa)**\n\n"
        "- **Cosa misura:** se BTC sta sovraperformando l’equity growth.\n"
        "- **Come leggerlo:** RS ↑ mentre Nasdaq è flat/giù = segnale forte.\n"
        "- **Implicazioni:** conferma “risk-on crypto” quando la macro è già ok.\n"
    ),
}


# =========================
# Data status table
# =========================
with st.expander("📦 Data status (ultimi aggiornamenti)", expanded=False):
    status_rows = [
        ("WALCL", "Fed Balance Sheet", last_date(walcl)),
        ("RRPONTSYD", "Reverse Repo", last_date(rrp)),
        ("DFII10", "10Y Real Yield", last_date(dfii10)),
        ("T10Y2Y", "Yield Curve 10Y–2Y", last_date(t10y2y)),
        ("DXY", "DXY", last_date(dxy)),
        ("VIX", "VIX", last_date(vix)),
        ("^IXIC", "Nasdaq", last_date(ixic)),
        ("BTC-USD", "BTC", last_date(btc)),
    ]
    st.dataframe(pd.DataFrame(status_rows, columns=["ID", "Indicatore", "Ultima data"]), use_container_width=True)


# =========================
# Signals + Score (robust)
# =========================
# Liquidity
walcl_ok = None
w = walcl.dropna()
if len(w) > 10:
    # proxy: "non sta scendendo" negli ultimi ~8 settimane (qui usiamo 60 osservazioni come proxy robusto)
    back = min(60, len(w) - 1)
    walcl_ok = (w.iloc[-1] - w.iloc[-back]) >= 0

rrp_trending_down = None
r = rrp.dropna()
if len(r) > rrp_trend_days + 5:
    rrp_ma = ma(r, rrp_trend_days).dropna()
    if len(rrp_ma) > 2:
        rrp_trending_down = (rrp_ma.diff().dropna().iloc[-1] < 0)

# Real rates
ry_last = safe_last(dfii10)
ry_ok_level = None if ry_last is None else (ry_last < real_yield_thr)

ry_change_60d_bps_series = bp_change_n(dfii10.dropna(), 60).dropna()
ry_drop_60d_bps = None if len(ry_change_60d_bps_series) == 0 else float(ry_change_60d_bps_series.iloc[-1])
ry_drop_fast = None if ry_drop_60d_bps is None else (ry_drop_60d_bps <= -float(ry_drop_bps))

# Risk
dxy_below_ma = None
d = dxy.dropna()
if len(d) > dxy_ma_days + 10:
    dxy_below_ma = d.iloc[-1] < ma(d, dxy_ma_days).iloc[-1]

vix_last = safe_last(vix)
vix_ok = None if vix_last is None else (vix_last < vix_thr)

# Crypto confirmation
btc_outperform = None
ratio = pd.Series(dtype=float)
b = btc.dropna()
n = ixic.dropna()
if len(b) > rs_days + 10 and len(n) > rs_days + 10:
    ratio = (b / n).dropna()
    rs = ratio.pct_change(rs_days).dropna()
    if len(rs) > 0:
        btc_outperform = float(rs.iloc[-1]) > 0

# Component scores in [0,1]
liquidity_score = mean_flags([walcl_ok, rrp_trending_down])
realrates_score = mean_flags([ry_ok_level, ry_drop_fast])
risk_score = mean_flags([dxy_below_ma, vix_ok])
crypto_score = mean_flags([btc_outperform])

# Weights sum to 1
W = {"liq": 0.30, "rr": 0.30, "risk": 0.25, "cr": 0.15}
regime_score = 100.0 * (
    W["liq"] * liquidity_score
    + W["rr"] * realrates_score
    + W["risk"] * risk_score
    + W["cr"] * crypto_score
)

if regime_score >= 70:
    regime_label = "RISK-ON ✅"
elif regime_score >= 40:
    regime_label = "NEUTRAL ⚖️"
else:
    regime_label = "RISK-OFF ⚠️"


# =========================
# Header: score + signals
# =========================
s1, s2, s3 = st.columns([1.2, 1.2, 2.6])

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
    st.subheader("Stato")
    st.markdown(f"### {regime_label}")
    st.caption("Score = media pesata dei blocchi: Liquidità, Real Rates, Risk, Conferma Crypto.")
    st.markdown(
        f"- **Real yield**: {fmt(ry_last,2,'%')} | **Δ60g**: {('n/a' if ry_drop_60d_bps is None else f'{ry_drop_60d_bps:.0f} bps')}\n"
        f"- **VIX**: {fmt(vix_last,2)}\n"
        f"- **DXY sotto MA{dxy_ma_days}**: {('n/a' if dxy_below_ma is None else ('✅' if dxy_below_ma else '—'))}\n"
        f"- **BTC RS ({rs_days}g)**: {('n/a' if btc_outperform is None else ('✅' if btc_outperform else '—'))}"
    )

with s3:
    st.subheader("Semafori (live)")
    signals = {
        "WALCL (trend 8w proxy) ≥ 0": walcl_ok,
        f"RRP MA{rrp_trend_days} in calo": rrp_trending_down,
        f"Real yield < {real_yield_thr:.2f}%": ry_ok_level,
        f"Real yield drop ≥ {ry_drop_bps} bps (60g)": ry_drop_fast,
        f"DXY sotto MA{dxy_ma_days}": dxy_below_ma,
        f"VIX < {vix_thr:.1f}": vix_ok,
        f"BTC sovraperforma Nasdaq ({rs_days}g)": btc_outperform,
    }
    df_sig = pd.DataFrame(
        {
            "signal": list(signals.keys()),
            "status": [
                ("n/a" if v is None else ("ON ✅" if v else "OFF —"))
                for v in signals.values()
            ],
        }
    )
    st.dataframe(df_sig, use_container_width=True, height=240)


# =========================
# Triggers recenti
# =========================
recent_start = pd.Timestamp.today().normalize() - pd.Timedelta(days=trigger_window_days)
trigger_items = []

# Real yield drop
if len(dfii10.dropna()) > 120:
    ry_chg = bp_change_n(dfii10.dropna(), 60).dropna()
    if len(ry_chg) > 0:
        ry_cond = (ry_chg <= -float(ry_drop_bps)).dropna()
        ry_trig_dates = [d for d in cond_turns_true_dates(ry_cond) if d >= recent_start]
        if len(ry_trig_dates) > 0:
            trigger_items.append(("Real yield drop (60g)", ry_trig_dates[-1].date().isoformat()))

# WALCL trend >=0
if len(w) > 120:
    back = min(60, len(w) - 1)
    slope = (w - w.shift(back)).dropna()
    walcl_cond = (slope >= 0).dropna()
    walcl_trig_dates = [d for d in cond_turns_true_dates(walcl_cond) if d >= recent_start]
    if len(walcl_trig_dates) > 0:
        trigger_items.append(("WALCL slope ≥ 0", walcl_trig_dates[-1].date().isoformat()))

# DXY below MA
if len(d) > dxy_ma_days + 30:
    cond = (d < ma(d, dxy_ma_days)).dropna()
    dxy_trig_dates = [x for x in cond_turns_true_dates(cond) if x >= recent_start]
    if len(dxy_trig_dates) > 0:
        trigger_items.append((f"DXY sotto MA{dxy_ma_days}", dxy_trig_dates[-1].date().isoformat()))

# VIX below threshold
vv = vix.dropna()
if len(vv) > 30:
    cond = (vv < vix_thr).dropna()
    vix_trig_dates = [x for x in cond_turns_true_dates(cond) if x >= recent_start]
    if len(vix_trig_dates) > 0:
        trigger_items.append((f"VIX < {vix_thr:.1f}", vix_trig_dates[-1].date().isoformat()))

# BTC outperform Nasdaq
if len(ratio.dropna()) > rs_days + 30:
    rs = ratio.dropna().pct_change(rs_days).dropna()
    cond = (rs > 0).dropna()
    rs_trig_dates = [x for x in cond_turns_true_dates(cond) if x >= recent_start]
    if len(rs_trig_dates) > 0:
        trigger_items.append((f"BTC RS>0 ({rs_days}g)", rs_trig_dates[-1].date().isoformat()))

with st.expander(f"⚡ Trigger recenti (ultimi {trigger_window_days} giorni)", expanded=True):
    if len(trigger_items) == 0:
        st.info("Nessun trigger rilevato nella finestra selezionata (o dati insufficienti).")
    else:
        st.dataframe(pd.DataFrame(trigger_items, columns=["Trigger", "Ultima attivazione"]), use_container_width=True)


# =========================
# Charts + explanations
# =========================
def chart_with_expl(title: str, series: pd.Series, expl_key: str, kind: str = "line"):
    left, right = st.columns([2.2, 1.0], vertical_alignment="top")
    with left:
        if series is None or len(series.dropna()) == 0:
            st.info(f"{title}: dati non disponibili.")
        else:
            if kind == "area":
                st.plotly_chart(px.area(series, title=title), use_container_width=True)
            else:
                st.plotly_chart(px.line(series, title=title), use_container_width=True)
    with right:
        st.markdown(EXPL.get(expl_key, ""))


b1, b2 = st.columns(2)

with b1:
    st.subheader("1) LIQUIDITÀ")
    chart_with_expl(f"Fed Balance Sheet (WALCL) — {view_mode}", walcl_v, "WALCL", kind="line")
    chart_with_expl(f"Reverse Repo (RRPONTSYD) — {view_mode}", rrp_v, "RRP", kind="area")

with b2:
    st.subheader("2) COSTO REALE DEL DENARO")
    chart_with_expl("10Y Real Yield (DFII10) — Raw", dfii10_v, "DFII10", kind="line")
    chart_with_expl("Yield Curve 10Y–2Y (T10Y2Y) — Raw", t10y2y_v, "T10Y2Y", kind="line")

b3, b4 = st.columns(2)

with b3:
    st.subheader("3) RISK SENTIMENT")
    chart_with_expl(f"DXY — {view_mode}", dxy_v, "DXY", kind="line")
    chart_with_expl("VIX — Raw", vix_v, "VIX", kind="line")

with b4:
    st.subheader("4) CRYPTO CONFIRMATION")
    chart_with_expl(f"BTC (BTC-USD) — {view_mode}", btc_v, "BTC", kind="line")

    rs_left, rs_right = st.columns([2.2, 1.0], vertical_alignment="top")
    with rs_left:
        if len(b) > 0 and len(n) > 0:
            ratio_chart = slice_lookback((b / n).dropna(), lookback)
            ratio_chart = normalize_series(ratio_chart, view_mode if view_mode != "Raw" else "Raw")
            if len(ratio_chart.dropna()) > 5:
                st.plotly_chart(px.line(ratio_chart, title=f"BTC / Nasdaq (Relative Strength) — {view_mode}"), use_container_width=True)
            else:
                st.info("Ratio BTC/Nasdaq non disponibile (serie troppo corta).")
        else:
            st.info("Ratio BTC/Nasdaq non disponibile (mancano dati BTC o Nasdaq).")
    with rs_right:
        st.markdown(EXPL["RS"])

st.caption(
    "Tip: usa **Index 100** per confrontare trend su scale diverse; usa **Z-score** per vedere quanto un indicatore è estremo rispetto al suo passato."
)

