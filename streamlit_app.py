import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import yfinance as yf

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Macro/Crypto Regime Dashboard", layout="wide")
st.title("Macro → Risk → Crypto: Regime Dashboard")

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


# ----------------------------
# Helpers (safe formatting)
# ----------------------------
def safe_last(s: pd.Series):
    if s is None or len(s) == 0:
        return None
    x = s.dropna()
    if len(x) == 0:
        return None
    return float(x.iloc[-1])


def fmt(x, digits=2, suffix=""):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}{suffix}"


def ma(s: pd.Series, n: int):
    return s.rolling(n).mean()


# ----------------------------
# Data sources
# ----------------------------
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
def yf_close_try(tickers, period="2y"):
    """
    tickers: lista di ticker in ordine di preferenza
    ritorna (serie, ticker_usato) oppure (serie vuota, None)
    """
    for t in tickers:
        for attempt in range(2):  # retry 2 volte
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
                    # Se troppo corta, trattala come "non valida"
                    if len(s) > 5:
                        s.name = t
                        return s, t
            except Exception:
                pass
            time.sleep(1)
    return pd.Series(dtype=float, name=str(tickers[0])), None


# ----------------------------
# Secrets
# ----------------------------
fred_key = st.secrets.get("FRED_API_KEY", "")
if not fred_key:
    st.error("Manca FRED_API_KEY nei Secrets di Streamlit Cloud.")
    st.stop()


# ----------------------------
# Load data
# ----------------------------
walcl, dfii10, rrp, t10y2y = load_macro(fred_key)

# Yahoo / yfinance (può fallire a volte su Cloud → fallback + no crash)
dxy, dxy_used = yf_close_try(["DX-Y.NYB", "^DXY"], period="2y")  # fallback
vix, vix_used = yf_close_try(["^VIX"], period="2y")
ixic, ixic_used = yf_close_try(["^IXIC"], period="2y")
btc, btc_used = yf_close_try(["BTC-USD"], period="2y")

# ----------------------------
# KPI row
# ----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Real Yield 10Y (DFII10)", fmt(safe_last(dfii10), 2, "%"))
last_rrp = safe_last(rrp)
c2.metric("RRP (latest)", "n/a" if last_rrp is None else f"{last_rrp:,.0f}")
c3.metric("DXY", fmt(safe_last(dxy), 2))
c4.metric("VIX", fmt(safe_last(vix), 2))

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


# ----------------------------
# Layout: 4 blocchi
# ----------------------------
b1, b2 = st.columns(2)

with b1:
    st.subheader("1) LIQUIDITÀ")
    st.plotly_chart(px.line(walcl, title="Fed Balance Sheet (WALCL)"), use_container_width=True)
    st.plotly_chart(px.area(rrp, title="Reverse Repo (RRPONTSYD)"), use_container_width=True)

with b2:
    st.subheader("2) COSTO REALE DEL DENARO")
    st.plotly_chart(px.line(dfii10, title="10Y Real Yield (DFII10)"), use_container_width=True)
    st.plotly_chart(px.line(t10y2y, title="Yield Curve (T10Y2Y)"), use_container_width=True)

b3, b4 = st.columns(2)

with b3:
    st.subheader("3) RISK SENTIMENT")
    if len(dxy) > 0:
        st.plotly_chart(px.line(dxy, title="DXY (DX-Y.NYB / ^DXY)"), use_container_width=True)
    else:
        st.info("DXY non disponibile al momento.")
    if len(vix) > 0:
        st.plotly_chart(px.line(vix, title="VIX (^VIX)"), use_container_width=True)
    else:
        st.info("VIX non disponibile al momento.")

with b4:
    st.subheader("4) CRYPTO CONFIRMATION")
    if len(btc) > 0:
        st.plotly_chart(px.line(btc, title="BTC (BTC-USD)"), use_container_width=True)
    else:
        st.info("BTC non disponibile al momento.")

    if len(btc) > 0 and len(ixic) > 0:
        ratio = (btc / ixic).dropna()
        if len(ratio) > 5:
            st.plotly_chart(px.line(ratio, title="BTC / Nasdaq (Relative Strength)"), use_container_width=True)
        else:
            st.info("Ratio BTC/Nasdaq non disponibile (serie troppo corta).")
    else:
        st.info("Ratio BTC/Nasdaq non disponibile (mancano dati BTC o Nasdaq).")


# ----------------------------
# Signals (bozza)
# ----------------------------
st.divider()
st.subheader("Regime Signals (bozza)")

# Calcoli "safe"
real_yield_ok = (safe_last(dfii10) is not None) and (safe_last(dfii10) < 1.75)

rrp_ma20 = ma(rrp, 20) if len(rrp) > 25 else pd.Series(dtype=float)
rrp_trending_down = False
if len(rrp_ma20) > 0 and len(rrp) > 25:
    # trend down: differenza dell'ultima MA vs giorno prima < 0
    rrp_trending_down = (rrp_ma20.diff().dropna().iloc[-1] < 0)

dxy_below_200ma = False
if len(dxy) > 220:
    dxy_below_200ma = (dxy.iloc[-1] < ma(dxy, 200).iloc[-1])

vix_ok = (safe_last(vix) is not None) and (safe_last(vix) < 20)

btc_outperform_ndx_30d = False
if len(btc) > 40 and len(ixic) > 40:
    ratio = (btc / ixic).dropna()
    if len(ratio) > 40:
        btc_outperform_ndx_30d = (ratio.pct_change(30).dropna().iloc[-1] > 0)

signals = {
    "Real yield < 1.75%?": real_yield_ok,
    "RRP trending down (20d MA)?": rrp_trending_down,
    "DXY below 200d MA?": dxy_below_200ma,
    "VIX < 20?": vix_ok,
    "BTC outperform Nasdaq (30d)?": btc_outperform_ndx_30d,
}

df_signals = pd.DataFrame({"signal": list(signals.keys()), "on": list(signals.values())})
st.dataframe(df_signals, use_container_width=True)

