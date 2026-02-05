import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import yfinance as yf

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

def fred_series(series_id: str, api_key: str) -> pd.Series:
    r = requests.get(
        FRED_BASE,
        params={"series_id": series_id, "api_key": api_key, "file_type": "json"},
        timeout=30,
    )
    r.raise_for_status()
    obs = r.json()["observations"]
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
def yf_close(ticker: str, period="2y") -> pd.Series:
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    s = df["Close"].dropna()
    s.name = ticker
    return s

def ma(s, n): 
    return s.rolling(n).mean()

st.set_page_config(page_title="Macro/Crypto Regime Dashboard", layout="wide")
st.title("Macro → Risk → Crypto: Regime Dashboard")

fred_key = st.secrets.get("FRED_API_KEY", "")

if not fred_key:
    st.error("Manca FRED_API_KEY nei Secrets. (Vedi Step 5)")
    st.stop()

walcl, dfii10, rrp, t10y2y = load_macro(fred_key)

dxy = yf_close("DX-Y.NYB")
vix = yf_close("^VIX")
ixic = yf_close("^IXIC")
btc = yf_close("BTC-USD")

# KPI row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Real Yield 10Y (DFII10)", f"{dfii10.iloc[-1]:.2f}%")
c2.metric("RRP (latest)", f"{rrp.iloc[-1]:,.0f}")
c3.metric("DXY", f"{dxy.iloc[-1]:.2f}")
c4.metric("VIX", f"{vix.iloc[-1]:.2f}")

# 4 blocchi
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
    st.plotly_chart(px.line(dxy, title="DXY (DX-Y.NYB)"), use_container_width=True)
    st.plotly_chart(px.line(vix, title="VIX (^VIX)"), use_container_width=True)

with b4:
    st.subheader("4) CRYPTO CONFIRMATION")
    st.plotly_chart(px.line(btc, title="BTC (BTC-USD)"), use_container_width=True)
    ratio = (btc / ixic).dropna()
    st.plotly_chart(px.line(ratio, title="BTC / Nasdaq (Relative Strength)"), use_container_width=True)

st.divider()
st.subheader("Regime Signals (bozza)")

signals = {
    "Real yield < 1.75%?": (dfii10.iloc[-1] < 1.75),
    "RRP trending down (20d)?": (ma(rrp, 20).diff().iloc[-1] < 0),
    "DXY below 200d MA?": (dxy.iloc[-1] < ma(dxy, 200).iloc[-1]),
    "VIX < 20?": (vix.iloc[-1] < 20),
    "BTC outperform Nasdaq (30d)?": (ratio.pct_change(30).iloc[-1] > 0),
}
st.dataframe(pd.DataFrame({"signal": list(signals.keys()), "on": list(signals.values())}))
