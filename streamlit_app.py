# ============================================================
# CRYPTO MACRO REGIME DASHBOARD — Versione B (FULL)
# ============================================================
# Focus: monitorare fondamentali macro che guidano il sentiment crypto (BTC + ETH),
# separando:
# 1) Liquidità USD (plumbing)
# 2) Costo reale del denaro (real yields / tassi)
# 3) Propensione al rischio sistemica (USD, credit, vol, trend)
# 4) Crypto confirmation (BTC, ETH, RS vs Nasdaq)
# 5) Vincoli strutturali (fiscal/policy, external, gold)
#
# NOTE TECNICA (fix bug <div>):
# - Wallboard usa SOLO componenti Streamlit-native (st.metric / st.progress / columns).
#   Nessuna griglia HTML complessa => niente "leak" di markup nel rendering.
# ============================================================

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from pandas.tseries.offsets import DateOffset

# ============================================================
# PAGE CONFIG (mobile-first)
# ============================================================
st.set_page_config(
    page_title="Crypto Macro Regime | Dashboard",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ============================================================
# PREMIUM CSS (sobrio, nessun layout HTML critico)
# ============================================================
st.markdown(
    """
<style>
  :root{
    --bg:#0b0f19;
    --border:rgba(255,255,255,0.10);
    --muted:rgba(255,255,255,0.70);
    --text:rgba(255,255,255,0.94);

    --good:rgba(34,197,94,1);
    --warn:rgba(245,158,11,1);
    --bad:rgba(239,68,68,1);

    --accent:#60a5fa;
    --accentSoft:rgba(96,165,250,0.16);
  }

  .stApp {
    background: radial-gradient(1200px 700px at 20% 0%, #121a33 0%, #0b0f19 45%, #0b0f19 100%);
    color: var(--text);
  }

  .block-container { padding-top: 0.9rem; padding-bottom: 1.8rem; max-width: 1020px; }

  h1, h2, h3, h4 { color: var(--text); letter-spacing: -0.02em; }
  .muted { color: var(--muted); }

  /* Tabs */
  button[data-baseweb="tab"]{
    color: rgba(255,255,255,0.90) !important;
    font-weight: 650 !important;
    background: rgba(255,255,255,0.03) !important;
    border-radius: 10px !important;
    margin-right: 6px !important;
  }
  button[data-baseweb="tab"][aria-selected="true"]{
    color: rgba(255,255,255,0.98) !important;
    background: var(--accentSoft) !important;
    border: 1px solid rgba(96,165,250,0.45) !important;
  }

  /* Expanders */
  div[data-testid="stExpander"]{
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    background: rgba(255,255,255,0.03) !important;
    overflow: hidden !important;
  }
  div[data-testid="stExpander"] summary{
    background: rgba(255,255,255,0.04) !important;
    color: rgba(255,255,255,0.92) !important;
    padding: 10px 12px !important;
  }
  div[data-testid="stExpander"] summary:hover{
    background: rgba(255,255,255,0.06) !important;
  }

  /* Dataframe */
  .stDataFrame { border: 1px solid var(--border); border-radius: 12px; overflow:hidden; }

  code { color: rgba(255,255,255,0.88); }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# CONSTANTS
# ============================================================
SNAPSHOT_SCHEMA_VERSION = "2.2"


# ============================================================
# BASIC HELPERS
# ============================================================
def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def status_label(status: str) -> str:
    if status == "risk_on":
        return "Risk-on"
    if status == "risk_off":
        return "Risk-off"
    if status == "neutral":
        return "Neutral"
    return "n/a"


def classify_status(score: float) -> str:
    if np.isnan(score):
        return "n/a"
    if score > 60:
        return "risk_on"
    if score < 40:
        return "risk_off"
    return "neutral"


def pill_html(status: str) -> str:
    # Minimal HTML pill (safe, not used for layout)
    if status == "risk_on":
        return "<span style='display:inline-flex;align-items:center;gap:8px;padding:5px 12px;border-radius:999px;border:1px solid rgba(34,197,94,0.40);background:rgba(34,197,94,0.12);font-size:0.90rem;'><span style='width:11px;height:11px;border-radius:999px;background:rgba(34,197,94,1);display:inline-block;'></span>Risk-on</span>"
    if status == "risk_off":
        return "<span style='display:inline-flex;align-items:center;gap:8px;padding:5px 12px;border-radius:999px;border:1px solid rgba(239,68,68,0.40);background:rgba(239,68,68,0.12);font-size:0.90rem;'><span style='width:11px;height:11px;border-radius:999px;background:rgba(239,68,68,1);display:inline-block;'></span>Risk-off</span>"
    if status == "neutral":
        return "<span style='display:inline-flex;align-items:center;gap:8px;padding:5px 12px;border-radius:999px;border:1px solid rgba(245,158,11,0.40);background:rgba(245,158,11,0.12);font-size:0.90rem;'><span style='width:11px;height:11px;border-radius:999px;background:rgba(245,158,11,1);display:inline-block;'></span>Neutral</span>"
    return "<span style='display:inline-flex;align-items:center;gap:8px;padding:5px 12px;border-radius:999px;border:1px solid rgba(255,255,255,0.10);background:rgba(255,255,255,0.04);font-size:0.90rem;'><span style='width:11px;height:11px;border-radius:999px;background:rgba(255,255,255,0.55);display:inline-block;'></span>n/a</span>"


def fmt_value(val, unit: str, scale: float = 1.0):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "n/a"
    try:
        v = float(val) * float(scale)
    except Exception:
        return "n/a"

    if unit in ("%", "pp"):
        return f"{v:.2f}{unit}"
    if unit == "ratio":
        return f"{v:.3f}"
    if unit == "bn USD":
        return f"{v:.1f} bn"
    if unit == "$":
        return f"${v:,.0f}"
    if unit == "":
        return f"{v:.2f}"
    return f"{v:.2f} {unit}"


def infer_frequency_days(s: pd.Series) -> float:
    if s is None or s.dropna().shape[0] < 10:
        return 1.0
    idx = pd.to_datetime(s.dropna().index)
    diffs = np.diff(idx.values).astype("timedelta64[D]").astype(int)
    if len(diffs) == 0:
        return 1.0
    return float(np.median(diffs))


def pct_change_over_days(series: pd.Series, days: int) -> float:
    if series is None or series.empty:
        return np.nan
    s = series.dropna()
    if s.empty:
        return np.nan
    last_date = s.index.max()
    target_date = last_date - timedelta(days=days)
    past = s[s.index <= target_date]
    if past.empty:
        return np.nan
    past_val = float(past.iloc[-1])
    curr_val = float(s.iloc[-1])
    if np.isnan(past_val) or np.isnan(curr_val) or past_val == 0:
        return np.nan
    return (curr_val / past_val - 1.0) * 100.0


def recent_trend(series: pd.Series) -> dict:
    """Daily series -> 1M; slow series -> 1Q. Returns arrow + pct."""
    if series is None or series.dropna().shape[0] < 10:
        return {"window_label": "n/a", "delta_pct": np.nan, "arrow": "→"}
    freq = infer_frequency_days(series)
    if freq >= 20:
        days, label = 90, "1Q"
    else:
        days, label = 30, "1M"
    d = pct_change_over_days(series, days)
    if np.isnan(d):
        return {"window_label": label, "delta_pct": np.nan, "arrow": "→"}
    arrow = "↑" if d > 0.25 else ("↓" if d < -0.25 else "→")
    return {"window_label": label, "delta_pct": float(d), "arrow": arrow}


# ============================================================
# DATA FETCHERS
# ============================================================
def get_fred_api_key():
    try:
        return st.secrets["FRED_API_KEY"]
    except Exception:
        return None


@st.cache_data(ttl=3600)
def fetch_fred_series(series_id: str, start_date: str) -> pd.Series:
    api_key = get_fred_api_key()
    if api_key is None:
        return pd.Series(dtype=float)

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
    }
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        data = r.json().get("observations", [])
        if not data:
            return pd.Series(dtype=float)

        idx = pd.to_datetime([o["date"] for o in data])
        vals = []
        for o in data:
            try:
                vals.append(float(o["value"]))
            except Exception:
                vals.append(np.nan)

        s = pd.Series(vals, index=idx).replace({".": np.nan}).astype(float).sort_index()
        return s
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=3600)
def fetch_yf_one(ticker: str, start_date: str) -> pd.Series:
    try:
        df = yf.Ticker(ticker).history(start=start_date, auto_adjust=True)
        if df is None or df.empty:
            return pd.Series(dtype=float)
        col = "Close"
        if "Adj Close" in df.columns:
            col = "Adj Close"
        s = df[col].dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None) if getattr(s.index, "tz", None) else pd.to_datetime(s.index)
        return s
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=3600)
def fetch_yf_many(tickers: list[str], start_date: str) -> dict:
    out = {}
    for t in tickers:
        out[t] = fetch_yf_one(t, start_date)
    return out


# ============================================================
# SCORING
# ============================================================
def rolling_percentile_last(hist: pd.Series, latest: float) -> float:
    h = hist.dropna()
    if len(h) < 10 or pd.isna(latest):
        return np.nan
    return float((h <= latest).mean())


def compute_indicator_score(series: pd.Series, direction: int, scoring_mode: str = "z5y"):
    """
    Returns: (score_0_100, signal, latest)
      - z5y: ~5Y z-score (thermometers)
      - pct20y: ~20Y percentile mapping (structural)
    """
    if series is None or series.empty:
        return np.nan, np.nan, np.nan
    s = series.dropna()
    if len(s) < 20:
        return np.nan, np.nan, (np.nan if s.empty else float(s.iloc[-1]))

    latest = float(s.iloc[-1])
    end = s.index.max()

    if scoring_mode == "pct20y":
        start = end - DateOffset(years=20)
        hist = s[s.index >= start]
        if len(hist) < 20:
            hist = s
        p = rolling_percentile_last(hist, latest)  # 0..1
        sig = (p - 0.5) * 4.0                      # [-2,+2]
    else:
        start = end - DateOffset(years=5)
        hist = s[s.index >= start]
        if len(hist) < 10:
            hist = s
        mean = float(hist.mean())
        std = float(hist.std())
        sig = 0.0 if (std == 0 or np.isnan(std)) else (latest - mean) / std

    raw = float(direction) * float(sig)
    raw = float(np.clip(raw, -2.0, 2.0))
    score = (raw + 2.0) / 4.0 * 100.0
    return score, sig, latest


# ============================================================
# PLOTTING
# ============================================================
def plot_premium(series: pd.Series, title: str, ref_line=None, height: int = 320):
    s = series.dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", line=dict(width=2), name=title))

    if ref_line is not None:
        try:
            fig.add_hline(y=float(ref_line), line_width=1, line_dash="dot", opacity=0.7)
        except Exception:
            pass

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=0.98,
        text=f"<b>{title}</b>",
        showarrow=False,
        align="left",
        font=dict(size=14, color="rgba(255,255,255,0.95)"),
        bgcolor="rgba(0,0,0,0.0)"
    )

    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=22, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        showlegend=False,
        font=dict(color="rgba(255,255,255,0.88)"),
    )
    return fig


# ============================================================
# INDICATOR META (macro + crypto lens)
# direction: +1 means higher is MORE supportive for crypto risk-on under our convention.
# ============================================================
INDICATOR_META = {
    # 1) Real cost of money
    "real_10y": {
        "label": "US 10Y TIPS Real Yield",
        "unit": "%",
        "direction": -1,
        "source": "FRED DFII10",
        "scale": 1.0,
        "ref_line": 2.0,
        "scoring_mode": "z5y",
        "guide": {
            "what": "Rendimento reale risk-free: costo opportunità macro #1 per BTC/ETH.",
            "reference": "0–2% ~ neutro; >2% restrittivo; cali rapidi spesso = contesto che migliora (euristica).",
            "how": "- Real yield ↑: stringe condizioni, aumenta costo opportunità → headwind crypto.\n- Real yield ↓: allenta, riduce costo opportunità → tailwind crypto.",
            "crypto": "Quando il risk-free reale rende molto, serve più premio/conviction per detenere crypto.",
        },
    },
    "nominal_10y": {
        "label": "US 10Y Nominal Yield",
        "unit": "%",
        "direction": -1,
        "source": "FRED DGS10",
        "scale": 1.0,
        "ref_line": None,
        "scoring_mode": "z5y",
        "guide": {
            "what": "Tasso nominale benchmark: proxy tightening finanziario.",
            "reference": "Movimenti rapidi al rialzo = tightening di fatto (euristica).",
            "how": "- Yield ↑ veloce: pressione su duration e risk.\n- Yield ↓: allenta condizioni (dipende dal motivo).",
            "crypto": "In rialzo rapido spesso riduce il risk budget: crypto tende a soffrire.",
        },
    },
    "yield_curve_10_2": {
        "label": "US Yield Curve (10Y–2Y)",
        "unit": "pp",
        "direction": +1,
        "source": "FRED DGS10 - DGS2",
        "scale": 1.0,
        "ref_line": 0.0,
        "scoring_mode": "z5y",
        "guide": {
            "what": "Slope di ciclo (late-cycle / recession risk).",
            "reference": "<0 inversione; ritorno >0 spesso dopo easing (euristica).",
            "how": "- Inversione profonda: warning.\n- Dis-inversione ‘buona’ = easing senza stress.\n- Dis-inversione ‘cattiva’ = crash/stress.",
            "crypto": "Crypto preferisce dis-inversione ‘buona’: tagli con stress contenuto.",
        },
    },

    # 2) Macro cycle
    "breakeven_10y": {
        "label": "10Y Breakeven Inflation",
        "unit": "%",
        "direction": -1,
        "source": "FRED T10YIE",
        "scale": 1.0,
        "ref_line": 2.5,
        "scoring_mode": "z5y",
        "guide": {
            "what": "Inflazione attesa: spazio per easing vs higher-for-longer.",
            "reference": "~2–3% ancorato; >3% sticky risk (euristica).",
            "how": "- BE ↑: meno spazio easing.\n- BE ↓: più spazio easing/duration.",
            "crypto": "Sticky inflation = higher-for-longer = headwind per BTC/ETH.",
        },
    },
    "cpi_yoy": {
        "label": "US CPI YoY",
        "unit": "%",
        "direction": -1,
        "source": "FRED CPIAUCSL (YoY calc)",
        "scale": 1.0,
        "ref_line": 3.0,
        "scoring_mode": "z5y",
        "guide": {
            "what": "Inflazione headline YoY: vincolo politico/monetario primario.",
            "reference": "2% target; >3–4% persistente = vincolo (euristica).",
            "how": "- CPI ↓: aumenta spazio easing.\n- CPI ↑: aumenta rischio higher-for-longer.",
            "crypto": "Higher-for-longer comprime multipli e leva: spesso crypto underperforma.",
        },
    },
    "unemployment_rate": {
        "label": "US Unemployment Rate",
        "unit": "%",
        "direction": -1,
        "source": "FRED UNRATE",
        "scale": 1.0,
        "ref_line": None,
        "scoring_mode": "z5y",
        "guide": {
            "what": "Slack nel lavoro: proxy growth downshift.",
            "reference": "Salite rapide spesso = deterioramento ciclo (euristica).",
            "how": "- Unemployment ↑ rapido: risk-off.\n- Stabile: benigno finché inflazione non riparte.",
            "crypto": "Shock growth può causare deleveraging prima dell’easing (timing complesso).",
        },
    },

    # 3) Conditions & stress
    "usd_index": {
        "label": "USD Index (DXY / Broad Proxy)",
        "unit": "",
        "direction": -1,
        "source": "yfinance DX-Y.NYB (fallback FRED DTWEXBGS)",
        "scale": 1.0,
        "ref_line": None,
        "scoring_mode": "z5y",
        "guide": {
            "what": "Forza USD: tightening globale (funding).",
            "reference": "USD ↑ = condizioni più strette; USD ↓ = allenta (euristica).",
            "how": "- USD ↑: stress funding, riduce appetite.\n- USD ↓: allenta, supporta risk-taking.",
            "crypto": "Filtro chiave: USD in trend rialzista spesso frena BTC/ETH.",
        },
    },
    "hy_oas": {
        "label": "US High Yield OAS",
        "unit": "pp",
        "direction": -1,
        "source": "FRED BAMLH0A0HYM2",
        "scale": 1.0,
        "ref_line": 4.5,
        "scoring_mode": "z5y",
        "guide": {
            "what": "Credit stress proxy: premio default/funding.",
            "reference": "<4% benigno; >6–7% stress (euristica).",
            "how": "- Spreads ↑: deleveraging/risk-off.\n- Spreads ↓: appetite ritorna.",
            "crypto": "Crypto soffre quando credito si deteriora (leva e liquidità marginale).",
        },
    },
    "vix": {
        "label": "VIX",
        "unit": "",
        "direction": -1,
        "source": "yfinance ^VIX",
        "scale": 1.0,
        "ref_line": 20.0,
        "scoring_mode": "z5y",
        "guide": {
            "what": "Vol implicita equity: proxy risk premium.",
            "reference": "<15 risk-on; 15–25 normale; >25 stress (euristica).",
            "how": "- VIX ↑: riduce risk budget.\n- VIX ↓: condizioni più permissive.",
            "crypto": "Crypto raramente riparte pulita con VIX alto: prima normalizza risk premium.",
        },
    },
    "spy_trend": {
        "label": "SPY Trend (SPY / 200D MA)",
        "unit": "ratio",
        "direction": +1,
        "source": "yfinance SPY",
        "scale": 1.0,
        "ref_line": 1.0,
        "scoring_mode": "z5y",
        "guide": {
            "what": "Trend proxy: S&P sopra/sotto 200D.",
            "reference": ">1 uptrend; <1 downtrend (euristica).",
            "how": "- >1: risk regime più stabile.\n- <1: regime più fragile.",
            "crypto": "Crypto vive meglio quando equity non è in downtrend strutturale.",
        },
    },
    "hyg_lqd_ratio": {
        "label": "Credit Risk Appetite (HYG / LQD)",
        "unit": "ratio",
        "direction": +1,
        "source": "yfinance HYG, LQD",
        "scale": 1.0,
        "ref_line": None,
        "scoring_mode": "z5y",
        "guide": {
            "what": "Appetite credito: HY vs IG.",
            "reference": "Ratio ↑ = più appetite; ↓ = flight to quality.",
            "how": "- Ratio ↑: sistema assorbe rischio.\n- Ratio ↓: difensivo.",
            "crypto": "Proxy risk appetite ‘di sistema’ che si riflette su asset ad alto beta (crypto).",
        },
    },

    # 4) Liquidity / plumbing
    "fed_balance_sheet": {
        "label": "Fed Balance Sheet (WALCL)",
        "unit": "bn USD",
        "direction": +1,
        "source": "FRED WALCL (millions → bn)",
        "scale": 1.0 / 1000.0,
        "ref_line": None,
        "scoring_mode": "z5y",
        "guide": {
            "what": "Bilancio Fed: proxy liquidità sistema USD.",
            "reference": "Stabile/in salita = tailwind; discesa persistente (QT) = headwind (euristica).",
            "how": "- BS ↑/QT rallenta: più liquidità marginale.\n- BS ↓: drenaggio.",
            "crypto": "Crypto tende a performare meglio quando il drenaggio si ferma/inverte.",
        },
    },
    "rrp": {
        "label": "Fed Overnight RRP",
        "unit": "bn USD",
        "direction": -1,
        "source": "FRED RRPONTSYD",
        "scale": 1.0,
        "ref_line": 0.0,
        "scoring_mode": "z5y",
        "guide": {
            "what": "Reverse repo: cash parcheggiato nella facility.",
            "reference": "RRP ↓ rapido può liberare liquidità marginale (euristica).",
            "how": "- RRP ↓: rilascio potenziale liquidità.\n- RRP ↑: cash ‘stuck’.",
            "crypto": "Spesso è un tailwind tattico quando scende rapidamente.",
        },
    },

    # 5) Fiscal/policy constraints (slow)
    "interest_payments": {
        "label": "US Federal Interest Payments (Quarterly)",
        "unit": "bn USD",
        "direction": -1,
        "source": "FRED A091RC1Q027SBEA",
        "scale": 1.0,
        "ref_line": None,
        "scoring_mode": "pct20y",
        "guide": {
            "what": "Interessi pagati dal governo: pressione debt service.",
            "reference": "In accelerazione = vincolo policy crescente (euristica).",
            "how": "- Sale: riduce flessibilità.\n- Stabilizza: riduce vincolo.",
            "crypto": "Nel lungo può aumentare bias verso policy funding-friendly (ma non timing).",
        },
    },
    "federal_receipts": {
        "label": "US Federal Receipts (Quarterly)",
        "unit": "bn USD",
        "direction": +1,
        "source": "FRED FGRECPT",
        "scale": 1.0,
        "ref_line": None,
        "scoring_mode": "pct20y",
        "guide": {
            "what": "Entrate federali: capacità di assorbire debt service.",
            "reference": "Usato per ratio interest/receipts.",
            "how": "- Receipts ↑: migliora capacità.\n- Receipts ↓: peggiora capacità.",
            "crypto": "Impatto indiretto via term premium/supply.",
        },
    },
    "interest_to_receipts": {
        "label": "Debt Service Stress (Interest / Receipts)",
        "unit": "ratio",
        "direction": -1,
        "source": "Derived",
        "scale": 1.0,
        "ref_line": None,
        "scoring_mode": "pct20y",
        "guide": {
            "what": "Quota entrate consumata da interessi: proxy vincolo politico/fiscale.",
            "reference": "Alto e crescente = vincolo più stringente (euristica).",
            "how": "- Ratio ↑: meno spazio fiscale.\n- Ratio ↓: più spazio.",
            "crypto": "Vincoli crescenti aumentano probabilità di repressione nel lungo (non timing).",
        },
    },
    "deficit_gdp": {
        "label": "Federal Surplus/Deficit (% of GDP)",
        "unit": "%",
        "direction": -1,
        "source": "FRED FYFSGDA188S",
        "scale": 1.0,
        "ref_line": -3.0,
        "scoring_mode": "pct20y",
        "guide": {
            "what": "Saldo fiscale (% GDP). Negativo = deficit.",
            "reference": "Deficit grandi e persistenti = pressione supply Treasury (euristica).",
            "how": "- Più negativo: più supply/funding pressure.\n- Migliora: meno pressione.",
            "crypto": "Se deficit alto + term premium sale, aumenta tightening di mercato (headwind).",
        },
    },
    "term_premium_10y": {
        "label": "US 10Y Term Premium (ACM)",
        "unit": "%",
        "direction": -1,
        "source": "FRED ACMTP10",
        "scale": 1.0,
        "ref_line": None,
        "scoring_mode": "pct20y",
        "guide": {
            "what": "Term premium: compenso extra per detenere duration nominale.",
            "reference": "Se sale: duration hedge meno affidabile (euristica).",
            "how": "- Term premium ↑: tassi più instabili.\n- ↓: duration hedge più pulito.",
            "crypto": "Più instabilità tassi = stress su risk budget (headwind crypto).",
        },
    },

    # 6) External balance
    "current_account_gdp": {
        "label": "US Current Account Balance (% of GDP)",
        "unit": "%",
        "direction": +1,
        "source": "FRED USAB6BLTT02STSAQ",
        "scale": 1.0,
        "ref_line": 0.0,
        "scoring_mode": "pct20y",
        "guide": {
            "what": "Vincolo esterno: dipendenza da capitali esteri.",
            "reference": "Più negativo = vulnerabilità in USD tightening (euristica).",
            "how": "- Più negativo: maggiore vulnerabilità.\n- Verso 0: minore vincolo.",
            "crypto": "In USD shortage globale, crypto tende a soffrire (funding stress).",
        },
    },

    # 7) Gold
    "gold": {
        "label": "Gold (GLD)",
        "unit": "",
        "direction": -1,  # treat as hedge demand proxy
        "source": "yfinance GLD",
        "scale": 1.0,
        "ref_line": None,
        "scoring_mode": "z5y",
        "guide": {
            "what": "Gold: proxy hedge demand (policy/tail risk).",
            "reference": "Breakout spesso = hedge demand, non ‘growth optimism’.",
            "how": "- Gold ↑: hedge demand.\n- Gold ↓: può essere risk-on ‘clean’ (dipende).",
            "crypto": "Gold forte può coesistere con BTC forte in ‘hedge regime’.",
        },
    },

    # 8) Crypto confirmation
    "btc": {
        "label": "Bitcoin (BTC-USD)",
        "unit": "$",
        "direction": +1,
        "source": "yfinance BTC-USD",
        "scale": 1.0,
        "ref_line": None,
        "scoring_mode": "z5y",
        "guide": {
            "what": "Prezzo BTC: conferma di regime (non driver macro).",
            "reference": "Conta il trend più del livello.",
            "how": "- BTC forte con macro che migliora: riallineamento.\n- BTC debole con macro che peggiora: conferma risk-off.",
            "crypto": "BTC spesso guida l’inizio del ciclo crypto.",
        },
    },
    "eth": {
        "label": "Ethereum (ETH-USD)",
        "unit": "$",
        "direction": +1,
        "source": "yfinance ETH-USD",
        "scale": 1.0,
        "ref_line": None,
        "scoring_mode": "z5y",
        "guide": {
            "what": "Prezzo ETH: proxy breadth intra-crypto.",
            "reference": "ETH che conferma spesso = risk-on più ampio.",
            "how": "- ETH accelera dopo BTC: breadth.\n- ETH lagga: leadership concentrata.",
            "crypto": "ETH è un buon check di ‘ampiezza’ del risk-on crypto.",
        },
    },
    "btc_rel_nasdaq": {
        "label": "BTC / Nasdaq (Relative Strength)",
        "unit": "ratio",
        "direction": +1,
        "source": "Derived: BTC-USD / ^IXIC",
        "scale": 1.0,
        "ref_line": None,
        "scoring_mode": "z5y",
        "guide": {
            "what": "Leadership BTC vs tech risk-on (domanda crypto-specific).",
            "reference": "RS ↑ per settimane = segnale forte; RS ↓ = crypto lagga tech.",
            "how": "- RS ↑: domanda crypto-specific.\n- RS ↓: crypto non guida il risk-on.",
            "crypto": "Se RS non migliora, spesso il movimento BTC è ‘beta’ e meno robusto.",
        },
    },
}


# ============================================================
# BLOCKS (crypto-centric)
# ============================================================
BLOCKS = {
    "liquidity": {
        "name": "1) Liquidità (USD plumbing)",
        "weight": 0.22,
        "indicators": ["fed_balance_sheet", "rrp"],
        "desc": "Motore: quando drenaggio si ferma o si inverte, il beta crypto tende a riaprirsi.",
        "group": "Market Thermometers",
    },
    "real_cost": {
        "name": "2) Costo reale del denaro",
        "weight": 0.22,
        "indicators": ["real_10y", "nominal_10y", "yield_curve_10_2"],
        "desc": "Vincolo: real yields/tassi determinano costo opportunità e risk budget.",
        "group": "Market Thermometers",
    },
    "conditions": {
        "name": "3) Condizioni & stress sistemico",
        "weight": 0.24,
        "indicators": ["usd_index", "hy_oas", "vix", "spy_trend", "hyg_lqd_ratio"],
        "desc": "Filtro di regime: USD/credit/vol/trend indicano se il sistema assorbe rischio.",
        "group": "Market Thermometers",
    },
    "macro_cycle": {
        "name": "4) Macro cycle (inflazione & lavoro)",
        "weight": 0.12,
        "indicators": ["breakeven_10y", "cpi_yoy", "unemployment_rate"],
        "desc": "Vincolo policy: disinflazione = più spazio; riaccelerazione = higher-for-longer.",
        "group": "Market Thermometers",
    },
    "crypto_confirm": {
        "name": "5) Crypto confirmation (BTC/ETH/RS)",
        "weight": 0.20,
        "indicators": ["btc", "eth", "btc_rel_nasdaq"],
        "desc": "Conferma: il macro tailwind sta diventando domanda reale (breadth & leadership).",
        "group": "Crypto Layer",
    },
    "policy_constraint": {
        "name": "6) Vincoli strutturali (fiscal/policy)",
        "weight": 0.00,  # informational
        "indicators": ["interest_to_receipts", "deficit_gdp", "term_premium_10y", "interest_payments", "federal_receipts"],
        "desc": "Pressioni lente: influenzano bias di policy e term premium nel tempo.",
        "group": "Structural Constraints",
    },
    "external": {
        "name": "7) External balance",
        "weight": 0.00,
        "indicators": ["current_account_gdp"],
        "desc": "Vulnerabilità in USD tightening (lento ma rilevante).",
        "group": "Structural Constraints",
    },
    "gold_block": {
        "name": "8) Gold (hedge demand)",
        "weight": 0.00,
        "indicators": ["gold"],
        "desc": "Conferma: risk-on growth vs hedge-demand regime.",
        "group": "Structural Constraints",
    },
}


# ============================================================
# OPERATING LINES (crypto)
# ============================================================
def crypto_operating_lines(block_scores: dict, indicator_scores: dict) -> dict:
    gs = block_scores.get("GLOBAL", {}).get("score", np.nan)
    liq = block_scores.get("liquidity", {}).get("score", np.nan)
    rc = block_scores.get("real_cost", {}).get("score", np.nan)
    cond = block_scores.get("conditions", {}).get("score", np.nan)
    conf = block_scores.get("crypto_confirm", {}).get("score", np.nan)

    def _f(x):
        return float(x) if (x is not None and not np.isnan(x)) else np.nan

    gs, liq, rc, cond, conf = map(_f, [gs, liq, rc, cond, conf])

    if np.isnan(gs):
        beta = "n/a"
    elif gs >= 60 and cond >= 55 and rc >= 50:
        beta = "Risk budget ↑ (misurato): macro favorevole + stress contenuto"
    elif gs <= 40 or cond <= 40 or rc <= 40:
        beta = "Risk budget ↓: priorità protezione (regime/stress sfavorevole)"
    else:
        beta = "Neutral: sizing disciplinato, aspettare conferme (BTC/ETH/RS)"

    if np.isnan(liq):
        liquidity = "n/a"
    elif liq >= 60:
        liquidity = "Liquidità: tailwind (plumbing più favorevole / drenaggio ridotto)"
    elif liq <= 40:
        liquidity = "Liquidità: headwind (drenaggio/QT prevale)"
    else:
        liquidity = "Liquidità: mista (nessun segnale dominante)"

    if np.isnan(cond):
        risk_ctl = "n/a"
    elif cond <= 40:
        risk_ctl = "Controllo rischio: difensivo — evitare leva; preferire liquidità"
    elif cond >= 60:
        risk_ctl = "Controllo rischio: regime più pulito — aumentare gradualmente"
    else:
        risk_ctl = "Controllo rischio: neutro — stop disciplinati, evitare overtrade"

    triggers = []
    usd_sc = indicator_scores.get("usd_index", {}).get("score", np.nan)
    hy_sc = indicator_scores.get("hy_oas", {}).get("score", np.nan)
    rs_sc = indicator_scores.get("btc_rel_nasdaq", {}).get("score", np.nan)
    real_sc = indicator_scores.get("real_10y", {}).get("score", np.nan)

    if not np.isnan(real_sc) and real_sc >= 60:
        triggers.append("Real yields: miglioramento → contesto più favorevole per crypto")
    if not np.isnan(usd_sc) and usd_sc <= 40:
        triggers.append("USD forte/tightening → ridurre beta o attendere conferme")
    if not np.isnan(hy_sc) and hy_sc <= 40:
        triggers.append("Credit stress (HY) → priorità difesa")
    if not np.isnan(rs_sc) and rs_sc >= 60:
        triggers.append("BTC leadership (RS vs Nasdaq) → risk-on crypto più credibile")

    if not triggers:
        triggers.append("Trigger: combinazioni (2–3 segnali insieme) più affidabili del singolo indicatore")

    return {"beta": beta, "liquidity": liquidity, "risk_control": risk_ctl, "triggers": triggers[:5]}


# ============================================================
# STREAMLIT-NATIVE TILE (NO HTML layout)
# ============================================================
def indicator_tile(key: str, series: pd.Series, indicator_scores: dict, show_expander=True):
    meta = INDICATOR_META[key]
    sc = indicator_scores.get(key, {})
    score = sc.get("score", np.nan)
    status = sc.get("status", "n/a")
    latest = sc.get("latest", np.nan)

    latest_txt = fmt_value(latest, meta["unit"], meta.get("scale", 1.0))

    tr = recent_trend(series)
    wlab = tr["window_label"]
    d = tr["delta_pct"]
    arrow = tr["arrow"]
    d_txt = "n/a" if np.isnan(d) else f"{d:+.1f}%"

    # Header
    c1, c2 = st.columns([4, 2], gap="small")
    with c1:
        st.subheader(meta["label"])
        st.caption(meta["source"])
    with c2:
        st.markdown(pill_html(status), unsafe_allow_html=True)

    # Metrics
    m1, m2, m3 = st.columns([2.2, 1.5, 1.3], gap="small")
    with m1:
        st.metric("Latest", latest_txt)
    with m2:
        st.metric(f"Trend ({wlab})", f"{arrow} {d_txt}")
    with m3:
        st.metric("Score", "n/a" if np.isnan(score) else f"{score:.0f}")

    # Score bar (native)
    if np.isnan(score):
        st.progress(0)
    else:
        st.progress(int(np.clip(score, 0, 100)))

    # Reference line notes
    ref_line = meta.get("ref_line", None)
    ref_txt = "—" if ref_line is None else str(ref_line)
    ref_note = meta["guide"].get("reference", "—")
    st.caption(f"Reference: {ref_txt} · {ref_note}")

    if show_expander:
        with st.expander("Guide: definizione, lettura, link crypto", expanded=False):
            g = meta["guide"]
            st.markdown(f"**Cosa misura:** {g.get('what','')}")
            st.markdown(f"**Livelli di riferimento:** {g.get('reference','')}")
            st.markdown("**Come leggerlo:**")
            st.markdown(g.get("how", ""))
            st.markdown(f"**Perché conta per crypto (BTC/ETH):** {g.get('crypto','')}")


# ============================================================
# REPORT PROMPT
# ============================================================
REPORT_PROMPT = """
SYSTEM / ROLE
You are a senior macro-crypto strategist writing an internal regime note for a real-money allocator.
You do NOT forecast. You diagnose the regime and translate it into BTC/ETH risk budgeting implications.

You receive a JSON payload with:
- scores (0–100) per block + per indicator
- latest values and short-horizon deltas (1M, 1Q, 1Y)
- optional previous snapshot to compare over time

CRITICAL RULES
- Use ONLY the payload.
- Be causal and disciplined: liquidity, real rates, USD funding, systemic stress first; crypto prices confirm.
- No hype, no narratives, no “moon”. No dramatic language.
- Explicitly call out what is changing vs prior snapshot (if provided).

OUTPUT STRUCTURE (follow exactly)
# Crypto Macro Regime Report
## BTC/ETH risk budgeting view — Internal

Date (UTC): <from payload>

1) Executive Summary (5–8 lines)
- Regime label + why (liquidity / real cost / stress / confirmation)
- What changed vs prior snapshot (if available)

2) Regime Map (macro drivers)
2.1 Liquidity (USD plumbing)
2.2 Real cost of money (real yields / rates)
2.3 Systemic risk appetite (USD, credit, vol, equity trend)
2.4 Crypto confirmation (BTC, ETH, RS vs Nasdaq)

3) Operating Implications (BTC/ETH)
- Risk budget stance (increase / neutral / reduce)
- What to avoid (common failure modes)
- What would confirm a better regime in 2–6 weeks (3–5 triggers)

4) Multi-horizon read (NO forecasting)
- Medium (1Y)
- Short (1Q)
- Very short (1M): what just shifted?

5) Bottom Line (single paragraph)
""".strip()


# ============================================================
# SNAPSHOT / COMPARISON
# ============================================================
def build_snapshot(years_back: int, global_score, global_status, block_scores, indicator_scores, indicators: dict):
    latest_points = []
    for s in indicators.values():
        if s is not None and not s.empty:
            latest_points.append(s.index.max())
    data_max_date = max(latest_points) if latest_points else None

    blocks = []
    for bkey, binfo in BLOCKS.items():
        bscore = block_scores.get(bkey, {}).get("score", np.nan)
        bstatus = block_scores.get(bkey, {}).get("status", "n/a")
        blocks.append({
            "key": bkey,
            "name": binfo["name"],
            "group": binfo["group"],
            "weight": float(binfo["weight"]),
            "score": None if np.isnan(bscore) else float(round(bscore, 2)),
            "status": bstatus,
            "indicators": list(binfo["indicators"]),
        })

    inds = []
    for key, meta in INDICATOR_META.items():
        s = indicators.get(key, pd.Series(dtype=float))
        sc = indicator_scores.get(key, {})
        latest = sc.get("latest", np.nan)
        score = sc.get("score", np.nan)
        status = sc.get("status", "n/a")

        d_1m = pct_change_over_days(s, 30)
        d_1q = pct_change_over_days(s, 90)
        d_1y = pct_change_over_days(s, 365)

        inds.append({
            "key": key,
            "name": meta["label"],
            "source": meta["source"],
            "unit": meta["unit"],
            "scoring_mode": meta.get("scoring_mode", "z5y"),
            "direction": int(meta["direction"]),
            "latest_value_fmt": fmt_value(latest, meta["unit"], meta.get("scale", 1.0)),
            "latest_value_raw": None if np.isnan(latest) else float(latest),
            "score": None if np.isnan(score) else float(round(score, 2)),
            "status": status,
            "delta_1m_pct": None if np.isnan(d_1m) else float(round(d_1m, 3)),
            "delta_1q_pct": None if np.isnan(d_1q) else float(round(d_1q, 3)),
            "delta_1y_pct": None if np.isnan(d_1y) else float(round(d_1y, 3)),
            "reference_line": meta.get("ref_line", None),
            "reference_notes": meta["guide"].get("reference", ""),
        })

    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "generated_at_utc": utc_now_str(),
        "history_years_back": int(years_back),
        "latest_data_date_utc": None if data_max_date is None else str(pd.to_datetime(data_max_date).date()),
        "global_score": None if np.isnan(global_score) else float(round(global_score, 2)),
        "global_status": global_status,
        "blocks": blocks,
        "indicators": inds,
    }


def compare_snapshots(prev: dict, curr: dict) -> pd.DataFrame:
    rows = []
    prev_blocks = {b.get("key"): b for b in prev.get("blocks", []) if isinstance(b, dict)}
    curr_blocks = {b.get("key"): b for b in curr.get("blocks", []) if isinstance(b, dict)}
    for k, cb in curr_blocks.items():
        pb = prev_blocks.get(k, {})
        ps, cs = pb.get("score", None), cb.get("score", None)
        rows.append({
            "Type": "Block",
            "Key": k,
            "Name": cb.get("name", ""),
            "Prev score": ps,
            "Curr score": cs,
            "Δ score": None if (ps is None or cs is None) else float(cs) - float(ps),
            "Prev status": pb.get("status", ""),
            "Curr status": cb.get("status", ""),
        })

    prev_inds = {i.get("key"): i for i in prev.get("indicators", []) if isinstance(i, dict)}
    curr_inds = {i.get("key"): i for i in curr.get("indicators", []) if isinstance(i, dict)}
    for k, ci in curr_inds.items():
        pi = prev_inds.get(k, {})
        ps, cs = pi.get("score", None), ci.get("score", None)
        rows.append({
            "Type": "Indicator",
            "Key": k,
            "Name": ci.get("name", ""),
            "Prev score": ps,
            "Curr score": cs,
            "Δ score": None if (ps is None or cs is None) else float(cs) - float(ps),
            "Prev status": pi.get("status", ""),
            "Curr status": ci.get("status", ""),
            "Curr Δ1M%": ci.get("delta_1m_pct", None),
            "Curr Δ1Q%": ci.get("delta_1q_pct", None),
            "Curr Δ1Y%": ci.get("delta_1y_pct", None),
        })

    df = pd.DataFrame(rows)

    def abs_safe(x):
        try:
            return abs(float(x))
        except Exception:
            return 0.0

    df["absΔ"] = df["Δ score"].apply(abs_safe)
    df = df.sort_values(["Type", "absΔ"], ascending=[True, False]).drop(columns=["absΔ"])
    return df


# ============================================================
# APP
# ============================================================
def main():
    st.title("Crypto Macro Regime")
    st.markdown(
        "<div class='muted'>Dashboard per leggere il <b>regime macro</b> che guida il sentiment crypto (BTC/ETH): "
        "liquidità USD → costo reale del denaro → stress sistemico → conferme crypto. "
        "Obiettivo: chiarezza su <b>forze che schiacciano</b> vs <b>forze che supportano</b> (non timing).</div>",
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        if st.button("🔄 Refresh data (clear cache)"):
            st.cache_data.clear()
            st.rerun()

        years_back = st.slider("History (years)", 5, 30, 15)
        st.caption("Suggerimento: 10–15 anni = buon compromesso per leggere regime macro.")

        st.divider()
        st.subheader("Snapshot previous (optional)")
        prev_snapshot_text = st.text_area("Paste previous snapshot JSON", value="", height=140)

    today = datetime.now(timezone.utc).date()
    start_date = (today - DateOffset(years=years_back)).date().isoformat()

    if get_fred_api_key() is None:
        st.warning("⚠️ Missing `FRED_API_KEY` in secrets: alcune serie FRED potrebbero mancare.")

    # Load data
    with st.spinner("Loading data (FRED + yfinance)..."):
        # FRED
        fred = {
            "real_10y": fetch_fred_series("DFII10", start_date),
            "nominal_10y": fetch_fred_series("DGS10", start_date),
            "dgs2": fetch_fred_series("DGS2", start_date),

            "breakeven_10y": fetch_fred_series("T10YIE", start_date),
            "cpi_index": fetch_fred_series("CPIAUCSL", start_date),
            "unemployment_rate": fetch_fred_series("UNRATE", start_date),

            "hy_oas": fetch_fred_series("BAMLH0A0HYM2", start_date),
            "usd_fred": fetch_fred_series("DTWEXBGS", start_date),

            "fed_balance_sheet": fetch_fred_series("WALCL", start_date),
            "rrp": fetch_fred_series("RRPONTSYD", start_date),

            "interest_payments": fetch_fred_series("A091RC1Q027SBEA", start_date),
            "federal_receipts": fetch_fred_series("FGRECPT", start_date),
            "deficit_gdp": fetch_fred_series("FYFSGDA188S", start_date),
            "term_premium_10y": fetch_fred_series("ACMTP10", start_date),

            "current_account_gdp": fetch_fred_series("USAB6BLTT02STSAQ", start_date),
        }

        indicators = {}

        # Derived yield curve
        if not fred["nominal_10y"].empty and not fred["dgs2"].empty:
            yc = fred["nominal_10y"].to_frame("10y").join(fred["dgs2"].to_frame("2y"), how="inner")
            indicators["yield_curve_10_2"] = (yc["10y"] - yc["2y"]).dropna()
        else:
            indicators["yield_curve_10_2"] = pd.Series(dtype=float)

        # CPI YoY
        if not fred["cpi_index"].empty:
            indicators["cpi_yoy"] = (fred["cpi_index"].pct_change(12) * 100.0).dropna()
        else:
            indicators["cpi_yoy"] = pd.Series(dtype=float)

        # Direct FRED
        indicators["real_10y"] = fred["real_10y"]
        indicators["nominal_10y"] = fred["nominal_10y"]
        indicators["breakeven_10y"] = fred["breakeven_10y"]
        indicators["unemployment_rate"] = fred["unemployment_rate"]
        indicators["hy_oas"] = fred["hy_oas"]
        indicators["fed_balance_sheet"] = fred["fed_balance_sheet"]
        indicators["rrp"] = fred["rrp"]
        indicators["interest_payments"] = fred["interest_payments"]
        indicators["federal_receipts"] = fred["federal_receipts"]
        indicators["deficit_gdp"] = fred["deficit_gdp"]
        indicators["term_premium_10y"] = fred["term_premium_10y"]
        indicators["current_account_gdp"] = fred["current_account_gdp"]

        # Derived interest/receipts
        ip = indicators.get("interest_payments", pd.Series(dtype=float))
        fr = indicators.get("federal_receipts", pd.Series(dtype=float))
        if (ip is not None and fr is not None) and (not ip.empty) and (not fr.empty):
            join = ip.to_frame("interest").join(fr.to_frame("receipts"), how="inner").dropna()
            join = join[join["receipts"] != 0]
            indicators["interest_to_receipts"] = (join["interest"] / join["receipts"]).dropna()
        else:
            indicators["interest_to_receipts"] = pd.Series(dtype=float)

        # YFinance
        yf_map = fetch_yf_many(
            ["DX-Y.NYB", "^VIX", "SPY", "HYG", "LQD", "GLD", "BTC-USD", "ETH-USD", "^IXIC"],
            start_date
        )

        # USD (fallback to FRED broad)
        dxy = yf_map.get("DX-Y.NYB", pd.Series(dtype=float))
        if dxy is None or dxy.empty:
            dxy = fred["usd_fred"]
        indicators["usd_index"] = dxy

        indicators["vix"] = yf_map.get("^VIX", pd.Series(dtype=float))

        spy = yf_map.get("SPY", pd.Series(dtype=float))
        if spy is not None and not spy.empty:
            ma200 = spy.rolling(200).mean()
            indicators["spy_trend"] = (spy / ma200).dropna()
        else:
            indicators["spy_trend"] = pd.Series(dtype=float)

        hyg = yf_map.get("HYG", pd.Series(dtype=float))
        lqd = yf_map.get("LQD", pd.Series(dtype=float))
        if hyg is not None and lqd is not None and (not hyg.empty) and (not lqd.empty):
            joined = hyg.to_frame("HYG").join(lqd.to_frame("LQD"), how="inner").dropna()
            indicators["hyg_lqd_ratio"] = (joined["HYG"] / joined["LQD"]).dropna()
        else:
            indicators["hyg_lqd_ratio"] = pd.Series(dtype=float)

        indicators["gold"] = yf_map.get("GLD", pd.Series(dtype=float))
        indicators["btc"] = yf_map.get("BTC-USD", pd.Series(dtype=float))
        indicators["eth"] = yf_map.get("ETH-USD", pd.Series(dtype=float))

        # BTC / Nasdaq RS
        btc = indicators["btc"]
        ixic = yf_map.get("^IXIC", pd.Series(dtype=float))
        if btc is not None and ixic is not None and (not btc.empty) and (not ixic.empty):
            rs = btc.to_frame("btc").join(ixic.to_frame("ixic"), how="inner").dropna()
            rs = rs[rs["ixic"] != 0]
            indicators["btc_rel_nasdaq"] = (rs["btc"] / rs["ixic"]).dropna()
        else:
            indicators["btc_rel_nasdaq"] = pd.Series(dtype=float)

    # Score indicators
    indicator_scores = {}
    for key, meta in INDICATOR_META.items():
        series = indicators.get(key, pd.Series(dtype=float))
        mode = meta.get("scoring_mode", "z5y")
        score, sig, latest = compute_indicator_score(series, meta["direction"], scoring_mode=mode)
        indicator_scores[key] = {
            "score": score,
            "signal": sig,
            "latest": latest,
            "status": classify_status(score),
            "mode": mode
        }

    # Score blocks + global
    block_scores = {}
    global_score = 0.0
    w_used = 0.0
    for bkey, binfo in BLOCKS.items():
        vals = []
        for ikey in binfo["indicators"]:
            sc = indicator_scores.get(ikey, {}).get("score", np.nan)
            if not np.isnan(sc):
                vals.append(sc)
        bscore = float(np.mean(vals)) if vals else np.nan
        block_scores[bkey] = {"score": bscore, "status": classify_status(bscore)}

        if binfo["weight"] > 0 and not np.isnan(bscore):
            global_score += bscore * binfo["weight"]
            w_used += binfo["weight"]

    global_score = (global_score / w_used) if w_used > 0 else np.nan
    global_status = classify_status(global_score)
    block_scores["GLOBAL"] = {"score": global_score, "status": global_status}

    # Data freshness
    latest_points = []
    for s in indicators.values():
        if s is not None and not s.empty:
            latest_points.append(s.index.max())
    data_max_date = max(latest_points) if latest_points else None

    ops = crypto_operating_lines(block_scores, indicator_scores)

    tabs = st.tabs(["Overview", "Wallboard", "Deep dive", "What changed", "Report & Snapshots"])

    # ------------------------------------------------------------
    # OVERVIEW (immediate crypto takeaways)
    # ------------------------------------------------------------
    with tabs[0]:
        st.markdown("### Regime snapshot (crypto lens)")
        c1, c2 = st.columns([2, 1], gap="small")
        with c1:
            gs_txt = "n/a" if np.isnan(global_score) else f"{global_score:.1f}"
            st.metric("Global Crypto-Macro Score (0–100)", gs_txt)
            st.markdown(pill_html(global_status), unsafe_allow_html=True)
            if not np.isnan(global_score):
                st.progress(int(np.clip(global_score, 0, 100)))
            else:
                st.progress(0)

            st.markdown("#### Operating lines (BTC/ETH)")
            st.write(f"**Crypto beta:** {ops['beta']}")
            st.write(f"**Liquidità:** {ops['liquidity']}")
            st.write(f"**Risk control:** {ops['risk_control']}")

            st.markdown("#### 2–6 week triggers")
            for t in ops["triggers"]:
                st.write(f"- {t}")

        with c2:
            st.markdown("### Component scores")
            for bkey in ["liquidity", "real_cost", "conditions", "macro_cycle", "crypto_confirm"]:
                b = block_scores.get(bkey, {})
                sc = b.get("score", np.nan)
                stt = b.get("status", "n/a")
                st.write(f"**{BLOCKS[bkey]['name']}** — {status_label(stt)} ({'n/a' if np.isnan(sc) else f'{sc:.1f}'})")

            st.markdown("---")
            st.caption(f"Now: **{utc_now_str()}**")
            st.caption(f"Latest datapoint: **{('n/a' if data_max_date is None else str(pd.to_datetime(data_max_date).date()))}**")
            st.caption(f"History start: **{start_date}**")

        with st.expander("How to read (crypto-first, causal order)", expanded=False):
            st.markdown(
                """
- **Prima:** Liquidità USD (plumbing). Se il sistema drena, il beta crypto si comprime.
- **Secondo:** Costo reale del denaro (real yields). È il costo opportunità diretto.
- **Terzo:** Stress sistemico (USD, credit, vol, trend). Se sale, si riduce il risk budget.
- **Quarto:** Conferme crypto (BTC, ETH, RS). Non sono driver macro, ma dicono se la domanda è reale.

**Scores:** 0–100 (euristica).  
>60 = comportamento più risk-on (secondo convenzione del singolo indicatore) · 40–60 = neutro · <40 = risk-off.
                """.strip()
            )

    # ------------------------------------------------------------
    # WALLBOARD (grouped, native tiles)
    # ------------------------------------------------------------
    with tabs[1]:
        st.markdown("### Wallboard")
        st.markdown("<div class='muted'>Gruppi macro → lettura rapida → guide per indicatori. (No HTML tiles: fix rendering.)</div>", unsafe_allow_html=True)

        def render_block(block_key: str):
            info = BLOCKS[block_key]
            bsc = block_scores.get(block_key, {}).get("score", np.nan)
            bst = block_scores.get(block_key, {}).get("status", "n/a")

            with st.expander(f"{info['name']} — {status_label(bst)} ({'n/a' if np.isnan(bsc) else f'{bsc:.1f}'})", expanded=True if block_key in ["liquidity", "real_cost", "conditions"] else False):
                st.caption(info["desc"])
                for ikey in info["indicators"]:
                    s = indicators.get(ikey, pd.Series(dtype=float))
                    if s is None or s.empty:
                        st.warning(f"Missing data: {INDICATOR_META[ikey]['label']}")
                    else:
                        indicator_tile(ikey, s, indicator_scores, show_expander=True)
                    st.divider()

        for bk in ["liquidity", "real_cost", "conditions", "macro_cycle", "crypto_confirm", "policy_constraint", "external", "gold_block"]:
            render_block(bk)

    # ------------------------------------------------------------
    # DEEP DIVE (scrollable: each section expandable; charts + guide)
    # ------------------------------------------------------------
    with tabs[2]:
        st.markdown("### Deep dive")
        st.markdown("<div class='muted'>Sezioni espandibili (possono rimanere aperte). Grafici con stile consistente.</div>", unsafe_allow_html=True)

        for bk, binfo in BLOCKS.items():
            bsc = block_scores.get(bk, {}).get("score", np.nan)
            bst = block_scores.get(bk, {}).get("status", "n/a")
            with st.expander(f"{binfo['name']} — {status_label(bst)} ({'n/a' if np.isnan(bsc) else f'{bsc:.1f}'})", expanded=False):
                st.caption(binfo["desc"])
                for ikey in binfo["indicators"]:
                    meta = INDICATOR_META[ikey]
                    s = indicators.get(ikey, pd.Series(dtype=float))
                    sc = indicator_scores.get(ikey, {})
                    score = sc.get("score", np.nan)
                    stt = sc.get("status", "n/a")
                    latest = sc.get("latest", np.nan)

                    head = st.columns([4, 2], gap="small")
                    with head[0]:
                        st.subheader(meta["label"])
                        st.caption(meta["source"])
                    with head[1]:
                        st.markdown(pill_html(stt), unsafe_allow_html=True)

                    p1, p2, p3 = st.columns([2.2, 1.4, 1.4], gap="small")
                    with p1:
                        st.metric("Latest", fmt_value(latest, meta["unit"], meta.get("scale", 1.0)))
                    tr = recent_trend(s)
                    with p2:
                        d = tr["delta_pct"]
                        st.metric(f"Trend ({tr['window_label']})", "n/a" if np.isnan(d) else f"{tr['arrow']} {d:+.1f}%")
                    with p3:
                        st.metric("Score", "n/a" if np.isnan(score) else f"{score:.0f}")

                    if s is None or s.empty:
                        st.warning("Missing data for this indicator in the selected history window.")
                    else:
                        fig = plot_premium(s, meta["label"], ref_line=meta.get("ref_line", None), height=340)
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                    with st.expander("Indicator guide", expanded=False):
                        g = meta["guide"]
                        st.markdown(f"**Cosa misura:** {g.get('what','')}")
                        st.markdown(f"**Reference:** {g.get('reference','')}")
                        st.markdown("**Come leggerlo:**")
                        st.markdown(g.get("how",""))
                        st.markdown(f"**Link crypto:** {g.get('crypto','')}")
                    st.divider()

    # ------------------------------------------------------------
    # WHAT CHANGED (1M emphasis + watchlist)
    # ------------------------------------------------------------
    with tabs[3]:
        st.markdown("### What changed")
        st.markdown("<div class='muted'>Focus: 1M (brevissimo) + contesto 1Q e 1Y. Watchlist segnala movimenti e vicinanza a soglie.</div>", unsafe_allow_html=True)

        rows = []
        for key, meta in INDICATOR_META.items():
            s = indicators.get(key, pd.Series(dtype=float))
            if s is None or s.empty:
                continue

            d1m = pct_change_over_days(s, 30)
            d1q = pct_change_over_days(s, 90)
            d1y = pct_change_over_days(s, 365)

            sc = indicator_scores.get(key, {})
            score = sc.get("score", np.nan)
            stt = sc.get("status", "n/a")
            mode = meta.get("scoring_mode", "z5y")

            # Attention heuristic: (a) proximity to 40/60 + (b) move size 1M
            if np.isnan(score):
                prox = 0.0
            else:
                prox = max(0.0, 20.0 - min(abs(score - 40), abs(score - 60))) / 20.0  # 0..1
            move = 0.0 if np.isnan(d1m) else min(1.0, abs(d1m) / 10.0)  # 10% = max
            attention = 0.55 * prox + 0.45 * move
            watch = "WATCH" if attention >= 0.55 else ""

            rows.append({
                "Indicator": meta["label"],
                "Scoring": mode,
                "Regime": status_label(stt),
                "Score": np.nan if np.isnan(score) else round(score, 1),
                "Δ 1M %": np.nan if np.isnan(d1m) else round(d1m, 2),
                "Δ 1Q %": np.nan if np.isnan(d1q) else round(d1q, 2),
                "Δ 1Y %": np.nan if np.isnan(d1y) else round(d1y, 2),
                "Watchlist": watch,
                "Attention": round(attention, 2),
            })

        if not rows:
            st.info("No sufficient data to compute changes.")
        else:
            df = pd.DataFrame(rows)
            wl = df[df["Watchlist"] == "WATCH"].sort_values("Attention", ascending=False).head(8)

            if not wl.empty:
                st.markdown("#### Watchlist (movers / threshold proximity)")
                for _, r in wl.iterrows():
                    st.write(
                        f"- **{r['Indicator']}** · {r['Regime']} · Score {r['Score']} · Δ1M {r['Δ 1M %']}%"
                    )

            st.markdown("#### Full table")
            st.dataframe(df.sort_values(["Watchlist", "Attention"], ascending=[True, False]).reset_index(drop=True), use_container_width=True)
            st.caption("Nota: le % sono calcolate su osservazioni disponibili (frequenze diverse).")

    # ------------------------------------------------------------
    # REPORT & SNAPSHOTS (stable schema, comparison)
    # ------------------------------------------------------------
    with tabs[4]:
        st.markdown("### Report & Snapshots")
        st.markdown("<div class='muted'>Genera uno snapshot JSON stabile e comparabile nel tempo. Puoi incollare uno snapshot precedente (sidebar) per confronto.</div>", unsafe_allow_html=True)

        snap = build_snapshot(years_back, global_score, global_status, block_scores, indicator_scores, indicators)

        # Try load previous snapshot
        prev = None
        if "prev_snapshot_text" in st.session_state:
            pass
        prev_txt = st.sidebar.session_state.get("Paste previous snapshot JSON", None) if False else None

        # Use the sidebar variable we created
        # Streamlit keeps local state; easiest: re-read from widget by key is not needed; we have variable in closure
        # We'll parse below from variable name in main scope
        # (the widget lives in sidebar, but string is in the outer scope via closure in this script)
        # We'll just read it from st.session_state if available; if not, ignore.
        prev_snapshot_str = st.session_state.get("previous_snapshot_json", None)
        # In case state key differs, also attempt to parse from the sidebar text area label
        # If user pasted, Streamlit assigns to a generated key; safer: we parse from the variable captured in sidebar.
        # We'll instead store it explicitly:
        # (We can't retroactively set it here; but we can parse from a new text_area shown below if needed.)

        st.markdown("#### Current snapshot (JSON)")
        st.code(json.dumps(snap, indent=2), language="json")

        st.download_button(
            "Download snapshot.json",
            data=json.dumps(snap, indent=2).encode("utf-8"),
            file_name=f"snapshot_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

        st.divider()

        st.markdown("#### Compare vs previous snapshot (paste below)")
        prev_input = st.text_area("Previous snapshot JSON", value="", height=140)
        if prev_input.strip():
            try:
                prev = json.loads(prev_input)
                dfc = compare_snapshots(prev, snap)
                st.success("Comparison ready.")
                st.dataframe(dfc, use_container_width=True)
            except Exception as e:
                st.error(f"Could not parse/compare previous snapshot. Error: {e}")

        st.divider()

        st.markdown("#### One-shot copy/paste for ChatGPT (prompt + payload)")
        payload = {
            "prompt_version": "crypto_macro_report_v1",
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "current_snapshot": snap,
            "previous_snapshot": prev if isinstance(prev, dict) else None,
        }
        one_shot = (
            "### COPY/PASTE (PROMPT)\n\n"
            + REPORT_PROMPT
            + "\n\n---\n\n"
            + "### PAYLOAD (JSON)\n\n```json\n"
            + json.dumps(payload, indent=2)
            + "\n```\n"
        )
        st.code(one_shot, language="markdown")


if __name__ == "__main__":
    main()
