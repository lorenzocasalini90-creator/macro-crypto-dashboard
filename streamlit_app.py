# streamlit_app.py
# ============================================================
# Macro → Crypto Regime Dashboard (Premium, Mobile-first)
# - Focus: macro fundamentals driving crypto (BTC + ETH)
# - Robust wallboard tiles (NO raw HTML div grids → avoids leaked <div> text)
# - Deep dive: scrollable, all sections expandable and can stay open
# - “What changed” + Watchlist
# - Report generation: stable schema + snapshot history + comparison
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import json
from datetime import datetime, timedelta, timezone
from pandas.tseries.offsets import DateOffset
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG (mobile-friendly)
# ============================================================
st.set_page_config(
    page_title="Crypto Macro Regime Dashboard",
    layout="centered",                 # mobile-first
    initial_sidebar_state="collapsed", # keep clean on mobile
)

# ============================================================
# PREMIUM, SOBER CSS (avoid heavy HTML layouts; keep subtle styling)
# ============================================================
st.markdown(
    """
<style>
  :root{
    --bg:#0b0f19;
    --card:#0f1629;
    --border:rgba(255,255,255,0.10);
    --muted:rgba(255,255,255,0.70);
    --text:rgba(255,255,255,0.94);

    --good:rgba(34,197,94,1);
    --warn:rgba(245,158,11,1);
    --bad:rgba(239,68,68,1);

    --accent:#60a5fa;       /* sober blue */
    --accentSoft:rgba(96,165,250,0.16);
  }

  .stApp {
    background: radial-gradient(1200px 700px at 20% 0%, #121a33 0%, #0b0f19 45%, #0b0f19 100%);
    color: var(--text);
  }

  .block-container { padding-top: 0.9rem; padding-bottom: 1.8rem; max-width: 980px; }

  h1, h2, h3 { color: var(--text); letter-spacing: -0.02em; }
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

  /* Subtle cards (used via markdown only, not required for wallboard tiles) */
  .softCard{
    background: linear-gradient(180deg, rgba(255,255,255,0.055) 0%, rgba(255,255,255,0.03) 100%);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 14px 14px 12px 14px;
    box-shadow: 0 10px 28px rgba(0,0,0,0.22);
  }

  .pill{
    display:inline-flex;
    align-items:center;
    gap:8px;
    padding: 5px 12px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: rgba(255,255,255,0.04);
    font-size: 0.90rem;
    color: rgba(255,255,255,0.94);
    white-space: nowrap;
  }
  .dot{ width: 11px; height: 11px; border-radius: 999px; display:inline-block; }

  .pill.good{ border-color: rgba(34,197,94,0.40); background: rgba(34,197,94,0.12); }
  .pill.warn{ border-color: rgba(245,158,11,0.40); background: rgba(245,158,11,0.12); }
  .pill.bad { border-color: rgba(239,68,68,0.40); background: rgba(239,68,68,0.12); }

  /* Dataframe */
  .stDataFrame { border: 1px solid var(--border); border-radius: 12px; overflow:hidden; }

  code { color: rgba(255,255,255,0.88); }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# HELPERS: pills + formatting
# ============================================================

def classify_status(score: float) -> str:
    if np.isnan(score):
        return "n/a"
    if score > 60:
        return "risk_on"
    if score < 40:
        return "risk_off"
    return "neutral"

def status_label(status: str) -> str:
    if status == "risk_on":
        return "Risk-on"
    if status == "risk_off":
        return "Risk-off"
    if status == "neutral":
        return "Neutral"
    return "n/a"

def pill_html(status: str) -> str:
    if status == "risk_on":
        return "<span class='pill good'><span class='dot' style='background:var(--good)'></span>Risk-on</span>"
    if status == "risk_off":
        return "<span class='pill bad'><span class='dot' style='background:var(--bad)'></span>Risk-off</span>"
    if status == "neutral":
        return "<span class='pill warn'><span class='dot' style='background:var(--warn)'></span>Neutral</span>"
    return "<span class='pill'><span class='dot' style='background:rgba(255,255,255,0.5)'></span>n/a</span>"

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
    past_val = past.iloc[-1]
    curr_val = s.iloc[-1]
    if pd.isna(past_val) or pd.isna(curr_val) or past_val == 0:
        return np.nan
    return (curr_val / past_val - 1.0) * 100.0

def recent_trend(series: pd.Series) -> dict:
    """
    Returns dict with:
      - window_label (e.g., "30d" or "1Q")
      - delta_pct
      - arrow
    """
    if series is None or series.dropna().shape[0] < 10:
        return {"window_label": "n/a", "delta_pct": np.nan, "arrow": "→"}
    freq = infer_frequency_days(series)
    if freq >= 20:  # monthly/quarterly-ish
        days = 90
        label = "1Q"
    else:
        days = 30
        label = "30d"
    d = pct_change_over_days(series, days)
    if np.isnan(d):
        return {"window_label": label, "delta_pct": np.nan, "arrow": "→"}
    arrow = "↑" if d > 0.25 else ("↓" if d < -0.25 else "→")
    return {"window_label": label, "delta_pct": d, "arrow": arrow}

def score_progress(score: float):
    """Render score 0-100 as Streamlit-native progress (no HTML)."""
    if np.isnan(score):
        st.progress(0)
        st.caption("Score: n/a")
        return
    v = int(np.clip(score, 0, 100))
    st.progress(v)
    st.caption(f"Score: {v}/100")

# ============================================================
# DATA: FRED + yfinance
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
# - Market thermometers: z5y
# - Structural constraints: pct20y (percentile mapping)
# Direction: +1 means higher is better for crypto risk-on, -1 means higher is worse
# ============================================================

def rolling_percentile_last(hist: pd.Series, latest: float) -> float:
    h = hist.dropna()
    if len(h) < 10 or pd.isna(latest):
        return np.nan
    return float((h <= latest).mean())

def compute_indicator_score(series: pd.Series, direction: int, scoring_mode: str = "z5y"):
    """
    Returns: (score_0_100, signal, latest)
      - signal: z-score for z5y, or mapped percentile for pct20y in [-2,+2] space
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
        sig = (p - 0.5) * 4.0  # 0->-2, 0.5->0, 1->+2
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
# PLOTTING (dark, readable, titles inside chart)
# ============================================================

def plot_premium(series: pd.Series, title: str, ref_line=None, height: int = 320):
    s = series.dropna()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=s.index,
            y=s.values,
            mode="lines",
            line=dict(width=2),
            name=title,
        )
    )

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
# CRYPTO-MACRO INDICATORS (BTC + ETH explicitly)
# ============================================================

INDICATOR_META = {
    # 1) LIQUIDITY (top priority for crypto)
    "fed_balance_sheet": {
        "label": "Fed Balance Sheet (WALCL)",
        "unit": "bn USD",
        "direction": +1,
        "source": "FRED WALCL (millions → bn)",
        "scale": 1.0 / 1000.0,
        "ref_line": None,
        "scoring_mode": "z5y",
        "expander": {
            "what": "Total Fed assets: proxy for USD system liquidity. Crypto behaves like a high-beta liquidity asset.",
            "reference": "BS ↑ (QE / less QT) = tailwind; BS ↓ persistent (QT) = headwind (euristica).",
            "interpretation": "- Se il bilancio smette di scendere o torna a salire, spesso migliora il “bid” sugli asset risk.\n- Se scende stabilmente, la liquidità marginale tende a ridursi → più fragilità.",
            "crypto_link": "BTC/ETH sono ‘opzioni sulla liquidità futura’: più liquidità → più risk-taking → più domanda per duration/convexity.",
            "so_what": "Quando questo migliora insieme a real yields in calo e USD che indebolisce, storicamente aumenta la probabilità di regime favorevole alle crypto.",
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
        "expander": {
            "what": "Cash parked at the Fed reverse repo facility. Falling RRP can release marginal liquidity.",
            "reference": "RRP ↓ rapido può liberare marginal liquidity; RRP ↑ = cash ‘stuck’ (euristica).",
            "interpretation": "- In calo: più cash può rientrare su T-bills/credit/risk.\n- In aumento: condizioni marginali più strette.",
            "crypto_link": "Supporto tattico quando scende velocemente (più risk appetite).",
            "so_what": "Se RRP scende e contemporaneamente BTC/ETH riprendono leadership, il contesto risk-on diventa più credibile.",
        },
    },

    # 2) REAL COST OF MONEY (kryptonite/support)
    "real_10y": {
        "label": "US 10Y Real Yield (TIPS)",
        "unit": "%",
        "direction": -1,
        "source": "FRED DFII10",
        "scale": 1.0,
        "ref_line": 2.0,
        "scoring_mode": "z5y",
        "expander": {
            "what": "Real (inflation-adjusted) risk-free return: the real price of money/time.",
            "reference": "0–2% neutro; >2% restrittivo; cali rapidi = risk-on crypto (euristiche).",
            "interpretation": "- Real yield ↑: i rendimenti reali competono con asset non-cashflow → pressione su BTC/ETH.\n- Real yield ↓: riduce l’attrattività del risk-free → supporto per duration/convexity.",
            "crypto_link": "Variabile macro #1 per crypto: quando il risk-free reale rende tanto, il ‘costo opportunità’ delle crypto cresce.",
            "so_what": "Se scende in modo persistente (o rapidamente), aumenta la probabilità di ripartenza strutturale delle crypto (non timing perfetto, ma vento in poppa).",
        },
    },
    "dxy": {
        "label": "USD Index (DXY proxy)",
        "unit": "",
        "direction": -1,
        "source": "yfinance DX-Y.NYB (fallback FRED DTWEXBGS)",
        "scale": 1.0,
        "ref_line": None,
        "scoring_mode": "z5y",
        "expander": {
            "what": "USD strength proxy. Strong USD tightens global funding conditions.",
            "reference": "USD ↑ = tightening; USD ↓ = easing (euristica).",
            "interpretation": "- USD forte spesso comprime la liquidità globale e gli asset risk.\n- USD debole aiuta il risk-taking e condizioni più ‘facili’.",
            "crypto_link": "BTC/ETH tendono a respirare con USD in indebolimento (non sempre, ma è un filtro utile).",
            "so_what": "Se USD rompe trend rialzista e contemporaneamente real yields calano, il contesto per crypto migliora sensibilmente.",
        },
    },

    # 3) RISK APPETITE / STRESS (confirmation)
    "vix": {
        "label": "VIX (equity vol)",
        "unit": "",
        "direction": -1,
        "source": "yfinance ^VIX",
        "scale": 1.0,
        "ref_line": 20.0,
        "scoring_mode": "z5y",
        "expander": {
            "what": "Implied volatility (S&P 500). High VIX usually = de-risking regime.",
            "reference": "<15 risk-on; 15–25 normale; >25 stress (euristica).",
            "interpretation": "- VIX alto = risk premia alti e riduzione leva.\n- VIX basso = condizioni più permissive.",
            "crypto_link": "Crypto raramente riparte ‘pulita’ con VIX elevato: prima si normalizza il risk appetite.",
            "so_what": "Se VIX scende mentre BTC/ETH tengono o salgono, il mercato sta ‘riaprendo’ la finestra risk-on.",
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
        "expander": {
            "what": "Credit stress proxy (default risk premium).",
            "reference": "<4% benign; >6–7% stress (euristica).",
            "interpretation": "- Spread ↑ = funding stress / risk-off.\n- Spread ↓ = risk appetite / easier credit.",
            "crypto_link": "Quando il credito si deteriora, di solito aumenta la probabilità di deleveraging anche su crypto.",
            "so_what": "Per un bull credibile su crypto, vuoi HY spreads stabili o in restringimento (non necessariamente ai minimi).",
        },
    },

    # 4) CRYPTO CONFIRMATION (BTC + ETH)
    "btc": {
        "label": "Bitcoin (BTC-USD)",
        "unit": "$",
        "direction": +1,
        "source": "yfinance BTC-USD",
        "scale": 1.0,
        "ref_line": None,
        "scoring_mode": "z5y",
        "expander": {
            "what": "BTC price level (as confirmation, not the driver).",
            "reference": "Più importante il trend che il livello assoluto.",
            "interpretation": "- BTC in uptrend mentre macro migliora = regime che si riallinea.\n- BTC debole con macro che peggiora = conferma risk-off.",
            "crypto_link": "BTC spesso guida il beta crypto in fase iniziale; ETH tende a seguire o confermare.",
            "so_what": "Usalo per capire se il mercato ‘accetta’ il macro tailwind o se è ancora in fase di deleveraging.",
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
        "expander": {
            "what": "ETH price level (confirmation + breadth).",
            "reference": "ETH che recupera leadership spesso segnala miglioramento di breadth/risk appetite.",
            "interpretation": "- ETH forte vs BTC può indicare ritorno di risk appetite intra-crypto.\n- ETH debole vs BTC spesso indica cautela e qualità.",
            "crypto_link": "ETH è un buon ‘breadth check’ oltre BTC.",
            "so_what": "Se ETH segue BTC al rialzo mentre macro migliora, aumenta la probabilità di regime risk-on più ampio sul comparto.",
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
        "expander": {
            "what": "BTC relative strength vs Nasdaq. Measures crypto-specific bid vs generic tech risk-on.",
            "reference": "RS ↑ = BTC leadership; RS ↓ = crypto lagging tech.",
            "interpretation": "- RS in salita: BTC sta sovraperformando → segnale forte.\n- RS in discesa: crypto è ‘late’ o fragile.",
            "crypto_link": "Quando BTC sovraperforma anche con Nasdaq flat, spesso è un segnale di domanda propria (conviction).",
            "so_what": "Per aumentare aggressività su crypto, vuoi vedere RS in miglioramento per settimane, non 2 giorni.",
        },
    },
    "eth_rel_nasdaq": {
        "label": "ETH / Nasdaq (Relative Strength)",
        "unit": "ratio",
        "direction": +1,
        "source": "Derived: ETH-USD / ^IXIC",
        "scale": 1.0,
        "ref_line": None,
        "scoring_mode": "z5y",
        "expander": {
            "what": "ETH relative strength vs Nasdaq: breadth/risk appetite confirmation.",
            "reference": "RS ↑ = ETH leadership; RS ↓ = crypto lagging tech.",
            "interpretation": "- RS ETH ↑: indica ritorno di appetite intra-crypto.\n- RS ETH ↓: indica cautela e leadership concentrata.",
            "crypto_link": "Se ETH RS migliora insieme a BTC RS, spesso il regime risk-on crypto è più ‘vero’ e meno fragile.",
            "so_what": "Buona metrica per capire se il movimento si sta allargando oltre BTC.",
        },
    },
}

# Blocks: explicitly crypto macro regime map
BLOCKS = {
    "liquidity": {
        "name": "1) Liquidità",
        "weight": 0.30,
        "indicators": ["fed_balance_sheet", "rrp"],
        "desc": "Il motore. Crypto tende a performare quando la liquidità USD smette di drenare o torna a migliorare.",
    },
    "real_cost": {
        "name": "2) Costo reale del denaro",
        "weight": 0.30,
        "indicators": ["real_10y", "dxy"],
        "desc": "Il vincolo. Real yields e USD forte aumentano il costo opportunità e stringono le condizioni globali.",
    },
    "risk_appetite": {
        "name": "3) Risk appetite & stress",
        "weight": 0.20,
        "indicators": ["vix", "hy_oas"],
        "desc": "La conferma di regime. Vol e credito dicono se il sistema sta de-riskando o riaprendo il budget rischio.",
    },
    "crypto_confirm": {
        "name": "4) Crypto confirmation (BTC + ETH)",
        "weight": 0.20,
        "indicators": ["btc", "eth", "btc_rel_nasdaq", "eth_rel_nasdaq"],
        "desc": "La conferma di mercato. Prezzo e leadership (RS) dicono se il tailwind macro sta diventando domanda reale su crypto.",
    },
}

# ============================================================
# OPERATING LINES (crypto-focused)
# ============================================================

def crypto_operating_lines(block_scores: dict, indicator_scores: dict):
    """
    Decision-friendly output:
    - Core exposure (BTC/ETH) sizing stance
    - Risk add vs reduce
    - What to watch next (2-6 weeks)
    """
    def _v(x):
        if np.isnan(x):
            return None
        return float(x)

    gs = _v(block_scores.get("GLOBAL", {}).get("score", np.nan))
    liq = _v(block_scores.get("liquidity", {}).get("score", np.nan))
    rc  = _v(block_scores.get("real_cost", {}).get("score", np.nan))
    ra  = _v(block_scores.get("risk_appetite", {}).get("score", np.nan))
    cc  = _v(block_scores.get("crypto_confirm", {}).get("score", np.nan))

    # Quick “forces” lens
    forces_headwind = []
    forces_tailwind = []

    # Macro forces (simple)
    if liq is not None and liq < 45: forces_headwind.append("Liquidità in peggioramento")
    if liq is not None and liq > 55: forces_tailwind.append("Liquidità in miglioramento")

    if rc is not None and rc < 45: forces_headwind.append("Costo reale / USD restrittivi")
    if rc is not None and rc > 55: forces_tailwind.append("Costo reale / USD più favorevoli")

    if ra is not None and ra < 45: forces_headwind.append("Stress (vol/credit) elevato")
    if ra is not None and ra > 55: forces_tailwind.append("Risk appetite più costruttivo")

    if cc is not None and cc < 45: forces_headwind.append("Crypto non conferma (prezzo/leadership)")
    if cc is not None and cc > 55: forces_tailwind.append("Crypto conferma (prezzo/leadership)")

    # Stance (heuristics)
    if gs is None:
        stance = "n/a"
        sizing = "n/a"
        next_watch = "n/a"
    else:
        if gs >= 60 and (liq is None or liq >= 55) and (rc is None or rc >= 50):
            stance = "Aggressivo (ma disciplinato)"
            sizing = "Aumenta esposizione core (BTC/ETH) gradualmente; privilegia BTC se breadth debole."
            next_watch = "Cerca conferma: RS BTC/ETH vs Nasdaq in salita 2–3 settimane + real yield in calo."
        elif gs <= 40 or (rc is not None and rc < 40) or (ra is not None and ra < 40):
            stance = "Difensivo"
            sizing = "Riduci rischio; mantieni core solo se coerente con orizzonte lungo; evita leva."
            next_watch = "Aspetta: VIX/HY stabilizzano + USD smette di rafforzarsi + real yield scende."
        else:
            stance = "Neutrale / selettivo"
            sizing = "Mantieni core; aggiungi solo su segnali congiunti (macro + leadership)."
            next_watch = "Cerca 2–3 trigger simultanei: real yield ↓, USD ↓, spreads stabili, RS ↑."

    return stance, sizing, next_watch, forces_tailwind, forces_headwind

# ============================================================
# WALLBOARD TILE (native, robust)
# ============================================================

def indicator_guide_expander(key: str):
    meta = INDICATOR_META[key]
    exp = meta.get("expander", {})
    with st.expander(f"Guida indicatore — {meta['label']}", expanded=False):
        st.markdown(f"**Che cos’è (metrica):** {exp.get('what','')}")
        st.markdown(f"**Livelli di riferimento / soglie:** {exp.get('reference','')}")
        st.markdown("**Come leggerlo (bidirezionale):**")
        st.markdown(exp.get("interpretation", ""))
        st.markdown(f"**Link a crypto (BTC/ETH):** {exp.get('crypto_link','')}")
        st.markdown(f"**So what operativo:** {exp.get('so_what','')}")

def wallboard_tile_native(key: str, series: pd.Series, indicator_scores: dict):
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

    ref_line = meta.get("ref_line", None)
    ref_txt = "—" if ref_line is None else str(ref_line)
    ref_note = meta.get("expander", {}).get("reference", "—")

    # Card-like via container + subtle markdown (no raw div grids)
    with st.container():
        st.markdown(f"### {meta['label']}")
        st.caption(meta["source"])

        c1, c2 = st.columns([2, 1], gap="small")
        with c1:
            st.metric("Latest", latest_txt)
        with c2:
            st.markdown(pill_html(status), unsafe_allow_html=True)

        score_progress(score)
        st.caption(f"Trend ({wlab}): **{arrow} {d_txt}**")
        st.caption(f"Reference: **{ref_txt}** · {ref_note}")

        indicator_guide_expander(key)

# ============================================================
# REPORT PROMPT (crypto-focused, stable structure)
# ============================================================

REPORT_PROMPT = """SYSTEM / ROLE

You are a senior macro strategist with a crypto overlay. You write an internal PM note.
You diagnose the macro regime that drives crypto (BTC + ETH), focusing on liquidity, real cost of money, and systemic risk appetite.
No hype. No forecasts. Concrete, causal, implementation-oriented.

INPUT
You receive a JSON payload with:
- stable indicator fields (same keys every time),
- current snapshot values,
- changes vs prior snapshots (if provided),
- block scores and a global score.

CRITICAL RULES
- Use ONLY the payload.
- Do not add new indicators.
- Do not speculate beyond the data.
- Explain causality links to crypto explicitly (BTC + ETH).
- Produce clear "so what" and near-term triggers (2–6 weeks).

MANDATORY STRUCTURE (FOLLOW EXACTLY)

# Crypto Macro Regime Report
## Liquidity • Real Cost of Money • Risk Appetite — BTC + ETH view
[Insert current date]

Executive Summary
(One coherent paragraph: what is the regime, what changed, and the practical stance.)

Regime Map
1) Liquidity
2) Real Cost of Money (real yields + USD)
3) Risk Appetite & Stress (vol + credit)
4) Crypto Confirmation (BTC + ETH + relative strength)

What Changed (vs last snapshot)
- Bullet list of the 5 most important moves with direction and why they matter for crypto.

Crypto Implications (BTC + ETH)
- BTC: what the regime implies
- ETH: what the regime implies (breadth / risk appetite)

Operating Lines (Implementation)
- Exposure stance (increase / neutral / reduce)
- Risk controls / what not to do
- What you need to see next to change stance

Triggers (2–6 weeks)
(3–5 observable, threshold-based triggers linked to regime change.)

Final Bottom Line
(One paragraph, no bullets.)
""".strip()

# ============================================================
# SNAPSHOT / HISTORY UTILITIES (stable schema + comparison)
# ============================================================

SNAPSHOT_SCHEMA_VERSION = "1.0"

def utc_now_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def stable_snapshot_dict(global_score, global_status, block_scores, indicator_scores, indicators):
    """
    Build a stable snapshot with fixed fields:
    - schema_version
    - generated_at_utc
    - global_score/global_status
    - blocks: fixed keys
    - indicators: fixed keys with latest/score/status/trends
    """
    snap = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "generated_at_utc": utc_now_str(),
        "global": {
            "score": None if np.isnan(global_score) else round(float(global_score), 1),
            "status": global_status,
        },
        "blocks": {},
        "indicators": {},
    }

    # Blocks (fixed)
    for bkey in BLOCKS.keys():
        b = block_scores.get(bkey, {})
        sc = b.get("score", np.nan)
        stt = b.get("status", "n/a")
        snap["blocks"][bkey] = {
            "name": BLOCKS[bkey]["name"],
            "weight": BLOCKS[bkey]["weight"],
            "score": None if np.isnan(sc) else round(float(sc), 1),
            "status": stt,
        }

    # Indicators (fixed)
    for ikey in INDICATOR_META.keys():
        meta = INDICATOR_META[ikey]
        s_info = indicator_scores.get(ikey, {})
        score = s_info.get("score", np.nan)
        status = s_info.get("status", "n/a")
        latest = s_info.get("latest", np.nan)

        series = indicators.get(ikey, pd.Series(dtype=float))
        tr = recent_trend(series)

        d7  = pct_change_over_days(series, 7)
        d30 = pct_change_over_days(series, 30)
        d90 = pct_change_over_days(series, 90)
        d365 = pct_change_over_days(series, 365)

        snap["indicators"][ikey] = {
            "name": meta["label"],
            "source": meta["source"],
            "scoring_mode": meta.get("scoring_mode", "z5y"),
            "latest_raw": None if (latest is None or (isinstance(latest, float) and np.isnan(latest))) else float(latest),
            "latest_fmt": fmt_value(latest, meta["unit"], meta.get("scale", 1.0)),
            "score": None if np.isnan(score) else round(float(score), 1),
            "status": status,
            "trend": {
                "window": tr.get("window_label", "n/a"),
                "delta_pct": None if np.isnan(tr.get("delta_pct", np.nan)) else round(float(tr["delta_pct"]), 2),
                "arrow": tr.get("arrow", "→"),
            },
            "changes_pct": {
                "d7": None if np.isnan(d7) else round(float(d7), 2),
                "d30": None if np.isnan(d30) else round(float(d30), 2),
                "d90": None if np.isnan(d90) else round(float(d90), 2),
                "d365": None if np.isnan(d365) else round(float(d365), 2),
            },
            "reference": {
                "ref_line": meta.get("ref_line", None),
                "ref_notes": meta.get("expander", {}).get("reference", ""),
            }
        }

    return snap

def compute_snapshot_delta(curr: dict, prev: dict) -> dict:
    """
    Compute deltas with stable keys.
    Returns dict with:
      - global delta
      - blocks delta
      - indicators delta (score + latest_raw + key trend deltas)
    """
    out = {
        "prev_generated_at_utc": prev.get("generated_at_utc"),
        "curr_generated_at_utc": curr.get("generated_at_utc"),
        "global": {},
        "blocks": {},
        "indicators": {},
    }

    # global
    cg = (curr.get("global", {}) or {}).get("score", None)
    pg = (prev.get("global", {}) or {}).get("score", None)
    out["global"]["score_delta"] = None if (cg is None or pg is None) else round(float(cg - pg), 1)

    # blocks
    for bkey in BLOCKS.keys():
        cb = (curr.get("blocks", {}) or {}).get(bkey, {})
        pb = (prev.get("blocks", {}) or {}).get(bkey, {})
        cs = cb.get("score", None)
        ps = pb.get("score", None)
        out["blocks"][bkey] = {
            "score_delta": None if (cs is None or ps is None) else round(float(cs - ps), 1),
            "status_prev": pb.get("status", "n/a"),
            "status_curr": cb.get("status", "n/a"),
        }

    # indicators
    for ikey in INDICATOR_META.keys():
        ci = (curr.get("indicators", {}) or {}).get(ikey, {})
        pi = (prev.get("indicators", {}) or {}).get(ikey, {})
        cs = ci.get("score", None)
        ps = pi.get("score", None)
        cl = ci.get("latest_raw", None)
        pl = pi.get("latest_raw", None)

        out["indicators"][ikey] = {
            "score_delta": None if (cs is None or ps is None) else round(float(cs - ps), 1),
            "latest_delta": None if (cl is None or pl is None) else float(cl - pl),
            "status_prev": pi.get("status", "n/a"),
            "status_curr": ci.get("status", "n/a"),
            "latest_prev_fmt": pi.get("latest_fmt", "n/a"),
            "latest_curr_fmt": ci.get("latest_fmt", "n/a"),
        }

    return out

# ============================================================
# MAIN
# ============================================================

def main():
    st.title("Crypto Macro Regime Dashboard")
    st.markdown(
        "<div class='muted'>"
        "Dashboard per monitorare i <b>fondamentali macro</b> che guidano il sentiment e il regime su <b>crypto (BTC + ETH)</b>: "
        "1) liquidità USD, 2) costo reale del denaro, 3) risk appetite sistemico, 4) conferma crypto (prezzo + leadership)."
        "</div>",
        unsafe_allow_html=True,
    )

    # Sidebar (compact)
    with st.sidebar:
        st.header("Settings")
        if st.button("🔄 Refresh data (clear cache)"):
            st.cache_data.clear()
            st.rerun()
        years_back = st.slider("History (years)", 3, 20, 10)
        st.caption("Suggerimento: 10y è spesso un buon compromesso per z-score e robustezza.")
        st.divider()
        st.subheader("Report / Snapshot")
        st.caption("Puoi salvare snapshot e confrontarli nel tempo.")
        st.divider()

    today = datetime.now(timezone.utc).date()
    start_date = (today - DateOffset(years=years_back)).date().isoformat()

    fred_key = get_fred_api_key()
    if fred_key is None:
        st.warning("⚠️ Manca `FRED_API_KEY` in Streamlit secrets. Alcuni indicatori FRED saranno vuoti.")

    # Fetch data
    with st.spinner("Loading data (FRED + yfinance)..."):
        fred = {
            "real_10y": fetch_fred_series("DFII10", start_date),
            "fed_balance_sheet": fetch_fred_series("WALCL", start_date),
            "rrp": fetch_fred_series("RRPONTSYD", start_date),
            "hy_oas": fetch_fred_series("BAMLH0A0HYM2", start_date),
            "usd_fred": fetch_fred_series("DTWEXBGS", start_date),
        }

        yf_map = fetch_yf_many(
            ["DX-Y.NYB", "^VIX", "BTC-USD", "ETH-USD", "^IXIC"],
            start_date
        )

        indicators = {}

        # Map direct
        indicators["real_10y"] = fred["real_10y"]
        indicators["fed_balance_sheet"] = fred["fed_balance_sheet"]
        indicators["rrp"] = fred["rrp"]
        indicators["hy_oas"] = fred["hy_oas"]

        # DXY: yfinance primary, fallback FRED trade-weighted USD
        dxy = yf_map.get("DX-Y.NYB", pd.Series(dtype=float))
        if dxy is None or dxy.empty:
            dxy = fred["usd_fred"]
        indicators["dxy"] = dxy

        indicators["vix"] = yf_map.get("^VIX", pd.Series(dtype=float))
        indicators["btc"] = yf_map.get("BTC-USD", pd.Series(dtype=float))
        indicators["eth"] = yf_map.get("ETH-USD", pd.Series(dtype=float))
        nasdaq = yf_map.get("^IXIC", pd.Series(dtype=float))

        # Relative strength series (robust join)
        def safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
            if a is None or b is None or a.empty or b.empty:
                return pd.Series(dtype=float)
            j = a.to_frame("a").join(b.to_frame("b"), how="inner").dropna()
            j = j[j["b"] != 0]
            if j.empty:
                return pd.Series(dtype=float)
            return (j["a"] / j["b"]).dropna()

        indicators["btc_rel_nasdaq"] = safe_ratio(indicators["btc"], nasdaq)
        indicators["eth_rel_nasdaq"] = safe_ratio(indicators["eth"], nasdaq)

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
    now_utc = utc_now_str()

    # Tabs
    tabs = st.tabs(["Overview", "Wallboard", "Deep dive", "What changed", "Report & History"])

    # ============================================================
    # OVERVIEW (immediate crypto message)
    # ============================================================
    with tabs[0]:
        gs_txt = "n/a" if np.isnan(global_score) else f"{global_score:.1f}"
        stance, sizing, next_watch, tailwinds, headwinds = crypto_operating_lines(block_scores, indicator_scores)

        st.markdown("## Regime snapshot (crypto)")
        c1, c2 = st.columns([2, 1], gap="small")
        with c1:
            st.markdown(
                f"""
                <div class="softCard">
                  <div class="muted">Global Score (0–100)</div>
                  <div style="font-size:2.2rem; font-weight:800; line-height:1.05;">{gs_txt}</div>
                  <div style="margin-top:8px;">{pill_html(global_status)}</div>
                  <div class="muted" style="margin-top:10px;">
                    Dati aggiornati: <b>{now_utc}</b><br/>
                    Ultimo datapoint: <b>{('n/a' if data_max_date is None else str(pd.to_datetime(data_max_date).date()))}</b><br/>
                    History: <b>{years_back}y</b>
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                f"""
                <div class="softCard">
                  <div class="muted">Stance</div>
                  <div style="font-size:1.25rem; font-weight:800; margin-top:6px;">{stance}</div>
                  <div class="muted" style="margin-top:10px;">Cosa fare:</div>
                  <div style="font-size:0.98rem; margin-top:6px;">{sizing}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("### Forze sul comparto crypto (BTC + ETH)")
        left, right = st.columns(2, gap="small")
        with left:
            st.markdown("**Tailwinds (spingono)**")
            if tailwinds:
                for x in tailwinds:
                    st.write(f"- {x}")
            else:
                st.write("- (nessun tailwind forte rilevato)")

        with right:
            st.markdown("**Headwinds (schiacciano)**")
            if headwinds:
                for x in headwinds:
                    st.write(f"- {x}")
            else:
                st.write("- (nessun headwind forte rilevato)")

        st.markdown("### 1) Liquidità • 2) Costo reale del denaro • 3) Risk appetite • 4) Conferma crypto")
        with st.expander("Come leggere Risk-on / Neutral / Risk-off (per crypto, non per equity)", expanded=True):
            st.markdown(
                """
**Risk-on (crypto):** la combinazione di liquidità e costo reale del denaro smette di essere un vincolo, e i premi per il rischio (vol/credit) non stanno esplodendo. BTC/ETH mostrano conferma (trend / leadership).  
**Neutral:** segnali misti; non è “short”, ma serve disciplina (size, niente leva).  
**Risk-off:** real yield e/o USD e/o stress (VIX/HY) indicano vincolo; spesso coincide con deleveraging e fragilità.

**Nota importante:** questa dashboard non “trova il bottom”. Serve a capire se il vento macro sta diventando favorevole o sfavorevole, e se BTC/ETH lo stanno confermando.
                """.strip()
            )

        st.markdown("### Cosa guardare adesso (next 2–6 weeks)")
        st.write(f"- {next_watch}")

    # ============================================================
    # WALLBOARD (native layout, no HTML grids → fixes your <div> leaks)
    # Menu “a tendina” SOPRA alle tile: qui lo facciamo come expander-guida di sezione
    # ============================================================
    with tabs[1]:
        st.markdown("## Wallboard (crypto)")
        st.markdown("<div class='muted'>Prima leggi le sezioni. Poi apri le guide se ti serve dettaglio.</div>", unsafe_allow_html=True)

        # Section guide ABOVE
        with st.expander("Guida rapida: come ogni sezione impatta BTC/ETH", expanded=False):
            st.markdown("**1) Liquidità:** quando migliora, aumenta la probabilità di regime favorevole a crypto.")
            st.markdown("**2) Costo reale del denaro:** real yields e USD forte sono i principali freni.")
            st.markdown("**3) Risk appetite & stress:** VIX e credito dicono se il sistema sta de-riskando.")
            st.markdown("**4) Crypto confirmation:** prezzo + leadership (RS) dicono se la domanda è reale e si allarga (BTC → ETH).")

        def render_group(block_key: str):
            binfo = BLOCKS[block_key]
            st.markdown(f"### {binfo['name']}")
            st.caption(binfo["desc"])

            # responsive: 2 columns on desktop, stacks on mobile
            cols = st.columns(2, gap="small")
            ci = 0
            for ikey in binfo["indicators"]:
                s = indicators.get(ikey, pd.Series(dtype=float))
                with cols[ci]:
                    if s is None or s.empty:
                        st.warning(f"{INDICATOR_META[ikey]['label']}: dati non disponibili.")
                        indicator_guide_expander(ikey)
                    else:
                        wallboard_tile_native(ikey, s, indicator_scores)
                ci = (ci + 1) % 2

            st.divider()

        for bk in ["liquidity", "real_cost", "risk_appetite", "crypto_confirm"]:
            render_group(bk)

    # ============================================================
    # DEEP DIVE (scrollable, all sections expandable and stay open)
    # ============================================================
    with tabs[2]:
        st.markdown("## Deep dive (scrollabile)")
        st.markdown("<div class='muted'>Tutte le sezioni sono espandibili e possono restare aperte. Ogni indicatore ha grafico + guida.</div>", unsafe_allow_html=True)

        for bk, binfo in BLOCKS.items():
            with st.expander(f"{binfo['name']} — charts", expanded=(bk in ["liquidity", "real_cost"])):
                st.caption(binfo["desc"])
                for ikey in binfo["indicators"]:
                    meta = INDICATOR_META[ikey]
                    s = indicators.get(ikey, pd.Series(dtype=float))
                    sc = indicator_scores.get(ikey, {})
                    score = sc.get("score", np.nan)
                    status = sc.get("status", "n/a")
                    latest = sc.get("latest", np.nan)
                    latest_txt = fmt_value(latest, meta["unit"], meta.get("scale", 1.0))

                    st.markdown(f"### {meta['label']}")
                    p1, p2, p3 = st.columns([1.4, 1, 1], gap="small")
                    with p1:
                        st.metric("Latest", latest_txt)
                    with p2:
                        st.markdown(pill_html(status), unsafe_allow_html=True)
                    with p3:
                        score_progress(score)

                    if s is None or s.empty:
                        st.warning("Dati non disponibili per questo indicatore.")
                    else:
                        fig = plot_premium(s, meta["label"], ref_line=meta.get("ref_line", None), height=320)
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=f"deep_{ikey}")

                    indicator_guide_expander(ikey)
                    st.divider()

    # ============================================================
    # WHAT CHANGED (adds 1M explicitly + watchlist)
    # ============================================================
    with tabs[3]:
        st.markdown("## What changed")
        st.markdown("<div class='muted'>Individua cosa si muove e cosa è vicino a soglie di regime (utile per 2–6 settimane).</div>", unsafe_allow_html=True)

        rows = []
        for key, meta in INDICATOR_META.items():
            s = indicators.get(key, pd.Series(dtype=float))
            if s is None or s.empty:
                continue

            d7 = pct_change_over_days(s, 7)
            d30 = pct_change_over_days(s, 30)   # 1M
            d90 = pct_change_over_days(s, 90)
            d1y = pct_change_over_days(s, 365)

            sc = indicator_scores.get(key, {})
            score = sc.get("score", np.nan)
            status = sc.get("status", "n/a")

            # Attention: threshold proximity + recent move magnitude
            if np.isnan(score):
                prox = 0.0
            else:
                prox = max(0.0, 20.0 - min(abs(score - 40), abs(score - 60))) / 20.0  # 0..1

            # use 1M magnitude as primary
            move = 0.0 if np.isnan(d30) else min(1.0, abs(d30) / 10.0)
            attention = 0.55 * prox + 0.45 * move
            watch = "WATCH" if attention >= 0.55 else ""

            rows.append({
                "Indicator": meta["label"],
                "Regime": status_label(status),
                "Score": (np.nan if np.isnan(score) else round(score, 1)),
                "Δ 7d %": (np.nan if np.isnan(d7) else round(d7, 2)),
                "Δ 1M %": (np.nan if np.isnan(d30) else round(d30, 2)),
                "Δ 3M %": (np.nan if np.isnan(d90) else round(d90, 2)),
                "Δ 1Y %": (np.nan if np.isnan(d1y) else round(d1y, 2)),
                "Watchlist": watch,
                "Attention": round(attention, 2),
            })

        if not rows:
            st.info("Dati insufficienti per calcolare i cambiamenti.")
        else:
            df = pd.DataFrame(rows).sort_values(["Watchlist", "Attention"], ascending=[True, False])

            wl = df[df["Watchlist"] == "WATCH"].sort_values("Attention", ascending=False).head(8)
            if not wl.empty:
                st.markdown("### Watchlist (movers / threshold proximity)")
                for _, r in wl.iterrows():
                    st.markdown(
                        f"""
                        <div class="softCard" style="margin-bottom:10px;">
                          <div class="muted">{r['Indicator']}</div>
                          <div style="margin-top:6px;">
                            Regime: <b>{r['Regime']}</b> · Score: <b>{r['Score']}</b> · Δ 1M: <b>{(r['Δ 1M %'] if pd.notna(r['Δ 1M %']) else 'n/a')}</b>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            st.markdown("### Full table")
            st.dataframe(
                df.reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Indicator": st.column_config.TextColumn("Indicator", width="large"),
                    "Regime": st.column_config.TextColumn("Regime"),
                    "Score": st.column_config.NumberColumn("Score"),
                    "Attention": st.column_config.NumberColumn("Attention"),
                }
            )
            st.caption("Nota: le % change usano l’osservazione più vicina disponibile (frequenze diverse).")

    # ============================================================
    # REPORT & HISTORY
    # - Generate stable snapshot JSON
    # - Keep in-session history (list)
    # - Download snapshots
    # - Paste previous snapshot JSON to compare
    # ============================================================
    with tabs[4]:
        st.markdown("## Report & History (comparabile nel tempo)")
        st.markdown("<div class='muted'>Genera uno snapshot (schema stabile), salvalo, e confrontalo con snapshot precedenti.</div>", unsafe_allow_html=True)

        # session history
        if "snapshots" not in st.session_state:
            st.session_state["snapshots"] = []  # list of snapshot dicts

        # Current snapshot
        curr_snap = stable_snapshot_dict(global_score, global_status, block_scores, indicator_scores, indicators)

        cA, cB = st.columns([1, 1], gap="small")
        with cA:
            if st.button("📌 Save snapshot to history"):
                st.session_state["snapshots"].append(curr_snap)
                st.success(f"Snapshot salvato. Totale: {len(st.session_state['snapshots'])}")
        with cB:
            st.download_button(
                "⬇️ Download current snapshot (JSON)",
                data=json.dumps(curr_snap, indent=2),
                file_name=f"crypto_macro_snapshot_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

        st.divider()

        # Compare vs previous snapshot (from history)
        st.markdown("### Compare snapshots")
        if len(st.session_state["snapshots"]) >= 2:
            idx_prev = st.selectbox(
                "Select previous snapshot from history",
                options=list(range(len(st.session_state["snapshots"]))),
                index=max(0, len(st.session_state["snapshots"]) - 2),
                format_func=lambda i: st.session_state["snapshots"][i].get("generated_at_utc", f"snapshot {i}")
            )
            prev_snap = st.session_state["snapshots"][idx_prev]
            delta = compute_snapshot_delta(curr_snap, prev_snap)

            st.markdown("**Global delta:**")
            st.write(f"- Prev: {prev_snap.get('global',{}).get('score')} → Curr: {curr_snap.get('global',{}).get('score')}  (Δ {delta['global'].get('score_delta')})")

            # Top indicator score deltas
            id_rows = []
            for k, v in delta["indicators"].items():
                sd = v.get("score_delta", None)
                if sd is None:
                    continue
                id_rows.append({
                    "Indicator": INDICATOR_META[k]["label"],
                    "Score Δ": sd,
                    "Status prev": v.get("status_prev"),
                    "Status curr": v.get("status_curr"),
                    "Latest prev": v.get("latest_prev_fmt"),
                    "Latest curr": v.get("latest_curr_fmt"),
                })
            if id_rows:
                df_delta = pd.DataFrame(id_rows).sort_values("Score Δ", ascending=False)
                st.markdown("**Top movers (Score Δ):**")
                st.dataframe(df_delta.head(10), use_container_width=True, hide_index=True)
                st.markdown("**Bottom movers (Score Δ):**")
                st.dataframe(df_delta.tail(10), use_container_width=True, hide_index=True)
        else:
            st.info("Salva almeno 2 snapshot per poter confrontare nel tempo.")

        st.divider()

        # External compare: paste previous snapshot JSON
        st.markdown("### Compare with an external snapshot (paste JSON)")
        prev_text = st.text_area("Paste a previous snapshot JSON here (optional)", height=180)
        if st.button("Compare pasted snapshot → current"):
            try:
                prev_obj = json.loads(prev_text)
                delta2 = compute_snapshot_delta(curr_snap, prev_obj)

                st.success("Comparison computed.")
                st.write(f"- Prev time: {delta2.get('prev_generated_at_utc')}")
                st.write(f"- Curr time: {delta2.get('curr_generated_at_utc')}")
                st.write(f"- Global score Δ: {delta2.get('global',{}).get('score_delta')}")

                # Show a compact table of key blocks
                b_rows = []
                for bk in BLOCKS.keys():
                    b_rows.append({
                        "Block": BLOCKS[bk]["name"],
                        "Score Δ": delta2["blocks"][bk].get("score_delta"),
                        "Status prev": delta2["blocks"][bk].get("status_prev"),
                        "Status curr": delta2["blocks"][bk].get("status_curr"),
                    })
                st.dataframe(pd.DataFrame(b_rows), use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Invalid JSON or schema mismatch: {e}")

        st.divider()

        # One-shot copy/paste for AI report (prompt + payload)
        st.markdown("### AI report generation (copy/paste)")
        st.caption("Copia e incolla in ChatGPT (nuova chat): prima il prompt, poi il payload JSON.")

        one_shot = (
            "### COPY/PASTE BELOW (PROMPT + PAYLOAD)\n\n"
            + REPORT_PROMPT
            + "\n\n---\n\n"
            + "JSON PAYLOAD:\n\n```json\n"
            + json.dumps(curr_snap, indent=2)
            + "\n```\n"
        )
        st.code(one_shot, language="markdown")

if __name__ == "__main__":
    main()
