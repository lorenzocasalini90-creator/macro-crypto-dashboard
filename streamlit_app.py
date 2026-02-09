import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from pandas.tseries.offsets import DateOffset

# ============================================================
# PAGE CONFIG (mobile-first)
# ============================================================
st.set_page_config(
    page_title="Crypto Macro Radar",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# PREMIUM SOBER CSS (mobile-first)
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

    --accent:rgba(148,163,184,1);         /* sober slate */
    --accentSoft:rgba(148,163,184,0.16);
    --accent2:rgba(99,102,241,1);         /* subtle indigo */
    --accent2Soft:rgba(99,102,241,0.14);
  }

  .stApp {
    background: radial-gradient(1100px 650px at 20% 0%, #121a33 0%, #0b0f19 45%, #0b0f19 100%);
    color: var(--text);
  }

  .block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1180px;
  }

  h1, h2, h3, h4 { color: var(--text); letter-spacing: -0.02em; }
  .muted { color: var(--muted); }

  /* Tabs */
  button[data-baseweb="tab"]{
    color: rgba(255,255,255,0.90) !important;
    font-weight: 700 !important;
    background: rgba(255,255,255,0.03) !important;
    border-radius: 10px !important;
    margin-right: 6px !important;
    padding: 10px 12px !important;
  }
  button[data-baseweb="tab"][aria-selected="true"]{
    color: rgba(255,255,255,0.98) !important;
    background: var(--accentSoft) !important;
    border: 1px solid rgba(148,163,184,0.45) !important;
  }

  /* Expander */
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

  /* Cards */
  .card{
    background: linear-gradient(180deg, rgba(255,255,255,0.055) 0%, rgba(255,255,255,0.03) 100%);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 16px 16px 14px 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
  }
  .cardTitle{ font-size: 0.92rem; color: var(--muted); margin-bottom: 6px; }
  .cardValue{ font-size: 2.05rem; font-weight: 820; line-height: 1.05; color: var(--text); }
  .cardSub{ margin-top: 8px; font-size: 0.96rem; color: var(--muted); line-height: 1.25rem; }

  .hr{
    height: 1px;
    background: rgba(255,255,255,0.10);
    margin: 14px 0 14px 0;
  }

  /* Responsive grids */
  .grid2{ display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
  .grid3{ display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
  .grid4{ display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }

  @media (max-width: 980px){
    .grid4{ grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .grid3{ grid-template-columns: repeat(2, minmax(0, 1fr)); }
  }
  @media (max-width: 720px){
    .grid4{ grid-template-columns: repeat(1, minmax(0, 1fr)); }
    .grid3{ grid-template-columns: repeat(1, minmax(0, 1fr)); }
    .grid2{ grid-template-columns: repeat(1, minmax(0, 1fr)); }
  }

  /* Pills */
  .pill{
    display:inline-flex;
    align-items:center;
    gap:8px;
    padding: 6px 12px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: rgba(255,255,255,0.04);
    font-size: 0.88rem;
    color: var(--text);
    white-space: nowrap;
  }
  .dot{ width: 10px; height: 10px; border-radius: 999px; display:inline-block; }
  .pill.good{ border-color: rgba(34,197,94,0.40); background: rgba(34,197,94,0.12); }
  .pill.warn{ border-color: rgba(245,158,11,0.40); background: rgba(245,158,11,0.12); }
  .pill.bad { border-color: rgba(239,68,68,0.40); background: rgba(239,68,68,0.12); }
  .pill.info{ border-color: rgba(99,102,241,0.35); background: rgba(99,102,241,0.12); }

  /* Section */
  .section{
    background: rgba(255,255,255,0.025);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 14px;
    box-shadow: 0 10px 28px rgba(0,0,0,0.20);
    margin-bottom: 14px;
  }
  .sectionHead{ display:flex; align-items:flex-start; justify-content:space-between; gap: 12px; margin-bottom: 10px; flex-wrap:wrap; }
  .sectionTitle{ font-size: 1.18rem; font-weight: 860; color: rgba(255,255,255,0.96); }
  .sectionDesc{ font-size: 0.95rem; color: var(--muted); margin-top: 3px; max-width: 920px; }

  /* Wallboard tiles */
  .wbGrid{
    display:grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
  }
  @media (max-width: 1200px){
    .wbGrid{ grid-template-columns: repeat(2, minmax(0, 1fr)); }
  }
  @media (max-width: 720px){
    .wbGrid{ grid-template-columns: repeat(1, minmax(0, 1fr)); }
  }

  .wbTile{
    background: rgba(255,255,255,0.028);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 14px 14px 12px 14px;
    box-shadow: 0 10px 26px rgba(0,0,0,0.18);
    min-height: 156px;
    display:flex;
    flex-direction:column;
    justify-content:space-between;
  }
  .wbName{ font-size: 0.98rem; font-weight: 860; color: rgba(255,255,255,0.96); margin-bottom: 2px; }
  .wbMeta{ font-size: 0.86rem; color: var(--muted); margin-bottom: 8px; }
  .wbRow{ display:flex; align-items:baseline; justify-content:space-between; gap: 10px; flex-wrap:wrap; }
  .wbVal{ font-size: 1.65rem; font-weight: 900; letter-spacing:-0.01em; }
  .wbSmall{ font-size: 0.88rem; color: var(--muted); }
  .wbFoot{ display:flex; align-items:center; justify-content:space-between; gap: 10px; margin-top: 10px; flex-wrap:wrap; }

  /* Score bar */
  .barWrap{
    height: 10px;
    border-radius: 999px;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.08);
    position: relative;
    overflow:hidden;
  }
  .barFill{
    height: 100%;
    border-radius: 999px;
    background: rgba(255,255,255,0.14);
    width: 100%;
    opacity: 0.55;
  }
  .barMark{
    position:absolute;
    top:-4px;
    width: 3px;
    height: 18px;
    border-radius: 2px;
    background: rgba(255,255,255,0.92);
    box-shadow: 0 0 0 2px rgba(0,0,0,0.20);
  }

  .stDataFrame { border: 1px solid var(--border); border-radius: 12px; overflow:hidden; }
  code { color: rgba(255,255,255,0.88); }

  .kpiRow{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# DATA: FETCHERS
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
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json().get("observations", [])
        if not data:
            return pd.Series(dtype=float)
        idx = pd.to_datetime([o["date"] for o in data])
        vals = []
        for o in data:
            v = o.get("value", np.nan)
            try:
                vals.append(float(v))
            except Exception:
                vals.append(np.nan)
        s = pd.Series(vals, index=idx).astype(float).sort_index()
        return s.dropna()
    except Exception:
        return pd.Series(dtype=float)

@st.cache_data(ttl=3600)
def fetch_yf_one(ticker: str, start_date: str) -> pd.Series:
    try:
        df = yf.Ticker(ticker).history(start=start_date, auto_adjust=True)
        if df is None or df.empty:
            return pd.Series(dtype=float)
        s = df["Close"].dropna()
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
# SCORING + HELPERS
# ============================================================

def rolling_percentile_last(hist: pd.Series, latest: float) -> float:
    h = hist.dropna()
    if len(h) < 10 or pd.isna(latest):
        return np.nan
    return float((h <= latest).mean())

def compute_indicator_score(series: pd.Series, direction: int, scoring_mode: str = "z5y"):
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
        p = rolling_percentile_last(hist, latest)
        sig = (p - 0.5) * 4.0
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

def classify_status(score: float) -> str:
    if np.isnan(score):
        return "n/a"
    if score > 60:
        return "risk_on"
    if score < 40:
        return "risk_off"
    return "neutral"

def status_label(status: str) -> str:
    return {"risk_on": "Risk-on", "risk_off": "Risk-off", "neutral": "Neutral"}.get(status, "n/a")

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
    if series is None or series.dropna().shape[0] < 10:
        return {"window_label": "n/a", "delta_pct": np.nan, "arrow": "→"}
    freq = infer_frequency_days(series)
    if freq >= 20:
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

def score_bar_html(score: float) -> str:
    if np.isnan(score):
        pos = 50
    else:
        pos = int(np.clip(score, 0, 100))
    return f"""
      <div class="barWrap">
        <div class="barFill"></div>
        <div class="barMark" style="left: calc({pos}% - 2px);"></div>
      </div>
    """

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
# CRYPTO-FOCUSED META (drivers of crypto sentiment)
# ============================================================

INDICATOR_META = {
    # 1) USD LIQUIDITY / PLUMBING
    "fed_balance_sheet": {
        "label": "Fed Balance Sheet (WALCL)",
        "unit": "bn USD",
        "direction": +1,
        "source": "FRED WALCL (millions → bn)",
        "scale": 1.0 / 1000.0,
        "ref_line": None,
        "scoring_mode": "z5y",
        "expander": {
            "what": "Totale asset FED: proxy QE/QT e liquidità di sistema (driver macro per asset risk, incl. crypto).",
            "reference": "BS ↑ o QT che rallenta = tailwind; BS ↓ persistente = headwind (euristica).",
            "interpretation": "- Fed BS ↑: più liquidità marginale → migliora sentiment su BTC/alt.\n- Fed BS ↓: drenaggio → crypto tende a fare fatica (soprattutto beta).",
            "bridge": "Crypto è una ‘opzione’ sulla liquidità futura: la plumbing cambia il price action prima delle narrative.",
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
            "what": "Reverse Repo: liquidità parcheggiata nel facility (cash non in risk).",
            "reference": "RRP ↓ rapido può liberare marginal liquidity (euristica).",
            "interpretation": "- RRP ↑: meno liquidità ‘disponibile’ → crypto risk appetite tende a peggiorare.\n- RRP ↓: potenziale tailwind tattico su BTC/alt.",
            "bridge": "Spesso è un ‘release valve’: quando scende, i mercati respirano.",
        },
    },

    # 2) REAL COST OF MONEY (kryptonite)
    "real_10y": {
        "label": "US 10Y TIPS Real Yield",
        "unit": "%",
        "direction": -1,
        "source": "FRED DFII10",
        "scale": 1.0,
        "ref_line": 2.0,
        "scoring_mode": "z5y",
        "expander": {
            "what": "Real yield: rendimento reale risk-free. È spesso la ‘kryptonite’ per crypto.",
            "reference": "0–2% neutro; >2% restrittivo; cali rapidi = risk-on crypto (euristiche).",
            "interpretation": "- Real yield ↑: competizione forte del risk-free → crypto soffre.\n- Real yield ↓: allenta il vincolo → BTC/alt possono riprendere (se stress non esplode).",
            "bridge": "Quando il risk-free reale rende molto, la domanda di rischio diminuisce.",
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
        "expander": {
            "what": "Yield nominale 10Y: proxy di tightening/discount rate (impatta soprattutto risk assets).",
            "reference": "Conta il delta: rialzi rapidi spesso equivalgono a tightening (euristica).",
            "interpretation": "- Yield ↑ veloce: pressione su risk assets → crypto tende a indebolirsi.\n- Yield ↓: supporta appetite (ma attenzione se è ‘growth scare’).",
            "bridge": "Il repricing dei tassi spesso guida i drawdown crypto più delle narrative.",
        },
    },

    # 3) USD FILTER (mandatory)
    "usd_index": {
        "label": "USD Strength (DXY / Broad)",
        "unit": "",
        "direction": -1,
        "source": "yfinance DX-Y.NYB (fallback FRED DTWEXBGS)",
        "scale": 1.0,
        "ref_line": None,
        "scoring_mode": "z5y",
        "expander": {
            "what": "Forza USD: filtro quasi obbligatorio per crypto (condizioni globali).",
            "reference": "USD ↑ = tightening globale → spesso headwind per crypto; USD ↓ = respiro (euristica).",
            "interpretation": "- USD ↑: funding stress globale → BTC/alt tendono a faticare.\n- USD ↓: condizioni più easy → supporto a crypto.",
            "bridge": "Molte posizioni/leverage e flussi globali sono USD-sensitive.",
        },
    },

    # 4) RISK APPETITE / STRESS (confirmation)
    "vix": {
        "label": "VIX",
        "unit": "",
        "direction": -1,
        "source": "yfinance ^VIX",
        "scale": 1.0,
        "ref_line": 25.0,
        "scoring_mode": "z5y",
        "expander": {
            "what": "Vol implicita equity: proxy stress sistemico. Crypto raramente riparte con VIX alto.",
            "reference": "<15 benigno; 15–25 normale; >25 stress (euristica).",
            "interpretation": "- VIX ↑: risk-off sistemico → crypto beta viene venduto.\n- VIX ↓: risk-taking più facile → crypto respira.",
            "bridge": "Quando la vol sale, i risk budgets si restringono (anche su crypto).",
        },
    },
    "hy_oas": {
        "label": "US High Yield OAS",
        "unit": "pp",
        "direction": -1,
        "source": "FRED BAMLH0A0HYM2",
        "scale": 1.0,
        "ref_line": 6.0,
        "scoring_mode": "z5y",
        "expander": {
            "what": "Spread HY: stress credito (spesso anticipa o conferma risk-off).",
            "reference": "<4% benigno; 4–6% warning; >6–7% stress (euristica).",
            "interpretation": "- Spread ↑: funding/credit stress → crypto tende a soffrire.\n- Spread ↓: appetite e carry → supportive per risk assets.",
            "bridge": "Credito che si rompe = deleveraging, e crypto di solito paga.",
        },
    },

    # 5) EQUITY RELATIVE CONFIRMATION
    "btc_price": {
        "label": "BTC Price (BTC-USD)",
        "unit": "",
        "direction": +1,
        "source": "yfinance BTC-USD",
        "scale": 1.0,
        "ref_line": None,
        "scoring_mode": "z5y",
        "expander": {
            "what": "Prezzo BTC: utile come conferma, ma va letto insieme ai driver macro sopra.",
            "reference": "Più importante il comportamento (trend/RS) che ‘il livello’.",
            "interpretation": "- BTC ↑ con macro che migliora: bull ‘pulito’.\n- BTC ↑ con macro che peggiora: spesso fragile / squeeze-driven.",
            "bridge": "Il punto è evitare FOMO: macro+stress ti dicono se la salita è sostenibile.",
        },
    },
    "nasdaq": {
        "label": "Nasdaq (QQQ)",
        "unit": "",
        "direction": +1,
        "source": "yfinance QQQ",
        "scale": 1.0,
        "ref_line": None,
        "scoring_mode": "z5y",
        "expander": {
            "what": "Equity risk proxy (tech) per confronto con BTC.",
            "reference": "BTC che sovraperforma QQQ è un segnale forte (euristica).",
            "interpretation": "- QQQ forte: risk-on generale.\n- QQQ debole con BTC forte: segnale di idiosincrasia crypto (molto interessante).",
            "bridge": "Serve per capire se crypto sta ‘guidando’ o solo seguendo l’equity.",
        },
    },
    "btc_vs_qqq": {
        "label": "BTC / Nasdaq Relative Strength (BTC / QQQ)",
        "unit": "ratio",
        "direction": +1,
        "source": "Derived (BTC-USD / QQQ)",
        "scale": 1.0,
        "ref_line": None,
        "scoring_mode": "z5y",
        "expander": {
            "what": "Forza relativa BTC vs Nasdaq: conferma che crypto sta outperformando risk tradizionale.",
            "reference": "RS ↑ per 2–3 settimane = segnale forte (euristica).",
            "interpretation": "- RS ↑: domanda crypto ‘reale’ / leadership.\n- RS ↓: crypto segue o underperforma (meno convincente).",
            "bridge": "Per timing: RS è una delle conferme migliori dopo macro/liquidità.",
        },
    },
}

BLOCKS = {
    "liquidity": {
        "name": "1) USD Liquidity / Plumbing",
        "weight": 0.30,
        "indicators": ["fed_balance_sheet", "rrp"],
        "desc": "Il motore: liquidità USD e plumbing FED (tailwind/headwind per crypto).",
        "group": "Macro Drivers",
    },
    "real_cost": {
        "name": "2) Real Cost of Money",
        "weight": 0.30,
        "indicators": ["real_10y", "nominal_10y"],
        "desc": "Il vincolo: real yields e tassi (kryptonite o supporto).",
        "group": "Macro Drivers",
    },
    "filters": {
        "name": "3) Stress & Filters",
        "weight": 0.25,
        "indicators": ["usd_index", "vix", "hy_oas"],
        "desc": "Filtri obbligatori: USD, vol e credito per evitare falsi risk-on.",
        "group": "Macro Drivers",
    },
    "crypto_confirm": {
        "name": "4) Crypto Confirmation",
        "weight": 0.15,
        "indicators": ["btc_price", "btc_vs_qqq"],
        "desc": "Conferme: BTC e forza relativa vs Nasdaq (leadership).",
        "group": "Crypto Confirmation",
    },
}

# ============================================================
# CRYPTO OPERATING LINES (clear, not trading advice)
# ============================================================

def crypto_operating_lines(block_scores: dict, indicator_scores: dict):
    gs = block_scores.get("GLOBAL", {}).get("score", np.nan)

    def _sg(x):
        if np.isnan(x):
            return np.nan
        return float(x)

    liq = _sg(block_scores.get("liquidity", {}).get("score", np.nan))
    cost = _sg(block_scores.get("real_cost", {}).get("score", np.nan))
    filt = _sg(block_scores.get("filters", {}).get("score", np.nan))
    conf = _sg(block_scores.get("crypto_confirm", {}).get("score", np.nan))

    def band(x):
        if np.isnan(x): return "n/a"
        if x >= 60: return "supportive"
        if x <= 40: return "adverse"
        return "mixed"

    # Stance: BTC / Alt beta / Leverage / Risk mgmt
    if np.isnan(gs):
        stance = "n/a"
        why = "Dati insufficienti."
    else:
        if (liq >= 60) and (cost >= 55) and (filt >= 55):
            stance = "Risk-on crypto (measured)"
            why = "Liquidità favorevole + costo reale meno restrittivo + filtri (USD/vol/credito) non stressati."
        elif (cost <= 40) or (filt <= 40):
            stance = "De-risk / defensive"
            why = "Real yields e/o stress filters avversi: crypto tende a soffrire (soprattutto beta e leverage)."
        else:
            stance = "Neutral / selective"
            why = "Segnali misti: evitare FOMO, privilegiare qualità/liquidità e conferme."

    # Allocation-style hints
    if not np.isnan(conf) and conf >= 60 and not np.isnan(liq) and liq >= 55:
        btc_vs_alt = "BTC-leading (better quality risk)"
        btc_vs_alt_why = "BTC strength + relative strength vs Nasdaq: domanda più ‘solida’."
    elif not np.isnan(filt) and filt <= 40:
        btc_vs_alt = "Prefer BTC / reduce alt beta"
        btc_vs_alt_why = "Quando stress sale, alt beta underperforma."
    else:
        btc_vs_alt = "Balanced"
        btc_vs_alt_why = "Aspettare conferme (RS, stress che rientra)."

    # Leverage heuristic
    vix_sc = indicator_scores.get("vix", {}).get("score", np.nan)
    hy_sc = indicator_scores.get("hy_oas", {}).get("score", np.nan)
    if (not np.isnan(vix_sc) and vix_sc <= 40) or (not np.isnan(hy_sc) and hy_sc <= 40):
        lev = "Keep leverage low"
        lev_why = "Stress premia alti: deleveraging risk."
    else:
        lev = "Moderate (if disciplined)"
        lev_why = "Nessun segnale dominante di stress; comunque sizing prudente."

    # Context
    context = f"GLOBAL={band(gs)}, LIQ={band(liq)}, REAL_COST={band(cost)}, FILTERS={band(filt)}, CONFIRM={band(conf)}"

    return {
        "Crypto stance": {"stance": stance, "why": why, "context": context},
        "BTC vs Alt beta": {"stance": btc_vs_alt, "why": btc_vs_alt_why, "context": f"CONFIRM={band(conf)}, FILTERS={band(filt)}"},
        "Leverage": {"stance": lev, "why": lev_why, "context": f"VIX_score={band(vix_sc)}, HY_score={band(hy_sc)}"},
    }

# ============================================================
# ALERTS / TRIGGERS
# ============================================================

def crossed_up(series: pd.Series, thresh: float) -> pd.Series:
    s = series.dropna()
    if s.empty:
        return pd.Series(dtype=bool)
    x = s > thresh
    return x & (~x.shift(1).fillna(False))

def crossed_down(series: pd.Series, thresh: float) -> pd.Series:
    s = series.dropna()
    if s.empty:
        return pd.Series(dtype=bool)
    x = s < thresh
    return x & (~x.shift(1).fillna(False))

def turned_true_dates(cond: pd.Series) -> list[pd.Timestamp]:
    c = cond.dropna()
    if c.empty:
        return []
    turned = c & (~c.shift(1).fillna(False))
    idx = c.index[turned.values]
    return list(pd.to_datetime(idx))

def last_n_triggers(cond: pd.Series, since: pd.Timestamp) -> list[str]:
    dates = [d for d in turned_true_dates(cond) if d >= since]
    return [pd.to_datetime(d).date().isoformat() for d in dates[-6:]]

def is_active(cond: pd.Series) -> bool:
    c = cond.dropna()
    if c.empty:
        return False
    return bool(c.iloc[-1])

def crypto_so_what_line(key: str, dwin: float) -> str:
    meta = INDICATOR_META[key]
    direction = meta["direction"]
    if np.isnan(dwin):
        return "Movimento recente non quantificabile (serie/frequenza)."

    # if direction = -1, up is worse for crypto; if +1, up is better
    if direction == -1:
        if dwin > 0:
            return "↑ tende a peggiorare le condizioni per crypto (headwind)."
        if dwin < 0:
            return "↓ tende a migliorare le condizioni per crypto (tailwind)."
        return "≈ nessun impulso evidente."
    else:
        if dwin > 0:
            return "↑ conferma demand/risk-on crypto (supportivo)."
        if dwin < 0:
            return "↓ segnala perdita di momentum/leadership (cautela)."
        return "≈ nessun impulso evidente."

# ============================================================
# WALLBOARD TILE
# ============================================================

def wallboard_tile(key: str, series: pd.Series, indicator_scores: dict):
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
    ref_note = meta["expander"].get("reference", "—")

    st.markdown(
        f"""
        <div class="wbTile">
          <div>
            <div class="wbName">{meta["label"]}</div>
            <div class="wbMeta">{meta["source"]}</div>

            <div class="wbRow">
              <div class="wbVal">{latest_txt}</div>
              <div>{pill_html(status)}</div>
            </div>

            <div style="margin-top:10px;">
              {score_bar_html(score)}
              <div class="wbFoot">
                <div class="wbSmall">Score: <b>{("n/a" if np.isnan(score) else f"{score:.0f}")}</b></div>
                <div class="wbSmall">Trend ({wlab}): <b>{arrow} {d_txt}</b></div>
              </div>
            </div>

            <div class="wbSmall" style="margin-top:8px;">
              Reference: <b>{ref_txt}</b> · {ref_note}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander(f"Indicator guide — {meta['label']}", expanded=False):
        exp = meta["expander"]
        st.markdown(f"**What it is:** {exp.get('what','')}")
        st.markdown(f"**Reference levels / thresholds:** {exp.get('reference','')}")
        st.markdown("**How to read it (crypto lens):**")
        st.markdown(exp.get("interpretation", ""))
        st.markdown(f"**Why it matters for crypto:** {exp.get('bridge','')}")

# ============================================================
# REPORT PROMPT (crypto-focused)
# ============================================================

REPORT_PROMPT = """SYSTEM / ROLE
You are a macro-driven crypto strategist writing an internal regime note.
You diagnose the crypto regime using macro liquidity, real rates, USD, and system risk appetite.
No narratives, no price predictions. Only what the indicators say.

You receive a YAML payload with:
- global score + block scores
- indicator latest values, scores, and recent changes
- active alerts + recent triggers
- crypto operating lines

RULES
- Use ONLY data in payload (no speculation, no new indicators).
- Separate: Macro Drivers vs Crypto Confirmations.
- Explain *why* the regime is Risk-on/Neutral/Risk-off for crypto.
- Provide clear, professional “so what” (BTC vs alts, leverage discipline, risk budget).

OUTPUT STRUCTURE (follow exactly)
# Crypto Macro Regime Note
[Insert date]

## Executive summary (max 8 lines)
## What changed (1M focus, then 30d/7d)
## Macro Drivers (the engine)
1) USD Liquidity / Plumbing
2) Real Cost of Money
3) Stress & Filters (USD, vol, credit)
## Crypto Confirmation
4) BTC & Relative Strength vs Nasdaq
## Active alerts & triggers (2–6 weeks)
## Operating lines (implementation)
- Crypto stance
- BTC vs Alt beta
- Leverage
## Bottom line (one paragraph)
""".strip()

# ============================================================
# MAIN
# ============================================================

def main():
    st.title("Crypto Macro Radar")
    st.markdown(
        "<div class='muted'>Dashboard <b>macro-driven</b> per monitorare i fondamentali che guidano il <b>sentiment crypto</b>: "
        "liquidità USD, costo reale del denaro, filtri di stress (USD/vol/credito) e conferme (BTC e forza relativa vs Nasdaq). "
        "Obiettivo: evitare FOMO e capire <b>quando il regime cambia</b>.</div>",
        unsafe_allow_html=True
    )

    # Sidebar
    st.sidebar.header("Settings")
    if st.sidebar.button("🔄 Refresh data (clear cache)"):
        st.cache_data.clear()
        st.rerun()

    years_back = st.sidebar.slider("History (years)", 5, 30, 10)
    layout_mode = st.sidebar.selectbox("Layout mode", ["Auto", "Wallboard-first", "Deep dive-first"], index=0)

    st.sidebar.divider()
    st.sidebar.subheader("Crypto alert thresholds (heuristics)")
    thr_real_yield = st.sidebar.slider("Real yield restrictive >", 0.0, 4.0, 2.0, 0.1)
    thr_vix = st.sidebar.slider("VIX stress >", 10.0, 60.0, 25.0, 0.5)
    thr_hy = st.sidebar.slider("HY OAS stress >", 3.0, 12.0, 6.0, 0.25)
    dxy_ma_days = st.sidebar.slider("USD trend filter MA (days)", 50, 300, 200, 10)
    rs_days = st.sidebar.slider("BTC/QQQ RS confirmation window (days)", 10, 90, 30, 5)

    today = datetime.now(timezone.utc).date()
    start_date = (today - DateOffset(years=years_back)).date().isoformat()

    fred_key = get_fred_api_key()
    if fred_key is None:
        st.sidebar.error("⚠️ Missing `FRED_API_KEY` in secrets.")

    # Fetch data
    with st.spinner("Loading data (FRED + yfinance)..."):
        fred = {
            "real_10y": fetch_fred_series("DFII10", start_date),
            "nominal_10y": fetch_fred_series("DGS10", start_date),
            "hy_oas": fetch_fred_series("BAMLH0A0HYM2", start_date),
            "fed_balance_sheet": fetch_fred_series("WALCL", start_date),
            "rrp": fetch_fred_series("RRPONTSYD", start_date),
            "usd_fred": fetch_fred_series("DTWEXBGS", start_date),
        }

        indicators = {}

        # FRED direct
        indicators["real_10y"] = fred["real_10y"]
        indicators["nominal_10y"] = fred["nominal_10y"]
        indicators["hy_oas"] = fred["hy_oas"]
        indicators["fed_balance_sheet"] = fred["fed_balance_sheet"]
        indicators["rrp"] = fred["rrp"]

        # YFinance
        yf_map = fetch_yf_many(["DX-Y.NYB", "^VIX", "BTC-USD", "QQQ"], start_date)

        dxy = yf_map.get("DX-Y.NYB", pd.Series(dtype=float))
        if dxy is None or dxy.empty:
            dxy = fred["usd_fred"]
        indicators["usd_index"] = dxy

        indicators["vix"] = yf_map.get("^VIX", pd.Series(dtype=float))
        btc = yf_map.get("BTC-USD", pd.Series(dtype=float))
        qqq = yf_map.get("QQQ", pd.Series(dtype=float))
        indicators["btc_price"] = btc
        indicators["nasdaq"] = qqq

        # Derived: BTC / QQQ RS
        if btc is not None and qqq is not None and (not btc.empty) and (not qqq.empty):
            joined = btc.to_frame("BTC").join(qqq.to_frame("QQQ"), how="inner").dropna()
            indicators["btc_vs_qqq"] = (joined["BTC"] / joined["QQQ"]).dropna()
        else:
            indicators["btc_vs_qqq"] = pd.Series(dtype=float)

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
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Operating lines (crypto)
    ops = crypto_operating_lines(block_scores, indicator_scores)

    # ============================================================
    # ALERTS (crypto-specific)
    # ============================================================
    s_real = indicators.get("real_10y", pd.Series(dtype=float)).dropna()
    s_vix = indicators.get("vix", pd.Series(dtype=float)).dropna()
    s_hy = indicators.get("hy_oas", pd.Series(dtype=float)).dropna()
    s_usd = indicators.get("usd_index", pd.Series(dtype=float)).dropna()
    s_rs = indicators.get("btc_vs_qqq", pd.Series(dtype=float)).dropna()

    usd_above_ma = pd.Series(dtype=bool)
    usd_below_ma = pd.Series(dtype=bool)
    if not s_usd.empty and len(s_usd) > dxy_ma_days + 10:
        usd_ma = s_usd.rolling(dxy_ma_days).mean()
        usd_above_ma = (s_usd > usd_ma).dropna()
        usd_below_ma = (s_usd < usd_ma).dropna()

    # RS confirmation: RS up over rs_days
    rs_up = pd.Series(dtype=bool)
    if not s_rs.empty and len(s_rs) > rs_days + 5:
        rs_up = (s_rs / s_rs.shift(rs_days) - 1.0) > 0.0
        rs_up = rs_up.dropna()

    cond_real_restrictive = (s_real > thr_real_yield) if not s_real.empty else pd.Series(dtype=bool)
    cond_vix_stress = (s_vix > thr_vix) if not s_vix.empty else pd.Series(dtype=bool)
    cond_hy_stress = (s_hy > thr_hy) if not s_hy.empty else pd.Series(dtype=bool)
    cond_usd_tight = usd_above_ma if len(usd_above_ma) else pd.Series(dtype=bool)
    cond_usd_easing = usd_below_ma if len(usd_below_ma) else pd.Series(dtype=bool)
    cond_rs_confirm = rs_up if len(rs_up) else pd.Series(dtype=bool)

    recent_start = pd.Timestamp(datetime.now(timezone.utc).date() - timedelta(days=30))

    alerts_def = [
        {
            "name": "Real yields restrictive (crypto headwind)",
            "cond": cond_real_restrictive,
            "severity": "bad",
            "so_what": "Real yield sopra soglia: risk-free reale competitivo → crypto tende a soffrire (soprattutto alt beta).",
            "trigger": f"DFII10 > {thr_real_yield:.1f}%",
        },
        {
            "name": "System stress (VIX high) — crypto risk-off",
            "cond": cond_vix_stress,
            "severity": "bad",
            "so_what": "VIX alto: risk budgets si stringono → crypto raramente riparte in modo sostenibile.",
            "trigger": f"VIX > {thr_vix:.1f}",
        },
        {
            "name": "Credit stress (HY spreads) — deleveraging risk",
            "cond": cond_hy_stress,
            "severity": "bad",
            "so_what": "Credito in stress: aumenta probabilità di deleveraging → crypto vulnerabile.",
            "trigger": f"HY OAS > {thr_hy:.2f}pp",
        },
        {
            "name": "USD tightening impulse — macro headwind",
            "cond": cond_usd_tight,
            "severity": "warn",
            "so_what": "USD sopra trend: tightening globale → spesso frena asset risk e crypto.",
            "trigger": f"USD > MA{dxy_ma_days}",
        },
        {
            "name": "USD easing impulse — macro tailwind",
            "cond": cond_usd_easing,
            "severity": "info",
            "so_what": "USD sotto trend: condizioni più easy → crypto respira (se stress non sale).",
            "trigger": f"USD < MA{dxy_ma_days}",
        },
        {
            "name": "BTC leadership confirmation (RS vs Nasdaq)",
            "cond": cond_rs_confirm,
            "severity": "info",
            "so_what": "BTC che sovraperforma Nasdaq per settimane: segnale di leadership e domanda crypto più solida.",
            "trigger": f"BTC/QQQ RS up over {rs_days}d",
        },
    ]

    active_alerts = []
    recent_triggers = []
    for a in alerts_def:
        if is_active(a["cond"]):
            active_alerts.append(a)
        dates = last_n_triggers(a["cond"], recent_start)
        if dates:
            recent_triggers.append({"Alert": a["name"], "Trigger": a["trigger"], "Dates (last 30d)": ", ".join(dates)})

    # ============================================================
    # TABS
    # ============================================================
    tabs = st.tabs(["Overview", "Wallboard", "Deep dive", "What changed", "Alerts", "Report"])

    # ============================================================
    # OVERVIEW
    # ============================================================
    with tabs[0]:
        st.markdown(
            "<div class='section'>"
            "<div class='sectionHead'>"
            "<div><div class='sectionTitle'>Crypto regime snapshot</div>"
            "<div class='sectionDesc'>Prima: regime complessivo per crypto. Secondo: blocchi driver. Terzo: operating lines (BTC/alt/leverage).</div></div>"
            "</div>",
            unsafe_allow_html=True
        )

        gs_txt = "n/a" if np.isnan(global_score) else f"{global_score:.1f}"
        st.markdown(
            f"""
            <div class="grid2">
              <div class="card">
                <div class="cardTitle">Global Score (0–100) — Crypto macro regime</div>
                <div class="cardValue">{gs_txt}</div>
                <div class="cardSub">{pill_html(global_status)}</div>
                <div class="cardSub">{score_bar_html(global_score)}</div>
                <div class="cardSub">
                  <span class="pill info"><span class="dot" style="background:var(--accent2)"></span>
                    Regime crypto = liquidità + real rates + stress filters, non narrative
                  </span>
                </div>
              </div>
              <div class="card">
                <div class="cardTitle">Data & display</div>
                <div class="cardSub">
                  Now: <b>{now_utc}</b><br/>
                  Latest datapoint: <b>{('n/a' if data_max_date is None else str(pd.to_datetime(data_max_date).date()))}</b><br/>
                  History window: <b>{years_back}y</b><br/>
                  Layout: <b>{layout_mode}</b>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        def block_line(bkey):
            name = BLOCKS[bkey]["name"]
            sc = block_scores[bkey]["score"]
            stt = block_scores[bkey]["status"]
            sc_txt = "n/a" if np.isnan(sc) else f"{sc:.1f}"
            return f"{name}: <b>{status_label(stt)}</b> ({sc_txt})"

        st.markdown(
            f"""
            <div class="grid2" style="margin-top:12px;">
              <div class="card">
                <div class="cardTitle">Macro drivers (what moves crypto)</div>
                <div class="cardSub">
                  {block_line("liquidity")}<br/>
                  {block_line("real_cost")}<br/>
                  {block_line("filters")}
                </div>
              </div>
              <div class="card">
                <div class="cardTitle">Crypto confirmation</div>
                <div class="cardSub">{block_line("crypto_confirm")}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        st.markdown("<div class='sectionTitle'>Operating lines (crypto)</div>", unsafe_allow_html=True)
        st.markdown("<div class='sectionDesc'>Sintesi operativa orientata a gestione rischio (non segnali di trading): stance, beta, leverage.</div>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="grid3" style="margin-top:10px;">
              <div class="card">
                <div class="cardTitle">Crypto stance</div>
                <div class="cardValue">{ops["Crypto stance"]["stance"]}</div>
                <div class="cardSub">{ops["Crypto stance"]["why"]}<br/><span class="muted">{ops["Crypto stance"]["context"]}</span></div>
              </div>
              <div class="card">
                <div class="cardTitle">BTC vs Alt beta</div>
                <div class="cardValue">{ops["BTC vs Alt beta"]["stance"]}</div>
                <div class="cardSub">{ops["BTC vs Alt beta"]["why"]}<br/><span class="muted">{ops["BTC vs Alt beta"]["context"]}</span></div>
              </div>
              <div class="card">
                <div class="cardTitle">Leverage discipline</div>
                <div class="cardValue">{ops["Leverage"]["stance"]}</div>
                <div class="cardSub">{ops["Leverage"]["why"]}<br/><span class="muted">{ops["Leverage"]["context"]}</span></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        with st.expander("How to read Risk-on / Neutral / Risk-off for crypto", expanded=False):
            st.markdown(
                """
**Risk-on (crypto):** liquidità USD non drena, real yields scendono o non stringono, USD/stress non peggiorano; BTC mostra leadership.  
**Neutral:** segnali misti → sizing prudente, preferire conferme (RS) ed evitare leverage.  
**Risk-off:** real yields alti e/o USD forte e/o vol/credito in stress → crypto beta viene venduto; proteggere downside.  

**Nota:** la dashboard non “prevede” prezzi: ti aiuta a capire se l’ambiente macro è **supportivo** o **restrittivo** per crypto.
                """.strip()
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # ============================================================
    # WALLBOARD
    # ============================================================
    with tabs[1]:
        st.markdown("## Wallboard")
        st.markdown("<div class='muted'>Tiles (senza grafici): lettura veloce del regime crypto e dei driver macro.</div>", unsafe_allow_html=True)

        gs_txt = "n/a" if np.isnan(global_score) else f"{global_score:.1f}"
        st.markdown(
            f"""
            <div class="grid2">
              <div class="card">
                <div class="cardTitle">Overall crypto regime</div>
                <div class="cardValue">{gs_txt}</div>
                <div class="cardSub">{pill_html(global_status)}</div>
                <div class="cardSub">{score_bar_html(global_score)}</div>
              </div>
              <div class="card">
                <div class="cardTitle">Active alerts (today)</div>
                <div class="cardSub">
                  {"None" if not active_alerts else "<br/>".join([f"<b>{a['name']}</b> — {a['trigger']}<br/><span class='muted'>{a['so_what']}</span>" for a in active_alerts[:6]])}
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        def render_group(title: str, desc: str, keys: list[str]):
            st.markdown(
                f"""
                <div class="section">
                  <div class="sectionHead">
                    <div>
                      <div class="sectionTitle">{title}</div>
                      <div class="sectionDesc">{desc}</div>
                    </div>
                  </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("<div class='wbGrid'>", unsafe_allow_html=True)
            for k in keys:
                s = indicators.get(k, pd.Series(dtype=float))
                if s is None or s.empty:
                    meta = INDICATOR_META[k]
                    st.markdown(
                        f"""
                        <div class="wbTile" style="opacity:0.85;">
                          <div>
                            <div class="wbName">{meta["label"]}</div>
                            <div class="wbMeta">{meta["source"]}</div>
                            <div class="wbRow">
                              <div class="wbVal">Missing</div>
                              <div>{pill_html("n/a")}</div>
                            </div>
                            <div class="wbSmall" style="margin-top:10px;">
                              No data available in the selected history window.
                            </div>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    with st.expander(f"Indicator guide — {meta['label']}", expanded=False):
                        exp = meta["expander"]
                        st.markdown(f"**What it is:** {exp.get('what','')}")
                        st.markdown(f"**Reference levels / thresholds:** {exp.get('reference','')}")
                        st.markdown("**How to read it (crypto lens):**")
                        st.markdown(exp.get("interpretation", ""))
                        st.markdown(f"**Why it matters for crypto:** {exp.get('bridge','')}")
                else:
                    wallboard_tile(k, s, indicator_scores)
            st.markdown("</div></div>", unsafe_allow_html=True)

        render_group("1) USD Liquidity / Plumbing", "Il motore: liquidità USD e plumbing FED (driver primario per crypto).", ["fed_balance_sheet", "rrp"])
        render_group("2) Real Cost of Money", "Il vincolo: real yields e tassi (kryptonite o supporto).", ["real_10y", "nominal_10y"])
        render_group("3) Stress & Filters", "Filtri obbligatori: USD, vol, credito (evitano falsi risk-on crypto).", ["usd_index", "vix", "hy_oas"])
        render_group("4) Crypto Confirmation", "Conferme: BTC e leadership vs Nasdaq (RS).", ["btc_price", "btc_vs_qqq"])

    # ============================================================
    # DEEP DIVE
    # ============================================================
    with tabs[2]:
        st.markdown("## Deep dive")
        st.markdown("<div class='muted'>Grafici completi + guida. Seleziona una sezione e scorri.</div>", unsafe_allow_html=True)

        group = st.selectbox(
            "Select section",
            ["USD Liquidity / Plumbing", "Real Cost of Money", "Stress & Filters", "Crypto Confirmation"],
            index=0
        )

        group_map = {
            "USD Liquidity / Plumbing": ["fed_balance_sheet", "rrp"],
            "Real Cost of Money": ["real_10y", "nominal_10y"],
            "Stress & Filters": ["usd_index", "vix", "hy_oas"],
            "Crypto Confirmation": ["btc_price", "btc_vs_qqq"],
        }

        keys = group_map[group]
        for k in keys:
            meta = INDICATOR_META[k]
            s = indicators.get(k, pd.Series(dtype=float))
            st.markdown("<div class='section'>", unsafe_allow_html=True)

            sc = indicator_scores.get(k, {})
            score = sc.get("score", np.nan)
            status = sc.get("status", "n/a")
            latest = sc.get("latest", np.nan)
            latest_txt = fmt_value(latest, meta["unit"], meta.get("scale", 1.0))

            tr = recent_trend(s)
            wlab = tr["window_label"]
            d = tr["delta_pct"]
            arrow = tr["arrow"]
            d_txt = "n/a" if np.isnan(d) else f"{d:+.1f}%"

            st.markdown(
                f"""
                <div class="sectionHead">
                  <div>
                    <div class="sectionTitle">{meta["label"]}</div>
                    <div class="sectionDesc">{meta["source"]}</div>
                  </div>
                  <div style="text-align:right;">
                    <div class="kpiRow">
                      <span class="pill">Latest: <b>{latest_txt}</b></span>
                      {pill_html(status)}
                      <span class="pill">Score: <b>{("n/a" if np.isnan(score) else f"{score:.0f}")}</b></span>
                      <span class="pill">Trend ({wlab}): <b>{arrow} {d_txt}</b></span>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if s is None or s.empty:
                st.warning("Missing data for this indicator in the selected history window.")
            else:
                fig = plot_premium(s, meta["label"], ref_line=meta.get("ref_line", None), height=340)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=f"deep_{k}")

            with st.expander("Indicator guide (definition, thresholds, why it matters for crypto)", expanded=False):
                exp = meta["expander"]
                st.markdown(f"**What it is:** {exp.get('what','')}")
                st.markdown(f"**Reference levels / thresholds:** {exp.get('reference','')}")
                st.markdown("**How to read it (crypto lens):**")
                st.markdown(exp.get("interpretation", ""))
                st.markdown(f"**Why it matters for crypto:** {exp.get('bridge','')}")

            st.markdown("</div>", unsafe_allow_html=True)

    # ============================================================
    # WHAT CHANGED (crypto language)
    # ============================================================
    with tabs[3]:
        st.markdown("## What changed")
        st.markdown(
            "<div class='muted'>Focus: cosa sta cambiando che può muovere il regime crypto (1M/30d/7d) e come impatta (tailwind/headwind).</div>",
            unsafe_allow_html=True
        )

        rows = []
        for key, meta in INDICATOR_META.items():
            s = indicators.get(key, pd.Series(dtype=float))
            if s is None or s.empty:
                continue

            tr = recent_trend(s)
            window = tr["window_label"]
            dwin = tr["delta_pct"]

            d7 = pct_change_over_days(s, 7)
            d30 = pct_change_over_days(s, 30)
            d90 = pct_change_over_days(s, 90)

            sc = indicator_scores.get(key, {})
            score = sc.get("score", np.nan)
            status = sc.get("status", "n/a")
            mode = meta.get("scoring_mode", "z5y")

            if np.isnan(score):
                prox = 0.0
            else:
                prox = max(0.0, 20.0 - min(abs(score - 40), abs(score - 60))) / 20.0
            move = 0.0 if np.isnan(dwin) else min(1.0, abs(dwin) / 10.0)
            attention = 0.55 * prox + 0.45 * move
            watch = "WATCH" if attention >= 0.55 else ""

            rows.append({
                "Indicator": meta["label"],
                "Scoring": mode,
                "Regime": status_label(status),
                "Score": (np.nan if np.isnan(score) else round(score, 1)),
                f"Trend ({window}) %": (np.nan if np.isnan(dwin) else round(dwin, 2)),
                "Δ 7d %": (np.nan if np.isnan(d7) else round(d7, 2)),
                "Δ 30d %": (np.nan if np.isnan(d30) else round(d30, 2)),
                "Δ 1Q %": (np.nan if np.isnan(d90) else round(d90, 2)),
                "Watch": watch,
                "Attention": round(attention, 2),
                "So what (crypto)": crypto_so_what_line(key, dwin),
                "_key": key,
            })

        if not rows:
            st.info("No sufficient data to compute changes.")
        else:
            df = pd.DataFrame(rows).sort_values(["Watch", "Attention"], ascending=[True, False]).reset_index(drop=True)

            wl = df[df["Watch"] == "WATCH"].sort_values("Attention", ascending=False).head(8)
            if not wl.empty:
                st.markdown("### Watchlist (likely regime relevance for crypto)")
                for _, r in wl.iterrows():
                    trend_col = [c for c in df.columns if c.startswith("Trend")][0]
                    st.markdown(
                        f"""
                        <div class='card' style='margin-bottom:10px;'>
                          <div class='cardTitle'>{r['Indicator']}</div>
                          <div class='cardSub'>
                            Regime: <b>{r['Regime']}</b> · Score: <b>{r['Score']}</b> · {trend_col}: <b>{r[trend_col]:+,.2f}%</b><br/>
                            <span class='muted'>{r['So what (crypto)']}</span>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            st.markdown("### Full table")
            show = df.drop(columns=["_key"])
            st.dataframe(show, use_container_width=True, hide_index=True)
            st.caption("Note: variazioni % basate sull’osservazione disponibile più vicina; frequenza serie diversa.")

    # ============================================================
    # ALERTS TAB
    # ============================================================
    with tabs[4]:
        st.markdown("## Alerts & triggers (crypto regime)")
        st.markdown("<div class='muted'>Alert = condizioni osservabili che spesso anticipano cambio di marcia nel sentiment crypto (2–6 settimane).</div>", unsafe_allow_html=True)

        st.markdown("<div class='grid2'>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='cardTitle'>Active alerts (today)</div>", unsafe_allow_html=True)
        if not active_alerts:
            st.markdown("<div class='cardSub'>Nessun alert attivo con le soglie attuali.</div>", unsafe_allow_html=True)
        else:
            for a in active_alerts:
                sev = a["severity"]
                pill_cls = "pill bad" if sev == "bad" else ("pill warn" if sev == "warn" else "pill info")
                dot_col = "var(--bad)" if sev == "bad" else ("var(--warn)" if sev == "warn" else "var(--accent2)")
                st.markdown(
                    f"<div class='cardSub'><span class='{pill_cls}'><span class='dot' style='background:{dot_col}'></span>{a['name']}</span>"
                    f" &nbsp; <b>{a['trigger']}</b><br/><span class='muted'>{a['so_what']}</span></div>",
                    unsafe_allow_html=True
                )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='cardTitle'>Recent triggers (last 30 days)</div>", unsafe_allow_html=True)
        if not recent_triggers:
            st.markdown("<div class='cardSub'>Nessun trigger recente rilevato.</div>", unsafe_allow_html=True)
        else:
            dfr = pd.DataFrame(recent_triggers)
            st.dataframe(dfr, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        st.markdown("<div class='sectionTitle'>How to react (crypto playbook)</div>", unsafe_allow_html=True)
        st.markdown(
            """
- **2+ alert “bad” insieme** → modalità difensiva: riduci alt beta/leverage, preferisci liquidità e qualità (BTC).
- **Real yields restrictive + USD tightening** → ambiente duro: evitare inseguimenti, aspettare miglioramento macro.
- **Stress (VIX/HY) che rientra + USD easing** → condizioni migliorano: aumentare rischio gradualmente.
- **RS BTC vs Nasdaq in salita per settimane** → conferma leadership: più affidabile un risk-on crypto (sempre con sizing).
            """.strip()
        )

    # ============================================================
    # REPORT TAB
    # ============================================================
    with tabs[5]:
        st.markdown("## Report (copy/paste)")
        st.markdown("<div class='muted'>Genera un blocco copiaincolla (prompt + payload YAML) per ottenere un report AI sul regime crypto in un’altra chat.</div>", unsafe_allow_html=True)

        def build_yaml_payload():
            payload_lines = []
            payload_lines.append("crypto_macro_payload:")
            payload_lines.append(f"  generated_at_utc: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
            payload_lines.append(f"  history_years: {years_back}")
            payload_lines.append(f"  latest_datapoint_date: {('null' if data_max_date is None else str(pd.to_datetime(data_max_date).date()))}")
            payload_lines.append(f"  global_score: {0.0 if np.isnan(global_score) else round(global_score, 1)}")
            payload_lines.append(f"  global_status: {global_status}")
            payload_lines.append("  thesis: \"Crypto sentiment is mainly driven by USD liquidity, real rates, and system risk appetite; narratives are secondary for timing.\"")

            payload_lines.append("  blocks:")
            for bkey, binfo in BLOCKS.items():
                bscore = block_scores[bkey]["score"]
                bstatus = block_scores[bkey]["status"]
                payload_lines.append(f"    - key: \"{bkey}\"")
                payload_lines.append(f"      name: \"{binfo['name']}\"")
                payload_lines.append(f"      group: \"{binfo['group']}\"")
                payload_lines.append(f"      weight: {binfo['weight']}")
                payload_lines.append(f"      score: {0.0 if np.isnan(bscore) else round(bscore, 1)}")
                payload_lines.append(f"      status: {bstatus}")

            payload_lines.append("  operating_lines:")
            for k in ["Crypto stance", "BTC vs Alt beta", "Leverage"]:
                payload_lines.append(f"    {k.lower().replace(' ','_')}:")
                payload_lines.append(f"      stance: \"{ops[k]['stance']}\"")
                payload_lines.append(f"      why: \"{ops[k]['why']}\"")
                payload_lines.append(f"      context: \"{ops[k]['context']}\"")

            payload_lines.append("  alerts:")
            payload_lines.append("    thresholds:")
            payload_lines.append(f"      real_yield_restrictive_gt: {thr_real_yield:.2f}")
            payload_lines.append(f"      vix_stress_gt: {thr_vix:.2f}")
            payload_lines.append(f"      hy_oas_stress_gt: {thr_hy:.2f}")
            payload_lines.append(f"      usd_ma_days: {dxy_ma_days}")
            payload_lines.append(f"      rs_window_days: {rs_days}")

            payload_lines.append("    active_today:")
            if not active_alerts:
                payload_lines.append("      - none")
            else:
                for a in active_alerts:
                    payload_lines.append(f"      - name: \"{a['name']}\"")
                    payload_lines.append(f"        trigger: \"{a['trigger']}\"")
                    payload_lines.append(f"        severity: \"{a['severity']}\"")
                    payload_lines.append(f"        so_what: \"{a['so_what']}\"")

            payload_lines.append("    recent_triggers_last_30d:")
            if not recent_triggers:
                payload_lines.append("      - none")
            else:
                for r in recent_triggers:
                    payload_lines.append(f"      - alert: \"{r['Alert']}\"")
                    payload_lines.append(f"        trigger: \"{r['Trigger']}\"")
                    payload_lines.append(f"        dates: \"{r['Dates (last 30d)']}\"")

            payload_lines.append("  indicators:")
            for key, meta in INDICATOR_META.items():
                s_info = indicator_scores.get(key, {})
                score = s_info.get("score", np.nan)
                status = s_info.get("status", "n/a")
                latest = s_info.get("latest", np.nan)
                series = indicators.get(key, pd.Series(dtype=float))

                tr = recent_trend(series)
                window = tr["window_label"]
                dwin = tr["delta_pct"]

                payload_lines.append(f"    - key: \"{key}\"")
                payload_lines.append(f"      name: \"{meta['label']}\"")
                payload_lines.append(f"      source: \"{meta['source']}\"")
                payload_lines.append(f"      scoring_mode: \"{meta.get('scoring_mode','z5y')}\"")
                payload_lines.append(f"      latest_value: \"{fmt_value(latest, meta['unit'], meta.get('scale', 1.0))}\"")
                payload_lines.append(f"      score: {0.0 if np.isnan(score) else round(score, 1)}")
                payload_lines.append(f"      status: {status}")
                payload_lines.append(f"      trend_window: \"{window}\"")
                payload_lines.append(f"      trend_change_pct: {0.0 if np.isnan(dwin) else round(dwin, 2)}")
                payload_lines.append(f"      reference_line: {('null' if meta.get('ref_line', None) is None else meta.get('ref_line'))}")
                payload_lines.append(f"      reference_notes: \"{meta['expander'].get('reference','')}\"")
                payload_lines.append(f"      so_what_crypto: \"{crypto_so_what_line(key, dwin)}\"")

            return "\n".join(payload_lines)

        generate = st.button("Generate prompt + payload", type="primary")
        if generate:
            payload_text = build_yaml_payload()
            one_shot = (
                "### COPY/PASTE BELOW (PROMPT + PAYLOAD)\n\n"
                + REPORT_PROMPT
                + "\n\n---\n\n"
                + "YAML PAYLOAD:\n\n```yaml\n"
                + payload_text
                + "\n```\n"
            )
            st.session_state["one_shot"] = one_shot

        one_shot = st.session_state.get("one_shot", "")
        if one_shot:
            st.text_area("Copy/paste block", one_shot, height=520)
            st.download_button("Download (.txt)", one_shot, file_name="crypto_macro_report_block.txt", mime="text/plain")
        else:
            st.info("Premi “Generate prompt + payload” per creare il blocco copiaincolla.")

if __name__ == "__main__":
    main()
