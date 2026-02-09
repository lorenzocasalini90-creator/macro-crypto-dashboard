import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from pandas.tseries.offsets import DateOffset

# Optional dependency (Streamlit Cloud typically has it; if not, we fallback)
try:
    import yaml
    HAS_YAML = True
except Exception:
    HAS_YAML = False


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Crypto Macro Radar",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# PREMIUM CSS (SOBER, MOBILE-FIRST, NO HTML LEAKAGE)
# ============================================================

st.markdown(
    """
<style>
  :root{
    --bg:#0b0f19;
    --card:#0f1629;
    --card2:#0c1324;
    --border:rgba(255,255,255,0.10);
    --muted:rgba(255,255,255,0.70);
    --text:rgba(255,255,255,0.94);

    --good:rgba(34,197,94,1);
    --warn:rgba(245,158,11,1);
    --bad:rgba(239,68,68,1);
    --info:rgba(99,102,241,1);

    --accent:rgba(99,102,241,1);
    --accentSoft:rgba(99,102,241,0.16);
  }

  .stApp {
    background: radial-gradient(1200px 700px at 20% 0%, #121a33 0%, #0b0f19 45%, #0b0f19 100%);
    color: var(--text);
  }

  /* container spacing */
  .block-container { padding-top: 1.0rem; padding-bottom: 2.0rem; max-width: 1220px; }

  h1, h2, h3, h4 { color: var(--text); letter-spacing: -0.02em; }
  .muted { color: var(--muted); }

  /* Tabs: readable + selected accent */
  button[data-baseweb="tab"]{
    color: rgba(255,255,255,0.90) !important;
    font-weight: 700 !important;
    background: rgba(255,255,255,0.03) !important;
    border-radius: 10px !important;
    margin-right: 6px !important;
    padding: 8px 12px !important;
  }
  button[data-baseweb="tab"][aria-selected="true"]{
    color: rgba(255,255,255,0.98) !important;
    background: var(--accentSoft) !important;
    border: 1px solid rgba(99,102,241,0.45) !important;
  }

  /* Expander consistent */
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
  .cardSub{ margin-top: 8px; font-size: 0.98rem; color: var(--muted); }

  .grid3{ display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
  .grid2{ display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
  .grid4{ display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }

  @media (max-width: 1100px){ .grid4{ grid-template-columns: repeat(2, minmax(0, 1fr)); } }
  @media (max-width: 800px){ .grid3, .grid2{ grid-template-columns: repeat(1, minmax(0, 1fr)); } }
  @media (max-width: 700px){ .grid4{ grid-template-columns: repeat(1, minmax(0, 1fr)); } }

  /* Pills */
  .pill{
    display:inline-flex;
    align-items:center;
    gap:8px;
    padding: 5px 12px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: rgba(255,255,255,0.04);
    font-size: 0.88rem;
    color: var(--text);
    white-space: nowrap;
  }
  .dot{ width: 11px; height: 11px; border-radius: 999px; display:inline-block; }
  .pill.good{ border-color: rgba(34,197,94,0.40); background: rgba(34,197,94,0.12); }
  .pill.warn{ border-color: rgba(245,158,11,0.40); background: rgba(245,158,11,0.12); }
  .pill.bad { border-color: rgba(239,68,68,0.40); background: rgba(239,68,68,0.12); }
  .pill.info{ border-color: rgba(99,102,241,0.40); background: rgba(99,102,241,0.12); }

  /* Section wrapper */
  .section{
    background: rgba(255,255,255,0.025);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 14px;
    box-shadow: 0 10px 28px rgba(0,0,0,0.20);
    margin-bottom: 14px;
  }
  .sectionHead{ display:flex; align-items:baseline; justify-content:space-between; gap: 12px; margin-bottom: 10px; }
  .sectionTitle{ font-size: 1.20rem; font-weight: 860; color: rgba(255,255,255,0.96); }
  .sectionDesc{ font-size: 0.96rem; color: var(--muted); margin-top: 2px; }

  /* Wallboard tiles */
  .wbGrid{
    display:grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
  }
  @media (max-width: 1100px){ .wbGrid{ grid-template-columns: repeat(2, minmax(0, 1fr)); } }
  @media (max-width: 700px){ .wbGrid{ grid-template-columns: repeat(1, minmax(0, 1fr)); } }

  .wbTile{
    background: rgba(255,255,255,0.028);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 14px 14px 12px 14px;
    box-shadow: 0 10px 26px rgba(0,0,0,0.18);
    min-height: 162px;
    display:flex;
    flex-direction:column;
    justify-content:space-between;
  }
  .wbName{ font-size: 0.98rem; font-weight: 860; color: rgba(255,255,255,0.96); margin-bottom: 2px; }
  .wbMeta{ font-size: 0.86rem; color: var(--muted); margin-bottom: 8px; }
  .wbRow{ display:flex; align-items:baseline; justify-content:space-between; gap: 10px; }
  .wbVal{ font-size: 1.65rem; font-weight: 900; letter-spacing:-0.01em; }
  .wbSmall{ font-size: 0.88rem; color: var(--muted); }
  .wbFoot{ display:flex; align-items:center; justify-content:space-between; gap: 10px; margin-top: 10px; }

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

  /* Dataframe tweaks */
  .stDataFrame { border: 1px solid var(--border); border-radius: 12px; overflow:hidden; }
  code { color: rgba(255,255,255,0.88); }

  /* Small divider */
  .hr{ height:1px; background: rgba(255,255,255,0.10); margin: 10px 0 14px 0; }

</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# META / CONFIG
# ============================================================

APP_NAME = "Crypto Macro Radar"
TZ = timezone.utc

def now_utc_str():
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M UTC")

def safe_float(x):
    try:
        if x is None:
            return np.nan
        if isinstance(x, (np.floating, float, int, np.integer)):
            return float(x)
        return float(x)
    except Exception:
        return np.nan

# ============================================================
# INDICATOR META (CRYPTO-FOCUSED)
# ============================================================

INDICATOR_META = {
    # 1) LIQUIDITY (TOP PRIORITY FOR CRYPTO)
    "fed_balance_sheet": {
        "label": "Fed Balance Sheet (WALCL)",
        "unit": "bn USD",
        "direction": +1,
        "source": "FRED WALCL (millions → bn)",
        "scale": 1.0 / 1000.0,  # millions -> bn
        "ref_line": None,
        "scoring_mode": "z5y",
        "group": "1) Liquidità USD",
        "expander": {
            "metric": "Totale attivi Fed (proxy della liquidità di sistema).",
            "what": "Quando la Fed espande (o smette di drenare) riduce la pressione di funding. Crypto tende a beneficiarne.",
            "reference": "BS ↑ o QT che rallenta = tailwind; BS ↓ persistente = headwind (euristica).",
            "interpretation": "- BS in salita: condizioni marginali meno restrittive.\n- BS in calo: drenaggio (QT) → pressione su asset risk/crypto.",
            "so_what": "Se BS smette di scendere mentre altri filtri non peggiorano, aumentano le probabilità di risk-on crypto sostenibile.",
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
        "group": "1) Liquidità USD",
        "expander": {
            "metric": "Cash parcheggiata nella facility RRP (liquidità “immobile”).",
            "what": "Un RRP in rapido calo può rilasciare liquidità marginale verso il sistema.",
            "reference": "RRP ↓ rapido può liberare marginal liquidity (euristica).",
            "interpretation": "- RRP in calo: potenziale tailwind tattico.\n- RRP in salita: cash si rifugia nel risk-free.",
            "so_what": "RRP che scende mentre BTC/ETH tengono o migliorano = spesso contesto più fertile per rialzi crypto.",
        },
    },

    # 2) REAL COST OF MONEY (CRYPTO KRYPTONITE)
    "real_10y": {
        "label": "US 10Y TIPS Real Yield",
        "unit": "%",
        "direction": -1,
        "source": "FRED DFII10",
        "scale": 1.0,
        "ref_line": 2.0,
        "scoring_mode": "z5y",
        "group": "2) Costo reale del denaro",
        "expander": {
            "metric": "Rendimento reale risk-free (competitor diretto della crypto).",
            "what": "Crypto soffre quando il risk-free reale paga bene. Il delta (salita/discesa) conta spesso più del livello.",
            "reference": "0–2% neutro; >2% restrittivo; cali rapidi = risk-on crypto (euristiche).",
            "interpretation": "- Real yield ↑: stringe condizioni, penalizza duration e asset speculativi.\n- Real yield ↓: allenta vincolo e supporta risk appetite.",
            "so_what": "Se real yield scende insieme a USD/stress in calo, aumentano odds di impulso risk-on crypto più pulito.",
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
        "group": "2) Costo reale del denaro",
        "expander": {
            "metric": "Tasso nominale benchmark (discount rate).",
            "what": "Per crypto conta soprattutto quando si muove velocemente (tightening shock) o quando scende insieme ai real yields.",
            "reference": "Salite rapide = tightening impulse (euristica).",
            "interpretation": "- Nominal ↑ veloce: pressione su risk.\n- Nominal ↓: può essere tailwind, ma se scende per paura crescita può essere risk-off.",
            "so_what": "Nominal ↓ + real ↓ + stress ↓ = combinazione più spesso costruttiva per crypto.",
        },
    },

    # 3) STRESS FILTERS (USD + VOL + CREDIT)
    "usd_index": {
        "label": "USD Index (DXY proxy)",
        "unit": "",
        "direction": -1,
        "source": "yfinance DX-Y.NYB (fallback: FRED DTWEXBGS)",
        "scale": 1.0,
        "ref_line": None,
        "scoring_mode": "z5y",
        "group": "3) Propensione al rischio sistemica",
        "expander": {
            "metric": "Forza USD (proxy di tightening globale).",
            "what": "USD forte = tightening delle condizioni globali, spesso frena crypto e asset risk.",
            "reference": "Trend ↓ = respiro per crypto; trend ↑ = headwind (euristica).",
            "interpretation": "- USD ↑: funding più stretto (specialmente fuori US).\n- USD ↓: condizioni più permissive.",
            "so_what": "USD che rompe al ribasso (trend) spesso anticipa migliori fasi per crypto, specie se real yields non salgono.",
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
        "group": "3) Propensione al rischio sistemica",
        "expander": {
            "metric": "Volatilità implicita equity (proxy di risk aversion).",
            "what": "Crypto raramente riparte bene con VIX alto: le condizioni di rischio restano strette.",
            "reference": "<15 risk-on; 15–25 neutro; >25 stress (euristica).",
            "interpretation": "- VIX ↑: risk premia aumentano, deleveraging più probabile.\n- VIX ↓: condizioni più adatte a risk-on.",
            "so_what": "Un VIX in calo mentre BTC/ETH confermano (RS) è uno dei filtri migliori per ‘risk-on credibile’.",
        },
    },
    "hy_oas": {
        "label": "US High Yield OAS",
        "unit": "pp",
        "direction": -1,
        "source": "FRED BAMLH0A0HYM2",
        "scale": 1.0,
        "ref_line": 5.0,
        "scoring_mode": "z5y",
        "group": "3) Propensione al rischio sistemica",
        "expander": {
            "metric": "Spread high yield (stress creditizio).",
            "what": "Quando gli spread si allargano, spesso c’è deleveraging e avversione al rischio: crypto ne risente.",
            "reference": "<4–5% benigno; >6–7% stress (euristica).",
            "interpretation": "- OAS ↑: rischio default/premio rischio ↑ → risk-off.\n- OAS ↓: credito più ‘tranquillo’ → risk-on più sostenibile.",
            "so_what": "OAS in rientro (non solo ‘stabile’) aiuta a rendere più robusto il risk-on crypto.",
        },
    },

    # 4) CRYPTO CONFIRMATION (BTC + ETH)
    "btc": {
        "label": "BTC Price",
        "unit": "USD",
        "direction": +1,
        "source": "yfinance BTC-USD",
        "scale": 1.0,
        "ref_line": None,
        "scoring_mode": "z5y",
        "group": "4) Conferma crypto (BTC & ETH)",
        "expander": {
            "metric": "Prezzo BTC (proxy principale risk-on crypto).",
            "what": "Da solo è rumoroso. Conta soprattutto quando conferma il macro (liquidità/stress).",
            "reference": "Trend e forza relativa contano più del livello.",
            "interpretation": "- BTC ↑ con macro favorevole: conferma.\n- BTC ↑ con macro ostile: può essere rally fragile.\n- BTC ↓ con macro ostile: coerente.",
            "so_what": "Cerca convergenza: macro migliora + BTC regge/rompe su = probabilità più alta di regime risk-on.",
        },
    },
    "eth": {
        "label": "ETH Price",
        "unit": "USD",
        "direction": +1,
        "source": "yfinance ETH-USD",
        "scale": 1.0,
        "ref_line": None,
        "scoring_mode": "z5y",
        "group": "4) Conferma crypto (BTC & ETH)",
        "expander": {
            "metric": "Prezzo ETH (beta più ‘risk-on’ dentro crypto).",
            "what": "ETH tende a sovraperformare quando il risk appetite è più pulito (o quando il mercato ‘allarga’ la partecipazione).",
            "reference": "ETH spesso anticipa ‘risk-on broadening’ quando ETH/BTC risale.",
            "interpretation": "- ETH ↑ con ETH/BTC ↑: segnale di espansione risk-on.\n- ETH ↑ ma ETH/BTC ↓: BTC-driven, più difensivo.",
            "so_what": "Se ETH/BTC smette di scendere e poi sale, spesso migliora la qualità del regime risk-on crypto.",
        },
    },
    "btc_qqq": {
        "label": "BTC vs Nasdaq (BTC/QQQ)",
        "unit": "ratio",
        "direction": +1,
        "source": "Derived (BTC-USD / QQQ)",
        "scale": 1.0,
        "ref_line": None,
        "scoring_mode": "z5y",
        "group": "4) Conferma crypto (BTC & ETH)",
        "expander": {
            "metric": "Forza relativa BTC vs equity growth.",
            "what": "Se BTC sovraperforma il Nasdaq, il bid su crypto è più ‘autonomo’ (domanda più forte).",
            "reference": "RS ↑ per settimane = conferma risk-on crypto (euristica).",
            "interpretation": "- RS ↑: BTC tiene meglio del risk proxy.\n- RS ↓: crypto è più fragile o subordinata al risk generale.",
            "so_what": "BTC/QQQ in trend rialzista è uno dei migliori filtri per capire se crypto ‘sta prendendo leadership’.",
        },
    },
    "eth_btc": {
        "label": "ETH vs BTC (ETH/BTC)",
        "unit": "ratio",
        "direction": +1,
        "source": "Derived (ETH-USD / BTC-USD)",
        "scale": 1.0,
        "ref_line": None,
        "scoring_mode": "z5y",
        "group": "4) Conferma crypto (BTC & ETH)",
        "expander": {
            "metric": "Ampiezza del risk-on dentro crypto (ETH beta).",
            "what": "ETH/BTC che sale spesso coincide con fasi di risk appetite crypto più ampio (alt-breadth).",
            "reference": "Stabilizza → ok; trend ↑ → qualità risk-on migliore (euristica).",
            "interpretation": "- ETH/BTC ↑: partecipazione più ampia.\n- ETH/BTC ↓: mercato più difensivo (preferenza BTC).",
            "so_what": "Un risk-on crypto ‘di qualità’ spesso richiede ETH/BTC non in caduta libera.",
        },
    },
}

# BLOCKS (CRYPTO-FOCUSED)
BLOCKS = {
    "liquidity": {
        "name": "1) Liquidità USD",
        "desc": "Il motore: disponibilità di liquidità e plumbing. Per crypto è spesso la forza #1 nel timing di regime.",
        "weight": 0.30,
        "indicators": ["fed_balance_sheet", "rrp"],
    },
    "real_cost": {
        "name": "2) Costo reale del denaro",
        "desc": "Il vincolo: real yields e tassi. Quando il risk-free reale paga, crypto fatica a competere.",
        "weight": 0.30,
        "indicators": ["real_10y", "nominal_10y"],
    },
    "stress": {
        "name": "3) Propensione al rischio sistemica",
        "desc": "Filtri: USD, vol e credito. Anche con buona liquidità, stress alto può bloccare crypto.",
        "weight": 0.25,
        "indicators": ["usd_index", "vix", "hy_oas"],
    },
    "crypto_confirm": {
        "name": "4) Conferma crypto (BTC & ETH)",
        "desc": "Conferme di prezzo e forza relativa: BTC/QQQ e ETH/BTC aiutano a capire la qualità del regime.",
        "weight": 0.15,
        "indicators": ["btc", "eth", "btc_qqq", "eth_btc"],
    },
}

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
        r = requests.get(url, params=params, timeout=14)
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
        # Normalize index to naive
        s.index = pd.to_datetime(s.index).tz_localize(None) if getattr(s.index, "tz", None) else pd.to_datetime(s.index)
        return s.sort_index()
    except Exception:
        return pd.Series(dtype=float)

@st.cache_data(ttl=3600)
def fetch_yf_many(tickers: list[str], start_date: str) -> dict:
    out = {}
    for t in tickers:
        out[t] = fetch_yf_one(t, start_date)
    return out

# ============================================================
# SCORING + UTILITIES (ROBUST)
# ============================================================

def rolling_percentile_last(hist: pd.Series, latest: float) -> float:
    h = hist.dropna()
    if len(h) < 10 or pd.isna(latest):
        return np.nan
    return float((h <= latest).mean())

def compute_indicator_score(series: pd.Series, direction: int, scoring_mode: str = "z5y"):
    """
    Returns: (score_0_100, signal_raw, latest)
    - z5y: z-score over ~5Y history (clamped)
    - pct20y: percentile over ~20Y history mapped to [-2,+2] (clamped)
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
        sig = (p - 0.5) * 4.0  # [-2,+2]
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
    return "<span class='pill info'><span class='dot' style='background:var(--info)'></span>n/a</span>"

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
        return f"{v:.4f}"
    if unit == "bn USD":
        return f"{v:.1f} bn"
    if unit == "USD":
        # crypto prices: compact
        if v >= 1000:
            return f"${v:,.0f}"
        return f"${v:,.2f}"
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
    """Returns delta window adapted by frequency (daily→30d; monthly/quarterly→1Q)."""
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
# PLOTTING (PREMIUM, DARK)
# ============================================================

def plot_premium(series: pd.Series, title: str, ref_line=None, height: int = 320):
    s = series.dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", line=dict(width=2), name=title))

    if ref_line is not None and not (isinstance(ref_line, float) and np.isnan(ref_line)):
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
# UI COMPONENTS
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

def indicator_guide_markdown(key: str):
    meta = INDICATOR_META[key]
    exp = meta["expander"]
    st.markdown(f"### {meta['label']}")
    st.markdown(f"**Metrica:** {exp.get('metric','')}")
    st.markdown(f"**Cosa cattura:** {exp.get('what','')}")
    st.markdown(f"**Valori di riferimento / soglie:** {exp.get('reference','')}")
    st.markdown("**Come leggerla (bi-direzionale):**")
    st.markdown(exp.get("interpretation", ""))
    st.markdown(f"**So what (crypto):** {exp.get('so_what','')}")

# ============================================================
# REPORT PROMPT (CRYPTO + COMPARISON OVER TIME)
# ============================================================

REPORT_PROMPT = """SYSTEM / ROLE

You are a senior macro-crypto strategist writing an internal PM regime note.
You diagnose the *crypto macro regime* (not price predictions), and translate it into actionable risk posture.
You focus on causal links: USD liquidity, real cost of money, systemic risk appetite, and crypto confirmation (BTC + ETH).

You receive TWO YAML payloads:
- current_payload: latest snapshot
- previous_payload: optional prior snapshot (if present, you MUST compare and explain deltas)

CRITICAL OUTPUT RULES (NON-NEGOTIABLE)

- Follow the exact structure and section order below.
- Do not reorder, merge, or omit sections.
- No speculation beyond the data provided.
- Be explicit about WHAT CHANGED vs previous payload when available.
- Use crisp PM language: drivers → implications → triggers.
- Avoid hype and timing calls. This is regime diagnosis.

MANDATORY REPORT STRUCTURE (FOLLOW EXACTLY)

# Crypto Macro Regime Report
## Internal PM Edition — Liquidity / Real Cost / Stress / Crypto Confirmation
[Insert current UTC date]

How to read “Risk-on / Neutral / Risk-off” for crypto
(Define as behavioral regime labels derived from the dashboard scores. Not forecasts.)

Executive Summary
- Current regime in one paragraph
- If previous_payload exists: what changed and why it matters

Regime Drivers (macro → crypto)
1) Liquidity (USD plumbing)
2) Real Cost of Money (real yields / rates)
3) Systemic Risk Appetite (USD, vol, credit stress)

Crypto Confirmation (BTC & ETH)
- BTC level + BTC/QQQ relative strength
- ETH level + ETH/BTC breadth signal
- Consistency vs macro drivers (confirm / diverge)

What Changed (vs previous snapshot)
- 5–10 bullet points max, sorted by significance
- Include both score changes and trend changes

Implications (risk posture, not trades)
- What to do more of / less of in crypto risk posture
- What requires caution

Key Triggers (2–6 week horizon)
- 3–6 triggers
- Observable thresholds linked to regime shifts

Final Bottom Line
(One paragraph, no bullets. No forecasts.)

STYLE
- Short declarative sentences.
- Cause → effect.
- Crypto-first language, not generic multi-asset commentary.
""".strip()

# ============================================================
# PARSING / COMPARISON FOR PREVIOUS PAYLOAD
# ============================================================

def yaml_load_any(text: str):
    if not text:
        return None
    if HAS_YAML:
        try:
            return yaml.safe_load(text)
        except Exception:
            return None
    # fallback: no yaml library
    return None

def compute_deltas(current: dict, prev: dict) -> dict:
    """
    Returns a dict with stable fields:
    {
      'regime': {'global_score_delta':..., 'global_status_change':...},
      'blocks': {block_key: {'score_delta':..., 'status_change':...}},
      'indicators': {ind_key: {'score_delta':..., 'status_change':..., 'trend_1m_delta':...}}
    }
    """
    out = {
        "regime": {"global_score_delta": None, "global_status_change": None},
        "blocks": {},
        "indicators": {}
    }
    if not prev:
        return out

    # Global
    try:
        cgs = safe_float(current.get("regime", {}).get("global_score"))
        pgs = safe_float(prev.get("regime", {}).get("global_score"))
        out["regime"]["global_score_delta"] = None if np.isnan(cgs) or np.isnan(pgs) else round(cgs - pgs, 2)
        cst = current.get("regime", {}).get("global_status")
        pst = prev.get("regime", {}).get("global_status")
        out["regime"]["global_status_change"] = (cst != pst) if (cst is not None and pst is not None) else None
    except Exception:
        pass

    # Blocks
    cblocks = current.get("blocks", {}) or {}
    pblocks = prev.get("blocks", {}) or {}
    for bk, cv in cblocks.items():
        pv = pblocks.get(bk, {})
        cscore = safe_float(cv.get("score"))
        pscore = safe_float(pv.get("score"))
        cst = cv.get("status")
        pst = pv.get("status")
        out["blocks"][bk] = {
            "score_delta": None if np.isnan(cscore) or np.isnan(pscore) else round(cscore - pscore, 2),
            "status_change": (cst != pst) if (cst is not None and pst is not None) else None,
        }

    # Indicators
    cinds = current.get("indicators", {}) or {}
    pinds = prev.get("indicators", {}) or {}
    for ik, cv in cinds.items():
        pv = pinds.get(ik, {})
        cscore = safe_float(cv.get("score"))
        pscore = safe_float(pv.get("score"))
        cst = cv.get("status")
        pst = pv.get("status")
        ctrend = safe_float(cv.get("trend_1m_pct"))
        ptrend = safe_float(pv.get("trend_1m_pct"))
        out["indicators"][ik] = {
            "score_delta": None if np.isnan(cscore) or np.isnan(pscore) else round(cscore - pscore, 2),
            "status_change": (cst != pst) if (cst is not None and pst is not None) else None,
            "trend_1m_delta": None if np.isnan(ctrend) or np.isnan(ptrend) else round(ctrend - ptrend, 2),
        }
    return out

# ============================================================
# MAIN
# ============================================================

def main():
    # HEADER
    st.markdown("## Crypto Macro Radar")
    st.markdown(
        "<div class='muted'>Dashboard per monitorare i fondamentali alla base del <b>sentiment crypto</b>: "
        "<b>1) Liquidità USD</b>, <b>2) Costo reale del denaro</b>, <b>3) Stress sistemico</b>, "
        "<b>4) Conferme BTC & ETH</b>. "
        "L’obiettivo è capire se il regime sta migliorando o deteriorando, con nessi causali chiari.</div>",
        unsafe_allow_html=True,
    )

    # SIDEBAR SETTINGS
    st.sidebar.header("Settings")
    if st.sidebar.button("🔄 Refresh data (clear cache)"):
        st.cache_data.clear()
        st.rerun()

    years_back = st.sidebar.slider("History (years)", 5, 20, 10)
    layout_mode = st.sidebar.selectbox("Layout mode", ["Auto", "Compact (mobile)", "Deep dive first"], index=0)

    # Alert thresholds (crypto-specific heuristics)
    st.sidebar.subheader("Crypto heuristics (alerts)")
    thr_real_restrictive = st.sidebar.slider("Real yield restrictive >", 0.0, 4.0, 2.0, 0.1)
    thr_vix_stress = st.sidebar.slider("VIX stress >", 10.0, 60.0, 25.0, 0.5)
    thr_hy_stress = st.sidebar.slider("HY OAS stress >", 2.0, 12.0, 6.0, 0.1)

    today = datetime.now(TZ).date()
    start_date = (today - DateOffset(years=years_back)).date().isoformat()
    st.sidebar.markdown(f"**Start date:** {start_date}")

    fred_key = get_fred_api_key()
    if fred_key is None:
        st.sidebar.error("⚠️ Missing `FRED_API_KEY` in secrets (Streamlit Cloud).")

    # Fetch data
    with st.spinner("Loading data (FRED + yfinance)..."):
        fred = {
            "real_10y": fetch_fred_series("DFII10", start_date),
            "nominal_10y": fetch_fred_series("DGS10", start_date),
            "fed_balance_sheet": fetch_fred_series("WALCL", start_date),
            "rrp": fetch_fred_series("RRPONTSYD", start_date),
            "hy_oas": fetch_fred_series("BAMLH0A0HYM2", start_date),
            "usd_fred": fetch_fred_series("DTWEXBGS", start_date),  # fallback
        }

        yf_map = fetch_yf_many(
            ["DX-Y.NYB", "^VIX", "BTC-USD", "ETH-USD", "QQQ"],
            start_date
        )

    # Assemble indicators series
    indicators = {}

    # FRED direct
    indicators["real_10y"] = fred["real_10y"]
    indicators["nominal_10y"] = fred["nominal_10y"]
    indicators["fed_balance_sheet"] = fred["fed_balance_sheet"]
    indicators["rrp"] = fred["rrp"]
    indicators["hy_oas"] = fred["hy_oas"]

    # DXY with fallback
    dxy = yf_map.get("DX-Y.NYB", pd.Series(dtype=float))
    if dxy is None or dxy.empty:
        dxy = fred["usd_fred"]
    indicators["usd_index"] = dxy

    indicators["vix"] = yf_map.get("^VIX", pd.Series(dtype=float))
    indicators["btc"] = yf_map.get("BTC-USD", pd.Series(dtype=float))
    indicators["eth"] = yf_map.get("ETH-USD", pd.Series(dtype=float))
    qqq = yf_map.get("QQQ", pd.Series(dtype=float))

    # Derived: BTC/QQQ and ETH/BTC
    btc = indicators["btc"]
    eth = indicators["eth"]

    if btc is not None and qqq is not None and (not btc.empty) and (not qqq.empty):
        j = btc.to_frame("btc").join(qqq.to_frame("qqq"), how="inner").dropna()
        indicators["btc_qqq"] = (j["btc"] / j["qqq"]).dropna()
    else:
        indicators["btc_qqq"] = pd.Series(dtype=float)

    if eth is not None and btc is not None and (not eth.empty) and (not btc.empty):
        j = eth.to_frame("eth").join(btc.to_frame("btc"), how="inner").dropna()
        indicators["eth_btc"] = (j["eth"] / j["btc"]).dropna()
    else:
        indicators["eth_btc"] = pd.Series(dtype=float)

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
    now_str = now_utc_str()

    # =========================
    # PREVIOUS PAYLOAD UPLOAD
    # =========================
    st.sidebar.subheader("Comparison (optional)")
    prev_file = st.sidebar.file_uploader("Upload previous YAML payload", type=["yaml", "yml", "txt"])
    prev_payload = None
    if prev_file is not None:
        try:
            prev_text = prev_file.read().decode("utf-8")
            prev_payload = yaml_load_any(prev_text)
            if prev_payload is None:
                st.sidebar.warning("Could not parse YAML (missing yaml lib or invalid file).")
        except Exception:
            st.sidebar.warning("Could not read uploaded file.")

    # =========================
    # BUILD CURRENT PAYLOAD (stable fields)
    # =========================

    # Trends for payload (fixed horizons)
    def trend_pack(series: pd.Series):
        return {
            "trend_7d_pct": None if np.isnan(pct_change_over_days(series, 7)) else round(pct_change_over_days(series, 7), 2),
            "trend_1m_pct": None if np.isnan(pct_change_over_days(series, 30)) else round(pct_change_over_days(series, 30), 2),
            "trend_3m_pct": None if np.isnan(pct_change_over_days(series, 90)) else round(pct_change_over_days(series, 90), 2),
            "trend_6m_pct": None if np.isnan(pct_change_over_days(series, 180)) else round(pct_change_over_days(series, 180), 2),
            "trend_1y_pct": None if np.isnan(pct_change_over_days(series, 365)) else round(pct_change_over_days(series, 365), 2),
        }

    current_payload = {
        "meta": {
            "app": APP_NAME,
            "generated_at_utc": datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "history_years": years_back,
            "latest_datapoint_date": None if data_max_date is None else str(pd.to_datetime(data_max_date).date()),
        },
        "regime": {
            "global_score": None if np.isnan(global_score) else round(global_score, 1),
            "global_status": global_status,
        },
        "blocks": {},
        "indicators": {}
    }

    # blocks payload (stable keys)
    for bkey, binfo in BLOCKS.items():
        bscore = block_scores.get(bkey, {}).get("score", np.nan)
        bstatus = block_scores.get(bkey, {}).get("status", "n/a")
        current_payload["blocks"][bkey] = {
            "name": binfo["name"],
            "weight": binfo["weight"],
            "score": None if np.isnan(bscore) else round(bscore, 1),
            "status": bstatus,
            "indicators": binfo["indicators"],
        }

    # indicators payload (stable keys)
    for key, meta in INDICATOR_META.items():
        series = indicators.get(key, pd.Series(dtype=float))
        s_info = indicator_scores.get(key, {})
        score = s_info.get("score", np.nan)
        status = s_info.get("status", "n/a")
        latest = s_info.get("latest", np.nan)

        tp = trend_pack(series)
        current_payload["indicators"][key] = {
            "name": meta["label"],
            "group": meta.get("group", ""),
            "source": meta.get("source", ""),
            "unit": meta.get("unit", ""),
            "latest_value": None if np.isnan(latest) else float(latest),
            "latest_value_fmt": fmt_value(latest, meta["unit"], meta.get("scale", 1.0)),
            "score": None if np.isnan(score) else round(score, 1),
            "status": status,
            **tp,
            "reference_line": None if meta.get("ref_line", None) is None else meta.get("ref_line"),
            "reference_notes": meta["expander"].get("reference", ""),
        }

    deltas = compute_deltas(current_payload, prev_payload) if prev_payload else {
        "regime": {"global_score_delta": None, "global_status_change": None},
        "blocks": {},
        "indicators": {}
    }

    # =========================
    # TABS
    # =========================

    tabs = st.tabs([
        "Overview",
        "Wallboard",
        "Deep dive",
        "What changed",
        "Alerts",
        "Report generation",
    ])

    # ============================================================
    # OVERVIEW (IMMEDIATE CRYPTO DIAGNOSIS)
    # ============================================================
    with tabs[0]:
        # Top: big regime card + what’s pushing / what’s helping
        gs_txt = "n/a" if np.isnan(global_score) else f"{global_score:.1f}"
        gsd = deltas["regime"].get("global_score_delta", None) if prev_payload else None
        gsd_txt = "" if gsd is None else f"Δ vs prev: {gsd:+.1f}"

        # Force diagnosis (simple heuristics)
        def is_bad(key):
            s = indicators.get(key, pd.Series(dtype=float))
            if s is None or s.empty:
                return None
            last = float(s.dropna().iloc[-1])
            if key == "real_10y":
                return last >= float(thr_real_restrictive)
            if key == "vix":
                return last >= float(thr_vix_stress)
            if key == "hy_oas":
                return last >= float(thr_hy_stress)
            return None

        real_bad = is_bad("real_10y")
        vix_bad = is_bad("vix")
        hy_bad = is_bad("hy_oas")

        # Narrative blocks
        headwinds = []
        tailwinds = []

        # Liquidity
        bs_tr = recent_trend(indicators.get("fed_balance_sheet", pd.Series(dtype=float)))
        rrp_tr = recent_trend(indicators.get("rrp", pd.Series(dtype=float)))

        if not np.isnan(bs_tr.get("delta_pct", np.nan)) and bs_tr["delta_pct"] < 0:
            headwinds.append("Fed BS in calo (QT) → drenaggio liquidità (headwind)")
        elif not np.isnan(bs_tr.get("delta_pct", np.nan)) and bs_tr["delta_pct"] >= 0:
            tailwinds.append("Fed BS stabile/in aumento → meno drenaggio (tailwind)")

        if not np.isnan(rrp_tr.get("delta_pct", np.nan)) and rrp_tr["delta_pct"] < 0:
            tailwinds.append("RRP in forte calo → possibile rilascio liquidità marginale")
        elif not np.isnan(rrp_tr.get("delta_pct", np.nan)) and rrp_tr["delta_pct"] > 0:
            headwinds.append("RRP in risalita → cash parcheggiata nel risk-free")

        # Real cost
        if real_bad is True:
            headwinds.append(f"Real yields sopra soglia ({thr_real_restrictive:.1f}%) → crypto compete col risk-free")
        elif real_bad is False:
            tailwinds.append("Real yields non restrittivi → vincolo meno duro")

        # Stress
        if vix_bad is True:
            headwinds.append(f"VIX alto (>{thr_vix_stress:.0f}) → risk aversion, difficile risk-on crypto")
        elif vix_bad is False:
            tailwinds.append("VIX non in stress → più spazio per risk-taking")

        if hy_bad is True:
            headwinds.append(f"HY OAS in stress (>{thr_hy_stress:.1f}) → deleveraging risk")
        elif hy_bad is False:
            tailwinds.append("HY OAS non in stress → contesto più stabile")

        # Crypto confirmations
        rs_btc = indicators.get("btc_qqq", pd.Series(dtype=float))
        rs_eth = indicators.get("eth_btc", pd.Series(dtype=float))
        rs_btc_tr = recent_trend(rs_btc)
        rs_eth_tr = recent_trend(rs_eth)

        if not np.isnan(rs_btc_tr.get("delta_pct", np.nan)) and rs_btc_tr["delta_pct"] > 0:
            tailwinds.append("BTC sovraperforma Nasdaq (RS ↑) → domanda crypto più ‘autonoma’")
        elif not np.isnan(rs_btc_tr.get("delta_pct", np.nan)) and rs_btc_tr["delta_pct"] < 0:
            headwinds.append("BTC underperforma Nasdaq (RS ↓) → regime crypto più fragile")

        if not np.isnan(rs_eth_tr.get("delta_pct", np.nan)) and rs_eth_tr["delta_pct"] > 0:
            tailwinds.append("ETH/BTC in rialzo → ‘breadth’ risk-on crypto migliore")
        elif not np.isnan(rs_eth_tr.get("delta_pct", np.nan)) and rs_eth_tr["delta_pct"] < 0:
            headwinds.append("ETH/BTC in calo → mercato crypto più difensivo (preferenza BTC)")

        # Crypto stance (non timing)
        stance = "Neutral"
        if global_status == "risk_on" and (real_bad is False) and (vix_bad is False):
            stance = "Risk-on (conferme buone, ma gestire sizing)"
        elif global_status == "risk_off" or (real_bad is True and vix_bad is True):
            stance = "Risk-off (priorità: protezione e liquidità)"
        else:
            stance = "Neutral (selettività, attendere convergenze)"

        st.markdown(
            f"""
            <div class="grid3">
              <div class="card">
                <div class="cardTitle">Global Crypto Regime Score (0–100)</div>
                <div class="cardValue">{gs_txt}</div>
                <div class="cardSub">{pill_html(global_status)} <span class="pill info">{gsd_txt}</span></div>
                <div class="cardSub" style="margin-top:10px;">
                  <b>Stance (non timing):</b> {stance}
                </div>
              </div>

              <div class="card">
                <div class="cardTitle">Forze che SCHIACCIANO / rallentano crypto</div>
                <div class="cardSub">
                  {"<br/>".join([f"• {x}" for x in headwinds[:6]]) if headwinds else "• Nessun headwind dominante rilevato (dati limitati o misti)."}
                </div>
              </div>

              <div class="card">
                <div class="cardTitle">Forze che SOSTENGONO crypto</div>
                <div class="cardSub">
                  {"<br/>".join([f"• {x}" for x in tailwinds[:6]]) if tailwinds else "• Nessun tailwind dominante rilevato (dati limitati o misti)."}
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

        # Sections 1-4 with big headings + short paragraphs (as requested)
        for bkey in ["liquidity", "real_cost", "stress", "crypto_confirm"]:
            binfo = BLOCKS[bkey]
            bsc = block_scores[bkey]["score"]
            bst = block_scores[bkey]["status"]
            btxt = "n/a" if np.isnan(bsc) else f"{bsc:.1f}"
            delta = deltas["blocks"].get(bkey, {}).get("score_delta", None) if prev_payload else None
            delta_txt = "" if delta is None else f"Δ vs prev: {delta:+.1f}"

            st.markdown(
                f"""
                <div class="section">
                  <div class="sectionHead">
                    <div>
                      <div class="sectionTitle">{binfo["name"]}</div>
                      <div class="sectionDesc">{binfo["desc"]}</div>
                    </div>
                    <div style="text-align:right; display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
                      {pill_html(bst)}
                      <span class="pill">Score: <b>{btxt}</b></span>
                      <span class="pill info">{delta_txt}</span>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Mechanics explainer (keep, but concise)
        with st.expander("Come leggere ‘Risk-on / Neutral / Risk-off’ (crypto)", expanded=False):
            st.markdown(
                """
**Risk-on (crypto):** condizioni più permissive (liquidità/marginal funding migliori), stress premia in calo, conferme BTC/ETH.  
**Neutral:** segnali misti; evitare FOMO; contano convergenze e sizing.  
**Risk-off:** stress o real yields rendono il risk-on fragile; priorità a protezione e optionalità.

**Score (0–100):** z-score su ~5 anni (clamp) mappato in 0–100 con convenzione direzionale per ogni metrica.
                """.strip()
            )

        st.markdown(
            f"""
            <div class="card">
              <div class="cardTitle">Data & display</div>
              <div class="cardSub">
                Now: <b>{now_str}</b><br/>
                Latest datapoint: <b>{('n/a' if data_max_date is None else str(pd.to_datetime(data_max_date).date()))}</b><br/>
                Layout mode: <b>{layout_mode}</b>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ============================================================
    # WALLBOARD (TILES) + GUIDE DROPDOWN ABOVE
    # ============================================================
    with tabs[1]:
        st.markdown("## Wallboard")
        st.markdown("<div class='muted'>Prima: quadro d’insieme. Poi: tiles per indicatori con reference, trend e score.</div>", unsafe_allow_html=True)

        # Guide dropdown ABOVE tiles/cards (as requested)
        with st.expander("📌 Guida indicatori (menu a tendina)", expanded=False):
            all_keys = list(INDICATOR_META.keys())
            pretty = [INDICATOR_META[k]["label"] for k in all_keys]
            sel = st.selectbox("Seleziona un indicatore per la guida", options=list(range(len(all_keys))), format_func=lambda i: pretty[i])
            indicator_guide_markdown(all_keys[sel])

        # Overall regime + components
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
                <div class="cardTitle">Block scorecard</div>
                <div class="cardSub">
                  • <b>{BLOCKS["liquidity"]["name"]}:</b> {status_label(block_scores["liquidity"]["status"])}<br/>
                  • <b>{BLOCKS["real_cost"]["name"]}:</b> {status_label(block_scores["real_cost"]["status"])}<br/>
                  • <b>{BLOCKS["stress"]["name"]}:</b> {status_label(block_scores["stress"]["status"])}<br/>
                  • <b>{BLOCKS["crypto_confirm"]["name"]}:</b> {status_label(block_scores["crypto_confirm"]["status"])}
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

        def render_group(block_key: str):
            binfo = BLOCKS[block_key]
            st.markdown(
                f"""
                <div class="section">
                  <div class="sectionHead">
                    <div>
                      <div class="sectionTitle">{binfo["name"]}</div>
                      <div class="sectionDesc">{binfo["desc"]}</div>
                    </div>
                  </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("<div class='wbGrid'>", unsafe_allow_html=True)

            for k in binfo["indicators"]:
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
                              No data available for this indicator in the selected history window.
                            </div>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    wallboard_tile(k, s, indicator_scores)

            st.markdown("</div></div>", unsafe_allow_html=True)

            # Keep an expander per group with crypto-tailored guidance (optional)
            with st.expander(f"Group notes — {binfo['name']}", expanded=False):
                st.markdown(binfo["desc"])
                st.markdown("**Indicatori inclusi:** " + ", ".join([INDICATOR_META[k]["label"] for k in binfo["indicators"]]))

        render_group("liquidity")
        render_group("real_cost")
        render_group("stress")
        render_group("crypto_confirm")

    # ============================================================
    # DEEP DIVE (SCROLLABLE, MULTI-SECTIONS OPEN)
    # ============================================================
    with tabs[2]:
        st.markdown("## Deep dive (scrollable)")
        st.markdown("<div class='muted'>Sezioni espandibili: puoi aprire più categorie insieme e scrollare.</div>", unsafe_allow_html=True)

        # Guide dropdown ABOVE deep dive too
        with st.expander("📌 Guida indicatori (menu a tendina)", expanded=False):
            all_keys = list(INDICATOR_META.keys())
            pretty = [INDICATOR_META[k]["label"] for k in all_keys]
            sel = st.selectbox("Seleziona un indicatore per la guida", options=list(range(len(all_keys))), format_func=lambda i: pretty[i], key="deep_guide_select")
            indicator_guide_markdown(all_keys[sel])

        def deep_section(block_key: str, expanded: bool = False):
            binfo = BLOCKS[block_key]
            with st.expander(f"{binfo['name']} — {binfo['desc']}", expanded=expanded):
                for k in binfo["indicators"]:
                    meta = INDICATOR_META[k]
                    s = indicators.get(k, pd.Series(dtype=float))
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
                        <div class="section">
                          <div class="sectionHead">
                            <div>
                              <div class="sectionTitle">{meta["label"]}</div>
                              <div class="sectionDesc">{meta["source"]}</div>
                            </div>
                            <div style="text-align:right;">
                              <div style="display:flex; gap:10px; justify-content:flex-end; flex-wrap:wrap;">
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
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=f"deep_plot_{k}")

                    with st.expander("Indicator guide (definition, thresholds, crypto so-what)", expanded=False):
                        indicator_guide_markdown(k)

                    st.markdown("</div>", unsafe_allow_html=True)

        # default expanded first two on mobile-friendly flow
        deep_section("liquidity", expanded=True if layout_mode != "Deep dive first" else True)
        deep_section("real_cost", expanded=True if layout_mode != "Auto" else False)
        deep_section("stress", expanded=False)
        deep_section("crypto_confirm", expanded=False)

    # ============================================================
    # WHAT CHANGED (CLEAN, NO DELTAGENERATOR LEAK)
    # ============================================================
    with tabs[3]:
        st.markdown("## What changed")
        st.markdown(
            "<div class='muted'>Tabella sinottica: trend su orizzonti fissi + score/status. "
            "Se hai caricato un report precedente, include anche Δ score vs prev.</div>",
            unsafe_allow_html=True
        )

        rows = []
        for key, meta in INDICATOR_META.items():
            s = indicators.get(key, pd.Series(dtype=float))
            if s is None or s.empty:
                continue

            d7 = pct_change_over_days(s, 7)
            d30 = pct_change_over_days(s, 30)
            d90 = pct_change_over_days(s, 90)
            d180 = pct_change_over_days(s, 180)
            d365 = pct_change_over_days(s, 365)

            sc = indicator_scores.get(key, {})
            score = sc.get("score", np.nan)
            status = sc.get("status", "n/a")

            ds = None
            if prev_payload:
                ds = deltas.get("indicators", {}).get(key, {}).get("score_delta", None)

            rows.append({
                "Group": meta.get("group", ""),
                "Indicator": meta["label"],
                "Regime": status_label(status),
                "Score": (np.nan if np.isnan(score) else round(score, 1)),
                "Δ Score vs prev": (np.nan if ds is None else ds),
                "Δ 7d %": (np.nan if np.isnan(d7) else round(d7, 2)),
                "Δ 1M %": (np.nan if np.isnan(d30) else round(d30, 2)),
                "Δ 3M %": (np.nan if np.isnan(d90) else round(d90, 2)),
                "Δ 6M %": (np.nan if np.isnan(d180) else round(d180, 2)),
                "Δ 1Y %": (np.nan if np.isnan(d365) else round(d365, 2)),
            })

        if not rows:
            st.info("Dati insufficienti per calcolare le variazioni.")
        else:
            df_wc = pd.DataFrame(rows)

            # Optional: highlight movers / significant changes
            df_wc["Abs Δ 1M"] = df_wc["Δ 1M %"].abs()
            df_wc["Abs Δ Score vs prev"] = df_wc["Δ Score vs prev"].abs() if prev_payload else np.nan

            # Sort by significance: score delta if exists, else 1M move
            if prev_payload:
                df_wc = df_wc.sort_values(["Abs Δ Score vs prev", "Abs Δ 1M"], ascending=[False, False])
            else:
                df_wc = df_wc.sort_values(["Abs Δ 1M"], ascending=False)

            # Show top movers summary
            st.markdown("### Top movers / regime-relevant (heuristic)")
            top = df_wc.head(8)
            for _, r in top.iterrows():
                ds_txt = "" if (not prev_payload or pd.isna(r["Δ Score vs prev"])) else f" · ΔScore {r['Δ Score vs prev']:+.2f}"
                st.markdown(
                    f"<div class='card' style='margin-bottom:10px;'>"
                    f"<div class='cardTitle'>{r['Group']}</div>"
                    f"<div class='cardSub'><b>{r['Indicator']}</b> — {r['Regime']} · Score {r['Score']}{ds_txt} · Δ1M {r['Δ 1M %']:+.2f}%</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            st.markdown("### Full table")
            show = df_wc.drop(columns=["Abs Δ 1M", "Abs Δ Score vs prev"]).reset_index(drop=True)
            st.dataframe(show, use_container_width=True, hide_index=True)

            st.caption(
                "Nota: le % sono calcolate sulla osservazione più vicina disponibile (frequenze diverse). "
                "Usa Wallboard/Deep dive per reference levels e lettura qualitativa."
            )

    # ============================================================
    # ALERTS (CRYPTO-SPECIFIC, PROFESSIONAL)
    # ============================================================
    with tabs[4]:
        st.markdown("## Alerts (crypto regime)")
        st.markdown("<div class='muted'>Alert euristici (non segnali di trading): servono a identificare cambi di regime.</div>", unsafe_allow_html=True)

        alerts = []

        # Helper to get last
        def last_val(key):
            s = indicators.get(key, pd.Series(dtype=float))
            if s is None or s.empty:
                return np.nan
            return float(s.dropna().iloc[-1])

        real_last = last_val("real_10y")
        vix_last = last_val("vix")
        hy_last = last_val("hy_oas")

        # Macro triggers
        if not np.isnan(real_last) and real_last >= thr_real_restrictive:
            alerts.append(("Headwind", "Real yield restrittivo", f"Real 10Y {real_last:.2f}% ≥ {thr_real_restrictive:.2f}%. Crypto compete con risk-free reale.", "risk_off"))
        if not np.isnan(vix_last) and vix_last >= thr_vix_stress:
            alerts.append(("Headwind", "Vol in stress", f"VIX {vix_last:.1f} ≥ {thr_vix_stress:.1f}. Risk aversion elevata.", "risk_off"))
        if not np.isnan(hy_last) and hy_last >= thr_hy_stress:
            alerts.append(("Headwind", "Credito in stress", f"HY OAS {hy_last:.2f}pp ≥ {thr_hy_stress:.2f}pp. Deleveraging risk.", "risk_off"))

        # Liquidity improvements
        bs = indicators.get("fed_balance_sheet", pd.Series(dtype=float))
        rrp = indicators.get("rrp", pd.Series(dtype=float))
        bs30 = pct_change_over_days(bs, 30)
        rrp30 = pct_change_over_days(rrp, 30)
        if not np.isnan(bs30) and bs30 >= 0:
            alerts.append(("Tailwind", "Fed BS non drena", f"Fed BS 30d: {bs30:+.2f}% (stabile/in salita) → minore drenaggio (QT).", "risk_on"))
        if not np.isnan(rrp30) and rrp30 <= -15:
            alerts.append(("Tailwind", "RRP in rilascio", f"RRP 30d: {rrp30:+.1f}% (calo forte) → possibile rilascio liquidità marginale.", "risk_on"))

        # Crypto confirmations
        btcqqq = indicators.get("btc_qqq", pd.Series(dtype=float))
        ethbtc = indicators.get("eth_btc", pd.Series(dtype=float))
        rs30 = pct_change_over_days(btcqqq, 30)
        eb30 = pct_change_over_days(ethbtc, 30)
        if not np.isnan(rs30) and rs30 > 2:
            alerts.append(("Confirm", "BTC leadership", f"BTC/QQQ 30d: {rs30:+.2f}% → BTC sovraperforma Nasdaq (conferma).", "risk_on"))
        if not np.isnan(eb30) and eb30 > 2:
            alerts.append(("Confirm", "ETH breadth", f"ETH/BTC 30d: {eb30:+.2f}% → breadth risk-on crypto migliore.", "risk_on"))

        if not alerts:
            st.info("Nessun alert dominante con le soglie attuali (o dati insufficienti).")
        else:
            # Group by type
            df_alerts = pd.DataFrame(alerts, columns=["Type", "Title", "Detail", "Tone"])
            for tname in ["Headwind", "Tailwind", "Confirm"]:
                part = df_alerts[df_alerts["Type"] == tname]
                if part.empty:
                    continue
                st.markdown(f"### {tname}")
                for _, r in part.iterrows():
                    pill = pill_html("risk_off") if r["Tone"] == "risk_off" else pill_html("risk_on")
                    st.markdown(
                        f"""
                        <div class="card" style="margin-bottom:10px;">
                          <div class="cardTitle">{r["Title"]} {pill}</div>
                          <div class="cardSub">{r["Detail"]}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            st.caption("Suggerimento: quando 2–3 alert ‘tailwind/confirm’ si accendono insieme, il regime tende a cambiare marcia (euristica).")

    # ============================================================
    # REPORT GENERATION (PROMPT + CURRENT PAYLOAD + OPTIONAL PREV)
    # ============================================================
    with tabs[5]:
        st.markdown("## Report generation (AI)")
        st.markdown(
            "<div class='muted'>Output pulito: prompt + payload YAML. "
            "Se hai caricato un payload precedente, include anche <b>previous_payload</b> per confronto nel tempo.</div>",
            unsafe_allow_html=True
        )

        if not HAS_YAML:
            st.warning("⚠️ PyYAML non disponibile nell'ambiente. Il download YAML potrebbe non funzionare. (Streamlit Cloud di solito lo include.)")

        # Show comparison summary if prev exists
        if prev_payload:
            gsd = deltas["regime"].get("global_score_delta", None)
            gsch = deltas["regime"].get("global_status_change", None)
            st.markdown(
                f"""
                <div class="card">
                  <div class="cardTitle">Comparison vs previous payload</div>
                  <div class="cardSub">
                    Global score Δ: <b>{("n/a" if gsd is None else f"{gsd:+.2f}")}</b><br/>
                    Global status changed: <b>{("n/a" if gsch is None else str(gsch))}</b>
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Build YAML texts
        def dump_yaml(obj: dict) -> str:
            if HAS_YAML:
                return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True)
            # fallback
            return str(obj)

        current_yaml = dump_yaml({"current_payload": current_payload})
        prev_yaml = dump_yaml({"previous_payload": prev_payload}) if prev_payload else ""

        one_shot = (
            "### COPY/PASTE BELOW (PROMPT + PAYLOADS)\n\n"
            + REPORT_PROMPT
            + "\n\n---\n\n"
            + "PAYLOADS:\n\n```yaml\n"
            + current_yaml.strip()
            + ("\n\n" + prev_yaml.strip() if prev_payload else "")
            + "\n```\n"
        )

        st.code(one_shot, language="markdown")

        # Download buttons (current + optional prev)
        st.download_button(
            "Download CURRENT payload YAML",
            current_yaml,
            file_name=f"crypto_macro_payload_{datetime.now(TZ).date().isoformat()}.yaml",
            mime="text/yaml"
        )
        if prev_payload:
            st.download_button(
                "Download PREVIOUS payload YAML (as imported)",
                prev_yaml,
                file_name=f"crypto_macro_payload_previous_{datetime.now(TZ).date().isoformat()}.yaml",
                mime="text/yaml"
            )

        st.caption("Workflow: salva il payload ogni volta (es. weekly). Al report successivo, ricarica il YAML precedente per avere confronto automatico.")

if __name__ == "__main__":
    main()
