import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from pandas.tseries.offsets import DateOffset

# ============================================================
# PAGE CONFIG (mobile-first: readable on mobile + desktop)
# ============================================================
st.set_page_config(
    page_title="Global Markets Radar",
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
    --card2:#0c1324;
    --border:rgba(255,255,255,0.10);
    --muted:rgba(255,255,255,0.70);
    --text:rgba(255,255,255,0.94);

    --good:rgba(34,197,94,1);
    --warn:rgba(245,158,11,1);
    --bad:rgba(239,68,68,1);

    --accent:rgba(148,163,184,1);         /* sober slate */
    --accentSoft:rgba(148,163,184,0.16);
    --accent2:rgba(99,102,241,1);         /* subtle indigo for highlights */
    --accent2Soft:rgba(99,102,241,0.14);
  }

  .stApp {
    background: radial-gradient(1100px 650px at 20% 0%, #121a33 0%, #0b0f19 45%, #0b0f19 100%);
    color: var(--text);
  }

  /* MOBILE-FIRST container: constrain width for readability */
  .block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1180px;
  }

  h1, h2, h3, h4 { color: var(--text); letter-spacing: -0.02em; }
  .muted { color: var(--muted); }

  /* Tabs: readable + sober selected */
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

  /* Expander: consistent */
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

  /* Responsive grids: mobile-first */
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

  /* Section wrapper */
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

  /* Dataframe tweaks */
  .stDataFrame { border: 1px solid var(--border); border-radius: 12px; overflow:hidden; }
  code { color: rgba(255,255,255,0.88); }

  /* Small helper */
  .kpiRow{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# INDICATORS & BLOCKS
# ============================================================

INDICATOR_META = {
    # 1) PRICE OF TIME
    "real_10y": {
        "label": "US 10Y TIPS Real Yield",
        "unit": "%",
        "direction": -1,
        "source": "FRED DFII10",
        "scale": 1.0,
        "ref_line": 2.0,  # practical "restrictive-ish"
        "scoring_mode": "z5y",
        "expander": {
            "what": "Real yield: il prezzo reale del denaro (discount rate reale).",
            "reference": "<0% molto easy; 0–2% neutro; >2% restrittivo (euristiche).",
            "interpretation": "- Real yield ↑: condizioni si stringono; soffrono duration lunga e equity growth.\n- Real yield ↓: allenta il vincolo; migliora la qualità del risk-on (se stress non esplode).",
            "bridge": "Quando il rendimento reale risk-free è alto, il sistema ‘non ha bisogno’ di rischio per ottenere rendimento.",
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
            "what": "Rendimento nominale 10Y: benchmark di sconto e proxy tightening.",
            "reference": "Conta soprattutto la velocità: upside rapida = tightening (euristica).",
            "interpretation": "- Yield ↑ veloce: pressione su equity e su bond già in portafoglio.\n- Yield ↓: supporto a duration; su equity dipende se è ‘good’ (disinflation) o ‘bad’ (growth scare).",
            "bridge": "Muove la sensibilità dei multipli e la dinamica di carry/roll sui bond.",
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
        "expander": {
            "what": "Pendenza curva 10–2: proxy ciclo / rischio recessivo.",
            "reference": "<0 inversione (late cycle); >0 normalizzazione (euristica).",
            "interpretation": "- Inversione profonda e persistente = warning.\n- Dis-inversione: bene se guidata da tagli (non da shock di crescita).",
            "bridge": "Curva invertita = policy stretta vs ciclo → aumenta probabilità di deleveraging.",
        },
    },

    # 2) MACRO CYCLE
    "breakeven_10y": {
        "label": "10Y Breakeven Inflation",
        "unit": "%",
        "direction": -1,
        "source": "FRED T10YIE",
        "scale": 1.0,
        "ref_line": 3.0,
        "scoring_mode": "z5y",
        "expander": {
            "what": "Inflazione attesa implicita dal mercato (10Y).",
            "reference": "~2–3% ancorata; >3% = rischio sticky (euristica).",
            "interpretation": "- Breakeven ↑: meno spazio per easing, term premium può salire.\n- Breakeven ↓/ancorata: supporta duration e risk budgeting.",
            "bridge": "Se l’inflazione attesa si disancora, la policy diventa più vincolata.",
        },
    },
    "cpi_yoy": {
        "label": "US CPI YoY",
        "unit": "%",
        "direction": -1,
        "source": "FRED CPIAUCSL (YoY)",
        "scale": 1.0,
        "ref_line": 3.0,
        "scoring_mode": "z5y",
        "expander": {
            "what": "Inflazione headline YoY (proxy).",
            "reference":  "2% target; >3–4% persistente = sticky risk (euristica).",
            "interpretation": "- Disinflazione: favorisce duration e spesso equity.\n- Riaccelerazione: aumenta rischio 'higher for longer'.",
            "bridge": "Inflazione persistente è il vincolo dominante per la politica monetaria.",
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
        "expander": {
            "what": "Proxy slack nel mercato del lavoro.",
            "reference": "Aumenti rapidi = segnali di rallentamento (euristica).",
            "interpretation": "- Unemployment ↑ rapido: spesso risk-off (growth scare).\n- Stabilità: benigno.",
            "bridge": "Slack + debito alto aumenta pressione per supporto policy/fiscale.",
        },
    },

    # 3) CONDITIONS & STRESS
    "usd_index": {
        "label": "USD Strength (DXY / Broad)",
        "unit": "",
        "direction": -1,
        "source": "yfinance DX-Y.NYB (fallback FRED DTWEXBGS)",
        "scale": 1.0,
        "ref_line": None,
        "scoring_mode": "z5y",
        "expander": {
            "what": "Forza USD: filtro condizioni finanziarie globali.",
            "reference": "USD ↑ = tightening globale (euristica).",
            "interpretation": "- USD ↑: stress funding globale, spesso headwind per risk.\n- USD ↓: allenta condizioni e supporta risk-on.",
            "bridge": "Molte passività globali sono USD-linked: USD forte = vincolo più duro.",
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
            "what": "Spread HY: stress credito / premio default.",
            "reference": "<4% benigno; 4–6% warning; >6–7% stress (euristica).",
            "interpretation": "- Spread ↑: risk-off (default premium e funding stress).\n- Spread ↓: risk appetite e carry più affidabile.",
            "bridge": "Credito è spesso la ‘cinghia’ che trasmette stress all’equity.",
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
        "expander": {
            "what": "Vol implicita equity USA (S&P).",
            "reference": "<15 low; 15–25 normale; >25 stress (euristica).",
            "interpretation": "- VIX ↑: premi al rischio salgono → tightening.\n- VIX ↓: facilita risk-taking.",
            "bridge": "La vol può stringere condizioni anche senza rialzi tassi.",
        },
    },
    "spy_trend": {
        "label": "Equity Trend (SPY / 200D MA)",
        "unit": "ratio",
        "direction": +1,
        "source": "yfinance SPY",
        "scale": 1.0,
        "ref_line": 1.0,
        "scoring_mode": "z5y",
        "expander": {
            "what": "Trend proxy: SPY sopra/sotto la media 200 giorni.",
            "reference": ">1 uptrend; <1 downtrend (euristica).",
            "interpretation": "- Sopra 1: momentum e risk appetite più solidi.\n- Sotto 1: regime più difensivo.",
            "bridge": "Trend down + credito che si allarga spesso segnala deleveraging.",
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
        "expander": {
            "what": "HY vs IG: appetito per rischio credito.",
            "reference": "Ratio ↑ = più appetite HY; ↓ = flight to quality.",
            "interpretation": "- Ratio ↑: supporta risk-on.\n- Ratio ↓: qualità/bilanci solidi preferiti.",
            "bridge": "È un termometro veloce dell’asset allocation del credito.",
        },
    },

    # 4) LIQUIDITY / PLUMBING
    "fed_balance_sheet": {
        "label": "Fed Balance Sheet (WALCL)",
        "unit": "bn USD",
        "direction": +1,
        "source": "FRED WALCL (millions → bn)",
        "scale": 1.0 / 1000.0,
        "ref_line": None,
        "scoring_mode": "z5y",
        "expander": {
            "what": "Asset FED: proxy di QE/QT e liquidità di sistema.",
            "reference": "Espansione = tailwind; contrazione = headwind (euristica).",
            "interpretation": "- BS ↑: spesso facilita risk.\n- BS ↓: drena liquidità marginale.",
            "bridge": "La ‘plumbing’ determina se i flussi alimentano o drenano risk assets.",
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
            "what": "Cash parcheggiato nel facility: liquidità non impiegata in risk.",
            "reference": "RRP alto = liquidità ‘in panchina’; calo può rilasciare margine (euristica).",
            "interpretation": "- RRP ↑: meno marginal liquidity.\n- RRP ↓: possibile tailwind tattico.",
            "bridge": "Agisce spesso come valvola tattica di liquidità.",
        },
    },

    # 5) STRUCTURAL: DEBT & FISCAL / POLICY LINK
    "interest_payments": {
        "label": "US Federal Interest Payments (Quarterly)",
        "unit": "bn USD",
        "direction": -1,
        "source": "FRED A091RC1Q027SBEA",
        "scale": 1.0,
        "ref_line": None,
        "scoring_mode": "pct20y",
        "expander": {
            "what": "Interessi pagati dal governo: pressione di debt-service.",
            "reference": "Accelerazione = vincolo policy più duro (euristica).",
            "interpretation": "- In aumento persistente: riduce flessibilità.\n- Stabilizzazione: vincolo meno stringente.",
            "bridge": "Debt-service alto aumenta incentivi a policy funding-friendly.",
        },
    },
    "federal_receipts": {
        "label": "US Federal Current Receipts (Quarterly)",
        "unit": "bn USD",
        "direction": +1,
        "source": "FRED FGRECPT",
        "scale": 1.0,
        "ref_line": None,
        "scoring_mode": "pct20y",
        "expander": {
            "what": "Entrate federali: supporto alla capacità di servizio del debito.",
            "reference": "Serve anche per il rapporto interest/receipts.",
            "interpretation": "- Entrate ↑: migliora sostenibilità (ceteris paribus).\n- Entrate ↓: aumenta vincolo.",
            "bridge": "Minore capacità di entrate rende più sensibile il funding.",
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
        "expander": {
            "what": "Quota delle entrate assorbita dagli interessi: proxy vincolo fiscale.",
            "reference": "Alto e in salita = vincolo politico crescente (euristica).",
            "interpretation": "- Ratio ↑: meno spazio per manovra.\n- Ratio ↓: più margine.",
            "bridge": "Se il vincolo fiscale cresce, aumenta probabilità di policy accomodante nel tempo.",
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
        "expander": {
            "what": "Saldo fiscale (% PIL). Negativo = deficit.",
            "reference": "Deficit ampi persistenti aumentano supply pressure (euristica).",
            "interpretation": "- Più negativo: più supply → rischio term premium.\n- Meno negativo: allevia pressione.",
            "bridge": "Supply pressure può ridurre qualità hedge dei bond nominali.",
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
        "expander": {
            "what": "Term premium: compenso per detenere duration nominale.",
            "reference": "Se sale, i long bond possono hedgiare meno (euristica).",
            "interpretation": "- Term premium ↑: duration più rischiosa.\n- Term premium ↓: hedge quality migliora.",
            "bridge": "Se sale per supply/funding, bond possono non proteggere equity drawdown.",
        },
    },

    # 6) STRUCTURAL: EXTERNAL BALANCE
    "current_account_gdp": {
        "label": "US Current Account Balance (% of GDP)",
        "unit": "%",
        "direction": +1,
        "source": "FRED USAB6BLTT02STSAQ",
        "scale": 1.0,
        "ref_line": 0.0,
        "scoring_mode": "pct20y",
        "expander": {
            "what": "Vincolo di funding esterno: deficit = dipendenza da capitali esteri.",
            "reference": "Più negativo = maggiore vulnerabilità con USD forte (euristica).",
            "interpretation": "- Più negativo: vulnerabilità in tightening USD.\n- Verso 0: meno vincolo.",
            "bridge": "Deficit esterni aumentano rischio quando condizioni USD stringono.",
        },
    },

    # 7) GOLD
    "gold": {
        "label": "Gold (GLD)",
        "unit": "",
        "direction": -1,
        "source": "yfinance GLD",
        "scale": 1.0,
        "ref_line": None,
        "scoring_mode": "z5y",
        "expander": {
            "what": "Oro: domanda di hedge (policy/inflazione/tail-risk).",
            "reference": "Rotture al rialzo spesso = hedge demand, non growth optimism (euristica).",
            "interpretation": "- Gold ↑: hedge demand / real returns compressi.\n- Gold ↓ in bull equity: risk-on ‘pulito’.",
            "bridge": "Aiuta quando il sistema cerca protezione da policy/funding risk.",
        },
    },
}

BLOCKS = {
    "price_of_time": {
        "name": "1) Price of Time",
        "weight": 0.20,
        "indicators": ["real_10y", "nominal_10y", "yield_curve_10_2"],
        "desc": "Tassi e curva: costo del capitale e segnali late-cycle.",
        "group": "Market Thermometers",
    },
    "macro": {
        "name": "2) Macro Cycle",
        "weight": 0.15,
        "indicators": ["breakeven_10y", "cpi_yoy", "unemployment_rate"],
        "desc": "Inflazione e crescita: vincolo alla reaction function.",
        "group": "Market Thermometers",
    },
    "conditions": {
        "name": "3) Conditions & Stress",
        "weight": 0.22,
        "indicators": ["usd_index", "hy_oas", "vix", "spy_trend", "hyg_lqd_ratio"],
        "desc": "Regime veloce: USD, credito, vol, trend e risk appetite.",
        "group": "Market Thermometers",
    },
    "plumbing": {
        "name": "4) Liquidity / Plumbing",
        "weight": 0.13,
        "indicators": ["fed_balance_sheet", "rrp"],
        "desc": "Liquidità di sistema: tailwind vs drain su risk assets.",
        "group": "Market Thermometers",
    },
    "policy_link": {
        "name": "5) Fiscal / Policy Constraint",
        "weight": 0.20,
        "indicators": ["interest_to_receipts", "deficit_gdp", "term_premium_10y", "interest_payments", "federal_receipts"],
        "desc": "Debt service, deficit e term premium: vincolo funding/policy.",
        "group": "Structural Constraints",
    },
    "external": {
        "name": "6) External Balance",
        "weight": 0.10,
        "indicators": ["current_account_gdp"],
        "desc": "Dipendenza dal funding estero: vulnerabilità in USD tightening.",
        "group": "Structural Constraints",
    },
    "gold_block": {
        "name": "7) Gold (hedge confirmation)",
        "weight": 0.00,
        "indicators": ["gold"],
        "desc": "Conferma hedge demand / policy risk (informativo).",
        "group": "Structural Constraints",
    },
}

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
        col = "Close"
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
# SCORING (z5y vs pct20y)
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
# PLOTTING (dark-friendly, minimal modebar)
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
# OPERATING LINES (ETF oriented, clearer + rationale)
# ============================================================

def operating_lines(block_scores: dict, indicator_scores: dict):
    gs = block_scores.get("GLOBAL", {}).get("score", np.nan)

    def _sg(x):
        if np.isnan(x):
            return np.nan
        return float(x)

    cond = _sg(block_scores.get("conditions", {}).get("score", np.nan))
    pot = _sg(block_scores.get("price_of_time", {}).get("score", np.nan))
    macro = _sg(block_scores.get("macro", {}).get("score", np.nan))
    policy = _sg(block_scores.get("policy_link", {}).get("score", np.nan))

    # Helpers
    def band(x):
        if np.isnan(x):
            return "n/a"
        if x >= 60: return "supportive"
        if x <= 40: return "adverse"
        return "mixed"

    # Equity exposure
    if np.isnan(gs):
        equity = "n/a"
        equity_why = "Dati insufficienti."
    else:
        if gs >= 65 and (not np.isnan(cond) and cond >= 55):
            equity = "Increase (measured)"
            equity_why = "Regime complessivamente supportive; stress/credito non stanno segnalando tightening marcato."
        elif gs <= 40 or (not np.isnan(cond) and cond <= 40):
            equity = "Reduce / de-risk"
            equity_why = "Stress/condizioni avverse: priorità a proteggere downside."
        else:
            equity = "Neutral / selective"
            equity_why = "Segnali misti: sizing disciplinato e preferenza per qualità."

    # Duration
    termp = _sg(indicator_scores.get("term_premium_10y", {}).get("score", np.nan))
    infl = _sg(indicator_scores.get("cpi_yoy", {}).get("score", np.nan))

    if (not np.isnan(termp) and termp <= 40) and (not np.isnan(infl) and infl <= 45):
        duration = "Short/neutral (be cautious on long nominals)"
        duration_why = "Term premium e/o inflazione non abbastanza benigni: hedge quality dei long bond può essere instabile."
    elif (not np.isnan(pot) and pot <= 40) and (not np.isnan(infl) and infl <= 45) and (not np.isnan(termp) and termp >= 55):
        duration = "Long (as hedge)"
        duration_why = "Disinflazione + costo del capitale in miglioramento: duration torna hedge più pulito."
    else:
        duration = "Neutral / barbell"
        duration_why = "Bilanciare rischio term-premium vs protezione in caso di growth scare."

    # Credit
    hy = _sg(indicator_scores.get("hy_oas", {}).get("score", np.nan))
    hyg = _sg(indicator_scores.get("hyg_lqd_ratio", {}).get("score", np.nan))
    ds = _sg(indicator_scores.get("interest_to_receipts", {}).get("score", np.nan))

    if (not np.isnan(hy) and hy <= 40) or (not np.isnan(hyg) and hyg <= 40) or (not np.isnan(ds) and ds <= 40):
        credit = "IG > HY (reduce default/funding risk)"
        credit_why = "Segnali di stress o flight-to-quality: evitare beta credito."
    elif (not np.isnan(hy) and hy >= 60) and (not np.isnan(hyg) and hyg >= 60) and (np.isnan(policy) or policy >= 50):
        credit = "Opportunistic HY (size disciplined)"
        credit_why = "Carry appetibile e risk appetite presente; usare sizing e stop di regime."
    else:
        credit = "Neutral (quality tilt)"
        credit_why = "Segnali misti: preferire qualità e selettività."

    # Hedges
    usd = _sg(indicator_scores.get("usd_index", {}).get("score", np.nan))
    gold = _sg(indicator_scores.get("gold", {}).get("score", np.nan))

    if (not np.isnan(policy) and policy <= 40) and (np.isnan(macro) or macro <= 55):
        hedges = "Gold / real-asset sleeve"
        hedges_why = "Vincolo fiscale/policy più duro: hedge demand può aumentare."
    elif (not np.isnan(usd) and usd <= 40) and (not np.isnan(cond) and cond <= 45):
        hedges = "USD / cash-like"
        hedges_why = "Funding stress: preferire liquidità e difesa."
    elif (not np.isnan(gold) and gold <= 40):
        hedges = "Keep small gold sleeve"
        hedges_why = "Hedge demand in aumento: utile opzionalità."
    else:
        hedges = "Light hedges (balanced)"
        hedges_why = "Nessun segnale dominante: hedges leggeri e tattici."

    return {
        "Equity": {"stance": equity, "why": equity_why, "context": f"GLOBAL={band(gs)}, CONDITIONS={band(cond)}"},
        "Duration": {"stance": duration, "why": duration_why, "context": f"PRICE_OF_TIME={band(pot)}, CPI={band(infl)}, TERM_PREM={band(termp)}"},
        "Credit": {"stance": credit, "why": credit_why, "context": f"HY_OAS={band(hy)}, HYG/LQD={band(hyg)}"},
        "Hedges": {"stance": hedges, "why": hedges_why, "context": f"USD={band(usd)}, POLICY={band(policy)}, GOLD={band(gold)}"},
    }

# ============================================================
# ALERTS (threshold-based) + trigger detection
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
    idx = c.index[turned.values]  # safe boolean mask
    return list(pd.to_datetime(idx))

def last_n_triggers(cond: pd.Series, since: pd.Timestamp) -> list[str]:
    dates = [d for d in turned_true_dates(cond) if d >= since]
    return [pd.to_datetime(d).date().isoformat() for d in dates[-6:]]

def so_what_line(key: str, dwin: float, score: float, status: str) -> str:
    # Deterministic, professional “so what” line (no AI)
    # Uses direction convention + whether move is meaningful
    meta = INDICATOR_META[key]
    label = meta["label"]
    direction = meta["direction"]
    move = "" if np.isnan(dwin) else ("up" if dwin > 0 else "down" if dwin < 0 else "flat")
    material = (not np.isnan(dwin)) and (abs(dwin) >= 3.0)  # heuristic
    stance = status_label(status)

    # Map to implication: if direction = -1, "up" is worse; if +1, "up" is better
    if np.isnan(dwin):
        return f"{label}: movimento recente non quantificabile (serie/frequenza)."
    if direction == -1:
        if dwin > 0:
            return f"{label} ↑: tende a stringere condizioni / aumentare vincoli (watch se persiste)."
        if dwin < 0:
            return f"{label} ↓: tende ad allentare condizioni / ridurre vincoli (supportivo se confermato)."
        return f"{label} ~: nessun impulso evidente dal movimento recente."
    else:
        if dwin > 0:
            return f"{label} ↑: conferma risk appetite / normalizzazione (supportivo se stabile)."
        if dwin < 0:
            return f"{label} ↓: segnala deterioramento appetite/condizioni (cautela se accelera)."
        return f"{label} ~: nessun impulso evidente dal movimento recente."

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
        st.markdown("**How to read it (bi-directional):**")
        st.markdown(exp.get("interpretation", ""))
        st.markdown(f"**Why it matters (funding/policy link):** {exp.get('bridge','')}")

# ============================================================
# REPORT PROMPT (PM/CIO style) — shorter but strict + actionable
# ============================================================

REPORT_PROMPT = """SYSTEM / ROLE
You are a senior multi-asset macro strategist writing an internal PM/CIO note.
You diagnose regime from *behavioral pricing + constraints* (no forecasts).

You receive a YAML payload with:
- global score + block scores
- indicator scores, latest values, and recent changes
- active alerts + recent triggers
- ETF-oriented operating lines

RULES
- Use ONLY data in payload (no speculation, no new indicators).
- Be concise, professional, implementation-oriented.
- Separate: Market Thermometers vs Structural Constraints.
- Always include “So what” implications (Equity / Duration / Credit / Hedges).
- Highlight what changed on 1M + 30d/7d.

OUTPUT STRUCTURE (follow exactly)
# Global Markets Regime Note
[Insert date]

## Executive summary (max 8 lines)
## What changed (1M focus, then 30d/7d)
## Market Thermometers (fast)
1) Price of Time
2) Macro Cycle
3) Conditions & Stress
4) Liquidity / Plumbing
## Structural Constraints (slow)
5) Fiscal / Policy Constraint
6) External Balance
7) Gold (informational)
## Active alerts & triggers (2–6 weeks)
## ETF-oriented operating lines
- Equity
- Duration
- Credit
- Hedges
## Bottom line (one paragraph)
""".strip()

# ============================================================
# MAIN
# ============================================================

def main():
    st.title("Global Markets Radar")
    st.markdown(
        "<div class='muted'>Dashboard macro-finanziaria per leggere il regime globale e tradurlo in logica ETF (equity / duration / credit / hedges). "
        "Obiettivo: capire <b>cosa guida</b> e <b>cosa cambia</b>, non ‘prevedere’.</div>",
        unsafe_allow_html=True
    )

    # Sidebar
    st.sidebar.header("Settings")
    if st.sidebar.button("🔄 Refresh data (clear cache)"):
        st.cache_data.clear()
        st.rerun()

    years_back = st.sidebar.slider("History (years)", 5, 30, 15)
    layout_mode = st.sidebar.selectbox("Layout mode", ["Auto", "Wallboard-first", "Deep dive-first"], index=0)

    st.sidebar.divider()
    st.sidebar.subheader("Alert thresholds (heuristics)")
    thr_real_yield = st.sidebar.slider("Real yield (DFII10) restrictive >", 0.0, 4.0, 2.0, 0.1)
    thr_vix = st.sidebar.slider("VIX stress >", 10.0, 50.0, 25.0, 0.5)
    thr_hy = st.sidebar.slider("HY OAS stress >", 3.0, 12.0, 6.0, 0.25)
    dxy_ma_days = st.sidebar.slider("USD trend filter MA (days)", 50, 300, 200, 10)
    spy_trend_thr = st.sidebar.slider("SPY trend threshold (SPY/MA200) <", 0.85, 1.10, 1.00, 0.01)

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

        # Derived: yield curve
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

        # Derived: interest / receipts ratio
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
            ["DX-Y.NYB", "^VIX", "SPY", "HYG", "LQD", "GLD"],
            start_date
        )

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

    # Operating lines
    ops = operating_lines(block_scores, indicator_scores)

    # ============================================================
    # ALERTS (active + recent triggers)
    # ============================================================
    alerts = []

    # Build series needed for alerts
    s_real = indicators.get("real_10y", pd.Series(dtype=float)).dropna()
    s_vix = indicators.get("vix", pd.Series(dtype=float)).dropna()
    s_hy = indicators.get("hy_oas", pd.Series(dtype=float)).dropna()
    s_usd = indicators.get("usd_index", pd.Series(dtype=float)).dropna()
    s_spytrend = indicators.get("spy_trend", pd.Series(dtype=float)).dropna()

    # USD MA filter (for trend)
    usd_below_ma = pd.Series(dtype=bool)
    usd_above_ma = pd.Series(dtype=bool)
    if not s_usd.empty and len(s_usd) > dxy_ma_days + 10:
        usd_ma = s_usd.rolling(dxy_ma_days).mean()
        usd_below_ma = (s_usd < usd_ma).dropna()
        usd_above_ma = (s_usd > usd_ma).dropna()

    # Alert conditions (boolean series)
    cond_real_restrictive = (s_real > thr_real_yield) if not s_real.empty else pd.Series(dtype=bool)
    cond_vix_stress = (s_vix > thr_vix) if not s_vix.empty else pd.Series(dtype=bool)
    cond_hy_stress = (s_hy > thr_hy) if not s_hy.empty else pd.Series(dtype=bool)
    cond_usd_tight = usd_above_ma if len(usd_above_ma) else pd.Series(dtype=bool)
    cond_spy_down = (s_spytrend < spy_trend_thr) if not s_spytrend.empty else pd.Series(dtype=bool)

    def is_active(cond: pd.Series) -> bool:
        c = cond.dropna()
        if c.empty:
            return False
        return bool(c.iloc[-1])

    recent_start = pd.Timestamp(datetime.now(timezone.utc).date() - timedelta(days=30))

    alerts_def = [
        {
            "name": "Real yields restrictive",
            "cond": cond_real_restrictive,
            "severity": "bad",
            "so_what": "Real yields sopra soglia: pressione su equity duration-lunga e su bond in prezzo.",
            "trigger": f"DFII10 > {thr_real_yield:.1f}%",
        },
        {
            "name": "Equity vol stress",
            "cond": cond_vix_stress,
            "severity": "bad",
            "so_what": "Vol alta: premi al rischio ↑ → sizing più difensivo.",
            "trigger": f"VIX > {thr_vix:.1f}",
        },
        {
            "name": "Credit spreads stress",
            "cond": cond_hy_stress,
            "severity": "bad",
            "so_what": "Credito che si allarga: attenzione a HY / beta equity.",
            "trigger": f"HY OAS > {thr_hy:.2f}pp",
        },
        {
            "name": "USD tightening impulse",
            "cond": cond_usd_tight,
            "severity": "warn",
            "so_what": "USD sopra trend: tightening globale (headwind per risk).",
            "trigger": f"USD > MA{dxy_ma_days}",
        },
        {
            "name": "Equity trend breakdown",
            "cond": cond_spy_down,
            "severity": "warn",
            "so_what": "SPY sotto MA200 (ratio sotto soglia): regime più difensivo.",
            "trigger": f"SPY/MA200 < {spy_trend_thr:.2f}",
        },
    ]

    active_alerts = []
    recent_triggers = []

    for a in alerts_def:
        cond = a["cond"]
        if is_active(cond):
            active_alerts.append(a)
        dates = last_n_triggers(cond, recent_start)
        if dates:
            recent_triggers.append({"Alert": a["name"], "Trigger": a["trigger"], "Dates (last 30d)": ", ".join(dates)})

    # ============================================================
    # TABS
    # ============================================================
    tabs = st.tabs([
        "Overview",
        "Wallboard",
        "Deep dive",
        "What changed",
        "Alerts",
        "Report",
    ])

    # ============================================================
    # OVERVIEW — immediate, professional, mobile-friendly
    # ============================================================
    with tabs[0]:
        st.markdown(
            "<div class='section'>"
            "<div class='sectionHead'>"
            "<div><div class='sectionTitle'>Regime snapshot</div>"
            "<div class='sectionDesc'>Prima: regime complessivo. Seconda riga: componenti. Terza: implicazioni operative ETF.</div></div>"
            "</div>",
            unsafe_allow_html=True
        )

        gs_txt = "n/a" if np.isnan(global_score) else f"{global_score:.1f}"
        st.markdown(
            f"""
            <div class="grid2">
              <div class="card">
                <div class="cardTitle">Global Score (0–100) — core blocks</div>
                <div class="cardValue">{gs_txt}</div>
                <div class="cardSub">{pill_html(global_status)}</div>
                <div class="cardSub">{score_bar_html(global_score)}</div>
                <div class="cardSub">
                  <span class="pill info"><span class="dot" style="background:var(--accent2)"></span>
                    Regime = comportamento prezzi + stress, non forecast
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

        market_blocks = ["price_of_time", "macro", "conditions", "plumbing"]
        structural_blocks = ["policy_link", "external", "gold_block"]

        st.markdown(
            f"""
            <div class="grid2" style="margin-top:12px;">
              <div class="card">
                <div class="cardTitle">Market Thermometers (fast)</div>
                <div class="cardSub">
                  {"<br/>".join([block_line(k) for k in market_blocks])}
                </div>
              </div>
              <div class="card">
                <div class="cardTitle">Structural Constraints (slow)</div>
                <div class="cardSub">
                  {"<br/>".join([block_line(k) for k in structural_blocks])}
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Operating lines (clear + rationale)
        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        st.markdown("<div class='sectionTitle'>ETF operating lines</div>", unsafe_allow_html=True)
        st.markdown("<div class='sectionDesc'>Sintesi operativa: cosa fare con equity/duration/credit/hedges dato il regime.</div>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="grid2" style="margin-top:10px;">
              <div class="card">
                <div class="cardTitle">Equity</div>
                <div class="cardValue">{ops["Equity"]["stance"]}</div>
                <div class="cardSub">{ops["Equity"]["why"]}<br/><span class="muted">{ops["Equity"]["context"]}</span></div>
              </div>
              <div class="card">
                <div class="cardTitle">Duration (bonds)</div>
                <div class="cardValue">{ops["Duration"]["stance"]}</div>
                <div class="cardSub">{ops["Duration"]["why"]}<br/><span class="muted">{ops["Duration"]["context"]}</span></div>
              </div>
              <div class="card">
                <div class="cardTitle">Credit</div>
                <div class="cardValue">{ops["Credit"]["stance"]}</div>
                <div class="cardSub">{ops["Credit"]["why"]}<br/><span class="muted">{ops["Credit"]["context"]}</span></div>
              </div>
              <div class="card">
                <div class="cardTitle">Hedges</div>
                <div class="cardValue">{ops["Hedges"]["stance"]}</div>
                <div class="cardSub">{ops["Hedges"]["why"]}<br/><span class="muted">{ops["Hedges"]["context"]}</span></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        with st.expander("How to read Risk-on / Neutral / Risk-off (clear + practical)", expanded=False):
            st.markdown(
                """
**Risk-on:** stress premia scendono, credito regge, trend equity è costruttivo, USD non stringe troppo.  
**Neutral:** segnali misti → sizing disciplina > convinzione direzionale.  
**Risk-off:** stress/tightening domina → proteggere downside (quality, liquidità, hedges).  

**Scoring (0–100):**
- **Market thermometers**: z-score ~5Y (z5y) → clamp [-2,+2] → 0–100.
- **Structural constraints**: percentile ~20Y (pct20y) → map [-2,+2] → 0–100.
- Soglie: >60 risk-on, 40–60 neutral, <40 risk-off (euristiche).  
                """.strip()
            )

        st.markdown("</div>", unsafe_allow_html=True)  # close section

    # ============================================================
    # WALLBOARD
    # ============================================================
    with tabs[1]:
        st.markdown("## Wallboard")
        st.markdown("<div class='muted'>Tiles senza grafici: leggi in 30 secondi, poi apri le guide se serve.</div>", unsafe_allow_html=True)

        gs_txt = "n/a" if np.isnan(global_score) else f"{global_score:.1f}"
        st.markdown(
            f"""
            <div class="grid2">
              <div class="card">
                <div class="cardTitle">Overall regime</div>
                <div class="cardValue">{gs_txt}</div>
                <div class="cardSub">{pill_html(global_status)}</div>
                <div class="cardSub">{score_bar_html(global_score)}</div>
              </div>
              <div class="card">
                <div class="cardTitle">Active alerts (today)</div>
                <div class="cardSub">
                  {"None" if not active_alerts else "<br/>".join([f"<b>{a['name']}</b> — {a['trigger']}<br/><span class='muted'>{a['so_what']}</span>" for a in active_alerts[:5]])}
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
                        st.markdown("**How to read it (bi-directional):**")
                        st.markdown(exp.get("interpretation", ""))
                        st.markdown(f"**Why it matters:** {exp.get('bridge','')}")
                else:
                    wallboard_tile(k, s, indicator_scores)
            st.markdown("</div></div>", unsafe_allow_html=True)

        render_group("1) Price of Time", "Tassi e curva: costo del capitale, sconto dei multipli e segnali di ciclo.", ["real_10y", "nominal_10y", "yield_curve_10_2"])
        render_group("2) Macro Cycle", "Inflazione e crescita: vincolo alla policy, rischio di disinflation vs sticky.", ["breakeven_10y", "cpi_yoy", "unemployment_rate"])
        render_group("3) Conditions & Stress", "USD, credito, vol e trend: regimi veloci e risk appetite.", ["usd_index", "hy_oas", "vix", "spy_trend", "hyg_lqd_ratio"])
        render_group("4) Liquidity / Plumbing", "Liquidità di sistema: tailwind/drain sulla propensione al rischio.", ["fed_balance_sheet", "rrp"])
        render_group("5) Fiscal / Policy Constraint", "Vincoli lenti: debito, deficit e term premium.", ["interest_to_receipts", "deficit_gdp", "term_premium_10y", "interest_payments", "federal_receipts"])
        render_group("6–7) External & Gold", "Vincolo esterno e conferma hedge demand.", ["current_account_gdp", "gold"])

    # ============================================================
    # DEEP DIVE
    # ============================================================
    with tabs[2]:
        st.markdown("## Deep dive")
        st.markdown("<div class='muted'>Grafici completi + guida. Seleziona una sezione e scorri.</div>", unsafe_allow_html=True)

        group = st.selectbox(
            "Select section",
            ["Price of Time", "Macro Cycle", "Conditions & Stress", "Liquidity / Plumbing", "Fiscal / Policy Constraint", "External & Gold"],
            index=0
        )

        group_map = {
            "Price of Time": ["real_10y", "nominal_10y", "yield_curve_10_2"],
            "Macro Cycle": ["breakeven_10y", "cpi_yoy", "unemployment_rate"],
            "Conditions & Stress": ["usd_index", "hy_oas", "vix", "spy_trend", "hyg_lqd_ratio"],
            "Liquidity / Plumbing": ["fed_balance_sheet", "rrp"],
            "Fiscal / Policy Constraint": ["interest_to_receipts", "deficit_gdp", "term_premium_10y", "interest_payments", "federal_receipts"],
            "External & Gold": ["current_account_gdp", "gold"],
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

            with st.expander("Indicator guide (definition, thresholds, why it matters)", expanded=False):
                exp = meta["expander"]
                st.markdown(f"**What it is:** {exp.get('what','')}")
                st.markdown(f"**Reference levels / thresholds:** {exp.get('reference','')}")
                st.markdown("**How to read it (bi-directional):**")
                st.markdown(exp.get("interpretation", ""))
                st.markdown(f"**Why it matters (funding/policy link):** {exp.get('bridge','')}")

            st.markdown("</div>", unsafe_allow_html=True)

    # ============================================================
    # WHAT CHANGED — upgraded + “So what” + watchlist
    # ============================================================
    with tabs[3]:
        st.markdown("## What changed")
        st.markdown(
            "<div class='muted'>Focus: cosa si muove in modo significativo e/o è vicino a soglie di regime. "
            "La colonna “So what” traduce il movimento in implicazioni di regime.</div>",
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
                "So what": so_what_line(key, dwin, score, status),
                "_key": key,
            })

        if not rows:
            st.info("No sufficient data to compute changes.")
        else:
            df = pd.DataFrame(rows).sort_values(["Watch", "Attention"], ascending=[True, False]).reset_index(drop=True)

            wl = df[df["Watch"] == "WATCH"].sort_values("Attention", ascending=False).head(8)
            if not wl.empty:
                st.markdown("### Watchlist (most relevant movers / threshold proximity)")
                for _, r in wl.iterrows():
                    trend_col = [c for c in df.columns if c.startswith("Trend")][0]
                    st.markdown(
                        f"""
                        <div class='card' style='margin-bottom:10px;'>
                          <div class='cardTitle'>{r['Indicator']}</div>
                          <div class='cardSub'>
                            Regime: <b>{r['Regime']}</b> · Score: <b>{r['Score']}</b> · {trend_col}: <b>{r[trend_col]:+,.2f}%</b><br/>
                            <span class='muted'>{r['So what']}</span>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            st.markdown("### Full table")
            show = df.drop(columns=["_key"])
            st.dataframe(show, use_container_width=True, hide_index=True)
            st.caption("Note: variazioni % basate sull’osservazione disponibile più vicina; frequenza serie diversa (giornaliera vs mensile vs trimestrale).")

    # ============================================================
    # ALERTS TAB — active + recent triggers + “what to do”
    # ============================================================
    with tabs[4]:
        st.markdown("## Alerts & triggers (2–6 weeks)")
        st.markdown("<div class='muted'>Alert = condizioni osservabili che spesso precedono un cambio di marcia su sizing/risk budget.</div>", unsafe_allow_html=True)

        st.markdown("<div class='grid2'>", unsafe_allow_html=True)

        # Active alerts card
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='cardTitle'>Active alerts (today)</div>", unsafe_allow_html=True)
        if not active_alerts:
            st.markdown("<div class='cardSub'>Nessun alert attivo con le soglie attuali.</div>", unsafe_allow_html=True)
        else:
            for a in active_alerts:
                sev = a["severity"]
                pill = "pill bad" if sev == "bad" else "pill warn"
                st.markdown(
                    f"<div class='cardSub'><span class='{pill}'><span class='dot' style='background:var(--bad)'></span>{a['name']}</span>"
                    f" &nbsp; <b>{a['trigger']}</b><br/><span class='muted'>{a['so_what']}</span></div>",
                    unsafe_allow_html=True
                )
        st.markdown("</div>", unsafe_allow_html=True)

        # Recent triggers card
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
        st.markdown("<div class='sectionTitle'>What to do when alerts cluster</div>", unsafe_allow_html=True)
        st.markdown(
            """
- **2+ alert “bad” insieme** → de-risk: riduci beta equity, preferisci qualità, accorcia rischio credito.
- **USD tightening + credit stress** → privilegia liquidità/hedges (cash-like, qualità IG, USD).
- **Real yields restrictive + trend equity fragile** → evita duration lunga + equity growth “pura”.
- **Alert che si spengono** → ri-aumenta rischio gradualmente (sizing misurato).
            """.strip()
        )

    # ============================================================
    # REPORT TAB — clean, hidden, copy/paste
    # ============================================================
    with tabs[5]:
        st.markdown("## Report (copy/paste)")
        st.markdown("<div class='muted'>Genera un blocco copiaincolla (prompt + payload YAML) per ottenere un report AI in un’altra chat.</div>", unsafe_allow_html=True)

        def build_yaml_payload():
            payload_lines = []
            payload_lines.append("macro_regime_payload:")
            payload_lines.append(f"  generated_at_utc: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
            payload_lines.append(f"  history_years: {years_back}")
            payload_lines.append(f"  latest_datapoint_date: {('null' if data_max_date is None else str(pd.to_datetime(data_max_date).date()))}")
            payload_lines.append(f"  global_score: {0.0 if np.isnan(global_score) else round(global_score, 1)}")
            payload_lines.append(f"  global_status: {global_status}")
            payload_lines.append("  scoring_notes: \"Market thermometers use z5y; structural constraints use pct20y\"")

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
            for k in ["Equity", "Duration", "Credit", "Hedges"]:
                payload_lines.append(f"    {k.lower()}:")
                payload_lines.append(f"      stance: \"{ops[k]['stance']}\"")
                payload_lines.append(f"      why: \"{ops[k]['why']}\"")
                payload_lines.append(f"      context: \"{ops[k]['context']}\"")

            payload_lines.append("  alerts:")
            payload_lines.append("    thresholds:")
            payload_lines.append(f"      real_yield_restrictive_gt: {thr_real_yield:.2f}")
            payload_lines.append(f"      vix_stress_gt: {thr_vix:.2f}")
            payload_lines.append(f"      hy_oas_stress_gt: {thr_hy:.2f}")
            payload_lines.append(f"      usd_ma_days: {dxy_ma_days}")
            payload_lines.append(f"      spy_trend_lt: {spy_trend_thr:.2f}")

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
                # brief guide keys (compact)
                payload_lines.append(f"      what: \"{meta['expander'].get('what','')}\"")
                payload_lines.append(f"      how_to_read: \"{meta['expander'].get('interpretation','').replace(chr(10),' / ')}\"")

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
            st.download_button("Download (.txt)", one_shot, file_name="global_markets_report_block.txt", mime="text/plain")
        else:
            st.info("Premi “Generate prompt + payload” per creare il blocco copiaincolla.")

if __name__ == "__main__":
    main()
