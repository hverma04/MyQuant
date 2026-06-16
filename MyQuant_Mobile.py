import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import numpy_financial as npf
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import plotly.express as px
import anthropic
import os
from datetime import date, datetime
import time
import functools
import requests_cache
requests_cache.install_cache('/tmp/yf_cache', backend='sqlite', expire_after=3600)


# ==========================================
# SECTION 2: ENGINES & DATA LAYER (shared logic with desktop)
# ==========================================
class FearZEngine:
    def __init__(self):
        self.params = {
            'Episodic':   {'p0': 22.3, 'lam': 0.083, 'mu': 0.65,  'min_ivr': 0},
            'Structural': {'p0': 28.7, 'lam': 0.046, 'mu': 1.14,  'min_ivr': 70},
            'Systemic':   {'p0': 34.8, 'lam': 0.021, 'mu': 2.03,  'min_ivr': 90}
        }

    def classify_shock(self, iv_rank):
        if iv_rank >= self.params['Systemic']['min_ivr']:   return 'Systemic'
        if iv_rank >= self.params['Structural']['min_ivr']: return 'Structural'
        return 'Episodic'

    def automate_gamma(self, vol_history):
        if vol_history is None or len(vol_history) < 10:
            return 0.12
        mean_iv      = vol_history.mean()
        y            = np.diff(vol_history)
        x            = mean_iv - vol_history[:-1].values
        covariance   = np.cov(x, y)[0, 1]
        variance     = np.var(x)
        ticker_gamma = covariance / variance if variance > 0 else 0.12
        return np.clip(ticker_gamma, 0.05, 0.25)

    def calculate_shelf(self, current_iv, iv_rank, vol_history):
        gamma     = self.automate_gamma(vol_history)
        threshold = 0.30 if iv_rank < 70 else 0.45
        z_days    = gamma * max(0, (current_iv * 100) - (threshold * 100))
        return round(z_days, 1), round(gamma, 3)

    def get_projection(self, t_days, current_iv, m_t0, z, category):
        p = self.params[category]
        if t_days <= z: return current_iv
        t_delta          = t_days - z
        inertia_friction = 1 + (p['p0'] / 100) + (p['mu'] * abs(m_t0))
        adjusted_lam     = p['lam'] / inertia_friction
        return round(current_iv * np.exp(-adjusted_lam * t_delta), 4)


def calculate_black_scholes(S, K, T, r, sigma, option_type="Call"):
    if T <= 0 or sigma <= 0:
        return max(0, S - K) if option_type == "Call" else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "Call":
        return (S * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
    return (K * np.exp(-r * T) * norm.cdf(-d2)) - (S * norm.cdf(-d1))

def calculate_greeks(S, K, T, r, sigma, option_type="Call"):
    if T <= 0 or sigma <= 0:
        return {"Delta": 0.0, "Gamma": 0.0, "Theta": 0.0, "Vega": 0.0, "Rho": 0.0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    if option_type == "Call":
        delta = norm.cdf(d1)
        rho   = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        delta = norm.cdf(d1) - 1
        rho   = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega  = S * pdf_d1 * np.sqrt(T) / 100
    return {"Delta": round(delta, 4), "Gamma": round(gamma, 4),
            "Theta": round(theta, 4), "Vega": round(vega, 4), "Rho": round(rho, 4)}

def kelly_position_size(win_prob, avg_win, avg_loss, account_size, fraction=0.5):
    if avg_loss <= 0 or account_size <= 0:
        return {"kelly_pct": 0, "recommended_dollars": 0, "kelly_raw": 0, "note": "Invalid inputs."}
    b          = avg_win / avg_loss
    p, q       = win_prob, 1 - win_prob
    kelly_raw  = (b * p - q) / b
    kelly_frac = max(0, kelly_raw * fraction)
    recommended = account_size * kelly_frac
    if kelly_raw <= 0:
        note = "Negative Kelly — edge is insufficient."
    elif kelly_raw > 0.25:
        note = f"Full Kelly ({kelly_raw:.1%}) is aggressive. Half-Kelly applied: {kelly_frac:.1%}."
    else:
        note = f"Kelly suggests {kelly_frac:.1%} of account (half-Kelly)."
    return {"kelly_pct": round(kelly_frac * 100, 2), "recommended_dollars": round(recommended, 2),
            "kelly_raw": round(kelly_raw * 100, 2), "note": note}

def trade_advisor_verdict(ev, premium, bs_fair_value, regime):
    score = 0.0; rules = []
    if ev > 0:
        score += 1.0; rules.append({"rule": "Expected Value", "result": "Pass", "detail": f"EV = ${ev:.2f} (positive)"})
    elif ev > -50:
        score += 0.5; rules.append({"rule": "Expected Value", "result": "Warn", "detail": f"EV = ${ev:.2f} (marginal)"})
    else:
        rules.append({"rule": "Expected Value", "result": "Fail", "detail": f"EV = ${ev:.2f} (negative)"})
    if bs_fair_value > 0:
        pct_over = (premium - bs_fair_value) / bs_fair_value
        if pct_over > 0.20:
            rules.append({"rule": "IV vs Black-Scholes", "result": "Fail", "detail": f"Premium {pct_over*100:.1f}% over BS fair value"})
        elif pct_over > 0.05:
            score += 0.5; rules.append({"rule": "IV vs Black-Scholes", "result": "Warn", "detail": f"Premium {pct_over*100:.1f}% over fair value"})
        else:
            score += 1.0; rules.append({"rule": "IV vs Black-Scholes", "result": "Pass", "detail": f"Premium within {abs(pct_over)*100:.1f}% of fair value"})
    else:
        score += 0.5; rules.append({"rule": "IV vs Black-Scholes", "result": "Warn", "detail": "Could not compute fair value"})
    if regime == 'Episodic':
        score += 1.0; rules.append({"rule": "Fear Z Regime", "result": "Pass", "detail": "Episodic — low behavioral risk"})
    elif regime == 'Structural':
        score += 0.5; rules.append({"rule": "Fear Z Regime", "result": "Warn", "detail": "Structural — moderate Panic Plateau"})
    else:
        rules.append({"rule": "Fear Z Regime", "result": "Fail", "detail": "Systemic — crisis state, IV inflated"})
    verdict = "BUY" if score >= 2.5 else ("HOLD" if score >= 1.5 else "SELL")
    return score, verdict, rules

def stock_advisor_verdict(momentum_5d, price_vs_sma, ivr, regime):
    score = 0.0; rules = []
    combined  = (momentum_5d * 100) + (price_vs_sma * 100)
    sma_label = "above" if price_vs_sma >= 0 else "below"
    if combined > 3:
        score += 1.0; rules.append({"rule": "Momentum", "result": "Pass", "detail": f"5d return {momentum_5d*100:+.1f}%, price {abs(price_vs_sma)*100:.1f}% {sma_label} 21-SMA"})
    elif combined > 0:
        score += 0.5; rules.append({"rule": "Momentum", "result": "Warn", "detail": f"5d return {momentum_5d*100:+.1f}%, marginal trend vs 21-SMA"})
    else:
        rules.append({"rule": "Momentum", "result": "Fail", "detail": f"5d return {momentum_5d*100:+.1f}%, price {sma_label} 21-SMA — bearish"})
    if ivr < 40:
        score += 1.0; rules.append({"rule": "Volatility Regime", "result": "Pass", "detail": f"IVR {ivr:.0f} — calm environment"})
    elif ivr < 70:
        score += 0.5; rules.append({"rule": "Volatility Regime", "result": "Warn", "detail": f"IVR {ivr:.0f} — elevated vol"})
    else:
        rules.append({"rule": "Volatility Regime", "result": "Fail", "detail": f"IVR {ivr:.0f} — high vol, risk-off"})
    if regime == 'Episodic':
        score += 1.0; rules.append({"rule": "Fear Z Regime", "result": "Pass", "detail": "Episodic — contained vol"})
    elif regime == 'Structural':
        score += 0.5; rules.append({"rule": "Fear Z Regime", "result": "Warn", "detail": "Structural — moderate stress"})
    else:
        rules.append({"rule": "Fear Z Regime", "result": "Fail", "detail": "Systemic — crisis state"})
    verdict = "BUY" if score >= 2.5 else ("HOLD" if score >= 1.5 else "SELL")
    return score, verdict, rules


@st.cache_data(ttl=900)
def fetch_chart_data(symbol, time_selection):
    t = yf.Ticker(symbol)
    intervals  = {"1 Day": "5m", "5 Days": "30m", "1 Month": "1d", "6 Months": "1d", "1 Year": "1d", "5 Years": "1wk"}
    period_map = {"1 Day": "1d", "5 Days": "5d", "1 Month": "1mo", "6 Months": "6mo", "1 Year": "1y", "5 Years": "5y"}
    return t.history(period=period_map.get(time_selection, "1y"), interval=intervals.get(time_selection, "1d"))

@st.cache_resource(ttl=3600)
def fetch_ticker_resource(symbol):
    t    = yf.Ticker(symbol)
    hist = t.history(period="1y")
    if hist.empty:
        return None, None, None, 0.042, 0.0, 50.0, None, None
    m_t0 = (hist["Close"].iloc[-1] / hist["Close"].iloc[-6]) - 1 if len(hist) > 5 else 0
    vols = hist["Close"].pct_change().rolling(21).std() * np.sqrt(252)
    vols = vols.dropna()
    if not vols.empty:
        vol_range = vols.max() - vols.min()
        ivr = (vols.iloc[-1] - vols.min()) / vol_range * 100 if vol_range > 0 else 50
    else:
        ivr = 50
    try:
        rf_rate = yf.Ticker("^IRX").history(period="1d")["Close"].iloc[-1] / 100
    except:
        rf_rate = 0.042
    return t, t.options, hist["Close"].iloc[-1], rf_rate, m_t0, ivr, vols, hist

@st.cache_data(ttl=3600)
def fetch_financials(symbol):
    t = yf.Ticker(symbol)
    try:
        return {"income": t.income_stmt, "balance": t.balance_sheet, "cashflow": t.cashflow, "info": t.info}
    except Exception:
        return None

@st.cache_data(ttl=600)
def fetch_company_news(symbol):
    try:
        return yf.Ticker(symbol).news or []
    except Exception:
        return []

@st.cache_data(ttl=3600)
def fetch_earnings_calendar(symbol: str):
    try:
        t   = yf.Ticker(symbol)
        cal = t.calendar
        if cal is None: return None
        if hasattr(cal, 'columns'):
            if 'Earnings Date' in cal.columns: raw = cal['Earnings Date'].iloc[0]
            elif 'Earnings Date' in cal.index:  raw = cal.loc['Earnings Date'].iloc[0]
            else: return None
        elif isinstance(cal, dict):
            raw = cal.get('Earnings Date', [None])[0]
        else:
            return None
        if raw is None: return None
        earn_date = pd.to_datetime(raw).date()
        days_away = (earn_date - date.today()).days
        return {"date": earn_date, "days_away": days_away}
    except Exception:
        return None

def _fin_val(df, *row_keys):
    if df is None or df.empty: return None
    for key in row_keys:
        matches = [i for i in df.index if key.lower().replace(" ", "") in str(i).lower().replace(" ", "")]
        if matches:
            try:
                v = df.loc[matches[0]].iloc[0]
                if pd.notna(v): return float(v)
            except Exception: continue
    return None

def _fin_val2(df, *row_keys):
    if df is None or df.empty or df.shape[1] < 2: return None, None
    for key in row_keys:
        matches = [i for i in df.index if key.lower().replace(" ", "") in str(i).lower().replace(" ", "")]
        if matches:
            try:
                v0 = df.loc[matches[0]].iloc[0]; v1 = df.loc[matches[0]].iloc[1]
                if pd.notna(v0) and pd.notna(v1): return float(v0), float(v1)
            except Exception: continue
    return None, None

def score_fundamentals(fin):
    if fin is None: return None
    inc  = fin.get("income"); bal = fin.get("balance"); cf = fin.get("cashflow"); info = fin.get("info", {}) or {}
    results = {"health": {}, "quality": {}, "growth": {}, "raw": {}}
    cur_assets = _fin_val(bal, "CurrentAssets", "Total Current Assets")
    cur_liab   = _fin_val(bal, "CurrentLiabilities", "Total Current Liabilities")
    inventory  = _fin_val(bal, "Inventory") or 0
    tot_debt   = _fin_val(bal, "TotalDebt", "LongTermDebt", "Long Term Debt", "Total Debt")
    equity     = _fin_val(bal, "StockholdersEquity", "CommonStockEquity", "Total Equity")
    ebit       = _fin_val(inc, "EBIT", "OperatingIncome", "Operating Income")
    int_exp    = _fin_val(inc, "InterestExpense", "Interest Expense")
    cr = cur_assets / cur_liab if cur_assets and cur_liab and cur_liab != 0 else None
    de = abs(tot_debt / equity) if tot_debt and equity and equity != 0 else None
    ic = abs(ebit / int_exp) if ebit and int_exp and int_exp != 0 else None
    qr = (cur_assets - inventory) / cur_liab if cur_assets and cur_liab and cur_liab != 0 else None
    if de is None:
        tot_liab = _fin_val(bal, "TotalLiabilities", "TotalLiabilitiesNetMinorityInterest")
        tot_assets = _fin_val(bal, "TotalAssets")
        if tot_liab is not None and tot_assets and tot_assets != 0: de = tot_liab / tot_assets
    if ic is None and (tot_debt is None or tot_debt == 0): ic = float("inf")
    if qr is None:
        cash = _fin_val(bal, "CashAndCashEquivalents", "Cash", "CashCashEquivalentsAndShortTermInvestments")
        recv = _fin_val(bal, "NetReceivables", "AccountsReceivable", "Receivables") or 0
        if cash is not None and cur_liab: qr = (cash + recv) / cur_liab
    results["health"]["Current Ratio"]     = (cr, (1.0 if cr and cr >= 2.0 else 0.5 if cr and cr >= 1.2 else 0.0), "≥2.0 strong / ≥1.2 ok")
    results["health"]["Debt/Equity"]       = (de, (1.0 if de is not None and de <= 0.5 else 0.5 if de is not None and de <= 1.0 else 0.0), "≤0.5 strong / ≤1.0 ok")
    results["health"]["Interest Coverage"] = (ic, (1.0 if ic and ic >= 5.0 else 0.5 if ic and ic >= 3.0 else 0.0), "≥5x strong / ≥3x ok")
    results["health"]["Quick Ratio"]       = (qr, (1.0 if qr and qr >= 1.0 else 0.5 if qr and qr >= 0.7 else 0.0), "≥1.0 strong / ≥0.7 ok")
    results["raw"].update({"Current Ratio": cr, "Debt/Equity": de, "Interest Coverage": ic, "Quick Ratio": qr})
    net_income = _fin_val(inc, "NetIncome", "Net Income"); rev = _fin_val(inc, "TotalRevenue", "Total Revenue")
    gross = _fin_val(inc, "GrossProfit", "Gross Profit"); op_cf = _fin_val(cf, "OperatingCashFlow", "Operating Cash Flow", "Cash Flow From Continuing Operating Activities")
    capex = _fin_val(cf, "CapitalExpenditure", "Capital Expenditure") or 0
    roe = net_income / equity if net_income and equity and equity != 0 else None
    net_marg = net_income / rev if net_income and rev and rev != 0 else None
    gr_marg = gross / rev if gross and rev and rev != 0 else None
    fcf = (op_cf + capex) if op_cf is not None else None
    eq_qual = op_cf / net_income if op_cf and net_income and net_income != 0 else None
    if roe is None:
        _info_roe = info.get("returnOnEquity")
        if _info_roe is not None:
            try: roe = float(_info_roe)
            except: pass
    if fcf is None and net_income is not None:
        dep = _fin_val(cf, "Depreciation", "DepreciationAndAmortization", "DepreciationAmortizationDepletion") or 0
        fcf = net_income + dep
    if net_marg is None and rev:
        ebitda = _fin_val(inc, "EBITDA", "NormalizedEBITDA")
        if ebitda and rev: net_marg = ebitda / rev
    results["quality"]["ROE"]              = (roe, (1.0 if roe and roe >= 0.20 else 0.5 if roe and roe >= 0.12 else 0.0), "≥20% strong / ≥12% ok")
    results["quality"]["Net Margin"]       = (net_marg, (1.0 if net_marg and net_marg >= 0.15 else 0.5 if net_marg and net_marg >= 0.07 else 0.0), "≥15% strong / ≥7% ok")
    results["quality"]["Gross Margin"]     = (gr_marg, (1.0 if gr_marg and gr_marg >= 0.40 else 0.5 if gr_marg and gr_marg >= 0.25 else 0.0), "≥40% strong / ≥25% ok")
    results["quality"]["Free Cash Flow"]   = (fcf, (1.0 if fcf is not None and fcf > 0 else 0.0), "Positive = pass")
    results["quality"]["Earnings Quality"] = (eq_qual, (1.0 if eq_qual and eq_qual >= 1.0 else 0.5 if eq_qual and eq_qual >= 0.8 else 0.0), "OCF/NI ≥1.0 strong")
    results["raw"].update({"ROE": roe, "Net Margin": net_marg, "Gross Margin": gr_marg, "FCF": fcf, "Earnings Quality": eq_qual})
    rev0, rev1 = _fin_val2(inc, "TotalRevenue", "Total Revenue"); gross0, gross1 = _fin_val2(inc, "GrossProfit", "Gross Profit")
    op_inc0 = _fin_val(inc, "OperatingIncome", "EBIT"); eps0, eps1 = _fin_val2(inc, "DilutedEPS", "BasicEPS", "Diluted EPS", "Basic EPS")
    rev_gr = (rev0 - rev1) / abs(rev1) if rev0 and rev1 and rev1 != 0 else None
    gm0 = gross0 / rev0 if gross0 and rev0 and rev0 != 0 else None
    gm1 = gross1 / rev1 if gross1 and rev1 and rev1 != 0 else None
    gm_trend = (gm0 - gm1) if gm0 is not None and gm1 is not None else None
    op_marg = op_inc0 / rev0 if op_inc0 and rev0 and rev0 != 0 else None
    eps_gr = (eps0 - eps1) / abs(eps1) if eps0 is not None and eps1 is not None and eps1 != 0 else None
    if eps_gr is None:
        _, ni1 = _fin_val2(inc, "NetIncome", "Net Income")
        if net_income and ni1 and ni1 != 0: eps_gr = (net_income - ni1) / abs(ni1)
    results["growth"]["Revenue Growth"]     = (rev_gr, (1.0 if rev_gr and rev_gr >= 0.15 else 0.5 if rev_gr and rev_gr >= 0.07 else 0.0), "≥15% strong / ≥7% ok")
    results["growth"]["Gross Margin Trend"] = (gm_trend, (1.0 if gm_trend and gm_trend > 0.01 else 0.5 if gm_trend and gm_trend >= -0.01 else 0.0), "Expanding strong / Flat ok")
    results["growth"]["Operating Margin"]   = (op_marg, (1.0 if op_marg and op_marg >= 0.15 else 0.5 if op_marg and op_marg >= 0.08 else 0.0), "≥15% strong / ≥8% ok")
    results["growth"]["EPS Growth"]         = (eps_gr, (1.0 if eps_gr is not None and eps_gr >= 0.10 else 0.5 if eps_gr is not None and eps_gr >= 0 else 0.0), "≥10% strong / ≥0% ok")
    results["raw"].update({"Revenue Growth": rev_gr, "GM Trend": gm_trend, "Operating Margin": op_marg, "EPS Growth": eps_gr})
    total_pts = 0.0; max_pts = 0.0
    for section in ("health", "quality", "growth"):
        for val, score, _ in results[section].values():
            if val is not None: total_pts += score; max_pts += 1.0
    results["total_score"]   = round(total_pts / max_pts * 10, 1) if max_pts > 0 else 0.0
    results["health_score"]  = sum(s for v, s, _ in results["health"].values()  if v is not None)
    results["quality_score"] = sum(s for v, s, _ in results["quality"].values() if v is not None)
    results["growth_score"]  = sum(s for v, s, _ in results["growth"].values()  if v is not None)
    return results

def _yf_retry(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for attempt in range(3):
            try:
                return func(*args, **kwargs)
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** (attempt + 1))
        return func(*args, **kwargs)
    return wrapper

@st.cache_data(ttl=300)
@_yf_retry
def fetch_market_overview():
    symbols = {"SPY": "S&P 500", "QQQ": "Nasdaq 100", "DIA": "Dow Jones", "^VIX": "VIX"}
    results = []
    for sym, name in symbols.items():
        try:
            hist = yf.Ticker(sym).history(period="2d")
            if len(hist) >= 2:
                prev = hist["Close"].iloc[-2]; curr = hist["Close"].iloc[-1]
                results.append({"Symbol": sym, "Name": name, "Price": curr, "Change": ((curr - prev) / prev) * 100})
        except: pass
    return results

_TOP10_CANDIDATES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "AMZN": "Amazon",
    "GOOGL": "Alphabet", "META": "Meta", "BRK-B": "Berkshire Hathaway",
    "AVGO": "Broadcom", "TSLA": "Tesla", "JPM": "JPMorgan Chase",
    "LLY": "Eli Lilly", "V": "Visa", "WMT": "Walmart", "COST": "Costco",
    "XOM": "ExxonMobil", "UNH": "UnitedHealth", "ORCL": "Oracle",
    "NFLX": "Netflix", "AMD": "AMD", "BAC": "Bank of America",
}

@st.cache_data(ttl=3600)
@_yf_retry
def fetch_top10_data():
    candidates = []
    for sym in _TOP10_CANDIDATES:
        try:
            fi = yf.Ticker(sym).fast_info
            mc = getattr(fi, "market_cap", None); price = getattr(fi, "last_price", None); prev = getattr(fi, "previous_close", None)
            if mc and price:
                chg = ((price - prev) / prev * 100) if prev and prev > 0 else 0.0
                candidates.append({"symbol": sym, "name": _TOP10_CANDIDATES[sym], "price": round(price, 2), "change": round(chg, 2), "market_cap": mc})
        except Exception: pass
    candidates.sort(key=lambda x: x["market_cap"], reverse=True)
    return candidates[:10]

@st.cache_data(ttl=600)
def fetch_market_news():
    try:
        return yf.Ticker("SPY").news or []
    except Exception:
        return []

STRIP_TICKERS  = ["SPY","QQQ","^VIX","^TNX","AAPL","MSFT","NVDA","TSLA","META","AMZN","AMD","PLTR","BTC-USD","GLD"]
NO_DOLLAR      = {"^VIX", "^TNX", "^GSPC"}
STRIP_LABELS   = {"^VIX": "VIX", "^TNX": "10Y", "^GSPC": "S&P", "BTC-USD": "BTC", "GLD": "Gold"}

@st.cache_data(ttl=300)
@_yf_retry
def fetch_ticker_live_strip():
    results = []
    try:
        data  = yf.download(STRIP_TICKERS, period="2d", interval="1d", auto_adjust=True, progress=False)
        close = data["Close"]
        for sym in STRIP_TICKERS:
            try:
                col = close[sym].dropna()
                if len(col) >= 2:
                    prev, curr = float(col.iloc[-2]), float(col.iloc[-1])
                    chg = ((curr - prev) / prev) * 100
                    results.append({"symbol": sym, "price": round(curr, 2), "change_pct": round(chg, 2)})
            except Exception: pass
    except Exception: pass
    return results

SCAN_UNIVERSE = [
    "SPY","QQQ","AAPL","MSFT","NVDA","TSLA","META","AMZN","GOOGL","AMD",
    "PLTR","SOFI","BAC","JPM","GS","WMT","COST","V","MA","DIS",
    "NFLX","UBER","COIN","MSTR","GME","F","GM","INTC","MU","SMCI",
    "ARM","TSM","AVGO","QCOM","CRM","ORCL","SNOW","RBLX","HOOD","RIVN",
    "NIO","BABA","SQ","PYPL","SHOP","ABNB","DASH","ROKU","SNAP","X"
]

def scan_single_ticker(symbol, option_type, holding_days, target_pct, stop_loss_pct):
    _fz = FearZEngine()
    try:
        t, expirations, spot, rf, m_t0, auto_ivr, vol_hist, _ = fetch_ticker_resource(symbol)
        if t is None or not expirations or spot is None: return None
        best_exp = None
        for exp in expirations:
            dte = (pd.to_datetime(exp) - pd.to_datetime("today")).days
            if 15 <= dte <= 60: best_exp = exp; break
        if best_exp is None:
            best_exp = min(expirations, key=lambda e: abs((pd.to_datetime(e) - pd.to_datetime("today")).days - 30))
        days_to_exp = (pd.to_datetime(best_exp) - pd.to_datetime("today")).days
        if days_to_exp < 1: return None
        opts  = t.option_chain(best_exp)
        chain = opts.calls if option_type == "Call" else opts.puts
        if chain.empty: return None
        atm_strike = chain.iloc[(chain["strike"] - spot).abs().argsort()[:1]]["strike"].values[0]
        row        = chain[chain["strike"] == atm_strike].iloc[0]
        premium    = row["ask"] if row["ask"] > 0 else row["lastPrice"]
        if premium <= 0: return None
        iv       = row["impliedVolatility"] if row["impliedVolatility"] > 0 else 0.001
        regime   = _fz.classify_shock(auto_ivr)
        shelf, _ = _fz.calculate_shelf(iv, auto_ivr, vol_hist)
        projected_iv = _fz.get_projection(holding_days, iv, m_t0, shelf, regime)
        vol_shock    = (projected_iv / iv) - 1
        target_price = spot * (1 + target_pct / 100) if option_type == "Call" else spot * (1 - target_pct / 100)
        breakeven    = atm_strike + premium if option_type == "Call" else atm_strike - premium
        adj_iv       = iv * (1 + vol_shock)
        adj_time     = max(holding_days, 1) / 365
        adj_piv      = max(adj_iv * np.sqrt(adj_time), 0.0001)
        time_to_exp  = max(days_to_exp, 1) / 365
        bs_fair      = calculate_black_scholes(spot, atm_strike, time_to_exp, rf, iv, option_type)
        drift = (rf - 0.5 * adj_iv**2) * adj_time
        t_z   = (np.log(target_price / spot) - drift) / adj_piv
        b_z   = (np.log(breakeven   / spot) - drift) / adj_piv
        if option_type == "Call":
            t_prob, b_prob = 1 - norm.cdf(t_z), 1 - norm.cdf(b_z)
            intrinsic = max(0, target_price - atm_strike)
        else:
            t_prob, b_prob = norm.cdf(t_z), norm.cdf(b_z)
            intrinsic = max(0, atm_strike - target_price)
        max_risk    = premium * 100
        risk_factor = stop_loss_pct if stop_loss_pct > 0 else 1.0
        ev          = (t_prob * (intrinsic - premium) * 100) - (((1 - b_prob) * max_risk) * risk_factor)
        score, verdict, _ = trade_advisor_verdict(ev, premium, bs_fair, regime)
        return {"Symbol": symbol, "Type": option_type, "Strike": atm_strike, "Expiry": best_exp,
                "DTE": days_to_exp, "Spot": round(spot, 2), "Premium": round(premium, 2),
                "IVR": round(auto_ivr, 1), "Regime": regime, "EV": round(ev, 2),
                "Score": round(score, 1), "Verdict": verdict, "P(Target)": f"{t_prob:.1%}"}
    except Exception: return None

def scan_single_stock(symbol, holding_days, target_pct):
    _fz = FearZEngine()
    try:
        _, _, spot, rf, m_t0, ivr, vol_hist, hist = fetch_ticker_resource(symbol)
        if spot is None or hist is None or len(hist) < 22: return None
        sma21        = hist["Close"].rolling(21).mean().iloc[-1]
        price_vs_sma = (spot / sma21) - 1 if sma21 > 0 else 0
        regime       = _fz.classify_shock(ivr)
        score, verdict, _ = stock_advisor_verdict(m_t0, price_vs_sma, ivr, regime)
        realized_vol = float(vol_hist.iloc[-1]) if not vol_hist.empty else 0.25
        adj_time     = max(holding_days, 1) / 365
        adj_piv      = max(realized_vol * np.sqrt(adj_time), 0.0001)
        drift        = (rf - 0.5 * realized_vol**2) * adj_time
        target       = spot * (1 + target_pct / 100)
        t_z          = (np.log(target / spot) - drift) / adj_piv
        p_target     = 1 - norm.cdf(t_z)
        sma_label    = "Above" if price_vs_sma >= 0 else "Below"
        return {"Symbol": symbol, "Spot": round(spot, 2), "IVR": round(ivr, 1),
                "Regime": regime, "Momentum": f"{m_t0*100:+.1f}%", "SMA21": sma_label,
                "Score": round(score, 1), "Verdict": verdict, "P(Target)": f"{p_target:.1%}"}
    except Exception: return None

def run_backtest(symbol, lookback_days=252, holding_days=21, target_pct=5.0, stop_loss_pct=10.0, direction="long"):
    _fz = FearZEngine(); _short = direction == "short"
    try:
        hist = yf.Ticker(symbol).history(period="2y")
        if hist is None or len(hist) < lookback_days + holding_days + 30: return None
        hist = hist.tail(lookback_days + holding_days + 30).copy()
        dates = hist.index
        hist["SMA21"] = hist["Close"].rolling(21).mean()
        hist["RVol"]  = hist["Close"].pct_change().rolling(21).std() * np.sqrt(252)
        hist["Ret5"]  = hist["Close"].pct_change(5)
        trades, equity, equity_curve, in_trade = [], 0.0, [], False
        for i in range(30, len(hist) - holding_days):
            if in_trade: continue
            row   = hist.iloc[i]; spot  = float(row["Close"]); sma21 = float(row["SMA21"]) if pd.notna(row["SMA21"]) else spot
            rvol  = float(row["RVol"])  if pd.notna(row["RVol"])  else 0.25
            ret5  = float(row["Ret5"])  if pd.notna(row["Ret5"])  else 0.0
            _vw   = hist["RVol"].iloc[max(0, i-252):i].dropna()
            ivr_proxy = ((rvol - _vw.min()) / (_vw.max() - _vw.min()) * 100) if len(_vw) > 1 and _vw.max() > _vw.min() else 50
            price_vs_sma = (spot / sma21) - 1 if sma21 > 0 else 0
            regime = _fz.classify_shock(ivr_proxy)
            score, verdict, _ = stock_advisor_verdict(ret5, price_vs_sma, ivr_proxy, regime)
            if verdict == "BUY":
                in_trade = True; entry_price = spot; entry_date = dates[i]
                exit_price, exit_date, outcome = spot, dates[min(i + holding_days, len(hist) - 1)], "held"
                stop_level   = entry_price * (1 + stop_loss_pct / 100) if _short else entry_price * (1 - stop_loss_pct / 100)
                target_level = entry_price * (1 - target_pct / 100) if _short else entry_price * (1 + target_pct / 100)
                for j in range(1, holding_days + 1):
                    if i + j >= len(hist): break
                    fp = float(hist.iloc[i + j]["Close"])
                    if _short:
                        if fp >= stop_level:   exit_price, exit_date, outcome = fp, dates[i+j], "stopped"; break
                        if fp <= target_level: exit_price, exit_date, outcome = fp, dates[i+j], "target";  break
                    else:
                        if fp <= stop_level:   exit_price, exit_date, outcome = fp, dates[i+j], "stopped"; break
                        if fp >= target_level: exit_price, exit_date, outcome = fp, dates[i+j], "target";  break
                    if j == holding_days: exit_price, exit_date, outcome = fp, dates[i+j-1], "held"
                raw_ret = (exit_price - entry_price) / entry_price
                pnl_pct = (-raw_ret if _short else raw_ret) * 100
                equity += pnl_pct; target_hit = exit_price <= target_level if _short else exit_price >= target_level
                trades.append({"Entry Date": entry_date.strftime("%Y-%m-%d"), "Exit Date": exit_date.strftime("%Y-%m-%d"),
                               "Entry Price": round(entry_price, 2), "Exit Price": round(exit_price, 2),
                               "P&L %": round(pnl_pct, 2), "Outcome": outcome, "Score": round(score, 1),
                               "Regime": regime, "Target Hit": target_hit})
                equity_curve.append({"Date": exit_date.strftime("%Y-%m-%d"), "Cumulative P&L %": round(equity, 2)})
                in_trade = False
        if not trades: return {"trades": [], "equity_curve": [], "stats": {}}
        n, nw = len(trades), sum(1 for t in trades if t["P&L %"] > 0)
        nt = sum(1 for t in trades if t["Target Hit"]); ns = sum(1 for t in trades if t["Outcome"] == "stopped")
        aw = np.mean([t["P&L %"] for t in trades if t["P&L %"] > 0]) if nw > 0 else 0
        al = np.mean([t["P&L %"] for t in trades if t["P&L %"] <= 0]) if n - nw > 0 else 0
        gp = sum(t["P&L %"] for t in trades if t["P&L %"] > 0); gl = abs(sum(t["P&L %"] for t in trades if t["P&L %"] <= 0))
        return {"trades": trades, "equity_curve": equity_curve,
                "stats": {"Total Trades": n, "Win Rate": round(nw/n*100, 1), "Target Hit Rate": round(nt/n*100, 1),
                          "Stop Out Rate": round(ns/n*100, 1), "Avg Win %": round(aw, 2), "Avg Loss %": round(al, 2),
                          "Profit Factor": round(gp/gl, 2) if gl > 0 else float("inf"), "Total Return %": round(equity, 2)}}
    except Exception: return None

def run_walkforward(symbol, is_months=12, oos_months=3, holding_days=21, target_pct=5.0, stop_loss_pct=10.0, direction="long"):
    _fz = FearZEngine(); _short = direction == "short"
    IS_DAYS = int(is_months * 21); OOS_DAYS = int(oos_months * 21)
    try:
        hist = yf.Ticker(symbol).history(period="5y")
        if hist is None or len(hist) < IS_DAYS + OOS_DAYS + 40: return None
        hist = hist.copy()
        hist["SMA21"] = hist["Close"].rolling(21).mean()
        hist["RVol"]  = hist["Close"].pct_change().rolling(21).std() * np.sqrt(252)
        hist["Ret5"]  = hist["Close"].pct_change(5)
        folds = []; oos_start = IS_DAYS; fold_num = 0
        while oos_start + OOS_DAYS <= len(hist) - holding_days:
            oos_end  = oos_start + OOS_DAYS
            oos_hist = hist.iloc[oos_start:oos_end]; fold_num += 1
            period_str = f"{oos_hist.index[0].strftime('%b %Y')} – {oos_hist.index[-1].strftime('%b %Y')}"
            regime_counts = {}
            for ii in range(len(oos_hist)):
                row_ii = oos_hist.iloc[ii]; rvol_ii = float(row_ii["RVol"]) if pd.notna(row_ii["RVol"]) else 0.25
                gi = oos_start + ii; _vw = hist["RVol"].iloc[max(0, gi-252):gi].dropna()
                ivr_ii = ((rvol_ii - _vw.min()) / (_vw.max() - _vw.min()) * 100) if len(_vw) > 1 and _vw.max() > _vw.min() else 50
                rg = _fz.classify_shock(ivr_ii); regime_counts[rg] = regime_counts.get(rg, 0) + 1
            dom_regime = max(regime_counts, key=regime_counts.get) if regime_counts else "Episodic"
            fold_trades = []; in_trade = False
            loop_start = min(30, max(0, len(oos_hist) - holding_days - 1))
            for ii in range(loop_start, len(oos_hist) - holding_days):
                if in_trade: continue
                row_ii = oos_hist.iloc[ii]; spot_ii = float(row_ii["Close"])
                sma21_ii = float(row_ii["SMA21"]) if pd.notna(row_ii["SMA21"]) else spot_ii
                rvol_ii  = float(row_ii["RVol"])  if pd.notna(row_ii["RVol"])  else 0.25
                ret5_ii  = float(row_ii["Ret5"])  if pd.notna(row_ii["Ret5"])  else 0.0
                gi = oos_start + ii; _vw = hist["RVol"].iloc[max(0, gi-252):gi].dropna()
                ivr_ii = ((rvol_ii - _vw.min()) / (_vw.max() - _vw.min()) * 100) if len(_vw) > 1 and _vw.max() > _vw.min() else 50
                pvsma = (spot_ii / sma21_ii) - 1 if sma21_ii > 0 else 0
                regime = _fz.classify_shock(ivr_ii)
                _, verdict, _ = stock_advisor_verdict(ret5_ii, pvsma, ivr_ii, regime)
                if verdict == "BUY":
                    in_trade = True; entry_price = spot_ii; exit_price = spot_ii; outcome = "held"
                    stop_level = entry_price * (1 + stop_loss_pct/100) if _short else entry_price * (1 - stop_loss_pct/100)
                    target_lvl = entry_price * (1 - target_pct/100)    if _short else entry_price * (1 + target_pct/100)
                    for jj in range(1, holding_days + 1):
                        if ii + jj >= len(oos_hist): break
                        fp = float(oos_hist.iloc[ii + jj]["Close"])
                        if _short:
                            if fp >= stop_level: exit_price, outcome = fp, "stopped"; break
                            if fp <= target_lvl: exit_price, outcome = fp, "target";  break
                        else:
                            if fp <= stop_level: exit_price, outcome = fp, "stopped"; break
                            if fp >= target_lvl: exit_price, outcome = fp, "target";  break
                        if jj == holding_days: exit_price = fp
                    raw = (exit_price - entry_price) / entry_price
                    pnl = (-raw if _short else raw) * 100
                    fold_trades.append({"pnl": pnl, "regime": regime, "outcome": outcome})
                    in_trade = False
            if fold_trades:
                pnls = [t["pnl"] for t in fold_trades]; n, nw = len(pnls), sum(1 for p in pnls if p > 0)
                avg = np.mean(pnls); std = np.std(pnls) if len(pnls) > 1 else 1.0
                cum = np.cumsum(pnls); maxdd = float(np.min(cum - np.maximum.accumulate(cum)))
                ann_f = np.sqrt(252 / max(holding_days, 1))
                sharpe = round(avg / std * ann_f, 2) if std > 0 else 0.0
            else:
                n = nw = 0; avg = maxdd = sharpe = 0.0
            folds.append({"fold": fold_num, "period": period_str, "dominant_regime": dom_regime,
                          "trades": n, "win_rate": round(nw/n*100, 1) if n > 0 else 0.0,
                          "avg_return": round(avg, 2), "max_dd": round(maxdd, 2),
                          "sharpe": round(sharpe, 2), "trade_details": fold_trades})
            oos_start += OOS_DAYS
        if not folds or sum(f["trades"] for f in folds) == 0: return None
        all_trades = [t for f in folds for t in f["trade_details"]]
        rg_groups = {}
        for t in all_trades: rg_groups.setdefault(t["regime"], []).append(t["pnl"])
        regime_attr = []
        for rg in ["Episodic", "Structural", "Systemic"]:
            pnls = rg_groups.get(rg)
            if not pnls: continue
            n, nw = len(pnls), sum(1 for p in pnls if p > 0); mean = np.mean(pnls)
            edge = "✅" if mean >= 1.0 else ("⚠️" if mean >= 0 else "❌")
            regime_attr.append({"Regime": f"{REGIME_ICON.get(rg,'⚪')} {rg}", "Trades": n,
                                "Win Rate": f"{nw/n*100:.0f}%", "Avg Return": f"{mean:+.2f}%",
                                "Total Return": f"{sum(pnls):+.2f}%", "Edge": edge})
        all_pnls = [t["pnl"] for t in all_trades]; std_all = np.std(all_pnls) if len(all_pnls) > 1 else 1.0
        ann_f = np.sqrt(252 / max(holding_days, 1))
        agg_sharpe = round(np.mean(all_pnls) / std_all * ann_f, 2) if std_all > 0 else 0.0
        worst_dd = min((f["max_dd"] for f in folds), default=0.0)
        return {"folds": folds, "regime_attribution": regime_attr, "agg_sharpe": agg_sharpe,
                "worst_fold_dd": round(worst_dd, 2), "total_oos_trades": len(all_trades), "n_folds": fold_num}
    except Exception: return None

@st.cache_data(ttl=600)
def analyze_watchlist_ticker(symbol):
    _fz = FearZEngine()
    _, _, spot, _, _, ivr, vol_hist, _ = fetch_ticker_resource(symbol)
    if spot is None: return None
    regime = _fz.classify_shock(ivr); shelf, gamma = _fz.calculate_shelf(0.25, ivr, vol_hist)
    return {"Symbol": symbol, "Price": round(spot, 2), "IVR": round(ivr, 1), "Regime": regime, "Shelf": f"{shelf}d", "Gamma": gamma}

def _get_anthropic_client():
    api_key = st.secrets.get("ANTHROPIC_API_KEY", None) or os.environ.get("ANTHROPIC_API_KEY", None)
    if not api_key: return None
    return anthropic.Anthropic(api_key=api_key)

_ADVISOR_SYSTEM = (
    "You are a friendly, experienced investment advisor speaking to someone who is investing their own money privately for the first time. "
    "Always reply in a notes style — short bullet points or numbered items, never long paragraphs. "
    "Briefly define any financial term you use in plain English (e.g. 'Sharpe Ratio — measures how much return you get per unit of risk'). "
    "Be warm, direct, and practical. If something carries real risk, say so plainly. Keep every bullet to 1-2 lines."
)

def _stream_ai_response(prompt, max_tokens=400, placeholder=None, system=None):
    client = _get_anthropic_client()
    if not client: return "Add `ANTHROPIC_API_KEY` to `.streamlit/secrets.toml` to enable AI features."
    full_text = ""
    try:
        stream_kwargs = dict(model="claude-sonnet-4-6", max_tokens=max_tokens,
                             messages=[{"role": "user", "content": prompt}])
        if system:
            stream_kwargs["system"] = system
        with client.messages.stream(**stream_kwargs) as stream:
            for text_chunk in stream.text_stream:
                full_text += text_chunk
                if placeholder: placeholder.markdown(f'<div class="briefing-card">{full_text}▌</div>', unsafe_allow_html=True)
        if placeholder: placeholder.markdown(f'<div class="briefing-card">{full_text}</div>', unsafe_allow_html=True)
    except Exception as e:
        full_text = f"Error: {e}"
        if placeholder: placeholder.error(full_text)
    return full_text

def generate_briefing(market_data, watchlist_data, placeholder=None):
    market_str    = "\n".join([f"- {r['Name']} ({r['Symbol']}): ${r['Price']:.2f} ({r['Change']:+.2f}%)" for r in market_data])
    watchlist_str = "\n".join([f"- {r['Symbol']}: IVR {r['IVR']} ({r['Regime']}), ${r['Price']:.2f}" for r in watchlist_data if r]) or "No watchlist data."
    prompt = f"""Morning briefing for {date.today().strftime('%B %d, %Y')}.

Market snapshot:
{market_str}

Watchlist readings:
{watchlist_str}

Give a quick morning brief using exactly these 3 bullet sections:
• 📊 What the market is doing today — plain summary of the numbers above
• 🧠 What to keep an eye on — explain any Fear Z regime (Episodic = short vol spike, Structural = sustained pressure, Systemic = market-wide fear) and why it matters today
• 🎯 How to approach today — one practical suggestion for a first-time investor

Keep each section to 2-3 lines. No jargon without a brief explanation."""
    return _stream_ai_response(prompt, max_tokens=400, placeholder=placeholder, system=_ADVISOR_SYSTEM)

def generate_trade_reasoning(symbol, trade_type, strike, expiry, premium, bs_fair, ev, regime, shelf, verdict, score, placeholder=None):
    prompt = f"""Options trade details:
- {symbol} {trade_type} | Strike ${strike:.2f} | Expires {expiry}
- Premium paid/received: ${premium:.2f} | Fair value estimate: ${bs_fair:.2f}
- Expected value: ${ev:.2f} | Market regime: {regime} ({shelf}-day window) | Score: {score:.1f}/3.0 → {verdict}

Explain this trade in 3 short bullets:
• Why the verdict is "{verdict}" — what the numbers are saying in plain English
• The biggest risk to this trade — what could go wrong
• One specific price level or date to watch

Briefly explain any options term you use (e.g. strike, premium, expiry)."""
    return _stream_ai_response(prompt, max_tokens=250, placeholder=placeholder, system=_ADVISOR_SYSTEM)

def generate_stock_reasoning(symbol, spot, momentum_5d, ivr, regime, verdict, score, target_pct, holding_days, placeholder=None):
    prompt = f"""Stock snapshot:
- {symbol} current price: ${spot:.2f}
- 5-day price momentum: {momentum_5d*100:+.1f}% (positive = trending up recently)
- IV Rank: {ivr:.0f}/100 (measures how 'expensive' options are right now — above 50 = elevated fear)
- Market regime: {regime} | Target: +{target_pct:.1f}% gain over {holding_days} days
- Score: {score:.1f}/3.0 → Verdict: {verdict}

Explain in 3 short bullets:
• Why the verdict is "{verdict}" — what the data is telling us
• The main risk — what could prevent this from working
• One catalyst or price level to keep an eye on"""
    return _stream_ai_response(prompt, max_tokens=250, placeholder=placeholder, system=_ADVISOR_SYSTEM)

def generate_fundamental_reasoning(symbol, scored, tech_verdict, tech_score, placeholder=None):
    raw = scored.get("raw", {})
    def _fmt(v, pct=False):
        if v is None: return "N/A"
        return f"{v*100:.1f}%" if pct else f"{v:.2f}"
    summary = (
        f"Health ({scored['health_score']:.1f}/4): CR {_fmt(raw.get('Current Ratio'))} D/E {_fmt(raw.get('Debt/Equity'))} IC {_fmt(raw.get('Interest Coverage'))}x\n"
        f"Quality ({scored['quality_score']:.1f}/5): ROE {_fmt(raw.get('ROE'), True)} NM {_fmt(raw.get('Net Margin'), True)} GM {_fmt(raw.get('Gross Margin'), True)}\n"
        f"Growth ({scored['growth_score']:.1f}/4): RevGr {_fmt(raw.get('Revenue Growth'), True)} EPS {_fmt(raw.get('EPS Growth'), True)}\n"
        f"Score: {scored['total_score']:.1f}/10 | Technical: {tech_verdict} ({tech_score:.1f}/3.0)"
    )
    prompt = f"""Company financial data for {symbol}:
{summary}

Explain this company's financial health in 4 short bullets:
• 💰 Overall quality — is this a financially healthy company? Give a plain verdict
• ⚠️ Biggest concern — the single weakest number and what it means in everyday terms
• 📈 Growth picture — is the business growing, and is that growth healthy?
• 🔗 Does it match the chart? — do the fundamentals support or contradict the technical signal?

When you mention any ratio (Current Ratio, D/E, ROE, etc.), add a one-phrase explanation in brackets."""
    return _stream_ai_response(prompt, max_tokens=350, placeholder=placeholder, system=_ADVISOR_SYSTEM)

def generate_news_summary(symbol, headlines, placeholder=None):
    headlines_str = "\n".join(f"- {h}" for h in headlines if h)
    prompt = f"""Recent news headlines for {symbol}:
{headlines_str}

Summarise in 3 short bullets:
• 🗞️ Overall mood — is the news positive, negative, or mixed?
• 📌 Key event — the single most important story and why it matters for the stock
• 👀 Near-term outlook — what should an investor watch for in the coming days?

Plain English only — no financial jargon without a brief explanation."""
    return _stream_ai_response(prompt, max_tokens=220, placeholder=placeholder, system=_ADVISOR_SYSTEM)

def generate_etf_analysis(portfolio, sharpe, sortino, beta, alpha_annual, max_dd, var_95, total_invested, total_fv, horizon, placeholder=None):
    holdings_str = "\n".join(f"- {t}: ${d:,.0f} ({d/total_invested*100:.1f}%)" for t, d in portfolio.items())
    prompt = f"""Portfolio holdings:
{holdings_str}

Risk numbers:
- Sharpe Ratio: {sharpe:.3f} (reward per unit of risk — above 1.0 is solid)
- Sortino Ratio: {sortino:.3f} (like Sharpe but only penalises downside risk)
- Beta vs S&P 500: {beta:.3f} (1.0 = moves with the market; above 1 = more volatile)
- Annualised Alpha: {alpha_annual:.2f}% (extra return beyond what the market gave you)
- Max Drawdown: {max_dd:.2f}% (worst peak-to-trough loss in the period)
- Daily VaR 95%: ${abs(var_95):,.0f} (on a bad day, you could lose up to this amount)
- Money invested: ${total_invested:,.0f} → projected in {horizon} years: ${total_fv:,.0f}

Explain this portfolio in 4 short bullets:
• 📊 Overall picture — is this a good portfolio for someone starting out?
• ⚖️ Diversification — is the money spread well or too concentrated in a few stocks?
• 🔢 Do the risk numbers stack up? — interpret the key metrics in plain language
• 💡 One thing to consider doing — a practical, specific suggestion"""
    return _stream_ai_response(prompt, max_tokens=400, placeholder=placeholder, system=_ADVISOR_SYSTEM)

def normalize_price_frame(raw: pd.DataFrame) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        if "Adj Close" in level0: return raw["Adj Close"].copy()
        return raw["Close"].copy()
    if "Adj Close" in raw.columns: return raw["Adj Close"].to_frame(name="SINGLE_TICKER")
    return raw["Close"].to_frame(name="SINGLE_TICKER")

def get_risk_free_daily(default_annual: float = 0.04) -> float:
    try:
        hist = yf.Ticker("^IRX").history(period="5d")
        if hist.empty or "Close" not in hist.columns: raise ValueError
        rate = float(hist["Close"].dropna().iloc[-1]) / 100.0
    except Exception: rate = default_annual
    return (1.0 + rate) ** (1.0 / 252.0) - 1.0

def etf_get_cagr(prices: pd.DataFrame, ticker: str, years: int = 10) -> float:
    if ticker not in prices.columns: return float("nan")
    series = prices[ticker].dropna()
    if series.empty: return float("nan")
    cutoff = series.index[-1] - pd.DateOffset(years=years)
    series = series[series.index >= cutoff]
    if len(series) < 2: return float("nan")
    actual_years = (series.index[-1] - series.index[0]).days / 365.25
    if actual_years <= 0: return float("nan")
    try: return float((float(series.iloc[-1]) / float(series.iloc[0])) ** (1.0 / actual_years) - 1.0)
    except Exception: return float("nan")

REGIME_ICON   = {"Episodic": "🟢", "Structural": "🟡", "Systemic": "🔴"}
RESULT_ICON   = {"Pass": "✅", "Warn": "⚠️", "Fail": "❌"}
RESULT_COLOR  = {"Pass": "#00d96f", "Warn": "#FFC107", "Fail": "#ff4b4b"}
VERDICT_COLOR = {"BUY": "#00d96f", "HOLD": "#FFC107", "SELL": "#ff4b4b"}


# ==========================================
# SECTION 3: PAGE CONFIG & MOBILE CSS
# ==========================================
st.set_page_config(page_title="MyQuant Mobile | KERN.", layout="centered", page_icon="📊")

def inject_mobile_css():
    st.markdown("""<style>
.stApp { background-color: #0a0a0c !important; }
.main .block-container { max-width: 490px !important; padding: 0 10px 16px 10px !important; margin: 0 auto !important; }
section[data-testid="stSidebar"], div[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; gap: 6px !important; }
.mob-header { display:flex; align-items:center; justify-content:space-between; padding:8px 0 6px 0; border-bottom:1px solid rgba(191,161,93,0.3); margin-bottom:8px; }
.mob-logo { font-family:'Times New Roman',serif; color:#bfa15d; letter-spacing:0.28rem; font-size:1.5rem; font-weight:700; text-transform:uppercase; }
.mob-app-name { font-family:'Courier New',monospace; font-size:0.64rem; color:rgba(191,161,93,0.65); letter-spacing:0.1em; }
.mob-inst-badge { background:rgba(191,161,93,0.12); border:1px solid rgba(191,161,93,0.4); border-radius:3px; padding:2px 7px; font-size:0.5rem; font-weight:700; letter-spacing:0.14em; text-transform:uppercase; color:#bfa15d; }
.mob-nav-active { border-bottom:2px solid #bfa15d; color:#bfa15d; font-size:0.58rem; font-weight:700; text-align:center; padding:4px 0; letter-spacing:0.05em; text-transform:uppercase; min-height:28px; display:flex; align-items:center; justify-content:center; white-space:nowrap; overflow:hidden; }
div[data-testid="stButton"] > button { font-size:0.58rem !important; padding:3px 1px !important; letter-spacing:0.04em !important; min-height:28px !important; white-space:nowrap !important; overflow:hidden !important; text-overflow:ellipsis !important; }
.mob-sh { display:flex; align-items:center; gap:8px; margin:8px 0 10px 0; }
.mob-sh-txt { color:#bfa15d; font-size:0.58rem; font-weight:700; letter-spacing:0.14em; text-transform:uppercase; white-space:nowrap; }
.mob-sh-line { flex:1; height:1px; background:linear-gradient(to right,rgba(191,161,93,0.5),rgba(191,161,93,0)); }
.mp-grid2 { display:grid; grid-template-columns:1fr 1fr; gap:7px; margin-bottom:9px; }
.mp { background:rgba(191,161,93,0.06); border:1px solid rgba(191,161,93,0.22); border-radius:6px; padding:7px 9px; min-height:50px; display:flex; flex-direction:column; justify-content:center; }
.mp-val { font-family:'Courier New',monospace; font-size:0.9rem; font-weight:700; line-height:1.1; color:#e8dfc8; }
.mp-lbl { font-size:0.54rem; text-transform:uppercase; letter-spacing:0.1em; color:#bfa15d; margin-top:3px; opacity:0.85; }
.mp-pos { font-size:0.58rem; color:#00d96f; margin-top:1px; }
.mp-neg { font-size:0.58rem; color:#ff4b4b; margin-top:1px; }
.mp-off { font-size:0.58rem; color:rgba(191,161,93,0.55); margin-top:1px; }
.verdict-chip { display:inline-flex; align-items:center; padding:3px 11px; border-radius:14px; font-size:0.7rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; }
.vc-buy  { background:rgba(0,217,111,0.12); color:#00d96f; border:1px solid #00d96f; }
.vc-hold { background:rgba(255,193,7,0.12);  color:#FFC107; border:1px solid #FFC107; }
.vc-sell { background:rgba(255,75,75,0.12);  color:#ff4b4b; border:1px solid #ff4b4b; }
.verdict-block { border-radius:8px; padding:14px 16px; margin-bottom:10px; border-width:1px; border-style:solid; text-align:center; }
.vb-buy  { border-color:#00d96f; background:rgba(0,217,111,0.05); }
.vb-hold { border-color:#FFC107; background:rgba(255,193,7,0.05); }
.vb-sell { border-color:#ff4b4b; background:rgba(255,75,75,0.05); }
.rp { display:inline-flex; align-items:center; gap:3px; padding:2px 8px; border-radius:10px; font-size:0.58rem; font-weight:600; letter-spacing:0.05em; }
.rp-ep { background:rgba(0,217,111,0.08); color:#00d96f; border:1px solid rgba(0,217,111,0.3); }
.rp-st { background:rgba(255,193,7,0.08);  color:#FFC107; border:1px solid rgba(255,193,7,0.3); }
.rp-sy { background:rgba(255,75,75,0.08);  color:#ff4b4b; border:1px solid rgba(255,75,75,0.3); }
.bl-row { display:flex; align-items:center; justify-content:space-between; padding:9px 0; border-bottom:1px solid rgba(255,255,255,0.04); min-height:38px; }
.bl-sym { font-family:'Courier New',monospace; color:#bfa15d; font-weight:700; font-size:0.8rem; min-width:50px; }
.bl-name { font-size:0.68rem; opacity:0.5; flex:1; padding:0 6px; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; }
.bl-price { font-family:'Courier New',monospace; font-size:0.8rem; font-weight:600; }
.bl-chg { font-family:'Courier New',monospace; font-size:0.75rem; min-width:62px; text-align:right; }
.bl-pos { color:#00d96f; }
.bl-neg { color:#ff4b4b; }
.sc-card { background:rgba(255,255,255,0.025); border:1px solid rgba(255,255,255,0.07); border-radius:7px; padding:10px 12px; margin-bottom:7px; }
.sc-hdr { display:flex; align-items:center; justify-content:space-between; margin-bottom:7px; }
.sc-sym { font-family:'Courier New',monospace; font-weight:700; font-size:0.86rem; color:#bfa15d; }
.sc-body { display:grid; grid-template-columns:1fr 1fr 1fr; gap:5px; font-size:0.67rem; font-family:'Courier New',monospace; }
.sc-lbl { font-size:0.52rem; text-transform:uppercase; color:#bfa15d; opacity:0.7; display:block; margin-bottom:1px; }
.news-card { border:1px solid rgba(255,255,255,0.06); border-radius:6px; padding:10px 12px; margin-bottom:7px; }
.news-title { font-size:0.8rem; font-weight:600; line-height:1.35; margin-bottom:4px; }
.news-meta { font-size:0.62rem; opacity:0.42; }
.briefing-card { background:rgba(191,161,93,0.05); border:1px solid rgba(191,161,93,0.28); border-radius:8px; padding:14px 16px; font-family:Georgia,serif; line-height:1.7; font-size:0.87rem; }
.mob-strip { width:100%; overflow:hidden; background:rgba(6,6,6,0.97); border-top:1px solid rgba(191,161,93,0.4); border-bottom:1px solid rgba(191,161,93,0.4); padding:5px 0; margin-bottom:10px; }
.mob-track { display:flex; white-space:nowrap; animation:mst 55s linear infinite; width:max-content; }
.mob-track:hover { animation-play-state:paused; }
.mob-item { display:inline-block; padding:0 16px; font-family:'Courier New',monospace; font-size:0.68rem; font-weight:600; color:#c8c8c8; border-right:1px solid rgba(191,161,93,0.18); }
.ms-sym { color:#bfa15d; font-weight:700; margin-right:4px; }
.ms-pos { color:#00d96f; }
.ms-neg { color:#ff4b4b; }
@keyframes mst { 0% { transform:translateX(0); } 100% { transform:translateX(-50%); } }
div[data-testid="stMetric"] { background:rgba(191,161,93,0.06); padding:8px 10px; border-radius:6px; border:1px solid rgba(191,161,93,0.25); min-height:54px; }
div[data-testid="stMetricLabel"] > div { color:#bfa15d !important; font-size:0.56rem !important; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; }
div[data-testid="stMetricValue"] { font-size:0.98rem !important; font-family:'Courier New',monospace !important; font-weight:700; }
.fa-strong { background:#1a3d2b; color:#00ffcc; border-radius:3px; padding:1px 6px; font-size:0.6rem; font-weight:700; }
.fa-ok     { background:#3d340a; color:#f5c842; border-radius:3px; padding:1px 6px; font-size:0.6rem; font-weight:700; }
.fa-weak   { background:#3d1a1a; color:#ff6b6b; border-radius:3px; padding:1px 6px; font-size:0.6rem; font-weight:700; }
div[data-testid="stButton"] button[kind="primary"] { width:100% !important; min-height:40px !important; font-size:0.8rem !important; }
[data-testid="stExpander"] summary { min-height:40px; font-size:0.82rem; }
</style>""", unsafe_allow_html=True)


# ==========================================
# SECTION 4: SESSION STATE & NAVIGATION
# ==========================================
def _init_state():
    for k, v in [("mob_page","home"),("positions",[]),("watchlist",["AAPL","NVDA","SPY","TSLA","META"]),
                 ("stock_watchlist",["AAPL","NVDA","MSFT","TSLA","GOOGL"]),
                 ("etf_portfolio",[]),("kelly_account",25000.0)]:
        if k not in st.session_state:
            st.session_state[k] = v

def render_nav(current_page):
    tabs = [("home","⌂ Home"),("trade","◈ Trade"),("stocks","▲ Stocks"),("scanner","⊙ Scan"),("portfolio","≡ Port")]
    cols = st.columns(5)
    for i, (key, label) in enumerate(tabs):
        if current_page == key:
            cols[i].markdown(f'<div class="mob-nav-active">{label}</div>', unsafe_allow_html=True)
        else:
            if cols[i].button(label, key=f"_mn_{key}", use_container_width=True):
                st.session_state.mob_page = key
                st.rerun()
    st.markdown('<hr style="margin:2px 0 8px;border:none;border-top:1px solid rgba(191,161,93,0.2);">', unsafe_allow_html=True)


# ==========================================
# SECTION 5: MOBILE HELPER COMPONENTS
# ==========================================
def _m_sh(label):
    st.markdown(f'<div class="mob-sh"><div class="mob-sh-txt">{label}</div><div class="mob-sh-line"></div></div>', unsafe_allow_html=True)

def _m_regime_pill(regime):
    cls = {"Episodic":"rp-ep","Structural":"rp-st","Systemic":"rp-sy"}.get(regime,"rp-ep")
    return f'<span class="rp {cls}">{REGIME_ICON.get(regime,"⚪")} {regime}</span>'

def _m_verdict_chip(verdict):
    return f'<span class="verdict-chip vc-{verdict.lower()}">{verdict}</span>'

def _m_mp2(items):
    html = "<div class='mp-grid2'>"
    for label, val, delta, dtype in items:
        dc = "mp-pos" if dtype == "pos" else ("mp-neg" if dtype == "neg" else "mp-off")
        d_html = f"<div class='{dc}'>{delta}</div>" if delta else ""
        html += f"<div class='mp'><div class='mp-val'>{val}</div><div class='mp-lbl'>{label}</div>{d_html}</div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def _m_rules(rules):
    for rule in rules:
        color = RESULT_COLOR[rule['result']]
        st.markdown(f"**{RESULT_ICON[rule['result']]} {rule['rule']}** — <span style='color:{color}'>{rule['result']}</span>: {rule['detail']}", unsafe_allow_html=True)

def _m_chart(data, height=240, target_line=None, entry_line=None):
    data = data.copy()
    data["SMA_21"] = data["Close"].rolling(21).mean()
    fig = go.Figure(data=[go.Candlestick(
        x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"],
        increasing_line_color="#00ffcc", decreasing_line_color="#ff4b4b", name="Price")])
    fig.add_trace(go.Scatter(x=data.index, y=data["SMA_21"], mode="lines",
        line=dict(color="#bfa15d", width=1), name="21 SMA"))
    if target_line: fig.add_hline(y=target_line, line_dash="dash", line_color="#00ffcc", opacity=0.65)
    if entry_line:  fig.add_hline(y=entry_line,  line_dash="dot",  line_color="#bfa15d", opacity=0.75)
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_rangeslider_visible=False, hovermode="closest", margin=dict(l=0,r=0,t=8,b=0),
        yaxis=dict(gridcolor="rgba(255,255,255,0.07)", tickfont=dict(size=8)),
        xaxis=dict(gridcolor="rgba(255,255,255,0.07)", tickfont=dict(size=8), rangebreaks=[dict(bounds=["sat","mon"])]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=7)))
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": False, "displayModeBar": False})

def _fa_badge(score):
    if score == 1.0: return '<span class="fa-strong">+ Strong</span>'
    if score == 0.5: return '<span class="fa-ok">~ OK</span>'
    return '<span class="fa-weak">x Weak</span>'

def _fa_fmt(name, val):
    if val is None: return "N/A"
    pct_set = {"ROE","Net Margin","Gross Margin","Revenue Growth","Gross Margin Trend","Operating Margin","EPS Growth"}
    if name in pct_set: return f"{val*100:.1f}%"
    if name == "Free Cash Flow": return f"${val/1e9:.2f}B" if abs(val) > 1e8 else f"${val/1e6:.1f}M"
    return f"{val:.2f}"


# ==========================================
# SECTION 6: COMPACT TICKER STRIP
# ==========================================
def render_compact_ticker_strip():
    strip_data = fetch_ticker_live_strip()
    if not strip_data:
        return
    items_html = '<span class="mob-item"><span style="font-family:Georgia,serif;font-weight:900;font-size:0.82rem;letter-spacing:0.18em;color:#bfa15d;font-style:italic;">KERN.</span></span>'
    for item in strip_data:
        sym       = item["symbol"]
        label     = STRIP_LABELS.get(sym, sym)
        price_str = f"{item['price']:.2f}" if sym in NO_DOLLAR else f"${item['price']:.2f}"
        chg       = item["change_pct"]
        arrow     = "▲" if chg >= 0 else "▼"
        css_cls   = "ms-pos" if chg >= 0 else "ms-neg"
        items_html += f'<span class="mob-item"><span class="ms-sym">{label}</span>{price_str} <span class="{css_cls}">{arrow}{abs(chg):.2f}%</span></span>'
    doubled = items_html + items_html
    st.markdown(f'<div class="mob-strip"><div class="mob-track">{doubled}</div></div>', unsafe_allow_html=True)


# ==========================================
# SECTION 7a: render_home()
# ==========================================
def render_home():
    _m_sh("Market Dashboard")

    # 2×2 market overview pills
    overview = fetch_market_overview()
    if overview:
        html = "<div class='mp-grid2'>"
        for item in overview[:4]:
            is_vix = item["Symbol"] == "^VIX"
            chg_pos = (item["Change"] < 0) if is_vix else (item["Change"] >= 0)
            dc = "mp-pos" if chg_pos else "mp-neg"
            arrow = "▲" if item["Change"] >= 0 else "▼"
            price_str = f"{item['Price']:.2f}" if is_vix else f"${item['Price']:.2f}"
            html += (f"<div class='mp'><div class='mp-val'>{price_str}</div>"
                     f"<div class='mp-lbl'>{item['Name']}</div>"
                     f"<div class='{dc}'>{arrow} {abs(item['Change']):.2f}%</div></div>")
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

    # SPY chart
    _m_sh("S&P 500")
    spy_tf = st.selectbox("Timeframe", ["1 Month","6 Months","1 Year"], index=0, key="mob_home_tf")
    spy_chart = fetch_chart_data("SPY", spy_tf)
    _m_chart(spy_chart, height=240)

    # Top 5 by market cap
    _m_sh("Top Companies by Market Cap")
    top10 = fetch_top10_data()
    if top10:
        for rank, row in enumerate(top10[:5], 1):
            arrow = "▲" if row["change"] >= 0 else "▼"
            pos_cls = "bl-pos" if row["change"] >= 0 else "bl-neg"
            mc = row.get("market_cap", 0)
            mc_str = f"${mc/1e12:.2f}T" if mc >= 1e12 else f"${mc/1e9:.1f}B"
            st.markdown(f"""<div class='bl-row'>
                <div><span class='bl-sym'>{row['symbol']}</span></div>
                <span class='bl-name'>#{rank} {row['name']} &nbsp; <span style='font-size:0.6rem;opacity:0.5;'>{mc_str}</span></span>
                <span class='bl-price'>${row['price']:,.2f}</span>
                <span class='bl-chg {pos_cls}'>{arrow}{abs(row['change']):.2f}%</span>
            </div>""", unsafe_allow_html=True)

    # Market news
    _m_sh("Market News")
    mkt_news = fetch_market_news()
    for ni in mkt_news[:4]:
        c = ni.get("content", ni)
        title = c.get("title", "")
        if not title: continue
        pub = c.get("provider", {}).get("displayName", ni.get("publisher", ""))
        raw_dt = c.get("pubDate", "")
        try:
            dt_str = datetime.strptime(raw_dt, "%Y-%m-%dT%H:%M:%SZ").strftime("%b %d") if raw_dt else ""
        except Exception:
            ts = ni.get("providerPublishTime", 0)
            dt_str = datetime.fromtimestamp(ts).strftime("%b %d") if ts else ""
        st.markdown(f'<div class="news-card"><div class="news-title">{title}</div><div class="news-meta">{pub} &middot; {dt_str}</div></div>', unsafe_allow_html=True)

    # AI Morning Briefing
    _m_sh("AI Morning Briefing")
    st.caption(date.today().strftime("%B %d, %Y"))
    if st.button("Generate Briefing", type="primary", key="mob_home_briefing"):
        market_data = fetch_market_overview()
        watchlist   = st.session_state.get("watchlist", ["SPY","QQQ","AAPL","NVDA"])
        wl_data     = [r for sym in watchlist[:5] if (r := analyze_watchlist_ticker(sym))]
        generate_briefing(market_data, wl_data, placeholder=st.empty())


# ==========================================
# SECTION 7b: render_trade()
# ==========================================
def render_trade():
    _fz = FearZEngine()
    _m_sh("Options Trade Advisor")

    _ta_data = st.session_state.get("mob_ta_data")
    form_expanded = _ta_data is None

    with st.expander("Trade Parameters", expanded=form_expanded):
        ta_ticker = st.text_input("Ticker Symbol", value=st.session_state.get("mob_ta_ticker_val","SPY"),
                                  key="mob_ta_ticker_inp").upper().strip()
        if ta_ticker:
            st.session_state["mob_ta_ticker_val"] = ta_ticker
            t_obj, expirations, spot, rf, m_t0, auto_ivr, vol_hist, _ = fetch_ticker_resource(ta_ticker)
            if t_obj is None or not expirations:
                st.error(f"No data for '{ta_ticker}'.")
            else:
                exp      = st.selectbox("Expiration", expirations, key="mob_ta_exp_sel")
                opt_type = st.radio("Option Type", ["Call","Put"], horizontal=True, key="mob_ta_type_sel")
                opts     = t_obj.option_chain(exp)
                chain    = opts.calls if opt_type == "Call" else opts.puts
                if chain.empty:
                    st.warning("No options chain for this expiration.")
                else:
                    strikes   = chain["strike"].tolist()
                    atm_idx   = int((chain["strike"] - spot).abs().argsort().iloc[0])
                    stored_s  = st.session_state.get("mob_ta_strike_val")
                    def_idx   = strikes.index(stored_s) if stored_s and stored_s in strikes else min(atm_idx, len(strikes)-1)
                    strike    = st.selectbox("Strike", strikes, index=def_idx, key="mob_ta_strike_sel")
                    st.session_state["mob_ta_strike_val"] = strike
                    row_s     = chain[chain["strike"] == strike].iloc[0]
                    iv        = row_s["impliedVolatility"] if row_s["impliedVolatility"] > 0 else 0.001
                    regime    = _fz.classify_shock(auto_ivr)
                    shelf, _  = _fz.calculate_shelf(iv, auto_ivr, vol_hist)

                    c1, c2 = st.columns(2)
                    c1.markdown(f"<div class='mp'><div class='mp-val'>${spot:.2f}</div><div class='mp-lbl'>Spot</div></div>", unsafe_allow_html=True)
                    c2.markdown(f"<div class='mp'><div class='mp-val'>{int(auto_ivr)}</div><div class='mp-lbl'>IV Rank</div><div class='mp-off'>{_m_regime_pill(regime)}</div></div>", unsafe_allow_html=True)

                    ivr_ov  = st.slider("IVR Stress Override", 0, 100, int(auto_ivr), key="mob_ta_ivr_sl")
                    target  = st.number_input("Target Price ($)", value=float(round(spot * 1.05, 2)), min_value=0.01, step=0.5, key="mob_ta_tgt")
                    c3, c4  = st.columns(2)
                    contr   = c3.number_input("Contracts", value=1, min_value=1, key="mob_ta_contr")
                    sl_pct  = c4.slider("Stop Loss %", 0, 100, 20, key="mob_ta_sl_sl") / 100

                    if st.button("Load Trade ▶", type="primary", key="mob_ta_load_btn"):
                        regime_ov   = _fz.classify_shock(ivr_ov)
                        shelf_ov, _ = _fz.calculate_shelf(iv, ivr_ov, vol_hist)
                        st.session_state["mob_ta_data"] = {
                            "ticker": ta_ticker, "exp": exp, "type": opt_type, "strike": strike,
                            "ask": float(row_s["ask"]) if row_s["ask"] > 0 else float(row_s["lastPrice"]),
                            "spot": spot, "rf": rf, "m_t0": m_t0, "ivr": ivr_ov, "iv": iv,
                            "shelf": shelf_ov, "regime": regime_ov,
                            "target": target, "contracts": int(contr), "sl": sl_pct,
                            "vol_hist": vol_hist,
                        }
                        st.rerun()

    if not _ta_data:
        st.info("Enter a ticker and click **Load Trade ▶** to begin.")
        return

    d           = _ta_data
    premium     = d["ask"]
    days_to_exp = max((pd.to_datetime(d["exp"]) - pd.to_datetime("today")).days, 1)
    time_to_exp = days_to_exp / 365
    breakeven   = d["strike"] + premium if d["type"] == "Call" else d["strike"] - premium

    adj_iv  = d["iv"]
    adj_t   = max(days_to_exp, 1) / 365
    adj_piv = max(adj_iv * np.sqrt(adj_t), 0.0001)
    bs_fair = calculate_black_scholes(d["spot"], d["strike"], time_to_exp, d["rf"], d["iv"], d["type"])
    drift   = (d["rf"] - 0.5 * adj_iv**2) * adj_t
    t_z     = (np.log(d["target"] / d["spot"]) - drift) / adj_piv
    b_z     = (np.log(breakeven / d["spot"])   - drift) / adj_piv
    s_z     = (np.log(d["strike"] / d["spot"]) - drift) / adj_piv
    if d["type"] == "Call":
        t_prob, b_prob, s_prob = 1-norm.cdf(t_z), 1-norm.cdf(b_z), 1-norm.cdf(s_z)
        intrinsic = max(0, d["target"] - d["strike"])
    else:
        t_prob, b_prob, s_prob = norm.cdf(t_z), norm.cdf(b_z), norm.cdf(s_z)
        intrinsic = max(0, d["strike"] - d["target"])
    total_pnl   = (intrinsic - premium) * 100 * d["contracts"]
    max_risk    = premium * d["contracts"] * 100
    rf_factor   = 1.0 if d["sl"] == 0 else d["sl"]
    ev          = (t_prob * total_pnl) - ((1 - b_prob) * max_risk * rf_factor)
    score, verdict, rules = trade_advisor_verdict(ev, premium, bs_fair, d["regime"])
    greeks      = calculate_greeks(d["spot"], d["strike"], time_to_exp, d["rf"], d["iv"], d["type"])
    pct_diff    = ((premium - bs_fair) / bs_fair * 100) if bs_fair > 0 else 0
    vc          = VERDICT_COLOR[verdict]

    _m_sh(f"{d['ticker']} {d['type']} ${d['strike']:.0f} | {d['exp']}")

    _m_mp2([
        ("Spot",         f"${d['spot']:.2f}",      None, "off"),
        ("Premium",      f"${premium:.2f}",         None, "off"),
        ("BS Fair",      f"${bs_fair:.2f}",         f"{pct_diff:+.1f}%", "pos" if pct_diff <= 0 else "neg"),
        ("Expected Val", f"${ev:.0f}",              None, "pos" if ev > 0 else "neg"),
        ("Breakeven",    f"${breakeven:.2f}",       None, "off"),
        ("IV",           f"{d['iv']*100:.1f}%",     None, "off"),
        ("Fear Z Shelf", f"{d['shelf']}d",          d["regime"], "off"),
        ("P(Target)",    f"{t_prob:.1%}",           None, "pos" if t_prob > 0.5 else "off"),
    ])

    st.markdown(f"Regime: {_m_regime_pill(d['regime'])}", unsafe_allow_html=True)

    _m_sh("Options Greeks")
    g_html = "<div style='display:flex;gap:6px;overflow-x:auto;padding-bottom:4px;margin-bottom:10px;'>"
    for gk, gv in greeks.items():
        g_html += f"<div class='mp' style='min-width:68px;'><div class='mp-val' style='font-size:0.76rem;'>{gv:+.3f}</div><div class='mp-lbl'>{gk}</div></div>"
    g_html += "</div>"
    st.markdown(g_html, unsafe_allow_html=True)

    _m_sh("Trade Verdict")
    st.markdown(f"""<div class='verdict-block vb-{verdict.lower()}'>
        <div style='font-size:2rem;font-weight:900;color:{vc};line-height:1;'>{verdict}</div>
        <div style='margin:6px 0 4px;'>{_m_verdict_chip(verdict)}</div>
        <div style='font-size:0.78rem;opacity:0.6;'>{score:.1f} / 3.0 pts</div>
    </div>""", unsafe_allow_html=True)
    _m_rules(rules)

    _m_sh("Kelly Criterion")
    k_account = st.number_input("Account Size ($)", min_value=1000.0,
                                value=float(st.session_state.get("kelly_account",25000.0)),
                                step=1000.0, key="mob_ta_kelly")
    st.session_state.kelly_account = k_account
    k_win_prob   = 0.30 + (score / 3.0) * 0.45
    k_pot_profit = max(0, d["target"] - breakeven) if d["type"] == "Call" else max(0, breakeven - d["target"])
    k_actual_rsk = premium * (rf_factor if rf_factor > 0 else 1.0)
    k_avg_win    = max(k_pot_profit * d["contracts"] * 100, 1)
    k_avg_loss   = max(k_actual_rsk * d["contracts"], 1)
    kd           = kelly_position_size(k_win_prob, k_avg_win, k_avg_loss, k_account)
    k_contracts  = max(1, int(kd["recommended_dollars"] / max(premium * 100, 1)))
    ka, kb = st.columns(2)
    ka.metric("Kelly $", f"${kd['recommended_dollars']:,.0f}")
    kb.metric("Kelly Contracts", str(k_contracts))
    st.caption(kd["note"])

    _m_sh(f"{d['ticker']} Price Chart")
    ta_tf = st.selectbox("Timeframe", ["1 Month","6 Months","1 Year","5 Years"], index=1, key="mob_ta_tf")
    chart_data = fetch_chart_data(d["ticker"], ta_tf)
    _m_chart(chart_data, height=240, target_line=d["target"], entry_line=d["spot"])

    _m_sh("Probability Summary")
    prob_html = f"""<div class='sc-card'><div class='sc-body' style='grid-template-columns:1fr 1fr 1fr;'>
        <div><span class='sc-lbl'>Level</span></div>
        <div><span class='sc-lbl'>Price</span></div>
        <div><span class='sc-lbl'>Probability</span></div>
        <div>Target</div><div>${d['target']:.2f}</div><div>{t_prob:.1%}</div>
        <div>Strike</div><div>${d['strike']:.2f}</div><div>{s_prob:.1%}</div>
        <div>Breakeven</div><div>${breakeven:.2f}</div><div>{b_prob:.1%}</div>
    </div></div>"""
    st.markdown(prob_html, unsafe_allow_html=True)

    _m_sh("Price Distribution (10,000 Simulations)")
    sim_prices = np.random.lognormal(np.log(d["spot"]) + drift, adj_piv, 10000)
    p5, p95    = np.percentile(sim_prices, [5, 95])
    fig_hist   = go.Figure()
    fig_hist.add_vrect(x0=p5, x1=p95, fillcolor="#bfa15d", opacity=0.1, layer="below", line_width=0)
    fig_hist.add_trace(go.Histogram(x=sim_prices, nbinsx=100, marker_color="#bfa15d", opacity=0.7))
    fig_hist.add_vline(x=d["spot"],  line_dash="dash",  line_color="#ffffff", opacity=0.7)
    fig_hist.add_vline(x=breakeven,  line_dash="solid", line_color="#ff4b4b")
    fig_hist.add_vline(x=d["target"],line_dash="dot",   line_color="#00ffcc")
    fig_hist.update_layout(height=200, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Price ($)", tickfont=dict(size=8)), yaxis=dict(title="Freq", tickfont=dict(size=8)),
        showlegend=False, bargap=0.15, margin=dict(l=0,r=0,t=12,b=0))
    st.plotly_chart(fig_hist, use_container_width=True, config={"scrollZoom": False, "displayModeBar": False})

    prob_hit = (sim_prices >= d["target"]).mean() if d["type"] == "Call" else (sim_prices <= d["target"]).mean()
    st.markdown(f'<div class="briefing-card"><strong>Simulation:</strong> In a <strong>{d["regime"]}</strong> regime, '
                f'Monte Carlo shows <strong style="color:#bfa15d;">{prob_hit:.1%}</strong> probability of reaching '
                f'<strong>${d["target"]:.2f}</strong>. Verdict: <strong style="color:{vc};">{verdict}</strong> ({score:.1f}/3.0).</div>',
                unsafe_allow_html=True)

    if st.button("Get AI Trade Reasoning", type="secondary", key="mob_ta_ai"):
        generate_trade_reasoning(d["ticker"], d["type"], d["strike"], d["exp"],
                                 premium, bs_fair, ev, d["regime"], d["shelf"], verdict, score, placeholder=st.empty())

    if st.button("Save to Positions", type="secondary", key="mob_ta_save"):
        if "positions" not in st.session_state: st.session_state.positions = []
        st.session_state.positions.append({"Symbol": d["ticker"], "Type": d["type"],
            "Strike": d["strike"], "Expiration": d["exp"], "Entry Premium": round(premium, 2),
            "Target": d["target"], "Contracts": d["contracts"], "Stop Loss": f"{int(d['sl']*100)}%",
            "Opened": date.today().isoformat()})
        st.success("Position saved to Portfolio tab.")

    if st.button("Clear / New Trade", type="secondary", key="mob_ta_clear"):
        st.session_state.pop("mob_ta_data", None)
        st.rerun()


# ==========================================
# SECTION 7c: render_stocks()
# ==========================================
def render_stocks():
    _m_sh("Stock Analytics")
    tab_adv, tab_wl = st.tabs(["Advisor", "Watchlist"])

    with tab_adv:
        sa_ticker = st.text_input("Ticker Symbol", value=st.session_state.get("mob_sa_ticker_val","AAPL"),
                                  key="mob_sa_ticker").upper().strip()
        c1, c2 = st.columns(2)
        sa_hold   = c1.slider("Holding (days)", 5, 120, 21, key="mob_sa_hold")
        sa_target = c2.number_input("Target Move (%)", value=5.0, min_value=0.1, step=0.5, key="mob_sa_tgt")
        sa_shares = st.number_input("Shares", value=100, min_value=1, key="mob_sa_shares")

        if st.button("Analyze Stock", type="primary", key="mob_sa_analyze") and sa_ticker:
            st.session_state["mob_sa_ticker_val"] = sa_ticker
            with st.spinner(f"Analyzing {sa_ticker}..."):
                _, _, spot, rf, m_t0, ivr, vol_hist, hist = fetch_ticker_resource(sa_ticker)
            if spot is None:
                st.error(f"No data for '{sa_ticker}'.")
            else:
                _fz = FearZEngine()
                regime       = _fz.classify_shock(ivr)
                sma21        = hist["Close"].rolling(21).mean().iloc[-1]
                price_vs_sma = (spot / sma21) - 1 if sma21 > 0 else 0
                high_52w     = hist["Close"].tail(252).max()
                low_52w      = hist["Close"].tail(252).min()
                pos_52w      = ((spot - low_52w) / (high_52w - low_52w) * 100) if high_52w > low_52w else 50
                day_hist     = hist["Close"].tail(2)
                day_chg_pct  = ((day_hist.iloc[-1] / day_hist.iloc[-2]) - 1) * 100 if len(day_hist) >= 2 else 0
                realized_vol = float(vol_hist.iloc[-1]) if not vol_hist.empty else 0.25
                adj_time     = max(sa_hold, 1) / 365
                adj_piv      = max(realized_vol * np.sqrt(adj_time), 0.0001)
                sa_drift     = (rf - 0.5 * realized_vol**2) * adj_time
                target_p     = spot * (1 + sa_target / 100)
                t_z          = (np.log(target_p / spot) - sa_drift) / adj_piv
                p_target     = 1 - norm.cdf(t_z)
                sa_gain      = (target_p - spot) * sa_shares
                sa_ev        = p_target * sa_gain
                score, verdict, rules = stock_advisor_verdict(m_t0, price_vs_sma, ivr, regime)
                st.session_state["mob_sa_data"] = {
                    "ticker": sa_ticker, "spot": spot, "rf": rf, "m_t0": m_t0, "ivr": ivr,
                    "vol_hist": vol_hist, "hist": hist, "regime": regime, "sma21": sma21,
                    "price_vs_sma": price_vs_sma, "pos_52w": pos_52w, "day_chg_pct": day_chg_pct,
                    "adj_piv": adj_piv, "sa_drift": sa_drift, "target_p": target_p,
                    "p_target": p_target, "sa_gain": sa_gain, "sa_ev": sa_ev,
                    "score": score, "verdict": verdict, "rules": rules,
                    "sa_hold": sa_hold, "sa_target": sa_target, "sa_shares": sa_shares,
                }
                st.rerun()

        d = st.session_state.get("mob_sa_data")
        if d:
            vc  = VERDICT_COLOR[d["verdict"]]
            sma = "▲ Above 21-SMA" if d["price_vs_sma"] >= 0 else "▼ Below 21-SMA"
            _m_sh(f"{d['ticker']} @ ${d['spot']:.2f}")
            _m_mp2([
                ("Spot",       f"${d['spot']:.2f}",     f"{d['day_chg_pct']:+.2f}%", "pos" if d["day_chg_pct"] >= 0 else "neg"),
                ("IV Rank",    f"{d['ivr']:.0f}",        None, "off"),
                ("5d Momentum",f"{d['m_t0']*100:+.1f}%", None, "pos" if d["m_t0"] >= 0 else "neg"),
                ("52W Position",f"{d['pos_52w']:.0f}%",  None, "off"),
                ("SMA Status", sma,                      None, "pos" if d["price_vs_sma"] >= 0 else "neg"),
                ("P(Target)",  f"{d['p_target']:.1%}",  None, "off"),
                ("Exp. Value", f"${d['sa_ev']:,.0f}",   None, "pos" if d["sa_ev"] >= 0 else "neg"),
                ("Regime",     REGIME_ICON.get(d['regime'],'⚪')+" "+d['regime'][:3], None, "off"),
            ])

            _m_sh("Stock Advisor Verdict")
            st.markdown(f"""<div class='verdict-block vb-{d["verdict"].lower()}'>
                <div style='font-size:2rem;font-weight:900;color:{vc};line-height:1;'>{d["verdict"]}</div>
                <div style='margin:6px 0 4px;'>{_m_verdict_chip(d["verdict"])}</div>
                <div style='font-size:0.78rem;opacity:0.6;'>{d["score"]:.1f} / 3.0 pts</div>
            </div>""", unsafe_allow_html=True)
            _m_rules(d["rules"])

            if st.button("Get AI Stock Reasoning", type="secondary", key="mob_sa_ai"):
                generate_stock_reasoning(d["ticker"], d["spot"], d["m_t0"], d["ivr"], d["regime"],
                                         d["verdict"], d["score"], d["sa_target"], d["sa_hold"], placeholder=st.empty())

            _m_sh(f"{d['ticker']} Price Chart")
            sa_tf = st.selectbox("Timeframe", ["1 Month","6 Months","1 Year","5 Years"], index=2, key="mob_sa_tf")
            chart_data = fetch_chart_data(d["ticker"], sa_tf)
            _m_chart(chart_data, height=240, target_line=d["target_p"], entry_line=d["spot"])

            _m_sh("Deep Fundamental Analysis")
            _fk = f"mob_fund_{d['ticker']}"
            if st.button("Run Deep Analysis", type="primary", key="mob_sa_fund"):
                with st.spinner("Fetching financials..."):
                    fin = fetch_financials(d["ticker"]); scored = score_fundamentals(fin)
                if scored: st.session_state[_fk] = scored
                else: st.error("Could not fetch financial data.")
            if _fk in st.session_state:
                sc = st.session_state[_fk]
                fc = "#00ffcc" if sc["total_score"] >= 7 else ("#f5c842" if sc["total_score"] >= 5 else "#ff6b6b")
                st.markdown(f'<div style="text-align:center;border:1px solid rgba(191,161,93,0.25);border-radius:8px;padding:10px;margin-bottom:10px;"><div style="font-size:0.58rem;color:#bfa15d;text-transform:uppercase;letter-spacing:0.1em;">Fundamental Score</div><div style="font-size:2.4rem;font-weight:900;color:{fc};">{sc["total_score"]}</div><div style="font-size:0.7rem;opacity:0.5;">/ 10.0</div></div>', unsafe_allow_html=True)
                for section_name, section_data, max_pts in [
                    (f"Financial Health — {sc['health_score']:.1f}/4.0", sc["health"], 4),
                    (f"Profitability — {sc['quality_score']:.1f}/5.0",   sc["quality"], 5),
                    (f"Growth — {sc['growth_score']:.1f}/4.0",           sc["growth"], 4),
                ]:
                    with st.expander(section_name):
                        for mn, (mv, ms, mt) in section_data.items():
                            st.markdown(f"{_fa_badge(ms)} **{mn}**: {_fa_fmt(mn, mv)}", unsafe_allow_html=True)
                            st.caption(mt)
                if st.button("Generate AI Fundamental Analysis", type="secondary", key="mob_sa_fund_ai"):
                    generate_fundamental_reasoning(d["ticker"], sc, d["verdict"], d["score"], placeholder=st.empty())

    with tab_wl:
        _m_sh("Stock Watchlist")
        if "stock_watchlist" not in st.session_state:
            st.session_state.stock_watchlist = ["AAPL","NVDA","MSFT","TSLA","GOOGL"]
        ac, bc = st.columns([3, 1])
        new_st = ac.text_input("Add Ticker", placeholder="e.g. AMZN", key="mob_sw_add").upper().strip()
        bc.write(""); bc.write("")
        if bc.button("Add", key="mob_sw_addbtn") and new_st and new_st not in st.session_state.stock_watchlist:
            st.session_state.stock_watchlist.append(new_st); st.rerun()
        rc, rbc = st.columns([3, 1])
        rm_st = rc.selectbox("Remove", ["---"] + st.session_state.stock_watchlist, key="mob_sw_rm")
        rbc.write(""); rbc.write("")
        if rbc.button("Remove", key="mob_sw_rmbtn") and rm_st != "---":
            st.session_state.stock_watchlist.remove(rm_st); st.rerun()
        st.caption(f"Watching: {', '.join(st.session_state.stock_watchlist)}")
        if st.button("Analyze Watchlist", type="primary", key="mob_sw_analyze"):
            with st.spinner("Analyzing..."):
                results = [scan_single_stock(sym, 21, 5) for sym in st.session_state.stock_watchlist]
            st.session_state["mob_sw_results"] = results; st.rerun()
        if "mob_sw_results" in st.session_state:
            for sym, r in zip(st.session_state.stock_watchlist, st.session_state["mob_sw_results"]):
                if r is None:
                    st.markdown(f'<div class="sc-card"><span class="sc-sym">{sym}</span> <span style="opacity:0.4;font-size:0.7rem;">No data</span></div>', unsafe_allow_html=True)
                    continue
                vc = VERDICT_COLOR.get(r["Verdict"], "#bfa15d")
                ri = REGIME_ICON.get(r["Regime"], "⚪")
                mom_col = "#00d96f" if "+" in str(r["Momentum"]) else "#ff4b4b"
                st.markdown(f"""<div class='sc-card'>
                    <div class='sc-hdr'><div><span class='sc-sym'>{r['Symbol']}</span> <span style='font-size:0.64rem;opacity:0.5;'>{ri} {r['Regime']}</span></div>
                    {_m_verdict_chip(r['Verdict'])}</div>
                    <div class='sc-body'>
                        <div><span class='sc-lbl'>Price</span>${r['Spot']:.2f}</div>
                        <div><span class='sc-lbl'>IVR</span>{r['IVR']:.0f}</div>
                        <div><span class='sc-lbl'>Score</span>{r['Score']}/3</div>
                        <div><span class='sc-lbl'>Momentum</span><span style='color:{mom_col};'>{r['Momentum']}</span></div>
                        <div><span class='sc-lbl'>SMA21</span>{r['SMA21']}</div>
                        <div><span class='sc-lbl'>P(Target)</span>{r['P(Target)']}</div>
                    </div></div>""", unsafe_allow_html=True)


# ==========================================
# SECTION 7d: render_scanner()
# ==========================================
def render_scanner():
    _m_sh("Market Scanner")
    tab_opt, tab_stk, tab_bt = st.tabs(["Options Scan", "Stock Scan", "Backtest"])

    with tab_opt:
        scan_type   = st.radio("Option Type", ["Call","Put","Both"], horizontal=True, key="mob_os_type")
        c1, c2      = st.columns(2)
        hold_days   = c1.slider("Holding (days)", 5, 60, 21, key="mob_os_hold")
        target_pct  = c2.slider("Target Move (%)", 1, 20, 5, key="mob_os_tgt")
        c3, c4      = st.columns(2)
        scan_sl     = c3.slider("Stop Loss (%)", 0, 100, 20, key="mob_os_sl") / 100
        min_score   = c4.slider("Min Score", 0.0, 3.0, 1.5, step=0.5, key="mob_os_minscore")

        if st.button("Run Options Scan", type="primary", key="mob_os_run"):
            types_to_scan = ["Call","Put"] if scan_type == "Both" else [scan_type]
            all_results   = []
            total = len(SCAN_UNIVERSE) * len(types_to_scan); done = 0
            progress = st.progress(0, text="Scanning options market...")
            for sym in SCAN_UNIVERSE:
                for ot in types_to_scan:
                    r = scan_single_ticker(sym, ot, hold_days, target_pct, scan_sl)
                    if r and r["Score"] >= min_score: all_results.append(r)
                    done += 1
                    progress.progress(done / total, text=f"Scanning {sym} {ot}...")
            progress.empty()
            st.session_state["mob_os_results"] = all_results; st.rerun()

        if "mob_os_results" in st.session_state:
            results = st.session_state["mob_os_results"]
            if not results:
                st.warning("No opportunities found. Try lowering the minimum score.")
            else:
                top10 = sorted(results, key=lambda x: (x["Score"], x["EV"]), reverse=True)[:10]
                st.success(f"Found {len(results)} opportunities. Top {len(top10)} shown.")
                for rank, r in enumerate(top10, 1):
                    ri = REGIME_ICON.get(r["Regime"], "⚪")
                    st.markdown(f"""<div class='sc-card'>
                        <div class='sc-hdr'>
                            <div><span class='sc-sym'>#{rank} {r['Symbol']}</span>
                            <span style='font-size:0.62rem;opacity:0.55;margin-left:6px;'>{r['Type']} ${r['Strike']:.0f} ({r['DTE']}d)</span></div>
                            {_m_verdict_chip(r['Verdict'])}</div>
                        <div class='sc-body'>
                            <div><span class='sc-lbl'>Spot</span>${r['Spot']:.2f}</div>
                            <div><span class='sc-lbl'>Premium</span>${r['Premium']:.2f}</div>
                            <div><span class='sc-lbl'>IVR</span>{r['IVR']:.0f}</div>
                            <div><span class='sc-lbl'>Regime</span>{ri} {r['Regime'][:3]}</div>
                            <div><span class='sc-lbl'>EV</span>${r['EV']:.0f}</div>
                            <div><span class='sc-lbl'>Score</span>{r['Score']}/3</div>
                        </div></div>""", unsafe_allow_html=True)
                    if st.button(f"Analyze {r['Symbol']} {r['Type']} →", key=f"mob_os_go_{rank}"):
                        st.session_state.mob_page = "trade"
                        st.session_state["mob_ta_ticker_val"] = r["Symbol"]
                        st.session_state.pop("mob_ta_data", None)
                        st.rerun()
                fig_bar = go.Figure(go.Bar(
                    x=[r["Symbol"]+r["Type"][0] for r in top10],
                    y=[r["Score"] for r in top10],
                    marker_color=[VERDICT_COLOR.get(r["Verdict"],"#bfa15d") for r in top10]))
                fig_bar.add_hline(y=2.5, line_dash="dot", line_color="#00d96f")
                fig_bar.add_hline(y=1.5, line_dash="dot", line_color="#FFC107")
                fig_bar.update_layout(height=200, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(range=[0,3.4], gridcolor="rgba(255,255,255,0.08)", tickfont=dict(size=8)),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickfont=dict(size=7)),
                    margin=dict(l=0,r=0,t=20,b=0), showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Set parameters and click **Run Options Scan**.")

    with tab_stk:
        c1, c2       = st.columns(2)
        ss_hold      = c1.slider("Holding (days)", 5, 120, 21, key="mob_ss_hold")
        ss_tgt       = c2.slider("Target Move (%)", 1, 30, 5, key="mob_ss_tgt")
        ss_min_score = st.slider("Min Score", 0.0, 3.0, 1.5, step=0.5, key="mob_ss_min")
        if st.button("Run Stock Scan", type="primary", key="mob_ss_run"):
            all_results = []
            progress    = st.progress(0, text="Scanning equities...")
            for i, sym in enumerate(SCAN_UNIVERSE):
                r = scan_single_stock(sym, ss_hold, ss_tgt)
                if r and r["Score"] >= ss_min_score: all_results.append(r)
                progress.progress((i+1)/len(SCAN_UNIVERSE), text=f"Scanning {sym}...")
            progress.empty()
            st.session_state["mob_ss_results"] = all_results; st.rerun()
        if "mob_ss_results" in st.session_state:
            results = st.session_state["mob_ss_results"]
            if not results:
                st.warning("No stocks found above minimum score.")
            else:
                top10 = sorted(results, key=lambda x: (x["Score"], x["P(Target)"]), reverse=True)[:10]
                st.success(f"Found {len(results)} stocks. Top {len(top10)} shown.")
                for rank, r in enumerate(top10, 1):
                    ri = REGIME_ICON.get(r["Regime"],"⚪")
                    mom_col = "#00d96f" if "+" in str(r["Momentum"]) else "#ff4b4b"
                    st.markdown(f"""<div class='sc-card'>
                        <div class='sc-hdr'>
                            <div><span class='sc-sym'>#{rank} {r['Symbol']}</span>
                            <span style='font-size:0.62rem;opacity:0.55;margin-left:6px;'>{ri} {r['Regime']}</span></div>
                            {_m_verdict_chip(r['Verdict'])}</div>
                        <div class='sc-body'>
                            <div><span class='sc-lbl'>Price</span>${r['Spot']:.2f}</div>
                            <div><span class='sc-lbl'>IVR</span>{r['IVR']:.0f}</div>
                            <div><span class='sc-lbl'>Score</span>{r['Score']}/3</div>
                            <div><span class='sc-lbl'>Momentum</span><span style='color:{mom_col};'>{r['Momentum']}</span></div>
                            <div><span class='sc-lbl'>SMA21</span>{r['SMA21']}</div>
                            <div><span class='sc-lbl'>P(Target)</span>{r['P(Target)']}</div>
                        </div></div>""", unsafe_allow_html=True)
                    ba, bb, bc = st.columns(3)
                    if ba.button(f"Analyze →", key=f"mob_ss_go_{rank}"):
                        st.session_state.mob_page = "stocks"
                        st.session_state["mob_sa_ticker_val"] = r["Symbol"]
                        st.session_state.pop("mob_sa_data", None)
                        st.rerun()
                    if bb.button(f"BT →", key=f"mob_ss_bt_{rank}", help=f"Backtest {r['Symbol']}"):
                        st.session_state["mob_bt_ticker_input"] = r["Symbol"]
                        st.session_state["mob_bt_hold_input"]   = ss_hold
                        st.session_state["mob_bt_tgt_input"]    = ss_tgt
                        st.session_state["mob_bt_mode_input"]   = "Simple"
                        st.toast(f"{r['Symbol']} loaded into Backtest tab ▸")
                        st.rerun()
                    if bc.button(f"ETF +", key=f"mob_ss_etf_{rank}", help=f"Add {r['Symbol']} to ETF Builder"):
                        if "etf_portfolio" not in st.session_state:
                            st.session_state.etf_portfolio = []
                        sym_etf = r["Symbol"]
                        existing = next((i for i, h in enumerate(st.session_state.etf_portfolio) if h["ticker"] == sym_etf), None)
                        if existing is None:
                            st.session_state.etf_portfolio.append({"ticker": sym_etf, "dollars": 10_000.0})
                            st.toast(f"{sym_etf} added to ETF Builder at $10,000.")
                        else:
                            st.toast(f"{sym_etf} is already in ETF Builder.")
                        st.session_state.pop("mob_etf_results", None)
                fig_sb = go.Figure(go.Bar(
                    x=[r["Symbol"] for r in top10], y=[r["Score"] for r in top10],
                    marker_color=[VERDICT_COLOR.get(r["Verdict"],"#bfa15d") for r in top10]))
                fig_sb.add_hline(y=2.5, line_dash="dot", line_color="#00d96f")
                fig_sb.add_hline(y=1.5, line_dash="dot", line_color="#FFC107")
                fig_sb.update_layout(height=200, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(range=[0,3.4], gridcolor="rgba(255,255,255,0.08)", tickfont=dict(size=8)),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickfont=dict(size=7)),
                    margin=dict(l=0,r=0,t=20,b=0), showlegend=False)
                st.plotly_chart(fig_sb, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Set parameters and click **Run Stock Scan**.")

    with tab_bt:
        # ── SHARED PARAMETERS (pre-fillable from Stock Scan via session state) ──
        bt_ticker = st.text_input(
            "Ticker Symbol",
            key="mob_bt_ticker_input",
        ).upper().strip()

        c1, c2  = st.columns(2)
        bt_hold = c1.slider("Holding (days)", 5, 60,
                            st.session_state.get("mob_bt_hold_input", 21),
                            key="mob_bt_hold_input")
        bt_tgt  = c2.slider("Target Move (%)", 2, 20,
                            st.session_state.get("mob_bt_tgt_input", 5),
                            key="mob_bt_tgt_input")
        bt_sl   = st.slider("Stop Loss (%)", 2, 30, 10, key="mob_bt_sl_input")

        # Mode radio — default from "BT →" button (mob_bt_mode_input) or Simple
        _default_mode = st.session_state.get("mob_bt_mode_input", "Simple")
        bt_mode = st.radio("Mode", ["Simple", "Walk-Forward"], horizontal=True,
                           index=0 if _default_mode == "Simple" else 1,
                           key="mob_bt_mode_radio")

        # ── SIMPLE BACKTEST ──
        if bt_mode == "Simple":
            bt_lookback = st.selectbox("Lookback", [63, 126, 252], index=2,
                                       format_func=lambda x: f"{x}d (~{x//21}mo)", key="mob_bt_lb")
            if st.button("Run Simple Backtest", type="primary", key="mob_bt_run") and bt_ticker:
                with st.spinner(f"Backtesting {bt_ticker}..."):
                    result = run_backtest(bt_ticker, bt_lookback, bt_hold, bt_tgt, bt_sl)
                st.session_state["mob_bt_result"] = result
                st.session_state["mob_bt_result_sym"] = bt_ticker
                st.session_state.pop("mob_wf_result", None)
                st.rerun()

            if "mob_bt_result" in st.session_state:
                bt  = st.session_state["mob_bt_result"]
                sym = st.session_state.get("mob_bt_result_sym", "")
                if not bt or not bt.get("trades"):
                    st.warning("No BUY signals generated. Try a longer lookback.")
                else:
                    s = bt["stats"]
                    st.success(f"**{sym}** — {s['Total Trades']} trades")
                    c1, c2 = st.columns(2)
                    c1.metric("Win Rate",        f"{s['Win Rate']:.1f}%")
                    c2.metric("Profit Factor",   f"{s['Profit Factor']:.2f}")
                    c1.metric("Total Return",    f"{s['Total Return %']:+.1f}%")
                    c2.metric("Target Hit Rate", f"{s['Target Hit Rate']:.1f}%")

                    _m_sh("Equity Curve")
                    eq_df = pd.DataFrame(bt["equity_curve"])
                    if not eq_df.empty:
                        lc  = "#00d96f" if s["Total Return %"] >= 0 else "#ff4b4b"
                        rgb = tuple(int(lc.lstrip('#')[ci:ci+2], 16) for ci in (0, 2, 4))
                        fig_eq = go.Figure()
                        fig_eq.add_trace(go.Scatter(
                            x=eq_df["Date"], y=eq_df["Cumulative P&L %"],
                            mode="lines", line=dict(color=lc, width=2),
                            fill="tozeroy", fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.08)"))
                        fig_eq.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                        fig_eq.update_layout(height=220,
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            yaxis=dict(title="Cum. P&L (%)", gridcolor="rgba(255,255,255,0.08)", tickfont=dict(size=8)),
                            xaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickfont=dict(size=8)),
                            margin=dict(l=0,r=0,t=10,b=0), showlegend=False)
                        st.plotly_chart(fig_eq, use_container_width=True, config={"displayModeBar": False})

                    _m_sh("Performance by Regime")
                    rg_grp = {}
                    for tr in bt["trades"]: rg_grp.setdefault(tr["Regime"], []).append(tr["P&L %"])
                    for rg, ps in rg_grp.items():
                        ri  = REGIME_ICON.get(rg, "⚪"); wr = sum(1 for p in ps if p > 0) / len(ps) * 100
                        avg = np.mean(ps); col = "#00d96f" if avg > 0 else "#ff4b4b"
                        st.markdown(f"""<div class='sc-card'>
                            <div class='sc-hdr'><span class='sc-sym'>{ri} {rg}</span>
                            <span style='font-size:0.7rem;color:{col};font-family:Courier New,monospace;'>{avg:+.2f}% avg</span></div>
                            <div class='sc-body'>
                                <div><span class='sc-lbl'>Trades</span>{len(ps)}</div>
                                <div><span class='sc-lbl'>Win Rate</span>{wr:.0f}%</div>
                                <div><span class='sc-lbl'>Total</span>{sum(ps):+.1f}%</div>
                            </div></div>""", unsafe_allow_html=True)

                    # Add to portfolio actions
                    _m_sh("Actions")
                    pa, pb = st.columns(2)
                    if pa.button("Add to Positions", key="mob_bt_add_pos"):
                        if "positions" not in st.session_state: st.session_state.positions = []
                        st.session_state.positions.append({
                            "Symbol": sym, "Type": "Stock", "Strike": "---",
                            "Expiration": "---", "Entry Premium": 0.0, "Target": 0.0,
                            "Contracts": 100, "Stop Loss": f"{bt_sl}%",
                            "Opened": date.today().isoformat()})
                        st.toast(f"{sym} added to Positions.")
                    if pb.button("Add to ETF Builder", key="mob_bt_add_etf"):
                        if "etf_portfolio" not in st.session_state: st.session_state.etf_portfolio = []
                        existing = next((i for i, h in enumerate(st.session_state.etf_portfolio) if h["ticker"] == sym), None)
                        if existing is None:
                            st.session_state.etf_portfolio.append({"ticker": sym, "dollars": 10_000.0})
                            st.toast(f"{sym} added to ETF Builder at $10,000.")
                        else:
                            st.toast(f"{sym} is already in ETF Builder.")
                        st.session_state.pop("mob_etf_results", None)
            else:
                st.info("Configure parameters above and click **Run Simple Backtest**.")

        # ── WALK-FORWARD ANALYSIS ──
        else:
            st.caption("Anti-overfitting test: rolling IS training + OOS test windows. 5 years of data required.")
            c1, c2  = st.columns(2)
            wf_is   = c1.selectbox("IS Window", [6, 9, 12, 18], index=2,
                                   format_func=lambda x: f"{x} mo", key="mob_wf_is")
            wf_oos  = c2.selectbox("OOS Window", [1, 2, 3, 6], index=2,
                                   format_func=lambda x: f"{x} mo", key="mob_wf_oos")

            if st.button("Run Walk-Forward", type="primary", key="mob_wf_run") and bt_ticker:
                with st.spinner(f"Running walk-forward on {bt_ticker}… (may take ~30s)"):
                    wf_result = run_walkforward(bt_ticker, wf_is, wf_oos, bt_hold, bt_tgt, bt_sl)
                st.session_state["mob_wf_result"]     = wf_result
                st.session_state["mob_wf_result_sym"] = bt_ticker
                st.session_state.pop("mob_bt_result", None)
                st.rerun()

            wf = st.session_state.get("mob_wf_result")
            wf_sym = st.session_state.get("mob_wf_result_sym", "")
            if wf is None:
                st.info("Configure parameters above and click **Run Walk-Forward**.")
            elif wf is False:
                st.warning("Insufficient data for walk-forward. Need at least 5 years of history.")
            else:
                # Summary metrics
                st.success(f"**{wf_sym}** — {wf['n_folds']} OOS folds · {wf['total_oos_trades']} trades")
                c1, c2 = st.columns(2)
                sharpe_col = "#00d96f" if wf["agg_sharpe"] >= 1.0 else ("#FFC107" if wf["agg_sharpe"] >= 0 else "#ff4b4b")
                dd_col = "#00d96f" if wf["worst_fold_dd"] >= -5 else ("#FFC107" if wf["worst_fold_dd"] >= -15 else "#ff4b4b")
                c1.metric("Aggregate OOS Sharpe", f"{wf['agg_sharpe']:.2f}")
                c2.metric("Worst Fold Drawdown",  f"{wf['worst_fold_dd']:+.1f}%")
                c1.metric("OOS Folds",            str(wf["n_folds"]))
                c2.metric("Total OOS Trades",     str(wf["total_oos_trades"]))

                # Fold-by-fold cards
                _m_sh("Fold-by-Fold OOS Results")
                for f in wf["folds"]:
                    ri   = REGIME_ICON.get(f["dominant_regime"], "⚪")
                    ret_col  = "#00d96f" if f["avg_return"] > 0 else "#ff4b4b"
                    sha_col  = "#00d96f" if f["sharpe"] >= 1.0 else ("#FFC107" if f["sharpe"] >= 0 else "#ff4b4b")
                    dd_col2  = "#00d96f" if f["max_dd"] >= -5 else ("#FFC107" if f["max_dd"] >= -15 else "#ff4b4b")
                    st.markdown(f"""<div class='sc-card'>
                        <div class='sc-hdr'>
                            <div><span class='sc-sym'>Fold {f['fold']}</span>
                            <span style='font-size:0.62rem;opacity:0.55;margin-left:6px;'>{f['period']}</span></div>
                            <span style='font-size:0.62rem;'>{ri} {f['dominant_regime'][:3]}</span>
                        </div>
                        <div class='sc-body'>
                            <div><span class='sc-lbl'>Trades</span>{f['trades']}</div>
                            <div><span class='sc-lbl'>Win Rate</span>{f['win_rate']:.0f}%</div>
                            <div><span class='sc-lbl'>Avg Ret</span><span style='color:{ret_col};'>{f['avg_return']:+.2f}%</span></div>
                            <div><span class='sc-lbl'>Max DD</span><span style='color:{dd_col2};'>{f['max_dd']:+.1f}%</span></div>
                            <div><span class='sc-lbl'>Sharpe</span><span style='color:{sha_col};'>{f['sharpe']:.2f}</span></div>
                            <div><span class='sc-lbl'>Regime</span>{ri}</div>
                        </div></div>""", unsafe_allow_html=True)

                # Regime attribution cards
                _m_sh("Regime Attribution")
                for ra in wf["regime_attribution"]:
                    edge_col = "#00d96f" if ra["Edge"] == "✅" else ("#FFC107" if ra["Edge"] == "⚠️" else "#ff4b4b")
                    avg_col  = "#00d96f" if "+" in ra["Avg Return"] else "#ff4b4b"
                    st.markdown(f"""<div class='sc-card'>
                        <div class='sc-hdr'>
                            <span class='sc-sym'>{ra['Regime']}</span>
                            <span style='color:{edge_col};font-size:0.9rem;'>{ra['Edge']}</span>
                        </div>
                        <div class='sc-body'>
                            <div><span class='sc-lbl'>Trades</span>{ra['Trades']}</div>
                            <div><span class='sc-lbl'>Win Rate</span>{ra['Win Rate']}</div>
                            <div><span class='sc-lbl'>Avg Return</span><span style='color:{avg_col};'>{ra['Avg Return']}</span></div>
                            <div><span class='sc-lbl'>Total Return</span>{ra['Total Return']}</div>
                        </div></div>""", unsafe_allow_html=True)

                # AI explanation
                _m_sh("AI Walk-Forward Explanation")
                if st.button("Generate AI Explanation", type="secondary", key="mob_wf_ai"):
                    ra_lines = "; ".join(
                        f"{r['Regime']} avg {r['Avg Return']} edge {r['Edge']}"
                        for r in wf["regime_attribution"]
                    )
                    prompt = (
                        f"Walk-forward test results for {wf_sym}:\n"
                        f"- Out-of-sample Sharpe: {wf['agg_sharpe']:.2f} | Worst drawdown: {wf['worst_fold_dd']:+.1f}%\n"
                        f"- Number of test periods (folds): {wf['n_folds']} | Total trades tested: {wf['total_oos_trades']}\n"
                        f"- Regime results: {ra_lines}\n\n"
                        "Explain in 5 short bullets:\n"
                        "• 📖 What walk-forward testing means — explain it like you're talking to someone who's never heard of it\n"
                        "• ✅ Is this strategy robust? — interpret the results honestly\n"
                        "• 🧠 Regime breakdown — which market conditions did this strategy thrive or struggle in?\n"
                        "• 📊 What the Sharpe Ratio means here — define it simply and say whether this number is good\n"
                        "• 💡 One practical next step — what should the investor do with this information?"
                    )
                    _stream_ai_response(prompt, max_tokens=600, placeholder=st.empty(), system=_ADVISOR_SYSTEM)

                # Add to portfolio actions
                _m_sh("Actions")
                pa, pb = st.columns(2)
                if pa.button("Add to Positions", key="mob_wf_add_pos"):
                    if "positions" not in st.session_state: st.session_state.positions = []
                    st.session_state.positions.append({
                        "Symbol": wf_sym, "Type": "Stock", "Strike": "---",
                        "Expiration": "---", "Entry Premium": 0.0, "Target": 0.0,
                        "Contracts": 100, "Stop Loss": f"{bt_sl}%",
                        "Opened": date.today().isoformat()})
                    st.toast(f"{wf_sym} added to Positions.")
                if pb.button("Add to ETF Builder", key="mob_wf_add_etf"):
                    if "etf_portfolio" not in st.session_state: st.session_state.etf_portfolio = []
                    existing = next((i for i, h in enumerate(st.session_state.etf_portfolio) if h["ticker"] == wf_sym), None)
                    if existing is None:
                        st.session_state.etf_portfolio.append({"ticker": wf_sym, "dollars": 10_000.0})
                        st.toast(f"{wf_sym} added to ETF Builder at $10,000.")
                    else:
                        st.toast(f"{wf_sym} is already in ETF Builder.")
                    st.session_state.pop("mob_etf_results", None)


# ==========================================
# SECTION 7e: render_portfolio()
# ==========================================
def render_portfolio():
    _m_sh("Portfolio")
    tab_pos, tab_etf = st.tabs(["Positions", "ETF Builder"])

    # ── POSITIONS ──
    with tab_pos:
        positions = st.session_state.get("positions", [])
        with st.expander("Add New Position", expanded=not positions):
            p_sym    = st.text_input("Symbol", key="mob_pos_sym").upper().strip()
            p_type   = st.selectbox("Type", ["Call","Put","Stock"], key="mob_pos_type")
            c1, c2   = st.columns(2)
            p_strike  = c1.number_input("Strike/Entry ($)", min_value=0.0, step=0.5, key="mob_pos_strike")
            p_premium = c2.number_input("Entry Premium ($)", min_value=0.0, step=0.01, key="mob_pos_prem")
            c3, c4   = st.columns(2)
            p_expiry  = c3.date_input("Expiry", key="mob_pos_expiry")
            p_target  = c4.number_input("Target ($)", min_value=0.0, step=0.5, key="mob_pos_target")
            c5, c6   = st.columns(2)
            p_contr   = c5.number_input("Qty/Contracts", min_value=1, value=1, key="mob_pos_contr")
            p_sl      = c6.slider("Stop Loss %", 0, 100, 20, key="mob_pos_sl")
            if st.button("Add Position", type="primary", key="mob_add_pos") and p_sym and p_premium > 0:
                st.session_state.positions.append({"Symbol": p_sym, "Type": p_type, "Strike": p_strike,
                    "Expiration": str(p_expiry), "Entry Premium": p_premium, "Target": p_target,
                    "Contracts": p_contr, "Stop Loss": f"{p_sl}%", "Opened": date.today().isoformat()})
                st.success(f"Added {p_sym}"); st.rerun()

        if not positions:
            st.info("No positions. Add one above or use 'Save to Positions' in the Trade tab.")
        else:
            if st.button("Refresh P&L", type="secondary", key="mob_pos_refresh"):
                enriched = []
                with st.spinner("Fetching current prices..."):
                    for pos in positions:
                        curr = pos["Entry Premium"]; pnl = 0.0; roi = 0.0
                        try:
                            if pos["Type"] == "Stock":
                                curr = yf.Ticker(pos["Symbol"]).history(period="1d")["Close"].iloc[-1]
                            else:
                                t = yf.Ticker(pos["Symbol"])
                                chain_df = t.option_chain(pos["Expiration"])
                                chain_df = chain_df.calls if pos["Type"] == "Call" else chain_df.puts
                                row_p = chain_df[chain_df["strike"] == pos["Strike"]]
                                if not row_p.empty: curr = row_p["lastPrice"].iloc[0]
                            mult = 1 if pos["Type"] == "Stock" else 100
                            pnl  = (curr - pos["Entry Premium"]) * pos["Contracts"] * mult
                            roi  = ((curr - pos["Entry Premium"]) / pos["Entry Premium"] * 100) if pos["Entry Premium"] > 0 else 0
                        except Exception: pass
                        enriched.append({**pos, "Current": round(curr, 2), "P&L": round(pnl, 2), "ROI%": round(roi, 1)})
                st.session_state["mob_enriched_pos"] = enriched

            display_pos = st.session_state.get("mob_enriched_pos", positions)
            for pos in display_pos:
                pnl  = pos.get("P&L", 0); roi = pos.get("ROI%", 0)
                curr = pos.get("Current", pos.get("Entry Premium"))
                col  = "#00d96f" if pnl >= 0 else "#ff4b4b"
                pnl_s = f"${pnl:+,.2f} ({roi:+.1f}%)" if "P&L" in pos else "—"
                earn = fetch_earnings_calendar(pos.get("Symbol","")) if pos.get("Type") != "Stock" else None
                earn_warn = ""
                if earn and 0 <= earn.get("days_away", 99) <= 14:
                    earn_warn = f" <span style='color:#FFC107;font-size:0.58rem;'>⚠ Earnings {earn['days_away']}d</span>"
                st.markdown(f"""<div class='sc-card'>
                    <div class='sc-hdr'>
                        <div><span class='sc-sym'>{pos['Symbol']}</span> <span style='font-size:0.65rem;opacity:0.55;'>{pos['Type']}</span>{earn_warn}</div>
                        <span style='font-family:Courier New,monospace;font-size:0.75rem;color:{col};font-weight:700;'>{pnl_s}</span>
                    </div>
                    <div class='sc-body'>
                        <div><span class='sc-lbl'>Entry</span>${pos['Entry Premium']:.2f}</div>
                        <div><span class='sc-lbl'>Current</span>${curr:.2f}</div>
                        <div><span class='sc-lbl'>Qty</span>{pos['Contracts']}</div>
                        <div><span class='sc-lbl'>Strike</span>{pos['Strike']}</div>
                        <div><span class='sc-lbl'>Expiry</span>{str(pos['Expiration'])[:10]}</div>
                        <div><span class='sc-lbl'>SL</span>{pos.get('Stop Loss','—')}</div>
                    </div></div>""", unsafe_allow_html=True)
            if st.button("Clear All Positions", key="mob_clear_pos"):
                st.session_state.positions = []; st.session_state.pop("mob_enriched_pos", None); st.rerun()

    # ── ETF BUILDER ──
    with tab_etf:
        if "etf_portfolio" not in st.session_state: st.session_state.etf_portfolio = []
        for h in st.session_state.etf_portfolio:
            if "pct" in h and "dollars" not in h:
                h["dollars"] = h["pct"] * 1000.0; del h["pct"]

        _m_sh("Portfolio Construction")
        fc1, fc2 = st.columns([2, 1])
        new_etf_ticker  = fc1.text_input("Ticker", placeholder="e.g. AAPL", key="mob_etf_ticker").upper().strip()
        new_etf_dollars = fc2.number_input("$ Amount", min_value=0.01, value=25000.0, step=1000.0, key="mob_etf_dollars")
        if st.button("Add Security", type="primary", key="mob_etf_add") and new_etf_ticker:
            existing = next((i for i, h in enumerate(st.session_state.etf_portfolio) if h["ticker"] == new_etf_ticker), None)
            if existing is not None:
                st.session_state.etf_portfolio[existing]["dollars"] = new_etf_dollars
                st.toast(f"Updated {new_etf_ticker}.")
            else:
                st.session_state.etf_portfolio.append({"ticker": new_etf_ticker, "dollars": new_etf_dollars})
                st.toast(f"Added {new_etf_ticker}.")
            st.session_state.pop("mob_etf_results", None); st.rerun()

        if not st.session_state.etf_portfolio:
            st.info("Add securities above to build your ETF. Example: AAPL $25,000 + MSFT $25,000 + NVDA $25,000")
        else:
            total_d = sum(h["dollars"] for h in st.session_state.etf_portfolio)
            _key_gen = st.session_state.get("mob_etf_key_gen", 0)
            _changed = False
            for i, h in enumerate(st.session_state.etf_portfolio):
                row = st.columns([1.4, 2.5, 1.2, 0.8])
                row[0].markdown(f"<div style='padding:9px 0;font-weight:700;color:#bfa15d;font-family:Courier New,monospace;'>{h['ticker']}</div>", unsafe_allow_html=True)
                new_amt = row[1].number_input("amt", min_value=0.01, value=max(0.01, float(h["dollars"])),
                                               step=1000.0, format="%.0f", key=f"mob_etf_amt_{h['ticker']}_{_key_gen}", label_visibility="collapsed")
                if abs(new_amt - h["dollars"]) > 0.01:
                    st.session_state.etf_portfolio[i]["dollars"] = new_amt
                    st.session_state.pop("mob_etf_results", None); _changed = True
                row[2].markdown(f"<div style='padding:9px 0;color:#bfa15d;font-size:0.78rem;'>{h['dollars']/total_d*100:.1f}%</div>", unsafe_allow_html=True)
                if row[3].button("✕", key=f"mob_etf_rm_{i}"):
                    st.session_state.etf_portfolio.pop(i); st.session_state.pop("mob_etf_results", None); st.rerun()
            if _changed: st.rerun()
            st.markdown(f"<div style='margin:8px 0;padding:8px 14px;border:1px solid rgba(191,161,93,0.3);border-radius:6px;display:flex;justify-content:space-between;'><span style='color:#bfa15d;font-weight:700;'>{len(st.session_state.etf_portfolio)} securities</span><span style='font-weight:700;'>${total_d:,.0f} total</span></div>", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            etf_horizon = c1.number_input("Horizon (years)", min_value=1, max_value=30, value=10, key="mob_etf_horizon")
            if c2.button("Clear All", type="secondary", key="mob_etf_clear"):
                st.session_state.etf_portfolio = []; st.session_state.pop("mob_etf_results", None); st.rerun()

            if st.button("Analyze ETF Portfolio", type="primary", key="mob_etf_analyze",
                         disabled=len(st.session_state.etf_portfolio) == 0):
                with st.spinner("Computing institutional metrics..."):
                    portfolio_snap = {h["ticker"]: h["dollars"] for h in st.session_state.etf_portfolio}
                    tickers_snap   = list(portfolio_snap.keys())
                    try:
                        today_str = pd.Timestamp.today().strftime("%Y-%m-%d")
                        raw       = yf.download(tickers_snap, start="2015-01-01", end=today_str, progress=False)
                        prices    = normalize_price_frame(raw)
                        if "SINGLE_TICKER" in prices.columns and len(tickers_snap) == 1:
                            prices = prices.rename(columns={"SINGLE_TICKER": tickers_snap[0]})
                        valid_cols  = [t for t in tickers_snap if t in prices.columns]
                        if not valid_cols:
                            st.error("No valid price data. Check ticker symbols.")
                        else:
                            dollars_arr = np.array([portfolio_snap[t] for t in valid_cols], dtype=float)
                            w_arr       = dollars_arr / dollars_arr.sum()
                            returns     = prices[valid_cols].pct_change().fillna(0.0)
                            port_return = (returns * w_arr).sum(axis=1)
                            spy_raw     = yf.download("SPY", start="2015-01-01", end=today_str, progress=False)
                            spy_prices  = normalize_price_frame(spy_raw)
                            if "SINGLE_TICKER" in spy_prices.columns:
                                spy_prices = spy_prices.rename(columns={"SINGLE_TICKER":"SPY"})
                            spy_ret = spy_prices["SPY"].pct_change().fillna(0.0) if "SPY" in spy_prices.columns else pd.Series(dtype=float)
                            rf_daily = get_risk_free_daily()
                            st.session_state["mob_etf_results"] = {
                                "portfolio": portfolio_snap, "prices": prices, "returns": returns,
                                "port_return": port_return, "valid_cols": valid_cols, "weights": w_arr,
                                "dollars_arr": dollars_arr, "rf_daily": rf_daily,
                                "spy_ret": spy_ret, "spy_prices": spy_prices,
                                "total_invested": dollars_arr.sum(), "horizon": etf_horizon,
                            }
                            st.success(f"Analysis complete for {len(valid_cols)} securities.")
                    except Exception as e:
                        st.error(f"Error: {e}")

        if "mob_etf_results" in st.session_state:
            res         = st.session_state["mob_etf_results"]
            port_return = res["port_return"]; valid_cols = res["valid_cols"]
            w           = res["weights"]; dollars_arr = res["dollars_arr"]
            rf_daily    = res["rf_daily"]; spy_ret = res["spy_ret"]
            spy_prices  = res["spy_prices"]; prices = res["prices"]
            total_inv   = res["total_invested"]; horizon = res["horizon"]

            excess      = port_return - rf_daily; vol_p = port_return.std(ddof=1)
            sharpe      = (excess.mean() / vol_p * np.sqrt(252)) if vol_p > 0 else float("nan")
            neg_ret     = port_return[port_return < 0]; downside = neg_ret.std(ddof=1) if len(neg_ret) > 1 else float("nan")
            sortino     = (excess.mean() / downside * np.sqrt(252)) if not (np.isnan(downside) or downside == 0) else float("nan")
            cumval      = (1 + port_return).cumprod(); roll_max = cumval.expanding().max()
            max_dd      = float(((cumval - roll_max) / roll_max).min()) * 100
            var_95      = float(np.percentile(port_return, 5)) * total_inv
            beta        = float("nan"); alpha_ann = float("nan"); alpha_daily = 0.0
            if len(spy_ret) > 10:
                aligned = pd.concat([port_return, spy_ret], axis=1).dropna(); aligned.columns = ["P","SPY"]
                if len(aligned) > 10:
                    reg = LinearRegression().fit(aligned["SPY"].values.reshape(-1,1), aligned["P"].values)
                    beta = float(reg.coef_[0]); alpha_daily = float(reg.intercept_); alpha_ann = alpha_daily * 252 * 100
            cagr_map    = {t: etf_get_cagr(prices, t, 10) for t in valid_cols}
            fv_map      = {}
            for i2, t in enumerate(valid_cols):
                cagr_t = cagr_map[t]; inv_t = float(dollars_arr[i2])
                fv_map[t] = npf.fv(rate=cagr_t, nper=horizon, pmt=0, pv=-inv_t) if not np.isnan(cagr_t) else float("nan")
            total_fv = sum(v for v in fv_map.values() if not np.isnan(v))

            _m_sh("Institutional Risk Metrics")
            c1, c2 = st.columns(2)
            c1.metric("Sharpe Ratio",  f"{sharpe:.3f}" if not np.isnan(sharpe) else "N/A")
            c2.metric("Sortino Ratio", f"{sortino:.3f}" if not np.isnan(sortino) else "N/A")
            c1.metric("Beta vs SPY",   f"{beta:.3f}" if not np.isnan(beta) else "N/A")
            c2.metric("Alpha (ann.)",  f"{alpha_ann:.2f}%" if not np.isnan(alpha_ann) else "N/A")
            c1.metric("Max Drawdown",  f"{max_dd:.2f}%")
            c2.metric("VaR 95% (1d)", f"${abs(var_95):,.0f}")

            _m_sh("Portfolio Weights")
            colors = px.colors.qualitative.Bold[:len(valid_cols)]
            fig_pie = go.Figure(go.Pie(
                labels=valid_cols, values=[float(dollars_arr[i2]) for i2 in range(len(valid_cols))],
                hole=0.4, marker=dict(colors=colors), textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<br>%{percent}<extra></extra>"))
            fig_pie.update_layout(height=240, paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0), showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

            _m_sh("Historical Growth vs SPY")
            port_cum = (1 + port_return).cumprod() * total_inv
            spy_al   = spy_ret.reindex(port_return.index).fillna(0.0)
            spy_cum  = (1 + spy_al).cumprod() * total_inv
            fig_g    = go.Figure()
            fig_g.add_trace(go.Scatter(x=port_cum.index, y=port_cum, mode="lines",
                name="Portfolio", line=dict(color="#bfa15d", width=2)))
            fig_g.add_trace(go.Scatter(x=spy_cum.index, y=spy_cum, mode="lines",
                name="SPY", line=dict(color="#7b68ee", width=1.5, dash="dash")))
            fig_g.update_layout(height=240, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0,r=0,t=10,b=0), hovermode="x unified",
                yaxis=dict(title="Value ($)", gridcolor="rgba(255,255,255,0.08)", tickfont=dict(size=8)),
                xaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickfont=dict(size=8)),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                            bgcolor="rgba(0,0,0,0)", font=dict(size=8)))
            st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar": False})

            _m_sh(f"CAGR & {horizon}-Year Projection")
            for i2, ticker in enumerate(valid_cols):
                cagr_t = cagr_map[ticker]; inv_t = float(dollars_arr[i2]); fv_t = fv_map[ticker]
                gain_t = fv_t - inv_t if not np.isnan(fv_t) else float("nan")
                mult_t = fv_t / inv_t if (not np.isnan(fv_t) and inv_t > 0) else float("nan")
                fv_s   = f"${fv_t:,.0f}" if not np.isnan(fv_t) else "N/A"
                cagr_s = f"{cagr_t*100:.1f}%" if not np.isnan(cagr_t) else "N/A"
                mult_s = f"{mult_t:.2f}x" if not np.isnan(mult_t) else "N/A"
                st.markdown(f"""<div class='bl-row'>
                    <span class='bl-sym'>{ticker}</span>
                    <span class='bl-name'>{w[i2]*100:.1f}% · ${inv_t:,.0f} · CAGR {cagr_s}</span>
                    <span class='bl-price' style='font-size:0.76rem;'>{fv_s}</span>
                    <span class='bl-chg bl-pos' style='font-size:0.7rem;'>{mult_s}</span>
                </div>""", unsafe_allow_html=True)
            st.markdown(f'<div style="padding:8px 0;border-top:1px solid rgba(191,161,93,0.25);display:flex;justify-content:space-between;"><span style="color:#bfa15d;font-weight:700;">Total ${total_inv:,.0f} → ${total_fv:,.0f}</span><span style="color:#00d96f;font-family:Courier New,monospace;">{(total_fv/total_inv-1)*100:.1f}% gain</span></div>', unsafe_allow_html=True)

            # Kelly (collapsed)
            with st.expander("Optimal Allocation — Kelly Criterion"):
                kelly_rows = []; kelly_raw_fracs = {}
                for t in valid_cols:
                    daily_r = res["returns"][t].dropna()
                    if len(daily_r) < 50: kelly_raw_fracs[t] = 0.0; continue
                    pos_r = daily_r[daily_r > 0]; neg_r = daily_r[daily_r < 0]
                    win_rate = len(pos_r) / len(daily_r)
                    avg_win  = float(pos_r.mean()) if len(pos_r) > 0 else 0.0
                    avg_loss = float(abs(neg_r.mean())) if len(neg_r) > 0 else 1e-9
                    b        = avg_win / avg_loss if avg_loss > 0 else 0.0
                    q        = 1 - win_rate
                    k_full   = (b * win_rate - q) / b if b > 0 else -1.0
                    k_half   = max(0.0, k_full * 0.5)
                    kelly_raw_fracs[t] = k_half
                    kelly_rows.append({"Ticker": t, "Win Rate": f"{win_rate*100:.1f}%",
                                       "Half Kelly": f"{k_half*100:.2f}%"})
                frac_total = sum(kelly_raw_fracs.values())
                kelly_pct  = {t: (kelly_raw_fracs[t]/frac_total*100) if frac_total > 0 else (100/len(valid_cols)) for t in valid_cols}
                kelly_dol  = {t: kelly_pct[t]/100*total_inv for t in valid_cols}
                for row_k in kelly_rows:
                    t = row_k["Ticker"]
                    row_k["Suggested %"] = f"{kelly_pct.get(t,0):.1f}%"
                    row_k["Suggested $"] = f"${kelly_dol.get(t,0):,.0f}"
                if kelly_rows: st.dataframe(pd.DataFrame(kelly_rows), hide_index=True, use_container_width=True)
                c_k1, c_k2 = st.columns(2)
                if c_k1.button("Apply Kelly", type="primary", key="mob_etf_apply_kelly"):
                    st.session_state["mob_etf_pre_kelly"] = [dict(h) for h in st.session_state.etf_portfolio]
                    for i3, h in enumerate(st.session_state.etf_portfolio):
                        if h["ticker"] in kelly_dol:
                            st.session_state.etf_portfolio[i3]["dollars"] = round(max(0.01, kelly_dol[h["ticker"]]), 2)
                    st.session_state["mob_etf_key_gen"] = st.session_state.get("mob_etf_key_gen", 0) + 1
                    st.session_state.pop("mob_etf_results", None); st.toast("Kelly applied."); st.rerun()
                if "mob_etf_pre_kelly" in st.session_state:
                    if c_k2.button("Revert", type="secondary", key="mob_etf_revert_kelly"):
                        st.session_state.etf_portfolio = st.session_state.pop("mob_etf_pre_kelly")
                        st.session_state["mob_etf_key_gen"] = st.session_state.get("mob_etf_key_gen", 0) + 1
                        st.session_state.pop("mob_etf_results", None); st.toast("Reverted."); st.rerun()

            _m_sh("AI Portfolio Analysis")
            if st.button("Generate Institutional Analysis", type="secondary", key="mob_etf_ai"):
                if _get_anthropic_client():
                    generate_etf_analysis(res["portfolio"], sharpe, sortino, beta, alpha_ann,
                                          max_dd, var_95, total_inv, total_fv, horizon, placeholder=st.empty())
                else:
                    st.warning("Add `ANTHROPIC_API_KEY` to `.streamlit/secrets.toml` to enable AI.")


# ==========================================
# SECTION 8: MAIN DISPATCH
# ==========================================
_init_state()
inject_mobile_css()

# KERN. compact header
st.markdown("""<div class='mob-header'>
    <div>
        <div class='mob-logo'>KERN.</div>
        <div class='mob-app-name'>MyQuant &nbsp;·&nbsp; Institutional Mobile</div>
    </div>
    <span class='mob-inst-badge'>INSTITUTIONAL</span>
</div>""", unsafe_allow_html=True)

render_compact_ticker_strip()
render_nav(st.session_state.mob_page)

_page = st.session_state.mob_page
if   _page == "home":      render_home()
elif _page == "trade":     render_trade()
elif _page == "stocks":    render_stocks()
elif _page == "scanner":   render_scanner()
elif _page == "portfolio": render_portfolio()

# streamlit run MyQuant_Mobile.py
