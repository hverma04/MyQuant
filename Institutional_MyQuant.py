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
# 1. FEAR Z BEHAVIORAL ENGINE
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


# ==========================================
# 2. MATH ENGINE
# ==========================================
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
        theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        delta = norm.cdf(d1) - 1
        rho   = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
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
        note = "Negative Kelly — edge is insufficient. Reduce position size or skip."
    elif kelly_raw > 0.25:
        note = f"Full Kelly ({kelly_raw:.1%}) is aggressive. Half-Kelly applied: {kelly_frac:.1%}."
    else:
        note = f"Kelly suggests {kelly_frac:.1%} of account (half-Kelly)."
    return {"kelly_pct": round(kelly_frac * 100, 2),
            "recommended_dollars": round(recommended, 2),
            "kelly_raw": round(kelly_raw * 100, 2),
            "note": note}


def trade_advisor_verdict(ev, premium, bs_fair_value, regime):
    score = 0.0
    rules = []
    if ev > 0:
        score += 1.0
        rules.append({"rule": "Expected Value", "result": "Pass", "detail": f"EV = ${ev:.2f} (positive)"})
    elif ev > -50:
        score += 0.5
        rules.append({"rule": "Expected Value", "result": "Warn", "detail": f"EV = ${ev:.2f} (marginal)"})
    else:
        rules.append({"rule": "Expected Value", "result": "Fail", "detail": f"EV = ${ev:.2f} (negative)"})
    if bs_fair_value > 0:
        pct_over = (premium - bs_fair_value) / bs_fair_value
        if pct_over > 0.20:
            rules.append({"rule": "IV vs Black-Scholes", "result": "Fail",
                          "detail": f"Premium {pct_over*100:.1f}% over BS fair value"})
        elif pct_over > 0.05:
            score += 0.5
            rules.append({"rule": "IV vs Black-Scholes", "result": "Warn",
                          "detail": f"Premium {pct_over*100:.1f}% over BS fair value"})
        else:
            score += 1.0
            rules.append({"rule": "IV vs Black-Scholes", "result": "Pass",
                          "detail": f"Premium within {abs(pct_over)*100:.1f}% of fair value"})
    else:
        score += 0.5
        rules.append({"rule": "IV vs Black-Scholes", "result": "Warn", "detail": "Could not compute fair value"})
    if regime == 'Episodic':
        score += 1.0
        rules.append({"rule": "Fear Z Regime", "result": "Pass", "detail": "Episodic — low behavioral risk"})
    elif regime == 'Structural':
        score += 0.5
        rules.append({"rule": "Fear Z Regime", "result": "Warn", "detail": "Structural — moderate Panic Plateau"})
    else:
        rules.append({"rule": "Fear Z Regime", "result": "Fail", "detail": "Systemic — crisis state, IV inflated"})
    verdict = "BUY" if score >= 2.5 else ("HOLD" if score >= 1.5 else "SELL")
    return score, verdict, rules


def stock_advisor_verdict(momentum_5d, price_vs_sma, ivr, regime):
    score = 0.0
    rules = []
    combined  = (momentum_5d * 100) + (price_vs_sma * 100)
    sma_label = "above" if price_vs_sma >= 0 else "below"
    if combined > 3:
        score += 1.0
        rules.append({"rule": "Momentum", "result": "Pass",
                      "detail": f"5d return {momentum_5d*100:+.1f}%, price {abs(price_vs_sma)*100:.1f}% {sma_label} 21-SMA"})
    elif combined > 0:
        score += 0.5
        rules.append({"rule": "Momentum", "result": "Warn",
                      "detail": f"5d return {momentum_5d*100:+.1f}%, marginal trend vs 21-SMA"})
    else:
        rules.append({"rule": "Momentum", "result": "Fail",
                      "detail": f"5d return {momentum_5d*100:+.1f}%, price {sma_label} 21-SMA — bearish"})
    if ivr < 40:
        score += 1.0
        rules.append({"rule": "Volatility Regime", "result": "Pass",
                      "detail": f"IVR {ivr:.0f} — calm environment, favorable for directional moves"})
    elif ivr < 70:
        score += 0.5
        rules.append({"rule": "Volatility Regime", "result": "Warn",
                      "detail": f"IVR {ivr:.0f} — elevated vol, uncertain conditions"})
    else:
        rules.append({"rule": "Volatility Regime", "result": "Fail",
                      "detail": f"IVR {ivr:.0f} — high vol, risk-off environment"})
    if regime == 'Episodic':
        score += 1.0
        rules.append({"rule": "Fear Z Regime", "result": "Pass", "detail": "Episodic — contained vol, normal behavior"})
    elif regime == 'Structural':
        score += 0.5
        rules.append({"rule": "Fear Z Regime", "result": "Warn", "detail": "Structural — moderate stress, watch catalysts"})
    else:
        rules.append({"rule": "Fear Z Regime", "result": "Fail", "detail": "Systemic — crisis state, preservation over growth"})
    verdict = "BUY" if score >= 2.5 else ("HOLD" if score >= 1.5 else "SELL")
    return score, verdict, rules


# ==========================================
# 3a. OPTIONS DATA FETCHING
# ==========================================
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
        return {"income": t.income_stmt, "balance": t.balance_sheet,
                "cashflow": t.cashflow, "info": t.info}
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
        if cal is None:
            return None
        if hasattr(cal, 'columns'):
            if 'Earnings Date' in cal.columns:
                raw = cal['Earnings Date'].iloc[0]
            elif 'Earnings Date' in cal.index:
                raw = cal.loc['Earnings Date'].iloc[0]
            else:
                return None
        elif isinstance(cal, dict):
            raw = cal.get('Earnings Date', [None])[0]
        else:
            return None
        if raw is None:
            return None
        earn_date = pd.to_datetime(raw).date()
        days_away = (earn_date - date.today()).days
        return {"date": earn_date, "days_away": days_away}
    except Exception:
        return None

def _fin_val(df, *row_keys):
    if df is None or df.empty:
        return None
    for key in row_keys:
        matches = [i for i in df.index if key.lower().replace(" ", "") in str(i).lower().replace(" ", "")]
        if matches:
            try:
                v = df.loc[matches[0]].iloc[0]
                if pd.notna(v):
                    return float(v)
            except Exception:
                continue
    return None

def _fin_val2(df, *row_keys):
    if df is None or df.empty or df.shape[1] < 2:
        return None, None
    for key in row_keys:
        matches = [i for i in df.index if key.lower().replace(" ", "") in str(i).lower().replace(" ", "")]
        if matches:
            try:
                v0 = df.loc[matches[0]].iloc[0]
                v1 = df.loc[matches[0]].iloc[1]
                if pd.notna(v0) and pd.notna(v1):
                    return float(v0), float(v1)
            except Exception:
                continue
    return None, None

def score_fundamentals(fin):
    if fin is None:
        return None
    inc  = fin.get("income")
    bal  = fin.get("balance")
    cf   = fin.get("cashflow")
    info = fin.get("info", {}) or {}
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
    ic = abs(ebit / int_exp)    if ebit and int_exp and int_exp != 0 else None
    qr = (cur_assets - inventory) / cur_liab if cur_assets and cur_liab and cur_liab != 0 else None

    if de is None:
        tot_liab   = _fin_val(bal, "TotalLiabilities", "TotalLiabilitiesNetMinorityInterest")
        tot_assets = _fin_val(bal, "TotalAssets")
        if tot_liab is not None and tot_assets and tot_assets != 0:
            de = tot_liab / tot_assets
    if ic is None and (tot_debt is None or tot_debt == 0):
        ic = float("inf")
    _qr_est = False
    if qr is None:
        cash = _fin_val(bal, "CashAndCashEquivalents", "Cash", "CashCashEquivalentsAndShortTermInvestments")
        recv = _fin_val(bal, "NetReceivables", "AccountsReceivable", "Receivables") or 0
        if cash is not None and cur_liab:
            qr = (cash + recv) / cur_liab
            _qr_est = True
    _ic_est = int_exp is None and ic is not None
    results["health"]["Current Ratio"]     = (cr, (1.0 if cr and cr >= 2.0 else 0.5 if cr and cr >= 1.2 else 0.0), "≥2.0 strong / ≥1.2 ok")
    results["health"]["Debt/Equity"]       = (de, (1.0 if de is not None and de <= 0.5 else 0.5 if de is not None and de <= 1.0 else 0.0), "≤0.5 strong / ≤1.0 ok")
    results["health"]["Interest Coverage"] = (ic, (1.0 if ic and ic >= 5.0 else 0.5 if ic and ic >= 3.0 else 0.0), "≥5x strong / ≥3x ok" + (" (est.)" if _ic_est else ""))
    results["health"]["Quick Ratio"]       = (qr, (1.0 if qr and qr >= 1.0 else 0.5 if qr and qr >= 0.7 else 0.0), "≥1.0 strong / ≥0.7 ok" + (" (est.)" if _qr_est else ""))
    results["raw"].update({"Current Ratio": cr, "Debt/Equity": de, "Interest Coverage": ic, "Quick Ratio": qr})

    net_income = _fin_val(inc, "NetIncome", "Net Income")
    rev        = _fin_val(inc, "TotalRevenue", "Total Revenue")
    gross      = _fin_val(inc, "GrossProfit", "Gross Profit")
    op_cf      = _fin_val(cf, "OperatingCashFlow", "Operating Cash Flow", "Cash Flow From Continuing Operating Activities")
    capex      = _fin_val(cf, "CapitalExpenditure", "Capital Expenditure") or 0

    roe      = net_income / equity if net_income and equity and equity != 0 else None
    net_marg = net_income / rev if net_income and rev and rev != 0 else None
    gr_marg  = gross / rev if gross and rev and rev != 0 else None
    fcf      = (op_cf + capex) if op_cf is not None else None
    eq_qual  = op_cf / net_income if op_cf and net_income and net_income != 0 else None

    _roe_est = False
    if roe is None:
        _info_roe = info.get("returnOnEquity")
        if _info_roe is not None:
            try:
                roe = float(_info_roe); _roe_est = True
            except (TypeError, ValueError):
                pass
    _fcf_est = False
    if fcf is None and net_income is not None:
        dep = _fin_val(cf, "Depreciation", "DepreciationAndAmortization", "DepreciationAmortizationDepletion") or 0
        fcf = net_income + dep; _fcf_est = True
    _nm_est = False
    if net_marg is None and rev:
        ebitda = _fin_val(inc, "EBITDA", "NormalizedEBITDA")
        if ebitda and rev:
            net_marg = ebitda / rev; _nm_est = True

    results["quality"]["ROE"]              = (roe,      (1.0 if roe and roe >= 0.20 else 0.5 if roe and roe >= 0.12 else 0.0),           "≥20% strong / ≥12% ok" + (" (est.)" if _roe_est else ""))
    results["quality"]["Net Margin"]       = (net_marg, (1.0 if net_marg and net_marg >= 0.15 else 0.5 if net_marg and net_marg >= 0.07 else 0.0), "≥15% strong / ≥7% ok" + (" (est.)" if _nm_est else ""))
    results["quality"]["Gross Margin"]     = (gr_marg,  (1.0 if gr_marg and gr_marg >= 0.40 else 0.5 if gr_marg and gr_marg >= 0.25 else 0.0),    "≥40% strong / ≥25% ok")
    results["quality"]["Free Cash Flow"]   = (fcf,      (1.0 if fcf is not None and fcf > 0 else 0.0),                                            "Positive = pass" + (" (est.)" if _fcf_est else ""))
    results["quality"]["Earnings Quality"] = (eq_qual,  (1.0 if eq_qual and eq_qual >= 1.0 else 0.5 if eq_qual and eq_qual >= 0.8 else 0.0),       "OCF/NI ≥1.0 strong / ≥0.8 ok")
    results["raw"].update({"ROE": roe, "Net Margin": net_marg, "Gross Margin": gr_marg, "FCF": fcf, "Earnings Quality": eq_qual})

    rev0, rev1     = _fin_val2(inc, "TotalRevenue", "Total Revenue")
    gross0, gross1 = _fin_val2(inc, "GrossProfit", "Gross Profit")
    op_inc0        = _fin_val(inc, "OperatingIncome", "EBIT")
    eps0, eps1     = _fin_val2(inc, "DilutedEPS", "BasicEPS", "Diluted EPS", "Basic EPS")

    rev_gr   = (rev0 - rev1) / abs(rev1) if rev0 and rev1 and rev1 != 0 else None
    gm0      = gross0 / rev0 if gross0 and rev0 and rev0 != 0 else None
    gm1      = gross1 / rev1 if gross1 and rev1 and rev1 != 0 else None
    gm_trend = (gm0 - gm1) if gm0 is not None and gm1 is not None else None
    op_marg  = op_inc0 / rev0 if op_inc0 and rev0 and rev0 != 0 else None
    eps_gr   = (eps0 - eps1) / abs(eps1) if eps0 is not None and eps1 is not None and eps1 != 0 else None

    _epsgr_est = False
    if eps_gr is None:
        ni0 = net_income
        _, ni1 = _fin_val2(inc, "NetIncome", "Net Income")
        if ni0 and ni1 and ni1 != 0:
            eps_gr = (ni0 - ni1) / abs(ni1); _epsgr_est = True

    results["growth"]["Revenue Growth"]     = (rev_gr,   (1.0 if rev_gr and rev_gr >= 0.15 else 0.5 if rev_gr and rev_gr >= 0.07 else 0.0),     "≥15% strong / ≥7% ok")
    results["growth"]["Gross Margin Trend"] = (gm_trend, (1.0 if gm_trend and gm_trend > 0.01 else 0.5 if gm_trend and gm_trend >= -0.01 else 0.0), "Expanding strong / Flat ok")
    results["growth"]["Operating Margin"]   = (op_marg,  (1.0 if op_marg and op_marg >= 0.15 else 0.5 if op_marg and op_marg >= 0.08 else 0.0),  "≥15% strong / ≥8% ok")
    results["growth"]["EPS Growth"]         = (eps_gr,   (1.0 if eps_gr is not None and eps_gr >= 0.10 else 0.5 if eps_gr is not None and eps_gr >= 0 else 0.0), "≥10% strong / ≥0% ok" + (" (est.)" if _epsgr_est else ""))
    results["raw"].update({"Revenue Growth": rev_gr, "GM Trend": gm_trend, "Operating Margin": op_marg, "EPS Growth": eps_gr})

    total_pts = 0.0; max_pts = 0.0
    for section in ("health", "quality", "growth"):
        for val, score, _ in results[section].values():
            if val is not None:
                total_pts += score; max_pts += 1.0
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
        except:
            pass
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
            mc    = getattr(fi, "market_cap", None)
            price = getattr(fi, "last_price", None)
            prev  = getattr(fi, "previous_close", None)
            if mc and price:
                chg = ((price - prev) / prev * 100) if prev and prev > 0 else 0.0
                candidates.append({"symbol": sym, "name": _TOP10_CANDIDATES[sym],
                                   "price": round(price, 2), "change": round(chg, 2), "market_cap": mc})
        except Exception:
            pass
    candidates.sort(key=lambda x: x["market_cap"], reverse=True)
    return candidates[:10]

@st.cache_data(ttl=600)
def fetch_market_news():
    try:
        return yf.Ticker("SPY").news or []
    except Exception:
        return []

@st.cache_data(ttl=600)
def analyze_watchlist_ticker(symbol):
    fz = FearZEngine()
    _, _, spot, _, _, ivr, vol_hist, _ = fetch_ticker_resource(symbol)
    if spot is None:
        return None
    regime       = fz.classify_shock(ivr)
    shelf, gamma = fz.calculate_shelf(0.25, ivr, vol_hist)
    return {"Symbol": symbol, "Price": round(spot, 2), "IVR": round(ivr, 1),
            "Regime": regime, "Shelf": f"{shelf}d", "Gamma": gamma}


# ==========================================
# 3b. OPTIONS MARKET SCANNER
# ==========================================
SCAN_UNIVERSE = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "META", "AMZN", "GOOGL", "AMD",
    "PLTR", "SOFI", "BAC", "JPM", "GS",   "WMT",  "COST", "V",    "MA",   "DIS",
    "NFLX", "UBER", "COIN", "MSTR", "GME",  "F",    "GM",   "INTC", "MU",   "SMCI",
    "ARM",  "TSM",  "AVGO", "QCOM", "CRM",  "ORCL", "SNOW", "RBLX", "HOOD", "RIVN",
    "NIO",  "BABA", "SQ",   "PYPL", "SHOP", "ABNB", "DASH", "ROKU", "SNAP", "X"
]

def scan_single_ticker(symbol, option_type, holding_days, target_pct, stop_loss_pct):
    fz = FearZEngine()
    try:
        t, expirations, spot, rf, m_t0, auto_ivr, vol_hist, _ = fetch_ticker_resource(symbol)
        if t is None or not expirations or spot is None:
            return None
        best_exp = None
        for exp in expirations:
            dte = (pd.to_datetime(exp) - pd.to_datetime("today")).days
            if 15 <= dte <= 60:
                best_exp = exp; break
        if best_exp is None:
            best_exp = min(expirations, key=lambda e: abs((pd.to_datetime(e) - pd.to_datetime("today")).days - 30))
        days_to_exp = (pd.to_datetime(best_exp) - pd.to_datetime("today")).days
        if days_to_exp < 1:
            return None
        opts  = t.option_chain(best_exp)
        chain = opts.calls if option_type == "Call" else opts.puts
        if chain.empty:
            return None
        atm_strike = chain.iloc[(chain["strike"] - spot).abs().argsort()[:1]]["strike"].values[0]
        row        = chain[chain["strike"] == atm_strike].iloc[0]
        premium    = row["ask"] if row["ask"] > 0 else row["lastPrice"]
        if premium <= 0:
            return None
        iv       = row["impliedVolatility"] if row["impliedVolatility"] > 0 else 0.001
        regime   = fz.classify_shock(auto_ivr)
        shelf, _ = fz.calculate_shelf(iv, auto_ivr, vol_hist)
        projected_iv = fz.get_projection(holding_days, iv, m_t0, shelf, regime)
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
    except Exception:
        return None


# ==========================================
# 3c. STOCK ANALYTICS ENGINE
# ==========================================
STRIP_TICKERS = [
    "SPY", "QQQ", "DIA", "IWM", "^VIX", "^TNX", "^GSPC",
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AMD",
    "AVGO", "NFLX", "JPM", "BAC", "PLTR", "COIN", "SOFI",
    "GLD", "BTC-USD", "ETH-USD", "XLK", "XLF", "XLE",
]
NO_DOLLAR  = {"^VIX", "^TNX", "^GSPC"}
STRIP_LABELS = {
    "^VIX": "VIX", "^TNX": "10Y Yield", "^GSPC": "S&P 500",
    "BTC-USD": "BTC", "ETH-USD": "ETH",
    "IWM": "Russell 2K", "GLD": "Gold",
    "XLK": "Tech ETF", "XLF": "Fin ETF", "XLE": "Energy ETF",
}

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
            except Exception:
                pass
    except Exception:
        pass
    return results

@st.cache_data(ttl=900)
def fetch_strip_news():
    headlines = []
    for sym in ["SPY", "AAPL", "NVDA", "MSFT", "TSLA"]:
        try:
            items = yf.Ticker(sym).news or []
            for item in items[:3]:
                content = item.get("content", item)
                title = content.get("title", "") if isinstance(content, dict) else item.get("title", "")
                if title and len(title) > 10:
                    short = title[:70] + ("…" if len(title) > 70 else "")
                    headlines.append(short); break
        except Exception:
            pass
        if len(headlines) >= 6:
            break
    return headlines[:6]

def scan_single_stock(symbol, holding_days, target_pct):
    fz = FearZEngine()
    try:
        _, _, spot, rf, m_t0, ivr, vol_hist, hist = fetch_ticker_resource(symbol)
        if spot is None or hist is None or len(hist) < 22:
            return None
        sma21        = hist["Close"].rolling(21).mean().iloc[-1]
        price_vs_sma = (spot / sma21) - 1 if sma21 > 0 else 0
        regime       = fz.classify_shock(ivr)
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
    except Exception:
        return None

def run_backtest(symbol, lookback_days=252, holding_days=21, target_pct=5.0, stop_loss_pct=10.0, direction="long"):
    fz = FearZEngine()
    _short = direction == "short"
    try:
        hist = yf.Ticker(symbol).history(period="2y")
        if hist is None or len(hist) < lookback_days + holding_days + 30:
            return None
        hist    = hist.tail(lookback_days + holding_days + 30).copy()
        dates   = hist.index
        hist["SMA21"] = hist["Close"].rolling(21).mean()
        hist["RVol"]  = hist["Close"].pct_change().rolling(21).std() * np.sqrt(252)
        hist["Ret5"]  = hist["Close"].pct_change(5)
        trades, equity, equity_curve, in_trade = [], 0.0, [], False
        for i in range(30, len(hist) - holding_days):
            if in_trade:
                continue
            row   = hist.iloc[i]
            spot  = float(row["Close"])
            sma21 = float(row["SMA21"]) if pd.notna(row["SMA21"]) else spot
            rvol  = float(row["RVol"])  if pd.notna(row["RVol"])  else 0.25
            ret5  = float(row["Ret5"])  if pd.notna(row["Ret5"])  else 0.0
            _vw   = hist["RVol"].iloc[max(0, i-252):i].dropna()
            ivr_proxy = ((rvol - _vw.min()) / (_vw.max() - _vw.min()) * 100) if len(_vw) > 1 and _vw.max() > _vw.min() else 50
            price_vs_sma = (spot / sma21) - 1 if sma21 > 0 else 0
            regime = fz.classify_shock(ivr_proxy)
            score, verdict, _ = stock_advisor_verdict(ret5, price_vs_sma, ivr_proxy, regime)
            if verdict == "BUY":
                in_trade    = True
                entry_price = spot
                entry_date  = dates[i]
                exit_price, exit_date, outcome = spot, dates[min(i + holding_days, len(hist) - 1)], "held"
                stop_level   = entry_price * (1 + stop_loss_pct / 100) if _short else entry_price * (1 - stop_loss_pct / 100)
                target_level = entry_price * (1 - target_pct / 100) if _short else entry_price * (1 + target_pct / 100)
                for j in range(1, holding_days + 1):
                    if i + j >= len(hist): break
                    fp = float(hist.iloc[i + j]["Close"])
                    if _short:
                        if fp >= stop_level:  exit_price, exit_date, outcome = fp, dates[i+j], "stopped"; break
                        if fp <= target_level: exit_price, exit_date, outcome = fp, dates[i+j], "target";  break
                    else:
                        if fp <= stop_level:  exit_price, exit_date, outcome = fp, dates[i+j], "stopped"; break
                        if fp >= target_level: exit_price, exit_date, outcome = fp, dates[i+j], "target";  break
                    if j == holding_days:
                        exit_price, exit_date, outcome = fp, dates[i+j-1], "held"
                raw_ret = (exit_price - entry_price) / entry_price
                pnl_pct = (-raw_ret if _short else raw_ret) * 100
                equity += pnl_pct
                target_hit = exit_price <= target_level if _short else exit_price >= target_level
                trades.append({"Entry Date": entry_date.strftime("%Y-%m-%d"), "Exit Date": exit_date.strftime("%Y-%m-%d"),
                               "Entry Price": round(entry_price, 2), "Exit Price": round(exit_price, 2),
                               "P&L %": round(pnl_pct, 2), "Outcome": outcome, "Score": round(score, 1),
                               "Regime": regime, "Target Hit": target_hit, "Direction": direction.capitalize()})
                equity_curve.append({"Date": exit_date.strftime("%Y-%m-%d"), "Cumulative P&L %": round(equity, 2)})
                in_trade = False
        if not trades:
            return {"trades": [], "equity_curve": [], "stats": {}}
        n, nw = len(trades), sum(1 for t in trades if t["P&L %"] > 0)
        nt    = sum(1 for t in trades if t["Target Hit"])
        ns    = sum(1 for t in trades if t["Outcome"] == "stopped")
        aw    = np.mean([t["P&L %"] for t in trades if t["P&L %"] > 0]) if nw > 0 else 0
        al    = np.mean([t["P&L %"] for t in trades if t["P&L %"] <= 0]) if n - nw > 0 else 0
        gp    = sum(t["P&L %"] for t in trades if t["P&L %"] > 0)
        gl    = abs(sum(t["P&L %"] for t in trades if t["P&L %"] <= 0))
        return {"trades": trades, "equity_curve": equity_curve,
                "stats": {"Total Trades": n, "Win Rate": round(nw/n*100, 1),
                          "Target Hit Rate": round(nt/n*100, 1), "Stop Out Rate": round(ns/n*100, 1),
                          "Avg Win %": round(aw, 2), "Avg Loss %": round(al, 2),
                          "Profit Factor": round(gp/gl, 2) if gl > 0 else float("inf"),
                          "Total Return %": round(equity, 2)}}
    except Exception:
        return None


def run_walkforward(symbol, is_months=12, oos_months=3, holding_days=21, target_pct=5.0, stop_loss_pct=10.0, direction="long"):
    fz     = FearZEngine()
    _short = direction == "short"
    IS_DAYS  = int(is_months * 21)
    OOS_DAYS = int(oos_months * 21)
    try:
        hist = yf.Ticker(symbol).history(period="5y")
        if hist is None or len(hist) < IS_DAYS + OOS_DAYS + 40:
            return None
        hist = hist.copy()
        hist["SMA21"] = hist["Close"].rolling(21).mean()
        hist["RVol"]  = hist["Close"].pct_change().rolling(21).std() * np.sqrt(252)
        hist["Ret5"]  = hist["Close"].pct_change(5)
        folds = []; oos_start = IS_DAYS; fold_num = 0
        while oos_start + OOS_DAYS <= len(hist) - holding_days:
            oos_end  = oos_start + OOS_DAYS
            oos_hist = hist.iloc[oos_start:oos_end]
            fold_num += 1
            period_str = f"{oos_hist.index[0].strftime('%b %Y')} – {oos_hist.index[-1].strftime('%b %Y')}"
            regime_counts = {}
            for i in range(len(oos_hist)):
                row  = oos_hist.iloc[i]
                rvol = float(row["RVol"]) if pd.notna(row["RVol"]) else 0.25
                gi   = oos_start + i
                _vw  = hist["RVol"].iloc[max(0, gi-252):gi].dropna()
                ivr  = ((rvol - _vw.min()) / (_vw.max() - _vw.min()) * 100) if len(_vw) > 1 and _vw.max() > _vw.min() else 50
                rg   = fz.classify_shock(ivr)
                regime_counts[rg] = regime_counts.get(rg, 0) + 1
            dom_regime = max(regime_counts, key=regime_counts.get) if regime_counts else "Episodic"
            fold_trades = []; in_trade = False
            loop_start  = min(30, max(0, len(oos_hist) - holding_days - 1))
            for i in range(loop_start, len(oos_hist) - holding_days):
                if in_trade: continue
                row   = oos_hist.iloc[i]
                spot  = float(row["Close"])
                sma21 = float(row["SMA21"]) if pd.notna(row["SMA21"]) else spot
                rvol  = float(row["RVol"])  if pd.notna(row["RVol"])  else 0.25
                ret5  = float(row["Ret5"])  if pd.notna(row["Ret5"])  else 0.0
                gi    = oos_start + i
                _vw   = hist["RVol"].iloc[max(0, gi-252):gi].dropna()
                ivr   = ((rvol - _vw.min()) / (_vw.max() - _vw.min()) * 100) if len(_vw) > 1 and _vw.max() > _vw.min() else 50
                pvsma = (spot / sma21) - 1 if sma21 > 0 else 0
                regime = fz.classify_shock(ivr)
                score, verdict, _ = stock_advisor_verdict(ret5, pvsma, ivr, regime)
                if verdict == "BUY":
                    in_trade    = True; entry_price = spot; exit_price = spot
                    exit_idx    = min(i + holding_days, len(oos_hist) - 1); outcome = "held"
                    stop_level  = entry_price * (1 + stop_loss_pct/100) if _short else entry_price * (1 - stop_loss_pct/100)
                    target_lvl  = entry_price * (1 - target_pct/100)    if _short else entry_price * (1 + target_pct/100)
                    for j in range(1, holding_days + 1):
                        if i + j >= len(oos_hist): break
                        fp = float(oos_hist.iloc[i + j]["Close"])
                        if _short:
                            if fp >= stop_level: exit_price, exit_idx, outcome = fp, i+j, "stopped"; break
                            if fp <= target_lvl: exit_price, exit_idx, outcome = fp, i+j, "target";  break
                        else:
                            if fp <= stop_level: exit_price, exit_idx, outcome = fp, i+j, "stopped"; break
                            if fp >= target_lvl: exit_price, exit_idx, outcome = fp, i+j, "target";  break
                        if j == holding_days:
                            exit_price, exit_idx = fp, i+j-1
                    raw = (exit_price - entry_price) / entry_price
                    pnl = (-raw if _short else raw) * 100
                    fold_trades.append({"pnl": pnl, "regime": regime, "outcome": outcome, "score": score})
                    in_trade = False
            if fold_trades:
                pnls  = [t["pnl"] for t in fold_trades]
                n, nw = len(pnls), sum(1 for p in pnls if p > 0)
                avg   = np.mean(pnls); std = np.std(pnls) if len(pnls) > 1 else 1.0
                cum   = np.cumsum(pnls)
                maxdd = float(np.min(cum - np.maximum.accumulate(cum)))
                ann_f = np.sqrt(252 / max(holding_days, 1))
                sharpe = round(avg / std * ann_f, 2) if std > 0 else 0.0
            else:
                n = nw = 0; avg = maxdd = sharpe = 0.0
            folds.append({"fold": fold_num, "period": period_str, "dominant_regime": dom_regime,
                          "trades": n, "win_rate": round(nw/n*100, 1) if n > 0 else 0.0,
                          "avg_return": round(avg, 2), "max_dd": round(maxdd, 2),
                          "sharpe": round(sharpe, 2), "trade_details": fold_trades})
            oos_start += OOS_DAYS
        if not folds or sum(f["trades"] for f in folds) == 0:
            return None
        all_trades = [t for f in folds for t in f["trade_details"]]
        rg_groups  = {}
        for t in all_trades:
            rg_groups.setdefault(t["regime"], []).append(t["pnl"])
        regime_attr = []
        for rg in ["Episodic", "Structural", "Systemic"]:
            pnls = rg_groups.get(rg)
            if not pnls: continue
            n, nw = len(pnls), sum(1 for p in pnls if p > 0)
            mean  = np.mean(pnls)
            edge  = "✅" if mean >= 1.0 else ("⚠️" if mean >= 0 else "❌")
            regime_attr.append({"Regime": f"{REGIME_ICON.get(rg,'⚪')} {rg}", "Trades": n,
                                "Win Rate": f"{nw/n*100:.0f}%", "Avg Return": f"{mean:+.2f}%",
                                "Total Return": f"{sum(pnls):+.2f}%", "Edge": edge})
        all_pnls   = [t["pnl"] for t in all_trades]
        std_all    = np.std(all_pnls) if len(all_pnls) > 1 else 1.0
        ann_f      = np.sqrt(252 / max(holding_days, 1))
        agg_sharpe = round(np.mean(all_pnls) / std_all * ann_f, 2) if std_all > 0 else 0.0
        worst_dd   = min((f["max_dd"] for f in folds), default=0.0)
        return {"folds": folds, "regime_attribution": regime_attr,
                "agg_sharpe": agg_sharpe, "worst_fold_dd": round(worst_dd, 2),
                "total_oos_trades": len(all_trades), "n_folds": fold_num}
    except Exception:
        return None


# ==========================================
# 4. AI NARRATIVE (Claude API)
# ==========================================
def _get_anthropic_client():
    api_key = st.secrets.get("ANTHROPIC_API_KEY", None) or os.environ.get("ANTHROPIC_API_KEY", None)
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)

_ADVISOR_SYSTEM = (
    "You are a friendly, experienced investment advisor speaking to someone who is investing their own money privately for the first time. "
    "Always reply in a notes style — short bullet points or numbered items, never long paragraphs. "
    "Briefly define any financial term you use in plain English (e.g. 'Sharpe Ratio — measures how much return you get per unit of risk'). "
    "Be warm, direct, and practical. If something carries real risk, say so plainly. Keep every bullet to 1-2 lines."
)

def _stream_ai_response(prompt, max_tokens=400, placeholder=None, system=None):
    client = _get_anthropic_client()
    if not client:
        return "Add `ANTHROPIC_API_KEY` to `.streamlit/secrets.toml` to enable AI features."
    full_text = ""
    try:
        stream_kwargs = dict(model="claude-sonnet-4-6", max_tokens=max_tokens,
                             messages=[{"role": "user", "content": prompt}])
        if system:
            stream_kwargs["system"] = system
        with client.messages.stream(**stream_kwargs) as stream:
            for text_chunk in stream.text_stream:
                full_text += text_chunk
                if placeholder:
                    placeholder.markdown(f'<div class="briefing-card">{full_text}▌</div>', unsafe_allow_html=True)
        if placeholder:
            placeholder.markdown(f'<div class="briefing-card">{full_text}</div>', unsafe_allow_html=True)
    except Exception as e:
        full_text = f"Streaming error: {e}"
        if placeholder:
            placeholder.error(full_text)
    return full_text

def generate_briefing(market_data, watchlist_data, placeholder=None):
    market_str    = "\n".join([f"- {r['Name']} ({r['Symbol']}): ${r['Price']:.2f} ({r['Change']:+.2f}%)" for r in market_data])
    watchlist_str = "\n".join([f"- {r['Symbol']}: IVR {r['IVR']} ({r['Regime']} regime), ${r['Price']:.2f}" for r in watchlist_data if r]) or "No watchlist data."
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
        f"Financial Health ({scored['health_score']:.1f}/4): "
        f"CR {_fmt(raw.get('Current Ratio'))} | D/E {_fmt(raw.get('Debt/Equity'))} | "
        f"IC {_fmt(raw.get('Interest Coverage'))}x | QR {_fmt(raw.get('Quick Ratio'))}\n"
        f"Profitability ({scored['quality_score']:.1f}/5): "
        f"ROE {_fmt(raw.get('ROE'), pct=True)} | NM {_fmt(raw.get('Net Margin'), pct=True)} | "
        f"GM {_fmt(raw.get('Gross Margin'), pct=True)} | FCF {_fmt(raw.get('FCF'))} | EQ {_fmt(raw.get('Earnings Quality'))}\n"
        f"Growth ({scored['growth_score']:.1f}/4): "
        f"RevGr {_fmt(raw.get('Revenue Growth'), pct=True)} | GM Trend {_fmt(raw.get('GM Trend'), pct=True)}pp | "
        f"OpM {_fmt(raw.get('Operating Margin'), pct=True)} | EPS {_fmt(raw.get('EPS Growth'), pct=True)}\n"
        f"Fundamental Score: {scored['total_score']:.1f}/10 | Technical: {tech_verdict} ({tech_score:.1f}/3.0)"
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


# ==========================================
# 4b. ETF BUILDER HELPERS
# ==========================================
def normalize_price_frame(raw: pd.DataFrame) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        if "Adj Close" in level0:
            return raw["Adj Close"].copy()
        return raw["Close"].copy()
    if "Adj Close" in raw.columns:
        return raw["Adj Close"].to_frame(name="SINGLE_TICKER")
    return raw["Close"].to_frame(name="SINGLE_TICKER")

def get_risk_free_daily(default_annual: float = 0.04) -> float:
    try:
        hist = yf.Ticker("^IRX").history(period="5d")
        if hist.empty or "Close" not in hist.columns:
            raise ValueError
        rate = float(hist["Close"].dropna().iloc[-1]) / 100.0
    except Exception:
        rate = default_annual
    return (1.0 + rate) ** (1.0 / 252.0) - 1.0

def etf_get_cagr(prices: pd.DataFrame, ticker: str, years: int = 10) -> float:
    if ticker not in prices.columns:
        return float("nan")
    series = prices[ticker].dropna()
    if series.empty:
        return float("nan")
    cutoff = series.index[-1] - pd.DateOffset(years=years)
    series = series[series.index >= cutoff]
    if len(series) < 2:
        return float("nan")
    actual_years = (series.index[-1] - series.index[0]).days / 365.25
    if actual_years <= 0:
        return float("nan")
    try:
        return float((float(series.iloc[-1]) / float(series.iloc[0])) ** (1.0 / actual_years) - 1.0)
    except Exception:
        return float("nan")


# ==========================================
# 5. PAGE CONFIG & CSS
# ==========================================
st.set_page_config(page_title="MyQuant Institutional | Kern", layout="wide", page_icon="📊")

st.markdown("""<style>
/* ── BRANDING ── */
.branding-row { display:flex; align-items:center; margin-bottom:12px; }
.logo-col { flex:0 0 160px; border-right:1px solid rgba(191,161,93,0.4); margin-right:22px; padding-top:4px; }
.logo-text-kern { font-family:'Times New Roman',serif; color:#bfa15d; letter-spacing:0.6rem; font-size:4.5rem; text-transform:uppercase; margin:0; line-height:1; }
.logo-text-span { font-family:'Times New Roman',serif; color:#bfa15d; letter-spacing:0.6rem; font-size:2.2rem; text-transform:uppercase; margin:0; line-height:1; }
.title-col { flex:1; }
.main-title-text { font-size:2.1rem; font-weight:700; margin:0; line-height:1.1; color:var(--text-color); }
.subtitle-text { font-size:0.95rem; opacity:0.7; margin-top:4px; font-family:serif; color:var(--text-color); letter-spacing:0.02em; }
.inst-badge { display:inline-block; background:rgba(191,161,93,0.15); border:1px solid rgba(191,161,93,0.5); border-radius:3px; padding:2px 10px; font-size:0.6rem; font-weight:700; letter-spacing:0.18em; text-transform:uppercase; color:#bfa15d; margin-left:10px; vertical-align:middle; }

/* ── LIVE TICKER BAR ── */
.ticker-wrapper { width:100%; overflow:hidden; background:rgba(8,8,8,0.97); border-top:1px solid rgba(191,161,93,0.5); border-bottom:1px solid rgba(191,161,93,0.5); padding:7px 0; margin-bottom:16px; }
.ticker-track { display:flex; white-space:nowrap; animation:ticker-scroll 55s linear infinite; width:max-content; }
.ticker-track:hover { animation-play-state:paused; }
.ticker-item { display:inline-block; padding:0 26px; font-family:'Courier New',monospace; font-size:0.8rem; font-weight:600; letter-spacing:0.03em; color:#c8c8c8; border-right:1px solid rgba(191,161,93,0.2); }
.t-sym { color:#bfa15d; font-weight:700; margin-right:5px; }
.t-pos { color:#00d96f; }
.t-neg { color:#ff4b4b; }
.ticker-news { color:#d0d0d0; font-style:italic; font-size:0.76rem; border-right:1px solid rgba(191,161,93,0.35); padding:0 28px; letter-spacing:0.01em; }
.ticker-news-bullet { color:#bfa15d; margin-right:6px; font-style:normal; font-weight:700; }
@keyframes ticker-scroll { 0% { transform:translateX(0); } 100% { transform:translateX(-50%); } }

/* ── METRIC CARDS ── */
div[data-testid="stMetric"] { background-color:rgba(191,161,93,0.06); padding:12px 14px; border-radius:8px; border:1px solid rgba(191,161,93,0.35); min-height:90px; display:flex; flex-direction:column; justify-content:center; }
div[data-testid="stMetric"] > div { width:100%; text-align:left; }
div[data-testid="stMetricLabel"] > div { color:#bfa15d !important; font-weight:700; font-size:0.72rem; letter-spacing:0.06em; text-transform:uppercase; }
div[data-testid="stMetricValue"] { font-size:1.35rem; font-weight:700; }

/* ── VERDICT BADGES ── */
.verdict-badge { display:inline-block; padding:5px 16px; border-radius:4px; font-size:0.8rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; }
.badge-buy  { background:rgba(0,217,111,0.15); color:#00d96f; border:1px solid #00d96f; }
.badge-hold { background:rgba(255,193,7,0.15);  color:#FFC107; border:1px solid #FFC107; }
.badge-sell { background:rgba(255,75,75,0.15);  color:#ff4b4b; border:1px solid #ff4b4b; }
.verdict-display { border-radius:10px; padding:22px; text-align:center; border-width:2px; border-style:solid; }
.vd-buy  { border-color:#00d96f; background:rgba(0,217,111,0.07); }
.vd-hold { border-color:#FFC107; background:rgba(255,193,7,0.07);  }
.vd-sell { border-color:#ff4b4b; background:rgba(255,75,75,0.07);  }

/* ── TOP NAV ── */
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button { padding:4px 6px !important; font-size:0.63rem !important; letter-spacing:0.07em !important; text-transform:uppercase !important; font-weight:600 !important; min-height:26px !important; line-height:1.1 !important; white-space:nowrap !important; }

/* ── PAGE SECTION HEADER ── */
.page-section-header { display:flex; align-items:center; gap:12px; margin:4px 0 18px 0; }
.psh-line { flex:1; height:1px; background:linear-gradient(to right, rgba(191,161,93,0.55), rgba(191,161,93,0)); }
.psh-text { color:#bfa15d; font-size:0.68rem; font-weight:700; letter-spacing:0.15em; text-transform:uppercase; white-space:nowrap; }

/* ── BRIEFING CARD ── */
.briefing-card { background:rgba(191,161,93,0.05); border:1px solid rgba(191,161,93,0.3); border-radius:10px; padding:22px; font-family:Georgia,serif; line-height:1.75; font-size:1rem; }

/* ── PIPELINE ── */
.pipeline-container { display:flex; justify-content:space-between; gap:14px; }
.pipeline-box { flex:1; padding:14px; background:rgba(0,0,0,0.2); border-radius:6px; }

/* ── MOBILE ── */
@media (max-width:768px) {
    .branding-row { flex-direction:column; text-align:center; }
    .logo-col { flex:1; border-right:none; border-bottom:1px solid rgba(191,161,93,0.4); margin-right:0; margin-bottom:12px; padding-bottom:12px; }
    .logo-text-kern { font-size:3.2rem; }
    .pipeline-container { flex-direction:column; }
}

/* ── KELLY METER ── */
.kelly-note { font-size:0.75rem; opacity:0.65; font-style:italic; margin-top:4px; }

/* ── BACKTEST TRADE LOG ── */
.bt-win  { color:#00d96f; font-weight:700; }
.bt-loss { color:#ff4b4b; font-weight:700; }

/* ── ETF INSTITUTIONAL ── */
.etf-metric-row { display:grid; grid-template-columns:repeat(6,1fr); gap:12px; margin-bottom:20px; }
</style>""", unsafe_allow_html=True)


# ==========================================
# 5b. LIVE TICKER BAR
# ==========================================
strip_data = fetch_ticker_live_strip()
_strip_news = fetch_strip_news()
if strip_data:
    _kern_badge = (
        '<span class="ticker-item" style="padding:0 32px 0 24px;border-right:2px solid rgba(191,161,93,0.5);">'
        '<span style="font-family:Georgia,serif;font-weight:900;font-size:0.95rem;'
        'letter-spacing:0.18em;color:#bfa15d;text-transform:uppercase;font-style:italic;">KERN</span>'
        '</span>'
    )
    items_html = _kern_badge
    _news_idx  = 0
    for idx, item in enumerate(strip_data):
        if _strip_news and idx > 0 and idx % 5 == 0 and _news_idx < len(_strip_news):
            _nl = _strip_news[_news_idx]; _news_idx += 1
            items_html += (f'<span class="ticker-item ticker-news">'
                           f'<span class="ticker-news-bullet">\u25cf</span>{_nl}</span>')
        sym       = item["symbol"]
        label     = STRIP_LABELS.get(sym, sym)
        price_str = f"{item['price']:.2f}" if sym in NO_DOLLAR else f"${item['price']:.2f}"
        chg       = item["change_pct"]
        arrow     = "\u25b2" if chg >= 0 else "\u25bc"
        css_class = "t-pos" if chg >= 0 else "t-neg"
        items_html += (f'<span class="ticker-item"><span class="t-sym">{label}</span>'
                       f'{price_str} <span class="{css_class}">{arrow} {abs(chg):.2f}%</span></span>')
    doubled = items_html + items_html
    st.markdown(f'<div class="ticker-wrapper"><div class="ticker-track" style="animation-duration:80s;">{doubled}</div></div>',
                unsafe_allow_html=True)


# ==========================================
# 6. NAVIGATION
# ==========================================
fz = FearZEngine()

_SECTION_PAGES = {
    "Dashboard":         ["Welcome", "Market Overview"],
    "Options Analytics": ["Trade Advisor", "Options Scanner", "Options Watchlist"],
    "Stock Analytics":   ["Stock Advisor", "Stock Scanner", "Stock Watchlist", "Backtest"],
    "Account":           ["Positions"],
    "Institutional":     ["ETF Builder"],
}
_PAGE_SECTION = {pg: sec for sec, pgs in _SECTION_PAGES.items() for pg in pgs}

_ta_exp = _ta_type = _ta_chain = _ta_row = _ta_ivr = _ta_iv = None
_ta_shelf = _ta_gamma = _ta_regime = _ta_target = _ta_orders = _ta_sl = 0

if "nav_section" not in st.session_state:
    st.session_state.nav_section = "Dashboard"
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Welcome"


def _nav_btn(col, pg_name):
    if st.session_state.nav_page == pg_name:
        col.markdown(
            f'<div style="text-align:center;padding:4px 2px 3px;'
            f'border-bottom:2px solid #bfa15d;color:#bfa15d;'
            f'font-size:0.63rem;font-weight:700;letter-spacing:0.07em;'
            f'text-transform:uppercase;white-space:nowrap;">{pg_name}</div>',
            unsafe_allow_html=True)
    elif col.button(pg_name, use_container_width=True, key=f"nav_{pg_name}"):
        st.session_state.nav_page    = pg_name
        st.session_state.nav_section = _PAGE_SECTION[pg_name]
        st.rerun()


def _nav_sec_label(text):
    st.markdown(
        f'<div style="color:#bfa15d;font-size:0.58rem;font-weight:700;'
        f'letter-spacing:0.14em;text-transform:uppercase;'
        f'padding-bottom:3px;margin-bottom:3px;'
        f'border-bottom:1px solid rgba(191,161,93,0.28);">{text}</div>',
        unsafe_allow_html=True)


# ── ROW 1: SECTION TABS (always one compact row) ─────────────────────────
_SEC_LABELS = {
    "Dashboard":         "Dashboard",
    "Options Analytics": "Options",
    "Stock Analytics":   "Stocks",
    "Account":           "Account",
    "Institutional":     "Institutional",
}
_sc = st.columns(len(_SECTION_PAGES))
for _i, _sec in enumerate(_SECTION_PAGES):
    if st.session_state.nav_section == _sec:
        _sc[_i].markdown(
            f'<div style="text-align:center;padding:5px 4px 4px;border-bottom:2px solid #bfa15d;'
            f'color:#bfa15d;font-size:0.63rem;font-weight:700;letter-spacing:0.1em;'
            f'text-transform:uppercase;white-space:nowrap;">{_SEC_LABELS[_sec]}</div>',
            unsafe_allow_html=True)
    elif _sc[_i].button(_SEC_LABELS[_sec], key=f"_sec_{_sec}", use_container_width=True):
        st.session_state.nav_section = _sec
        st.session_state.nav_page    = _SECTION_PAGES[_sec][0]
        st.rerun()

# ── ROW 2: SUB-PAGES (only shown when section has multiple pages) ─────────
_active_pages = _SECTION_PAGES[st.session_state.nav_section]
if len(_active_pages) > 1:
    _pc = st.columns(len(_active_pages))
    for _i, _pg in enumerate(_active_pages):
        _nav_btn(_pc[_i], _pg)

st.markdown('<hr style="margin:3px 0 12px;border:none;border-top:1px solid rgba(191,161,93,0.25);">', unsafe_allow_html=True)

page    = st.session_state.nav_page
section = st.session_state.nav_section

if page not in ("Trade Advisor", "Stock Advisor"):
    st.markdown('<style>section[data-testid="stSidebar"],div[data-testid="collapsedControl"]{display:none!important;}</style>',
                unsafe_allow_html=True)

with st.sidebar:
    if page == "Trade Advisor":
        st.header("Trade Parameters")
        ta_ticker_input = st.text_input("Ticker Symbol", value="SPY", key="ta_ticker").upper().strip()
        if ta_ticker_input:
            _ticker_obj, _expirations, _spot, _rf, _m_t0, _auto_ivr, _vol_hist, _ = fetch_ticker_resource(ta_ticker_input)
            if _ticker_obj is None or not _expirations:
                st.error(f"No data for '{ta_ticker_input}'.")
            else:
                _stored_exp = st.session_state.get("ta_exp")
                if _stored_exp and _stored_exp not in _expirations:
                    del st.session_state["ta_exp"]
                _ta_exp  = st.selectbox("Expiration Date", _expirations, key="ta_exp")
                _ta_type = st.radio("Option Type", ["Call", "Put"], key="ta_type")
                _opts     = _ticker_obj.option_chain(_ta_exp)
                _ta_chain = _opts.calls if _ta_type == "Call" else _opts.puts
                if not _ta_chain.empty:
                    _strikes = _ta_chain["strike"].tolist()
                    _stored_strike = st.session_state.get("ta_strike")
                    if _stored_strike is not None and _stored_strike not in _strikes:
                        st.session_state["ta_strike"] = min(_strikes, key=lambda s: abs(s - _stored_strike))
                    _ta_strike = st.selectbox("Strike Price", _strikes, key="ta_strike")
                    _ta_row    = _ta_chain[_ta_chain["strike"] == _ta_strike].iloc[0]
                    st.divider()
                    st.markdown("### Behavioral Adjustment")
                    st.markdown(f"Live IV Rank: <span style='color:#bfa15d;font-weight:bold;font-size:1.05rem;'>{int(_auto_ivr)}</span>", unsafe_allow_html=True)
                    _ta_ivr              = st.slider("Stress Test Override", 0, 100, int(_auto_ivr))
                    _ta_regime           = fz.classify_shock(_ta_ivr)
                    _ta_iv               = _ta_row["impliedVolatility"] if _ta_row["impliedVolatility"] > 0 else 0.001
                    _ta_shelf, _ta_gamma = fz.calculate_shelf(_ta_iv, _ta_ivr, _vol_hist)
                    st.info(f"Regime: **{_ta_regime}**\n\nGamma: **{_ta_gamma}**\n\nShelf: **{_ta_shelf} Days**")
                    st.divider()
                    st.markdown("### Strategy")
                    _ta_target = st.number_input("Target Price ($)", value=float(_spot))
                    _ta_orders = st.number_input("Contracts", value=1, min_value=1)
                    _ta_sl     = st.slider("Stop Loss (%)", 0, 100, 20) / 100
                else:
                    st.warning("No options chain for this expiration.")
    elif page == "Stock Advisor":
        st.header("Stock Parameters")
        st.text_input("Ticker Symbol", value="AAPL", key="sa_ticker")
        _sa_holding = st.slider("Holding Period (Days)", 5, 120, 21)
        _sa_target  = st.number_input("Target Move (%)", value=5.0, min_value=0.1, step=0.5)
        _sa_shares  = st.number_input("Shares", value=100, min_value=1)
        st.divider()
        st.markdown("**Company News**")
        _sidebar_ticker = st.session_state.get("sa_ticker", "").strip().upper()
        if _sidebar_ticker:
            _news_items = fetch_company_news(_sidebar_ticker)
            if not _news_items:
                st.caption("No recent news found.")
            else:
                for _ni in _news_items[:6]:
                    _c     = _ni.get("content", _ni)
                    _title = _c.get("title", "")
                    _pub   = _c.get("provider", {}).get("displayName", _ni.get("publisher", ""))
                    _raw_date = _c.get("pubDate", "")
                    try:
                        _dt = datetime.strptime(_raw_date, "%Y-%m-%dT%H:%M:%SZ").strftime("%b %d") if _raw_date else ""
                    except Exception:
                        _ts = _ni.get("providerPublishTime", 0)
                        _dt = datetime.fromtimestamp(_ts).strftime("%b %d") if _ts else ""
                    if not _title: continue
                    st.markdown(
                        f"<div style='margin-bottom:8px;padding-bottom:8px;"
                        f"border-bottom:1px solid rgba(255,255,255,0.06);'>"
                        f"<div style='font-size:0.77rem;line-height:1.35;'>{_title}</div>"
                        f"<div style='font-size:0.65rem;opacity:0.45;margin-top:3px;'>{_pub} \u00b7 {_dt}</div>"
                        f"</div>", unsafe_allow_html=True)
                _news_sum_key = f"news_summary_{_sidebar_ticker}"
                if st.button("Summarize with AI", key="sa_news_sum_btn"):
                    _headlines = [_n.get("content", _n).get("title", "") for _n in _news_items[:8]]
                    _news_ph   = st.empty()
                    _summary   = generate_news_summary(_sidebar_ticker, _headlines, placeholder=_news_ph)
                    st.session_state[_news_sum_key] = _summary
                if _news_sum_key in st.session_state:
                    st.markdown(
                        f"<div style='background:rgba(191,161,93,0.07);border-left:2px solid #bfa15d;"
                        f"padding:8px 10px;border-radius:4px;font-size:0.75rem;line-height:1.5;"
                        f"margin-top:6px;'>{st.session_state[_news_sum_key]}</div>", unsafe_allow_html=True)
        st.divider()
        st.markdown("**Earnings Calendar**")
        _earn = fetch_earnings_calendar(_sidebar_ticker) if _sidebar_ticker else None
        if _earn:
            _earn_days     = _earn["days_away"]
            _earn_date_str = _earn["date"].strftime("%b %d, %Y")
            if _earn_days < 0:   st.caption(f"Last earnings: {_earn_date_str}")
            elif _earn_days <= 7:  st.error(f"Earnings in **{_earn_days} days** ({_earn_date_str}) — IV crush risk.")
            elif _earn_days <= 21: st.warning(f"Earnings in **{_earn_days} days** ({_earn_date_str})")
            else:                  st.info(f"Next earnings: {_earn_date_str} ({_earn_days} days away)")
        else:
            st.caption("Earnings date unavailable.")


# ── BRANDING HEADER ──
st.markdown("""
<div class="branding-row">
    <div class="logo-col"><span class="logo-text-span"><p class="logo-text-kern">KERN.</p></span></div>
    <div class="title-col">
        <div class="main-title-text">MyQuant | Institutional Terminal <span class="inst-badge">INSTITUTIONAL</span></div>
        <p class="subtitle-text">Institutional-grade investing. Options, equities, and ETF portfolio construction in one platform.</p>
    </div>
</div>""", unsafe_allow_html=True)


def _section_header(label):
    st.markdown(f'<div class="page-section-header"><div class="psh-text">{label}</div><div class="psh-line"></div></div>', unsafe_allow_html=True)

def _verdict_badge(verdict):
    cls = f"badge-{verdict.lower()}"
    return f'<span class="verdict-badge {cls}">{verdict}</span>'

RESULT_ICON   = {"Pass": "\u2705", "Warn": "\u26a0\ufe0f", "Fail": "\u274c"}
RESULT_COLOR  = {"Pass": "#00d96f", "Warn": "#FFC107", "Fail": "#ff4b4b"}
REGIME_ICON   = {"Episodic": "\U0001f7e2", "Structural": "\U0001f7e1", "Systemic": "\U0001f534"}
VERDICT_COLOR = {"BUY": "#00d96f", "HOLD": "#FFC107", "SELL": "#ff4b4b"}


def _render_options_scan_results(results):
    top10 = sorted(results, key=lambda x: (x["Score"], x["EV"]), reverse=True)[:10]
    st.success(f"Found **{len(results)}** opportunities. Top **{len(top10)}** shown.")
    col_w = [0.4, 0.8, 0.6, 0.9, 1.1, 0.7, 0.6, 1.0, 0.7, 0.6, 0.8, 0.6, 0.65]
    headers = st.columns(col_w)
    for col, lbl in zip(headers, ["#","Symbol","Type","Strike/DTE","Expiry","Prem","IVR","Regime","EV","Score","Verdict","",""]):
        col.markdown(f"<span style='color:#bfa15d;font-size:0.75rem;font-weight:700;'>{lbl}</span>", unsafe_allow_html=True)
    for rank, r in enumerate(top10, 1):
        ri   = REGIME_ICON.get(r["Regime"], "\u26aa")
        cols = st.columns(col_w)
        cols[0].markdown(f"**#{rank}**")
        cols[1].markdown(f"**{r['Symbol']}**")
        cols[2].markdown(r["Type"])
        cols[3].markdown(f"${r['Strike']:.0f} ({r['DTE']}d)")
        cols[4].markdown(r["Expiry"])
        cols[5].markdown(f"${r['Premium']:.2f}")
        cols[6].markdown(f"{r['IVR']:.0f}")
        cols[7].markdown(f"{ri} {r['Regime']}")
        cols[8].markdown(f"${r['EV']:.0f}")
        cols[9].markdown(f"**{r['Score']}/3**")
        cols[10].markdown(_verdict_badge(r["Verdict"]), unsafe_allow_html=True)
        if cols[11].button("Analyze \u2192", key=f"os_analyze_{rank}"):
            st.session_state.nav_section  = "Options Analytics"
            st.session_state.nav_page     = "Trade Advisor"
            st.session_state["ta_ticker"] = r["Symbol"]
            st.session_state["ta_exp"]    = r["Expiry"]
            st.session_state["ta_type"]   = r["Type"]
            st.session_state["ta_strike"] = float(r["Strike"])
            st.rerun()
        if cols[12].button("BT \u2192", key=f"os_bt_{rank}"):
            st.session_state["bt_prefill"] = {
                "ticker": r["Symbol"], "mode": "options",
                "direction": "long" if r["Type"] == "Call" else "short",
                "holding": min(r["DTE"], 60), "target": 5, "sl": 20,
            }
            st.session_state.nav_section = "Stock Analytics"
            st.session_state.nav_page    = "Backtest"
            st.rerun()
    st.divider()
    st.download_button("Download CSV", pd.DataFrame(top10).to_csv(index=False), "options_scan.csv", "text/csv")
    fig_bar = go.Figure(go.Bar(
        x=[r["Symbol"] + " " + r["Type"][0] for r in top10],
        y=[r["Score"] for r in top10],
        marker_color=[VERDICT_COLOR.get(r["Verdict"], "#bfa15d") for r in top10],
        text=[r["Verdict"] for r in top10], textposition="outside"))
    fig_bar.add_hline(y=2.5, line_dash="dot", line_color="#00d96f", annotation_text="BUY", annotation_position="right")
    fig_bar.add_hline(y=1.5, line_dash="dot", line_color="#FFC107", annotation_text="HOLD", annotation_position="right")
    fig_bar.update_layout(title="Top Opportunities — Trade Advisor Score",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="Score (out of 3)", range=[0,3.4], gridcolor="rgba(255,255,255,0.08)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"), margin=dict(l=0,r=0,t=40,b=0), showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})


def _render_stock_scan_results(results):
    top10 = sorted(results, key=lambda x: (x["Score"], x["P(Target)"]), reverse=True)[:10]
    st.success(f"Found **{len(results)}** opportunities. Top **{len(top10)}** shown.")
    col_w = [0.4, 0.8, 0.7, 0.6, 1.1, 0.8, 0.8, 0.6, 0.8, 0.6, 0.65, 0.65]
    headers = st.columns(col_w)
    for col, lbl in zip(headers, ["#","Symbol","Spot","IVR","Regime","Momentum","SMA21","Score","Verdict","","","ETF"]):
        col.markdown(f"<span style='color:#bfa15d;font-size:0.75rem;font-weight:700;'>{lbl}</span>", unsafe_allow_html=True)
    for rank, r in enumerate(top10, 1):
        ri   = REGIME_ICON.get(r["Regime"], "\u26aa")
        cols = st.columns(col_w)
        cols[0].markdown(f"**#{rank}**")
        cols[1].markdown(f"**{r['Symbol']}**")
        cols[2].markdown(f"${r['Spot']:.2f}")
        cols[3].markdown(f"{r['IVR']:.0f}")
        cols[4].markdown(f"{ri} {r['Regime']}")
        cols[5].markdown(r["Momentum"])
        cols[6].markdown(r["SMA21"])
        cols[7].markdown(f"**{r['Score']}/3**")
        cols[8].markdown(_verdict_badge(r["Verdict"]), unsafe_allow_html=True)
        if cols[9].button("Analyze \u2192", key=f"ss_analyze_{rank}"):
            st.session_state.nav_section  = "Stock Analytics"
            st.session_state.nav_page     = "Stock Advisor"
            st.session_state["sa_ticker"] = r["Symbol"]
            st.rerun()
        if cols[10].button("BT \u2192", key=f"ss_bt_{rank}"):
            st.session_state["bt_prefill"] = {
                "ticker": r["Symbol"], "mode": "stock",
                "direction": "long", "holding": 21, "target": 5, "sl": 10,
            }
            st.session_state.nav_section = "Stock Analytics"
            st.session_state.nav_page    = "Backtest"
            st.rerun()
        if cols[11].button("ETF+", key=f"ss_etf_{rank}", help=f"Add {r['Symbol']} to ETF Builder"):
            if "etf_portfolio" not in st.session_state:
                st.session_state.etf_portfolio = []
            _sym = r["Symbol"]
            _existing = next((i for i, h in enumerate(st.session_state.etf_portfolio) if h["ticker"] == _sym), None)
            if _existing is None:
                st.session_state.etf_portfolio.append({"ticker": _sym, "dollars": 10_000.0})
                st.toast(f"{_sym} added to ETF Builder at $10,000. Adjust amount in the ETF Builder tab.")
            else:
                st.toast(f"{_sym} is already in your ETF Builder portfolio.")
            st.session_state.pop("etf_results", None)
    st.divider()
    st.download_button("Download CSV", pd.DataFrame(top10).to_csv(index=False), "stock_scan.csv", "text/csv")
    fig_sb = go.Figure(go.Bar(
        x=[r["Symbol"] for r in top10], y=[r["Score"] for r in top10],
        marker_color=[VERDICT_COLOR.get(r["Verdict"], "#bfa15d") for r in top10],
        text=[r["Verdict"] for r in top10], textposition="outside"))
    fig_sb.add_hline(y=2.5, line_dash="dot", line_color="#00d96f", annotation_text="BUY", annotation_position="right")
    fig_sb.add_hline(y=1.5, line_dash="dot", line_color="#FFC107", annotation_text="HOLD", annotation_position="right")
    fig_sb.update_layout(title="Stock Advisor Score",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="Score (out of 3)", range=[0,3.4], gridcolor="rgba(255,255,255,0.08)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"), margin=dict(l=0,r=0,t=40,b=0), showlegend=False)
    st.plotly_chart(fig_sb, use_container_width=True, config={"displayModeBar": False})


# ==========================================
# PAGE: WELCOME
# ==========================================
if page == "Welcome":
    st.markdown(f"""
<div style="border:1px solid rgba(191,161,93,0.3);border-radius:10px;padding:28px 32px;margin-bottom:24px;background:rgba(191,161,93,0.04);">
  <div style="font-size:0.7rem;color:#bfa15d;letter-spacing:0.18em;text-transform:uppercase;margin-bottom:6px;">KERN. — Institutional Investment Intelligence</div>
  <div style="font-family:'Times New Roman',serif;font-size:1.9rem;color:#e8dfc8;font-weight:400;letter-spacing:0.04em;margin-bottom:12px;">The Complete Institutional Platform</div>
  <p style="font-size:0.97rem;line-height:1.75;opacity:0.85;max-width:860px;margin-bottom:18px;">
    MyQuant Institutional synthesizes real-time market data, volatility modeling, probability mathematics, AI-driven analysis, and custom ETF construction into a unified environment designed for institutional rigor.
  </p>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-top:4px;">
    <div style="border-left:2px solid #bfa15d;padding-left:14px;">
      <div style="font-size:0.72rem;color:#bfa15d;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">Fear Z Behavioral Engine</div>
      <div style="font-size:0.83rem;opacity:0.8;line-height:1.55;">Episodic, Structural, and Systemic regime classification driving disciplined position sizing.</div>
    </div>
    <div style="border-left:2px solid #bfa15d;padding-left:14px;">
      <div style="font-size:0.72rem;color:#bfa15d;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">Options and Equity Advisory</div>
      <div style="font-size:0.83rem;opacity:0.8;line-height:1.55;">Black-Scholes pricing, Greeks, Monte Carlo simulation, and fundamental scoring before any trade.</div>
    </div>
    <div style="border-left:2px solid #bfa15d;padding-left:14px;">
      <div style="font-size:0.72rem;color:#bfa15d;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">Scanner Suite</div>
      <div style="font-size:0.83rem;opacity:0.8;line-height:1.55;">Automated screening across equities and options ranked by quantitative score with AI analysis.</div>
    </div>
    <div style="border-left:2px solid #bfa15d;padding-left:14px;">
      <div style="font-size:0.72rem;color:#bfa15d;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">ETF Builder</div>
      <div style="font-size:0.83rem;opacity:0.8;line-height:1.55;">Custom ETF construction with Sharpe, Sortino, Beta, Alpha, Max Drawdown, VaR, and projections.</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    st.divider()
    _section_header("S&P 500 Performance")
    _overview = fetch_market_overview()
    if _overview:
        _ov_cols = st.columns(len(_overview))
        for _i, _item in enumerate(_overview):
            _ov_cols[_i].metric(
                _item["Name"],
                f"{_item['Price']:.2f}" if _item["Symbol"] == "^VIX" else f"${_item['Price']:.2f}",
                delta=f"{_item['Change']:+.2f}%",
                delta_color="inverse" if _item["Symbol"] == "^VIX" else "normal")

    _sp_tf_col, _sp_ind_col = st.columns([1, 2])
    _sp_tf = _sp_tf_col.selectbox("Chart Timeframe", ["1 Day", "5 Days", "1 Month", "6 Months", "1 Year"], index=2, key="wb_sp_tf")
    _sp_indicators = _sp_ind_col.multiselect("Indicators", ["21 SMA", "200 SMA", "21 EMA", "VWAP"], default=["21 SMA"], key="wb_sp_indicators")
    _sp_chart = fetch_chart_data("SPY", _sp_tf)
    _sp_chart["SMA_21"]  = _sp_chart["Close"].rolling(21).mean()
    _sp_chart["SMA_200"] = _sp_chart["Close"].rolling(200).mean()
    _sp_chart["EMA_21"]  = _sp_chart["Close"].ewm(span=21, adjust=False).mean()
    _sp_tp               = (_sp_chart["High"] + _sp_chart["Low"] + _sp_chart["Close"]) / 3
    _sp_chart["VWAP"]    = (_sp_tp * _sp_chart["Volume"]).cumsum() / _sp_chart["Volume"].cumsum()
    _sp_breaks = [dict(bounds=["sat","mon"]), dict(bounds=[16,9.5], pattern="hour")] \
                 if _sp_tf in ["1 Day","5 Days"] else [dict(bounds=["sat","mon"])]
    _fig_sp = go.Figure(data=[go.Candlestick(
        x=_sp_chart.index, open=_sp_chart["Open"], high=_sp_chart["High"],
        low=_sp_chart["Low"], close=_sp_chart["Close"],
        increasing_line_color="#00ffcc", decreasing_line_color="#ff4b4b", name="SPY")])
    if "21 SMA"  in _sp_indicators: _fig_sp.add_trace(go.Scatter(x=_sp_chart.index, y=_sp_chart["SMA_21"],  mode="lines", line=dict(color="#bfa15d", width=1.5), name="21 SMA"))
    if "200 SMA" in _sp_indicators: _fig_sp.add_trace(go.Scatter(x=_sp_chart.index, y=_sp_chart["SMA_200"], mode="lines", line=dict(color="#7b68ee", width=1.5), name="200 SMA"))
    if "21 EMA"  in _sp_indicators: _fig_sp.add_trace(go.Scatter(x=_sp_chart.index, y=_sp_chart["EMA_21"],  mode="lines", line=dict(color="#00d9ff", width=1.5), name="21 EMA"))
    if "VWAP"    in _sp_indicators: _fig_sp.add_trace(go.Scatter(x=_sp_chart.index, y=_sp_chart["VWAP"],    mode="lines", line=dict(color="#ff8c42", width=1.5), name="VWAP"))
    _fig_sp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=10,b=0), hovermode="x unified",
        yaxis=dict(title="Price ($)", gridcolor="rgba(255,255,255,0.08)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", rangebreaks=_sp_breaks),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(_fig_sp, use_container_width=True, config={"displayModeBar": False})

    st.divider()
    _section_header("Top 10 Companies by Market Cap")
    _top10 = fetch_top10_data()
    if _top10:
        _t10_hdr = st.columns([0.4, 2.8, 1.0, 1.3, 1.2, 1.4])
        for _col, _lbl in zip(_t10_hdr, ["#", "Company", "Ticker", "Market Cap", "Price", "Day Change"]):
            _col.markdown(f"<div style='font-size:0.68rem;color:#bfa15d;text-transform:uppercase;letter-spacing:0.1em;padding-bottom:4px;border-bottom:1px solid rgba(191,161,93,0.25);'>{_lbl}</div>", unsafe_allow_html=True)
        for _rank, _row in enumerate(_top10, 1):
            _chg_color = "#00ffcc" if _row["change"] >= 0 else "#ff4b4b"
            _arrow = "\u25b2" if _row["change"] >= 0 else "\u25bc"
            _mc = _row.get("market_cap", 0)
            _mc_str = f"${_mc/1e12:.2f}T" if _mc >= 1e12 else f"${_mc/1e9:.1f}B"
            _rc = st.columns([0.4, 2.8, 1.0, 1.3, 1.2, 1.4])
            _rc[0].markdown(f"<div style='padding:6px 0;opacity:0.5;font-size:0.82rem;'>{_rank}</div>", unsafe_allow_html=True)
            _rc[1].markdown(f"<div style='padding:6px 0;font-size:0.88rem;'>{_row['name']}</div>", unsafe_allow_html=True)
            _rc[2].markdown(f"<div style='padding:6px 0;font-size:0.82rem;color:#bfa15d;'>{_row['symbol']}</div>", unsafe_allow_html=True)
            _rc[3].markdown(f"<div style='padding:6px 0;font-size:0.88rem;font-weight:600;'>{_mc_str}</div>", unsafe_allow_html=True)
            _rc[4].markdown(f"<div style='padding:6px 0;font-size:0.88rem;'>${_row['price']:,.2f}</div>", unsafe_allow_html=True)
            _rc[5].markdown(f"<div style='padding:6px 0;font-size:0.88rem;color:{_chg_color};'>{_arrow} {abs(_row['change']):.2f}%</div>", unsafe_allow_html=True)

    st.divider()
    _section_header("Market News")
    _mkt_news = fetch_market_news()
    if _mkt_news:
        _nl, _nr = st.columns(2)
        for _idx, _ni in enumerate(_mkt_news[:10]):
            _c = _ni.get("content", _ni)
            _title = _c.get("title", "")
            if not _title: continue
            _pub    = _c.get("provider", {}).get("displayName", _ni.get("publisher", ""))
            _raw_dt = _c.get("pubDate", "")
            try:
                _dt_str = datetime.strptime(_raw_dt, "%Y-%m-%dT%H:%M:%SZ").strftime("%b %d, %Y") if _raw_dt else ""
            except Exception:
                _ts = _ni.get("providerPublishTime", 0)
                _dt_str = datetime.fromtimestamp(_ts).strftime("%b %d, %Y") if _ts else ""
            _desc = _c.get("summary", "")[:180] + "..." if _c.get("summary", "") else ""
            _card = (f"<div style='border:1px solid rgba(255,255,255,0.07);border-radius:7px;padding:14px 16px;margin-bottom:12px;'>"
                     f"<div style='font-size:0.85rem;font-weight:600;line-height:1.4;margin-bottom:6px;'>{_title}</div>"
                     f"<div style='font-size:0.75rem;opacity:0.55;margin-bottom:8px;'>{_pub} &middot; {_dt_str}</div>"
                     f"<div style='font-size:0.78rem;opacity:0.72;line-height:1.5;'>{_desc}</div></div>")
            (_nl if _idx % 2 == 0 else _nr).markdown(_card, unsafe_allow_html=True)

    st.divider()
    _section_header("AI Morning Briefing")
    st.caption(date.today().strftime("%B %d, %Y"))
    if not _get_anthropic_client():
        st.warning("Add `ANTHROPIC_API_KEY` to `.streamlit/secrets.toml` to enable AI briefings.")
    if st.button("Generate Briefing", type="primary"):
        _market_data    = fetch_market_overview()
        _watchlist      = st.session_state.get("watchlist", ["SPY", "QQQ", "AAPL", "NVDA"])
        _watchlist_data = [r for sym in _watchlist[:5] if (r := analyze_watchlist_ticker(sym))]
        generate_briefing(_market_data, _watchlist_data, placeholder=st.empty())


# ==========================================
# PAGE: MARKET OVERVIEW
# ==========================================
elif page == "Market Overview":
    _section_header("Dashboard — Market Overview")
    overview = fetch_market_overview()
    if overview:
        cols = st.columns(len(overview))
        for i, item in enumerate(overview):
            cols[i].metric(item["Name"],
                           f"{item['Price']:.2f}" if item["Symbol"] == "^VIX" else f"${item['Price']:.2f}",
                           delta=f"{item['Change']:+.2f}%",
                           delta_color="inverse" if item["Symbol"] == "^VIX" else "normal")
    st.divider()
    idx_c, tf_c, ind_c = st.columns([3, 1, 1.5])
    with idx_c: st.subheader("Index Chart")
    with tf_c:
        index_choice = st.selectbox("Index", ["SPY", "QQQ", "DIA"])
        timeframe    = st.selectbox("Timeframe", ["1 Day", "5 Days", "1 Month", "6 Months", "1 Year"])
    with ind_c:
        mo_indicators = st.multiselect("Indicators", ["21 SMA", "200 SMA", "21 EMA", "VWAP"], default=["21 SMA"], key="mo_indicators")
    chart_data = fetch_chart_data(index_choice, timeframe)
    chart_data["SMA_21"]  = chart_data["Close"].rolling(21).mean()
    chart_data["SMA_200"] = chart_data["Close"].rolling(200).mean()
    chart_data["EMA_21"]  = chart_data["Close"].ewm(span=21, adjust=False).mean()
    _mo_tp                = (chart_data["High"] + chart_data["Low"] + chart_data["Close"]) / 3
    chart_data["VWAP"]    = (_mo_tp * chart_data["Volume"]).cumsum() / chart_data["Volume"].cumsum()
    x_breaks = [dict(bounds=["sat","mon"]), dict(bounds=[16,9.5], pattern="hour")] \
               if timeframe in ["1 Day","5 Days"] else [dict(bounds=["sat","mon"])]
    fig = go.Figure(data=[go.Candlestick(x=chart_data.index, open=chart_data["Open"], high=chart_data["High"],
        low=chart_data["Low"], close=chart_data["Close"],
        increasing_line_color="#00ffcc", decreasing_line_color="#ff4b4b", name="Price")])
    if "21 SMA"  in mo_indicators: fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data["SMA_21"],  mode="lines", line=dict(color="#bfa15d", width=1.5), name="21 SMA"))
    if "200 SMA" in mo_indicators: fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data["SMA_200"], mode="lines", line=dict(color="#7b68ee", width=1.5), name="200 SMA"))
    if "21 EMA"  in mo_indicators: fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data["EMA_21"],  mode="lines", line=dict(color="#00d9ff", width=1.5), name="21 EMA"))
    if "VWAP"    in mo_indicators: fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data["VWAP"],    mode="lines", line=dict(color="#ff8c42", width=1.5), name="VWAP"))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=10,b=0),
        yaxis=dict(title="Price ($)", gridcolor="rgba(255,255,255,0.08)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", rangebreaks=x_breaks),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ==========================================
# PAGE: TRADE ADVISOR
# ==========================================
elif page == "Trade Advisor":
    _section_header("Options Analytics — Trade Advisor")
    with st.expander("Learn More: Fear Z Behavioral Engine"):
        st.markdown("""<div class="pipeline-container">
            <div class="pipeline-box" style="border-top:3px solid #00ffcc;"><strong style="color:#00ffcc;">1. IV Rank Input</strong><p style="font-size:0.85rem;opacity:0.8;margin:6px 0 0;">Classifies true severity of market panic using 1-year IV history.</p></div>
            <div class="pipeline-box" style="border-top:3px solid #ff4b4b;"><strong style="color:#ff4b4b;">2. Behavioral Math</strong><p style="font-size:0.85rem;opacity:0.8;margin:6px 0 0;">Emotional Inertia + Momentum Drag = Panic Plateau shelf duration.</p></div>
            <div class="pipeline-box" style="border-top:3px solid #FFC107;"><strong style="color:#FFC107;">3. IV Projection</strong><p style="font-size:0.85rem;opacity:0.8;margin:6px 0 0;">Predicts exact vol crush % over your holding period.</p></div>
            <div class="pipeline-box" style="border-top:3px solid #bfa15d;"><strong style="color:#bfa15d;">4. Trade Verdict</strong><p style="font-size:0.85rem;opacity:0.8;margin:6px 0 0;">3-rule scoring: EV + BS mispricing + Fear Z = BUY / HOLD / SELL.</p></div>
        </div><br>""", unsafe_allow_html=True)

    ta_ticker_input = st.session_state.get("ta_ticker", "")
    if not ta_ticker_input or _ta_row is None:
        st.info("Enter a ticker symbol and select a strike in the sidebar to begin.")
        st.stop()

    premium     = _ta_row["ask"] if _ta_row["ask"] > 0 else _ta_row["lastPrice"]
    days_to_exp = (pd.to_datetime(_ta_exp) - pd.to_datetime("today")).days
    time_to_exp = max(days_to_exp, 1) / 365
    breakeven   = _ta_strike + premium if _ta_type == "Call" else _ta_strike - premium

    sim_col, shock_col = st.columns(2)
    with sim_col:
        days_to_hold = st.slider("Holding Period (Days)", 1, max(days_to_exp, 2), max(days_to_exp, 1)) if days_to_exp > 1 else 1
        projected_iv = fz.get_projection(days_to_hold, _ta_iv, _m_t0, _ta_shelf, _ta_regime)
    with shock_col:
        vol_shock_suggested = (projected_iv / _ta_iv) - 1
        vol_shock = st.slider("Custom Vol Shock (%)", -50, 150, int(vol_shock_suggested * 100)) / 100

    adj_iv          = _ta_iv * (1 + vol_shock)
    adj_time        = max(days_to_hold, 1) / 365
    adj_periodic_iv = max(adj_iv * np.sqrt(adj_time), 0.0001)
    bs_fair_value   = calculate_black_scholes(_spot, _ta_strike, time_to_exp, _rf, _ta_iv, _ta_type)
    drift = (_rf - 0.5 * adj_iv**2) * adj_time
    t_z   = (np.log(_ta_target / _spot) - drift) / adj_periodic_iv
    s_z   = (np.log(_ta_strike / _spot) - drift) / adj_periodic_iv
    b_z   = (np.log(breakeven  / _spot) - drift) / adj_periodic_iv
    if _ta_type == "Call":
        t_prob, s_prob, b_prob = 1 - norm.cdf(t_z), 1 - norm.cdf(s_z), 1 - norm.cdf(b_z)
        intrinsic = max(0, _ta_target - _ta_strike)
    else:
        t_prob, s_prob, b_prob = norm.cdf(t_z), norm.cdf(s_z), norm.cdf(b_z)
        intrinsic = max(0, _ta_strike - _ta_target)
    pnl_per_contract  = (intrinsic - premium) * 100
    total_pnl         = pnl_per_contract * _ta_orders
    max_risk          = premium * _ta_orders * 100
    risk_factor       = 1.0 if _ta_sl == 0.0 else _ta_sl
    ev                = (t_prob * total_pnl) - (((1 - b_prob) * max_risk) * risk_factor)
    days_remaining    = max(0, days_to_exp - days_to_hold)
    projected_premium = calculate_black_scholes(_ta_target, _ta_strike, days_remaining / 365, _rf, projected_iv, _ta_type)
    projected_roi     = ((projected_premium - premium) / premium * 100) if premium > 0 else 0
    score, verdict, rules = trade_advisor_verdict(ev, premium, bs_fair_value, _ta_regime)

    valuation_label = "Overvalued" if premium > bs_fair_value else "Undervalued"
    pct_diff = ((premium - bs_fair_value) / bs_fair_value * 100) if bs_fair_value > 0 else 0
    r1 = st.columns(4)
    r1[0].metric("Spot Price",     f"${_spot:.2f}")
    r1[1].metric("Market Premium", f"${premium:.2f}")
    r1[2].metric("Black-Scholes",  f"${bs_fair_value:.2f}", delta=f"{pct_diff:.1f}% {valuation_label}", delta_color="inverse")
    r1[3].metric("Exit Premium",   f"${projected_premium:.2f}", delta=f"{projected_roi:.1f}% ROI")
    r2 = st.columns(4)
    r2[0].metric(f"IV: {ta_ticker_input}", f"{_ta_iv*100:.1f}%")
    r2[1].metric("Fear Z Shelf",   f"{_ta_shelf}d", delta=_ta_regime, delta_color="off")
    r2[2].metric("Expected Value", f"${ev:.2f}")
    r2[3].metric("Breakeven",      f"${breakeven:.2f}")

    _greeks = calculate_greeks(_spot, _ta_strike, time_to_exp, _rf, _ta_iv, _ta_type)
    _section_header("Options Greeks")
    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("Delta", f"{_greeks['Delta']:+.4f}")
    g2.metric("Gamma", f"{_greeks['Gamma']:.4f}")
    g3.metric("Theta", f"${_greeks['Theta']:.4f}/day")
    g4.metric("Vega",  f"{_greeks['Vega']:.4f}")
    g5.metric("Rho",   f"{_greeks['Rho']:.4f}")

    st.divider()
    _section_header("Position Sizing — Kelly Criterion")
    _k_col1, _k_col2 = st.columns([2, 3])
    with _k_col1:
        if "kelly_account" not in st.session_state: st.session_state.kelly_account = 25000.0
        _k_account = st.number_input("Account Size ($)", min_value=1000.0, max_value=10_000_000.0,
                                     value=st.session_state.kelly_account, step=1000.0, key="ta_kelly_account")
        st.session_state.kelly_account = _k_account
    with _k_col2:
        _k_win_prob   = 0.30 + (score / 3.0) * 0.45
        _k_pot_profit = max(0, _ta_target - breakeven) if _ta_type == "Call" else max(0, breakeven - _ta_target)
        _k_actual_risk = premium * (risk_factor if risk_factor > 0 else 1.0)
        _k_avg_win    = max(_k_pot_profit * _ta_orders * 100, 1)
        _k_avg_loss   = max(_k_actual_risk * _ta_orders, 1)
        _kd           = kelly_position_size(_k_win_prob, _k_avg_win, _k_avg_loss, _k_account)
        _k_contracts  = max(1, int(_kd["recommended_dollars"] / max(premium * 100, 1)))
        _kc1, _kc2, _kc3 = st.columns(3)
        _kc1.metric("Kelly Allocation",  f"{_kd['kelly_pct']:.1f}%")
        _kc2.metric("Recommended $",     f"${_kd['recommended_dollars']:,.0f}")
        _kc3.metric("Implied Contracts", str(_k_contracts))
        st.caption(_kd["note"])

    st.divider()
    _section_header("Trade Advisor Verdict")
    vc = VERDICT_COLOR[verdict]; vd_class = f"vd-{verdict.lower()}"
    vadv_col, rules_col = st.columns([1, 2])
    with vadv_col:
        st.markdown(f"""<div class="verdict-display {vd_class}">
            <div style="font-size:2.8rem;font-weight:900;color:{vc};">{verdict}</div>
            <div style="margin-top:8px;">{_verdict_badge(verdict)}</div>
            <div style="font-size:0.9rem;opacity:0.7;margin-top:8px;">{score:.1f} / 3.0 pts</div>
        </div>""", unsafe_allow_html=True)
    with rules_col:
        for rule in rules:
            st.markdown(f"**{RESULT_ICON[rule['result']]} {rule['rule']}** — <span style='color:{RESULT_COLOR[rule['result']]}'>{rule['result']}</span>: {rule['detail']}", unsafe_allow_html=True)

    _ta_btn1, _ta_btn2 = st.columns(2)
    if _ta_btn1.button("Get AI Trade Reasoning", type="secondary"):
        generate_trade_reasoning(ta_ticker_input, _ta_type, _ta_strike, _ta_exp,
                                 premium, bs_fair_value, ev, _ta_regime, _ta_shelf, verdict, score, placeholder=st.empty())
    if _ta_btn2.button("Backtest This Trade", type="secondary", key="ta_bt_btn"):
        try: _bt_dte = max(5, min(int((pd.to_datetime(_ta_exp) - pd.Timestamp.today()).days), 60))
        except Exception: _bt_dte = 21
        _bt_tgt = round(abs(_ta_target - _spot) / _spot * 100, 1) if _spot > 0 else 5.0
        st.session_state["bt_prefill"] = {"ticker": ta_ticker_input, "mode": "options",
            "direction": "long" if _ta_type == "Call" else "short",
            "holding": _bt_dte, "target": max(1, min(int(_bt_tgt), 20)), "sl": 20}
        st.session_state.nav_section = "Stock Analytics"; st.session_state.nav_page = "Backtest"; st.rerun()

    st.divider()
    _section_header("Live Chart")
    cc1, cc2, cc3 = st.columns([4, 1, 1.5])
    with cc1: st.subheader(f"{ta_ticker_input}")
    with cc2: timeframe = st.selectbox("Timeframe", ["1 Day","5 Days","1 Month","6 Months","1 Year","5 Years"], index=4)
    with cc3: ta_indicators = st.multiselect("Indicators", ["21 SMA","200 SMA","21 EMA","VWAP"], default=["21 SMA"], key="ta_indicators")
    chart_data = fetch_chart_data(ta_ticker_input, timeframe)
    chart_data["SMA_21"]  = chart_data["Close"].rolling(21).mean()
    chart_data["SMA_200"] = chart_data["Close"].rolling(200).mean()
    chart_data["EMA_21"]  = chart_data["Close"].ewm(span=21, adjust=False).mean()
    _ta_tp = (chart_data["High"] + chart_data["Low"] + chart_data["Close"]) / 3
    chart_data["VWAP"] = (_ta_tp * chart_data["Volume"]).cumsum() / chart_data["Volume"].cumsum()
    x_breaks = [dict(bounds=["sat","mon"]), dict(bounds=[16,9.5], pattern="hour")] if timeframe in ["1 Day","5 Days"] else [dict(bounds=["sat","mon"])]
    fig_c = go.Figure(data=[go.Candlestick(x=chart_data.index, open=chart_data["Open"], high=chart_data["High"],
        low=chart_data["Low"], close=chart_data["Close"], increasing_line_color="#00ffcc", decreasing_line_color="#ff4b4b", name="Price")])
    if "21 SMA"  in ta_indicators: fig_c.add_trace(go.Scatter(x=chart_data.index, y=chart_data["SMA_21"],  mode="lines", line=dict(color="#bfa15d", width=1.5), name="21 SMA"))
    if "200 SMA" in ta_indicators: fig_c.add_trace(go.Scatter(x=chart_data.index, y=chart_data["SMA_200"], mode="lines", line=dict(color="#7b68ee", width=1.5), name="200 SMA"))
    if "21 EMA"  in ta_indicators: fig_c.add_trace(go.Scatter(x=chart_data.index, y=chart_data["EMA_21"],  mode="lines", line=dict(color="#00d9ff", width=1.5), name="21 EMA"))
    if "VWAP"    in ta_indicators: fig_c.add_trace(go.Scatter(x=chart_data.index, y=chart_data["VWAP"],    mode="lines", line=dict(color="#ff8c42", width=1.5), name="VWAP"))
    fig_c.add_hline(y=_ta_target, line_dash="dash", line_color="#00ffcc", opacity=0.6)
    fig_c.add_hline(y=breakeven,  line_dash="solid", line_color="#ff4b4b", opacity=0.6)
    fig_c.add_hline(y=_spot, line_dash="dot", line_color="#bfa15d", annotation_text=f"  ${_spot:.2f}", annotation_position="bottom right")
    fig_c.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=10,b=0), hovermode="x unified",
        yaxis=dict(title="Price ($)", gridcolor="rgba(255,255,255,0.08)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", rangebreaks=x_breaks),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_c, use_container_width=True, config={"scrollZoom": False, "displayModeBar": False})
    st.divider()

    col_l, col_r = st.columns([2, 3])
    with col_l:
        _section_header("Probability Summary")
        st.dataframe(pd.DataFrame({"Level": ["Target","Strike","Breakeven"],
            "Price": [f"${_ta_target:.2f}",f"${_ta_strike:.2f}",f"${breakeven:.2f}"],
            "Probability": [f"{t_prob:.2%}",f"{s_prob:.2%}",f"{b_prob:.2%}"]}), hide_index=True, use_container_width=True)
        _section_header("Trade Analysis")
        actual_risk = premium * risk_factor
        pot_profit  = max(0, _ta_target - breakeven) if _ta_type == "Call" else max(0, breakeven - _ta_target)
        rr          = (pot_profit / actual_risk) if actual_risk > 0 else 0
        for k, v in {"Contracts": _ta_orders, "Cash at Risk": f"${actual_risk*_ta_orders*100:.2f}",
                     "Stop Loss": "None" if _ta_sl == 0 else f"{_ta_sl*100:.0f}%",
                     "Potential Profit": f"${pot_profit:.2f}/cnt", "R/R Ratio": f"{rr:.2f}"}.items():
            if k == "R/R Ratio": st.markdown(f"**{k}:** <span style='color:#bfa15d;font-weight:bold;'>{v}</span>", unsafe_allow_html=True)
            else: st.write(f"**{k}:** {v}")
        if st.button("Save to Positions"):
            if "positions" not in st.session_state: st.session_state.positions = []
            st.session_state.positions.append({"Symbol": ta_ticker_input, "Type": _ta_type,
                "Strike": _ta_strike, "Expiration": _ta_exp, "Entry Premium": round(premium, 2),
                "Target": _ta_target, "Contracts": _ta_orders, "Stop Loss": f"{int(_ta_sl*100)}%",
                "Opened": date.today().isoformat()})
            st.success("Position saved.")
    with col_r:
        _section_header("Price Distribution (10,000 Simulations)")
        sim_prices = np.random.lognormal(np.log(_spot) + drift, adj_periodic_iv, 10000)
        p5, p95 = np.percentile(sim_prices, [5, 95])
        fig_hist = go.Figure()
        fig_hist.add_vrect(x0=p5, x1=p95, fillcolor="#bfa15d", opacity=0.12, layer="below", line_width=0)
        fig_hist.add_trace(go.Histogram(x=sim_prices, nbinsx=150, marker_color="#bfa15d", opacity=0.75))
        fig_hist.add_vline(x=_spot,      line_dash="dash",  line_color="#ffffff", opacity=0.8)
        fig_hist.add_vline(x=breakeven,  line_dash="solid", line_color="#ff4b4b")
        fig_hist.add_vline(x=_ta_target, line_dash="dot",   line_color="#00ffcc")
        fig_hist.update_layout(title=f"Price in {days_to_hold} Days", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Price ($)"), yaxis=dict(title="Frequency"), showlegend=False, bargap=0.2, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_hist, use_container_width=True, config={"scrollZoom": False, "displayModeBar": False})
        prob_hit = (sim_prices >= _ta_target).mean() if _ta_type == "Call" else (sim_prices <= _ta_target).mean()
        st.markdown(f"""<div class="briefing-card">
            <strong>Simulation Insight:</strong> In a <strong>{_ta_regime}</strong> regime, Fear Z projects IV to decay to
            <strong>{projected_iv*100:.1f}%</strong> over <strong>{days_to_hold} days</strong>.
            Monte Carlo: <strong style="color:#bfa15d;">{prob_hit:.1%}</strong> probability of reaching <strong>${_ta_target:.2f}</strong>.
            Verdict: <strong style="color:{vc};">{verdict}</strong> ({score:.1f}/3.0).
        </div>""", unsafe_allow_html=True)


# ==========================================
# PAGE: OPTIONS SCANNER
# ==========================================
elif page == "Options Scanner":
    _section_header("Options Analytics — Market Scanner")
    sc1, sc2, sc3, sc4 = st.columns(4)
    scan_type    = sc1.radio("Option Type", ["Call", "Put", "Both"])
    holding_days = sc2.slider("Holding (days)", 5, 60, 21)
    target_pct   = sc3.slider("Target Move (%)", 1, 20, 5)
    scan_sl      = sc4.slider("Stop Loss (%)", 0, 100, 20) / 100
    min_score    = st.slider("Minimum Score", 0.0, 3.0, 1.5, step=0.5)
    st.divider()
    if st.button("Run Options Scan", type="primary"):
        types_to_scan = ["Call", "Put"] if scan_type == "Both" else [scan_type]
        all_results   = []
        progress      = st.progress(0, text="Scanning options market...")
        total = len(SCAN_UNIVERSE) * len(types_to_scan); done = 0
        for sym in SCAN_UNIVERSE:
            for ot in types_to_scan:
                r = scan_single_ticker(sym, ot, holding_days, target_pct, scan_sl)
                if r and r["Score"] >= min_score: all_results.append(r)
                done += 1
                progress.progress(done / total, text=f"Scanning {sym} {ot}...")
        progress.empty()
        st.session_state["last_scan"] = all_results
        if not all_results: st.warning("No opportunities found. Try lowering the minimum score.")
        else: _render_options_scan_results(all_results)
    elif st.session_state.get("last_scan"):
        _render_options_scan_results(st.session_state["last_scan"])
    else:
        st.info("Configure parameters above and click **Run Options Scan**.")


# ==========================================
# PAGE: OPTIONS WATCHLIST
# ==========================================
elif page == "Options Watchlist":
    _section_header("Options Analytics — Watchlist")
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = ["AAPL", "NVDA", "SPY", "TSLA", "META"]
    ac, bc = st.columns([3, 1])
    with ac: new_t = st.text_input("Add Ticker", placeholder="e.g. MSFT").upper().strip()
    with bc:
        st.write(""); st.write("")
        if st.button("Add") and new_t and new_t not in st.session_state.watchlist:
            st.session_state.watchlist.append(new_t); st.rerun()
    rc, rbc = st.columns([3, 1])
    with rc: rm_t = st.selectbox("Remove", ["---"] + st.session_state.watchlist)
    with rbc:
        st.write(""); st.write("")
        if st.button("Remove") and rm_t != "---":
            st.session_state.watchlist.remove(rm_t); st.rerun()
    st.caption(f"Watching: {', '.join(st.session_state.watchlist)}")
    st.divider()
    if st.button("Analyze Options Watchlist", type="primary"):
        results = []
        with st.spinner("Running Fear Z analysis..."):
            for sym in st.session_state.watchlist:
                r = analyze_watchlist_ticker(sym)
                results.append(r if r else {"Symbol": sym, "Price": "N/A", "IVR": "N/A", "Regime": "Error", "Shelf": "---", "Gamma": "---"})
        df = pd.DataFrame(results)
        if "Regime" in df.columns:
            df["Regime"] = df["Regime"].apply(lambda x: f"{REGIME_ICON.get(x,'?')} {x}")
        st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.info("Click **Analyze Options Watchlist** to run Fear Z scoring.")


# ==========================================
# PAGE: STOCK ADVISOR
# ==========================================
elif page == "Stock Advisor":
    _section_header("Stock Analytics — Stock Advisor")
    sa_ticker  = st.session_state.get("sa_ticker", "AAPL")
    sa_holding = _sa_holding if '_sa_holding' in dir() else 21
    sa_target  = _sa_target  if '_sa_target'  in dir() else 5.0
    sa_shares  = _sa_shares  if '_sa_shares'  in dir() else 100

    if not sa_ticker: st.info("Enter a ticker symbol in the sidebar."); st.stop()
    _, _, sa_spot, sa_rf, sa_m_t0, sa_ivr, sa_vol_hist, sa_hist = fetch_ticker_resource(sa_ticker)
    if sa_spot is None: st.error(f"No data found for '{sa_ticker}'."); st.stop()

    sa_regime    = fz.classify_shock(sa_ivr)
    sma21        = sa_hist["Close"].rolling(21).mean().iloc[-1]
    price_vs_sma = (sa_spot / sma21) - 1 if sma21 > 0 else 0
    sma_label    = "Above 21-SMA" if price_vs_sma >= 0 else "Below 21-SMA"
    high_52w     = sa_hist["Close"].tail(252).max()
    low_52w      = sa_hist["Close"].tail(252).min()
    pos_52w      = ((sa_spot - low_52w) / (high_52w - low_52w) * 100) if high_52w > low_52w else 50
    day_hist     = sa_hist["Close"].tail(2)
    day_chg_pct  = ((day_hist.iloc[-1] / day_hist.iloc[-2]) - 1) * 100 if len(day_hist) >= 2 else 0
    day_chg_dol  = day_hist.iloc[-1] - day_hist.iloc[-2] if len(day_hist) >= 2 else 0
    realized_vol = float(sa_vol_hist.iloc[-1]) if sa_vol_hist is not None and not sa_vol_hist.empty else 0.25
    adj_time     = max(sa_holding, 1) / 365
    adj_piv      = max(realized_vol * np.sqrt(adj_time), 0.0001)
    sa_drift     = (sa_rf - 0.5 * realized_vol**2) * adj_time
    sa_target_p  = sa_spot * (1 + sa_target / 100)
    t_z_s        = (np.log(sa_target_p / sa_spot) - sa_drift) / adj_piv
    p_target_s   = 1 - norm.cdf(t_z_s)
    sa_gain      = (sa_target_p - sa_spot) * sa_shares
    sa_ev        = p_target_s * sa_gain
    sa_score, sa_verdict, sa_rules = stock_advisor_verdict(sa_m_t0, price_vs_sma, sa_ivr, sa_regime)

    r1 = st.columns(4)
    r1[0].metric("Spot Price",    f"${sa_spot:.2f}")
    r1[1].metric("Day Change",    f"${day_chg_dol:+.2f}", delta=f"{day_chg_pct:+.2f}%", delta_color="normal" if day_chg_pct >= 0 else "inverse")
    r1[2].metric("52W Position",  f"{pos_52w:.0f}%", delta="of 52W range", delta_color="off")
    r1[3].metric("IV Rank",       f"{sa_ivr:.0f}")
    r2 = st.columns(4)
    r2[0].metric("5d Momentum",   f"{sa_m_t0*100:+.1f}%", delta_color="normal" if sa_m_t0 >= 0 else "inverse")
    r2[1].metric("21-SMA Status", sma_label, delta=f"{price_vs_sma*100:+.1f}%", delta_color="normal" if price_vs_sma >= 0 else "inverse")
    r2[2].metric("P(Target)",     f"{p_target_s:.1%}")
    r2[3].metric("Expected Value", f"${sa_ev:,.0f}")

    st.divider()
    _section_header("Stock Advisor Verdict")
    sa_vc = VERDICT_COLOR[sa_verdict]; sa_vd_cl = f"vd-{sa_verdict.lower()}"
    v_col, r_col = st.columns([1, 2])
    with v_col:
        st.markdown(f"""<div class="verdict-display {sa_vd_cl}">
            <div style="font-size:2.8rem;font-weight:900;color:{sa_vc};">{sa_verdict}</div>
            <div style="margin-top:8px;">{_verdict_badge(sa_verdict)}</div>
            <div style="font-size:0.9rem;opacity:0.7;margin-top:8px;">{sa_score:.1f} / 3.0 pts</div>
        </div>""", unsafe_allow_html=True)
    with r_col:
        for rule in sa_rules:
            st.markdown(f"**{RESULT_ICON[rule['result']]} {rule['rule']}** — <span style='color:{RESULT_COLOR[rule['result']]}'>{rule['result']}</span>: {rule['detail']}", unsafe_allow_html=True)

    _sa_btn1, _sa_btn2 = st.columns(2)
    if _sa_btn1.button("Get AI Stock Reasoning", type="secondary"):
        generate_stock_reasoning(sa_ticker, sa_spot, sa_m_t0, sa_ivr, sa_regime,
                                 sa_verdict, sa_score, sa_target, sa_holding, placeholder=st.empty())
    if _sa_btn2.button("Backtest This Stock", type="secondary", key="sa_bt_btn"):
        st.session_state["bt_prefill"] = {"ticker": sa_ticker, "mode": "stock", "direction": "long",
            "holding": sa_holding, "target": max(2, min(int(sa_target), 20)), "sl": 10}
        st.session_state.nav_section = "Stock Analytics"; st.session_state.nav_page = "Backtest"; st.rerun()

    st.divider()
    _section_header("Deep Fundamental Analysis")
    _fund_key = f"fundamentals_{sa_ticker}"; _fund_ai_key = f"fund_ai_{sa_ticker}"
    if st.button("Run Deep Analysis", type="primary", key="sa_run_fund"):
        with st.spinner("Fetching financials..."):
            _fin_data = fetch_financials(sa_ticker); _scored = score_fundamentals(_fin_data)
        if _scored is None: st.error(f"Could not fetch financial statements for {sa_ticker}.")
        else:
            st.session_state[_fund_key] = _scored
            if _fund_ai_key in st.session_state: del st.session_state[_fund_ai_key]
    if _fund_key not in st.session_state:
        st.caption("Click **Run Deep Analysis** to score Income Statement, Balance Sheet, and Cash Flow.")
    else:
        _sc = st.session_state[_fund_key]
        def _fa_badge(score):
            if score == 1.0: return '<span style="background:#1a3d2b;color:#00ffcc;border-radius:4px;padding:1px 8px;font-size:0.68rem;font-weight:700;">+ Strong</span>'
            if score == 0.5: return '<span style="background:#3d340a;color:#f5c842;border-radius:4px;padding:1px 8px;font-size:0.68rem;font-weight:700;">~ OK</span>'
            return '<span style="background:#3d1a1a;color:#ff6b6b;border-radius:4px;padding:1px 8px;font-size:0.68rem;font-weight:700;">x Weak</span>'
        def _fa_fmt(name, val):
            if val is None: return "N/A"
            pct_set = {"ROE","Net Margin","Gross Margin","Revenue Growth","Gross Margin Trend","Operating Margin","EPS Growth"}
            if name in pct_set: return f"{val*100:.1f}%"
            if name == "Free Cash Flow": return f"${val/1e9:.2f}B" if abs(val) > 1e8 else f"${val/1e6:.1f}M"
            return f"{val:.2f}"
        _fh, _fq, _fg, _fs = st.columns(4)
        with _fh:
            st.markdown(f"**Financial Health** &nbsp; <span style='color:#bfa15d;font-size:0.82rem;'>{_sc['health_score']:.1f} / 4.0</span>", unsafe_allow_html=True)
            for _mn, (_mv, _ms, _mt) in _sc["health"].items():
                st.markdown(f"{_fa_badge(_ms)} **{_mn}**: {_fa_fmt(_mn, _mv)}<br><small style='opacity:0.45;font-size:0.65rem;'>{_mt}</small>", unsafe_allow_html=True)
        with _fq:
            st.markdown(f"**Profitability & Cash** &nbsp; <span style='color:#bfa15d;font-size:0.82rem;'>{_sc['quality_score']:.1f} / 5.0</span>", unsafe_allow_html=True)
            for _mn, (_mv, _ms, _mt) in _sc["quality"].items():
                st.markdown(f"{_fa_badge(_ms)} **{_mn}**: {_fa_fmt(_mn, _mv)}<br><small style='opacity:0.45;font-size:0.65rem;'>{_mt}</small>", unsafe_allow_html=True)
        with _fg:
            st.markdown(f"**Growth / GARP** &nbsp; <span style='color:#bfa15d;font-size:0.82rem;'>{_sc['growth_score']:.1f} / 4.0</span>", unsafe_allow_html=True)
            for _mn, (_mv, _ms, _mt) in _sc["growth"].items():
                st.markdown(f"{_fa_badge(_ms)} **{_mn}**: {_fa_fmt(_mn, _mv)}<br><small style='opacity:0.45;font-size:0.65rem;'>{_mt}</small>", unsafe_allow_html=True)
        with _fs:
            _fc = "#00ffcc" if _sc["total_score"] >= 7 else ("#f5c842" if _sc["total_score"] >= 5 else "#ff6b6b")
            st.markdown(f"""<div style="text-align:center;padding:18px 8px;border:1px solid rgba(191,161,93,0.25);border-radius:8px;margin-top:4px;">
                <div style="font-size:0.7rem;color:#bfa15d;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:6px;">Fundamental Score</div>
                <div style="font-size:3rem;font-weight:900;color:{_fc};line-height:1;">{_sc['total_score']}</div>
                <div style="font-size:0.78rem;opacity:0.55;margin-top:2px;">/ 10.0</div>
            </div>""", unsafe_allow_html=True)
        if st.button("Generate AI Fundamental Analysis", type="secondary", key="sa_fund_ai_btn"):
            _fund_text = generate_fundamental_reasoning(sa_ticker, _sc, sa_verdict, sa_score, placeholder=st.empty())
            st.session_state[_fund_ai_key] = _fund_text
        elif _fund_ai_key in st.session_state:
            st.markdown(f'<div class="briefing-card">{st.session_state[_fund_ai_key]}</div>', unsafe_allow_html=True)

    st.divider()
    _section_header("Live Chart")
    sc1, sc2, sc3 = st.columns([4, 1, 1.5])
    with sc1: st.subheader(sa_ticker)
    with sc2: sa_tf = st.selectbox("Timeframe", ["1 Day","5 Days","1 Month","6 Months","1 Year","5 Years"], index=4, key="sa_tf")
    with sc3: sa_indicators = st.multiselect("Indicators", ["21 SMA","200 SMA","21 EMA","VWAP"], default=["21 SMA"], key="sa_indicators")
    sa_chart = fetch_chart_data(sa_ticker, sa_tf)
    sa_chart["SMA_21"]  = sa_chart["Close"].rolling(21).mean()
    sa_chart["SMA_200"] = sa_chart["Close"].rolling(200).mean()
    sa_chart["EMA_21"]  = sa_chart["Close"].ewm(span=21, adjust=False).mean()
    _sa_tp = (sa_chart["High"] + sa_chart["Low"] + sa_chart["Close"]) / 3
    sa_chart["VWAP"] = (_sa_tp * sa_chart["Volume"]).cumsum() / sa_chart["Volume"].cumsum()
    sa_x_breaks = [dict(bounds=["sat","mon"]), dict(bounds=[16,9.5], pattern="hour")] if sa_tf in ["1 Day","5 Days"] else [dict(bounds=["sat","mon"])]
    fig_sa = go.Figure(data=[go.Candlestick(x=sa_chart.index, open=sa_chart["Open"], high=sa_chart["High"],
        low=sa_chart["Low"], close=sa_chart["Close"], increasing_line_color="#00ffcc", decreasing_line_color="#ff4b4b", name="Price")])
    if "21 SMA"  in sa_indicators: fig_sa.add_trace(go.Scatter(x=sa_chart.index, y=sa_chart["SMA_21"],  mode="lines", line=dict(color="#bfa15d", width=1.5), name="21 SMA"))
    if "200 SMA" in sa_indicators: fig_sa.add_trace(go.Scatter(x=sa_chart.index, y=sa_chart["SMA_200"], mode="lines", line=dict(color="#7b68ee", width=1.5), name="200 SMA"))
    if "21 EMA"  in sa_indicators: fig_sa.add_trace(go.Scatter(x=sa_chart.index, y=sa_chart["EMA_21"],  mode="lines", line=dict(color="#00d9ff", width=1.5), name="21 EMA"))
    if "VWAP"    in sa_indicators: fig_sa.add_trace(go.Scatter(x=sa_chart.index, y=sa_chart["VWAP"],    mode="lines", line=dict(color="#ff8c42", width=1.5), name="VWAP"))
    fig_sa.add_hline(y=sa_target_p, line_dash="dash", line_color="#00ffcc", opacity=0.6)
    fig_sa.add_hline(y=sa_spot,     line_dash="dot",  line_color="#bfa15d")
    fig_sa.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=10,b=0), hovermode="x unified",
        yaxis=dict(title="Price ($)", gridcolor="rgba(255,255,255,0.08)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", rangebreaks=sa_x_breaks),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_sa, use_container_width=True, config={"scrollZoom": False, "displayModeBar": False})
    st.divider()

    sa_l, sa_r = st.columns([2, 3])
    with sa_l:
        _section_header("Trade Analysis")
        for k, v in {"Shares": sa_shares, "Entry Cost": f"${sa_spot * sa_shares:,.2f}",
                     "Target Price": f"${sa_target_p:.2f} (+{sa_target:.1f}%)",
                     "Expected Gain": f"${sa_gain:,.2f}", "Expected Value": f"${sa_ev:,.2f}"}.items():
            if k == "Expected Value":
                color = "#00d96f" if sa_ev >= 0 else "#ff4b4b"
                st.markdown(f"**{k}:** <span style='color:{color};font-weight:bold;'>{v}</span>", unsafe_allow_html=True)
            else: st.write(f"**{k}:** {v}")
        st.markdown("---")
        st.markdown("**Kelly Position Sizing**")
        _sk_account = st.number_input("Account Size ($)", min_value=1000.0, max_value=10_000_000.0,
                                      value=st.session_state.get("kelly_account", 25000.0), step=1000.0, key="sa_kelly_account")
        st.session_state.kelly_account = _sk_account
        _sk_win_prob = 0.30 + (sa_score / 3.0) * 0.45
        _sk_avg_win  = max(sa_gain, 1)
        _sk_avg_loss = max(sa_spot * sa_shares * 0.10, 1)
        _skd         = kelly_position_size(_sk_win_prob, _sk_avg_win, _sk_avg_loss, _sk_account)
        _sk_shares_k = max(1, int(_skd["recommended_dollars"] / max(sa_spot, 1)))
        _ska, _skb   = st.columns(2)
        _ska.metric("Kelly $",      f"${_skd['recommended_dollars']:,.0f}")
        _skb.metric("Kelly Shares", str(_sk_shares_k))
        st.caption(_skd["note"])
        if st.button("Save Stock Position"):
            if "positions" not in st.session_state: st.session_state.positions = []
            st.session_state.positions.append({"Symbol": sa_ticker, "Type": "Stock", "Strike": "---",
                "Expiration": "---", "Entry Premium": round(sa_spot, 2), "Target": sa_target_p,
                "Contracts": sa_shares, "Stop Loss": "---", "Opened": date.today().isoformat()})
            st.success("Stock position saved.")

        st.markdown("---")
        st.markdown("**Add to ETF Builder**")
        _etf_amt_col, _etf_btn_col = st.columns([2, 1])
        with _etf_amt_col:
            _sa_etf_dollars = st.number_input(
                "Dollar Amount ($)", min_value=0.01, value=10_000.0, step=1_000.0,
                format="%.2f", key="sa_etf_dollars",
                help="Dollar amount to allocate to this security in your ETF portfolio."
            )
        with _etf_btn_col:
            st.write("")
            if st.button("Add to ETF Builder", type="secondary", key="sa_add_etf"):
                if "etf_portfolio" not in st.session_state:
                    st.session_state.etf_portfolio = []
                _existing_etf = next((i for i, h in enumerate(st.session_state.etf_portfolio) if h["ticker"] == sa_ticker), None)
                if _existing_etf is not None:
                    st.session_state.etf_portfolio[_existing_etf]["dollars"] = _sa_etf_dollars
                    st.success(f"Updated {sa_ticker} in ETF Builder to ${_sa_etf_dollars:,.0f}.")
                else:
                    st.session_state.etf_portfolio.append({"ticker": sa_ticker, "dollars": _sa_etf_dollars})
                    st.success(f"{sa_ticker} added to ETF Builder at ${_sa_etf_dollars:,.0f}.")
                st.session_state.pop("etf_results", None)
        if st.session_state.get("etf_portfolio"):
            _etf_tickers_now = [h["ticker"] for h in st.session_state["etf_portfolio"]]
            _etf_total_now   = sum(h["dollars"] for h in st.session_state["etf_portfolio"])
            st.caption(f"ETF Builder: {len(_etf_tickers_now)} securities, ${_etf_total_now:,.0f} total — {', '.join(_etf_tickers_now)}")
            if st.button("Go to ETF Builder", key="sa_goto_etf"):
                st.session_state.nav_section = "Institutional"
                st.session_state.nav_page    = "ETF Builder"
                st.rerun()

    with sa_r:
        _section_header(f"Price Distribution ({sa_holding} Days, 10,000 Simulations)")
        sa_sims = np.random.lognormal(np.log(sa_spot) + sa_drift, adj_piv, 10000)
        sa_p5, sa_p95 = np.percentile(sa_sims, [5, 95])
        fig_sh = go.Figure()
        fig_sh.add_vrect(x0=sa_p5, x1=sa_p95, fillcolor="#bfa15d", opacity=0.12, layer="below", line_width=0)
        fig_sh.add_trace(go.Histogram(x=sa_sims, nbinsx=150, marker_color="#bfa15d", opacity=0.75))
        fig_sh.add_vline(x=sa_spot,     line_dash="dash", line_color="#ffffff", opacity=0.8)
        fig_sh.add_vline(x=sa_target_p, line_dash="dot",  line_color="#00ffcc")
        fig_sh.update_layout(title=f"{sa_ticker} Distribution in {sa_holding} Days",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Price ($)"), yaxis=dict(title="Frequency"),
            showlegend=False, bargap=0.2, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_sh, use_container_width=True, config={"scrollZoom": False, "displayModeBar": False})
        st.markdown(f"""<div class="briefing-card">
            <strong>Simulation Insight:</strong> In a <strong>{sa_regime}</strong> behavioral regime,
            Monte Carlo projects <strong style="color:#bfa15d;">{p_target_s:.1%}</strong> probability
            of reaching <strong>${sa_target_p:.2f}</strong> (+{sa_target:.1f}%) over <strong>{sa_holding} days</strong>.
            Stock Advisor: <strong style="color:{sa_vc};">{sa_verdict}</strong> ({sa_score:.1f}/3.0).
        </div>""", unsafe_allow_html=True)


# ==========================================
# PAGE: STOCK SCANNER
# ==========================================
elif page == "Stock Scanner":
    _section_header("Stock Analytics — Stock Scanner")
    ss1, ss2, ss3 = st.columns(3)
    ss_holding   = ss1.slider("Holding Period (days)", 5, 120, 21, key="ss_hold")
    ss_target    = ss2.slider("Target Move (%)", 1, 30, 5, key="ss_tgt")
    ss_min_score = ss3.slider("Minimum Score", 0.0, 3.0, 1.5, step=0.5, key="ss_min")
    st.divider()
    if st.button("Run Stock Scan", type="primary"):
        all_results = []
        progress    = st.progress(0, text="Scanning equities...")
        for i, sym in enumerate(SCAN_UNIVERSE):
            r = scan_single_stock(sym, ss_holding, ss_target)
            if r and r["Score"] >= ss_min_score: all_results.append(r)
            progress.progress((i + 1) / len(SCAN_UNIVERSE), text=f"Scanning {sym}...")
        progress.empty()
        st.session_state["last_stock_scan"] = all_results
        if not all_results: st.warning("No stocks found above the minimum score.")
        else: _render_stock_scan_results(all_results)
    elif st.session_state.get("last_stock_scan"):
        _render_stock_scan_results(st.session_state["last_stock_scan"])
    else:
        st.info("Click **Run Stock Scan** to find the best equity opportunities.")


# ==========================================
# PAGE: STOCK WATCHLIST
# ==========================================
elif page == "Stock Watchlist":
    _section_header("Stock Analytics — Watchlist")
    if "stock_watchlist" not in st.session_state:
        st.session_state.stock_watchlist = ["AAPL", "NVDA", "MSFT", "TSLA", "GOOGL"]
    ac, bc = st.columns([3, 1])
    with ac: new_st = st.text_input("Add Ticker", placeholder="e.g. AMZN", key="sw_add").upper().strip()
    with bc:
        st.write(""); st.write("")
        if st.button("Add", key="sw_addbtn") and new_st and new_st not in st.session_state.stock_watchlist:
            st.session_state.stock_watchlist.append(new_st); st.rerun()
    rc, rbc = st.columns([3, 1])
    with rc: rm_st = st.selectbox("Remove", ["---"] + st.session_state.stock_watchlist, key="sw_rm")
    with rbc:
        st.write(""); st.write("")
        if st.button("Remove", key="sw_rmbtn") and rm_st != "---":
            st.session_state.stock_watchlist.remove(rm_st); st.rerun()
    st.caption(f"Watching: {', '.join(st.session_state.stock_watchlist)}")
    st.divider()
    if st.button("Analyze Stock Watchlist", type="primary"):
        results = []
        with st.spinner("Running stock analysis..."):
            for sym in st.session_state.stock_watchlist:
                r = scan_single_stock(sym, 21, 5)
                results.append(r if r else {"Symbol": sym, "Spot": "N/A", "IVR": "N/A",
                                            "Regime": "Error", "Momentum": "---", "SMA21": "---",
                                            "Score": "---", "Verdict": "---", "P(Target)": "---"})
        df = pd.DataFrame(results)
        if "Regime" in df.columns:
            df["Regime"] = df["Regime"].apply(lambda x: f"{REGIME_ICON.get(x,'?')} {x}" if isinstance(x, str) else x)
        st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.info("Click **Analyze Stock Watchlist** to run scoring on all tickers.")


# ==========================================
# PAGE: BACKTEST
# ==========================================
elif page == "Backtest":
    _section_header("Stock Analytics — Strategy Backtest")
    _bt_pre  = st.session_state.pop("bt_prefill", {})
    _bt_mode = _bt_pre.get("mode", "stock")
    _bt_dir  = _bt_pre.get("direction", "long")
    _tab_simple, _tab_wf, _tab_dict = st.tabs(["Simple Backtest", "Walk-Forward Analysis", "Dictionary & AI Explanation"])

    with _tab_simple:
        if _bt_mode == "options":
            _dir_label = "CALL — Long Bias" if _bt_dir == "long" else "PUT — Short Bias"
            st.info(f"**Options Backtest: {_dir_label}**")
        bt1, bt2, bt3, bt4 = st.columns(4)
        bt_ticker   = bt1.text_input("Ticker", value=_bt_pre.get("ticker", "AAPL"), key="bt_ticker").upper().strip()
        bt_lookback = bt2.selectbox("Lookback", [63, 126, 252], index=2, format_func=lambda x: f"{x}d (~{x//21}mo)")
        bt_holding  = bt3.slider("Holding Period (days)", 5, 60, _bt_pre.get("holding", 21), key="bt_hold")
        bt_target   = bt4.slider("Target Move (%)", 2, 20, _bt_pre.get("target", 5), key="bt_tgt")
        bt_sl       = st.slider("Stop Loss (%)", 2, 30, _bt_pre.get("sl", 10), key="bt_sl")
        st.divider()
        if st.button("Run Backtest", type="primary", key="run_simple_bt"):
            with st.spinner(f"Backtesting {bt_ticker}..."):
                _bt_result = run_backtest(bt_ticker, bt_lookback, bt_holding, bt_target, bt_sl, direction=_bt_dir)
            st.session_state["last_backtest"] = _bt_result
            st.session_state["last_backtest_params"] = {"ticker": bt_ticker, "lookback": bt_lookback,
                "holding": bt_holding, "target": bt_target, "sl": bt_sl, "mode": _bt_mode, "direction": _bt_dir}
        _bt   = st.session_state.get("last_backtest")
        _bt_p = st.session_state.get("last_backtest_params", {})
        if _bt is None: st.info("Configure parameters and click **Run Backtest**.")
        elif not _bt.get("trades"): st.warning("No BUY signals generated. Try a longer lookback.")
        else:
            _s = _bt["stats"]; _bt_sym = _bt_p.get("ticker", bt_ticker)
            st.success(f"Backtest complete: **{_bt_sym}** — {_s['Total Trades']} trades")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Win Rate",        f"{_s['Win Rate']:.1f}%")
            m2.metric("Profit Factor",   f"{_s['Profit Factor']:.2f}")
            m3.metric("Total Return",    f"{_s['Total Return %']:+.1f}%")
            m4.metric("Target Hit Rate", f"{_s['Target Hit Rate']:.1f}%")
            m5, m6, m7, m8 = st.columns(4)
            m5.metric("Total Trades",  str(_s['Total Trades']))
            m6.metric("Avg Win %",     f"+{_s['Avg Win %']:.2f}%")
            m7.metric("Avg Loss %",    f"{_s['Avg Loss %']:.2f}%")
            m8.metric("Stop Out Rate", f"{_s['Stop Out Rate']:.1f}%")
            st.divider()
            _section_header("Cumulative P&L Curve")
            _eq_df = pd.DataFrame(_bt["equity_curve"])
            if not _eq_df.empty:
                _lc  = "#00d96f" if _s["Total Return %"] >= 0 else "#ff4b4b"
                _rgb = tuple(int(_lc.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(x=_eq_df["Date"], y=_eq_df["Cumulative P&L %"],
                    mode="lines+markers", line=dict(color=_lc, width=2), marker=dict(size=5, color=_lc),
                    fill="tozeroy", fillcolor=f"rgba({_rgb[0]},{_rgb[1]},{_rgb[2]},0.08)"))
                fig_eq.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                fig_eq.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(title="Cumulative P&L (%)", gridcolor="rgba(255,255,255,0.08)"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.08)"), margin=dict(l=0,r=0,t=10,b=0), showlegend=False)
                st.plotly_chart(fig_eq, use_container_width=True, config={"displayModeBar": False})
            st.divider()
            _section_header("Trade Log")
            _trade_df = pd.DataFrame(_bt["trades"])
            if not _trade_df.empty:
                def _color_pnl(val):
                    if isinstance(val, (int, float)): return f"color: {'#00d96f' if val > 0 else '#ff4b4b'}"
                    return ""
                st.dataframe(_trade_df.style.applymap(_color_pnl, subset=["P&L %"]), hide_index=True, use_container_width=True)
                st.download_button("Download Trade Log CSV", _trade_df.to_csv(index=False), f"{_bt_sym}_backtest.csv", "text/csv")
            st.divider()
            _section_header("Performance by Fear Z Regime")
            _rg_groups = {}
            for _tr in _bt["trades"]: _rg_groups.setdefault(_tr["Regime"], []).append(_tr["P&L %"])
            _rg_rows = [{"Regime": f"{REGIME_ICON.get(_rg,'?')} {_rg}", "Trades": len(_ps),
                         "Win Rate": f"{sum(1 for p in _ps if p>0)/len(_ps)*100:.0f}%",
                         "Avg Return": f"{np.mean(_ps):+.2f}%", "Total": f"{sum(_ps):+.2f}%"}
                        for _rg, _ps in _rg_groups.items()]
            if _rg_rows: st.dataframe(pd.DataFrame(_rg_rows), hide_index=True, use_container_width=True)

    with _tab_wf:
        wf1, wf2, wf3, wf4, wf5 = st.columns(5)
        wf_ticker  = wf1.text_input("Ticker", value=_bt_pre.get("ticker", "AAPL"), key="wf_ticker").upper().strip()
        wf_is      = wf2.selectbox("IS Window", [6, 9, 12, 18], index=2, format_func=lambda x: f"{x} mo", key="wf_is")
        wf_oos     = wf3.selectbox("OOS Window", [1, 2, 3, 6], index=2, format_func=lambda x: f"{x} mo", key="wf_oos")
        wf_holding = wf4.slider("Holding (days)", 5, 60, _bt_pre.get("holding", 21), key="wf_hold")
        wf_target  = wf5.slider("Target (%)", 2, 20, _bt_pre.get("target", 5), key="wf_tgt")
        wf_sl      = st.slider("Stop Loss (%)", 2, 30, _bt_pre.get("sl", 10), key="wf_sl")
        wf_dir     = _bt_dir
        st.divider()
        if st.button("Run Walk-Forward", type="primary", key="run_wf"):
            with st.spinner(f"Running walk-forward on {wf_ticker}..."):
                _wf_result = run_walkforward(wf_ticker, wf_is, wf_oos, wf_holding, wf_target, wf_sl, direction=wf_dir)
            st.session_state["last_wf"] = _wf_result
            st.session_state["last_wf_ticker"] = wf_ticker
        _wf = st.session_state.get("last_wf")
        _wf_sym = st.session_state.get("last_wf_ticker", wf_ticker)
        if _wf is None: st.info("Configure parameters and click **Run Walk-Forward**.")
        else:
            wm1, wm2, wm3, wm4 = st.columns(4)
            wm1.metric("Folds Completed",     str(_wf["n_folds"]))
            wm2.metric("Total OOS Trades",    str(_wf["total_oos_trades"]))
            wm3.metric("Aggregate OOS Sharpe", f"{_wf['agg_sharpe']:.2f}")
            wm4.metric("Worst Fold Drawdown",  f"{_wf['worst_fold_dd']:+.1f}%")
            st.divider()
            _section_header("Fold-by-Fold OOS Results")
            _fold_hdr = st.columns([0.5, 2.5, 1.8, 0.8, 1.1, 1.3, 1.2, 1.1])
            for _fc2, _fl in zip(_fold_hdr, ["#","Period","Regime","Trades","Win Rate","Avg Return","Max DD","Sharpe"]):
                _fc2.markdown(f"<div style='font-size:0.68rem;color:#bfa15d;text-transform:uppercase;padding-bottom:4px;border-bottom:1px solid rgba(191,161,93,0.25);'>{_fl}</div>", unsafe_allow_html=True)
            for _f in _wf["folds"]:
                _ri  = REGIME_ICON.get(_f["dominant_regime"], "?")
                _rc  = "#00d96f" if _f["avg_return"] > 0 else "#ff4b4b"
                _sc2 = "#00d96f" if _f["sharpe"] >= 1.0 else ("#FFC107" if _f["sharpe"] >= 0 else "#ff4b4b")
                _ddc = "#00d96f" if _f["max_dd"] >= -5 else ("#FFC107" if _f["max_dd"] >= -15 else "#ff4b4b")
                _frow = st.columns([0.5, 2.5, 1.8, 0.8, 1.1, 1.3, 1.2, 1.1])
                _frow[0].markdown(f"<div style='padding:5px 0;opacity:0.5;font-size:0.82rem;'>{_f['fold']}</div>", unsafe_allow_html=True)
                _frow[1].markdown(f"<div style='padding:5px 0;font-size:0.82rem;'>{_f['period']}</div>", unsafe_allow_html=True)
                _frow[2].markdown(f"<div style='padding:5px 0;font-size:0.82rem;'>{_ri} {_f['dominant_regime']}</div>", unsafe_allow_html=True)
                _frow[3].markdown(f"<div style='padding:5px 0;font-size:0.82rem;'>{_f['trades']}</div>", unsafe_allow_html=True)
                _frow[4].markdown(f"<div style='padding:5px 0;font-size:0.82rem;'>{_f['win_rate']:.0f}%</div>", unsafe_allow_html=True)
                _frow[5].markdown(f"<div style='padding:5px 0;font-size:0.82rem;color:{_rc};font-weight:600;'>{_f['avg_return']:+.2f}%</div>", unsafe_allow_html=True)
                _frow[6].markdown(f"<div style='padding:5px 0;font-size:0.82rem;color:{_ddc};'>{_f['max_dd']:+.1f}%</div>", unsafe_allow_html=True)
                _frow[7].markdown(f"<div style='padding:5px 0;font-size:0.82rem;color:{_sc2};font-weight:600;'>{_f['sharpe']:.2f}</div>", unsafe_allow_html=True)
            st.divider()
            _section_header("Regime Attribution")
            _ra_df = pd.DataFrame(_wf["regime_attribution"])
            if not _ra_df.empty: st.dataframe(_ra_df, hide_index=True, use_container_width=True)
            _wf_export = pd.DataFrame([{"Fold": f["fold"], "Period": f["period"], "Regime": f["dominant_regime"],
                 "Trades": f["trades"], "Win Rate %": f["win_rate"], "Avg Return %": f["avg_return"],
                 "Max DD %": f["max_dd"], "Sharpe": f["sharpe"]} for f in _wf["folds"]])
            st.download_button("Download Walk-Forward CSV", _wf_export.to_csv(index=False), f"{_wf_sym}_walkforward.csv", "text/csv")

    with _tab_dict:
        _bt_res   = st.session_state.get("last_backtest")
        _bt_par   = st.session_state.get("last_backtest_params", {})
        _wf_res   = st.session_state.get("last_wf")
        _wf_sym_d = st.session_state.get("last_wf_ticker", "")
        _section_header("AI Backtest Explanation")
        _has_simple = _bt_res and _bt_res.get("trades")
        _has_wf     = _wf_res and _wf_res.get("folds")
        if not _has_simple and not _has_wf:
            st.info("Run a Simple Backtest or Walk-Forward first.")
        else:
            _ai_mode = st.radio("Explain results from:", ["Simple Backtest", "Walk-Forward Analysis"], horizontal=True, key="dict_ai_mode")
            if st.button("Generate AI Explanation", type="primary", key="dict_ai_btn"):
                _explain_ph = st.empty()
                if _ai_mode == "Simple Backtest" and _has_simple:
                    _s = _bt_res["stats"]
                    _rg_grp = {}
                    for _t in _bt_res["trades"]: _rg_grp.setdefault(_t["Regime"], []).append(_t["P&L %"])
                    _rg_lines = [f"  {rg}: {len(ps)} trades, avg {sum(ps)/len(ps):+.2f}%" for rg, ps in _rg_grp.items()]
                    _prompt = (
                        f"Backtest results for {_bt_par.get('ticker','N/A')}:\n"
                        f"- Direction: {_bt_par.get('direction','long')} | Hold each trade for: {_bt_par.get('holding',21)} days\n"
                        f"- Target gain: {_bt_par.get('target',5)}% | Stop loss: {_bt_par.get('sl',10)}%\n"
                        f"- Total trades: {_s['Total Trades']} | Win rate: {_s['Win Rate']}%"
                        f" | Profit factor: {_s['Profit Factor']} | Total return: {_s['Total Return %']:+.1f}%\n"
                        f"- Regime breakdown: {'; '.join(_rg_lines)}\n\n"
                        "Explain these results in 4 short bullets:\n"
                        "• 📖 What a backtest is — explain it simply for a first-time investor\n"
                        "• 📊 How did this strategy do? — interpret win rate, profit factor, and total return in plain English\n"
                        "• 🧠 Regime breakdown — explain what each regime means and which performed best\n"
                        "• 💡 One way to improve — a practical suggestion"
                    )
                else:
                    _folds = _wf_res["folds"]; _ra = _wf_res["regime_attribution"]
                    _ra_str = "; ".join(f'{r["Regime"]} avg {r["Avg Return"]} edge {r["Edge"]}' for r in _ra)
                    _prompt = (
                        f"Walk-forward test results for {_wf_sym_d}:\n"
                        f"- Out-of-sample Sharpe: {_wf_res['agg_sharpe']:.2f} | Worst drawdown: {_wf_res['worst_fold_dd']:+.1f}%\n"
                        f"- Number of test periods (folds): {_wf_res['n_folds']} | Total trades tested: {_wf_res['total_oos_trades']}\n"
                        f"- Regime results: {_ra_str}\n\n"
                        "Explain in 5 short bullets:\n"
                        "• 📖 What walk-forward testing means — explain it like you're talking to someone who's never heard of it\n"
                        "• ✅ Is this strategy robust? — interpret the results honestly\n"
                        "• 🧠 Regime breakdown — which market conditions did this strategy thrive or struggle in?\n"
                        "• 📊 What the Sharpe Ratio means here — define it simply and say whether this number is good\n"
                        "• 💡 One practical next step — what should the investor do with this information?"
                    )
                _explain_text = _stream_ai_response(_prompt, max_tokens=700, placeholder=_explain_ph, system=_ADVISOR_SYSTEM)
                st.session_state["bt_ai_explanation"] = _explain_text
            elif "bt_ai_explanation" in st.session_state:
                st.markdown(f'<div class="briefing-card">{st.session_state["bt_ai_explanation"]}</div>', unsafe_allow_html=True)
        st.divider()
        _section_header("Backtesting Dictionary")
        for _term, _def in [
            ("Simple Backtest", "Runs the Stock Advisor 3-rule signal over a single historical window. Tests in-sample data."),
            ("Walk-Forward Analysis", "Anti-overfitting test: rolling IS training + OOS test windows. The honest performance measure."),
            ("Sharpe Ratio", "Mean return / standard deviation, annualized. Above 1.0 is institutional-grade."),
            ("Profit Factor", "Gross profit / gross loss. Above 1.5 strong. Above 2.0 exceptional."),
            ("Max Drawdown", "Largest peak-to-trough cumulative loss in the test period."),
            ("Regime Attribution", "OOS performance decomposed by Fear Z regime. Shows where the edge lives."),
        ]:
            with st.expander(_term):
                st.markdown(f"<div style='font-size:0.9rem;line-height:1.65;opacity:0.88;'>{_def}</div>", unsafe_allow_html=True)


# ==========================================
# PAGE: POSITIONS
# ==========================================
elif page == "Positions":
    _section_header("Account — Positions Tracker")
    if "positions" not in st.session_state: st.session_state.positions = []
    with st.expander("Add New Position", expanded=not st.session_state.positions):
        pc1, pc2, pc3 = st.columns(3)
        p_sym    = pc1.text_input("Symbol").upper().strip()
        p_type   = pc2.selectbox("Type", ["Call", "Put", "Stock"])
        p_strike = pc3.number_input("Strike / Entry ($)", min_value=0.0, step=0.5)
        pc4, pc5, pc6 = st.columns(3)
        p_expiry  = pc4.date_input("Expiration (N/A for stocks)")
        p_premium = pc5.number_input("Entry Premium / Price ($)", min_value=0.0, step=0.01)
        p_target  = pc6.number_input("Target Price ($)", min_value=0.0, step=0.5)
        pc7, pc8  = st.columns(2)
        p_contr   = pc7.number_input("Contracts / Shares", min_value=1, value=1)
        p_sl      = pc8.slider("Stop Loss (%)", 0, 100, 20)
        if st.button("Add Position", type="primary") and p_sym and p_premium > 0:
            st.session_state.positions.append({"Symbol": p_sym, "Type": p_type, "Strike": p_strike,
                "Expiration": str(p_expiry), "Entry Premium": p_premium, "Target": p_target,
                "Contracts": p_contr, "Stop Loss": f"{p_sl}%", "Opened": date.today().isoformat()})
            st.success(f"Added {p_sym} {p_type}"); st.rerun()
    if st.session_state.positions:
        enriched = []
        for pos in st.session_state.positions:
            curr_premium, pnl, roi = pos["Entry Premium"], 0.0, 0.0
            try:
                if pos["Type"] == "Stock":
                    curr_premium = yf.Ticker(pos["Symbol"]).history(period="1d")["Close"].iloc[-1]
                else:
                    t = yf.Ticker(pos["Symbol"])
                    chain_df = t.option_chain(pos["Expiration"])
                    chain_df = chain_df.calls if pos["Type"] == "Call" else chain_df.puts
                    row = chain_df[chain_df["strike"] == pos["Strike"]]
                    if not row.empty: curr_premium = row["lastPrice"].iloc[0]
                pnl = (curr_premium - pos["Entry Premium"]) * pos["Contracts"] * (1 if pos["Type"] == "Stock" else 100)
                roi = ((curr_premium - pos["Entry Premium"]) / pos["Entry Premium"] * 100) if pos["Entry Premium"] > 0 else 0
            except: pass
            enriched.append({**pos, "Current": round(curr_premium, 2), "P&L": round(pnl, 2), "ROI%": round(roi, 1)})
        st.dataframe(pd.DataFrame(enriched), hide_index=True, use_container_width=True)
        _earn_warnings = []
        for _pos in st.session_state.positions:
            _sym = _pos.get("Symbol", "")
            if not _sym or _pos.get("Type") == "Stock": continue
            _ec = fetch_earnings_calendar(_sym)
            if not _ec: continue
            _days_to_earn = _ec["days_away"]; _expiry_str = _pos.get("Expiration", "")
            if _expiry_str and _expiry_str not in ("---", ""):
                try:
                    _dte = (pd.to_datetime(_expiry_str).date() - date.today()).days
                    if 0 <= _days_to_earn <= 14 and _days_to_earn <= _dte:
                        _earn_warnings.append(f"**{_sym}** {_pos['Type']} {_expiry_str}: earnings in {_days_to_earn} days — IV crush risk.")
                except Exception: pass
        if _earn_warnings: st.warning("**Earnings Risk Flags**\n\n" + "\n\n".join(f"! {w}" for w in _earn_warnings))
        if st.button("Clear All Positions"):
            st.session_state.positions = []; st.rerun()
    else:
        st.info("No positions yet. Add one above or use 'Save to Positions' in Trade/Stock Advisor.")




# ==========================================
# PAGE: ETF BUILDER (INSTITUTIONAL)
# ==========================================
elif page == "ETF Builder":
    _section_header("Institutional — ETF Builder & Portfolio Analytics")

    st.markdown("""
<div style="border:1px solid rgba(191,161,93,0.3);border-radius:10px;padding:20px 28px;margin-bottom:20px;background:rgba(191,161,93,0.04);">
  <div style="font-size:0.7rem;color:#bfa15d;letter-spacing:0.18em;text-transform:uppercase;margin-bottom:4px;">Institutional Portfolio Construction</div>
  <div style="font-size:1.1rem;color:#e8dfc8;margin-bottom:8px;">Build and analyze a custom ETF with institutional-grade risk metrics</div>
  <div style="font-size:0.85rem;opacity:0.75;line-height:1.6;">
    Add each security with its own dollar allocation. Edit amounts directly in the table at any time.
    The <strong>Optimal Allocation</strong> engine applies Kelly Criterion position sizing to suggest
    mathematically optimal weights based on each security's historical return distribution.
  </div>
</div>""", unsafe_allow_html=True)

    # ── SESSION STATE INIT ──
    if "etf_portfolio" not in st.session_state:
        st.session_state.etf_portfolio = []

    # Migrate legacy pct-based holdings to dollar-based (use 10,000 per pct point as fallback)
    for _h in st.session_state.etf_portfolio:
        if "pct" in _h and "dollars" not in _h:
            _h["dollars"] = _h["pct"] * 1_000.0
            del _h["pct"]

    # ── ADD SECURITY FORM ──
    _section_header("Portfolio Construction")
    fc1, fc2, fc3 = st.columns([2, 2, 1])
    with fc1:
        new_etf_ticker = st.text_input(
            "Ticker Symbol", placeholder="e.g. AAPL, NVDA, MSFT",
            key="etf_add_ticker"
        ).upper().strip()
    with fc2:
        new_etf_dollars = st.number_input(
            "Dollar Amount ($)", min_value=0.01, value=25_000.0, step=1_000.0,
            format="%.2f", key="etf_add_dollars"
        )
    with fc3:
        st.write(""); st.write("")
        if st.button("Add Security", type="primary", key="etf_add_btn") and new_etf_ticker:
            _existing_idx = next(
                (i for i, h in enumerate(st.session_state.etf_portfolio) if h["ticker"] == new_etf_ticker),
                None
            )
            if _existing_idx is not None:
                st.session_state.etf_portfolio[_existing_idx]["dollars"] = new_etf_dollars
                st.toast(f"Updated {new_etf_ticker} to ${new_etf_dollars:,.0f}.")
            else:
                st.session_state.etf_portfolio.append({"ticker": new_etf_ticker, "dollars": new_etf_dollars})
                st.toast(f"Added {new_etf_ticker}: ${new_etf_dollars:,.0f}.")
            st.session_state.pop("etf_results", None)
            st.rerun()

    # ── CURRENT PORTFOLIO TABLE (inline editable) ──
    if not st.session_state.etf_portfolio:
        st.info("Add securities above to build your ETF portfolio. Example: AAPL $25,000 + MSFT $25,000 + NVDA $25,000 + GOOGL $25,000")
    else:
        _total_dollars = sum(h["dollars"] for h in st.session_state.etf_portfolio)

        # Table header
        _th = st.columns([0.35, 1.6, 2.5, 1.4, 0.85])
        for _col, _lbl in zip(_th, ["#", "Ticker", "Amount ($)", "Weight", ""]):
            _col.markdown(
                f"<span style='color:#bfa15d;font-size:0.72rem;font-weight:700;"
                f"text-transform:uppercase;letter-spacing:0.08em;'>{_lbl}</span>",
                unsafe_allow_html=True
            )

        # Generation counter: incrementing forces new widget keys, making Streamlit
        # treat them as fresh widgets whose value= argument is respected (not stale state).
        _key_gen = st.session_state.get("etf_key_gen", 0)
        _any_changed = False
        for i, h in enumerate(st.session_state.etf_portfolio):
            _row = st.columns([0.35, 1.6, 2.5, 1.4, 0.85])
            _row[0].markdown(
                f"<div style='padding:9px 0;opacity:0.5;font-size:0.82rem;'>{i+1}</div>",
                unsafe_allow_html=True
            )
            _row[1].markdown(f"<div style='padding:9px 0;font-weight:700;'>{h['ticker']}</div>", unsafe_allow_html=True)
            _new_amt = _row[2].number_input(
                label="amt", min_value=0.01, value=max(0.01, float(h["dollars"])),
                step=1_000.0, format="%.2f",
                key=f"etf_amt_{h['ticker']}_{_key_gen}",
                label_visibility="collapsed"
            )
            if abs(_new_amt - h["dollars"]) > 0.001:
                st.session_state.etf_portfolio[i]["dollars"] = _new_amt
                st.session_state.pop("etf_results", None)
                _any_changed = True
            _pct_display = h["dollars"] / _total_dollars * 100 if _total_dollars > 0 else 0
            _row[3].markdown(
                f"<div style='padding:9px 0;color:#bfa15d;font-weight:600;'>{_pct_display:.1f}%</div>",
                unsafe_allow_html=True
            )
            if _row[4].button("Remove", key=f"etf_rm_{i}"):
                st.session_state.etf_portfolio.pop(i)
                st.session_state.pop("etf_results", None)
                st.rerun()

        if _any_changed:
            st.rerun()

        # Summary bar
        st.markdown(
            f"<div style='margin:10px 0;padding:10px 18px;border:1px solid rgba(191,161,93,0.35);"
            f"border-radius:6px;background:rgba(0,0,0,0.15);display:flex;justify-content:space-between;align-items:center;'>"
            f"<span style='font-weight:700;color:#bfa15d;'>{len(st.session_state.etf_portfolio)} securities</span>"
            f"<span style='font-size:1rem;font-weight:700;'>${_total_dollars:,.2f} total deployed</span>"
            f"</div>",
            unsafe_allow_html=True
        )

        st.divider()

        # Action row
        ac1, ac2, ac3 = st.columns([1, 1, 2])
        with ac1:
            if st.button("Clear All", type="secondary", key="etf_clear"):
                st.session_state.etf_portfolio = []
                st.session_state.pop("etf_results", None)
                st.rerun()
        with ac2:
            etf_horizon = st.number_input(
                "Horizon (years)", min_value=1, max_value=30, value=10, key="etf_horizon_input"
            )
        with ac3:
            run_etf = st.button(
                "Analyze ETF Portfolio", type="primary", key="etf_analyze",
                use_container_width=True,
                disabled=(len(st.session_state.etf_portfolio) == 0)
            )

        if run_etf:
            with st.spinner("Downloading price data and computing institutional metrics..."):
                portfolio_snap = {h["ticker"]: h["dollars"] for h in st.session_state.etf_portfolio}
                tickers_snap   = list(portfolio_snap.keys())
                try:
                    today_str = pd.Timestamp.today().strftime("%Y-%m-%d")
                    raw = yf.download(tickers_snap, start="2015-01-01", end=today_str, progress=False)
                    prices = normalize_price_frame(raw)
                    if "SINGLE_TICKER" in prices.columns and len(tickers_snap) == 1:
                        prices = prices.rename(columns={"SINGLE_TICKER": tickers_snap[0]})
                    valid_cols = [t for t in tickers_snap if t in prices.columns]
                    if not valid_cols:
                        st.error("No valid price data returned. Check ticker symbols.")
                    else:
                        dollars_arr = np.array([portfolio_snap[t] for t in valid_cols], dtype=float)
                        w_arr       = dollars_arr / dollars_arr.sum()
                        returns     = prices[valid_cols].pct_change().fillna(0.0)
                        port_return = (returns * w_arr).sum(axis=1)

                        spy_raw    = yf.download("SPY", start="2015-01-01", end=today_str, progress=False)
                        spy_prices = normalize_price_frame(spy_raw)
                        if "SINGLE_TICKER" in spy_prices.columns:
                            spy_prices = spy_prices.rename(columns={"SINGLE_TICKER": "SPY"})
                        spy_ret = spy_prices["SPY"].pct_change().fillna(0.0) if "SPY" in spy_prices.columns else pd.Series(dtype=float)

                        rf_daily = get_risk_free_daily()
                        st.session_state["etf_results"] = {
                            "portfolio":      portfolio_snap,
                            "prices":         prices,
                            "returns":        returns,
                            "port_return":    port_return,
                            "valid_cols":     valid_cols,
                            "weights":        w_arr,
                            "dollars_arr":    dollars_arr,
                            "rf_daily":       rf_daily,
                            "spy_ret":        spy_ret,
                            "spy_prices":     spy_prices,
                            "total_invested": dollars_arr.sum(),
                            "horizon":        etf_horizon,
                        }
                        st.success(f"Analysis complete for {len(valid_cols)} securities.")
                except Exception as e:
                    st.error(f"Error fetching data: {e}")

        # ==========================================
        # OPTIMAL ALLOCATION — KELLY CRITERION
        # ==========================================
        if "etf_results" in st.session_state:
            res_k     = st.session_state["etf_results"]
            returns_k = res_k["returns"]
            valid_k   = res_k["valid_cols"]
            total_k   = res_k["total_invested"]

            st.divider()
            _section_header("Optimal Allocation — Kelly Criterion")
            st.markdown(
                "<div style='font-size:0.85rem;opacity:0.75;line-height:1.6;margin-bottom:12px;'>"
                "The Kelly Criterion maximizes long-run geometric portfolio growth by sizing each position "
                "proportional to its historical edge. <strong>Half-Kelly</strong> is applied and normalized "
                "across the portfolio. Tickers with negative edge receive 0% allocation. "
                "Clicking <strong>Apply Kelly Allocation</strong> redistributes your current total "
                f"investment of <strong>${total_k:,.0f}</strong> according to these weights."
                "</div>", unsafe_allow_html=True
            )

            kelly_rows = []
            kelly_raw_fracs = {}
            for t in valid_k:
                daily_r  = returns_k[t].dropna()
                if len(daily_r) < 50:
                    kelly_raw_fracs[t] = 0.0; continue
                pos_r    = daily_r[daily_r > 0]
                neg_r    = daily_r[daily_r < 0]
                win_rate = len(pos_r) / len(daily_r)
                avg_win  = float(pos_r.mean())  if len(pos_r) > 0 else 0.0
                avg_loss = float(abs(neg_r.mean())) if len(neg_r) > 0 else 1e-9
                b        = avg_win / avg_loss if avg_loss > 0 else 0.0
                q        = 1 - win_rate
                k_full   = (b * win_rate - q) / b if b > 0 else -1.0
                k_half   = max(0.0, k_full * 0.5)
                kelly_raw_fracs[t] = k_half
                kelly_rows.append({
                    "Ticker":        t,
                    "Win Rate":      f"{win_rate*100:.1f}%",
                    "Avg Win/Day":   f"{avg_win*100:.3f}%",
                    "Avg Loss/Day":  f"{avg_loss*100:.3f}%",
                    "b (Win/Loss)":  f"{b:.3f}",
                    "Full Kelly":    f"{k_full*100:.2f}%",
                    "Half Kelly":    f"{k_half*100:.2f}%",
                })

            frac_total = sum(kelly_raw_fracs.values())
            kelly_suggested_pct = {
                t: (kelly_raw_fracs[t] / frac_total * 100) if frac_total > 0 else (100 / len(valid_k))
                for t in valid_k
            }
            kelly_suggested_dollars = {t: kelly_suggested_pct[t] / 100 * total_k for t in valid_k}

            for row in kelly_rows:
                t = row["Ticker"]
                row["Suggested %"]  = f"{kelly_suggested_pct.get(t,0):.1f}%"
                row["Suggested $"]  = f"${kelly_suggested_dollars.get(t,0):,.0f}"

            if kelly_rows:
                st.dataframe(pd.DataFrame(kelly_rows), hide_index=True, use_container_width=True)

                # Bar chart: current vs Kelly
                _cur_dollars = {h["ticker"]: h["dollars"] for h in st.session_state.etf_portfolio}
                bar_fig = go.Figure()
                bar_fig.add_trace(go.Bar(
                    name="Current ($)", x=valid_k,
                    y=[_cur_dollars.get(t, 0) for t in valid_k],
                    marker_color="#7b68ee", opacity=0.8,
                    text=[f"${_cur_dollars.get(t,0):,.0f}" for t in valid_k],
                    textposition="outside"))
                bar_fig.add_trace(go.Bar(
                    name="Kelly Suggested ($)", x=valid_k,
                    y=[kelly_suggested_dollars.get(t, 0) for t in valid_k],
                    marker_color="#bfa15d", opacity=0.9,
                    text=[f"${kelly_suggested_dollars.get(t,0):,.0f}" for t in valid_k],
                    textposition="outside"))
                bar_fig.update_layout(
                    barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(title="Allocation ($)", gridcolor="rgba(255,255,255,0.08)"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
                    margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False})

                k_col1, k_col2, k_col3 = st.columns([2, 2, 3])
                with k_col1:
                    if st.button("Apply Kelly Allocation", type="primary", key="etf_apply_kelly"):
                        # Save snapshot for revert
                        st.session_state["etf_portfolio_pre_kelly"] = [dict(h) for h in st.session_state.etf_portfolio]
                        # Apply Kelly amounts to portfolio list
                        for i2, h in enumerate(st.session_state.etf_portfolio):
                            t2 = h["ticker"]
                            if t2 in kelly_suggested_dollars:
                                st.session_state.etf_portfolio[i2]["dollars"] = round(max(0.01, kelly_suggested_dollars[t2]), 2)
                        # Increment generation → forces brand-new widget keys on rerun
                        # so value= is respected instead of stale widget state
                        st.session_state["etf_key_gen"] = st.session_state.get("etf_key_gen", 0) + 1
                        st.session_state.pop("etf_results", None)
                        st.toast("Kelly allocation applied. Click 'Revert' to undo.")
                        st.rerun()
                with k_col2:
                    if "etf_portfolio_pre_kelly" in st.session_state:
                        if st.button("Revert to Previous", type="secondary", key="etf_revert_kelly"):
                            _snapshot = st.session_state.pop("etf_portfolio_pre_kelly")
                            st.session_state.etf_portfolio = _snapshot
                            # Increment generation → fresh widgets pick up reverted values
                            st.session_state["etf_key_gen"] = st.session_state.get("etf_key_gen", 0) + 1
                            st.session_state.pop("etf_results", None)
                            st.toast("Reverted to your previous allocation.")
                            st.rerun()
                with k_col3:
                    _has_snap = "etf_portfolio_pre_kelly" in st.session_state
                    st.markdown(
                        "<div style='font-size:0.8rem;opacity:0.6;padding-top:6px;line-height:1.5;'>"
                        f"Redistributes your ${total_k:,.0f} total using half-Kelly weights. "
                        + ("Revert button active — click to undo." if _has_snap else "Always review before deploying capital.")
                        + "</div>", unsafe_allow_html=True)
            else:
                st.caption("Insufficient data for Kelly. Need at least 50 trading days per security.")

        # ==========================================
        # ANALYSIS RESULTS
        # ==========================================
        if "etf_results" in st.session_state:
            res         = st.session_state["etf_results"]
            portfolio   = res["portfolio"]
            prices      = res["prices"]
            returns     = res["returns"]
            port_return = res["port_return"]
            valid_cols  = res["valid_cols"]
            w           = res["weights"]
            dollars_arr = res["dollars_arr"]
            rf_daily    = res["rf_daily"]
            spy_ret     = res["spy_ret"]
            spy_prices  = res["spy_prices"]
            total_inv2  = res["total_invested"]
            horizon     = res.get("horizon", 10)

            # ── COMPUTE ALL METRICS ──
            excess       = port_return - rf_daily
            vol_port     = port_return.std(ddof=1)
            sharpe       = (excess.mean() / vol_port * np.sqrt(252)) if vol_port > 0 else float("nan")

            neg_returns  = port_return[port_return < 0]
            downside_std = neg_returns.std(ddof=1) if len(neg_returns) > 1 else float("nan")
            sortino      = (excess.mean() / downside_std * np.sqrt(252)) if not (np.isnan(downside_std) or downside_std == 0) else float("nan")

            cumval       = (1 + port_return).cumprod()
            rolling_max  = cumval.expanding().max()
            drawdown_ser = (cumval - rolling_max) / rolling_max
            max_dd       = float(drawdown_ser.min()) * 100

            var_95       = float(np.percentile(port_return, 5)) * total_inv2

            beta         = float("nan"); alpha_ann = float("nan"); alpha_daily = 0.0
            if len(spy_ret) > 10:
                aligned_df = pd.concat([port_return, spy_ret], axis=1).dropna()
                aligned_df.columns = ["Portfolio", "SPY"]
                if len(aligned_df) > 10:
                    Xb  = aligned_df["SPY"].values.reshape(-1, 1)
                    yb  = aligned_df["Portfolio"].values
                    reg = LinearRegression().fit(Xb, yb)
                    beta        = float(reg.coef_[0])
                    alpha_daily = float(reg.intercept_)
                    alpha_ann   = alpha_daily * 252 * 100

            corr     = returns.corr() if len(valid_cols) > 1 else None
            cagr_map = {t: etf_get_cagr(prices, t, 10) for t in valid_cols}

            fv_map = {}
            for i2, t in enumerate(valid_cols):
                cagr_t = cagr_map[t]
                inv_t  = float(dollars_arr[i2])
                fv_map[t] = npf.fv(rate=cagr_t, nper=horizon, pmt=0, pv=-inv_t) if not np.isnan(cagr_t) else float("nan")
            total_fv = sum(v for v in fv_map.values() if not np.isnan(v))

            # ── RISK METRICS CARDS ──
            st.divider()
            _section_header("Institutional Risk Metrics")
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Sharpe Ratio",  f"{sharpe:.3f}" if not np.isnan(sharpe) else "N/A",
                      help="Annualized risk-adjusted return vs risk-free rate. >1.0 is institutional-grade.")
            m2.metric("Sortino Ratio", f"{sortino:.3f}" if not np.isnan(sortino) else "N/A",
                      help="Like Sharpe but penalizes only downside volatility.")
            m3.metric("Beta vs SPY",   f"{beta:.3f}" if not np.isnan(beta) else "N/A",
                      help="Portfolio sensitivity to S&P 500 daily movements.")
            m4.metric("Alpha (ann.)",  f"{alpha_ann:.2f}%" if not np.isnan(alpha_ann) else "N/A",
                      help="Annualized excess return vs SPY after adjusting for beta.")
            m5.metric("Max Drawdown",  f"{max_dd:.2f}%",
                      help="Worst historical peak-to-trough decline in portfolio value.")
            m6.metric("VaR 95% (1d)", f"${abs(var_95):,.0f}",
                      help="Historical Value at Risk: worst expected daily loss 95% of the time.")

            st.divider()

            # ── PIE + GROWTH ──
            chart_c1, chart_c2 = st.columns([1, 2])
            with chart_c1:
                _section_header("Portfolio Weights")
                _colors = px.colors.qualitative.Bold[:len(valid_cols)]
                fig_pie = go.Figure(go.Pie(
                    labels=valid_cols,
                    values=[float(dollars_arr[i2]) for i2 in range(len(valid_cols))],
                    hole=0.4, marker=dict(colors=_colors),
                    textinfo="label+percent",
                    hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<br>%{percent}<extra></extra>"))
                fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0), showlegend=False)
                st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

            with chart_c2:
                _section_header("Historical Growth vs SPY Benchmark")
                port_cumulative = (1 + port_return).cumprod() * total_inv2
                spy_aligned     = spy_ret.reindex(port_return.index).fillna(0.0)
                spy_cumulative  = (1 + spy_aligned).cumprod() * total_inv2
                fig_growth = go.Figure()
                fig_growth.add_trace(go.Scatter(x=port_cumulative.index, y=port_cumulative,
                    mode="lines", name="ETF Portfolio", line=dict(color="#bfa15d", width=2),
                    hovertemplate="<b>Portfolio</b><br>%{x|%b %d, %Y}<br>$%{y:,.2f}<extra></extra>"))
                fig_growth.add_trace(go.Scatter(x=spy_cumulative.index, y=spy_cumulative,
                    mode="lines", name="SPY Benchmark", line=dict(color="#7b68ee", width=1.5, dash="dash"),
                    hovertemplate="<b>SPY</b><br>%{x|%b %d, %Y}<br>$%{y:,.2f}<extra></extra>"))
                fig_growth.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0,r=0,t=10,b=0), hovermode="x unified",
                    yaxis=dict(title="Portfolio Value ($)", gridcolor="rgba(255,255,255,0.08)"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig_growth, use_container_width=True, config={"displayModeBar": False})

            st.divider()

            # ── 2-YEAR PROJECTION ──
            _section_header("2-Year Forward Projection")
            end_date_proj = port_return.index[-1]
            hist_2y       = port_cumulative[port_cumulative.index >= end_date_proj - pd.DateOffset(years=2)].copy()
            if len(hist_2y) >= 20:
                X_hist      = np.arange(len(hist_2y)).reshape(-1, 1)
                y_hist      = hist_2y.values
                reg_pred    = LinearRegression().fit(X_hist, y_hist)
                future_days = 2 * 252
                X_future    = np.arange(len(hist_2y), len(hist_2y) + future_days).reshape(-1, 1)
                future_pred = reg_pred.predict(X_future).flatten()
                std_resid   = float(np.std(y_hist - reg_pred.predict(X_hist).flatten()))
                future_dates = pd.date_range(start=hist_2y.index[-1], periods=future_days + 1, freq="B")[1:]
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=hist_2y.index, y=hist_2y, mode="lines", name="Historical (2Y)",
                    line=dict(color="#00ffcc", width=2),
                    hovertemplate="<b>Historical</b><br>%{x|%b %d, %Y}<br>$%{y:,.2f}<extra></extra>"))
                fig_pred.add_trace(go.Scatter(x=future_dates, y=future_pred, mode="lines", name="Projected Trend",
                    line=dict(color="#bfa15d", width=1.5, dash="dash"),
                    hovertemplate="<b>Trend</b><br>%{x|%b %d, %Y}<br>$%{y:,.2f}<extra></extra>"))
                fig_pred.add_trace(go.Scatter(
                    x=list(future_dates) + list(future_dates[::-1]),
                    y=list(future_pred + std_resid) + list((future_pred - std_resid)[::-1]),
                    fill="toself", fillcolor="rgba(191,161,93,0.1)", line=dict(color="rgba(0,0,0,0)"),
                    name="+/-1 Std Dev", hoverinfo="skip"))
                fig_pred.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0,r=0,t=10,b=0), hovermode="x unified",
                    yaxis=dict(title="Portfolio Value ($)", gridcolor="rgba(255,255,255,0.08)"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig_pred, use_container_width=True, config={"displayModeBar": False})
            else:
                st.caption("Insufficient history for 2-year projection.")

            st.divider()

            # ── BETA SCATTER + CORRELATION ──
            beta_c, corr_c = st.columns([1, 1])
            with beta_c:
                _section_header("Beta Analysis vs SPY")
                if not np.isnan(beta) and len(spy_ret) > 10:
                    aligned_df2 = pd.concat([port_return, spy_ret], axis=1).dropna()
                    aligned_df2.columns = ["Portfolio", "SPY"]
                    X_line = np.linspace(aligned_df2["SPY"].min(), aligned_df2["SPY"].max(), 100)
                    Y_line = beta * X_line + alpha_daily
                    fig_beta = go.Figure()
                    fig_beta.add_trace(go.Scatter(x=aligned_df2["SPY"], y=aligned_df2["Portfolio"],
                        mode="markers", marker=dict(color="#bfa15d", size=3, opacity=0.4), name="Daily Returns"))
                    fig_beta.add_trace(go.Scatter(x=X_line, y=Y_line, mode="lines",
                        name=f"Fit (beta={beta:.2f})", line=dict(color="#ff4b4b", width=2)))
                    fig_beta.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=0,r=0,t=10,b=0),
                        xaxis=dict(title="SPY Daily Return", gridcolor="rgba(255,255,255,0.08)", tickformat=".2%"),
                        yaxis=dict(title="Portfolio Daily Return", gridcolor="rgba(255,255,255,0.08)", tickformat=".2%"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"))
                    st.plotly_chart(fig_beta, use_container_width=True, config={"displayModeBar": False})
                else:
                    st.caption("Beta chart requires SPY data alignment.")

            with corr_c:
                _section_header("Correlation Matrix")
                if corr is not None:
                    fig_corr = go.Figure(go.Heatmap(
                        z=corr.values, x=list(corr.columns), y=list(corr.index),
                        colorscale="RdYlGn", zmin=-1, zmax=1,
                        text=corr.round(2).values, texttemplate="%{text}",
                        hovertemplate="<b>%{x} / %{y}</b><br>r = %{z:.3f}<extra></extra>",
                        colorbar=dict(title="r", len=0.8)))
                    fig_corr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=0,r=0,t=10,b=0))
                    st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar": False})
                else:
                    st.caption("Correlation matrix requires at least 2 securities.")

            st.divider()

            # ── CAGR & VALUATION TABLE ──
            _section_header(f"Security CAGR & {horizon}-Year Projection")
            cagr_rows = []
            for i2, ticker in enumerate(valid_cols):
                cagr_t  = cagr_map[ticker]
                inv_t   = float(dollars_arr[i2])
                fv_t    = fv_map[ticker]
                gain_t  = fv_t - inv_t if not np.isnan(fv_t) else float("nan")
                mult_t  = fv_t / inv_t if (not np.isnan(fv_t) and inv_t > 0) else float("nan")
                cagr_rows.append({
                    "Ticker":                  ticker,
                    "Weight":                  f"{w[i2]*100:.1f}%",
                    "Invested":                f"${inv_t:,.2f}",
                    "10Y CAGR":                f"{cagr_t*100:.2f}%" if not np.isnan(cagr_t) else "N/A",
                    f"Projected ({horizon}Y)": f"${fv_t:,.2f}" if not np.isnan(fv_t) else "N/A",
                    f"Gain ({horizon}Y)":      f"${gain_t:,.2f}" if not np.isnan(gain_t) else "N/A",
                    "Multiple":                f"{mult_t:.2f}x" if not np.isnan(mult_t) else "N/A",
                })
            st.dataframe(pd.DataFrame(cagr_rows), hide_index=True, use_container_width=True)

            tc1, tc2, tc3 = st.columns(3)
            tc1.metric("Total Invested",               f"${total_inv2:,.2f}")
            tc2.metric(f"Total Projected ({horizon}Y)", f"${total_fv:,.2f}")
            tc3.metric(f"Total Gain ({horizon}Y)",
                       f"${total_fv - total_inv2:,.2f}",
                       delta=f"{(total_fv/total_inv2 - 1)*100:.1f}%" if total_inv2 > 0 else None)

            st.divider()

            # ── INDIVIDUAL SECURITY DEEP-DIVE ──
            _section_header("Individual Security Deep-Dive")
            dd_ticker = st.selectbox("Select Security", valid_cols, key="etf_dd_ticker")
            if dd_ticker:
                dd_invested = float(portfolio.get(dd_ticker, 0))
                dd_cagr     = cagr_map.get(dd_ticker, float("nan"))
                dd_fv       = fv_map.get(dd_ticker, float("nan"))
                dd_spot     = float(prices[dd_ticker].dropna().iloc[-1]) if dd_ticker in prices.columns else None
                dd_gain     = dd_fv - dd_invested if not np.isnan(dd_fv) else float("nan")
                dd_mult     = dd_fv / dd_invested if (not np.isnan(dd_fv) and dd_invested > 0) else float("nan")
                dd_weight   = dd_invested / total_inv2 * 100 if total_inv2 > 0 else 0

                ddc1, ddc2, ddc3, ddc4, ddc5 = st.columns(5)
                ddc1.metric("Current Price",               f"${dd_spot:,.2f}" if dd_spot else "N/A")
                ddc2.metric("Invested",                    f"${dd_invested:,.2f} ({dd_weight:.1f}%)")
                ddc3.metric("10Y CAGR",                    f"{dd_cagr*100:.2f}%" if not np.isnan(dd_cagr) else "N/A")
                ddc4.metric(f"Projected FV ({horizon}Y)",  f"${dd_fv:,.2f}" if not np.isnan(dd_fv) else "N/A")
                ddc5.metric(f"Gain ({horizon}Y)",
                            f"${dd_gain:,.2f}" if not np.isnan(dd_gain) else "N/A",
                            delta=f"{dd_mult:.2f}x" if not np.isnan(dd_mult) else None)

                if st.button("Get Stock Advisor Signal", type="secondary", key="etf_dd_advisor"):
                    with st.spinner(f"Fetching live data for {dd_ticker}..."):
                        _, _, dd_spot_live, _, dd_m_t0, dd_ivr, _, _ = fetch_ticker_resource(dd_ticker)
                    if dd_spot_live:
                        dd_regime = fz.classify_shock(dd_ivr)
                        dd_score, dd_verdict, _ = stock_advisor_verdict(dd_m_t0, 0, dd_ivr, dd_regime)
                        vc_dd = VERDICT_COLOR[dd_verdict]
                        st.markdown(f"""<div style="border:1px solid {vc_dd};border-radius:8px;padding:14px 20px;margin-top:8px;">
                            <strong style="color:{vc_dd};font-size:1.1rem;">{dd_verdict}</strong>
                            <span style="opacity:0.6;font-size:0.85rem;margin-left:8px;">{dd_score:.1f}/3.0 | IVR {dd_ivr:.0f} | {dd_regime}</span>
                        </div>""", unsafe_allow_html=True)
                        generate_stock_reasoning(dd_ticker, dd_spot_live, dd_m_t0, dd_ivr, dd_regime,
                                                 dd_verdict, dd_score, 5.0, 21, placeholder=st.empty())
                    else:
                        st.warning(f"Could not fetch live data for {dd_ticker}.")

            st.divider()

            # ── AI PORTFOLIO ANALYSIS ──
            _section_header("AI Portfolio Analysis")
            if st.button("Generate Institutional Portfolio Analysis", type="secondary", key="etf_ai_btn"):
                if _get_anthropic_client():
                    generate_etf_analysis(portfolio, sharpe, sortino, beta, alpha_ann,
                                         max_dd, var_95, total_inv2, total_fv, horizon, placeholder=st.empty())
                else:
                    st.warning("Add `ANTHROPIC_API_KEY` to `.streamlit/secrets.toml` to enable AI analysis.")

            st.divider()

            # ── EXPORT ──
            _section_header("Export")
            export_metrics = pd.DataFrame({
                "Metric": ["Sharpe Ratio", "Sortino Ratio", "Beta vs SPY", "Alpha (Ann. %)",
                           "Max Drawdown (%)", "VaR 95% 1d ($)", "Total Invested ($)",
                           f"Projected Total ({horizon}Y) ($)", f"Projected Gain ({horizon}Y) ($)"],
                "Value": [
                    f"{sharpe:.4f}" if not np.isnan(sharpe) else "N/A",
                    f"{sortino:.4f}" if not np.isnan(sortino) else "N/A",
                    f"{beta:.4f}" if not np.isnan(beta) else "N/A",
                    f"{alpha_ann:.4f}" if not np.isnan(alpha_ann) else "N/A",
                    f"{max_dd:.4f}",
                    f"{abs(var_95):,.2f}",
                    f"{total_inv2:,.2f}",
                    f"{total_fv:,.2f}",
                    f"{total_fv - total_inv2:,.2f}",
                ]
            })
            exp_c1, exp_c2, exp_c3 = st.columns(3)
            exp_c1.download_button("Download Risk Metrics CSV",
                                   export_metrics.to_csv(index=False), "etf_risk_metrics.csv", "text/csv")
            if corr is not None:
                exp_c2.download_button("Download Correlation Matrix CSV",
                                       corr.to_csv(), "etf_correlation.csv", "text/csv")
            holdings_export = pd.DataFrame([{
                "Ticker": t, "Invested ($)": portfolio.get(t, 0),
                "Weight %": w[i2] * 100,
                "10Y CAGR %": cagr_map.get(t, float("nan")) * 100,
                f"Projected FV ({horizon}Y) ($)": fv_map.get(t, float("nan")),
            } for i2, t in enumerate(valid_cols)])
            exp_c3.download_button("Download Holdings CSV",
                                   holdings_export.to_csv(index=False), "etf_holdings.csv", "text/csv")


# ==========================================
# END
# ==========================================
# streamlit run Institutional_MyQuant.py
# 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
