import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go

# --- 1. FEAR Z BEHAVIORAL ENGINE ---
# --- 1. FEAR Z BEHAVIORAL ENGINE ---
class FearZEngine:
    def __init__(self):
        self.params = {
            'Episodic':   {'p0': 22.3, 'lam': 0.083, 'mu': 0.65, 'min_ivr': 0},
            'Structural': {'p0': 28.7, 'lam': 0.046, 'mu': 1.14, 'min_ivr': 70},
            'Systemic':   {'p0': 34.8, 'lam': 0.021, 'mu': 2.03, 'min_ivr': 90}
        }

    def classify_shock(self, iv_rank):
        if iv_rank >= self.params['Systemic']['min_ivr']: return 'Systemic'
        if iv_rank >= self.params['Structural']['min_ivr']: return 'Structural'
        return 'Episodic'

    def automate_gamma(self, vol_history):
        """Calculates ticker-specific gamma based on IV mean reversion speed."""
        # Safety check: if vol_history is empty or too short, return baseline 0.12
        if vol_history is None or len(vol_history) < 10:
            return 0.12
        
        # Calculate daily change in IV (y) and distance from mean (x)
        mean_iv = vol_history.mean()
        y = np.diff(vol_history) 
        x = mean_iv - vol_history[:-1].values 
        
        # Linear Regression: ΔIV = Gamma * (Mean - Current_IV)
        covariance = np.cov(x, y)[0, 1]
        variance = np.var(x)
        
        ticker_gamma = covariance / variance if variance > 0 else 0.12
        return np.clip(ticker_gamma, 0.05, 0.25)

    def calculate_shelf(self, current_iv, iv_rank, vol_history):
        # 1. Get dynamic gamma
        gamma = self.automate_gamma(vol_history)
        
        # 2. Threshold logic
        threshold = 0.30 if iv_rank < 70 else 0.45
        z_days = gamma * max(0, (current_iv * 100) - (threshold * 100))
        
        # 3. CRITICAL: Return BOTH values as a tuple so the sidebar can 'unpack' them
        return round(z_days, 1), round(gamma, 3)

    # ... (keep get_projection exactly as it was)

    def get_projection(self, t_days, current_iv, m_t0, z, category):
        p = self.params[category]
        if t_days <= z: return current_iv 
        
        t_delta = t_days - z
        
        # CORRECTED MATH: Inertia and Momentum act as friction on the decay rate (lambda)
        # This stretches the "Emotional Tail" without causing artificial spikes.
        inertia_friction = 1 + (p['p0'] / 100) + (p['mu'] * abs(m_t0))
        adjusted_lam = p['lam'] / inertia_friction
        
        decay = current_iv * np.exp(-adjusted_lam * t_delta)
        return round(decay, 4)

# --- 2. PAGE SETUP & CSS (RESTORED EXACTLY) ---
st.set_page_config(page_title="MyQuant Analytics", layout="wide")

st.markdown("""
    <style>
    /* --- BRANDING HEADER LAYOUT --- */
    .branding-row {
        display: flex;
        align-items: center; 
        margin-bottom: 20px;
    }
    .logo-col {
        flex: 0 0 170px; /* INCREASED: From 130px to 170px to fit the larger logo */
        border-right: 1px solid rgba(191, 161, 93, 0.4); 
        margin-right: 25px;
        padding-top: 5px;
    }
    .logo-text-kern {
        font-family: 'Times New Roman', Times, serif;
        color: #bfa15d;
        letter-spacing: 0.6rem;
        font-size: 5rem; /* INCREASED: From 1.1rem to 1.5rem */
        text-transform: uppercase;
        margin: 0;
        line-height: 1;
    }
    }
    .title-col {
        flex: 1;
    }
    .main-title-text {
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.1;
        color: var(--text-color); /* Adapts to your theme */
    }
    .subtitle-text {
        font-size: 1.05rem;
        opacity: 0.8;
        margin-top: 4px;
        font-family: 'serif';
        color: var(--text-color);
    }

    /* --- METRIC CARDS (UNTOUCHED) --- */
    div[data-testid="stMetric"] {
        background-color: rgba(191, 161, 93, 0.08);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #bfa15d;
        min-height: 134px; 
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    div[data-testid="stMetric"] > div {
        width: 100%;
        text-align: left;
    }

    div[data-testid="stMetricLabel"] > div {
        color: #bfa15d !important;
        font-weight: bold;
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- BRANDED HEADER RENDERING ---
st.markdown("""
    <div class="branding-row">
        <div class="logo-col">
            <p class="logo-text-kern">KERN.</p>
        </div>
        <div class="title-col">
            <div class="main-title-text">MyQuant | Advanced Options Analytics</div>
            <p class="subtitle-text">Institutional-grade probability modeling built for the Retail Investor.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# --- 3. MATH & DATA FETCHING ---
def calculate_black_scholes(S, K, T, r, sigma, option_type="Call"):
    if T <= 0 or sigma <= 0:
        return max(0, S - K) if option_type == "Call" else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "Call":
        return (S * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
    return (K * np.exp(-r * T) * norm.cdf(-d2)) - (S * norm.cdf(-d1))

@st.cache_data(ttl=900) # Caches chart data for 15 mins to stay fast but live
def fetch_chart_data(symbol, time_selection):
    t = yf.Ticker(symbol)
    
    # yfinance requires different intervals for short timeframes to show actual candles
    if time_selection == "1 Day":
        return t.history(period="1d", interval="5m")
    elif time_selection == "5 Days":
        return t.history(period="5d", interval="30m")
    elif time_selection == "1 Month":
        return t.history(period="1mo", interval="1d")
    elif time_selection == "6 Months":
        return t.history(period="6mo", interval="1d")
    elif time_selection == "1 Year":
        return t.history(period="1y", interval="1d")
    elif time_selection == "5 Years":
        return t.history(period="5y", interval="1wk")
    
    return t.history(period="1y", interval="1d")

@st.cache_resource(ttl=3600)
def fetch_ticker_resource(symbol):
    t = yf.Ticker(symbol)
    hist = t.history(period="1y")
    if hist.empty: return None, None, None, 0.042, 0.0, 0.0
    
    m_t0 = (hist["Close"].iloc[-1] / hist["Close"].iloc[-6]) - 1 if len(hist) > 5 else 0
    vols = hist["Close"].pct_change().rolling(21).std() * np.sqrt(252)
    vols = vols.dropna()
    # DEBUGGED: Added ZeroDivisionError safety net
    if not vols.empty:
        current_vol = vols.iloc[-1]
        vol_range = vols.max() - vols.min()
        ivr = (current_vol - vols.min()) / vol_range * 100 if vol_range > 0 else 50
    else:
        ivr = 50
        
    try:
        rf_rate = yf.Ticker("^IRX").history(period="1d")["Close"].iloc[-1] / 100
    except:
        rf_rate = 0.042 
        
    return t, t.options, hist["Close"].iloc[-1], rf_rate, m_t0, ivr, vols, hist


# --- 4. SIDEBAR INPUTS ---
with st.sidebar:
    st.header("Trade Parameters")
    ticker_input = st.text_input("Ticker Symbol", value="SPY").upper().strip()
    if not ticker_input:
        st.info("Enter a ticker symbol (e.g., AAPL, NVDA, SPY) to begin analysis.")
        st.stop()
    
    # 1. NEW UNPACKING (Added vol_history)
    ticker_obj, expirations, spot_price, risk_free_rate, m_t0, auto_ivr, vol_history, hist_data = fetch_ticker_resource(ticker_input)
    
    if ticker_obj is None or not expirations:
        st.error(f"No data found for '{ticker_input}'. Please check the symbol.")
        st.stop()

    selected_exp = st.selectbox("Expiration Date", expirations)
    trade_type = st.radio("Option Type", ["Call", "Put"])
    
    opts = ticker_obj.option_chain(selected_exp)
    chain = opts.calls if trade_type == "Call" else opts.puts
    if chain.empty:
        st.warning("No options chain found for this expiration.")
        st.stop()

    strike_price = st.selectbox("Strike Price", chain["strike"].tolist())
    option_row = chain[chain["strike"] == strike_price].iloc[0]

    st.divider()
    st.markdown("### Behavioral Adjustment")
    
    st.markdown(f"Live Market IV Rank: <span style='color:#bfa15d; font-weight:bold; font-size:1.1rem;'>{int(auto_ivr)}</span>", unsafe_allow_html=True)
    
    ivr = st.slider("Stress Test Override", 0, 100, int(auto_ivr))
    
    # 2. NEW DYNAMIC ENGINE LOGIC
    fz = FearZEngine()
    regime = fz.classify_shock(ivr)
    iv = option_row["impliedVolatility"] if option_row["impliedVolatility"] > 0 else 0.001
    
    # Passing vol_history here allows the engine to calculate a ticker-specific gamma
    shelf, dynamic_gamma = fz.calculate_shelf(iv, ivr, vol_history) 
    
    # Updated UI to show the "Story" of the dynamic gamma
    st.info(f"Regime: **{regime}**\n\nTicker Gamma: **{dynamic_gamma}**\n\nShelf Duration: **{shelf} Days**")

    st.divider()
    # ... (rest of sidebar stays the same)
    st.markdown("### Strategy Adjustment")
    target_price = st.number_input("Target Price ($)", value=float(spot_price))
    order_size = st.number_input("Contracts", value=1, min_value=1)
    stop_loss_pct = st.slider("Stop Loss (%)", 0, 100, 20) / 100

# --- 5. PRE-CALCULATIONS & SHOCKS ---
premium = option_row["ask"] if option_row["ask"] > 0 else option_row["lastPrice"]
days_to_exp = (pd.to_datetime(selected_exp) - pd.to_datetime("today")).days
time_to_exp = max(days_to_exp, 1) / 365
breakeven = strike_price + premium if trade_type == "Call" else strike_price - premium

col_sim1, col_sim2 = st.columns(2)
with col_sim1:
    if days_to_exp > 1:
        days_to_hold = st.slider("Holding Period (Days)", 1, days_to_exp, days_to_exp)
    else:
        st.slider("Holding Period (Days)", 0, 2, 1, disabled=True)
        days_to_hold = 1
    projected_iv = fz.get_projection(days_to_hold, iv, m_t0, shelf, regime)

with col_sim2:
    vol_shock_suggested = (projected_iv / iv) - 1
    vol_shock = st.slider("Custom Shock (%) and Manual Adjustment", -50, 150, int(vol_shock_suggested * 100)) / 100

adj_iv = iv * (1 + vol_shock)
adj_time = max(days_to_hold, 1) / 365
adj_periodic_iv = adj_iv * np.sqrt(adj_time)
if adj_periodic_iv <= 0: adj_periodic_iv = 0.0001

bs_fair_value = calculate_black_scholes(spot_price, strike_price, time_to_exp, risk_free_rate, iv, trade_type)

# --- NEW MATH: Risk-Neutral Drift ---
# Calculates the expected directional drift of the stock based on the risk-free rate and volatility drag
drift = (risk_free_rate - 0.5 * adj_iv**2) * adj_time

# Apply the drift to the Z-Score calculations
t_z = (np.log(target_price / spot_price) - drift) / adj_periodic_iv
s_z = (np.log(strike_price / spot_price) - drift) / adj_periodic_iv
b_z = (np.log(breakeven / spot_price) - drift) / adj_periodic_iv

if trade_type == "Call":
    t_prob, s_prob, b_prob = 1 - norm.cdf(t_z), 1 - norm.cdf(s_z), 1 - norm.cdf(b_z)
    intrinsic = max(0, target_price - strike_price)
else:
    t_prob, s_prob, b_prob = norm.cdf(t_z), norm.cdf(s_z), norm.cdf(b_z)
    intrinsic = max(0, strike_price - target_price)

pnl_per_contract = (intrinsic - premium) * 100
total_pnl = pnl_per_contract * order_size
max_risk = premium * order_size * 100
risk_factor = 1.0 if stop_loss_pct == 0.0 else stop_loss_pct

# EV Calculation remains the same
ev = (t_prob * total_pnl) - (((1 - b_prob) * max_risk) * risk_factor)

# --- NEW MATH: Projected Exit Premium ---
# We calculate what the option will be worth at the END of your holding period
days_remaining_at_exit = max(0, days_to_exp - days_to_hold)
time_remaining_at_exit = days_remaining_at_exit / 365

projected_premium = calculate_black_scholes(
    S=target_price,               # The price you hope it hits
    K=strike_price,               # Your fixed strike
    T=time_remaining_at_exit,     # The time left when you sell
    r=risk_free_rate,             # Interest rate
    sigma=projected_iv,           # Your Fear Z projected IV
    option_type=trade_type
)

# Calculate the projected ROI %
projected_roi = ((projected_premium - premium) / premium) * 100 if premium > 0 else 0

# --- 6. DASHBOARD METRICS (STACKED 4x4) ---
valuation_label = "Overvalued" if premium > bs_fair_value else "Undervalued"
pct_diff = ((premium - bs_fair_value) / bs_fair_value * 100) if bs_fair_value > 0 else 0

# Row 1
r1 = st.columns(4)
r1[0].metric("Spot Price", f"${spot_price:.2f}")
r1[1].metric("Market Premium", f"${premium:.2f}")
r1[2].metric("Black-Scholes", f"${bs_fair_value:.2f}", delta=f"{pct_diff:.1f}% {valuation_label}", delta_color="inverse")
r1[3].metric("Exit Premium", f"${projected_premium:.2f}", delta=f"{projected_roi:.1f}% ROI")
# Row 2
r2 = st.columns(4)
r2[0].metric(f"IV: {ticker_input}", f"{iv*100:.1f}%")
r2[1].metric("Fear Z Shelf", f"{shelf}d", delta=regime, delta_color="off")
r2[2].metric("Expected Value", f"${ev:.2f}")
r2[3].metric("Breakeven", f"${breakeven:.2f}")

st.divider()

# --- 6.5 LIVE MARKET CHART ---
st.divider()

# 1. Setup the Layout for Title and Dropdown
chart_col1, chart_col2 = st.columns([4, 1])
with chart_col1:
    st.subheader(f"Live Market Action: {ticker_input}")
with chart_col2:
    timeframe = st.selectbox(
        "Chart Timeframe",
        options=["1 Day", "5 Days", "1 Month", "6 Months", "1 Year", "5 Years"],
        index=4 # Defaults to 1 Year
    )

# 2. Fetch the dynamic data specifically for the visual chart
chart_data = fetch_chart_data(ticker_input, timeframe)

# 3. Recalculate a dynamic SMA based on the data interval
chart_data['SMA_21'] = chart_data['Close'].rolling(window=21).mean()

# 4. Create Candlestick using the new dynamic chart_data
fig_candle = go.Figure(data=[go.Candlestick(
    x=chart_data.index,
    open=chart_data['Open'],
    high=chart_data['High'],
    low=chart_data['Low'],
    close=chart_data['Close'],
    
    # 1. COLOR LOGIC (Keep existing palette)
    increasing_line_color='#00ffcc', # Cyan for Bullish
    decreasing_line_color='#ff4b4b', # Red for Bearish
    
    # 2. CUSTOM HOVER (Styled for readability)
    hovertemplate="""
    <span style='color:#bfa15d'><b>Date:</b></span> %{x}<br>
    <span style='color:#bfa15d'><b>Open:</b></span> $%{open:.2f}<br>
    <span style='color:#bfa15d'><b>Close:</b></span> $%{close:.2f}<br>
    <span style='color:#bfa15d'><b>Range:</b></span> $%{low:.2f} - $%{high:.2f}
    <extra></extra>""",
    
    name="Price"
)])
# Add Target and Breakeven visual anchors
fig_candle.add_hline(y=target_price, line_dash="dash", line_color="#00ffcc", opacity=0.5)
fig_candle.add_hline(y=breakeven, line_dash="solid", line_color="#ff4b4b", opacity=0.5)
# Add the Moving Average
fig_candle.add_trace(go.Scatter(
    x=chart_data.index, y=chart_data['SMA_21'], 
    mode='lines', line=dict(color='#bfa15d', width=1.5), 
    name='21-Period SMA'
))

# --- NEW: Conditional Rangebreaks ---
# Intraday gets hour & weekend filtering. Daily/Weekly only gets weekend filtering.
if timeframe in ["1 Day", "5 Days"]:
    x_breaks = [
        dict(bounds=["sat", "mon"]), 
        dict(bounds=[16, 9.5], pattern="hour") 
    ]
else:
    x_breaks = [
        dict(bounds=["sat", "mon"])
    ]

# Add a horizontal line for the current Spot Price
fig_candle.add_hline(
    y=spot_price, 
    line_dash="dot", 
    line_color="#bfa15d", 
    annotation_text=f"  Current: ${spot_price:.2f}", 
    annotation_position="bottom right"
)

# --- ADDING LEGEND-BASED PRICE ANCHORS ---

# 1. Current Spot Price (Trace for Legend)
fig_candle.add_trace(go.Scatter(
    x=[chart_data.index[-1]], 
    y=[spot_price],
    mode="markers",
    marker=dict(color="#bfa15d", size=10, symbol="diamond"),
    name=f"Current: ${spot_price:.2f}",
    hoverinfo="skip"
))

# 2. Horizontal Line for the Current Price (Visual Anchor)
fig_candle.add_hline(
    y=spot_price, 
    line_dash="dot", 
    line_color="#bfa15d", 
    opacity=0.8
)

# 3. Target Price Trace (Matches your Cyan theme)
fig_candle.add_trace(go.Scatter(
    x=[chart_data.index[0], chart_data.index[-1]], 
    y=[target_price, target_price],
    mode="lines",
    line=dict(color="#00ffcc", width=2, dash="dash"),
    name=f"Target: ${target_price:.2f}"
))

# 4. Breakeven Price Trace (Matches your Red theme)
fig_candle.add_trace(go.Scatter(
    x=[chart_data.index[0], chart_data.index[-1]], 
    y=[breakeven, breakeven],
    mode="lines",
    line=dict(color="#ff4b4b", width=2, dash="solid"),
    name=f"Breakeven: ${breakeven:.2f}"
))
# Format to match your MyQuant dark theme
fig_candle.update_layout(
    paper_bgcolor='rgba(0,0,0,0)', 
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis_rangeslider_visible=False,
    margin=dict(l=0, r=0, t=10, b=0),
    hovermode="x unified",
    yaxis=dict(title="Price ($)", gridcolor="rgba(255,255,255,0.1)"),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.1)",
        rangebreaks=x_breaks # <--- APPLIED HERE
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        bgcolor="rgba(0,0,0,0)"
    )
)

st.plotly_chart(fig_candle, use_container_width=True)
st.divider()

# --- 7. INTERACTIVE LAYOUT (RESTORED EXACTLY) ---
col_left, col_right = st.columns([2, 3])

with col_left:
    st.subheader("Probability Summary")
    prob_data = {
        "Level": ["Target Price", "Strike Price", "Breakeven"],
        "Price": [f"${target_price:.2f}", f"${strike_price:.2f}", f"${breakeven:.2f}"],
        "Probability to Reach": [f"{t_prob:.2%}", f"{s_prob:.2%}", f"{b_prob:.2%}"]
    }
    st.table(pd.DataFrame(prob_data))
    
    st.subheader("Trade Analysis")
    actual_risk_per_cnt = premium * risk_factor
    potential_profit = max(0, target_price - breakeven) if trade_type == "Call" else max(0, breakeven - target_price)
    rr_ratio = (potential_profit / actual_risk_per_cnt) if actual_risk_per_cnt > 0 else 0
    stop_label = "None (Max Risk)" if stop_loss_pct == 0.0 else f"{stop_loss_pct*100:.0f}% of Premium"
    
    details = {
        "Contracts": order_size,
        "Total Cash Risk": f"${actual_risk_per_cnt * order_size * 100:.2f}",
        "Stop Loss Limit": stop_label,
        "Potential Profit": f"${potential_profit:.2f}/cnt",
        "Adjusted R/R Ratio": f"{rr_ratio:.2f}" 
    }
    
    for k, v in details.items():
        if k == "Adjusted R/R Ratio":
            st.markdown(f"**{k}:** <span style='color: #bfa15d; font-weight: bold;'>{v}</span>", unsafe_allow_html=True)
        else:
            st.write(f"**{k}:** {v}")

with col_right:
    st.subheader("Interactive Price Projection")
    # --- NEW MATH: Added 'drift' to the lognormal mean to synchronize with Black-Scholes probabilities ---
    sim_prices = np.random.lognormal(np.log(spot_price) + drift, adj_periodic_iv, 10000)
    p5, p95 = np.percentile(sim_prices, [5, 95])
    
    fig = go.Figure()
    
    fig.add_vrect(
        x0=p5, x1=p95,
        fillcolor="#bfa15d",
        opacity=0.15,
        layer="below",
        line_width=0,
    )
    
    fig.add_trace(go.Histogram(
        x=sim_prices, 
        nbinsx=150,             
        name="Frequency",
        marker_color='#bfa15d', 
        opacity=0.8,
        hovertemplate="<b>Price:</b> %{x:$.2f}<br><b>Frequency:</b> %{y}<extra></extra>"
    ))
    
    fig.add_vline(x=spot_price, line_dash="dash", line_color="#ffffff")
    fig.add_vline(x=breakeven, line_dash="solid", line_color="#ff4b4b")
    fig.add_vline(x=target_price, line_dash="dot", line_color="#00ffcc")
    
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='#ffffff', dash='dash'), name='Spot Price'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='#ff4b4b', dash='solid'), name='Breakeven'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='#00ffcc', dash='dot'), name='Target Price'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='rgba(191, 161, 93, 0.4)', symbol='square', size=15), name='90% Confidence Interval'))
    
    fig.update_layout(
        title=f"Price Distribution in {days_to_hold} Days",
        xaxis=dict(title="Price ($)"),
        yaxis=dict(title="Frequency"),
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        hovermode="x unified",
        showlegend=True,
        bargap=0.25,
        hoverlabel=dict(
            font_size=16,          
            font_family="serif",   
        ),            
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)"
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

    prob_hit_target = (sim_prices >= target_price).mean() if trade_type == "Call" else (sim_prices <= target_price).mean()
    
    st.markdown(f"""
        <div style="background-color: rgba(191, 161, 93, 0.08); padding: 15px; border-radius: 10px; border: 1px solid #bfa15d; margin-top: -10px;">
            <p style="color: var(--text-color); margin: 0; font-size: 1.05rem; font-family: 'serif';">
                <strong>Simulation Insight:</strong> Factoring in a <strong>{regime}</strong> shock regime, the Fear Z model projects IV will decay to <strong>{projected_iv*100:.1f}%</strong> over <strong>{days_to_hold} days</strong>. 
                There is a <strong style="color: #bfa15d;">{prob_hit_target:.1%}</strong> mathematical probability of reaching your target of <strong>${target_price:.2f}</strong>. 
            </p>
            <p style="color: var(--text-color); margin: 10px 0 0 0; font-size: 0.85rem; opacity: 0.7; font-style: italic; text-align: right;">
                *Double click graph to reset view | Highlight graph sections to zoom in for more detail
            </p>
        </div>
    """, unsafe_allow_html=True)


# --- 8. REGIME CLASSIFICATION LEGEND ---
st.divider()
st.subheader("MyQuant's Behavioral Pipeline")

st.markdown("""<div style="display: flex; justify-content: space-between; gap: 15px; text-align: left;">
<div style="flex: 1; padding: 15px; background: rgba(0,0,0,0.2); border-radius: 6px; border-top: 3px solid #00ffcc;">
<strong style="color: #00ffcc; font-size: 0.95rem;">1. The Input (IV Rank)</strong><br>
<p style="font-size: 0.85rem; opacity: 0.8; margin-top: 5px; margin-bottom: 0;">
The engine reads the stock's current Implied Volatility against its 1-year history to classify the true severity of the market panic.
</p>
</div>
<div style="flex: 1; padding: 15px; background: rgba(0,0,0,0.2); border-radius: 6px; border-top: 3px solid #ff4b4b;">
<strong style="color: #ff4b4b; font-size: 0.95rem;">2. Behavioral Math</strong><br>
<p style="font-size: 0.85rem; opacity: 0.8; margin-top: 5px; margin-bottom: 0;">
Applies Emotional Inertia and Momentum Drag to calculate if option premiums are frozen in a "Panic Plateau" and determines the amount of days it will take for volatility decay.
</p>
</div>
<div style="flex: 1; padding: 15px; background: rgba(0,0,0,0.2); border-radius: 6px; border-top: 3px solid #FFC107;">
<strong style="color: #FFC107; font-size: 0.95rem;">3. Smart Projection</strong><br>
<p style="font-size: 0.85rem; opacity: 0.8; margin-top: 5px; margin-bottom: 0;">
The model predicts the exact percentage that Volatility will crush over your specific holding period (The Suggested Shock) and applies it to the manual adjustments.
</p>
</div>
<div style="flex: 1; padding: 15px; background: rgba(0,0,0,0.2); border-radius: 6px; border-top: 3px solid #bfa15d;">
<strong style="color: #bfa15d; font-size: 0.95rem;">4. Adjusted EV and Price Distribution</strong><br>
<p style="font-size: 0.85rem; opacity: 0.8; margin-top: 5px; margin-bottom: 0;">
Black-Scholes runs using the fear-adjusted IV, outputting a highly accurate Expected Value and price probability distribution via Monte Carlo Simulation calculating 10,000 possible scenarios.
</p>
</div>
</div><br>""", unsafe_allow_html=True)
st.subheader("Fear Z Regime Guide")

# DEBUGGED: Removed trailing comma causing tuple unpacking error
guide_c1, guide_c2, guide_c3 = st.columns(3)

with guide_c1:
    st.markdown("""
<div style="background-color: rgba(191, 161, 93, 0.05); padding: 15px; border-radius: 8px; border-left: 4px solid #4CAF50; height: 100%;">
    <h4 style="margin-top:0; color: var(--text-color);">🟢 Episodic (IVR 0-69)</h4>
    <p style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 0;">
        <strong>Contained</strong><br>
        Standard market reaction to typical news. Fear is fleeting, the Panic Plateau is non-existent, and volatility decays rapidly back to its baseline.
    </p>
</div>
""", unsafe_allow_html=True)

with guide_c2:
    st.markdown("""
<div style="background-color: rgba(191, 161, 93, 0.05); padding: 15px; border-radius: 8px; border-left: 4px solid #FFC107; height: 100%;">
    <h4 style="margin-top:0; color: var(--text-color);">🟡 Structural (IVR 70-89)</h4>
    <p style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 0;">
        <strong>Regime Uncertainty</strong><br>
        Investors are fundamentally questioning the stock's valuation. Triggers a moderate "Panic Plateau" where option premiums freeze before beginning a slow decay.
    </p>
</div>
""", unsafe_allow_html=True)

with guide_c3:
    st.markdown("""
<div style="background-color: rgba(191, 161, 93, 0.05); padding: 15px; border-radius: 8px; border-left: 4px solid #ff4b4b; height: 100%;">
    <h4 style="margin-top:0; color: var(--text-color);">🔴 Systemic (IVR 90+)</h4>
    <p style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 0;">
        <strong>Crisis Detection</strong><br>
        Peak market panic and capitulation. Creates a massive "Shelf" where implied volatility stays artificially inflated for days due to deep behavioral uncertainty.
    </p>
</div>
""", unsafe_allow_html=True)

    # python -m streamlit run MyQuant_V1.py