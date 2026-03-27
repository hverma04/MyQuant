import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.express as px
import plotly.graph_objects as go
import os
#python -m streamlit run MyQuant_V1.py
#PAGE SETUP & CSS
st.set_page_config(page_title="Kern | MyQuant", layout="wide")

#CUSTOM CSS
st.markdown("""
    <style>
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: rgba(191, 161, 93, 0.08); /* Subtle gold tint */
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #bfa15d;
    }
    /* Keep Metric Labels Gold */
    div[data-testid="stMetricLabel"] > div {
        color: #bfa15d !important;
        font-weight: bold;
    }
    /* Keep Sidebar Labels consistent */
    div[data-testid="stSidebar"] label {
        font-family: 'serif';
        font-size: 1.1rem;
    }
    </style>
    """, unsafe_allow_html=True)

#APP HEADER
col_logo, col_title = st.columns([1, 1])

with col_logo:
    st.markdown("""
        <h1 style="
            font-family: 'Times New Roman', Times, serif; 
            color: #bfa15d; 
            font-size: 5rem; 
            font-weight: 300;     
            margin-top: 15px;      
            margin-left: 20px;
            margin-right: 100px;   
            letter-spacing: 8px;    
            text-transform: uppercase;
        ">KERN<span style="color: #bfa15d; margin-left: -5px;">.</span></h1>
    """, unsafe_allow_html=True)

with col_title:
    st.title("MyQuant | Advanced Options Analytics")
    st.write("Institutional-grade probability modeling built for the Retail Investor.")

st.divider()


#MATH FUNCTIONS & DATA CACHING
def calculate_black_scholes(S, K, T, r, sigma, option_type="Call"):
    """Calculates the theoretical Black-Scholes price."""
    if T <= 0 or sigma <= 0:
        return max(0, S - K) if option_type == "Call" else max(0, K - S)
        
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "Call":
        price = (S * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
    else:
        price = (K * np.exp(-r * T) * norm.cdf(-d2)) - (S * norm.cdf(-d1))
    return price

@st.cache_resource(ttl=3600)
def fetch_ticker_resource(symbol):
    t = yf.Ticker(symbol)
    hist = t.history(period="1d")
    if hist.empty:
        return None, None, None, 0.042
    
    try:
        rf_ticker = yf.Ticker("^IRX")
        rf_rate = rf_ticker.history(period="1d")["Close"].iloc[-1] / 100
    except:
        rf_rate = 0.042 
        
    return t, t.options, hist["Close"].iloc[0], rf_rate

#SIDEBAR & INPUTS
with st.sidebar:
    st.header("Trade Parameters")
    
    ticker_input = st.text_input("Ticker Symbol", value="SPY").upper().strip()
    
    if not ticker_input:
        st.info("Enter a ticker symbol (e.g., AAPL, NVDA, SPY) to begin analysis.")
        st.stop()
    
    ticker_obj, expirations, spot_price, risk_free_rate = fetch_ticker_resource(ticker_input)

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
    st.markdown("### Strategy Adjustment")
    target_price = st.number_input("Target Price ($)", value=float(spot_price))
    order_size = st.number_input("Contracts", value=1, min_value=1)
    stop_loss_pct = st.slider("Stop Loss (%)", 0, 100, 50) / 100

#PRE-CALCULATIONS
premium = option_row["ask"] if option_row["ask"] > 0 else option_row["lastPrice"]
iv = option_row["impliedVolatility"]
if iv == 0 or pd.isna(iv):
    iv = 0.001

days_to_exp = (pd.to_datetime(selected_exp) - pd.to_datetime("today")).days
time_to_exp = max(days_to_exp, 1) / 365
breakeven = strike_price + premium if trade_type == "Call" else strike_price - premium

# --- 1. DYNAMIC INPUTS & SHOCKS ---
col_sim1, col_sim2 = st.columns(2)
with col_sim1:
    vol_shock = st.slider("IV Shock (%)", -50, 100, 0) / 100
with col_sim2:
    # Guard against 0 days to expiration / 1 (dte) options crashing the slider
    if days_to_exp > 1:
        days_to_hold = st.slider("Holding Period (Days)", 1, days_to_exp, days_to_exp)
    else:
        # Renders a greyed-out slider for options expiring immediately
        st.slider("Holding Period (Days)", 0, 2, 1, disabled=True)
        days_to_hold = 1

adj_iv = iv * (1 + vol_shock)
adj_time = max(days_to_hold, 1) / 365
adj_periodic_iv = adj_iv * np.sqrt(adj_time)

#IV Math logic
adj_iv = iv * (1 + vol_shock)
adj_time = max(days_to_hold, 1) / 365
adj_periodic_iv = adj_iv * np.sqrt(adj_time)

#Safeguard against 0 volatility
if adj_periodic_iv <= 0:
    adj_periodic_iv = 0.0001

bs_fair_value = calculate_black_scholes(
    spot_price, strike_price, time_to_exp, risk_free_rate, iv, trade_type
)

t_z = np.log(target_price / spot_price) / adj_periodic_iv
s_z = np.log(strike_price / spot_price) / adj_periodic_iv
b_z = np.log(breakeven / spot_price) / adj_periodic_iv

if trade_type == "Call":
    t_prob, s_prob, b_prob = 1 - norm.cdf(t_z), 1 - norm.cdf(s_z), 1 - norm.cdf(b_z)
    intrinsic = max(0, target_price - strike_price)
else:
    t_prob, s_prob, b_prob = norm.cdf(t_z), norm.cdf(s_z), norm.cdf(b_z)
    intrinsic = max(0, strike_price - target_price)

pnl_per_contract = (intrinsic - premium) * 100
total_pnl = pnl_per_contract * order_size
max_risk = premium * order_size * 100
ev = (b_prob * total_pnl) - (((1 - b_prob) * max_risk) * stop_loss_pct)

#DASHBOARD METRICS
valuation_label = "Overvalued" if premium > bs_fair_value else "Undervalued"
pct_diff = ((premium - bs_fair_value) / bs_fair_value * 100) if bs_fair_value > 0 else 0

m1, m4, m5, m6, m2, m3 = st.columns(6)
m1.metric("Spot Price", f"${spot_price:.2f}")
m2.metric("Market Premium", f"${premium:.2f}")
m3.metric("Black-Scholes Fair Value", f"${bs_fair_value:.2f}", delta=f"{pct_diff:.1f}% {valuation_label}", delta_color="inverse")
m4.metric("Breakeven Price", f"${breakeven:.2f}")
m5.metric("Implied Vol (IV)", f"{iv*100:.1f}%")
m6.metric("Expected Value (EV)", f"${ev:.2f}")

st.divider()

#INTERACTIVE LAYOUT
col_left, col_right = st.columns([2, 3])

with col_left:
    st.subheader("Probability Summary")
    prob_data = {
        "Level": ["Target Price", "Strike Price", "Breakeven"],
        "Price": [f"${target_price:.2f}", f"${strike_price:.2f}", f"${breakeven:.2f}"],
        "Probability": [f"{t_prob:.2%}", f"{s_prob:.2%}", f"{b_prob:.2%}"]
    }
    st.table(pd.DataFrame(prob_data))
    
    st.subheader("Trade Details")
    st.write(f"**Contracts:** {order_size}")
    st.write(f"**Total Risk:** ${max_risk:.2f}")
    st.write(f"**Risk-Free Rate:** {risk_free_rate:.2%}")

with col_right:
    st.subheader("Interactive Price Projection")
    # Run Simulation based on slider shocks
    sim_prices = np.random.lognormal(np.log(spot_price), adj_periodic_iv, 10000)
    
    # Calculate the 90% Confidence Interval (5th to 95th percentile)
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
        opacity=0.8
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
        template="plotly_dark", 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        hovermode="x unified",
        showlegend=True,
        bargap=0.25,            
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
    
    prob_hit_target = (sim_prices >= target_price).mean() if trade_type == "Call" else (sim_prices <= target_price).mean()
    
    st.markdown(f"""
        <div style="background-color: rgba(191, 161, 93, 0.08); padding: 15px; border-radius: 10px; border: 1px solid #bfa15d; margin-top: -10px;">
            <p style="color: var(--text-color); margin: 0; font-size: 1.05rem; font-family: 'serif';">
                <strong>Simulation Insight:</strong> Factoring in a holding period of <strong>{days_to_hold} days</strong> and your volatility adjustments, 
                there is a <strong style="color: #bfa15d;">{prob_hit_target:.1%}</strong> mathematical probability of the underlying asset reaching your target price of <strong>${target_price:.2f}</strong>. 
                The 90% confidence interval projects the price will land between <strong>${p5:.2f}</strong> and <strong>${p95:.2f}</strong>.
            </p>
            <p style="color: var(--text-color); margin: 10px 0 0 0; font-size: 0.85rem; opacity: 0.7; font-style: italic; text-align: right;">
                *Double click graph to reset view
            </p>
            <p style="color: var(--text-color); margin: 10px 0 0 0; font-size: 0.85rem; opacity: 0.7; font-style: italic; text-align: right;">
                *Highlight graph sections to zoom in for more detail
            </p>
        </div>
    """, unsafe_allow_html=True)