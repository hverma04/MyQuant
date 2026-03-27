import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import os
#HOW TO RUN = python -m streamlit run MyQuant_V1.py 

def calculate_black_scholes(S, K, T, r, sigma, option_type="Call"):
    """Calculates the theoretical Black-Scholes price."""
    # Avoid division by zero
    if T <= 0 or sigma <= 0:
        return max(0, S - K) if option_type == "Call" else max(0, K - S)
        
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "Call":
        price = (S * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
    else:
        price = (K * np.exp(-r * T) * norm.cdf(-d2)) - (S * norm.cdf(-d1))
    return price

st.set_page_config(page_title="Kern | MyQuant", layout="wide")

#CUSTOM CSS FOR STYLING THE DASHBOARD
st.markdown("""
    <style>
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: rgba(191, 161, 93, 0.08); /* Subtle gold tint */
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #bfa15d;
    }
    div[data-testid="stMetricLabel"] {
        color: #bfa15d !important;
        font-weight: bold;
    }
    div[data-testid="stSidebar"] label {
        color: #ffffff !important;
        font-family: 'serif';
        font-size: 1.1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# APP HEADER 
col_logo, col_title = st.columns([1, 1])
with col_logo:
    # Stylized text logo that defines the Kern brand
    st.markdown("""
        <h1 style="
            font-family: 'Times New Roman', Times, serif; 
            color: #bfa15d; 
            font-size: 3rem; 
            font-weight: 300;     
            margin-top: 15px;      
            margin-left: 130px;   
            letter-spacing: 8px;    
            text-transform: uppercase;
        ">KERN<span style="color: #bfa15d; margin-left: -5px;">.</span></h1>
    """, unsafe_allow_html=True)

with col_title:
    st.title("MyQuant | Advanced Options Analytics")
    st.write("Institutional-grade probability modeling built for the Retail Investor.")

st.divider()

#  DATA CACHING 
@st.cache_resource(ttl=3600)
def fetch_ticker_resource(symbol):
    t = yf.Ticker(symbol)
    hist = t.history(period="1d")
    if hist.empty:
        return None, None, None
    return t, t.options, hist["Close"].iloc[0]

#  SIDEBAR / INPUTS 
with st.sidebar:
    st.header("MyQuant | KERN.\nTrade Parameters")
    ticker_input = st.text_input("Ticker Symbol", value="SPY").upper()
    
    ticker, expirations, spot_price = fetch_ticker_resource(ticker_input)

    if ticker is not None and expirations:
        selected_exp = st.selectbox("Expiration Date", expirations)
        trade_type = st.radio("Option Type", ["Call", "Put"])
        
        opts = ticker.option_chain(selected_exp)
        chain = opts.calls if trade_type == "Call" else opts.puts
        
        strike_price = st.selectbox("Strike Price", chain["strike"].tolist())
        option_row = chain[chain["strike"] == strike_price].iloc[0]
        
        st.divider()
        target_price = st.number_input("Target Price ($)", value=float(spot_price))
        order_size = st.number_input("Contracts", value=1, min_value=1)
        stop_loss_pct = st.slider("Stop Loss (%)", 0, 100, 50) / 100 
    else:
        st.error("Invalid Ticker or No Options Data.")
        st.stop()

# MATH LOGIC
# --- UPDATED MATH LOGIC ---
premium = option_row["ask"]
iv = option_row["impliedVolatility"]
risk_free_rate = 0.04  # 4% proxy for risk-free rate

days_to_exp = (pd.to_datetime(selected_exp) - pd.to_datetime("today")).days
time_to_exp = max(days_to_exp, 1) / 365

# 1. Calculate Black-Scholes Fair Value
bs_fair_value = calculate_black_scholes(
    spot_price, strike_price, time_to_exp, risk_free_rate, iv, trade_type
)

# 2. Existing Probability Logic
breakeven = strike_price + premium if trade_type == "Call" else strike_price - premium
periodic_iv = iv * np.sqrt(time_to_exp)
o = periodic_iv

t_z = np.log(target_price / spot_price) / o
s_z = np.log(strike_price / spot_price) / o
b_z = np.log(breakeven / spot_price) / o

if trade_type == "Call":
    t_prob, s_prob, b_prob = 1 - norm.cdf(t_z), 1 - norm.cdf(s_z), 1 - norm.cdf(b_z)
    intrinsic = max(0, target_price - strike_price)
else:
    t_prob, s_prob, b_prob = norm.cdf(t_z), norm.cdf(s_z), norm.cdf(b_z)
    intrinsic = max(0, strike_price - target_price)

# 3. PnL Calculations
pnl_per_contract = (intrinsic - premium) * 100
total_pnl = pnl_per_contract * order_size
max_risk = premium * order_size * 100
ev = (b_prob * total_pnl) - (((1 - b_prob) * max_risk) * stop_loss_pct)

# --- DASHBOARD LAYOUT ---
valuation_label = "Overvalued" if premium > bs_fair_value else "Undervalued"
pct_diff = ((premium - bs_fair_value) / bs_fair_value) * 100
m1, m4, m5, m6, m2, m3 = st.columns(6)
m1.metric("Spot Price", f"${spot_price:.2f}")
m2.metric("Market Premium", f"${premium:.2f}")
m3.metric("Black-Scholes Fair Value", f"${bs_fair_value:.2f}", delta=f"{pct_diff:.1f}% {valuation_label}", delta_color="inverse")
m4.metric("Breakeven", f"${breakeven:.2f}")
m5.metric("Implied Vol (IV)", f"{iv*100:.1f}%")
m6.metric("Exp. Value (EV) - Per Trade", f"${ev:.2f}")

col_left, col_right = st.columns([2, 3])

with col_left:
    st.subheader("Probability Summary")
    prob_data = {
        "Level": ["Target Price", "Strike Price", "Breakeven"],
        "Price": [f"${target_price:.2f}", f"${strike_price:.2f}", f"${breakeven:.2f}"],
        "Probability to Reach": [f"{t_prob:.2%}", f"{s_prob:.2%}", f"{b_prob:.2%}"]
    }
    st.table(pd.DataFrame(prob_data))
    
    st.subheader("Trade Details")
    details = {
        "Contracts": order_size,
        "Premium Paid": f"${premium:.2f}/contract",
        "Total Risk": f"${max_risk:.2f}",
        "Days to Expiry": days_to_exp
    }
    for k, v in details.items():
        st.write(f"**{k}:** {v}")

with col_right:
    st.subheader("Monte Carlo Price Projection")
    sim_prices = np.random.lognormal(np.log(spot_price), o, 10000)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_alpha(0.0) 
    ax.patch.set_alpha(0.0)
    
    ax.hist(sim_prices, bins=70, color='#1a1c2c', alpha=0.6, edgecolor="#ffffff")
    ax.axvline(spot_price, color='#bfa15d', linestyle='--', linewidth=1, label='Current Price')
    ax.axvline(breakeven, color='red', linestyle='--', linewidth=1, label='Breakeven')
    
    # --- NEW: INCREASE X-AXIS INCREMENTS ---
    # This creates 15 evenly spaced labels between the min and max simulated price
    import numpy as np
    xticks = np.linspace(sim_prices.min(), sim_prices.max(), 15)
    ax.set_xticks(xticks)
    # Format the labels to 0 decimal places for a cleaner look
    ax.set_xticklabels([f"${x:.0f}" for x in xticks])
    
    ax.set_title(f"{ticker_input} Simulated Distribution at Expiry", fontsize=14, color="#595a67")
    ax.set_xlabel("Price ($)", fontsize=10, color="#595a67")
    ax.set_ylabel("Frequency", fontsize=10, color="#595a67")
    ax.tick_params(colors="#595a67", labelsize=8) # Smaller labelsize so they don't overlap
    ax.legend()
    
    st.pyplot(fig)