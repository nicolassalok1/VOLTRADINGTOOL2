import streamlit as st
import numpy as np
from scipy.stats import norm
import pandas as pd

st.set_page_config(page_title="Volatility Crush Trade Analyzer", layout="wide")

st.title("📊 Volatility Crush Trade Analyzer")

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def calculate_delta(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    else:
        return -norm.cdf(-d1)

def calculate_gamma(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def calculate_vega(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T) / 100

def calculate_theta(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r*T) * norm.cdf(d2)) / 365
    else:
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 + r * K * np.exp(-r*T) * norm.cdf(-d2)) / 365
    return theta

col1, col2 = st.columns(2)

with col1:
    st.header("📈 Market Data & Parameters")
    
    ticker = st.text_input("Ticker Symbol", value="NVDA")
    spot_price = st.number_input("Spot Price ($)", min_value=0.01, value=100.00, step=0.01)
    strike_price = st.number_input("Strike Price ($)", min_value=0.01, value=100.00, step=0.01)
    iv_percent = st.number_input("Implied Volatility (%)", min_value=0.1, value=50.0, step=0.1)
    days_to_expiry = st.number_input("Days to Expiry", min_value=1, value=30, step=1)
    risk_free_rate = st.number_input("Risk-Free Rate (%)", min_value=0.0, value=5.0, step=0.1) / 100
    
    st.markdown("---")
    
    if st.button("💰 Price Straddle", type="primary"):
        T = days_to_expiry / 365.0
        iv = iv_percent / 100.0
        
        call_price = black_scholes_call(spot_price, strike_price, T, risk_free_rate, iv)
        put_price = black_scholes_put(spot_price, strike_price, T, risk_free_rate, iv)
        straddle_price = call_price + put_price
        
        delta_call = calculate_delta(spot_price, strike_price, T, risk_free_rate, iv, 'call')
        delta_put = calculate_delta(spot_price, strike_price, T, risk_free_rate, iv, 'put')
        delta_straddle = delta_call + delta_put
        
        gamma = calculate_gamma(spot_price, strike_price, T, risk_free_rate, iv)
        vega = calculate_vega(spot_price, strike_price, T, risk_free_rate, iv) * 2
        
        theta_call = calculate_theta(spot_price, strike_price, T, risk_free_rate, iv, 'call')
        theta_put = calculate_theta(spot_price, strike_price, T, risk_free_rate, iv, 'put')
        theta_straddle = theta_call + theta_put
        
        st.session_state['current_data'] = {
            'call_price': call_price,
            'put_price': put_price,
            'straddle_price': straddle_price,
            'delta': delta_straddle,
            'gamma': gamma,
            'vega': vega,
            'theta': theta_straddle,
            'spot_price': spot_price,
            'strike_price': strike_price,
            'T': T,
            'iv': iv,
            'risk_free_rate': risk_free_rate
        }

if 'current_data' in st.session_state:
    data = st.session_state['current_data']
    
    with col1:
        st.subheader("📊 Current Straddle Price")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Call Price", f"${data['call_price']:.2f}")
        with col_b:
            st.metric("Put Price", f"${data['put_price']:.2f}")
        
        st.metric("**Straddle Price**", f"${data['straddle_price']:.2f}")
        
        st.subheader("🔤 Current Greeks")
        col_c, col_d = st.columns(2)
        with col_c:
            st.metric("Delta", f"{data['delta']:.4f}")
            st.metric("Vega", f"{data['vega']:.2f}")
        with col_d:
            st.metric("Gamma", f"{data['gamma']:.4f}")
            st.metric("Theta", f"{data['theta']:.2f}")

with col2:
    st.header("🔮 Scenario Analysis")
    
    new_spot_price = st.number_input("New Spot Price ($)", min_value=0.01, value=data['spot_price'] if 'current_data' in st.session_state else 100.00, step=0.01, key="new_spot")
    new_iv_percent = st.number_input("New Implied Volatility (%)", min_value=0.1, value=(data['iv']*100) if 'current_data' in st.session_state else 50.0, step=0.1, key="new_iv")
    
    st.markdown("---")
    
    if st.button("🎯 Analyze Scenario", type="primary") and 'current_data' in st.session_state:
        data = st.session_state['current_data']
        new_iv = new_iv_percent / 100.0
        
        new_call_price = black_scholes_call(new_spot_price, data['strike_price'], data['T'], data['risk_free_rate'], new_iv)
        new_put_price = black_scholes_put(new_spot_price, data['strike_price'], data['T'], data['risk_free_rate'], new_iv)
        new_straddle_price = new_call_price + new_put_price
        
        pnl_long = new_straddle_price - data['straddle_price']
        pnl_short = -pnl_long
        
        new_delta_call = calculate_delta(new_spot_price, data['strike_price'], data['T'], data['risk_free_rate'], new_iv, 'call')
        new_delta_put = calculate_delta(new_spot_price, data['strike_price'], data['T'], data['risk_free_rate'], new_iv, 'put')
        new_delta = new_delta_call + new_delta_put
        
        new_gamma = calculate_gamma(new_spot_price, data['strike_price'], data['T'], data['risk_free_rate'], new_iv)
        new_vega = calculate_vega(new_spot_price, data['strike_price'], data['T'], data['risk_free_rate'], new_iv) * 2
        
        new_theta_call = calculate_theta(new_spot_price, data['strike_price'], data['T'], data['risk_free_rate'], new_iv, 'call')
        new_theta_put = calculate_theta(new_spot_price, data['strike_price'], data['T'], data['risk_free_rate'], new_iv, 'put')
        new_theta = new_theta_call + new_theta_put
        
        st.session_state['scenario_data'] = {
            'new_straddle_price': new_straddle_price,
            'pnl_long': pnl_long,
            'pnl_short': pnl_short,
            'new_delta': new_delta,
            'new_gamma': new_gamma,
            'new_vega': new_vega,
            'new_theta': new_theta
        }

if 'scenario_data' in st.session_state:
    scenario = st.session_state['scenario_data']
    
    with col2:
        st.subheader("💵 P/L Analysis")
        
        st.metric("New Straddle Price", f"${scenario['new_straddle_price']:.2f}")
        
        col_e, col_f = st.columns(2)
        with col_e:
            st.metric("Long Straddle P/L", 
                     f"${scenario['pnl_long']:+.2f}",
                     delta=f"{scenario['pnl_long']:+.2f}",
                     delta_color="normal")
        with col_f:
            st.metric("Short Straddle P/L", 
                     f"${scenario['pnl_short']:+.2f}",
                     delta=f"{scenario['pnl_short']:+.2f}",
                     delta_color="inverse")
        
        st.subheader("🔤 New Scenario Greeks")
        col_g, col_h = st.columns(2)
        with col_g:
            st.metric("Delta", f"{scenario['new_delta']:.4f}")
            st.metric("Vega", f"{scenario['new_vega']:.2f}")
        with col_h:
            st.metric("Gamma", f"{scenario['new_gamma']:.4f}")
            st.metric("Theta", f"{scenario['new_theta']:.2f}")

st.markdown("---")
st.info("ℹ️ This tool analyzes volatility crush trades using Black-Scholes pricing. Enter market parameters on the left and scenario parameters on the right to analyze potential P/L.")
