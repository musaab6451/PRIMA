import streamlit as st
import pandas as pd
import yfinance as yf
from nselib import capital_market
import ta
import time
from datetime import datetime
import secrets
import plotly.express as px # Added for the Risk Dashboard

# --- CONFIG ---
st.set_page_config(page_title="PRIMA v9.1 | Risk Analytics", layout="wide")

# --- SIDEBAR: STRATEGY & RISK ---
with st.sidebar:
    st.title("🔱 PRIMA Control")
    active_strat = st.selectbox("Trading Engine", 
                                ["RSI Mean Reversion", "EMA Crossover (9/21)", "Bollinger Mean Reversion", "VWAP Scalping"])
    
    st.divider()
    st.subheader("💰 Capital Control")
    total_cap = st.number_input("Total Capital (₹)", value=50000.0)
    max_trades = st.slider("Max Open Positions", 1, 10, 5)
    risk_per_trade = total_cap / max_trades
    
    st.divider()
    if st.button("🚨 GLOBAL KILL SWITCH", type="primary", use_container_width=True):
        st.session_state.portfolio = pd.DataFrame(columns=["ID", "Symbol", "Sector", "Qty", "Entry", "LTP", "P&L", "Target", "Stop"])
        st.rerun()

# --- STATE INITIALIZATION ---
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=["ID", "Symbol", "Sector", "Qty", "Entry", "LTP", "P&L", "Target", "Stop"])
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["Time", "Symbol", "Entry", "Exit", "P&L"])
if 'balance' not in st.session_state:
    st.session_state.balance = total_cap
if 'last_scan' not in st.session_state:
    st.session_state.last_scan = 0

# --- DATA HELPERS ---
@st.cache_data(ttl=86400)
def get_master_data():
    df = capital_market.equity_list()
    # Flexible column detection for Sector/Industry
    sec_col = next((c for c in ['SERIES', 'INDUSTRY', 'GROUP'] if c in df.columns), 'SYMBOL')
    name_col = next((c for c in ['NAME_OF_COMPANY', 'NAME'] if c in df.columns), 'SYMBOL')
    
    df = df.rename(columns={sec_col: 'Sector', name_col: 'Company'})
    return df[['SYMBOL', 'Company', 'Sector']].head(100)

master_df = get_master_data()

# --- REAL-TIME CALCULATIONS ---
# P&L Formula: $$ P\&L = (LTP - Entry) \times Qty $$
current_pos_val = (st.session_state.portfolio['LTP'] * st.session_state.portfolio['Qty']).sum()
live_pnl = st.session_state.portfolio['P&L'].sum()

# --- UI HEADER ---
st.title("🔱 PRIMA RISK COMMAND CENTER")
h1, h2, h3, h4 = st.columns(4)
h1.metric("Available Cash", f"₹{round(st.session_state.balance, 2)}")
h2.metric("Total Position Value", f"₹{round(current_pos_val, 2)}")
h3.metric("Live Session P&L", f"₹{round(live_pnl, 2)}", delta=f"{round(live_pnl, 2)}")
h4.metric("Risk Utilization", f"{round((current_pos_val/total_cap)*100, 1)}%")

tab1, tab2, tab3, tab4 = st.tabs(["📺 Market Watch", "⚔️ Active Positions", "📊 Risk Analytics", "📜 History"])

# --- STRATEGY ENGINE ---
def check_signal(df, strategy):
    cp = df['Close'].iloc[-1]
    if strategy == "RSI Mean Reversion":
        return ta.momentum.RSIIndicator(df['Close']).rsi().iloc[-1] < 30
    elif strategy == "EMA Crossover (9/21)":
        return ta.trend.EMAIndicator(df['Close'], 9).ema_indicator().iloc[-1] > ta.trend.EMAIndicator(df['Close'], 21).ema_indicator().iloc[-1]
    return False

# --- MAIN LOOP ---
watch_space = tab1.empty()
pos_space = tab2.empty()
risk_space = tab4.empty() # Placeholder for Risk tab

while True:
    # 1. POSITION REFRESH (1-SEC)
    if not st.session_state.portfolio.empty:
        for idx, row in st.session_state.portfolio.iterrows():
            try:
                t = yf.Ticker(f"{row['Symbol']}.NS")
                ltp = t.fast_info['last_price']
                st.session_state.portfolio.at[idx, 'LTP'] = round(ltp, 2)
                st.session_state.portfolio.at[idx, 'P&L'] = round((ltp - row['Entry']) * row['Qty'], 2)
                
                # Risk Guard: SL/TP
                if ltp <= row['Stop'] or ltp >= row['Target']:
                    st.session_state.balance += (ltp * row['Qty'])
                    st.session_state.portfolio = st.session_state.portfolio.drop(idx)
                    st.rerun()
            except: continue

    with pos_space.container():
        st.dataframe(st.session_state.portfolio.drop(columns=['ID']), use_container_width=True, hide_index=True)

    # 2. RISK DASHBOARD (TAB 3)
    with tab3:
        if not st.session_state.portfolio.empty:
            st.subheader("Sector Exposure")
            # Image of a sector allocation pie chart for a stock portfolio
            
            sector_data = st.session_state.portfolio.groupby('Sector')['Qty'].sum().reset_index()
            fig = px.pie(sector_data, values='Qty', names='Sector', hole=0.4, title="Capital Distribution by Sector")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No active risk to analyze.")

    # 3. SCANNER (2-MIN)
    if time.time() - st.session_state.last_scan > 120:
        results = []
        for _, stock in master_df.iterrows():
            try:
                data = yf.download(f"{stock['SYMBOL']}.NS", period="2d", interval="15m", progress=False)
                if check_signal(data, active_strat):
                    # Smart Allocation
                    if len(st.session_state.portfolio) < max_trades and st.session_state.balance >= risk_per_trade:
                        cp = data['Close'].iloc[-1]
                        qty = int(risk_per_trade / cp)
                        uid = f"{stock['SYMBOL']}_{secrets.token_hex(2)}"
                        new_trade = {"ID": uid, "Symbol": stock['SYMBOL'], "Sector": stock['Sector'], 
                                     "Qty": qty, "Entry": cp, "LTP": cp, "P&L": 0.0, 
                                     "Target": cp*1.03, "Stop": cp*0.98}
                        st.session_state.portfolio = pd.concat([st.session_state.portfolio, pd.DataFrame([new_trade])], ignore_index=True)
                        st.session_state.balance -= (cp * qty)
                results.append({"Symbol": stock['SYMBOL'], "Sector": stock['Sector'], "Price": data['Close'].iloc[-1]})
            except: continue
        
        st.session_state.last_scan = time.time()
        with watch_space.container():
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

    time.sleep(1)
