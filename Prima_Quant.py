import streamlit as st
import pandas as pd
import yfinance as yf
from nselib import capital_market
import ta
import time
from datetime import datetime
import secrets
import plotly.express as px

# --- CONFIG ---
st.set_page_config(page_title="PRIMA v9.2 | Pro Terminal", layout="wide")

# --- SIDEBAR: STRATEGY & RISK ---
with st.sidebar:
    st.title("🔱 PRIMA Control")
    active_strat = st.selectbox("Trading Engine", 
                                ["RSI Mean Reversion", "EMA Crossover (9/21)", "Bollinger Mean Reversion", "VWAP Scalping"])
    
    st.divider()
    st.subheader("💰 Capital Control")
    total_cap = st.number_input("Total Capital (₹)", value=50000.0)
    max_trades = st.slider("Max Open Positions", 1, 10, 5)
    sector_limit = st.slider("Max Sector Exposure %", 10, 100, 40) # New Feature!
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

# --- FIXING THE KEYERROR: ROBUST DATA ENGINE ---
@st.cache_data(ttl=86400)
def get_master_data():
    try:
        raw_df = capital_market.equity_list()
        # 1. Standardize column names to uppercase to avoid case-sensitivity issues
        raw_df.columns = [c.upper().strip() for c in raw_df.columns]
        
        # 2. Identify critical columns
        sym_col = next((c for c in ['SYMBOL', 'SYM', 'TOKEN'] if c in raw_df.columns), raw_df.columns[0])
        sec_col = next((c for c in ['INDUSTRY', 'GROUP', 'SERIES'] if c in raw_df.columns), None)
        
        # 3. Create a clean mapping without destroying the original index
        processed_df = pd.DataFrame()
        processed_df['SYMBOL'] = raw_df[sym_col]
        processed_df['SECTOR'] = raw_df[sec_col] if sec_col else "General"
        
        return processed_df.head(100)
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return pd.DataFrame({'SYMBOL': ['RELIANCE', 'TCS'], 'SECTOR': ['Energy', 'IT']})

master_df = get_master_data()

# --- STRATEGY ENGINE ---
def check_signal(df, strategy):
    try:
        cp = df['Close'].iloc[-1]
        if strategy == "RSI Mean Reversion":
            return ta.momentum.RSIIndicator(df['Close']).rsi().iloc[-1] < 30
        elif strategy == "EMA Crossover (9/21)":
            ema9 = ta.trend.EMAIndicator(df['Close'], 9).ema_indicator().iloc[-1]
            ema21 = ta.trend.EMAIndicator(df['Close'], 21).ema_indicator().iloc[-1]
            return ema9 > ema21
        elif strategy == "Bollinger Mean Reversion":
            bb = ta.volatility.BollingerBands(df['Close'])
            return cp < bb.bollinger_lband().iloc[-1]
        elif strategy == "VWAP Scalping":
            vwap = ((df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()).iloc[-1]
            return cp < (vwap * 0.985)
    except: return False
    return False

# --- REAL-TIME HEADER ---
current_pos_val = (st.session_state.portfolio['LTP'] * st.session_state.portfolio['Qty']).sum()
live_pnl = st.session_state.portfolio['P&L'].sum()

st.title("🔱 PRIMA COMMAND v9.2")
h1, h2, h3 = st.columns(3)
h1.metric("Cash Balance", f"₹{round(st.session_state.balance, 2)}")
h2.metric("Portfolio Value", f"₹{round(current_pos_val, 2)}", delta=f"{round(live_pnl, 2)}")
h3.metric("Live P&L", f"₹{round(live_pnl, 2)}")

tab1, tab2, tab3 = st.tabs(["📺 Market Watch", "⚔️ Positions", "📊 Risk Dashboard"])

# --- MAIN LOOP ---
watch_space = tab1.empty()
pos_space = tab2.empty()

while True:
    # 1. POSITION REFRESH (1-SEC)
    if not st.session_state.portfolio.empty:
        for idx, row in st.session_state.portfolio.iterrows():
            try:
                t = yf.Ticker(f"{row['Symbol']}.NS")
                ltp = t.fast_info['last_price']
                st.session_state.portfolio.at[idx, 'LTP'] = round(ltp, 2)
                st.session_state.portfolio.at[idx, 'P&L'] = round((ltp - row['Entry']) * row['Qty'], 2)
                
                # SL/TP Guard
                if ltp <= row['Stop'] or ltp >= row['Target']:
                    st.session_state.balance += (ltp * row['Qty'])
                    st.session_state.portfolio = st.session_state.portfolio.drop(idx)
                    st.rerun()
            except: continue

    with pos_space.container():
        st.dataframe(st.session_state.portfolio.drop(columns=['ID']), use_container_width=True, hide_index=True)
        for i, r in st.session_state.portfolio.iterrows():
            if st.button(f"KILL {r['Symbol']}", key=f"k_{r['ID']}_{time.time()}"):
                st.session_state.balance += (r['LTP'] * r['Qty'])
                st.session_state.portfolio = st.session_state.portfolio.drop(i)
                st.rerun()

    # 2. RISK ANALYTICS (Real-Time)
    with tab3:
        if not st.session_state.portfolio.empty:
            sector_counts = st.session_state.portfolio.groupby('Sector')['Qty'].count()
            fig = px.pie(names=sector_counts.index, values=sector_counts.values, hole=0.5, title="Sector Distribution")
            st.plotly_chart(fig, use_container_width=True)

    # 3. SCANNER (2-MIN)
    if time.time() - st.session_state.last_scan > 120:
        results = []
        for _, stock in master_df.iterrows():
            try:
                sym = stock['SYMBOL']
                data = yf.download(f"{sym}.NS", period="2d", interval="15m", progress=False)
                if check_signal(data, active_strat):
                    # Smart Allocation + Sector Limit check
                    current_sector_count = len(st.session_state.portfolio[st.session_state.portfolio['Sector'] == stock['SECTOR']])
                    sector_limit_reached = (current_sector_count / max_trades * 100) >= sector_limit
                    
                    if len(st.session_state.portfolio) < max_trades and not sector_limit_reached:
                        cp = data['Close'].iloc[-1]
                        qty = int(risk_per_trade / cp)
                        if qty > 0:
                            uid = f"{sym}_{secrets.token_hex(2)}"
                            new_trade = {"ID": uid, "Symbol": sym, "Sector": stock['SECTOR'], "Qty": qty, 
                                         "Entry": cp, "LTP": cp, "P&L": 0.0, "Target": cp*1.03, "Stop": cp*0.98}
                            st.session_state.portfolio = pd.concat([st.session_state.portfolio, pd.DataFrame([new_trade])], ignore_index=True)
                            st.session_state.balance -= (cp * qty)
                results.append({"Symbol": sym, "Price": data['Close'].iloc[-1]})
            except: continue
        
        st.session_state.last_scan = time.time()
        with watch_space.container():
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

    time.sleep(1)
