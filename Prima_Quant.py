import streamlit as st
import pandas as pd
import yfinance as yf
from nselib import capital_market
import ta
import time
import requests
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="PRIMA High-Speed Terminal", layout="wide")

# RISK PARAMS
INITIAL_BALANCE = 50000.0
STOP_LOSS_PCT = 1.5  
TAKE_PROFIT_PCT = 3.0 

# TELEGRAM
def broadcast(msg):
    token = "8563714849:AAGYRRGGQupxvvU16RHovQ5QSMOd6vkSS_o"
    chat_ids = ["1303832128","1287509530"] 
    for cid in chat_ids:
        try: requests.get(f"https://api.telegram.org/bot{token}/sendMessage", 
                          params={"chat_id": cid, "text": f"🔱 PRIMA: {msg}"}, timeout=1)
        except: pass

# --- SESSION STATE ---
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=["Symbol", "Side", "Qty", "Entry", "LTP", "P&L", "Target", "Stop"])
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["Time", "Symbol", "Side", "Entry", "Exit", "P&L", "Reason"])
if 'balance' not in st.session_state:
    st.session_state.balance = INITIAL_BALANCE

# --- DATA ENGINES ---
@st.cache_data(ttl=86400)
def get_100_symbols():
    df = capital_market.equity_list()
    return df[['SYMBOL', 'NAME_OF_COMPANY']].head(100).to_dict('records')

def manage_risk():
    if st.session_state.portfolio.empty: return
    for idx, row in st.session_state.portfolio.iterrows():
        try:
            # Fast fetch for LTP
            ticker = yf.Ticker(f"{row['Symbol']}.NS")
            current_ltp = ticker.fast_info['last_price']
            
            st.session_state.portfolio.at[idx, 'LTP'] = round(current_ltp, 2)
            pnl = (current_ltp - row['Entry']) * row['Qty'] if row['Side'] == "BUY" else (row['Entry'] - current_ltp) * row['Qty']
            st.session_state.portfolio.at[idx, 'P&L'] = round(pnl, 2)

            # RISK CHECK
            reason = ""
            if current_ltp <= row['Stop']: reason = "STOP LOSS 🛑"
            elif current_ltp >= row['Target']: reason = "TARGET HIT 🎯"
            
            if reason:
                st.session_state.balance += (current_ltp * row['Qty'])
                hist_entry = {"Time": datetime.now().strftime("%H:%M"), "Symbol": row['Symbol'], "Side": row['Side'], 
                              "Entry": row['Entry'], "Exit": round(current_ltp, 2), "P&L": round(pnl, 2), "Reason": reason}
                st.session_state.history = pd.concat([st.session_state.history, pd.DataFrame([hist_entry])], ignore_index=True)
                broadcast(f"CLOSED: {row['Symbol']} @ {round(current_ltp,2)} ({reason})")
                st.session_state.portfolio = st.session_state.portfolio.drop(idx)
                st.rerun()
        except: continue

# --- UI ---
st.title("🔱 PRIMA HIGH-SPEED TERMINAL v7.1")
m1, m2, m3 = st.columns(3)
total_pnl = st.session_state.portfolio['P&L'].sum() if not st.session_state.portfolio.empty else 0
m1.metric("Wallet Balance", f"₹{round(st.session_state.balance, 2)}")
m2.metric("Open Positions", len(st.session_state.portfolio))
m3.metric("Live Session P&L", f"₹{round(total_pnl, 2)}", delta=f"{round(total_pnl, 2)}")

tab1, tab2, tab3 = st.tabs(["📺 Market Scanner (100 Stocks)", "💰 Active Positions", "📜 Trade History"])

# --- EXECUTION LOOP ---
master_list = get_100_symbols()
scan_placeholder = tab1.empty()

while True:
    # 1. RISK MANAGER (Runs every loop - ~1s)
    manage_risk()
    with tab2:
        st.dataframe(st.session_state.portfolio, width="stretch", hide_index=True)
    with tab3:
        st.dataframe(st.session_state.history, width="stretch", hide_index=True)

    # 2. FULL SCANNER (Batch Process)
    scan_data = []
    progress_bar = st.progress(0)
    
    for i, item in enumerate(master_list):
        progress_bar.progress((i + 1) / len(master_list))
        sym = item['SYMBOL']
        try:
            df = yf.download(f"{sym}.NS", period="5d", interval="15m", progress=False)
            if df.empty: continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

            cp = float(df['Close'].iloc[-1])
            rsi = ta.momentum.RSIIndicator(df['Close']).rsi().iloc[-1]
            vwap = ((df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()).iloc[-1]
            dist = ((cp - vwap) / vwap) * 100
            
            # AUTOMATIC BUY LOGIC
            if rsi < 30 and dist < -1.5 and st.session_state.balance > (cp * 10):
                if sym not in st.session_state.portfolio['Symbol'].values:
                    qty = 10
                    new_trade = {"Symbol": sym, "Side": "BUY", "Qty": qty, "Entry": cp, "LTP": cp, 
                                 "P&L": 0.0, "Target": cp*1.03, "Stop": cp*0.985}
                    st.session_state.portfolio = pd.concat([st.session_state.portfolio, pd.DataFrame([new_trade])], ignore_index=True)
                    st.session_state.balance -= (cp * qty)
                    broadcast(f"BUY EXECUTION: {sym} @ {cp}")

            scan_data.append({"Company": item['NAME_OF_COMPANY'], "Symbol": sym, "LTP": cp, "RSI": round(rsi,1)})
        except: continue
    
    # After full scan, update the Market Watch table
    with scan_placeholder.container():
        st.dataframe(pd.DataFrame(scan_data), width="stretch", hide_index=True)
    
    progress_bar.empty()
    time.sleep(1) # Small rest between full scan cycles
