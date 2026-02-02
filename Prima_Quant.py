import streamlit as st
import pandas as pd
import yfinance as yf
from nselib import capital_market
import ta
import time
import requests
from datetime import datetime

# --- CONFIG & STYLING ---
st.set_page_config(page_title="PRIMA Terminal v7.6", layout="wide")

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("⚙️ Strategy Settings")
    rsi_threshold = st.slider("RSI Buy Threshold", 10, 50, 30)
    vwap_dist_threshold = st.slider("VWAP Dist % (Negative)", -5.0, -0.5, -1.5)
    stop_loss_val = st.slider("Stop Loss %", 0.5, 5.0, 1.5)
    target_val = st.slider("Target Profit %", 1.0, 10.0, 3.0)
    st.divider()
    if st.button("RESET SESSION"):
        st.session_state.clear()
        st.rerun()

# Telegram Broadcaster
def broadcast(msg):
    token = "8563714849:AAGYRRGGQupxvvU16RHovQ5QSMOd6vkSS_o"
    chat_ids = ["1303832128","1287509530"] 
    for cid in chat_ids:
        try:
            requests.get(f"https://api.telegram.org/bot{token}/sendMessage", 
                         params={"chat_id": cid, "text": f"🔱 PRIMA: {msg}", "parse_mode": "Markdown"}, timeout=1)
        except: pass

# --- SESSION STATE ---
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=["ID", "Symbol", "Side", "Qty", "Entry", "LTP", "P&L", "Target", "Stop"])
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["Time", "Symbol", "Side", "Entry", "Exit", "P&L", "Reason"])
if 'balance' not in st.session_state:
    st.session_state.balance = 50000.0

def style_pnl(val):
    color = 'green' if val > 0 else 'red' if val < 0 else 'white'
    return f'color: {color}; font-weight: bold'

# --- THE ENGINE ---
@st.cache_data(ttl=86400)
def get_100_symbols():
    df = capital_market.equity_list()
    possible_headers = ['NAME_OF_COMPANY', 'NAME OF COMPANY', 'NAME']
    name_col = next((col for col in possible_headers if col in df.columns), 'SYMBOL')
    df = df.rename(columns={name_col: 'Company'})
    return df[['SYMBOL', 'Company']].head(100).to_dict('records')

def close_position(row_id, reason="MANUAL KILL ⚔️"):
    # Find the specific row by its unique ID
    idx = st.session_state.portfolio[st.session_state.portfolio['ID'] == row_id].index[0]
    row = st.session_state.portfolio.loc[idx]
    
    st.session_state.balance += (row['LTP'] * row['Qty'])
    hist_entry = {"Time": datetime.now().strftime("%H:%M"), "Symbol": row['Symbol'], "Side": row['Side'], 
                  "Entry": row['Entry'], "Exit": row['LTP'], "P&L": row['P&L'], "Reason": reason}
    st.session_state.history = pd.concat([st.session_state.history, pd.DataFrame([hist_entry])], ignore_index=True)
    broadcast(f"*CLOSED:* {row['Symbol']} @ {row['LTP']} | P&L: ₹{row['P&L']} ({reason})")
    st.session_state.portfolio = st.session_state.portfolio.drop(idx).reset_index(drop=True)
    st.rerun()

# --- UI LAYOUT ---
st.title("🔱 PRIMA AUTONOMOUS COMMAND v7.6")
m1, m2, m3 = st.columns(3)
total_pnl = st.session_state.portfolio['P&L'].sum() if not st.session_state.portfolio.empty else 0
m1.metric("Wallet Balance", f"₹{round(st.session_state.balance, 2)}")
m2.metric("Positions", len(st.session_state.portfolio))
m3.metric("Live P&L", f"₹{round(total_pnl, 2)}", delta=f"{round(total_pnl, 2)}")

tab1, tab2, tab3 = st.tabs(["📺 Market Watch", "⚔️ Position Control", "📜 History"])

# --- EXECUTION LOOP ---
master_list = get_100_symbols()
scan_placeholder = tab1.empty()

while True:
    # 1. RISK MANAGER & P&L UPDATE
    if not st.session_state.portfolio.empty:
        for idx, row in st.session_state.portfolio.iterrows():
            ticker = yf.Ticker(f"{row['Symbol']}.NS")
            current_ltp = ticker.fast_info['last_price']
            st.session_state.portfolio.at[idx, 'LTP'] = round(current_ltp, 2)
            pnl = (current_ltp - row['Entry']) * row['Qty'] if row['Side'] == "BUY" else (row['Entry'] - current_ltp) * row['Qty']
            st.session_state.portfolio.at[idx, 'P&L'] = round(pnl, 2)
            
            # Auto-Exit Logic
            if current_ltp <= row['Stop']: close_position(row['ID'], "STOP LOSS 🛑")
            elif current_ltp >= row['Target']: close_position(row['ID'], "TARGET HIT 🎯")

    # 2. POSITION CONTROL (Unique Key Fix)
    with tab2:
        if st.session_state.portfolio.empty:
            st.info("No active positions.")
        else:
            st.dataframe(st.session_state.portfolio.drop(columns=['ID']).style.map(style_pnl, subset=['P&L']), width="stretch", hide_index=True)
            # Create unique buttons using ID
            for idx, row in st.session_state.portfolio.iterrows():
                if st.button(f"KILL {row['Symbol']} (ID: {row['ID']})", key=f"kill_{row['ID']}"):
                    close_position(row['ID'])

    # 3. SCANNER
    scan_results = []
    for item in master_list:
        sym = item['SYMBOL']
        try:
            df = yf.download(f"{sym}.NS", period="2d", interval="15m", progress=False)
            if df.empty: continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            cp = float(df['Close'].iloc[-1])
            rsi = ta.momentum.RSIIndicator(df['Close']).rsi().iloc[-1]
            vwap = ((df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()).iloc[-1]
            dist = ((cp - vwap) / vwap) * 100
            
            # Use Sidebar thresholds
            if rsi < rsi_threshold and dist < vwap_dist_threshold:
                if sym not in st.session_state.portfolio['Symbol'].values and st.session_state.balance > (cp * 10):
                    # UNIQUE ID GENERATION
                    unique_id = f"{sym}_{int(time.time())}"
                    new_trade = {
                        "ID": unique_id, "Symbol": sym, "Side": "BUY", "Qty": 10, "Entry": cp, 
                        "LTP": cp, "P&L": 0.0, "Target": cp * (1 + target_val/100), "Stop": cp * (1 - stop_loss_val/100)
                    }
                    st.session_state.portfolio = pd.concat([st.session_state.portfolio, pd.DataFrame([new_trade])], ignore_index=True)
                    st.session_state.balance -= (cp * 10)
                    broadcast(f"🚀 *ENTRY:* {sym} @ {cp}")

            scan_results.append({"Company": item['Company'], "Symbol": sym, "LTP": cp, "RSI": round(rsi,1)})
        except: continue
        
    with scan_placeholder.container():
        st.dataframe(pd.DataFrame(scan_results), width="stretch", hide_index=True)
    
    time.sleep(1)
