import streamlit as st
import pandas as pd
import yfinance as yf
from nselib import capital_market
import ta
import time
import requests
from datetime import datetime
import secrets

# --- CONFIG ---
st.set_page_config(page_title="PRIMA Terminal v8.0", layout="wide")

# Telegram Broadcaster
def broadcast(msg):
    token = "8563714849:AAGYRRGGQupxvvU16RHovQ5QSMOd6vkSS_o"
    chat_ids = ["1303832128","1287509530"] 
    for cid in chat_ids:
        try:
            requests.get(f"https://api.telegram.org/bot{token}/sendMessage", 
                         params={"chat_id": cid, "text": f"🔱 PRIMA: {msg}", "parse_mode": "Markdown"}, timeout=1)
        except Exception: pass

# --- SESSION STATE ---
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=["ID", "Symbol", "Side", "Qty", "Entry", "LTP", "P&L", "Target", "Stop"])
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["Time", "Symbol", "Side", "Entry", "Exit", "P&L", "Reason"])
if 'balance' not in st.session_state:
    st.session_state.balance = 50000.0

# --- FUNCTIONS ---
def close_position(row_id, reason="MANUAL KILL ⚔️"):
    if row_id in st.session_state.portfolio['ID'].values:
        idx = st.session_state.portfolio[st.session_state.portfolio['ID'] == row_id].index[0]
        row = st.session_state.portfolio.loc[idx]
        st.session_state.balance += (row['LTP'] * row['Qty'])
        
        new_hist = pd.DataFrame([{"Time": datetime.now().strftime("%H:%M:%S"), "Symbol": row['Symbol'], "Side": row['Side'], 
                                  "Entry": row['Entry'], "Exit": row['LTP'], "P&L": row['P&L'], "Reason": reason}])
        st.session_state.history = pd.concat([st.session_state.history, new_hist], ignore_index=True)
        
        broadcast(f"*CLOSED:* {row['Symbol']} | P&L: ₹{row['P&L']}")
        st.session_state.portfolio = st.session_state.portfolio.drop(idx).reset_index(drop=True)
        st.rerun()

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔱 Controls")
    if st.button("🚨 GLOBAL KILL", use_container_width=True, type="primary"):
        if not st.session_state.portfolio.empty:
            for rid in st.session_state.portfolio['ID'].tolist():
                close_position(rid, "GLOBAL SHUTDOWN 🔌")
    
    rsi_buy = st.slider("RSI Buy Zone", 10, 50, 30)
    vwap_dist = st.slider("VWAP Dist %", -5.0, -0.5, -1.5)

# --- UI TABS ---
st.title("🔱 PRIMA AUTONOMOUS COMMAND v8.0")
m1, m2, m3 = st.columns(3)
total_pnl = st.session_state.portfolio['P&L'].sum() if not st.session_state.portfolio.empty else 0
m1.metric("Wallet", f"₹{round(st.session_state.balance, 2)}")
m2.metric("Positions", len(st.session_state.portfolio))
m3.metric("Live P&L", f"₹{round(total_pnl, 2)}", delta=f"{round(total_pnl, 2)}")

tab1, tab2, tab3 = st.tabs(["📺 Market Watch", "⚔️ Position Control", "📜 History"])

# --- THE FIX: EMPTY PLACEHOLDERS ---
# We define these OUTSIDE the loop so they are only created once
portfolio_container = tab2.empty()
history_container = tab3.empty()
scanner_container = tab1.empty()

@st.cache_data(ttl=86400)
def get_100_symbols():
    df = capital_market.equity_list()
    name_col = next((c for c in ['NAME_OF_COMPANY', 'NAME OF COMPANY', 'NAME'] if c in df.columns), 'SYMBOL')
    return df.rename(columns={name_col: 'Company'})[['SYMBOL', 'Company']].head(100).to_dict('records')

master_list = get_100_symbols()

# --- MAIN LOOP ---
while True:
    # 1. RISK & P&L (Back-end logic)
    if not st.session_state.portfolio.empty:
        for idx, row in st.session_state.portfolio.iterrows():
            try:
                ticker = yf.Ticker(f"{row['Symbol']}.NS")
                current_ltp = ticker.fast_info['last_price']
                st.session_state.portfolio.at[idx, 'LTP'] = round(current_ltp, 2)
                st.session_state.portfolio.at[idx, 'P&L'] = round((current_ltp - row['Entry']) * row['Qty'], 2)
                
                if current_ltp <= row['Stop']: close_position(row['ID'], "STOP LOSS 🛑")
                elif current_ltp >= row['Target']: close_position(row['ID'], "TARGET HIT 🎯")
            except: continue

    # 2. RENDER PORTFOLIO (Tab 2)
    with portfolio_container.container():
        if st.session_state.portfolio.empty:
            st.info("No active trades.")
        else:
            # We display the data but only create buttons if the tab is being viewed
            st.dataframe(st.session_state.portfolio.drop(columns=['ID']), width="stretch", hide_index=True)
            for idx, row in st.session_state.portfolio.iterrows():
                # Added a secondary random salt to the key to be absolutely safe
                if st.button(f"KILL {row['Symbol']}", key=f"btn_{row['ID']}_{secrets.token_hex(2)}"):
                    close_position(row['ID'])

    # 3. RENDER HISTORY (Tab 3)
    with history_container.container():
        st.dataframe(st.session_state.history, width="stretch", hide_index=True)

    # 4. SCANNER (Tab 1)
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
            
            if rsi < rsi_buy and dist < vwap_dist:
                if sym not in st.session_state.portfolio['Symbol'].values and st.session_state.balance > (cp * 10):
                    uid = f"{sym}_{int(time.time())}_{secrets.token_hex(3)}"
                    new_trade = {"ID": uid, "Symbol": sym, "Side": "BUY", "Qty": 10, "Entry": cp, "LTP": cp, "P&L": 0.0, "Target": cp*1.03, "Stop": cp*0.985}
                    st.session_state.portfolio = pd.concat([st.session_state.portfolio, pd.DataFrame([new_trade])], ignore_index=True)
                    st.session_state.balance -= (cp * 10)
                    broadcast(f"🚀 *ENTRY:* {sym} @ {cp}")

            scan_results.append({"Company": item['Company'], "Symbol": sym, "LTP": cp, "RSI": round(rsi,1)})
        except: continue
        
    with scanner_container.container():
        st.dataframe(pd.DataFrame(scan_results), width="stretch", hide_index=True)
    
    time.sleep(1)
