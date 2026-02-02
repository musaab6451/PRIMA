import streamlit as st
import pandas as pd
import yfinance as yf
from nselib import capital_market
import ta
import time
import requests
from datetime import datetime
import io

# --- CONFIG ---
st.set_page_config(page_title="PRIMA Terminal v7.8", layout="wide")

# Telegram Broadcaster
def broadcast(msg):
    token = "8563714849:AAGYRRGGQupxvvU16RHovQ5QSMOd6vkSS_o"
    chat_ids = ["1303832128","1287509530"] 
    for cid in chat_ids:
        try:
            requests.get(f"https://api.telegram.org/bot{token}/sendMessage", 
                         params={"chat_id": cid, "text": f"🔱 PRIMA: {msg}", "parse_mode": "Markdown"}, timeout=1)
        except Exception:
            pass

# --- SESSION STATE INITIALIZATION ---
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=["ID", "Symbol", "Side", "Qty", "Entry", "LTP", "P&L", "Target", "Stop"])
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["Time", "Symbol", "Side", "Entry", "Exit", "P&L", "Reason"])
if 'balance' not in st.session_state:
    st.session_state.balance = 50000.0

# --- CORE FUNCTIONS ---
def close_position(row_id, reason="MANUAL KILL ⚔️"):
    if row_id in st.session_state.portfolio['ID'].values:
        idx = st.session_state.portfolio[st.session_state.portfolio['ID'] == row_id].index[0]
        row = st.session_state.portfolio.loc[idx]
        
        st.session_state.balance += (row['LTP'] * row['Qty'])
        hist_entry = {
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Symbol": row['Symbol'],
            "Side": row['Side'], 
            "Entry": row['Entry'],
            "Exit": row['LTP'],
            "P&L": row['P&L'],
            "Reason": reason
        }
        st.session_state.history = pd.concat([st.session_state.history, pd.DataFrame([hist_entry])], ignore_index=True)
        broadcast(f"*CLOSED:* {row['Symbol']} | P&L: ₹{row['P&L']} ({reason})")
        st.session_state.portfolio = st.session_state.portfolio.drop(idx).reset_index(drop=True)
        return True
    return False

# --- SIDEBAR & GLOBAL CONTROLS ---
with st.sidebar:
    st.header("🔱 Global Controls")
    
    if st.button("🚨 GLOBAL KILL SWITCH", use_container_width=True, type="primary"):
        if not st.session_state.portfolio.empty:
            # Create a copy of IDs to avoid "index changed during iteration" errors
            ids = st.session_state.portfolio['ID'].tolist()
            for rid in ids:
                close_position(rid, "GLOBAL SHUTDOWN 🔌")
            st.success("All positions closed.")
            st.rerun()

    st.header("⚙️ Strategy Tuning")
    rsi_threshold = st.slider("RSI Buy Zone", 10, 50, 30)
    vwap_dist_threshold = st.slider("VWAP Dist %", -5.0, -0.5, -1.5)
    
    st.divider()
    
    # Export History Feature
    if not st.session_state.history.empty:
        csv = st.session_state.history.to_csv(index=False).encode('utf-8')
        st.download_button("📥 DOWNLOAD TRADE LOG", data=csv, file_name=f"PRIMA_Log_{datetime.now().date()}.csv", mime='text/csv', use_container_width=True)

# --- UI TABS ---
st.title("🔱 PRIMA AUTONOMOUS COMMAND v7.8")
m1, m2, m3 = st.columns(3)
total_pnl = st.session_state.portfolio['P&L'].sum() if not st.session_state.portfolio.empty else 0
m1.metric("Wallet", f"₹{round(st.session_state.balance, 2)}")
m2.metric("Positions", len(st.session_state.portfolio))
m3.metric("Live P&L", f"₹{round(total_pnl, 2)}", delta=f"{round(total_pnl, 2)}")

tab1, tab2, tab3 = st.tabs(["📺 Market Watch", "⚔️ Position Control", "📜 History"])

# --- DATA FETCHING ---
@st.cache_data(ttl=86400)
def get_100_symbols():
    try:
        df = capital_market.equity_list()
        name_col = next((c for c in ['NAME_OF_COMPANY', 'NAME OF COMPANY', 'NAME'] if c in df.columns), 'SYMBOL')
        return df.rename(columns={name_col: 'Company'})[['SYMBOL', 'Company']].head(100).to_dict('records')
    except Exception:
        return [{"SYMBOL": "RELIANCE", "Company": "RELIANCE INDUSTRIES"}]

master_list = get_100_symbols()
scan_placeholder = tab1.empty()

# --- MAIN LOOP ---
while True:
    # 1. LIVE RISK MANAGEMENT (1-SEC)
    if not st.session_state.portfolio.empty:
        for idx, row in st.session_state.portfolio.iterrows():
            try:
                ticker = yf.Ticker(f"{row['Symbol']}.NS")
                current_ltp = ticker.fast_info['last_price']
                st.session_state.portfolio.at[idx, 'LTP'] = round(current_ltp, 2)
                pnl = (current_ltp - row['Entry']) * row['Qty']
                st.session_state.portfolio.at[idx, 'P&L'] = round(pnl, 2)
                
                # Check Auto-Exits
                if current_ltp <= row['Stop']: 
                    close_position(row['ID'], "STOP LOSS 🛑")
                    st.rerun()
                elif current_ltp >= row['Target']: 
                    close_position(row['ID'], "TARGET HIT 🎯")
                    st.rerun()
            except Exception:
                continue

    # 2. POSITION CONTROL UI
    with tab2:
        if st.session_state.portfolio.empty:
            st.info("No active trades.")
        else:
            display_df = st.session_state.portfolio.copy()
            if 'ID' in display_df.columns: display_df = display_df.drop(columns=['ID'])
            st.dataframe(display_df, width="stretch", hide_index=True)
            
            # Individual Kill Buttons
            for idx, row in st.session_state.portfolio.iterrows():
                if st.button(f"KILL {row['Symbol']}", key=f"k_{row['ID']}"):
                    close_position(row['ID'])
                    st.rerun()

    # 3. SCANNER ENGINE (60-SEC)
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
            
            # TRADE NOTIFICATION & ENTRY
            if rsi < rsi_threshold and dist < vwap_dist_threshold:
                # Notify on Telegram even if we don't buy
                broadcast(f"⚡ *MATCH:* {sym} (RSI: {round(rsi,1)})")
                
                if sym not in st.session_state.portfolio['Symbol'].values and st.session_state.balance > (cp * 10):
                    uid = f"{sym}_{int(time.time())}"
                    new_trade = {
                        "ID": uid, "Symbol": sym, "Side": "BUY", "Qty": 10, "Entry": cp, 
                        "LTP": cp, "P&L": 0.0, "Target": cp*1.03, "Stop": cp*0.985
                    }
                    st.session_state.portfolio = pd.concat([st.session_state.portfolio, pd.DataFrame([new_trade])], ignore_index=True)
                    st.session_state.balance -= (cp * 10)
                    broadcast(f"🚀 *ENTRY:* {sym} @ {cp}")

            scan_results.append({"Company": item['Company'], "Symbol": sym, "LTP": cp, "RSI": round(rsi,1)})
        except Exception:
            continue
        
    with scan_placeholder.container():
        st.dataframe(pd.DataFrame(scan_results), width="stretch", hide_index=True)
        with tab3:
            st.dataframe(st.session_state.history, width="stretch", hide_index=True)
    
    time.sleep(1)
