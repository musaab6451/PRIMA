import streamlit as st
import pandas as pd
import yfinance as yf
from nselib import capital_market
import ta
import time
import requests
from datetime import datetime

# --- CONFIG & BRANDING ---
st.set_page_config(page_title="PRIMA High-Speed Terminal", layout="wide")

# --- RISK SETTINGS ---
INITIAL_BALANCE = 50000.0
STOP_LOSS_PCT = 1.5  # Auto-exit if stock drops 1.5%
TAKE_PROFIT_PCT = 3.0 # Auto-exit if stock gains 3.0%

# --- TELEGRAM BROADCAST ---
def broadcast(msg):
    token = "8563714849:AAGYRRGGQupxvvU16RHovQ5QSMOd6vkSS_o"
    chat_ids = ["1303832128","1287509530"] # Update with your IDs
    for cid in chat_ids:
        try:
            requests.get(f"https://api.telegram.org/bot{token}/sendMessage", 
                         params={"chat_id": cid, "text": f"🔱 PRIMA: {msg}"}, timeout=2)
        except: pass

# --- SESSION STATE ---
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=["Symbol", "Side", "Qty", "Entry", "LTP", "P&L", "Target", "Stop"])
if 'balance' not in st.session_state:
    st.session_state.balance = INITIAL_BALANCE

# --- DATA ENGINES ---
@st.cache_data(ttl=86400)
def get_100_symbols():
    try:
        df = capital_market.equity_list()
        return df[['SYMBOL', 'NAME_OF_COMPANY']].head(100).to_dict('records')
    except:
        return [{"SYMBOL": "RELIANCE", "NAME_OF_COMPANY": "RELIANCE INDUSTRIES"}]

# --- RISK MANAGEMENT ENGINE (REAL-TIME 1-SEC) ---
def manage_risk():
    if st.session_state.portfolio.empty:
        return

    for idx, row in st.session_state.portfolio.iterrows():
        try:
            # High-speed check for current price
            ticker = yf.Ticker(f"{row['Symbol']}.NS")
            current_ltp = ticker.fast_info['last_price']
            
            # Update LTP and P&L
            st.session_state.portfolio.at[idx, 'LTP'] = round(current_ltp, 2)
            pnl = (current_ltp - row['Entry']) * row['Qty'] if row['Side'] == "BUY" else (row['Entry'] - current_ltp) * row['Qty']
            st.session_state.portfolio.at[idx, 'P&L'] = round(pnl, 2)

            # AUTO-EXIT LOGIC
            exit_triggered = False
            reason = ""
            
            if current_ltp <= row['Stop'] and row['Side'] == "BUY":
                exit_triggered = True; reason = "STOP LOSS 🛑"
            elif current_ltp >= row['Target'] and row['Side'] == "BUY":
                exit_triggered = True; reason = "PROFIT TARGET 🎯"
            
            if exit_triggered:
                st.session_state.balance += (current_ltp * row['Qty']) # Return capital
                broadcast(f"AUTO-EXIT: {row['Symbol']} @ {round(current_ltp,2)} ({reason})")
                st.session_state.portfolio = st.session_state.portfolio.drop(idx)
                st.rerun()
        except:
            continue

# --- UI LAYOUT ---
st.title("🔱 PRIMA HIGH-SPEED TERMINAL v7.0")

# Header Stats
c1, c2, c3 = st.columns(3)
total_pnl = st.session_state.portfolio['P&L'].sum() if not st.session_state.portfolio.empty else 0
c1.metric("Wallet Balance", f"₹{round(st.session_state.balance, 2)}")
c2.metric("Open Positions", len(st.session_state.portfolio))
c3.metric("Live Session P&L", f"₹{round(total_pnl, 2)}", delta=f"{round(total_pnl, 2)}")

tab1, tab2 = st.tabs(["📺 Market Watch (60s)", "💰 Live Portfolio (1s)"])

# --- MAIN OPERATIONAL LOOP ---
master_list = get_100_symbols()
last_scan_time = 0

while True:
    # 1. THE 1-SECOND MONITOR (Portfolio & Risk)
    manage_risk()
    with tab2:
        st.subheader("Active Risk Management")
        if not st.session_state.portfolio.empty:
            st.dataframe(st.session_state.portfolio, width="stretch", hide_index=True)
        else:
            st.info("No active trades. Scanning for Golden entries...")

    # 2. THE 60-SECOND SCANNER (New Trades)
    current_time = time.time()
    if current_time - last_scan_time > 60:
        scan_results = []
        progress = st.progress(0)
        
        for i, item in enumerate(master_list):
            progress.progress((i + 1) / len(master_list))
            sym = item['SYMBOL']
            try:
                df = yf.download(f"{sym}.NS", period="5d", interval="15m", progress=False)
                if df.empty or len(df) < 20: continue
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

                cp = float(df['Close'].iloc[-1])
                rsi = ta.momentum.RSIIndicator(df['Close']).rsi().iloc[-1]
                vwap = ((df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()).iloc[-1]
                dist = ((cp - vwap) / vwap) * 100
                
                # Signal Logic
                if rsi < 30 and dist < -1.5 and st.session_state.balance > (cp * 10):
                    # EXECUTE BUY
                    qty = 10 # Sample fixed quantity
                    stop_price = cp * (1 - STOP_LOSS_PCT/100)
                    target_price = cp * (1 + TAKE_PROFIT_PCT/100)
                    
                    new_trade = {
                        "Symbol": sym, "Side": "BUY", "Qty": qty, "Entry": cp, 
                        "LTP": cp, "P&L": 0.0, "Target": target_price, "Stop": stop_price
                    }
                    st.session_state.portfolio = pd.concat([st.session_state.portfolio, pd.DataFrame([new_trade])], ignore_index=True)
                    st.session_state.balance -= (cp * qty)
                    broadcast(f"ENTRY: BUY {sym} @ {cp} | SL: {round(stop_price,2)}")
                
                scan_results.append({"Company": item['NAME_OF_COMPANY'], "Symbol": sym, "LTP": cp, "RSI": round(rsi,1)})
            except: continue
        
        with tab1:
            st.dataframe(pd.DataFrame(scan_results), width="stretch", hide_index=True)
        
        progress.empty()
        last_scan_time = current_time

    time.sleep(1) # Frequency of the Risk Manager
