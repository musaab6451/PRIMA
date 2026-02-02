import streamlit as st
import pandas as pd
import yfinance as yf
from nselib import capital_market
import ta
import time
import requests
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="PRIMA Autonomous", layout="wide")

# Updated Telegram with proper error blocks
def broadcast(msg):
    token = "8563714849:AAGYRRGGQupxvvU16RHovQ5QSMOd6vkSS_o"
    chat_ids = ["1303832128","1287509530"] # Add your IDs here
    for cid in chat_ids:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.get(url, params={"chat_id": cid, "text": f"🔱 PRIMA: {msg}"}, timeout=5)
        except Exception:
            pass 

# --- SESSION STATE ---
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=["Time", "Symbol", "Side", "Qty", "Entry", "LTP", "P&L"])
if 'total_pnl' not in st.session_state:
    st.session_state.total_pnl = 0.0

# --- DATA HELPERS ---
@st.cache_data(ttl=86400)
def get_master_list():
    try:
        df = capital_market.equity_list()
        # Rename columns for clarity and filter for top 50
        df = df[['SYMBOL', 'NAME_OF_COMPANY']].head(50)
        return df.to_dict('records')
    except Exception:
        return [{"SYMBOL": "RELIANCE", "NAME_OF_COMPANY": "RELIANCE INDUSTRIES LTD"}]

def get_clean_data(symbol):
    try:
        df = yf.download(f"{symbol}.NS", period="5d", interval="15m", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return None

# --- ENGINE ---
def execute_auto_trade(sym, side, price):
    # Check if already holding this stock to prevent over-exposure
    if not st.session_state.portfolio.empty:
        if sym in st.session_state.portfolio['Symbol'].values:
            return

    new_trade = {
        "Time": datetime.now().strftime("%H:%M"),
        "Symbol": sym,
        "Side": side,
        "Qty": 100,
        "Entry": round(price, 2),
        "LTP": round(price, 2),
        "P&L": 0.0
    }
    st.session_state.portfolio = pd.concat([st.session_state.portfolio, pd.DataFrame([new_trade])], ignore_index=True)
    broadcast(f"AUTO-TRADE: {side} {sym} @ {price}")

# --- UI DISPLAY ---
st.title("🔱 PRIMA AUTONOMOUS TERMINAL v6.1")

# Top Metric Bar
m1, m2 = st.columns(2)
m1.metric("Active Positions", len(st.session_state.portfolio))
m2.metric("Total Session P&L", f"₹{st.session_state.portfolio['P&L'].sum() if not st.session_state.portfolio.empty else 0.0}")

tab1, tab2 = st.tabs(["⚡ Live Scanner", "💼 Portfolio & Management"])

master_list = get_master_list()

with tab1:
    engine_ui = st.empty()

with tab2:
    if not st.session_state.portfolio.empty:
        # We display the portfolio with a "Close" button for each
        for idx, row in st.session_state.portfolio.iterrows():
            c1, c2, c3 = st.columns([2, 2, 1])
            c1.write(f"**{row['Symbol']}** ({row['Side']})")
            c2.write(f"Entry: {row['Entry']} | P&L: ₹{row['P&L']}")
            if c3.button("EXIT", key=f"exit_{idx}"):
                st.session_state.portfolio = st.session_state.portfolio.drop(idx)
                st.rerun()
    else:
        st.info("No active trades found.")

# --- MAIN LOOP ---
while True:
    results = []
    for item in master_list:
        sym = item['SYMBOL']
        name = item['NAME_OF_COMPANY']
        df = get_clean_data(sym)
        
        if df is not None and len(df) > 20:
            cp = float(df['Close'].iloc[-1])
            rsi = ta.momentum.RSIIndicator(df['Close']).rsi().iloc[-1]
            vwap = ((df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()).iloc[-1]
            dist = ((cp - vwap) / vwap) * 100
            
            status = "Scanning"
            # Logic: RSI + VWAP Distance
            if rsi < 30 and dist < -1.2:
                status = "⭐ BUY"
                execute_auto_trade(sym, "BUY", cp)
            elif rsi > 70 and dist > 1.2:
                status = "⭐ SELL"
                execute_auto_trade(sym, "SELL", cp)
                
            results.append({"Company": name, "Symbol": sym, "LTP": cp, "RSI": round(rsi, 1), "Signal": status})

    # Update Active P&L in background
    if not st.session_state.portfolio.empty:
        for idx, row in st.session_state.portfolio.iterrows():
            price_data = yf.download(f"{row['Symbol']}.NS", period="1d", interval="1m", progress=False)
            if not price_data.empty:
                current_ltp = price_data['Close'].iloc[-1]
                st.session_state.portfolio.at[idx, 'LTP'] = round(current_ltp, 2)
                pnl = (current_ltp - row['Entry']) * row['Qty'] if row['Side'] == "BUY" else (row['Entry'] - current_ltp) * row['Qty']
                st.session_state.portfolio.at[idx, 'P&L'] = round(pnl, 2)

    with engine_ui.container():
        st.dataframe(pd.DataFrame(results), width="stretch", hide_index=True)
    
    time.sleep(60)
