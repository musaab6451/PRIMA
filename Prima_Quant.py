import streamlit as st
import pandas as pd
import yfinance as yf
from nselib import capital_market
import ta
import time
import plotly.graph_objects as go
from datetime import datetime
import requests

# --- CONFIG & TELEGRAM ---
st.set_page_config(page_title="PRIMA Autonomous Terminal", layout="wide", page_icon="🔱")
TELEGRAM_TOKEN = "8563714849:AAGYRRGGQupxvvU16RHovQ5QSMOd6vkSS_o"
CHAT_IDS = ["1303832128","1287509530"]

def broadcast(msg):
    for cid in CHAT_IDS:
        try: requests.get(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                          params={"chat_id": cid, "text": f"🔱 PRIMA: {msg}", "parse_mode": "Markdown"})
    except: pass

# --- STATE MANAGEMENT ---
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=["Time", "Symbol", "Side", "Qty", "Entry", "LTP", "P&L"])

# --- DATA HELPERS ---
@st.cache_data(ttl=86400)
def get_master_list():
    df = capital_market.equity_list()
    # Keeping it to top 50 for faster autonomous performance
    return df[['SYMBOL', 'NAME_OF_COMPANY']].head(50).to_dict('records')

def get_clean_data(symbol):
    df = yf.download(f"{symbol}.NS", period="5d", interval="15m", progress=False)
    if df.empty: return None
    # Fix the MultiIndex issue that causes the ValueError
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

# --- SCANNER ENGINE ---
def run_autonomous_engine(master_list):
    results = []
    progress = st.progress(0)
    
    for i, item in enumerate(master_list):
        sym = item['SYMBOL']
        name = item['NAME_OF_COMPANY']
        progress.progress((i + 1) / len(master_list))
        
        df = get_clean_data(sym)
        if df is None or len(df) < 20: continue

        try:
            cp = float(df['Close'].iloc[-1])
            rsi = ta.momentum.RSIIndicator(df['Close']).rsi().iloc[-1]
            vwap = ((df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()).iloc[-1]
            vwap_dist = ((cp - vwap) / vwap) * 100
            vol_spike = df['Volume'].iloc[-1] / df['Volume'].rolling(10).mean().iloc[-1]

            strategy = "Scanning"
            # AUTO-TRADE TRIGGER
            if rsi < 30 and vwap_dist < -1.2 and vol_spike > 1.5:
                strategy = "⭐ GOLDEN BUY"
                execute_auto_trade(sym, "BUY", cp)
            elif rsi > 70 and vwap_dist > 1.2 and vol_spike > 1.5:
                strategy = "⭐ GOLDEN SELL"
                execute_auto_trade(sym, "SELL", cp)

            results.append({"Company": name, "Symbol": sym, "LTP": round(cp, 2), "RSI": round(rsi, 1), "Strategy": strategy})
        except: continue
    return pd.DataFrame(results)

def execute_auto_trade(sym, side, price):
    # Prevent duplicate trades if already open
    if not st.session_state.portfolio.empty:
        if sym in st.session_state.portfolio['Symbol'].values: return
    
    new_trade = {"Time": datetime.now().strftime("%H:%M"), "Symbol": sym, "Side": side, "Qty": 100, 
                 "Entry": round(price,2), "LTP": round(price,2), "P&L": 0.0}
    st.session_state
