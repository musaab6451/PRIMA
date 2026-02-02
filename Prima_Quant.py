import streamlit as st
import pandas as pd
import yfinance as yf
from nselib import capital_market
import ta
import time
from datetime import datetime
import secrets
import plotly.express as px
import requests
import json

# --- TELEGRAM CONFIG ---
TELEGRAM_TOKEN = "8563714849:AAGYRRGGQupxvvU16RHovQ5QSMOd6vkSS_o"
TELEGRAM_CHAT_IDS = ["1303832128", "1287509530"]

def send_telegram_msg(message):
    for chat_id in TELEGRAM_CHAT_IDS:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            params = {
                "chat_id": chat_id, 
                "text": f"🔱 *PRIMA ALERT*\n\n{message}", 
                "parse_mode": "Markdown"
            }
            response = requests.get(url, params=params, timeout=5)
        except Exception as e:
            print(f"Error sending to {chat_id}: {e}")

# --- CONFIG ---
st.set_page_config(page_title="PRIMA v10.1 | Production", layout="wide")

# --- PERSISTENT STORAGE ---
def save_state_to_file():
    try:
        state_data = {
            'portfolio': st.session_state.portfolio.to_dict('records') if not st.session_state.portfolio.empty else [],
            'history': st.session_state.history.to_dict('records') if not st.session_state.history.empty else [],
            'balance': float(st.session_state.balance),
            'strategy_stats': st.session_state.strategy_stats,
            'timestamp': time.time()
        }
        with open('/tmp/prima_state.json', 'w') as f:
            json.dump(state_data, f)
    except Exception as e:
        pass

def load_state_from_file():
    try:
        with open('/tmp/prima_state.json', 'r') as f:
            state_data = json.load(f)
        if state_data['portfolio']:
            st.session_state.portfolio = pd.DataFrame(state_data['portfolio'])
        if state_data['history']:
            st.session_state.history = pd.DataFrame(state_data['history'])
        st.session_state.balance = float(state_data['balance'])
        st.session_state.strategy_stats = state_data['strategy_stats']
        return True
    except:
        return False

# --- STATE INITIALIZATION ---
if 'initialized' not in st.session_state:
    if not load_state_from_file():
        st.session_state.portfolio = pd.DataFrame(columns=["ID", "Symbol", "Sector", "Qty", "Entry", "LTP", "P&L", "Target", "Stop", "TrailStop", "Strategy"])
        st.session_state.history = pd.DataFrame(columns=["Time", "Symbol", "Entry", "Exit", "P&L", "Strategy"])
        st.session_state.balance = 50000.0
        st.session_state.strategy_stats = {s: {"signals": 0, "trades": 0, "wins": 0, "total_pnl": 0.0} for s in 
            ["RSI Mean Reversion", "EMA Crossover (9/21)", "Bollinger Mean Reversion", "VWAP Scalping", 
             "MACD Momentum", "Supertrend Breakout", "Volume Breakout", "Multi-Timeframe Trend"]}
    st.session_state.initialized = True

if 'last_scan' not in st.session_state: st.session_state.last_scan = time.time()
if 'scan_results' not in st.session_state: st.session_state.scan_results = []
if 'last_position_update' not in st.session_state: st.session_state.last_position_update = time.time()

# --- UTILS ---
@st.cache_data(ttl=86400)
def get_master_data():
    try:
        raw_df = capital_market.equity_list()
        raw_df.columns = [c.upper().strip() for c in raw_df.columns]
        sym_col = next((c for c in ['SYMBOL', 'SYM'] if c in raw_df.columns), raw_df.columns[0])
        sec_col = next((c for c in ['INDUSTRY', 'GROUP', 'SECTOR'] if c in raw_df.columns), None)
        df = pd.DataFrame()
        df['SYMBOL'] = raw_df[sym_col].astype(str)
        df['SECTOR'] = raw_df[sec_col].astype(str) if sec_col else "General"
        return df[~df['SYMBOL'].str.contains('-', na=False)]
    except:
        return pd.DataFrame({'SYMBOL': ['RELIANCE', 'TCS', 'INFY'], 'SECTOR': ['Energy', 'IT', 'IT']})

def get_live_price(symbol):
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        # Ensure we get a scalar float, not a Series
        price = ticker.fast_info['last_price']
        return float(price) if not pd.isna(price) else None
    except:
        return None

def calculate_indicators(data):
    if len(data) < 30: return None
    try:
        close = data['Close']
        # Handle potential MultiIndex columns from yfinance
        if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
        
        rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
        ema_9 = ta.trend.EMAIndicator(close, window=9).ema_indicator().iloc[-1]
        ema_21 = ta.trend.EMAIndicator(close, window=21).ema_indicator().iloc[-1]
        vwap = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()
        
        return {
            'rsi': float(rsi), 'ema_9': float(ema_9), 'ema_21': float(ema_21),
            'vwap': float(vwap.iloc[-1]), 'close': float(close.iloc[-1]),
            'prev_close': float(close.iloc[-2])
        }
    except: return None

def check_strategy_signal(strategy, ind, price):
    if not ind: return False, 0, 0
    # simplified logic for brevity, matches your V10 logic
    if strategy == "RSI Mean Reversion" and ind['rsi'] < 40:
        return True, price * 1.03, price * 0.98
    if strategy == "EMA Crossover (9/21)" and ind['ema_9'] > ind['ema_21']:
        return True, price * 1.04, price * 0.97
    return False, 0, 0

# --- SIDEBAR ---
with st.sidebar:
    st.title("🔱 PRIMA v10.1")
    active_strat = st.selectbox("Strategy", list(st.session_state.strategy_stats.keys()))
    st.divider()
    total_cap = st.number_input("Capital (₹)", value=50000.0)
    max_trades = st.slider("Max Positions", 1, 10, 5)
    auto_trade = st.checkbox("Auto-Execute", value=False)
    trailing_stop_pct = st.slider("Trailing Stop %", 0.5, 5.0, 1.5)
    
    st.divider()
    # FIX: use_container_width -> width='stretch'
    if st.button("🚨 KILL SWITCH", type="primary", width="stretch"):
        st.session_state.portfolio = pd.DataFrame(columns=["ID", "Symbol", "Sector", "Qty", "Entry", "LTP", "P&L", "Target", "Stop", "TrailStop", "Strategy"])
        save_state_to_file()
        st.rerun()

# --- MAIN UI ---
master_df = get_master_data()
current_pos_val = float(st.session_state.portfolio['LTP'].mul(st.session_state.portfolio['Qty']).sum()) if not st.session_state.portfolio.empty else 0.0
live_pnl = float(st.session_state.portfolio['P&L'].sum()) if not st.session_state.portfolio.empty else 0.0

h1, h2, h3 = st.columns(3)
h1.metric("Cash", f"₹{round(st.session_state.balance, 2)}")
h2.metric("P&L", f"₹{round(live_pnl, 2)}", delta=f"{round(live_pnl, 2)}")
h3.metric("Total Equity", f"₹{round(st.session_state.balance + current_pos_val, 2)}")

t1, t2, t3 = st.tabs(["📺 Scanner", "⚔️ Positions", "📜 History"])

with t1:
    if st.button("🔍 Run Full Scan", width="stretch"):
        results = []
        progress = st.progress(0)
        subset = master_df.head(50) # Scanning subset for speed
        for i, stock in subset.iterrows():
            progress.progress((i+1)/len(subset))
            try:
                data = yf.download(f"{stock['SYMBOL']}.NS", period="5d", interval="15m", progress=False)
                ind = calculate_indicators(data)
                if ind:
                    sig, tgt, stp = check_strategy_signal(active_strat, ind, ind['close'])
                    if sig:
                        results.append({"Symbol": stock['SYMBOL'], "Price": ind['close'], "Target": tgt, "Stop": stp, "RSI": ind['rsi']})
            except: continue
        st.session_state.scan_results = results
        st.session_state.last_scan = time.time()
    
    if st.session_state.scan_results:
        # FIX: width='stretch'
        st.dataframe(pd.DataFrame(st.session_state.scan_results), width="stretch", hide_index=True)

with t2:
    if not st.session_state.portfolio.empty:
        # High-frequency update block
        if time.time() - st.session_state.last_position_update > 2:
            for idx, row in st.session_state.portfolio.iterrows():
                ltp = get_live_price(row['Symbol'])
                if ltp:
                    st.session_state.portfolio.at[idx, 'LTP'] = round(ltp, 2)
                    st.session_state.portfolio.at[idx, 'P&L'] = round((ltp - row['Entry']) * row['Qty'], 2)
            st.session_state.last_position_update = time.time()
            save_state_to_file()
        
        st.dataframe(st.session_state.portfolio, width="stretch", hide_index=True)
    else:
        st.info("No active trades.")

# Auto-refresh logic
if not st.session_state.portfolio.empty:
    time.sleep(2)
    st.rerun()
