import streamlit as st
import pandas as pd
from kiteconnect import KiteConnect
from nselib import capital_market
import ta
import time
from datetime import datetime, timedelta
import secrets
import requests
import json

# --- KITE CONFIG (From image_2d7620.png) ---
API_KEY = "pcngqkv3k0i0i35o" 
API_SECRET = "4m8oueyj8m4e44qaym3elkla6rfptn27"

# --- TELEGRAM CONFIG ---
TELEGRAM_TOKEN = "8563714849:AAGYRRGGQupxvvU16RHovQ5QSMOd6vkSS_o"
TELEGRAM_CHAT_IDS = ["1303832128", "1287509530"]

def send_telegram_msg(message):
    for chat_id in TELEGRAM_CHAT_IDS:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            params = {"chat_id": chat_id, "text": f"🔱 *PRIMA ALERT*\n\n{message}", "parse_mode": "Markdown"}
            requests.get(url, params=params, timeout=5)
        except: pass

# --- CONFIG ---
st.set_page_config(page_title="PRIMA v10.0 | Kite Connect", layout="wide")

# --- KITE SESSION MANAGEMENT ---
if 'kite' not in st.session_state:
    st.session_state.kite = KiteConnect(api_key=API_KEY)
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'instrument_map' not in st.session_state:
    st.session_state.instrument_map = {}

# --- HELPER FUNCTIONS ---
def get_historical_data(symbol, interval="15minute", days=5):
    """Fetches real-time historical candles from Zerodha"""
    try:
        token = st.session_state.instrument_map.get(symbol)
        if not token: return pd.DataFrame()
        
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        
        records = st.session_state.kite.historical_data(token, from_date, to_date, interval)
        df = pd.DataFrame(records)
        # Rename to match 'ta' library expectations
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        return df
    except Exception as e:
        print(f"Error fetching historical for {symbol}: {e}")
        return pd.DataFrame()

def get_live_price(symbol):
    """Fetch live LTP via Kite"""
    try:
        resp = st.session_state.kite.ltp(f"NSE:{symbol}")
        return float(resp[f"NSE:{symbol}"]["last_price"])
    except:
        return None

# --- STATE INITIALIZATION ---
if 'initialized' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=["ID", "Symbol", "Sector", "Qty", "Entry", "LTP", "P&L", "Target", "Stop", "TrailStop", "Strategy"])
    st.session_state.history = pd.DataFrame(columns=["Time", "Symbol", "Entry", "Exit", "P&L", "Strategy"])
    st.session_state.balance = 50000.0
    st.session_state.strategy_stats = {
        "RSI Mean Reversion": {"signals": 0, "trades": 0, "wins": 0, "total_pnl": 0.0},
        "EMA Crossover (9/21)": {"signals": 0, "trades": 0, "wins": 0, "total_pnl": 0.0},
        "Bollinger Mean Reversion": {"signals": 0, "trades": 0, "wins": 0, "total_pnl": 0.0},
        "VWAP Scalping": {"signals": 0, "trades": 0, "wins": 0, "total_pnl": 0.0},
        "MACD Momentum": {"signals": 0, "trades": 0, "wins": 0, "total_pnl": 0.0},
        "Supertrend Breakout": {"signals": 0, "trades": 0, "wins": 0, "total_pnl": 0.0},
        "Volume Breakout": {"signals": 0, "trades": 0, "wins": 0, "total_pnl": 0.0},
        "Multi-Timeframe Trend": {"signals": 0, "trades": 0, "wins": 0, "total_pnl": 0.0}
    }
    st.session_state.last_scan = time.time()
    st.session_state.scan_results = []
    st.session_state.last_position_update = time.time()
    st.session_state.initialized = True

# --- SIDEBAR: AUTHENTICATION & CONTROLS ---
with st.sidebar:
    st.title("🔱 PRIMA v10.0")
    
    # Kite Authentication Flow
    if not st.session_state.authenticated:
        st.warning("Kite Session Expired")
        login_url = st.session_state.kite.login_url()
        st.link_button("🔑 Login to Zerodha", login_url)
        token = st.text_input("Paste Request Token from URL:")
        if st.button("Activate Terminal"):
            try:
                data = st.session_state.kite.generate_session(token, api_secret=API_SECRET)
                st.session_state.kite.set_access_token(data["access_token"])
                # Cache instruments for fast lookup
                instruments = st.session_state.kite.instruments("NSE")
                st.session_state.instrument_map = {i['tradingsymbol']: i['instrument_token'] for i in instruments}
                st.session_state.authenticated = True
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.success("✅ Connected to Zerodha")

    active_strat = st.selectbox("Strategy", list(st.session_state.strategy_stats.keys()))
    
    st.divider()
    st.subheader("💰 Capital")
    total_cap = st.number_input("Total Capital (₹)", value=50000.0)
    max_trades = st.slider("Max Positions", 1, 10, 5)
    
    st.divider()
    auto_trade = st.checkbox("Auto-Execute", value=False)
    scan_interval = st.slider("Scan Interval (sec)", 30, 300, 120)
    trailing_stop_pct = st.slider("Trailing Stop %", 0.5, 5.0, 1.5)

# --- INDICATOR CALCULATIONS ---
def calculate_indicators(data):
    try:
        if len(data) < 30: return None
        rsi = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
        ema_9 = ta.trend.EMAIndicator(data['Close'], window=9).ema_indicator()
        ema_21 = ta.trend.EMAIndicator(data['Close'], window=21).ema_indicator()
        vwap = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()
        
        return {
            'rsi': rsi.iloc[-1], 'ema_9': ema_9.iloc[-1], 'ema_21': ema_21.iloc[-1],
            'vwap': vwap.iloc[-1], 'close': float(data['Close'].iloc[-1]),
            'prev_close': float(data['Close'].iloc[-2])
        }
    except: return None

# --- MAIN UI ---
if st.session_state.authenticated:
    # Header Metrics
    current_pos_val = float((st.session_state.portfolio['LTP'] * st.session_state.portfolio['Qty']).sum()) if not st.session_state.portfolio.empty else 0.0
    live_pnl = float(st.session_state.portfolio['P&L'].sum()) if not st.session_state.portfolio.empty else 0.0
    
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Cash", f"₹{round(st.session_state.balance, 2)}")
    h2.metric("Portfolio", f"₹{round(current_pos_val, 2)}")
    h3.metric("P&L", f"₹{round(live_pnl, 2)}")
    h4.metric("Total", f"₹{round(st.session_state.balance + current_pos_val, 2)}")

    tab1, tab2 = st.tabs(["📺 Scanner", "⚔️ Positions"])

    # --- SCANNER ---
    with tab1:
        if st.button("🔍 Scan Market"):
            # Use top liquid symbols for the scanner
            master_list = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'SBIN', 'BHARTIARTL', 'ITC'] 
            results = []
            for sym in master_list:
                data = get_historical_data(sym)
                inds = calculate_indicators(data)
                if inds and inds['rsi'] < 40: # Example logic
                    results.append({"Symbol": sym, "Price": inds['close'], "RSI": round(inds['rsi'], 2)})
            st.session_state.scan_results = results
        
        if st.session_state.scan_results:
            st.table(st.session_state.scan_results)

    # --- POSITIONS (Updates every 2 seconds) ---
    with tab2:
        if not st.session_state.portfolio.empty:
            for idx, row in st.session_state.portfolio.iterrows():
                ltp = get_live_price(row['Symbol'])
                if ltp:
                    st.session_state.portfolio.at[idx, 'LTP'] = ltp
                    st.session_state.portfolio.at[idx, 'P&L'] = (ltp - row['Entry']) * row['Qty']
            st.dataframe(st.session_state.portfolio, use_container_width=True)
            time.sleep(2)
            st.rerun()
        else:
            st.info("No open positions.")
else:
    st.info("Please login via the sidebar to start the terminal.")
