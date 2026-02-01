import streamlit as st
import pandas as pd
import yfinance as yf
from nselib import capital_market
import ta
import time
import plotly.graph_objects as go
from datetime import datetime
from playsound3 import playsound
import requests

# --- CONFIG & PRIMA BRANDING ---
st.set_page_config(page_title="PRIMA Command Center", layout="wide", page_icon="🔱")

# --- USER SETTINGS (TELEGRAM) ---
# To get these: Message @BotFather on Telegram to create a bot and get a TOKEN
TELEGRAM_TOKEN = "8563714849:AAGYRRGGQupxvvU16RHovQ5QSMOd6vkSS_o" 
TELEGRAM_CHAT_ID = "1303832128"

def send_telegram_msg(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage?chat_id={TELEGRAM_CHAT_ID}&text={message}"
        requests.get(url)
    except: pass

# --- INITIALIZATION ---
if 'trade_log' not in st.session_state:
    st.session_state.trade_log = pd.DataFrame(columns=["Time", "Symbol", "LTP", "Strategy", "RSI", "P&L"])
if 'paper_portfolio' not in st.session_state:
    st.session_state.paper_portfolio = {} # {Symbol: Entry_Price}

# --- ENHANCED CHARTING ---
def plot_advanced_chart(symbol):
    df = yf.download(f"{symbol}.NS", period="5d", interval="15m", progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
    
    # Add Technical Overlays for the Chart Tab
    df['EMA20'] = ta.trend.ema_indicator(df['Close'], window=20)
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='orange', width=1), name="EMA 20"))
    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='cyan', width=1, dash='dash'), name="VWAP"))
    
    fig.update_layout(title=f"PRIMA Technical View: {symbol}", template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
    return fig

# --- THE ENGINE ---
def get_symbols():
    try: return capital_market.equity_list()['SYMBOL'].head(100).tolist()
    except: return ['RELIANCE', 'TCS', 'INFY']

def run_prima_engine(symbols):
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, symbol in enumerate(symbols):
        # Update Progress Bar
        progress = (i + 1) / len(symbols)
        progress_bar.progress(progress)
        status_text.text(f"PRIMA Scanning: {symbol} ({i+1}/{len(symbols)})")
        
        try:
            ticker = f"{symbol}.NS"
            df = yf.download(ticker, period="5d", interval="15m", progress=False)
            if df.empty: continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]

            cp = float(df['Close'].iloc[-1])
            rsi = ta.momentum.RSIIndicator(df['Close']).rsi().iloc[-1]
            vwap = ((df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()).iloc[-1]
            vwap_dist = ((cp - vwap) / vwap) * 100
            vol_spike = df['Volume'].iloc[-1] / df['Volume'].rolling(10).mean().iloc[-1]

            strategy = "Wait"
            priority = 3
            
            # GOLDEN LOGIC
            if rsi < 30 and vwap_dist < -1.2 and vol_spike > 1.5:
                strategy = "⭐ GOLDEN BUY"; priority = 1
            elif rsi > 70 and vwap_dist > 1.2 and vol_spike > 1.5:
                strategy = "⭐ GOLDEN SELL"; priority = 1

            if priority == 1:
                # 1. Sound Alert
                try: playsound("alert.mp3")
                except: pass
                # 2. Telegram Alert
                send_telegram_msg(f"PRIMA SIGNAL: {strategy} in {symbol} at {cp}")
                # 3. Log it
                new_row = {"Time": datetime.now().strftime("%H:%M:%S"), "Symbol": symbol, "LTP": cp, "Strategy": strategy, "RSI": round(rsi,1), "P&L": "0.0"}
                st.session_state.trade_log = pd.concat([st.session_state.trade_log, pd.DataFrame([new_row])], ignore_index=True)

            results.append({"Symbol": symbol, "LTP": cp, "RSI": round(rsi,1), "VWAP_Dist%": round(vwap_dist,2), "Vol_Spike": round(vol_spike,2), "Strategy": strategy, "prio": priority})
        except: continue
    
    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(results)

# --- UI TABS ---
st.title("🔱 PRIMA COMMAND CENTER v4.0")
tab1, tab2, tab3, tab4 = st.tabs(["⚡ Live Scan", "📜 Signal Log", "📊 Technical Charts", "💼 Paper Portfolio"])

symbols_list = get_symbols()

with tab1:
    live_area = st.empty()

with tab2:
    st.subheader("Golden Signal History")
    st.dataframe(st.session_state.trade_log, width="stretch", hide_index=True)

with tab3:
    col_a, col_b = st.columns([1, 4])
    with col_a:
        chart_sym = st.selectbox("Analyze Symbol", symbols_list)
    with col_b:
        if chart_sym:
            st.plotly_chart(plot_advanced_chart(chart_sym), use_container_width=True)

with tab4:
    st.info("Coming Soon: Real-time P&L tracking for paper trades.")
    # Here we will add the 'Buy' button logic next!

# --- MAIN LOOP ---
while True:
    report = run_prima_engine(symbols_list)
    if not report.empty:
        report = report.sort_values("prio").drop(columns="prio")
        with live_area.container():
            st.write(f"Refreshed at: {datetime.now().strftime('%H:%M:%S')}")
            st.dataframe(report.style.background_gradient(subset=['RSI'], cmap='coolwarm'), width="stretch", hide_index=True)
    time.sleep(60) # Scanning 100 stocks takes time; 1 min interval is safer for yfinance
