import streamlit as st
import pandas as pd
import yfinance as yf
from nselib import capital_market
import ta
import time
import plotly.graph_objects as go
from datetime import datetime
import requests
from playsound3 import playsound

# --- 1. PRIMA CONFIG & BRANDING ---
st.set_page_config(page_title="PRIMA Command Center", layout="wide", page_icon="🔱")

# TELEGRAM CONFIG
# Replace with your actual IDs
TELEGRAM_TOKEN = "8563714849:AAGYRRGGQupxvvU16RHovQ5QSMOd6vkSS_o"
TELEGRAM_CHAT_ID = "1303832128" 

def send_telegram_msg(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        params = {"chat_id": TELEGRAM_CHAT_ID, "text": f"🔱 *PRIMA:* {message}", "parse_mode": "Markdown"}
        requests.get(url, params=params)
    except: pass

# --- 2. SESSION STATE INITIALIZATION ---
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=["Time", "Symbol", "Side", "Qty", "Entry", "LTP", "P&L"])
if 'trade_log' not in st.session_state:
    st.session_state.trade_log = pd.DataFrame(columns=["Time", "Symbol", "LTP", "Strategy", "Score"])

# --- 3. DATA & ANALYSIS ENGINES ---
@st.cache_data(ttl=3600)
def get_nse_watchlist():
    try:
        return capital_market.equity_list()['SYMBOL'].head(100).tolist()
    except:
        return ["RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN", "ICICIBANK"]

def plot_prima_chart(symbol):
    # Fetching 1 Year of data for clean historical visibility
    df = yf.download(f"{symbol}.NS", period="1y", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
    
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()

    fig = go.Figure()
    # High-Visibility Area Line
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', line=dict(color='#58a6ff', width=3),
                             name="Price", fill='tozeroy', fillcolor='rgba(88,166,255,0.1)'))
    # Trendlines
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], line=dict(color='orange', width=1.5, dash='dot'), name="50 DMA"))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], line=dict(color='red', width=1.5), name="200 DMA"))

    fig.update_layout(title=f"🏛️ {symbol} Historical Trend", template="plotly_dark", height=600,
                      hovermode="x unified", xaxis=dict(rangeslider=dict(visible=True)))
    return fig

# --- 4. THE CORE SCANNER ENGINE ---
def run_scanner(symbols):
    results = []
    progress_bar = st.progress(0)
    for i, sym in enumerate(symbols):
        progress_bar.progress((i + 1) / len(symbols))
        try:
            df = yf.download(f"{sym}.NS", period="5d", interval="15m", progress=False)
            if df.empty or len(df) < 20: continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]

            cp = float(df['Close'].iloc[-1])
            rsi = ta.momentum.RSIIndicator(df['Close']).rsi().iloc[-1]
            vwap = ((df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()).iloc[-1]
            vwap_dist = ((cp - vwap) / vwap) * 100
            vol_spike = df['Volume'].iloc[-1] / df['Volume'].rolling(10).mean().iloc[-1]

            strategy = "Scanning"
            score = 0
            if rsi < 30 and vwap_dist < -1.2 and vol_spike > 1.5:
                strategy = "⭐ GOLDEN BUY"; score = 10
            elif rsi > 70 and vwap_dist > 1.2 and vol_spike > 1.5:
                strategy = "⭐ GOLDEN SELL"; score = 10
            
            if score >= 10:
                send_telegram_msg(f"{strategy} | {sym} at ₹{cp}")
                try: playsound("alert.mp3") 
                except: pass

            results.append({"Symbol": sym, "LTP": cp, "RSI": round(rsi, 1), "VWAP_Dist": round(vwap_dist, 2), "Strategy": strategy, "Score": score})
        except: continue
    progress_bar.empty()
    return pd.DataFrame(results)

# --- 5. UI LAYOUT ---
st.title("🔱 PRIMA COMMAND CENTER v5.0")
tab1, tab2, tab3 = st.tabs(["📊 Market Scanner", "💼 Paper Portfolio", "📈 Historical Analysis"])

watchlist = get_nse_watchlist()

# SIDEBAR: QUICK EXECUTION
with st.sidebar:
    st.header("🚀 Execution")
    with st.form("trade_form"):
        trade_sym = st.selectbox("Symbol", watchlist)
        side = st.radio("Side", ["BUY", "SELL"])
        qty = st.number_input("Qty", 1, 10000, 100)
        if st.form_submit_button("EXECUTE"):
            price = yf.download(f"{trade_sym}.NS", period="1d", interval="1m", progress=False)['Close'].iloc[-1]
            new_trade = {"Time": datetime.now().strftime("%H:%M"), "Symbol": trade_sym, "Side": side, "Qty": qty, "Entry": round(price,2), "LTP": round(price,2), "P&L": 0.0}
            st.session_state.portfolio = pd.concat([st.session_state.portfolio, pd.DataFrame([new_trade])], ignore_index=True)
            st.toast(f"Trade Entered: {trade_sym}")

with tab1:
    scan_area = st.empty()

with tab2:
    st.subheader("Live Paper P&L")
    if not st.session_state.portfolio.empty:
        # Update Portfolio LTP
        for idx, row in st.session_state.portfolio.iterrows():
            curr_price = yf.download(f"{row['Symbol']}.NS", period="1d", interval="1m", progress=False)['Close'].iloc[-1]
            st.session_state.portfolio.at[idx, 'LTP'] = round(curr_price, 2)
            pnl = (curr_price - row['Entry']) * row['Qty'] if row['Side'] == "BUY" else (row['Entry'] - curr_price) * row['Qty']
            st.session_state.portfolio.at[idx, 'P&L'] = round(pnl, 2)
        st.dataframe(st.session_state.portfolio, width="stretch", hide_index=True)
    else:
        st.info("No active trades. Use the sidebar to enter a position.")

with tab3:
    analysis_sym = st.selectbox("Symbol to Analyze", watchlist)
    if analysis_sym:
        st.plotly_chart(plot_prima_chart(analysis_sym), use_container_width=True)

# --- 6. AUTO-REFRESH LOOP ---
while True:
    data = run_scanner(watchlist)
    if not data.empty:
        with scan_area.container():
            st.dataframe(data.sort_values("Score", ascending=False).style.background_gradient(subset=['Score'], cmap='RdYlGn'), 
                         width="stretch", hide_index=True)
    time.sleep(60)
