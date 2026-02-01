import streamlit as st
import pandas as pd
import yfinance as yf
from nselib import capital_market
import ta
import time
import plotly.graph_objects as go
from datetime import datetime
from playsound3 import playsound

# --- PRIMA THEME & CONFIG ---
st.set_page_config(page_title="PRIMA Quant Terminal", layout="wide", page_icon="📈")

st.markdown("""
    <style>
    .main { background-color: #0b0e14; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 10px; border-radius: 5px; }
    .prima-header { color: #58a6ff; font-family: 'Courier New', monospace; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- INITIALIZATION ---
if 'trade_log' not in st.session_state:
    st.session_state.trade_log = pd.DataFrame(columns=["Time", "Symbol", "LTP", "Strategy", "RSI"])

# --- CORE FUNCTIONS ---
@st.cache_data(ttl=3600) # Cache symbol list for 1 hour
def get_symbols():
    try:
        df_all = capital_market.equity_list()
        return df_all['SYMBOL'].head(100).tolist()
    except:
        return ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'SBIN', 'ICICIBANK']

def trigger_alert():
    try: playsound("alert.mp3")
    except: pass

def get_golden_signals(symbols):
    results = []
    alert_triggered = False
    
    for symbol in symbols:
        try:
            ticker = f"{symbol}.NS"
            df = yf.download(ticker, period="5d", interval="15m", progress=False)
            if df.empty or len(df) < 20: continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]

            # Indicators
            df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
            df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
            
            cp = float(df['Close'].iloc[-1])
            current_rsi = float(df['RSI'].iloc[-1])
            current_vwap = float(df['VWAP'].iloc[-1])
            
            avg_vol = df['Volume'].rolling(window=10).mean().iloc[-1]
            vol_spike = (df['Volume'].iloc[-1] / avg_vol) if avg_vol > 0 else 0
            vwap_dist = ((cp - current_vwap) / current_vwap) * 100
            
            # --- PRIMA GOLDEN LOGIC ---
            strategy = "Scanning"
            priority = 3
            
            if current_rsi < 30 and vwap_dist < -1.0 and vol_spike > 1.5:
                strategy = "⭐ GOLDEN BUY"; priority = 1
            elif current_rsi > 70 and vwap_dist > 1.0 and vol_spike > 1.5:
                strategy = "⭐ GOLDEN SELL"; priority = 1
            elif current_rsi < 30 and cp < current_vwap:
                strategy = "Standard BUY"; priority = 2
            elif current_rsi > 70 and cp > current_vwap:
                strategy = "Standard SELL"; priority = 2

            res = {
                "Symbol": symbol, "LTP": round(cp, 2), "RSI": round(current_rsi, 1),
                "VWAP_Dist%": round(vwap_dist, 2), "Vol_Spike": round(vol_spike, 2),
                "Strategy": strategy, "priority": priority
            }
            results.append(res)
            
            if priority == 1:
                # Log to session state
                if st.session_state.trade_log.empty or st.session_state.trade_log.iloc[-1]['Symbol'] != symbol:
                    log_entry = pd.DataFrame([{"Time": datetime.now().strftime("%H:%M:%S"), 
                                               "Symbol": symbol, "LTP": cp, "Strategy": strategy, "RSI": current_rsi}])
                    st.session_state.trade_log = pd.concat([st.session_state.trade_log, log_entry], ignore_index=True)
                    alert_triggered = True
        except: continue
        
    if alert_triggered: trigger_alert()
    return pd.DataFrame(results)

def plot_stock(symbol):
    df = yf.download(f"{symbol}.NS", period="1d", interval="1m", progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
    fig.update_layout(title=f"{symbol} Intraday", template="plotly_dark", height=400)
    return fig

# --- UI EXECUTION ---
st.markdown("<h1 class='prima-header'>PRIMA QUANT TERMINAL v3.0</h1>", unsafe_allow_html=True)

symbols = get_symbols()
tab1, tab2, tab3 = st.tabs(["⚡ Live Screener", "📜 Trade Log", "📊 Analysis"])

with tab1:
    live_placeholder = st.empty()

with tab2:
    st.dataframe(st.session_state.trade_log.tail(20), width="stretch", hide_index=True)

with tab3:
    selected_stock = st.selectbox("Select Symbol for Chart", symbols)
    if selected_stock:
        st.plotly_chart(plot_stock(selected_stock), use_container_width=True)

# --- LOOP ---
while True:
    report = get_golden_signals(symbols)
    if not report.empty:
        # Sorting: Golden Trades First
        report = report.sort_values(by="priority").drop(columns=['priority'])
        
        with live_placeholder.container():
            st.write(f"Last Scan: {datetime.now().strftime('%H:%M:%S')} | Symbols: {len(symbols)}")
            st.dataframe(
                report.style.apply(lambda x: ['background-color: #1e3a20' if 'BUY' in str(v) else 'background-color: #3a1e1e' if 'SELL' in str(v) else '' for v in x], axis=1),
                width="stretch", hide_index=True
            )
    time.sleep(30) # 30 seconds to respect API limits for 100 symbols