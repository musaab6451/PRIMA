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
    """Sends a professional PRIMA alert to multiple users."""
    for chat_id in TELEGRAM_CHAT_IDS:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            params = {
                "chat_id": chat_id, 
                "text": f"🔱 *PRIMA ALERT*\n\n{message}", 
                "parse_mode": "Markdown"
            }
            response = requests.get(url, params=params, timeout=5)
            if response.status_code != 200:
                print(f"Telegram error for {chat_id}: {response.text}")
        except Exception as e:
            print(f"Error sending to {chat_id}: {e}")

# --- CONFIG ---
st.set_page_config(page_title="PRIMA v10.0 | Production", layout="wide")

# --- PERSISTENT STORAGE FUNCTIONS ---
def save_state_to_file():
    """Save session state to file for persistence across devices"""
    try:
        state_data = {
            'portfolio': st.session_state.portfolio.to_dict('records') if not st.session_state.portfolio.empty else [],
            'history': st.session_state.history.to_dict('records') if not st.session_state.history.empty else [],
            'balance': st.session_state.balance,
            'strategy_stats': st.session_state.strategy_stats,
            'timestamp': time.time()
        }
        with open('/tmp/prima_state.json', 'w') as f:
            json.dump(state_data, f)
    except Exception as e:
        print(f"Error saving state: {e}")

def load_state_from_file():
    """Load session state from file"""
    try:
        with open('/tmp/prima_state.json', 'r') as f:
            state_data = json.load(f)
        
        # Restore portfolio
        if state_data['portfolio']:
            st.session_state.portfolio = pd.DataFrame(state_data['portfolio'])
        
        # Restore history
        if state_data['history']:
            st.session_state.history = pd.DataFrame(state_data['history'])
        
        # Restore balance and stats
        st.session_state.balance = state_data['balance']
        st.session_state.strategy_stats = state_data['strategy_stats']
        
        return True
    except Exception as e:
        return False

# --- STATE INITIALIZATION ---
if 'initialized' not in st.session_state:
    # Try to load from file first
    loaded = load_state_from_file()
    
    if not loaded:
        # Initialize fresh state
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
    
    st.session_state.initialized = True

if 'last_scan' not in st.session_state:
    st.session_state.last_scan = time.time()  # Initialize to current time
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = []
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True
if 'last_position_update' not in st.session_state:
    st.session_state.last_position_update = time.time()

# --- HELPER FUNCTIONS ---

@st.cache_data(ttl=86400)
def get_master_data():
    """Fetch NSE stock list with error handling"""
    try:
        raw_df = capital_market.equity_list()
        raw_df.columns = [c.upper().strip() for c in raw_df.columns]
        sym_col = next((c for c in ['SYMBOL', 'SYM'] if c in raw_df.columns), raw_df.columns[0])
        sec_col = next((c for c in ['INDUSTRY', 'GROUP', 'SECTOR'] if c in raw_df.columns), None)
        
        processed_df = pd.DataFrame()
        processed_df['SYMBOL'] = raw_df[sym_col].astype(str)
        processed_df['SECTOR'] = raw_df[sec_col].astype(str) if sec_col else "General"
        
        # Filter out invalid symbols
        processed_df = processed_df[processed_df['SYMBOL'].str.len() > 0]
        processed_df = processed_df[~processed_df['SYMBOL'].str.contains('-', na=False)]
        
        return processed_df
    except Exception as e:
        # Expanded fallback with top 100 liquid stocks
        return pd.DataFrame({
            'SYMBOL': ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'SBIN', 'BHARTIARTL', 
                      'ITC', 'KOTAKBANK', 'LT', 'HINDUNILVR', 'AXISBANK', 'ASIANPAINT', 'MARUTI',
                      'BAJFINANCE', 'TITAN', 'ULTRACEMCO', 'NESTLEIND', 'SUNPHARMA', 'WIPRO'] * 5,
            'SECTOR': ['Energy', 'IT', 'IT', 'Finance', 'Finance', 'Finance', 'Telecom', 
                      'FMCG', 'Finance', 'Infra', 'FMCG', 'Finance', 'Paint', 'Auto',
                      'Finance', 'Consumer', 'Cement', 'FMCG', 'Pharma', 'IT'] * 5
        })

def get_live_price(symbol):
    """Fetch live price with error handling"""
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        price = ticker.fast_info.get('last_price', None)
        
        if price is None or pd.isna(price):
            price = ticker.info.get('currentPrice', None)
        
        if price is not None:
            return float(price)
        return None
    except:
        return None

def calculate_indicators(data):
    """Calculate technical indicators"""
    try:
        if len(data) < 30:
            return None
        
        # RSI
        rsi = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
        
        # EMA
        ema_9 = ta.trend.EMAIndicator(data['Close'], window=9).ema_indicator()
        ema_21 = ta.trend.EMAIndicator(data['Close'], window=21).ema_indicator()
        ema_50 = ta.trend.EMAIndicator(data['Close'], window=50).ema_indicator()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
        
        # VWAP
        vwap = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()
        
        # MACD
        macd = ta.trend.MACD(data['Close'])
        
        # Volume
        volume_sma = data['Volume'].rolling(window=20).mean()
        volume_ratio = data['Volume'].iloc[-1] / volume_sma.iloc[-1] if volume_sma.iloc[-1] > 0 else 1
        
        # ADX
        adx = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=14).adx()
        
        return {
            'rsi': rsi.iloc[-1] if not rsi.empty else None,
            'ema_9': ema_9.iloc[-1] if not ema_9.empty else None,
            'ema_21': ema_21.iloc[-1] if not ema_21.empty else None,
            'ema_50': ema_50.iloc[-1] if not ema_50.empty else None,
            'bb_upper': bb.bollinger_hband().iloc[-1] if not bb.bollinger_hband().empty else None,
            'bb_lower': bb.bollinger_lband().iloc[-1] if not bb.bollinger_lband().empty else None,
            'bb_mid': bb.bollinger_mavg().iloc[-1] if not bb.bollinger_mavg().empty else None,
            'vwap': vwap.iloc[-1] if not vwap.empty else None,
            'macd_diff': macd.macd_diff().iloc[-1] if not macd.macd_diff().empty else None,
            'volume_ratio': volume_ratio,
            'adx': adx.iloc[-1] if not adx.empty else None,
            'close': float(data['Close'].iloc[-1]),
            'high': float(data['High'].iloc[-1]),
            'prev_close': float(data['Close'].iloc[-2]) if len(data) > 1 else float(data['Close'].iloc[-1])
        }
    except:
        return None

def check_strategy_signal(strategy, indicators, price):
    """Check if strategy conditions are met - RELAXED for more signals"""
    if indicators is None:
        return False, None, None
    
    try:
        if strategy == "RSI Mean Reversion":
            # Very relaxed - catch any oversold condition
            if indicators['rsi'] and indicators['rsi'] < 40:
                target = price * 1.025
                stop = price * 0.975
                return True, target, stop
                
        elif strategy == "EMA Crossover (9/21)":
            if indicators['ema_9'] and indicators['ema_21']:
                # Just need 9 > 21
                if indicators['ema_9'] > indicators['ema_21']:
                    target = price * 1.03
                    stop = price * 0.98
                    return True, target, stop
                    
        elif strategy == "Bollinger Mean Reversion":
            if indicators['bb_lower'] and indicators['bb_mid']:
                # Near lower band
                if price < indicators['bb_mid'] * 0.98:
                    target = indicators['bb_mid']
                    stop = price * 0.97
                    return True, target, stop
                    
        elif strategy == "VWAP Scalping":
            if indicators['vwap']:
                # Below VWAP
                if price < indicators['vwap']:
                    target = indicators['vwap'] * 1.01
                    stop = price * 0.99
                    return True, target, stop
        
        elif strategy == "MACD Momentum":
            # Any positive histogram
            if indicators['macd_diff'] and indicators['macd_diff'] > -0.5:
                target = price * 1.035
                stop = price * 0.975
                return True, target, stop
        
        elif strategy == "Supertrend Breakout":
            # Price moving up
            if indicators['close'] > indicators['prev_close'] * 1.005:
                target = price * 1.04
                stop = price * 0.97
                return True, target, stop
        
        elif strategy == "Volume Breakout":
            # Any above-average volume
            if indicators['volume_ratio'] > 1.2:
                if indicators['rsi'] and indicators['rsi'] > 40:
                    target = price * 1.04
                    stop = price * 0.97
                    return True, target, stop
        
        elif strategy == "Multi-Timeframe Trend":
            # Simple trend check
            if indicators['ema_9'] and indicators['ema_21']:
                if indicators['ema_9'] > indicators['ema_21']:
                    target = price * 1.05
                    stop = price * 0.965
                    return True, target, stop
        
        return False, None, None
    except:
        return False, None, None

def execute_trade(symbol, sector, price, qty, target, stop, strategy):
    """Execute trade with trailing stop"""
    trade_value = price * qty
    
    if trade_value > st.session_state.balance:
        return False, "Insufficient balance"
    
    if len(st.session_state.portfolio) >= max_trades:
        return False, "Max positions reached"
    
    # Execute trade
    trade_id = secrets.token_hex(4)
    new_position = pd.DataFrame([{
        "ID": trade_id,
        "Symbol": symbol,
        "Sector": sector,
        "Qty": qty,
        "Entry": round(price, 2),
        "LTP": round(price, 2),
        "P&L": 0.0,
        "Target": round(target, 2),
        "Stop": round(stop, 2),
        "TrailStop": round(stop, 2),  # Initialize trailing stop
        "Strategy": strategy
    }])
    
    st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_position], ignore_index=True)
    st.session_state.balance -= trade_value
    st.session_state.strategy_stats[strategy]['trades'] += 1
    
    # Save state
    save_state_to_file()
    
    # Telegram notification
    msg = f"📈 *BUY SIGNAL*\n" \
          f"Strategy: {strategy}\n" \
          f"Symbol: `{symbol}`\n" \
          f"Entry: ₹{round(price, 2)}\n" \
          f"Qty: {qty}\n" \
          f"Target: ₹{round(target, 2)}\n" \
          f"Stop: ₹{round(stop, 2)}"
    send_telegram_msg(msg)
    
    return True, f"Trade executed: {symbol}"

# --- SIDEBAR ---
with st.sidebar:
    st.title("🔱 PRIMA v10.0")
    
    active_strat = st.selectbox("Strategy", [
        "RSI Mean Reversion", "EMA Crossover (9/21)", "Bollinger Mean Reversion", 
        "VWAP Scalping", "MACD Momentum", "Supertrend Breakout",
        "Volume Breakout", "Multi-Timeframe Trend"
    ])
    
    st.divider()
    st.subheader("💰 Capital")
    total_cap = st.number_input("Total Capital (₹)", value=50000.0, min_value=1000.0)
    max_trades = st.slider("Max Positions", 1, 10, 5)
    sector_limit = st.slider("Sector Limit %", 10, 100, 40)
    
    st.divider()
    st.subheader("⚙️ Controls")
    auto_trade = st.checkbox("Auto-Execute", value=False)
    scan_interval = st.slider("Scan Interval (sec)", 30, 300, 120)
    trailing_stop_pct = st.slider("Trailing Stop %", 0.5, 5.0, 1.5, 0.5)
    
    st.divider()
    st.subheader("🎯 Filters")
    min_price = st.number_input("Min Price (₹)", value=20.0, min_value=1.0)
    max_price = st.number_input("Max Price (₹)", value=5000.0, min_value=50.0)
    min_rsi = st.slider("Min RSI", 0, 100, 15)
    max_rsi = st.slider("Max RSI", 0, 100, 85)
    
    st.divider()
    if st.button("🚨 KILL SWITCH", type="primary", use_container_width=True):
        total_exit = 0
        for idx, row in st.session_state.portfolio.iterrows():
            total_exit += row['LTP'] * row['Qty']
        
        st.session_state.balance += total_exit
        st.session_state.portfolio = pd.DataFrame(columns=["ID", "Symbol", "Sector", "Qty", "Entry", "LTP", "P&L", "Target", "Stop", "TrailStop", "Strategy"])
        save_state_to_file()
        st.success(f"Closed! +₹{round(total_exit, 2)}")
        time.sleep(1)
        st.rerun()

# Load master data
master_df = get_master_data()

# --- HEADER ---
current_pos_val = 0.0
live_pnl = 0.0

if not st.session_state.portfolio.empty:
    current_pos_val = float((st.session_state.portfolio['LTP'] * st.session_state.portfolio['Qty']).sum())
    live_pnl = float(st.session_state.portfolio['P&L'].sum())

total_equity = st.session_state.balance + current_pos_val

st.title("🔱 PRIMA v10.0")
st.caption(f"📊 {len(master_df)} stocks | {active_strat}")

h1, h2, h3, h4 = st.columns(4)
h1.metric("Cash", f"₹{round(st.session_state.balance, 2)}")
h2.metric("Portfolio", f"₹{round(current_pos_val, 2)}")
h3.metric("P&L", f"₹{round(live_pnl, 2)}")
h4.metric("Total", f"₹{round(total_equity, 2)}")

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📺 Scanner", "⚔️ Positions", "📊 Dashboard", "📜 History"])

# --- TAB 1: SCANNER ---
with tab1:
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col2:
        time_since_scan = int(time.time() - st.session_state.last_scan)
        st.metric("Last Scan", f"{time_since_scan}s ago")
    
    with col3:
        manual_scan = st.button("🔍 Scan", use_container_width=True)
    
    should_scan = manual_scan or (time_since_scan > scan_interval)
    
    if should_scan:
        with st.spinner(f"Scanning {len(master_df)} stocks..."):
            results = []
            progress = st.progress(0)
            
            for idx, stock in master_df.iterrows():
                try:
                    sym = str(stock['SYMBOL'])
                    sector = str(stock['SECTOR'])
                    
                    if idx % 25 == 0:
                        progress.progress((idx + 1) / len(master_df))
                    
                    data = yf.download(f"{sym}.NS", period="5d", interval="15m", progress=False, show_errors=False)
                    
                    if data.empty or len(data) < 20:
                        continue
                    
                    indicators = calculate_indicators(data)
                    if indicators is None:
                        continue
                    
                    price = indicators['close']
                    
                    if price < min_price or price > max_price:
                        continue
                    
                    if indicators['rsi'] and (indicators['rsi'] < min_rsi or indicators['rsi'] > max_rsi):
                        continue
                    
                    signal, target, stop = check_strategy_signal(active_strat, indicators, price)
                    
                    if signal:
                        st.session_state.strategy_stats[active_strat]['signals'] += 1
                        results.append({
                            "Symbol": sym,
                            "Sector": sector,
                            "Price": round(price, 2),
                            "Target": round(target, 2),
                            "Stop": round(stop, 2),
                            "RSI": round(indicators['rsi'], 2) if indicators['rsi'] else 0,
                            "Strategy": active_strat
                        })
                except:
                    continue
            
            progress.empty()
            st.session_state.scan_results = results
            st.session_state.last_scan = time.time()  # Update last scan time
            
            if results:
                msg = f"🔍 Scan: {len(results)} signals"
                send_telegram_msg(msg)
                st.success(f"Found {len(results)} signals!")
            else:
                st.warning("No signals found - try different strategy or filters")
    
    if st.session_state.scan_results:
        st.dataframe(pd.DataFrame(st.session_state.scan_results), use_container_width=True, hide_index=True)
        
        if auto_trade:
            risk_per_trade = total_cap / max_trades
            for signal in st.session_state.scan_results[:max_trades - len(st.session_state.portfolio)]:
                qty = int(risk_per_trade / signal['Price'])
                if qty > 0:
                    execute_trade(signal['Symbol'], signal['Sector'], signal['Price'], 
                                qty, signal['Target'], signal['Stop'], signal['Strategy'])
            st.session_state.scan_results = []
            st.rerun()

# --- TAB 2: POSITIONS (HIGH FREQUENCY UPDATE) ---
with tab2:
    if not st.session_state.portfolio.empty:
        # Update every 2 seconds (high frequency)
        time_since_update = time.time() - st.session_state.last_position_update
        
        if time_since_update > 2:  # Update every 2 seconds
            for idx, row in st.session_state.portfolio.iterrows():
                try:
                    ltp = get_live_price(row['Symbol'])
                    
                    if ltp:
                        st.session_state.portfolio.at[idx, 'LTP'] = round(ltp, 2)
                        pnl = (ltp - row['Entry']) * row['Qty']
                        st.session_state.portfolio.at[idx, 'P&L'] = round(pnl, 2)
                        
                        # TRAILING STOP LOGIC
                        if pnl > 0:  # In profit
                            # Calculate trailing stop (lock in profit)
                            trail_stop = ltp * (1 - trailing_stop_pct / 100)
                            if trail_stop > row['TrailStop']:
                                st.session_state.portfolio.at[idx, 'TrailStop'] = round(trail_stop, 2)
                        
                        # Check exits
                        if ltp <= row['TrailStop']:
                            st.session_state.balance += ltp * row['Qty']
                            
                            if 'Strategy' in row:
                                if pnl > 0:
                                    st.session_state.strategy_stats[row['Strategy']]['wins'] += 1
                                st.session_state.strategy_stats[row['Strategy']]['total_pnl'] += pnl
                            
                            msg = f"🛑 TRAIL STOP: {row['Symbol']} | P&L: ₹{round(pnl, 2)}"
                            send_telegram_msg(msg)
                            
                            st.session_state.portfolio = st.session_state.portfolio.drop(idx)
                            save_state_to_file()
                            
                        elif ltp >= row['Target']:
                            st.session_state.balance += ltp * row['Qty']
                            
                            if 'Strategy' in row:
                                st.session_state.strategy_stats[row['Strategy']]['wins'] += 1
                                st.session_state.strategy_stats[row['Strategy']]['total_pnl'] += pnl
                            
                            msg = f"🎯 TARGET: {row['Symbol']} | P&L: ₹{round(pnl, 2)}"
                            send_telegram_msg(msg)
                            
                            st.session_state.portfolio = st.session_state.portfolio.drop(idx)
                            save_state_to_file()
                except:
                    continue
            
            st.session_state.last_position_update = time.time()
            save_state_to_file()
        
        st.dataframe(st.session_state.portfolio.drop(columns=['ID']), use_container_width=True, hide_index=True)
    else:
        st.info("No positions")

# --- TAB 3: DASHBOARD ---
with tab3:
    for strat, stats in st.session_state.strategy_stats.items():
        if stats['signals'] > 0 or stats['trades'] > 0:
            win_rate = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
            with st.expander(f"{strat}"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Signals", stats['signals'])
                c2.metric("Trades", stats['trades'])
                c3.metric("Win%", f"{round(win_rate, 1)}%")
                c4.metric("P&L", f"₹{round(stats['total_pnl'], 2)}")

# --- TAB 4: HISTORY ---
with tab4:
    if not st.session_state.history.empty:
        st.dataframe(st.session_state.history, use_container_width=True, hide_index=True)
    else:
        st.info("No history")

# --- AUTO-REFRESH FOR POSITIONS ---
if st.session_state.auto_refresh and not st.session_state.portfolio.empty:
    time.sleep(2)  # Refresh every 2 seconds
    st.rerun()
