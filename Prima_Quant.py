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
st.set_page_config(page_title="PRIMA v9.4 | 2026 Fixed", layout="wide")

# --- STATE INITIALIZATION (Must be first) ---
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=["ID", "Symbol", "Sector", "Qty", "Entry", "LTP", "P&L", "Target", "Stop", "Strategy"])
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["Time", "Symbol", "Entry", "Exit", "P&L"])
if 'balance' not in st.session_state:
    st.session_state.balance = 50000.0  # Default balance
if 'last_scan' not in st.session_state:
    st.session_state.last_scan = 0
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = []
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True
if 'strategy_stats' not in st.session_state:
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

# --- SIDEBAR: STRATEGY & RISK ---
with st.sidebar:
    st.title("🔱 PRIMA Control")
    active_strat = st.selectbox("Trading Engine", 
                                ["RSI Mean Reversion", 
                                 "EMA Crossover (9/21)", 
                                 "Bollinger Mean Reversion", 
                                 "VWAP Scalping",
                                 "MACD Momentum",
                                 "Supertrend Breakout",
                                 "Volume Breakout",
                                 "Multi-Timeframe Trend"])
    
    st.divider()
    st.subheader("💰 Capital Control")
    total_cap = st.number_input("Total Capital (₹)", value=50000.0, min_value=1000.0)
    max_trades = st.slider("Max Open Positions", 1, 10, 5)
    sector_limit = st.slider("Max Sector Exposure %", 10, 100, 40)
    risk_per_trade = total_cap / max_trades
    
    st.metric("Risk Per Trade", f"₹{round(risk_per_trade, 2)}")
    
    st.divider()
    st.subheader("⚙️ Trading Controls")
    auto_trade = st.checkbox("Auto-Execute Signals", value=False)
    scan_interval = st.slider("Scan Interval (seconds)", 30, 300, 120)
    
    st.divider()
    st.subheader("📊 Strategy Performance")
    for strat, stats in st.session_state.strategy_stats.items():
        if stats['signals'] > 0 or stats['trades'] > 0:
            win_rate = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
            with st.expander(f"{strat}"):
                col1, col2 = st.columns(2)
                col1.metric("Signals", stats['signals'])
                col2.metric("Trades", stats['trades'])
                col1.metric("Win Rate", f"{round(win_rate, 1)}%")
                col2.metric("Total P&L", f"₹{round(stats['total_pnl'], 2)}")
    
    st.divider()
    if st.button("🚨 GLOBAL KILL SWITCH", type="primary", use_container_width=True):
        # Exit all positions at market price
        total_exit_value = 0
        total_pnl = 0
        positions_closed = []
        
        for idx, row in st.session_state.portfolio.iterrows():
            exit_value = row['LTP'] * row['Qty']
            total_exit_value += exit_value
            total_pnl += row['P&L']
            positions_closed.append(f"{row['Symbol']}: ₹{round(row['P&L'], 2)}")
            
            # Log to history
            new_history = pd.DataFrame([{
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Symbol": row['Symbol'],
                "Entry": row['Entry'],
                "Exit": row['LTP'],
                "P&L": row['P&L']
            }])
            st.session_state.history = pd.concat([st.session_state.history, new_history], ignore_index=True)
        
        # Send Telegram notification
        if positions_closed:
            msg = f"🚨 *KILL SWITCH ACTIVATED*\n\n" \
                  f"Total Positions Closed: {len(positions_closed)}\n" \
                  f"Total P&L: ₹{round(total_pnl, 2)}\n" \
                  f"Recovered: ₹{round(total_exit_value, 2)}\n\n" \
                  f"Positions:\n" + "\n".join(positions_closed[:5])
            if len(positions_closed) > 5:
                msg += f"\n...and {len(positions_closed)-5} more"
            send_telegram_msg(msg)
        
        st.session_state.balance += total_exit_value
        st.session_state.portfolio = pd.DataFrame(columns=["ID", "Symbol", "Sector", "Qty", "Entry", "LTP", "P&L", "Target", "Stop", "Strategy"])
        st.success(f"All positions closed! ₹{round(total_exit_value, 2)} returned to balance")
        time.sleep(2)
        st.rerun()
    
    st.divider()
    auto_refresh_toggle = st.checkbox("Auto-Refresh Positions", value=True)
    if auto_refresh_toggle != st.session_state.auto_refresh:
        st.session_state.auto_refresh = auto_refresh_toggle
    
    if st.session_state.auto_refresh:
        st.info("🔄 Auto-refresh: ON (5s)")

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
        return processed_df.head(100)
    except Exception as e:
        st.sidebar.warning(f"Using fallback stock list: {str(e)}")
        # Fallback to major stocks
        return pd.DataFrame({
            'SYMBOL': ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'SBIN', 'BHARTIARTL', 'ITC', 'KOTAKBANK', 'LT'],
            'SECTOR': ['Energy', 'IT', 'IT', 'Finance', 'Finance', 'Finance', 'Telecom', 'FMCG', 'Finance', 'Infra']
        })

def get_live_price(symbol):
    """Fetch live price with error handling"""
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        price = ticker.fast_info.get('last_price', None)
        
        if price is None or pd.isna(price):
            # Fallback to regular info
            price = ticker.info.get('currentPrice', None)
        
        if price is not None:
            return float(price)
        else:
            return None
    except Exception as e:
        return None

def calculate_indicators(data):
    """Calculate technical indicators with error handling"""
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
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        bb_mid = bb.bollinger_mavg()
        
        # VWAP (approximation)
        vwap = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()
        
        # MACD
        macd = ta.trend.MACD(data['Close'])
        macd_line = macd.macd()
        macd_signal = macd.macd_signal()
        macd_diff = macd.macd_diff()
        
        # Supertrend (using ATR)
        atr = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=10).average_true_range()
        hl_avg = (data['High'] + data['Low']) / 2
        multiplier = 3
        upperband = hl_avg + (multiplier * atr)
        lowerband = hl_avg - (multiplier * atr)
        supertrend = pd.Series(index=data.index, dtype=float)
        supertrend.iloc[0] = upperband.iloc[0]
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > supertrend.iloc[i-1]:
                supertrend.iloc[i] = lowerband.iloc[i]
            else:
                supertrend.iloc[i] = upperband.iloc[i]
        
        # Volume indicators
        volume_sma = data['Volume'].rolling(window=20).mean()
        volume_ratio = data['Volume'].iloc[-1] / volume_sma.iloc[-1] if volume_sma.iloc[-1] > 0 else 1
        
        # ADX (Trend Strength)
        adx = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=14).adx()
        
        return {
            'rsi': rsi.iloc[-1] if not rsi.empty else None,
            'ema_9': ema_9.iloc[-1] if not ema_9.empty else None,
            'ema_21': ema_21.iloc[-1] if not ema_21.empty else None,
            'ema_50': ema_50.iloc[-1] if not ema_50.empty else None,
            'bb_upper': bb_upper.iloc[-1] if not bb_upper.empty else None,
            'bb_lower': bb_lower.iloc[-1] if not bb_lower.empty else None,
            'bb_mid': bb_mid.iloc[-1] if not bb_mid.empty else None,
            'vwap': vwap.iloc[-1] if not vwap.empty else None,
            'macd': macd_line.iloc[-1] if not macd_line.empty else None,
            'macd_signal': macd_signal.iloc[-1] if not macd_signal.empty else None,
            'macd_diff': macd_diff.iloc[-1] if not macd_diff.empty else None,
            'supertrend': supertrend.iloc[-1] if not supertrend.empty else None,
            'volume_ratio': volume_ratio,
            'adx': adx.iloc[-1] if not adx.empty else None,
            'close': float(data['Close'].iloc[-1]),
            'prev_close': float(data['Close'].iloc[-2]) if len(data) > 1 else float(data['Close'].iloc[-1])
        }
    except Exception as e:
        return None

def check_strategy_signal(strategy, indicators, price):
    """Check if strategy conditions are met"""
    if indicators is None:
        return False, None, None
    
    try:
        if strategy == "RSI Mean Reversion":
            if indicators['rsi'] and indicators['rsi'] < 30:  # Oversold
                target = price * 1.02  # 2% target
                stop = price * 0.98    # 2% stop
                return True, target, stop
                
        elif strategy == "EMA Crossover (9/21)":
            if indicators['ema_9'] and indicators['ema_21']:
                if indicators['ema_9'] > indicators['ema_21']:  # Bullish crossover
                    target = price * 1.025  # 2.5% target
                    stop = price * 0.985    # 1.5% stop
                    return True, target, stop
                    
        elif strategy == "Bollinger Mean Reversion":
            if indicators['bb_lower'] and price < indicators['bb_lower']:  # Below lower band
                target = indicators['bb_mid'] if indicators['bb_mid'] else price * 1.02
                stop = price * 0.98
                return True, target, stop
                
        elif strategy == "VWAP Scalping":
            if indicators['vwap'] and price < indicators['vwap'] * 0.998:  # Slight discount to VWAP
                target = indicators['vwap']
                stop = price * 0.995
                return True, target, stop
        
        elif strategy == "MACD Momentum":
            # MACD crosses above signal line (bullish crossover)
            if indicators['macd'] and indicators['macd_signal'] and indicators['macd_diff']:
                if indicators['macd_diff'] > 0 and indicators['macd'] > 0:  # Bullish momentum
                    target = price * 1.03  # 3% target
                    stop = price * 0.98    # 2% stop
                    return True, target, stop
        
        elif strategy == "Supertrend Breakout":
            # Price breaks above Supertrend
            if indicators['supertrend'] and indicators['close'] > indicators['supertrend']:
                if indicators['prev_close'] <= indicators['supertrend']:  # Fresh breakout
                    target = price * 1.04  # 4% target
                    stop = indicators['supertrend']  # Supertrend as stop
                    return True, target, stop
        
        elif strategy == "Volume Breakout":
            # High volume + price breakout
            if indicators['volume_ratio'] > 2.0:  # 2x average volume
                if indicators['rsi'] and indicators['rsi'] > 50:  # Bullish momentum
                    target = price * 1.035  # 3.5% target
                    stop = price * 0.975    # 2.5% stop
                    return True, target, stop
        
        elif strategy == "Multi-Timeframe Trend":
            # All EMAs aligned + strong trend (ADX > 25)
            if indicators['ema_9'] and indicators['ema_21'] and indicators['ema_50']:
                if indicators['ema_9'] > indicators['ema_21'] > indicators['ema_50']:
                    if indicators['adx'] and indicators['adx'] > 25:  # Strong trend
                        target = price * 1.045  # 4.5% target
                        stop = price * 0.97     # 3% stop
                        return True, target, stop
        
        return False, None, None
    except:
        return False, None, None

def execute_trade(symbol, sector, price, qty, target, stop, strategy):
    """Execute a trade and add to portfolio"""
    trade_value = price * qty
    
    # Check if we have enough balance
    if trade_value > st.session_state.balance:
        return False, "Insufficient balance"
    
    # Check position limits
    if len(st.session_state.portfolio) >= max_trades:
        return False, "Max positions reached"
    
    # Check sector exposure
    sector_exposure = st.session_state.portfolio[st.session_state.portfolio['Sector'] == sector]['Qty'].sum()
    total_positions = len(st.session_state.portfolio)
    if total_positions > 0:
        sector_pct = (sector_exposure / total_positions) * 100
        if sector_pct >= sector_limit:
            return False, f"Sector limit reached ({sector_pct:.1f}%)"
    
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
        "Strategy": strategy
    }])
    
    st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_position], ignore_index=True)
    st.session_state.balance -= trade_value
    
    # Track strategy stats
    st.session_state.strategy_stats[strategy]['trades'] += 1
    
    # Send Telegram notification
    msg = f"📈 *BUY SIGNAL*\n" \
          f"Strategy: {strategy}\n" \
          f"Symbol: `{symbol}`\n" \
          f"Sector: {sector}\n" \
          f"Entry: ₹{round(price, 2)}\n" \
          f"Quantity: {qty}\n" \
          f"Target: ₹{round(target, 2)}\n" \
          f"Stop Loss: ₹{round(stop, 2)}\n" \
          f"Investment: ₹{round(trade_value, 2)}"
    send_telegram_msg(msg)
    
    return True, f"Trade executed: {symbol} @ ₹{round(price, 2)}"

# --- LOAD MASTER DATA ---
master_df = get_master_data()

# --- REAL-TIME HEADER ---
current_pos_val = 0.0
live_pnl = 0.0

if not st.session_state.portfolio.empty:
    current_pos_val = float((st.session_state.portfolio['LTP'] * st.session_state.portfolio['Qty']).sum())
    live_pnl = float(st.session_state.portfolio['P&L'].sum())

total_equity = st.session_state.balance + current_pos_val

st.title("🔱 PRIMA COMMAND v9.4")
h1, h2, h3, h4 = st.columns(4)
h1.metric("Cash Balance", f"₹{round(st.session_state.balance, 2)}")
h2.metric("Portfolio Value", f"₹{round(current_pos_val, 2)}")
h3.metric("Live P&L", f"₹{round(live_pnl, 2)}", delta=f"{round((live_pnl/total_cap)*100, 2) if total_cap > 0 else 0}%")
h4.metric("Total Equity", f"₹{round(total_equity, 2)}")

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📺 Market Scanner", "⚔️ Active Positions", "📊 Risk Dashboard", "📜 Trade History"])

# --- TAB 1: MARKET SCANNER ---
with tab1:
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader(f"Active Strategy: {active_strat}")
    
    with col2:
        time_since_scan = int(time.time() - st.session_state.last_scan)
        st.metric("Last Scan", f"{time_since_scan}s ago")
    
    with col3:
        manual_scan = st.button("🔍 Scan Now", use_container_width=True)
    
    # Scanner logic
    should_scan = manual_scan or (time.time() - st.session_state.last_scan > scan_interval)
    
    if should_scan:
        with st.spinner("Scanning market..."):
            results = []
            progress_bar = st.progress(0)
            
            for idx, stock in master_df.iterrows():
                try:
                    sym = str(stock['SYMBOL'])
                    sector = str(stock['SECTOR'])
                    
                    # Download recent data
                    data = yf.download(f"{sym}.NS", period="5d", interval="15m", progress=False)
                    
                    if data.empty or len(data) < 20:
                        continue
                    
                    # Calculate indicators
                    indicators = calculate_indicators(data)
                    
                    if indicators is None:
                        continue
                    
                    price = indicators['close']
                    
                    # Check strategy signal
                    signal, target, stop = check_strategy_signal(active_strat, indicators, price)
                    
                    if signal:
                        # Track signal count
                        st.session_state.strategy_stats[active_strat]['signals'] += 1
                        
                        results.append({
                            "Symbol": sym,
                            "Sector": sector,
                            "Price": round(price, 2),
                            "Target": round(target, 2) if target else 0,
                            "Stop": round(stop, 2) if stop else 0,
                            "RSI": round(indicators['rsi'], 2) if indicators['rsi'] else 0,
                            "Signal": "BUY",
                            "Strategy": active_strat
                        })
                    
                    progress_bar.progress((idx + 1) / len(master_df))
                    
                except Exception as e:
                    continue
            
            progress_bar.empty()
            st.session_state.scan_results = results
            st.session_state.last_scan = time.time()
            
            # Send Telegram notification if signals found
            if results:
                top_signals = results[:3]  # Top 3 signals
                signal_list = "\n".join([f"• {s['Symbol']} @ ₹{s['Price']} (Target: ₹{s['Target']})" for s in top_signals])
                msg = f"🔍 *SCAN COMPLETE*\n\n" \
                      f"Strategy: {active_strat}\n" \
                      f"Signals Found: {len(results)}\n\n" \
                      f"Top Picks:\n{signal_list}"
                if len(results) > 3:
                    msg += f"\n\n...and {len(results)-3} more signals"
                send_telegram_msg(msg)
            
            st.success(f"Scan complete! Found {len(results)} signals")
    
    # Display scan results
    if st.session_state.scan_results:
        st.dataframe(pd.DataFrame(st.session_state.scan_results), use_container_width=True, hide_index=True)
        
        # Auto-execute if enabled
        if auto_trade and st.session_state.scan_results:
            for signal in st.session_state.scan_results[:max_trades - len(st.session_state.portfolio)]:
                qty = int(risk_per_trade / signal['Price'])
                if qty > 0:
                    success, msg = execute_trade(
                        signal['Symbol'],
                        signal['Sector'],
                        signal['Price'],
                        qty,
                        signal['Target'],
                        signal['Stop'],
                        signal['Strategy']
                    )
                    if success:
                        st.toast(msg, icon="✅")
            st.session_state.scan_results = []  # Clear after execution
    else:
        st.info("No active signals. Click 'Scan Now' or wait for auto-scan.")

# --- TAB 2: ACTIVE POSITIONS ---
with tab2:
    if not st.session_state.portfolio.empty:
        # Update positions
        positions_updated = False
        for idx, row in st.session_state.portfolio.iterrows():
            try:
                ltp = get_live_price(row['Symbol'])
                
                if ltp is not None:
                    st.session_state.portfolio.at[idx, 'LTP'] = round(ltp, 2)
                    pnl = (ltp - row['Entry']) * row['Qty']
                    st.session_state.portfolio.at[idx, 'P&L'] = round(pnl, 2)
                    positions_updated = True
                    
                    # Check exit conditions
                    if ltp <= row['Stop']:
                        # Stop loss hit
                        st.session_state.balance += (ltp * row['Qty'])
                        
                        # Track strategy stats (loss)
                        if 'Strategy' in row and row['Strategy'] in st.session_state.strategy_stats:
                            st.session_state.strategy_stats[row['Strategy']]['total_pnl'] += pnl
                        
                        # Log to history
                        new_history = pd.DataFrame([{
                            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Symbol": row['Symbol'],
                            "Entry": row['Entry'],
                            "Exit": ltp,
                            "P&L": pnl
                        }])
                        st.session_state.history = pd.concat([st.session_state.history, new_history], ignore_index=True)
                        
                        # Send Telegram notification
                        msg = f"🛑 *STOP LOSS HIT*\n" \
                              f"Symbol: `{row['Symbol']}`\n" \
                              f"Entry: ₹{round(row['Entry'], 2)}\n" \
                              f"Exit: ₹{round(ltp, 2)}\n" \
                              f"P&L: ₹{round(pnl, 2)}\n" \
                              f"Loss: {round((pnl/row['Entry']/row['Qty'])*100, 2)}%"
                        send_telegram_msg(msg)
                        
                        st.session_state.portfolio = st.session_state.portfolio.drop(idx)
                        st.toast(f"🛑 Stop Loss: {row['Symbol']} @ ₹{round(ltp, 2)}", icon="🔴")
                        positions_updated = True
                        
                    elif ltp >= row['Target']:
                        # Target hit
                        st.session_state.balance += (ltp * row['Qty'])
                        
                        # Track strategy stats (win)
                        if 'Strategy' in row and row['Strategy'] in st.session_state.strategy_stats:
                            st.session_state.strategy_stats[row['Strategy']]['wins'] += 1
                            st.session_state.strategy_stats[row['Strategy']]['total_pnl'] += pnl
                        
                        # Log to history
                        new_history = pd.DataFrame([{
                            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Symbol": row['Symbol'],
                            "Entry": row['Entry'],
                            "Exit": ltp,
                            "P&L": pnl
                        }])
                        st.session_state.history = pd.concat([st.session_state.history, new_history], ignore_index=True)
                        
                        # Send Telegram notification
                        msg = f"🎯 *TARGET HIT*\n" \
                              f"Symbol: `{row['Symbol']}`\n" \
                              f"Entry: ₹{round(row['Entry'], 2)}\n" \
                              f"Exit: ₹{round(ltp, 2)}\n" \
                              f"P&L: ₹{round(pnl, 2)}\n" \
                              f"Profit: {round((pnl/row['Entry']/row['Qty'])*100, 2)}%"
                        send_telegram_msg(msg)
                        
                        st.session_state.portfolio = st.session_state.portfolio.drop(idx)
                        st.toast(f"🎯 Target Hit: {row['Symbol']} @ ₹{round(ltp, 2)}", icon="🟢")
                        positions_updated = True
                        
            except Exception as e:
                continue
        
        if positions_updated:
            st.rerun()
        
        # Display positions
        display_df = st.session_state.portfolio.drop(columns=['ID'])
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Individual kill buttons
        st.divider()
        cols = st.columns(min(5, len(st.session_state.portfolio)))
        for i, (idx, row) in enumerate(st.session_state.portfolio.iterrows()):
            with cols[i % 5]:
                if st.button(f"❌ Exit {row['Symbol']}", key=f"kill_{row['ID']}", use_container_width=True):
                    exit_value = row['LTP'] * row['Qty']
                    st.session_state.balance += exit_value
                    
                    # Log to history
                    new_history = pd.DataFrame([{
                        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Symbol": row['Symbol'],
                        "Entry": row['Entry'],
                        "Exit": row['LTP'],
                        "P&L": row['P&L']
                    }])
                    st.session_state.history = pd.concat([st.session_state.history, new_history], ignore_index=True)
                    
                    # Send Telegram notification
                    msg = f"🔄 *MANUAL EXIT*\n" \
                          f"Symbol: `{row['Symbol']}`\n" \
                          f"Entry: ₹{round(row['Entry'], 2)}\n" \
                          f"Exit: ₹{round(row['LTP'], 2)}\n" \
                          f"P&L: ₹{round(row['P&L'], 2)}\n" \
                          f"Return: {round((row['P&L']/(row['Entry']*row['Qty']))*100, 2)}%"
                    send_telegram_msg(msg)
                    
                    st.session_state.portfolio = st.session_state.portfolio.drop(idx)
                    st.rerun()
    else:
        st.info("No active positions. Start scanning for signals!")

# --- TAB 3: RISK DASHBOARD ---
with tab3:
    if not st.session_state.portfolio.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sector Allocation")
            sector_counts = st.session_state.portfolio.groupby('Sector')['Qty'].sum()
            fig = px.pie(names=sector_counts.index, values=sector_counts.values, hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Position Metrics")
            st.metric("Open Positions", len(st.session_state.portfolio))
            st.metric("Largest Position", f"₹{round(st.session_state.portfolio['LTP'].max() * st.session_state.portfolio['Qty'].max(), 2)}")
            st.metric("Avg P&L per Position", f"₹{round(st.session_state.portfolio['P&L'].mean(), 2)}")
            
            winning = len(st.session_state.portfolio[st.session_state.portfolio['P&L'] > 0])
            total = len(st.session_state.portfolio)
            win_rate = (winning / total * 100) if total > 0 else 0
            st.metric("Win Rate", f"{round(win_rate, 1)}%")
    else:
        st.info("No positions to analyze")

# --- TAB 4: TRADE HISTORY ---
with tab4:
    if not st.session_state.history.empty:
        st.dataframe(st.session_state.history, use_container_width=True, hide_index=True)
        
        total_trades = len(st.session_state.history)
        total_pnl = st.session_state.history['P&L'].sum()
        winning_trades = len(st.session_state.history[st.session_state.history['P&L'] > 0])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Trades", total_trades)
        col2.metric("Total P&L", f"₹{round(total_pnl, 2)}")
        col3.metric("Win Rate", f"{round((winning_trades/total_trades)*100, 1) if total_trades > 0 else 0}%")
    else:
        st.info("No trade history yet")

# --- AUTO-REFRESH ---
if st.session_state.auto_refresh and not st.session_state.portfolio.empty:
    time.sleep(5)
    st.rerun()
