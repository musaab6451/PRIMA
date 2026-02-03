import streamlit as st
import pandas as pd
from kiteconnect import KiteConnect, KiteTicker
import ta
import time
from datetime import datetime, timedelta
import secrets
import requests
import json
import threading
from queue import Queue

# --- KITE CONFIG ---
API_KEY = "2kcgsxf407fpuvif" 
API_SECRET = "l5h1w3wbshj70pxaayr7i48qd3plamfc"

# --- TELEGRAM CONFIG ---
TELEGRAM_TOKEN = "8563714849:AAGYRRGGQupxvvU16RHovQ5QSMOd6vkSS_o"
TELEGRAM_CHAT_IDS = ["1303832128", "1287509530"]

def send_telegram_msg(message):
    for chat_id in TELEGRAM_CHAT_IDS:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            params = {"chat_id": chat_id, "text": f"🔱 *PRIMA v12*\n\n{message}", "parse_mode": "Markdown"}
            requests.get(url, params=params, timeout=5)
        except: pass

# --- CONFIG ---
st.set_page_config(page_title="PRIMA v12.0 | Real-Time Terminal", layout="wide")

# --- PERSISTENT STORAGE ---
def save_state():
    try:
        state = {
            'portfolio': st.session_state.portfolio.to_dict('records') if not st.session_state.portfolio.empty else [],
            'history': st.session_state.history.to_dict('records') if not st.session_state.history.empty else [],
            'balance': float(st.session_state.balance),
            'strategy_stats': st.session_state.strategy_stats,
            'access_token': st.session_state.get('access_token', ''),
            'bot_active': st.session_state.get('bot_active', False)
        }
        with open('/tmp/prima_v12_state.json', 'w') as f:
            json.dump(state, f)
    except: pass

def load_state():
    try:
        with open('/tmp/prima_v12_state.json', 'r') as f:
            state = json.load(f)
        if state['portfolio']:
            st.session_state.portfolio = pd.DataFrame(state['portfolio'])
        if state['history']:
            st.session_state.history = pd.DataFrame(state['history'])
        st.session_state.balance = float(state['balance'])
        st.session_state.strategy_stats = state['strategy_stats']
        st.session_state.bot_active = state.get('bot_active', False)
        if state.get('access_token'):
            st.session_state.access_token = state['access_token']
            st.session_state.kite.set_access_token(state['access_token'])
            st.session_state.authenticated = True
        return True
    except:
        return False

# --- KITE INITIALIZATION ---
if 'kite' not in st.session_state:
    st.session_state.kite = KiteConnect(api_key=API_KEY)
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'instrument_map' not in st.session_state:
    st.session_state.instrument_map = {}
if 'token_to_symbol' not in st.session_state:
    st.session_state.token_to_symbol = {}
if 'live_data' not in st.session_state:
    st.session_state.live_data = {}
if 'websocket_active' not in st.session_state:
    st.session_state.websocket_active = False

# --- STATE INITIALIZATION ---
if 'initialized' not in st.session_state:
    loaded = load_state()
    if not loaded:
        st.session_state.portfolio = pd.DataFrame(columns=[
            "ID", "Symbol", "Qty", "Entry", "LTP", "P&L", "%Change", "Target", "Stop", "TrailStop", "Strategy"
        ])
        st.session_state.history = pd.DataFrame(columns=[
            "Time", "Symbol", "Entry", "Exit", "P&L", "%Return", "Strategy", "Duration"
        ])
        st.session_state.balance = 50000.0
        st.session_state.strategy_stats = {s: {"signals": 0, "trades": 0, "wins": 0, "total_pnl": 0.0, "active": True} for s in 
            ["RSI Mean Reversion", "EMA Crossover (9/21)", "Bollinger Mean Reversion", "VWAP Scalping", 
             "MACD Momentum", "Supertrend Breakout", "Volume Breakout", "Multi-Timeframe Trend"]}
    st.session_state.initialized = True

if 'scan_results' not in st.session_state: st.session_state.scan_results = []
if 'scanning' not in st.session_state: st.session_state.scanning = False
if 'scan_progress' not in st.session_state: st.session_state.scan_progress = {"current": 0, "total": 0, "found": 0, "symbol": ""}
if 'last_position_update' not in st.session_state: st.session_state.last_position_update = time.time()
if 'market_snapshot' not in st.session_state: st.session_state.market_snapshot = pd.DataFrame()

# --- WEBSOCKET HANDLER ---
class WebSocketHandler:
    def __init__(self, access_token, api_key):
        self.kws = KiteTicker(api_key, access_token)
        self.kws.on_ticks = self.on_ticks
        self.kws.on_connect = self.on_connect
        self.kws.on_close = self.on_close
        self.kws.on_error = self.on_error
        self.running = False
        
    def on_ticks(self, ws, ticks):
        """Handle incoming tick data"""
        for tick in ticks:
            token = tick['instrument_token']
            if token in st.session_state.token_to_symbol:
                symbol = st.session_state.token_to_symbol[token]
                st.session_state.live_data[symbol] = {
                    'ltp': tick.get('last_price', 0),
                    'change': tick.get('change', 0),
                    'volume': tick.get('volume', 0),
                    'oi': tick.get('oi', 0),
                    'timestamp': datetime.now()
                }
    
    def on_connect(self, ws, response):
        """Subscribe to instruments on connection"""
        tokens = list(st.session_state.instrument_map.values())
        if tokens:
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_FULL, tokens)
    
    def on_close(self, ws, code, reason):
        st.session_state.websocket_active = False
    
    def on_error(self, ws, code, reason):
        pass
    
    def start(self):
        if not self.running:
            self.running = True
            threading.Thread(target=self.kws.connect, daemon=True).start()
    
    def stop(self):
        if self.running:
            self.kws.close()
            self.running = False

# --- KITE HELPER FUNCTIONS ---
def get_historical_data(symbol, interval="15minute", days=5):
    try:
        token = st.session_state.instrument_map.get(symbol)
        if not token: return pd.DataFrame()
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        records = st.session_state.kite.historical_data(token, from_date, to_date, interval)
        df = pd.DataFrame(records)
        if df.empty: return pd.DataFrame()
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'oi']
        return df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
    except: return pd.DataFrame()

def get_live_price(symbol):
    """Get live price from WebSocket data or fallback to API"""
    if symbol in st.session_state.live_data:
        return st.session_state.live_data[symbol]['ltp']
    try:
        resp = st.session_state.kite.ltp([f"NSE:{symbol}"])
        return float(resp[f"NSE:{symbol}"]["last_price"])
    except: return None

def place_order(symbol, qty, transaction_type="BUY"):
    try:
        order_id = st.session_state.kite.place_order(
            variety=st.session_state.kite.VARIETY_REGULAR,
            exchange=st.session_state.kite.EXCHANGE_NSE,
            tradingsymbol=symbol,
            transaction_type=transaction_type,
            quantity=qty,
            product=st.session_state.kite.PRODUCT_MIS,
            order_type=st.session_state.kite.ORDER_TYPE_MARKET
        )
        return order_id
    except: return None

# --- TECHNICAL INDICATORS ---
def calculate_indicators(data):
    try:
        if len(data) < 30: return None
        c, h, l, v = data['Close'], data['High'], data['Low'], data['Volume']
        
        # RSI
        rsi = ta.momentum.RSIIndicator(c, window=14).rsi()
        
        # EMA
        ema9 = ta.trend.EMAIndicator(c, window=9).ema_indicator()
        ema21 = ta.trend.EMAIndicator(c, window=21).ema_indicator()
        ema50 = ta.trend.EMAIndicator(c, window=50).ema_indicator()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
        
        # VWAP
        vwap = (v * (h + l + c) / 3).cumsum() / v.cumsum()
        
        # MACD
        macd = ta.trend.MACD(c)
        
        # ATR for volatility
        atr = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()
        
        # Volume
        vol_ratio = float(v.iloc[-1] / v.rolling(20).mean().iloc[-1]) if v.rolling(20).mean().iloc[-1] > 0 else 1.0
        
        return {
            'rsi': float(rsi.iloc[-1]) if not rsi.empty else None,
            'ema_9': float(ema9.iloc[-1]) if not ema9.empty else None,
            'ema_21': float(ema21.iloc[-1]) if not ema21.empty else None,
            'ema_50': float(ema50.iloc[-1]) if not ema50.empty else None,
            'bb_upper': float(bb.bollinger_hband().iloc[-1]) if not bb.bollinger_hband().empty else None,
            'bb_lower': float(bb.bollinger_lband().iloc[-1]) if not bb.bollinger_lband().empty else None,
            'bb_mid': float(bb.bollinger_mavg().iloc[-1]) if not bb.bollinger_mavg().empty else None,
            'vwap': float(vwap.iloc[-1]) if not vwap.empty else None,
            'macd_diff': float(macd.macd_diff().iloc[-1]) if not macd.macd_diff().empty else None,
            'atr': float(atr.iloc[-1]) if not atr.empty else None,
            'volume_ratio': vol_ratio,
            'close': float(c.iloc[-1]),
            'prev_close': float(c.iloc[-2]) if len(c) > 1 else float(c.iloc[-1])
        }
    except: return None

# --- STRATEGY SIGNALS ---
def check_all_strategies(ind, price):
    """Check all strategies and return best signal"""
    if not ind:
        return None, None, None, None
    
    signals = []
    
    # RSI Mean Reversion
    if ind['rsi'] and ind['rsi'] < 35:
        target = price * 1.025
        stop = price * 0.975
        signals.append(("RSI Mean Reversion", "BUY", target, stop, ind['rsi']))
    
    # EMA Crossover
    if ind['ema_9'] and ind['ema_21'] and ind['ema_9'] > ind['ema_21']:
        target = price * 1.03
        stop = price * 0.98
        signals.append(("EMA Crossover (9/21)", "BUY", target, stop, 0))
    
    # Bollinger Mean Reversion
    if ind['bb_lower'] and ind['bb_mid'] and price < ind['bb_mid'] * 0.98:
        target = ind['bb_mid']
        stop = price * 0.97
        signals.append(("Bollinger Mean Reversion", "BUY", target, stop, 0))
    
    # VWAP Scalping
    if ind['vwap'] and price < ind['vwap']:
        target = ind['vwap'] * 1.01
        stop = price * 0.99
        signals.append(("VWAP Scalping", "BUY", target, stop, 0))
    
    # MACD Momentum
    if ind['macd_diff'] and ind['macd_diff'] > -0.3:
        target = price * 1.035
        stop = price * 0.975
        signals.append(("MACD Momentum", "BUY", target, stop, 0))
    
    # Volume Breakout
    if ind['volume_ratio'] > 1.5 and ind['rsi'] and ind['rsi'] > 45:
        target = price * 1.04
        stop = price * 0.97
        signals.append(("Volume Breakout", "BUY", target, stop, 0))
    
    # Supertrend
    if ind['close'] > ind['prev_close'] * 1.01:
        target = price * 1.04
        stop = price * 0.97
        signals.append(("Supertrend Breakout", "BUY", target, stop, 0))
    
    # Multi-Timeframe Trend
    if ind['ema_9'] and ind['ema_21'] and ind['ema_50']:
        if ind['ema_9'] > ind['ema_21'] > ind['ema_50']:
            target = price * 1.05
            stop = price * 0.965
            signals.append(("Multi-Timeframe Trend", "BUY", target, stop, 0))
    
    # Return best signal (prioritize by strength)
    if signals:
        # Sort by risk-reward ratio
        best = max(signals, key=lambda x: (x[2] - price) / (price - x[3]))
        return best[0], best[1], best[2], best[3]
    
    return None, None, None, None

# --- TRADE EXECUTION ---
def execute_trade(symbol, price, qty, target, stop, strategy, live_trading=False):
    trade_value = price * qty
    if trade_value > st.session_state.balance: return False, "Insufficient balance"
    if len(st.session_state.portfolio) >= max_trades: return False, "Max positions reached"
    
    if live_trading:
        order_id = place_order(symbol, qty, "BUY")
        if not order_id: return False, "Order failed"
    
    trade_id = secrets.token_hex(4)
    new_position = pd.DataFrame([{
        "ID": trade_id, "Symbol": symbol, "Qty": qty, "Entry": round(price, 2), "LTP": round(price, 2),
        "P&L": 0.0, "%Change": 0.0, "Target": round(target, 2), "Stop": round(stop, 2), 
        "TrailStop": round(stop, 2), "Strategy": strategy
    }])
    
    st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_position], ignore_index=True)
    st.session_state.balance -= trade_value
    st.session_state.strategy_stats[strategy]['trades'] += 1
    save_state()
    
    send_telegram_msg(f"📈 BUY {symbol} @ ₹{round(price, 2)}\n{strategy}\nQty: {qty}\nTarget: ₹{round(target, 2)}")
    return True, f"Executed: {symbol}"

# --- REAL-TIME MARKET SCANNER ---
def scan_market_realtime(symbols, active_strategies, min_price, max_price, min_rsi, max_rsi):
    """Scan all symbols with real-time WebSocket data"""
    results = []
    total = len(symbols)
    
    for idx, symbol in enumerate(symbols):
        try:
            st.session_state.scan_progress = {
                "current": idx + 1, 
                "total": total, 
                "found": len(results), 
                "symbol": symbol
            }
            
            # Get live price from WebSocket
            if symbol in st.session_state.live_data:
                live_data = st.session_state.live_data[symbol]
                price = live_data['ltp']
                change_pct = live_data['change']
            else:
                price = get_live_price(symbol)
                if not price:
                    continue
                change_pct = 0
            
            # Apply filters
            if price < min_price or price > max_price:
                continue
            
            # Get historical data for indicators
            hist_data = get_historical_data(symbol, interval="15minute", days=3)
            
            if hist_data.empty or len(hist_data) < 30:
                continue
            
            # Calculate indicators
            indicators = calculate_indicators(hist_data)
            
            if not indicators:
                continue
            
            # RSI filter
            if indicators['rsi'] and (indicators['rsi'] < min_rsi or indicators['rsi'] > max_rsi):
                continue
            
            # Check all strategies
            strategy, action, target, stop = check_all_strategies(indicators, price)
            
            if strategy and strategy in active_strategies:
                st.session_state.strategy_stats[strategy]['signals'] += 1
                
                # Calculate % targets
                target_pct = ((target - price) / price) * 100 if target else 0
                stop_pct = ((price - stop) / price) * 100 if stop else 0
                risk_reward = target_pct / stop_pct if stop_pct > 0 else 0
                
                results.append({
                    "Symbol": symbol,
                    "LTP": round(price, 2),
                    "Change%": round(change_pct, 2),
                    "Strategy": strategy,
                    "Action": action,
                    "Target": round(target, 2),
                    "Target%": round(target_pct, 2),
                    "Stop": round(stop, 2),
                    "Stop%": round(stop_pct, 2),
                    "R:R": round(risk_reward, 2),
                    "RSI": round(indicators['rsi'], 2) if indicators['rsi'] else 0,
                    "Volume": round(indicators['volume_ratio'], 2)
                })
        
        except Exception as e:
            continue
    
    return results

# --- SIDEBAR ---
with st.sidebar:
    st.title("🔱 PRIMA v12.0")
    st.caption("Real-Time WebSocket Terminal")
    
    # --- KITE AUTHENTICATION ---
    if not st.session_state.authenticated:
        st.warning("⚠️ Not Connected")
        login_url = st.session_state.kite.login_url()
        st.link_button("🔑 Login to Zerodha", login_url)
        request_token = st.text_input("Request Token:", type="password")
        
        if st.button("🚀 Activate"):
            try:
                data = st.session_state.kite.generate_session(request_token, api_secret=API_SECRET)
                st.session_state.access_token = data["access_token"]
                st.session_state.kite.set_access_token(data["access_token"])
                
                # Load instruments
                instruments = st.session_state.kite.instruments("NSE")
                st.session_state.instrument_map = {i['tradingsymbol']: i['instrument_token'] for i in instruments}
                st.session_state.token_to_symbol = {i['instrument_token']: i['tradingsymbol'] for i in instruments}
                
                st.session_state.authenticated = True
                save_state()
                st.success("✅ Connected!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.success("✅ Connected to Zerodha")
        
        # WebSocket status
        if st.session_state.websocket_active:
            st.success("🟢 WebSocket: LIVE")
        else:
            st.warning("🟡 WebSocket: Disconnected")
            if st.button("🔌 Connect WebSocket"):
                try:
                    ws_handler = WebSocketHandler(st.session_state.access_token, API_KEY)
                    st.session_state.ws_handler = ws_handler
                    ws_handler.start()
                    st.session_state.websocket_active = True
                    st.success("WebSocket connected!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"WebSocket error: {e}")
    
    st.divider()
    
    # --- BOT CONTROL ---
    st.subheader("🤖 Bot Control")
    
    if 'bot_active' not in st.session_state:
        st.session_state.bot_active = False
    
    bot_toggle = st.toggle("🤖 TRADING BOT", value=st.session_state.bot_active)
    
    if bot_toggle != st.session_state.bot_active:
        st.session_state.bot_active = bot_toggle
        save_state()
        if bot_toggle:
            send_telegram_msg("🟢 Trading bot ACTIVATED")
            st.success("🟢 Bot ACTIVE")
        else:
            send_telegram_msg("🔴 Trading bot DEACTIVATED")
            st.warning("🔴 Bot INACTIVE")
        time.sleep(1)
        st.rerun()
    
    if st.session_state.bot_active:
        st.success("🟢 **BOT: ACTIVE**")
    else:
        st.error("🔴 **BOT: INACTIVE**")
    
    st.divider()
    
    # --- TRADING SETTINGS ---
    st.subheader("💰 Capital")
    total_cap = st.number_input("Capital (₹)", value=50000.0)
    max_trades = st.slider("Max Positions", 1, 20, 5)
    
    st.divider()
    st.subheader("⚙️ Trading Mode")
    live_trading = st.checkbox("🔴 LIVE TRADING", value=False, disabled=not st.session_state.bot_active)
    auto_trade = st.checkbox("Auto-Execute", value=False, disabled=not st.session_state.bot_active)
    trailing_stop_pct = st.slider("Trailing Stop %", 0.5, 5.0, 1.5)
    
    st.divider()
    st.subheader("🎯 Filters")
    min_price = st.number_input("Min Price (₹)", value=20.0)
    max_price = st.number_input("Max Price (₹)", value=5000.0)
    min_rsi = st.slider("Min RSI", 0, 100, 15)
    max_rsi = st.slider("Max RSI", 0, 100, 85)
    
    st.divider()
    st.subheader("📊 Strategies")
    for strategy in list(st.session_state.strategy_stats.keys()):
        st.session_state.strategy_stats[strategy]['active'] = st.checkbox(
            strategy[:20], value=st.session_state.strategy_stats[strategy]['active'], key=f"s_{strategy[:10]}")
    
    st.divider()
    
    if st.button("🚨 EXIT ALL", type="primary"):
        total_exit = sum(row['LTP'] * row['Qty'] for _, row in st.session_state.portfolio.iterrows())
        if live_trading:
            for _, row in st.session_state.portfolio.iterrows():
                place_order(row['Symbol'], row['Qty'], "SELL")
        st.session_state.balance += total_exit
        st.session_state.portfolio = pd.DataFrame(columns=[
            "ID", "Symbol", "Qty", "Entry", "LTP", "P&L", "%Change", "Target", "Stop", "TrailStop", "Strategy"
        ])
        save_state()
        st.success(f"Closed: +₹{round(total_exit, 2)}")
        time.sleep(1)
        st.rerun()

# --- MAIN UI ---
if st.session_state.authenticated:
    current_pos_val = float((st.session_state.portfolio['LTP'] * st.session_state.portfolio['Qty']).sum()) if not st.session_state.portfolio.empty else 0.0
    live_pnl = float(st.session_state.portfolio['P&L'].sum()) if not st.session_state.portfolio.empty else 0.0
    
    st.title("🔱 PRIMA Real-Time Terminal v12.0")
    
    # Bot status banner
    if st.session_state.get('bot_active', False):
        st.success("🟢 **TRADING BOT: ACTIVE** | WebSocket streaming live data")
    else:
        st.info("🔴 **TRADING BOT: INACTIVE** | Manual mode")
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Cash", f"₹{round(st.session_state.balance, 2)}")
    c2.metric("Portfolio", f"₹{round(current_pos_val, 2)}")
    c3.metric("P&L", f"₹{round(live_pnl, 2)}")
    c4.metric("Total", f"₹{round(st.session_state.balance + current_pos_val, 2)}")
    c5.metric("Positions", f"{len(st.session_state.portfolio)}/{max_trades}")
    
    st.divider()
    
    tab1, tab2, tab3, tab4 = st.tabs(["📡 Live Scanner", "⚔️ Positions", "📊 Analytics", "📜 History"])
    
    # --- TAB 1: LIVE SCANNER ---
    with tab1:
        col_scan1, col_scan2, col_scan3 = st.columns([2, 1, 1])
        
        with col_scan1:
            if st.button("🔍 SCAN MARKET (Real-Time)", type="primary", disabled=not st.session_state.bot_active):
                st.session_state.scanning = True
        
        with col_scan2:
            refresh_rate = st.selectbox("Refresh", ["1s", "2s", "5s"], index=1)
        
        with col_scan3:
            auto_scan = st.checkbox("🔄 Auto-Scan", disabled=not st.session_state.bot_active)
        
        if not st.session_state.bot_active:
            st.warning("⚠️ **Enable Trading Bot** to start scanning")
        
        st.divider()
        
        # LIVE PROGRESS
        if st.session_state.scanning or st.session_state.scan_progress['total'] > 0:
            prog = st.session_state.scan_progress
            col1, col2, col3 = st.columns(3)
            col1.metric("Scanned", f"{prog['current']}/{prog['total']}")
            col2.metric("Signals", prog['found'])
            col3.metric("Current", prog.get('symbol', '-'))
            if prog['total'] > 0:
                st.progress(prog['current'] / prog['total'])
        
        # RUN SCAN
        if st.session_state.scanning:
            # Top 100 liquid stocks
            nifty_100 = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'SBIN', 'BHARTIARTL', 'ITC', 
                        'KOTAKBANK', 'LT', 'HINDUNILVR', 'AXISBANK', 'BAJFINANCE', 'ASIANPAINT', 'MARUTI', 
                        'TITAN', 'SUNPHARMA', 'ULTRACEMCO', 'NESTLEIND', 'WIPRO', 'HCLTECH', 'TATAMOTORS',
                        'ONGC', 'NTPC', 'POWERGRID', 'M&M', 'ADANIPORTS', 'BAJAJFINSV', 'DRREDDY', 'TECHM',
                        'COALINDIA', 'CIPLA', 'DIVISLAB', 'GRASIM', 'EICHERMOT', 'BRITANNIA', 'SHREECEM',
                        'INDUSINDBK', 'BPCL', 'JSWSTEEL', 'TATACONSUM', 'TATASTEEL', 'APOLLOHOSP', 'HINDALCO',
                        'ADANIENT', 'HEROMOTOCO', 'UPL', 'BAJAJ-AUTO', 'LTIM', 'SBILIFE', 'DABUR', 'PIDILITIND',
                        'AMBUJACEM', 'DLF', 'GODREJCP', 'HAVELLS', 'VEDL', 'BERGEPAINT', 'BOSCHLTD', 'MUTHOOTFIN',
                        'GAIL', 'SIEMENS', 'ABB', 'LUPIN', 'BANKBARODA', 'PNB', 'INDIGO', 'ADANIGREEN', 
                        'BANDHANBNK', 'TATAPOWER', 'INDUSTOWER', 'CANBK', 'JUBLFOOD', 'MCDOWELL-N', 'BEL',
                        'CHOLAFIN', 'TORNTPHARM', 'TRENT', 'LICI', 'ZOMATO', 'NYKAA', 'PAYTM', 'IRCTC',
                        'IRFC', 'ZYDUSLIFE', 'CONCOR', 'HAL', 'SAIL', 'RECLTD', 'GNFC', 'NMDC']
            
            active_strategies = [s for s, stats in st.session_state.strategy_stats.items() if stats['active']]
            
            if not active_strategies:
                st.warning("⚠️ Enable strategies in sidebar!")
                st.session_state.scanning = False
            else:
                results = scan_market_realtime(nifty_100, active_strategies, min_price, max_price, min_rsi, max_rsi)
                st.session_state.scan_results = results
                st.session_state.scanning = False
                
                if results:
                    send_telegram_msg(f"📊 Live Scan: {len(results)} signals found")
                    st.success(f"✅ {len(results)} live signals!")
                else:
                    st.info("No signals - market conditions not met")
        
        # DISPLAY RESULTS TABLE
        if st.session_state.scan_results:
            st.subheader(f"🎯 {len(st.session_state.scan_results)} Live Signals")
            
            df_results = pd.DataFrame(st.session_state.scan_results)
            
            # Color coding for better visualization
            st.dataframe(
                df_results,
                column_config={
                    "LTP": st.column_config.NumberColumn("LTP", format="₹%.2f"),
                    "Change%": st.column_config.NumberColumn("Change%", format="%.2f%%"),
                    "Target": st.column_config.NumberColumn("Target", format="₹%.2f"),
                    "Target%": st.column_config.NumberColumn("Target%", format="%.2f%%"),
                    "Stop": st.column_config.NumberColumn("Stop Loss", format="₹%.2f"),
                    "Stop%": st.column_config.NumberColumn("Risk%", format="%.2f%%"),
                    "R:R": st.column_config.NumberColumn("Risk:Reward", format="%.2f"),
                    "RSI": st.column_config.NumberColumn("RSI", format="%.1f"),
                    "Volume": st.column_config.NumberColumn("Vol Ratio", format="%.2fx"),
                },
                hide_index=True,
                height=400
            )
            
            st.divider()
            
            # Auto-execute
            if auto_trade and st.session_state.bot_active:
                st.info("🤖 Auto-executing top signals...")
                risk_per_trade = total_cap / max_trades
                
                for signal in st.session_state.scan_results[:max_trades - len(st.session_state.portfolio)]:
                    qty = int(risk_per_trade / signal['LTP'])
                    if qty > 0:
                        success, msg = execute_trade(
                            signal['Symbol'], signal['LTP'], qty, 
                            signal['Target'], signal['Stop'], 
                            signal['Strategy'], live_trading
                        )
                        if success:
                            st.toast(f"✅ {msg}", icon="📈")
                
                st.session_state.scan_results = []
                save_state()
                time.sleep(1)
                st.rerun()
    
    # --- TAB 2: POSITIONS ---
    with tab2:
        if not st.session_state.portfolio.empty:
            # High-frequency updates
            if time.time() - st.session_state.last_position_update > 1:
                for idx, row in st.session_state.portfolio.iterrows():
                    try:
                        ltp = get_live_price(row['Symbol'])
                        
                        if ltp:
                            st.session_state.portfolio.at[idx, 'LTP'] = round(ltp, 2)
                            pnl = (ltp - row['Entry']) * row['Qty']
                            change_pct = ((ltp - row['Entry']) / row['Entry']) * 100
                            
                            st.session_state.portfolio.at[idx, 'P&L'] = round(pnl, 2)
                            st.session_state.portfolio.at[idx, '%Change'] = round(change_pct, 2)
                            
                            # Trailing stop
                            if pnl > 0:
                                trail_stop = ltp * (1 - trailing_stop_pct / 100)
                                if trail_stop > row['TrailStop']:
                                    st.session_state.portfolio.at[idx, 'TrailStop'] = round(trail_stop, 2)
                            
                            # Check exits
                            if ltp <= row['TrailStop'] or ltp >= row['Target']:
                                if live_trading:
                                    place_order(row['Symbol'], row['Qty'], "SELL")
                                
                                st.session_state.balance += ltp * row['Qty']
                                
                                if pnl > 0:
                                    st.session_state.strategy_stats[row['Strategy']]['wins'] += 1
                                st.session_state.strategy_stats[row['Strategy']]['total_pnl'] += pnl
                                
                                # Add to history
                                return_pct = (pnl / (row['Entry'] * row['Qty'])) * 100
                                new_history = pd.DataFrame([{
                                    "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "Symbol": row['Symbol'],
                                    "Entry": row['Entry'],
                                    "Exit": ltp,
                                    "P&L": pnl,
                                    "%Return": round(return_pct, 2),
                                    "Strategy": row['Strategy'],
                                    "Duration": "~1h"
                                }])
                                st.session_state.history = pd.concat([st.session_state.history, new_history], ignore_index=True)
                                
                                exit_type = "TARGET" if ltp >= row['Target'] else "TRAIL STOP"
                                send_telegram_msg(f"{'🎯' if exit_type == 'TARGET' else '🛑'} {exit_type}: {row['Symbol']} | ₹{round(pnl, 2)} ({round(return_pct, 2)}%)")
                                
                                st.session_state.portfolio = st.session_state.portfolio.drop(idx)
                                save_state()
                    except:
                        continue
                
                st.session_state.last_position_update = time.time()
                save_state()
            
            # Display positions
            st.dataframe(
                st.session_state.portfolio.drop(columns=['ID']),
                column_config={
                    "Entry": st.column_config.NumberColumn("Entry", format="₹%.2f"),
                    "LTP": st.column_config.NumberColumn("LTP", format="₹%.2f"),
                    "P&L": st.column_config.NumberColumn("P&L", format="₹%.2f"),
                    "%Change": st.column_config.NumberColumn("Change%", format="%.2f%%"),
                    "Target": st.column_config.NumberColumn("Target", format="₹%.2f"),
                    "TrailStop": st.column_config.NumberColumn("Trail", format="₹%.2f"),
                },
                hide_index=True,
                height=400
            )
        else:
            st.info("📭 No active positions")
    
    # --- TAB 3: ANALYTICS ---
    with tab3:
        strategy_data = []
        for strat, stats in st.session_state.strategy_stats.items():
            if stats['trades'] > 0 or stats['signals'] > 0:
                win_rate = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
                avg_pnl = stats['total_pnl'] / stats['trades'] if stats['trades'] > 0 else 0
                strategy_data.append({
                    "Strategy": strat,
                    "Signals": stats['signals'],
                    "Trades": stats['trades'],
                    "Wins": stats['wins'],
                    "Win%": round(win_rate, 1),
                    "Total P&L": round(stats['total_pnl'], 2),
                    "Avg P&L": round(avg_pnl, 2)
                })
        
        if strategy_data:
            st.dataframe(
                pd.DataFrame(strategy_data),
                column_config={
                    "Total P&L": st.column_config.NumberColumn("Total P&L", format="₹%.2f"),
                    "Avg P&L": st.column_config.NumberColumn("Avg P&L", format="₹%.2f"),
                },
                hide_index=True
            )
        else:
            st.info("No analytics data yet")
    
    # --- TAB 4: HISTORY ---
    with tab4:
        if not st.session_state.history.empty:
            st.dataframe(
                st.session_state.history,
                column_config={
                    "Entry": st.column_config.NumberColumn("Entry", format="₹%.2f"),
                    "Exit": st.column_config.NumberColumn("Exit", format="₹%.2f"),
                    "P&L": st.column_config.NumberColumn("P&L", format="₹%.2f"),
                    "%Return": st.column_config.NumberColumn("Return%", format="%.2f%%"),
                },
                hide_index=True
            )
            
            total_trades = len(st.session_state.history)
            total_pnl = st.session_state.history['P&L'].sum()
            winning = len(st.session_state.history[st.session_state.history['P&L'] > 0])
            avg_return = st.session_state.history['%Return'].mean()
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Trades", total_trades)
            c2.metric("Total P&L", f"₹{round(total_pnl, 2)}")
            c3.metric("Win Rate", f"{round((winning/total_trades)*100, 1)}%")
            c4.metric("Avg Return", f"{round(avg_return, 2)}%")
        else:
            st.info("📭 No trade history")
    
    # AUTO-REFRESH
    if not st.session_state.portfolio.empty:
        time.sleep(1)
        st.rerun()
else:
    st.info("🔐 Authenticate with Zerodha in sidebar")
