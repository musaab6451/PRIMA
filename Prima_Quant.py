import streamlit as st
import pandas as pd
from kiteconnect import KiteConnect
import ta
import time
from datetime import datetime, timedelta
import secrets
import requests
import json

# --- KITE CONFIG ---
API_KEY = "pcngqkv3k0i0i35o" 
API_SECRET = "4m8oueyj8m4e44qaym3elkla6rfptn27"

# --- TELEGRAM CONFIG ---
TELEGRAM_TOKEN = "8563714849:AAGYRRGGQupxvvU16RHovQ5QSMOd6vkSS_o"
TELEGRAM_CHAT_IDS = ["1303832128", "1287509530"]

def send_telegram_msg(message):
    for chat_id in TELEGRAM_CHAT_IDS:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            params = {"chat_id": chat_id, "text": f"🔱 *PRIMA v11*\n\n{message}", "parse_mode": "Markdown"}
            requests.get(url, params=params, timeout=5)
        except: pass

# --- CONFIG ---
st.set_page_config(page_title="PRIMA v11.0 | Professional Terminal", layout="wide")

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
        with open('/tmp/prima_v11_state.json', 'w') as f:
            json.dump(state, f)
    except: pass

def load_state():
    try:
        with open('/tmp/prima_v11_state.json', 'r') as f:
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

# --- STATE INITIALIZATION ---
if 'initialized' not in st.session_state:
    loaded = load_state()
    if not loaded:
        st.session_state.portfolio = pd.DataFrame(columns=["ID", "Symbol", "Qty", "Entry", "LTP", "P&L", "Target", "Stop", "TrailStop", "Strategy"])
        st.session_state.history = pd.DataFrame(columns=["Time", "Symbol", "Entry", "Exit", "P&L", "Strategy"])
        st.session_state.balance = 50000.0
        st.session_state.strategy_stats = {s: {"signals": 0, "trades": 0, "wins": 0, "total_pnl": 0.0, "active": True} for s in 
            ["RSI Mean Reversion", "EMA Crossover (9/21)", "Bollinger Mean Reversion", "VWAP Scalping", 
             "MACD Momentum", "Supertrend Breakout", "Volume Breakout", "Multi-Timeframe Trend"]}
    st.session_state.initialized = True

if 'scan_results' not in st.session_state: st.session_state.scan_results = []
if 'scanning' not in st.session_state: st.session_state.scanning = False
if 'scan_progress' not in st.session_state: st.session_state.scan_progress = {"current": 0, "total": 0, "found": 0, "symbol": ""}
if 'last_position_update' not in st.session_state: st.session_state.last_position_update = time.time()

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
        rsi = ta.momentum.RSIIndicator(c, window=14).rsi()
        ema9 = ta.trend.EMAIndicator(c, window=9).ema_indicator()
        ema21 = ta.trend.EMAIndicator(c, window=21).ema_indicator()
        bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
        vwap = (v * (h + l + c) / 3).cumsum() / v.cumsum()
        macd = ta.trend.MACD(c)
        vol_ratio = float(v.iloc[-1] / v.rolling(20).mean().iloc[-1]) if v.rolling(20).mean().iloc[-1] > 0 else 1.0
        return {
            'rsi': float(rsi.iloc[-1]) if not rsi.empty else None,
            'ema_9': float(ema9.iloc[-1]) if not ema9.empty else None,
            'ema_21': float(ema21.iloc[-1]) if not ema21.empty else None,
            'bb_lower': float(bb.bollinger_lband().iloc[-1]) if not bb.bollinger_lband().empty else None,
            'bb_mid': float(bb.bollinger_mavg().iloc[-1]) if not bb.bollinger_mavg().empty else None,
            'vwap': float(vwap.iloc[-1]) if not vwap.empty else None,
            'macd_diff': float(macd.macd_diff().iloc[-1]) if not macd.macd_diff().empty else None,
            'volume_ratio': vol_ratio,
            'close': float(c.iloc[-1]),
            'prev_close': float(c.iloc[-2]) if len(c) > 1 else float(c.iloc[-1])
        }
    except: return None

# --- STRATEGY SIGNALS ---
def check_strategy_signal(strategy, ind, price):
    if not ind: return False, 0, 0
    try:
        if strategy == "RSI Mean Reversion" and ind['rsi'] and ind['rsi'] < 35:
            return True, price * 1.025, price * 0.975
        if strategy == "EMA Crossover (9/21)" and ind['ema_9'] and ind['ema_21'] and ind['ema_9'] > ind['ema_21']:
            return True, price * 1.03, price * 0.98
        if strategy == "Bollinger Mean Reversion" and ind['bb_lower'] and price < ind['bb_mid'] * 0.98:
            return True, ind['bb_mid'], price * 0.97
        if strategy == "VWAP Scalping" and ind['vwap'] and price < ind['vwap']:
            return True, ind['vwap'] * 1.01, price * 0.99
        if strategy == "MACD Momentum" and ind['macd_diff'] and ind['macd_diff'] > -0.3:
            return True, price * 1.035, price * 0.975
        if strategy == "Supertrend Breakout" and ind['close'] > ind['prev_close'] * 1.005:
            return True, price * 1.04, price * 0.97
        if strategy == "Volume Breakout" and ind['volume_ratio'] > 1.2 and ind['rsi'] and ind['rsi'] > 40:
            return True, price * 1.04, price * 0.97
        if strategy == "Multi-Timeframe Trend" and ind['ema_9'] and ind['ema_21'] and ind['ema_9'] > ind['ema_21']:
            return True, price * 1.05, price * 0.965
        return False, 0, 0
    except: return False, 0, 0

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
        "P&L": 0.0, "Target": round(target, 2), "Stop": round(stop, 2), "TrailStop": round(stop, 2), "Strategy": strategy
    }])
    st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_position], ignore_index=True)
    st.session_state.balance -= trade_value
    st.session_state.strategy_stats[strategy]['trades'] += 1
    save_state()
    send_telegram_msg(f"📈 BUY {symbol} @ ₹{round(price, 2)} | {strategy}")
    return True, f"Executed: {symbol}"

# --- MULTI-STRATEGY SCANNER ---
def scan_all_strategies(symbols, active_strategies, min_price, max_price, min_rsi, max_rsi):
    results = []
    total = len(symbols)
    for idx, symbol in enumerate(symbols):
        try:
            st.session_state.scan_progress = {"current": idx + 1, "total": total, "found": len(results), "symbol": symbol}
            data = get_historical_data(symbol)
            if data.empty or len(data) < 30: continue
            ind = calculate_indicators(data)
            if not ind: continue
            price = ind['close']
            if price < min_price or price > max_price: continue
            if ind['rsi'] and (ind['rsi'] < min_rsi or ind['rsi'] > max_rsi): continue
            for strategy in active_strategies:
                signal, target, stop = check_strategy_signal(strategy, ind, price)
                if signal:
                    st.session_state.strategy_stats[strategy]['signals'] += 1
                    results.append({
                        "Symbol": symbol, "Strategy": strategy, "Price": round(price, 2),
                        "Target": round(target, 2), "Stop": round(stop, 2),
                        "RSI": round(ind['rsi'], 2) if ind['rsi'] else 0,
                        "Volume": round(ind['volume_ratio'], 2)
                    })
                    break
        except: continue
    return results

# --- SIDEBAR ---
with st.sidebar:
    st.title("🔱 PRIMA v11.0")
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
                instruments = st.session_state.kite.instruments("NSE")
                st.session_state.instrument_map = {i['tradingsymbol']: i['instrument_token'] for i in instruments}
                st.session_state.authenticated = True
                save_state()
                st.success("✅ Connected!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.success("✅ Connected")
    
    st.divider()
    
    # --- MASTER BOT CONTROL ---
    st.subheader("🤖 Bot Control")
    
    if 'bot_active' not in st.session_state:
        st.session_state.bot_active = False
    
    # Big visual toggle
    bot_toggle = st.toggle("🤖 TRADING BOT", value=st.session_state.bot_active, help="Master switch for automated trading")
    
    if bot_toggle != st.session_state.bot_active:
        st.session_state.bot_active = bot_toggle
        save_state()
        if bot_toggle:
            send_telegram_msg("🟢 Trading bot ACTIVATED")
            st.success("🟢 Bot is now ACTIVE")
        else:
            send_telegram_msg("🔴 Trading bot DEACTIVATED")
            st.warning("🔴 Bot is now INACTIVE")
        time.sleep(1)
        st.rerun()
    
    # Visual status indicator
    if st.session_state.bot_active:
        st.success("🟢 **BOT STATUS: ACTIVE**")
        st.caption("Bot will auto-scan and execute trades")
    else:
        st.error("🔴 **BOT STATUS: INACTIVE**")
        st.caption("Bot is paused - manual mode only")
    
    st.divider()
    
    # --- TRADING SETTINGS ---
    st.subheader("💰 Capital")
    total_cap = st.number_input("Capital (₹)", value=50000.0)
    max_trades = st.slider("Max Positions", 1, 20, 5)
    
    st.divider()
    st.subheader("⚙️ Trading Mode")
    live_trading = st.checkbox("🔴 LIVE TRADING", value=False, help="Execute real orders on Zerodha", disabled=not st.session_state.bot_active)
    auto_trade = st.checkbox("Auto-Execute Signals", value=False, help="Automatically enter trades", disabled=not st.session_state.bot_active)
    trailing_stop_pct = st.slider("Trailing Stop %", 0.5, 5.0, 1.5)
    
    if not st.session_state.bot_active:
        st.caption("⚠️ Enable Trading Bot to use auto features")
    
    st.divider()
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
        st.session_state.portfolio = pd.DataFrame(columns=["ID", "Symbol", "Qty", "Entry", "LTP", "P&L", "Target", "Stop", "TrailStop", "Strategy"])
        save_state()
        st.success(f"Closed: +₹{round(total_exit, 2)}")
        time.sleep(1)
        st.rerun()

# --- MAIN UI ---
if st.session_state.authenticated:
    current_pos_val = float((st.session_state.portfolio['LTP'] * st.session_state.portfolio['Qty']).sum()) if not st.session_state.portfolio.empty else 0.0
    live_pnl = float(st.session_state.portfolio['P&L'].sum()) if not st.session_state.portfolio.empty else 0.0
    
    st.title("🔱 PRIMA Professional Terminal")
    
    # Bot status banner
    if st.session_state.get('bot_active', False):
        st.success("🟢 **TRADING BOT: ACTIVE** | Auto-scanning and executing trades")
    else:
        st.info("🔴 **TRADING BOT: INACTIVE** | Manual mode - scans require bot activation")
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Cash", f"₹{round(st.session_state.balance, 2)}")
    c2.metric("Portfolio", f"₹{round(current_pos_val, 2)}")
    c3.metric("P&L", f"₹{round(live_pnl, 2)}")
    c4.metric("Total", f"₹{round(st.session_state.balance + current_pos_val, 2)}")
    c5.metric("Positions", f"{len(st.session_state.portfolio)}/{max_trades}")
    
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["📡 Scanner", "⚔️ Positions", "📊 Analytics"])
    
    # --- SCANNER TAB ---
    with tab1:
        col_scan1, col_scan2 = st.columns([3, 1])
        
        with col_scan1:
            if st.button("🔍 SCAN ALL STRATEGIES", type="primary", disabled=not st.session_state.bot_active):
                st.session_state.scanning = True
        
        with col_scan2:
            auto_scan = st.checkbox("🔄 Auto-Scan", value=False, disabled=not st.session_state.bot_active, 
                                   help="Automatically scan every 60s when bot is active")
        
        if not st.session_state.bot_active:
            st.warning("⚠️ **Trading Bot is OFF** - Enable in sidebar to start scanning")
        
        st.divider()
        
        # LIVE PROGRESS BAR
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
            nifty_50 = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'SBIN', 'BHARTIARTL', 'ITC', 'KOTAKBANK', 'LT',
                       'HINDUNILVR', 'AXISBANK', 'BAJFINANCE', 'ASIANPAINT', 'MARUTI', 'TITAN', 'SUNPHARMA', 'ULTRACEMCO',
                       'NESTLEIND', 'WIPRO', 'HCLTECH', 'TATAMOTORS', 'ONGC', 'NTPC', 'POWERGRID', 'M&M', 'ADANIPORTS',
                       'BAJAJFINSV', 'DRREDDY', 'TECHM', 'COALINDIA', 'CIPLA', 'DIVISLAB', 'GRASIM', 'EICHERMOT', 'BRITANNIA',
                       'SHREECEM', 'INDUSINDBK', 'BPCL', 'JSWSTEEL', 'TATACONSUM', 'TATASTEEL', 'APOLLOHOSP', 'HINDALCO',
                       'ADANIENT', 'HEROMOTOCO', 'UPL', 'BAJAJ-AUTO', 'LTIM', 'SBILIFE']
            
            active_strategies = [s for s, stats in st.session_state.strategy_stats.items() if stats['active']]
            
            if not active_strategies:
                st.warning("⚠️ Enable strategies in sidebar!")
                st.session_state.scanning = False
            else:
                results = scan_all_strategies(nifty_50, active_strategies, min_price, max_price, min_rsi, max_rsi)
                st.session_state.scan_results = results
                st.session_state.scanning = False
                if results:
                    send_telegram_msg(f"Scan: {len(results)} signals")
                    st.success(f"✅ {len(results)} signals!")
                else:
                    st.info("No signals")
        
        # DISPLAY RESULTS
        if st.session_state.scan_results:
            st.subheader(f"🎯 {len(st.session_state.scan_results)} Signals")
            st.dataframe(pd.DataFrame(st.session_state.scan_results), column_config={
                "Price": st.column_config.NumberColumn("Price", format="₹%.2f"),
                "Target": st.column_config.NumberColumn("Target", format="₹%.2f"),
                "Stop": st.column_config.NumberColumn("Stop", format="₹%.2f"),
            }, hide_index=True)
            
            if auto_trade and st.session_state.bot_active:
                risk_per_trade = total_cap / max_trades
                for signal in st.session_state.scan_results[:max_trades - len(st.session_state.portfolio)]:
                    qty = int(risk_per_trade / signal['Price'])
                    if qty > 0:
                        execute_trade(signal['Symbol'], signal['Price'], qty, signal['Target'], signal['Stop'], signal['Strategy'], live_trading)
                st.session_state.scan_results = []
                save_state()
                time.sleep(1)
                st.rerun()
            elif auto_trade and not st.session_state.bot_active:
                st.warning("⚠️ Trading bot is OFF - enable it in sidebar to auto-execute")
    
    # --- POSITIONS TAB ---
    with tab2:
        if not st.session_state.portfolio.empty:
            if time.time() - st.session_state.last_position_update > 1:
                for idx, row in st.session_state.portfolio.iterrows():
                    try:
                        ltp = get_live_price(row['Symbol'])
                        if ltp:
                            st.session_state.portfolio.at[idx, 'LTP'] = round(ltp, 2)
                            pnl = (ltp - row['Entry']) * row['Qty']
                            st.session_state.portfolio.at[idx, 'P&L'] = round(pnl, 2)
                            if pnl > 0:
                                trail_stop = ltp * (1 - trailing_stop_pct / 100)
                                if trail_stop > row['TrailStop']:
                                    st.session_state.portfolio.at[idx, 'TrailStop'] = round(trail_stop, 2)
                            if ltp <= row['TrailStop'] or ltp >= row['Target']:
                                if live_trading:
                                    place_order(row['Symbol'], row['Qty'], "SELL")
                                st.session_state.balance += ltp * row['Qty']
                                if pnl > 0:
                                    st.session_state.strategy_stats[row['Strategy']]['wins'] += 1
                                st.session_state.strategy_stats[row['Strategy']]['total_pnl'] += pnl
                                new_history = pd.DataFrame([{"Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "Symbol": row['Symbol'], "Entry": row['Entry'], "Exit": ltp, "P&L": pnl, "Strategy": row['Strategy']}])
                                st.session_state.history = pd.concat([st.session_state.history, new_history], ignore_index=True)
                                send_telegram_msg(f"{'🎯 TARGET' if ltp >= row['Target'] else '🛑 STOP'}: {row['Symbol']} | ₹{round(pnl, 2)}")
                                st.session_state.portfolio = st.session_state.portfolio.drop(idx)
                                save_state()
                    except: continue
                st.session_state.last_position_update = time.time()
                save_state()
            
            st.dataframe(st.session_state.portfolio.drop(columns=['ID']), column_config={
                "Entry": st.column_config.NumberColumn("Entry", format="₹%.2f"),
                "LTP": st.column_config.NumberColumn("LTP", format="₹%.2f"),
                "P&L": st.column_config.NumberColumn("P&L", format="₹%.2f"),
                "TrailStop": st.column_config.NumberColumn("Trail", format="₹%.2f"),
            }, hide_index=True)
        else:
            st.info("📭 No positions")
    
    # --- ANALYTICS TAB ---
    with tab3:
        strategy_data = []
        for strat, stats in st.session_state.strategy_stats.items():
            if stats['trades'] > 0 or stats['signals'] > 0:
                win_rate = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
                strategy_data.append({"Strategy": strat, "Signals": stats['signals'], "Trades": stats['trades'],
                    "Wins": stats['wins'], "Win%": round(win_rate, 1), "P&L": round(stats['total_pnl'], 2)})
        if strategy_data:
            st.dataframe(pd.DataFrame(strategy_data), column_config={
                "P&L": st.column_config.NumberColumn("P&L", format="₹%.2f")}, hide_index=True)
        else:
            st.info("No data yet")
    
    # AUTO-REFRESH
    if not st.session_state.portfolio.empty:
        time.sleep(1)
        st.rerun()
else:
    st.info("🔐 Authenticate with Zerodha in sidebar")
