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
from collections import defaultdict

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
            params = {"chat_id": chat_id, "text": f"🔱 *PRIMA v13*\n\n{message}", "parse_mode": "Markdown"}
            requests.get(url, params=params, timeout=5)
        except: pass

# --- PROFESSIONAL UI CONFIG ---
st.set_page_config(
    page_title="PRIMA v13 | Professional Terminal", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "PRIMA v13.0 - Professional Algorithmic Trading Terminal"
    }
)

# Custom CSS for professional look
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #0e1117;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
    }
    
    /* Tables */
    .dataframe {
        font-family: 'Monaco', 'Menlo', monospace;
        font-size: 12px;
    }
    
    /* Headers */
    h1 {
        color: #00ff88;
        font-weight: 800;
        letter-spacing: 2px;
    }
    
    h2, h3 {
        color: #00ccff;
        font-weight: 600;
    }
    
    /* Status indicators */
    .status-live {
        color: #00ff88;
        font-weight: bold;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    /* Buttons */
    .stButton>button {
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,255,136,0.3);
    }
</style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
def init_session_state():
    if 'kite' not in st.session_state:
        st.session_state.kite = KiteConnect(api_key=API_KEY)
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'access_token' not in st.session_state:
        st.session_state.access_token = None
    if 'instrument_map' not in st.session_state:
        st.session_state.instrument_map = {}
    if 'token_to_symbol' not in st.session_state:
        st.session_state.token_to_symbol = {}
    if 'live_data' not in st.session_state:
        st.session_state.live_data = {}
    if 'market_status' not in st.session_state:
        st.session_state.market_status = "UNKNOWN"
    if 'zerodha_positions' not in st.session_state:
        st.session_state.zerodha_positions = pd.DataFrame()
    if 'zerodha_holdings' not in st.session_state:
        st.session_state.zerodha_holdings = pd.DataFrame()
    if 'zerodha_funds' not in st.session_state:
        st.session_state.zerodha_funds = {}
    if 'stock_universe' not in st.session_state:
        st.session_state.stock_universe = pd.DataFrame()
    if 'websocket_active' not in st.session_state:
        st.session_state.websocket_active = False
    if 'bot_active' not in st.session_state:
        st.session_state.bot_active = False
    if 'last_sync' not in st.session_state:
        st.session_state.last_sync = 0

init_session_state()

# --- ZERODHA ACCOUNT SYNC ---
def sync_zerodha_account():
    """Sync live Zerodha account data"""
    try:
        # Get positions
        positions = st.session_state.kite.positions()
        if positions and 'net' in positions:
            pos_data = []
            for p in positions['net']:
                if p['quantity'] != 0:
                    pos_data.append({
                        'Symbol': p['tradingsymbol'],
                        'Qty': p['quantity'],
                        'Avg Price': p['average_price'],
                        'LTP': p['last_price'],
                        'P&L': p['pnl'],
                        '%Change': round(((p['last_price'] - p['average_price']) / p['average_price'] * 100), 2),
                        'Product': p['product']
                    })
            st.session_state.zerodha_positions = pd.DataFrame(pos_data)
        
        # Get holdings
        holdings = st.session_state.kite.holdings()
        if holdings:
            hold_data = []
            for h in holdings:
                hold_data.append({
                    'Symbol': h['tradingsymbol'],
                    'Qty': h['quantity'],
                    'Avg Price': h['average_price'],
                    'LTP': h['last_price'],
                    'P&L': h['pnl'],
                    'Day Change%': round(h['day_change_percentage'], 2),
                    'Value': h['last_price'] * h['quantity']
                })
            st.session_state.zerodha_holdings = pd.DataFrame(hold_data)
        
        # Get funds
        funds = st.session_state.kite.margins()
        if funds and 'equity' in funds:
            st.session_state.zerodha_funds = {
                'available': funds['equity']['available']['live_balance'],
                'used': funds['equity']['utilised']['debits'],
                'total': funds['equity']['net']
            }
        
        st.session_state.last_sync = time.time()
        return True
    except Exception as e:
        return False

# --- MARKET STATUS ---
def get_market_status():
    """Get live market status from Zerodha"""
    try:
        # Check if it's a trading day and within market hours
        now = datetime.now()
        
        # Simple check: Monday-Friday, 9:15 AM - 3:30 PM IST
        if now.weekday() >= 5:  # Saturday or Sunday
            return "CLOSED"
        
        market_open = now.replace(hour=9, minute=15, second=0)
        market_close = now.replace(hour=15, minute=30, second=0)
        
        if market_open <= now <= market_close:
            return "OPEN"
        elif now < market_open:
            return "PRE-MARKET"
        else:
            return "CLOSED"
    except:
        return "UNKNOWN"

# --- WEBSOCKET HANDLER ---
class WebSocketHandler:
    def __init__(self, access_token, api_key):
        self.kws = KiteTicker(api_key, access_token)
        self.kws.on_ticks = self.on_ticks
        self.kws.on_connect = self.on_connect
        self.kws.on_close = self.on_close
        self.running = False
        
    def on_ticks(self, ws, ticks):
        for tick in ticks:
            token = tick['instrument_token']
            if token in st.session_state.token_to_symbol:
                symbol = st.session_state.token_to_symbol[token]
                st.session_state.live_data[symbol] = {
                    'ltp': tick.get('last_price', 0),
                    'change': tick.get('change', 0),
                    'volume': tick.get('volume_traded', 0),
                    'oi': tick.get('oi', 0),
                    'high': tick.get('ohlc', {}).get('high', 0),
                    'low': tick.get('ohlc', {}).get('low', 0),
                    'open': tick.get('ohlc', {}).get('open', 0),
                    'close': tick.get('ohlc', {}).get('close', 0)
                }
    
    def on_connect(self, ws, response):
        tokens = list(st.session_state.instrument_map.values())
        if tokens:
            # Subscribe in batches to avoid overload
            batch_size = 500
            for i in range(0, len(tokens), batch_size):
                batch = tokens[i:i+batch_size]
                ws.subscribe(batch)
                ws.set_mode(ws.MODE_FULL, batch)
    
    def on_close(self, ws, code, reason):
        st.session_state.websocket_active = False
    
    def start(self):
        if not self.running:
            self.running = True
            threading.Thread(target=self.kws.connect, daemon=True).start()
    
    def stop(self):
        if self.running:
            self.kws.close()
            self.running = False

# --- PATTERN DETECTION ---
def detect_patterns(data, symbol, price):
    """Detect chart patterns and breakout levels"""
    try:
        if len(data) < 50:
            return []
        
        patterns = []
        close = data['Close']
        high = data['High']
        low = data['Low']
        
        # 1. Breakout Detection
        resistance = high.rolling(20).max().iloc[-1]
        support = low.rolling(20).min().iloc[-1]
        
        if price >= resistance * 0.99:  # Within 1% of resistance
            patterns.append({
                'Pattern': 'BREAKOUT',
                'Level': round(resistance, 2),
                'Distance%': round(((price - resistance) / resistance * 100), 2),
                'Signal': 'BULLISH'
            })
        
        if price <= support * 1.01:  # Within 1% of support
            patterns.append({
                'Pattern': 'SUPPORT BOUNCE',
                'Level': round(support, 2),
                'Distance%': round(((support - price) / support * 100), 2),
                'Signal': 'BULLISH'
            })
        
        # 2. Triangle Pattern (converging range)
        recent_highs = high.tail(20)
        recent_lows = low.tail(20)
        
        high_trend = recent_highs.iloc[-1] < recent_highs.iloc[0]
        low_trend = recent_lows.iloc[-1] > recent_lows.iloc[0]
        
        if high_trend and low_trend:
            patterns.append({
                'Pattern': 'TRIANGLE',
                'Level': round((recent_highs.iloc[-1] + recent_lows.iloc[-1]) / 2, 2),
                'Distance%': 0,
                'Signal': 'CONSOLIDATION'
            })
        
        # 3. Double Bottom/Top
        lows_20 = low.tail(20)
        highs_20 = high.tail(20)
        
        min_idx = lows_20.idxmin()
        if len(lows_20) > 10:
            second_half = lows_20[min_idx+5:]
            if not second_half.empty and abs(lows_20.min() - second_half.min()) / lows_20.min() < 0.02:
                patterns.append({
                    'Pattern': 'DOUBLE BOTTOM',
                    'Level': round(lows_20.min(), 2),
                    'Distance%': round(((price - lows_20.min()) / lows_20.min() * 100), 2),
                    'Signal': 'BULLISH'
                })
        
        return patterns
    except:
        return []

# --- LOAD FULL STOCK UNIVERSE ---
def load_stock_universe():
    """Load complete NSE stock list"""
    try:
        instruments = st.session_state.kite.instruments("NSE")
        
        # Filter for equity stocks only
        stocks = [i for i in instruments if i['segment'] == 'NSE' and i['instrument_type'] == 'EQ']
        
        stock_data = []
        for stock in stocks:
            stock_data.append({
                'Symbol': stock['tradingsymbol'],
                'Token': stock['instrument_token'],
                'Name': stock['name'],
                'Exchange': stock['exchange'],
                'Lot Size': stock.get('lot_size', 1)
            })
        
        df = pd.DataFrame(stock_data)
        
        # Store mappings
        st.session_state.instrument_map = {s['tradingsymbol']: s['instrument_token'] for s in stocks}
        st.session_state.token_to_symbol = {s['instrument_token']: s['tradingsymbol'] for s in stocks}
        
        return df
    except:
        return pd.DataFrame()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("# 🔱 PRIMA v13")
    st.markdown("### Professional Terminal")
    
    # --- AUTHENTICATION ---
    if not st.session_state.authenticated:
        st.error("🔴 **NOT CONNECTED**")
        st.markdown("---")
        
        login_url = st.session_state.kite.login_url()
        st.link_button("🔑 Login to Zerodha", login_url, use_container_width=True)
        
        request_token = st.text_input("Request Token:", type="password")
        
        if st.button("🚀 CONNECT", type="primary", use_container_width=True):
            try:
                with st.spinner("Authenticating..."):
                    data = st.session_state.kite.generate_session(request_token, api_secret=API_SECRET)
                    st.session_state.access_token = data["access_token"]
                    st.session_state.kite.set_access_token(data["access_token"])
                    
                    # Load stock universe
                    st.session_state.stock_universe = load_stock_universe()
                    
                    # Sync account
                    sync_zerodha_account()
                    
                    st.session_state.authenticated = True
                    st.success("✅ Connected!")
                    time.sleep(1)
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    else:
        st.success("🟢 **CONNECTED**")
        
        # Account info
        if st.session_state.zerodha_funds:
            st.markdown("---")
            st.markdown("### 💰 Account")
            st.metric("Available Cash", f"₹{round(st.session_state.zerodha_funds.get('available', 0), 2):,}")
            st.metric("Used Margin", f"₹{round(st.session_state.zerodha_funds.get('used', 0), 2):,}")
        
        st.markdown("---")
        
        # WebSocket control
        st.markdown("### 📡 Data Feed")
        
        if st.session_state.websocket_active:
            st.success("🟢 WebSocket LIVE")
            if st.button("🔌 Disconnect", use_container_width=True):
                if hasattr(st.session_state, 'ws_handler'):
                    st.session_state.ws_handler.stop()
                st.session_state.websocket_active = False
                st.rerun()
        else:
            st.warning("🟡 WebSocket OFF")
            if st.button("🔌 Connect WebSocket", type="primary", use_container_width=True):
                try:
                    ws_handler = WebSocketHandler(st.session_state.access_token, API_KEY)
                    st.session_state.ws_handler = ws_handler
                    ws_handler.start()
                    st.session_state.websocket_active = True
                    st.success("✅ WebSocket connected!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        
        st.markdown("---")
        
        # Bot control
        st.markdown("### 🤖 Trading Bot")
        
        bot_status = st.toggle("Enable Bot", value=st.session_state.bot_active, key="bot_toggle")
        
        if bot_status != st.session_state.bot_active:
            st.session_state.bot_active = bot_status
            if bot_status:
                send_telegram_msg("🟢 Bot ACTIVATED")
            else:
                send_telegram_msg("🔴 Bot DEACTIVATED")
        
        if st.session_state.bot_active:
            st.success("🟢 Bot ACTIVE")
            
            st.markdown("#### Settings")
            auto_trade = st.checkbox("Auto-Execute", value=False)
            max_trades = st.number_input("Max Positions", 1, 20, 5)
            trailing_stop = st.slider("Trailing Stop %", 0.5, 5.0, 1.5, 0.5)
        else:
            st.error("🔴 Bot INACTIVE")
        
        st.markdown("---")
        
        # Sync button
        if st.button("🔄 Sync Account", use_container_width=True):
            with st.spinner("Syncing..."):
                sync_zerodha_account()
            st.success("✅ Synced!")
            time.sleep(0.5)
            st.rerun()
        
        # Last sync time
        if st.session_state.last_sync > 0:
            time_ago = int(time.time() - st.session_state.last_sync)
            st.caption(f"Last sync: {time_ago}s ago")

# --- MAIN UI ---
if st.session_state.authenticated:
    
    # Update market status
    st.session_state.market_status = get_market_status()
    
    # Header
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("# 🔱 PRIMA PROFESSIONAL TERMINAL")
    
    with col2:
        # Market status indicator
        if st.session_state.market_status == "OPEN":
            st.markdown('<p class="status-live">🟢 MARKET OPEN</p>', unsafe_allow_html=True)
        elif st.session_state.market_status == "CLOSED":
            st.markdown("🔴 **MARKET CLOSED**")
        elif st.session_state.market_status == "PRE-MARKET":
            st.markdown("🟡 **PRE-MARKET**")
    
    st.markdown("---")
    
    # Account metrics
    if st.session_state.zerodha_funds:
        c1, c2, c3, c4 = st.columns(4)
        
        available = st.session_state.zerodha_funds.get('available', 0)
        used = st.session_state.zerodha_funds.get('used', 0)
        total = st.session_state.zerodha_funds.get('total', 0)
        
        # Calculate portfolio value from holdings
        portfolio_value = 0
        if not st.session_state.zerodha_holdings.empty:
            portfolio_value = st.session_state.zerodha_holdings['Value'].sum()
        
        c1.metric("💵 Available Cash", f"₹{round(available, 2):,}")
        c2.metric("📊 Portfolio Value", f"₹{round(portfolio_value, 2):,}")
        c3.metric("⚡ Used Margin", f"₹{round(used, 2):,}")
        c4.metric("💎 Total Equity", f"₹{round(total, 2):,}")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Market", "⚔️ Your Positions", "💼 Holdings", "🎯 Patterns & Signals"])
    
    # --- TAB 1: LIVE MARKET ---
    with tab1:
        st.markdown("### 📊 Live Stock Universe")
        
        if st.session_state.market_status != "OPEN":
            st.warning(f"⚠️ Market is {st.session_state.market_status}. Live data streaming paused.")
        
        # Controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            search = st.text_input("🔍 Search Symbol", placeholder="e.g. RELIANCE, TCS, INFY")
        
        with col2:
            sort_by = st.selectbox("Sort By", ["Change%", "LTP", "Volume", "Symbol"])
        
        with col3:
            filter_type = st.selectbox("Filter", ["All", "Gainers", "Losers", "High Volume"])
        
        # Build live table
        if not st.session_state.stock_universe.empty:
            
            # Get subset for performance (top 200 liquid stocks)
            display_stocks = st.session_state.stock_universe.head(200).copy()
            
            if search:
                display_stocks = display_stocks[display_stocks['Symbol'].str.contains(search.upper(), na=False)]
            
            # Add live data
            live_table = []
            for _, stock in display_stocks.iterrows():
                symbol = stock['Symbol']
                
                # Get live price
                if symbol in st.session_state.live_data:
                    data = st.session_state.live_data[symbol]
                    ltp = data['ltp']
                    change = data['change']
                    volume = data['volume']
                    high = data['high']
                    low = data['low']
                else:
                    # Fallback to API call if not in WebSocket data
                    try:
                        quote = st.session_state.kite.quote([f"NSE:{symbol}"])
                        if f"NSE:{symbol}" in quote:
                            q = quote[f"NSE:{symbol}"]
                            ltp = q['last_price']
                            change = q['net_change']
                            volume = q['volume']
                            high = q['ohlc']['high']
                            low = q['ohlc']['low']
                        else:
                            continue
                    except:
                        continue
                
                # Calculate change %
                if ltp > 0 and change != 0:
                    prev_close = ltp - change
                    change_pct = (change / prev_close * 100) if prev_close > 0 else 0
                else:
                    change_pct = 0
                
                # Apply filters
                if filter_type == "Gainers" and change_pct <= 0:
                    continue
                elif filter_type == "Losers" and change_pct >= 0:
                    continue
                elif filter_type == "High Volume" and volume < 100000:
                    continue
                
                live_table.append({
                    'Symbol': symbol,
                    'LTP': round(ltp, 2),
                    'Change': round(change, 2),
                    'Change%': round(change_pct, 2),
                    'High': round(high, 2),
                    'Low': round(low, 2),
                    'Volume': volume,
                    'Status': '🟢' if change_pct > 0 else '🔴' if change_pct < 0 else '⚪'
                })
            
            if live_table:
                df_live = pd.DataFrame(live_table)
                
                # Sort
                if sort_by == "Change%":
                    df_live = df_live.sort_values('Change%', ascending=False)
                elif sort_by == "LTP":
                    df_live = df_live.sort_values('LTP', ascending=False)
                elif sort_by == "Volume":
                    df_live = df_live.sort_values('Volume', ascending=False)
                else:
                    df_live = df_live.sort_values('Symbol')
                
                # Display with styling
                st.dataframe(
                    df_live,
                    column_config={
                        'Symbol': st.column_config.TextColumn('Symbol', width='small'),
                        'LTP': st.column_config.NumberColumn('LTP', format='₹%.2f'),
                        'Change': st.column_config.NumberColumn('Change', format='₹%.2f'),
                        'Change%': st.column_config.NumberColumn('Change%', format='%.2f%%'),
                        'High': st.column_config.NumberColumn('High', format='₹%.2f'),
                        'Low': st.column_config.NumberColumn('Low', format='₹%.2f'),
                        'Volume': st.column_config.NumberColumn('Volume', format='%d'),
                        'Status': st.column_config.TextColumn('', width='small')
                    },
                    hide_index=True,
                    height=600
                )
                
                st.caption(f"Showing {len(df_live)} stocks | Last updated: {datetime.now().strftime('%H:%M:%S')}")
            else:
                st.info("No stocks match your filters")
    
    # --- TAB 2: YOUR POSITIONS ---
    with tab2:
        st.markdown("### ⚔️ Live Positions from Zerodha")
        
        if not st.session_state.zerodha_positions.empty:
            st.dataframe(
                st.session_state.zerodha_positions,
                column_config={
                    'Avg Price': st.column_config.NumberColumn('Avg Price', format='₹%.2f'),
                    'LTP': st.column_config.NumberColumn('LTP', format='₹%.2f'),
                    'P&L': st.column_config.NumberColumn('P&L', format='₹%.2f'),
                    '%Change': st.column_config.NumberColumn('Change%', format='%.2f%%')
                },
                hide_index=True,
                height=400
            )
            
            # Summary
            total_pnl = st.session_state.zerodha_positions['P&L'].sum()
            winning = len(st.session_state.zerodha_positions[st.session_state.zerodha_positions['P&L'] > 0])
            total = len(st.session_state.zerodha_positions)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Positions", total)
            c2.metric("Total P&L", f"₹{round(total_pnl, 2):,}")
            c3.metric("Winning Positions", f"{winning}/{total}")
        else:
            st.info("📭 No open positions in your Zerodha account")
    
    # --- TAB 3: HOLDINGS ---
    with tab3:
        st.markdown("### 💼 Your Holdings from Zerodha")
        
        if not st.session_state.zerodha_holdings.empty:
            st.dataframe(
                st.session_state.zerodha_holdings,
                column_config={
                    'Avg Price': st.column_config.NumberColumn('Avg Price', format='₹%.2f'),
                    'LTP': st.column_config.NumberColumn('LTP', format='₹%.2f'),
                    'P&L': st.column_config.NumberColumn('P&L', format='₹%.2f'),
                    'Day Change%': st.column_config.NumberColumn('Day Change%', format='%.2f%%'),
                    'Value': st.column_config.NumberColumn('Value', format='₹%.2f')
                },
                hide_index=True,
                height=400
            )
            
            # Summary
            total_value = st.session_state.zerodha_holdings['Value'].sum()
            total_pnl = st.session_state.zerodha_holdings['P&L'].sum()
            
            c1, c2 = st.columns(2)
            c1.metric("Total Value", f"₹{round(total_value, 2):,}")
            c2.metric("Total P&L", f"₹{round(total_pnl, 2):,}")
        else:
            st.info("📭 No holdings in your Zerodha account")
    
    # --- TAB 4: PATTERNS & SIGNALS ---
    with tab4:
        st.markdown("### 🎯 Pattern Detection & Breakout Analysis")
        
        if st.session_state.market_status != "OPEN":
            st.warning("⚠️ Pattern detection works during market hours")
        
        if st.button("🔍 Scan for Patterns", type="primary"):
            with st.spinner("Scanning for patterns..."):
                
                # Top 50 liquid stocks
                scan_symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'SBIN', 
                               'BHARTIARTL', 'ITC', 'KOTAKBANK', 'LT', 'HINDUNILVR', 'AXISBANK',
                               'BAJFINANCE', 'ASIANPAINT', 'MARUTI', 'TITAN', 'SUNPHARMA', 
                               'ULTRACEMCO', 'NESTLEIND', 'WIPRO', 'HCLTECH', 'TATAMOTORS']
                
                pattern_results = []
                
                for symbol in scan_symbols:
                    try:
                        # Get historical data
                        token = st.session_state.instrument_map.get(symbol)
                        if not token:
                            continue
                        
                        to_date = datetime.now()
                        from_date = to_date - timedelta(days=30)
                        
                        records = st.session_state.kite.historical_data(token, from_date, to_date, "day")
                        df = pd.DataFrame(records)
                        
                        if df.empty:
                            continue
                        
                        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'oi']
                        df = df.rename(columns={'close': 'Close', 'high': 'High', 'low': 'Low'})
                        
                        # Get current price
                        price = df['Close'].iloc[-1]
                        
                        # Detect patterns
                        patterns = detect_patterns(df, symbol, price)
                        
                        for p in patterns:
                            pattern_results.append({
                                'Symbol': symbol,
                                'Pattern': p['Pattern'],
                                'Signal': p['Signal'],
                                'Level': p['Level'],
                                'Distance%': p['Distance%'],
                                'LTP': round(price, 2)
                            })
                    except:
                        continue
                
                if pattern_results:
                    st.success(f"✅ Found {len(pattern_results)} patterns!")
                    
                    df_patterns = pd.DataFrame(pattern_results)
                    
                    st.dataframe(
                        df_patterns,
                        column_config={
                            'Level': st.column_config.NumberColumn('Level', format='₹%.2f'),
                            'Distance%': st.column_config.NumberColumn('Distance%', format='%.2f%%'),
                            'LTP': st.column_config.NumberColumn('LTP', format='₹%.2f')
                        },
                        hide_index=True,
                        height=400
                    )
                else:
                    st.info("No significant patterns detected")
    
    # Auto-refresh every 2 seconds
    time.sleep(2)
    st.rerun()

else:
    # Not authenticated
    st.markdown("# 🔱 PRIMA PROFESSIONAL TERMINAL")
    st.markdown("---")
    st.info("👈 Please login with your Zerodha account in the sidebar to start")
    
    st.markdown("""
    ### Features:
    - ✅ Live Zerodha account integration
    - ✅ Real-time WebSocket data for all NSE stocks
    - ✅ Your positions & holdings monitoring
    - ✅ Pattern detection & breakout analysis
    - ✅ Professional institutional-grade UI
    - ✅ Market status monitoring
    """)
