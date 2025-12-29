import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import time
import numpy as np

# Try to import yfinance (install if needed: pip install yfinance)
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    st.warning("⚠️ yfinance not installed. Run: pip install yfinance")

# Page config
st.set_page_config(page_title="Trading Signals Pro", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #0e1117;}
    .stSelectbox {margin-bottom: 10px;}
    .signal-buy {
        background-color: #00ff00;
        color: black;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
    }
    .signal-sell {
        background-color: #ff0000;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
    }
    .signal-neutral {
        background-color: #ffa500;
        color: black;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Trading instruments by category
INSTRUMENTS = {
    "Stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "V", "WMT"],
    "Indices": ["SPY", "QQQ", "DIA", "IWM", "VIX"],
    "Commodities": ["GC=F", "CL=F", "NG=F", "ZC=F", "ZS=F"],
    "Metals": ["GC=F", "SI=F", "PL=F", "PA=F", "HG=F"],
    "Forex": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X"]
}

# Initialize session state
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False

# Sidebar
with st.sidebar:
    st.title("⚡ Trading Signals Pro")
    st.markdown("---")
    
    # Category selection
    category = st.selectbox("📊 Select Market", list(INSTRUMENTS.keys()))
    
    # Instrument selection
    symbol = st.selectbox("🎯 Select Instrument", INSTRUMENTS[category])
    
    # Timeframe selection
    timeframe = st.selectbox("⏱️ Timeframe", ["5min", "15min", "30min", "1hour"], index=1)
    
    st.markdown("---")
    st.markdown("### 🔑 API Configuration")
    
    # API Provider selection
    api_provider = st.selectbox(
        "📡 API Provider",
        ["Yahoo Finance (FREE - No API Key)", "Alpha Vantage (Stocks only)", "Polygon.io (All markets)", "Twelve Data (All markets)", "Demo Data"],
        help="Yahoo Finance works for all instruments without API key!"
    )
    
    # API Key input (not needed for Yahoo Finance)
    if "Yahoo Finance" not in api_provider:
        api_key = st.text_input(
            "🔑 API Key", 
            type="password", 
            help=f"Enter your {api_provider.split('(')[0].strip()} API key",
            key="api_key_input"
        )
        
        if api_key:
            st.success(f"✅ API Key entered ({len(api_key)} chars)")
        else:
            st.info("💡 Enter API key for live data")
    else:
        api_key = None
        st.success("✅ Yahoo Finance - No API key needed!")
    
    # Show API signup links
    with st.expander("🔗 Get Free API Keys"):
        st.markdown("""
        **Yahoo Finance** (FREE - No API needed!)
        - Works for: Stocks, Commodities, Forex, Indices
        - No signup required!
        
        **Alpha Vantage** (Stocks only - 25 calls/day)
        - Get key: https://www.alphavantage.co/support/#api-key
        
        **Polygon.io** (All markets - 5 calls/min free)
        - Get key: https://polygon.io/dashboard/signup
        - Best for: Stocks, Forex, Crypto, Commodities
        
        **Twelve Data** (All markets - 800 calls/day free)
        - Get key: https://twelvedata.com/pricing
        - Best for: Commodities, Forex, Stocks
        """)
    
    st.markdown("---")
    
    # Auto-refresh toggle
    st.session_state.auto_refresh = st.checkbox("🔄 Auto-refresh (30s)", value=st.session_state.auto_refresh)
    
    # Manual refresh button
    refresh_btn = st.button("🔃 Refresh Now", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 📈 Signal Settings")
    rsi_period = st.slider("RSI Period", 5, 30, 14)
    ema_fast = st.slider("Fast EMA", 5, 50, 9)
    ema_slow = st.slider("Slow EMA", 20, 200, 21)


def fetch_live_data(symbol, timeframe, api_key, api_provider):
    """
    Fetch live data from multiple API providers
    """
    if "Yahoo Finance" in api_provider:
        return fetch_yahoo_finance(symbol, timeframe)
    elif not api_key or len(api_key) < 10 or "Demo Data" in api_provider:
        # Generate dummy data for demo
        dates = pd.date_range(end=datetime.now(), periods=100, freq='5min')
        base_price = np.random.uniform(100, 200)
        df = pd.DataFrame({
            'timestamp': dates,
            'open': base_price + np.random.uniform(-5, 5, 100),
            'high': base_price + np.random.uniform(0, 8, 100),
            'low': base_price + np.random.uniform(-8, 0, 100),
            'close': base_price + np.random.uniform(-5, 5, 100),
            'volume': np.random.randint(1000000, 10000000, 100)
        })
        return df
    
    # Route to appropriate API based on provider
    if "Alpha Vantage" in api_provider:
        return fetch_alpha_vantage(symbol, timeframe, api_key)
    elif "Polygon" in api_provider:
        return fetch_polygon(symbol, timeframe, api_key)
    elif "Twelve Data" in api_provider:
        return fetch_twelve_data(symbol, timeframe, api_key)
    else:
        return None


def fetch_yahoo_finance(symbol, timeframe):
    """Fetch data from Yahoo Finance (FREE - No API key needed)"""
    if not YFINANCE_AVAILABLE:
        st.error("❌ yfinance not installed. Run: pip install yfinance")
        return None
    
    try:
        # Map timeframe to yfinance interval
        interval_map = {
            "5min": "5m",
            "15min": "15m",
            "30min": "30m",
            "1hour": "1h"
        }
        interval = interval_map.get(timeframe, "15m")
        
        # Download data
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="5d", interval=interval)
        
        if df.empty:
            st.warning(f"⚠️ No data for {symbol}. Check symbol format.")
            return None
        
        # Reset index and rename columns
        df = df.reset_index()
        df.columns = [col.lower() if col != 'Datetime' else 'timestamp' for col in df.columns]
        
        # Keep only needed columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.tail(100)
        
        st.success(f"✅ Live data loaded from Yahoo Finance (FREE)!")
        return df
        
    except Exception as e:
        st.error(f"❌ Yahoo Finance Error: {str(e)}")
        return None


def fetch_alpha_vantage(symbol, timeframe, api_key):
    """Fetch data from Alpha Vantage"""
    try:
        interval_map = {"5min": "5min", "15min": "15min", "30min": "30min", "1hour": "60min"}
        interval = interval_map.get(timeframe, "15min")
        
        clean_symbol = symbol.replace("=X", "").replace("=F", "")
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={clean_symbol}&interval={interval}&apikey={api_key}&outputsize=full"
        
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if "Error Message" in data:
            st.error(f"API Error: {data['Error Message']}")
            return None
        
        if "Note" in data:
            st.warning(f"API Limit: {data['Note']}")
            return None
        
        time_series_key = f"Time Series ({interval})"
        if time_series_key not in data:
            st.warning("⚠️ No data from Alpha Vantage. Try a US stock symbol (AAPL, MSFT) or use Polygon/Twelve Data for commodities.")
            return None
        
        time_series = data[time_series_key]
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        df = df.reset_index()
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = df.tail(100)
        
        st.success(f"✅ Live data loaded from Alpha Vantage!")
        return df
        
    except Exception as e:
        st.error(f"Alpha Vantage Error: {str(e)}")
        return None


def fetch_polygon(symbol, timeframe, api_key):
    """Fetch data from Polygon.io"""
    try:
        # Map symbols for Polygon format
        symbol_map = {
            "GC=F": "X:XAUUSD",  # Gold (Forex pair format)
            "SI=F": "X:XAGUSD",  # Silver
            "CL=F": "X:WTIUSD",  # Crude Oil
            "NG=F": "X:NGUSD",   # Natural Gas
            "EURUSD=X": "C:EURUSD",
            "GBPUSD=X": "C:GBPUSD",
        }
        
        polygon_symbol = symbol_map.get(symbol, symbol)
        
        # Map timeframe
        multiplier_map = {"5min": 5, "15min": 15, "30min": 30, "1hour": 60}
        multiplier = multiplier_map.get(timeframe, 15)
        
        # Get date range - use more recent dates
        to_date = datetime.now()
        from_date = to_date - timedelta(days=2)
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{polygon_symbol}/range/{multiplier}/minute/{from_date.strftime('%Y-%m-%d')}/{to_date.strftime('%Y-%m-%d')}?adjusted=true&sort=asc&apiKey={api_key}"
        
        st.info(f"🔍 Trying Polygon with symbol: {polygon_symbol}")
        
        response = requests.get(url, timeout=10)
        data = response.json()
        
        # Better error handling
        if data.get('status') == 'ERROR':
            error_msg = data.get('error', 'Unknown error')
            st.error(f"❌ Polygon API Error: {error_msg}")
            st.warning("💡 Polygon free tier requires verification. Try Twelve Data or use stocks with Alpha Vantage.")
            return None
        
        if data.get('status') != 'OK':
            st.warning(f"⚠️ Polygon response: {data.get('status', 'Unknown')} - {data.get('message', '')}")
            return None
        
        if 'results' not in data or not data['results']:
            st.warning("⚠️ No data from Polygon. This symbol might require a paid plan.")
            return None
        
        results = data['results']
        df = pd.DataFrame(results)
        
        # Rename columns to match our format
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
        df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.tail(100)
        
        st.success(f"✅ Live data loaded from Polygon.io!")
        return df
        
    except Exception as e:
        st.error(f"❌ Polygon Error: {str(e)}")
        st.info("💡 Try Twelve Data or select a US stock (AAPL, MSFT) with Alpha Vantage")
        return None


def fetch_twelve_data(symbol, timeframe, api_key):
    """Fetch data from Twelve Data"""
    try:
        # Map symbols for Twelve Data
        symbol_map = {
            "GC=F": "XAU/USD",  # Gold
            "SI=F": "XAG/USD",  # Silver
            "CL=F": "WTI/USD",  # Crude Oil
            "EURUSD=X": "EUR/USD",
            "GBPUSD=X": "GBP/USD",
        }
        
        twelve_symbol = symbol_map.get(symbol, symbol)
        
        # Map timeframe
        interval_map = {"5min": "5min", "15min": "15min", "30min": "30min", "1hour": "1h"}
        interval = interval_map.get(timeframe, "15min")
        
        url = f"https://api.twelvedata.com/time_series?symbol={twelve_symbol}&interval={interval}&outputsize=100&apikey={api_key}"
        
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if data.get('status') == 'error':
            st.error(f"Twelve Data Error: {data.get('message', 'Unknown error')}")
            return None
        
        if 'values' not in data:
            st.warning("⚠️ No data from Twelve Data. Check symbol or API limits.")
            return None
        
        values = data['values']
        df = pd.DataFrame(values)
        
        df['timestamp'] = pd.to_datetime(df['datetime'])
        
        # Handle missing volume for forex/commodities
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col])
        
        if 'volume' not in df.columns or df['volume'].isna().all():
            df['volume'] = 1000000  # Default volume for forex/commodities
        else:
            df['volume'] = pd.to_numeric(df['volume'])
        
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.sort_values('timestamp')
        df = df.tail(100)
        
        st.success(f"✅ Live data loaded from Twelve Data!")
        return df
        
    except Exception as e:
        st.error(f"Twelve Data Error: {str(e)}")
        return None


def calculate_indicators(df):
    """Calculate technical indicators"""
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # EMAs
    df['ema_fast'] = df['close'].ewm(span=ema_fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=ema_slow, adjust=False).mean()
    
    # Volume analysis
    df['vol_ma'] = df['volume'].rolling(window=20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma']
    
    # VWAP (Volume Weighted Average Price)
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    
    return df


def generate_signals(df):
    """Generate buy/sell signals"""
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    signals = []
    signal_type = "NEUTRAL"
    
    # EMA Crossover
    if latest['ema_fast'] > latest['ema_slow'] and prev['ema_fast'] <= prev['ema_slow']:
        signals.append("🟢 EMA Bullish Crossover")
        signal_type = "BUY"
    elif latest['ema_fast'] < latest['ema_slow'] and prev['ema_fast'] >= prev['ema_slow']:
        signals.append("🔴 EMA Bearish Crossover")
        signal_type = "SELL"
    
    # RSI Signals
    if latest['rsi'] < 30:
        signals.append("🟢 RSI Oversold (< 30)")
        if signal_type != "SELL":
            signal_type = "BUY"
    elif latest['rsi'] > 70:
        signals.append("🔴 RSI Overbought (> 70)")
        if signal_type != "BUY":
            signal_type = "SELL"
    
    # Volume spike
    if latest['vol_ratio'] > 1.5:
        signals.append("📊 High Volume Alert")
    
    # Price vs VWAP
    if latest['close'] > latest['vwap']:
        signals.append("📈 Price Above VWAP")
    else:
        signals.append("📉 Price Below VWAP")
    
    return signal_type, signals, latest


def create_chart(df):
    """Create interactive candlestick chart with indicators"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=('Price Action', 'RSI', 'Volume')
    )
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price'
    ), row=1, col=1)
    
    # EMAs
    fig.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['ema_fast'],
        name=f'EMA {ema_fast}',
        line=dict(color='cyan', width=1)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['ema_slow'],
        name=f'EMA {ema_slow}',
        line=dict(color='orange', width=1)
    ), row=1, col=1)
    
    # VWAP
    fig.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['vwap'],
        name='VWAP',
        line=dict(color='purple', width=1, dash='dash')
    ), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['rsi'],
        name='RSI',
        line=dict(color='yellow', width=2)
    ), row=2, col=1)
    
    # RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # Volume
    colors = ['red' if df['close'].iloc[i] < df['open'].iloc[i] else 'green' 
              for i in range(len(df))]
    
    fig.add_trace(go.Bar(
        x=df['timestamp'], 
        y=df['volume'],
        name='Volume',
        marker_color=colors
    ), row=3, col=1)
    
    fig.update_layout(
        title=f'{symbol} - {timeframe} Chart',
        yaxis_title='Price',
        template='plotly_dark',
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="Volume", row=3, col=1)
    
    return fig


# Main content
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown(f"## 📊 {symbol} - {timeframe}")

with col2:
    st.markdown(f"**Last Update:** {datetime.now().strftime('%H:%M:%S')}")

with col3:
    status = "🟢 LIVE" if st.session_state.auto_refresh else "⚪ PAUSED"
    st.markdown(f"**Status:** {status}")

# Fetch and process data
df = fetch_live_data(symbol, timeframe, api_key, api_provider)

if df is not None and len(df) > 0:
    df = calculate_indicators(df)
    signal_type, signals, latest = generate_signals(df)
    
    # Display signal box
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        signal_class = f"signal-{signal_type.lower()}"
        st.markdown(f'<div class="{signal_class}">SIGNAL: {signal_type}</div>', unsafe_allow_html=True)
    
    with col2:
        st.metric("Current Price", f"${latest['close']:.2f}", 
                  f"{((latest['close'] - df.iloc[-2]['close']) / df.iloc[-2]['close'] * 100):.2f}%")
    
    with col3:
        st.metric("RSI", f"{latest['rsi']:.1f}")
    
    with col4:
        st.metric("Volume Ratio", f"{latest['vol_ratio']:.2f}x")
    
    # Signal details
    st.markdown("### 🎯 Active Signals")
    for signal in signals:
        st.markdown(f"- {signal}")
    
    # Chart
    st.plotly_chart(create_chart(df), use_container_width=True)
    
    # Market metrics
    st.markdown("---")
    st.markdown("### 📊 Market Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("24h High", f"${df['high'].max():.2f}")
    with col2:
        st.metric("24h Low", f"${df['low'].min():.2f}")
    with col3:
        st.metric("Avg Volume", f"{df['volume'].mean()/1e6:.2f}M")
    with col4:
        volatility = ((df['high'] - df['low']) / df['close'] * 100).mean()
        st.metric("Avg Volatility", f"{volatility:.2f}%")
    
    # Recent data table
    with st.expander("📋 Recent Price Data"):
        st.dataframe(df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'rsi']].tail(20).iloc[::-1],
                    use_container_width=True)

else:
    st.warning("⚠️ No data available")
    st.info("💡 **To get live data:**\n\n1. Get FREE API key: https://www.alphavantage.co/support/#api-key\n2. Paste your API key in the sidebar (🔑 API Key field)\n3. Select a US stock symbol (AAPL, MSFT, TSLA, etc.)\n4. Click 'Refresh Now' button\n\n**Current Status:**")
    
    col1, col2 = st.columns(2)
    with col1:
        if api_key:
            st.success(f"✅ API Key: Entered ({len(api_key)} characters)")
        else:
            st.error("❌ API Key: Not entered")
    
    with col2:
        st.info(f"📊 Symbol: {symbol}")
        st.info(f"⏱️ Timeframe: {timeframe}")
    
    st.markdown("---")
    st.markdown("**Test your API key here:** https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=YOUR_KEY")
    st.markdown("Replace YOUR_KEY with your actual key and open in browser to test.")

# Auto-refresh logic
if st.session_state.auto_refresh:
    time.sleep(30)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("**⚠️ Disclaimer:** This tool is for educational purposes only. Always do your own research before making trading decisions.")
