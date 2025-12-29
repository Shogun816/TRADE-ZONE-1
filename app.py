import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import time
import numpy as np

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
    
    # API Key input
    api_key = st.text_input("🔑 API Key", type="password", help="Enter your financial data API key (Alpha Vantage, Polygon, etc.)")
    
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


def fetch_live_data(symbol, timeframe, api_key):
    """
    Fetch live data from your API provider
    Replace this with your actual API integration
    """
    # Example using Alpha Vantage (replace with your API)
    if not api_key:
        # Generate dummy data for demo
        dates = pd.date_range(end=datetime.now(), periods=100, freq='5min')
        df = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(105, 115, 100),
            'low': np.random.uniform(95, 105, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.randint(1000000, 10000000, 100)
        })
        return df
    
    # TODO: Implement your actual API call here
    # Example for Alpha Vantage:
    # url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={timeframe}&apikey={api_key}"
    # response = requests.get(url)
    # data = response.json()
    # Parse and return DataFrame
    
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
df = fetch_live_data(symbol, timeframe, api_key)

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
    st.warning("⚠️ Please enter your API key to fetch live data, or the app will use demo data.")
    st.info("💡 **Setup Instructions:**\n1. Get a free API key from Alpha Vantage, Polygon.io, or your preferred provider\n2. Enter the API key in the sidebar\n3. Select your instrument and timeframe\n4. Enable auto-refresh for real-time updates")

# Auto-refresh logic
if st.session_state.auto_refresh:
    time.sleep(30)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("**⚠️ Disclaimer:** This tool is for educational purposes only. Always do your own research before making trading decisions.")
