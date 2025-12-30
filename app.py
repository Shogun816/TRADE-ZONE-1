import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import time
import numpy as np

# Try to import yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Page config
st.set_page_config(page_title="Trading Signals Pro", layout="wide", initial_sidebar_state="expanded")

# Custom CSS with enhanced styling
st.markdown("""
<style>
    .main {background-color: #0e1117;}
    .signal-strong-buy {
        background: linear-gradient(135deg, #00ff00 0%, #00cc00 100%);
        color: black;
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
        font-size: 20px;
        text-align: center;
        animation: pulse 2s infinite;
        box-shadow: 0 0 20px #00ff00;
    }
    .signal-buy {
        background-color: #00ff00;
        color: black;
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
        font-size: 18px;
        text-align: center;
    }
    .signal-strong-sell {
        background: linear-gradient(135deg, #ff0000 0%, #cc0000 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
        font-size: 20px;
        text-align: center;
        animation: pulse 2s infinite;
        box-shadow: 0 0 20px #ff0000;
    }
    .signal-sell {
        background-color: #ff0000;
        color: white;
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
        font-size: 18px;
        text-align: center;
    }
    .signal-neutral {
        background-color: #ffa500;
        color: black;
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
        font-size: 18px;
        text-align: center;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    .strategy-box {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .scalping {
        background-color: #ff6b6b;
        color: white;
    }
    .daytrading {
        background-color: #4ecdc4;
        color: black;
    }
    .swing {
        background-color: #45b7d1;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Trading instruments by category
INSTRUMENTS = {
    "Stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "V", "WMT"],
    "Indices": ["^GSPC", "^IXIC", "^DJI", "SPY", "QQQ", "DIA", "IWM", "^VIX"],
    "Commodities": ["GC=F", "CL=F", "NG=F", "ZC=F", "ZS=F"],
    "Metals": ["GC=F", "SI=F", "PL=F", "PA=F", "HG=F"],
    "Forex": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X"]
}

# Trading strategy presets
STRATEGY_PRESETS = {
    "Scalping (5min)": {
        "timeframe": "5min",
        "rsi_period": 9,
        "ema_fast": 5,
        "ema_slow": 13,
        "rsi_oversold": 25,
        "rsi_overbought": 75,
        "color": "#ff6b6b"
    },
    "Day Trading (15min)": {
        "timeframe": "15min",
        "rsi_period": 14,
        "ema_fast": 9,
        "ema_slow": 21,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "color": "#4ecdc4"
    },
    "Swing Trading (1hour)": {
        "timeframe": "1hour",
        "rsi_period": 14,
        "ema_fast": 21,
        "ema_slow": 50,
        "rsi_oversold": 35,
        "rsi_overbought": 65,
        "color": "#45b7d1"
    }
}

# Initialize session state
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False
if 'last_signal' not in st.session_state:
    st.session_state.last_signal = "NEUTRAL"
if 'sound_enabled' not in st.session_state:
    st.session_state.sound_enabled = True

# Sidebar
with st.sidebar:
    st.title("⚡ Trading Signals Pro")
    st.markdown("---")
    
    # Strategy selection
    st.markdown("### 🎯 Trading Strategy")
    strategy = st.selectbox(
        "Select Strategy",
        list(STRATEGY_PRESETS.keys()),
        help="Pre-configured settings for different trading styles"
    )
    
    preset = STRATEGY_PRESETS[strategy]
    st.markdown(f'<div class="strategy-box {strategy.split()[0].lower()}">{strategy}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Category selection
    category = st.selectbox("📊 Select Market", list(INSTRUMENTS.keys()))
    
    # Instrument selection
    symbol = st.selectbox("🎯 Select Instrument", INSTRUMENTS[category])
    
    # Timeframe (auto-set by strategy but can override)
    timeframe = st.selectbox("⏱️ Timeframe", ["5min", "15min", "30min", "1hour"], 
                             index=["5min", "15min", "30min", "1hour"].index(preset["timeframe"]))
    
    st.markdown("---")
    st.markdown("### 🔑 API Configuration")
    
    api_provider = st.selectbox(
        "📡 API Provider",
        ["Yahoo Finance (FREE - No API Key)", "Alpha Vantage (Stocks only)", "Demo Data"],
        help="Yahoo Finance works for all instruments without API key!"
    )
    
    if "Yahoo Finance" not in api_provider:
        api_key = st.text_input("🔑 API Key", type="password", key="api_key_input")
        if api_key:
            st.success(f"✅ API Key entered ({len(api_key)} chars)")
    else:
        api_key = None
        st.success("✅ Yahoo Finance - No API key needed!")
    
    st.markdown("---")
    
    # Advanced settings
    with st.expander("⚙️ Advanced Settings", expanded=False):
        st.markdown("**Customize Indicator Settings**")
        rsi_period = st.slider("RSI Period", 5, 30, preset["rsi_period"])
        ema_fast = st.slider("Fast EMA", 5, 50, preset["ema_fast"])
        ema_slow = st.slider("Slow EMA", 10, 200, preset["ema_slow"])
        
        st.markdown("**RSI Levels**")
        rsi_oversold = st.slider("Oversold Level", 20, 40, preset["rsi_oversold"])
        rsi_overbought = st.slider("Overbought Level", 60, 80, preset["rsi_overbought"])
    
    st.markdown("---")
    
    # Control buttons
    st.session_state.auto_refresh = st.checkbox("🔄 Auto-refresh (30s)", value=st.session_state.auto_refresh)
    st.session_state.sound_enabled = st.checkbox("🔊 Sound Alerts", value=st.session_state.sound_enabled)
    refresh_btn = st.button("🔃 Refresh Now", use_container_width=True)


def fetch_yahoo_finance(symbol, timeframe):
    """Fetch data from Yahoo Finance"""
    if not YFINANCE_AVAILABLE:
        st.error("❌ yfinance not installed. Run: pip install yfinance")
        return None
    
    try:
        interval_map = {"5min": "5m", "15min": "15m", "30min": "30m", "1hour": "1h"}
        interval = interval_map.get(timeframe, "15m")
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="5d", interval=interval)
        
        if df.empty:
            st.warning(f"⚠️ No data for {symbol}. Check symbol format.")
            return None
        
        df = df.reset_index()
        df.columns = [col.lower() if col != 'Datetime' else 'timestamp' for col in df.columns]
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.tail(100)
        
        st.success(f"✅ Live data loaded from Yahoo Finance!")
        return df
        
    except Exception as e:
        st.error(f"❌ Yahoo Finance Error: {str(e)}")
        return None


def fetch_live_data(symbol, timeframe, api_key, api_provider):
    """Fetch live data from provider"""
    if "Yahoo Finance" in api_provider:
        return fetch_yahoo_finance(symbol, timeframe)
    else:
        # Generate demo data
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


def calculate_indicators(df, rsi_period, ema_fast, ema_slow):
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
    
    # VWAP
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    
    # Price momentum
    df['momentum'] = df['close'].pct_change(periods=5) * 100
    
    return df


def generate_signals(df, rsi_oversold, rsi_overbought):
    """Generate advanced buy/sell signals"""
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    signals = []
    signal_strength = 0  # Track signal strength
    
    # EMA Crossover (Weight: 3)
    if latest['ema_fast'] > latest['ema_slow'] and prev['ema_fast'] <= prev['ema_slow']:
        signals.append("🟢 EMA Bullish Crossover")
        signal_strength += 3
    elif latest['ema_fast'] < latest['ema_slow'] and prev['ema_fast'] >= prev['ema_slow']:
        signals.append("🔴 EMA Bearish Crossover")
        signal_strength -= 3
    
    # RSI Signals (Weight: 2)
    if latest['rsi'] < rsi_oversold:
        signals.append(f"🟢 RSI Oversold ({latest['rsi']:.1f})")
        signal_strength += 2
    elif latest['rsi'] > rsi_overbought:
        signals.append(f"🔴 RSI Overbought ({latest['rsi']:.1f})")
        signal_strength -= 2
    
    # Volume spike (Weight: 1)
    if latest['vol_ratio'] > 1.5:
        signals.append(f"📊 High Volume ({latest['vol_ratio']:.2f}x)")
        signal_strength += 1
    
    # Price vs VWAP (Weight: 1)
    if latest['close'] > latest['vwap']:
        signals.append("📈 Price Above VWAP")
        signal_strength += 0.5
    else:
        signals.append("📉 Price Below VWAP")
        signal_strength -= 0.5
    
    # Momentum (Weight: 1)
    if latest['momentum'] > 1:
        signals.append(f"🚀 Strong Upward Momentum (+{latest['momentum']:.2f}%)")
        signal_strength += 1
    elif latest['momentum'] < -1:
        signals.append(f"⚠️ Strong Downward Momentum ({latest['momentum']:.2f}%)")
        signal_strength -= 1
    
    # Determine overall signal
    if signal_strength >= 4:
        signal_type = "STRONG BUY"
    elif signal_strength >= 2:
        signal_type = "BUY"
    elif signal_strength <= -4:
        signal_type = "STRONG SELL"
    elif signal_strength <= -2:
        signal_type = "SELL"
    else:
        signal_type = "NEUTRAL"
    
    return signal_type, signals, latest, signal_strength


def create_advanced_chart(df, ema_fast, ema_slow, rsi_oversold, rsi_overbought, symbol, timeframe):
    """Create advanced trading chart with color-coded signals"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f'{symbol} - {timeframe} - Price Action', 'RSI Indicator', 'Volume Profile')
    )
    
    # Candlestick with custom colors
    colors = ['red' if row['close'] < row['open'] else 'green' for _, row in df.iterrows()]
    
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price',
        increasing_line_color='#00ff00',
        decreasing_line_color='#ff0000'
    ), row=1, col=1)
    
    # EMAs with thicker lines
    fig.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['ema_fast'],
        name=f'EMA {ema_fast}',
        line=dict(color='cyan', width=2)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['ema_slow'],
        name=f'EMA {ema_slow}',
        line=dict(color='orange', width=2)
    ), row=1, col=1)
    
    # VWAP
    fig.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['vwap'],
        name='VWAP',
        line=dict(color='purple', width=2, dash='dash')
    ), row=1, col=1)
    
    # Mark EMA crossovers
    for i in range(1, len(df)):
        if df['ema_fast'].iloc[i] > df['ema_slow'].iloc[i] and df['ema_fast'].iloc[i-1] <= df['ema_slow'].iloc[i-1]:
            fig.add_annotation(
                x=df['timestamp'].iloc[i],
                y=df['low'].iloc[i],
                text="▲ BUY",
                showarrow=True,
                arrowhead=2,
                arrowcolor="green",
                ax=0,
                ay=40,
                font=dict(color="green", size=12),
                row=1, col=1
            )
        elif df['ema_fast'].iloc[i] < df['ema_slow'].iloc[i] and df['ema_fast'].iloc[i-1] >= df['ema_slow'].iloc[i-1]:
            fig.add_annotation(
                x=df['timestamp'].iloc[i],
                y=df['high'].iloc[i],
                text="▼ SELL",
                showarrow=True,
                arrowhead=2,
                arrowcolor="red",
                ax=0,
                ay=-40,
                font=dict(color="red", size=12),
                row=1, col=1
            )
    
    # RSI with zones
    fig.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['rsi'],
        name='RSI',
        line=dict(color='yellow', width=3)
    ), row=2, col=1)
    
    # RSI levels
    fig.add_hline(y=rsi_overbought, line_dash="dash", line_color="red", line_width=2, row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="white", line_width=1, row=2, col=1)
    fig.add_hline(y=rsi_oversold, line_dash="dash", line_color="green", line_width=2, row=2, col=1)
    
    # Color zones for RSI
    fig.add_hrect(y0=rsi_overbought, y1=100, fillcolor="red", opacity=0.1, row=2, col=1)
    fig.add_hrect(y0=0, y1=rsi_oversold, fillcolor="green", opacity=0.1, row=2, col=1)
    
    # Volume with colors
    volume_colors = ['red' if df['close'].iloc[i] < df['open'].iloc[i] else 'green' 
                     for i in range(len(df))]
    
    fig.add_trace(go.Bar(
        x=df['timestamp'], 
        y=df['volume'],
        name='Volume',
        marker_color=volume_colors,
        showlegend=False
    ), row=3, col=1)
    
    # Volume MA
    fig.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['vol_ma'],
        name='Vol MA',
        line=dict(color='white', width=1, dash='dash')
    ), row=3, col=1)
    
    fig.update_layout(
        title=dict(
            text=f'{symbol} - {timeframe} Live Chart',
            font=dict(size=24, color='white')
        ),
        template='plotly_dark',
        height=900,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="Volume", row=3, col=1)
    
    return fig


def play_alert_sound(signal_type):
    """Play sound alert for signals"""
    if st.session_state.sound_enabled and signal_type != st.session_state.last_signal:
        if "BUY" in signal_type:
            st.markdown("""
                <audio autoplay>
                    <source src="data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSx+zfPaizsIHGS86+OSUAoNU6Ln7bFiFgY5j9Xxy3osBSh4yO/ekUEKFV2y6OypWBENR57e8L11KAU" type="audio/wav">
                </audio>
            """, unsafe_allow_html=True)
        elif "SELL" in signal_type:
            st.markdown("""
                <audio autoplay>
                    <source src="data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAADB2e3z7+vn497W0MrEvrm0r6qlo56Zm5SQi4aAe3hzb2tnamZjYF5bWVdUUlBOS0lHRUNBPz07OTc1MzEvLCooJiQiIB4cGhgWFBIQDgwKCAYEAgDC3O/z8O3o5N/Z0s3Hw8C7trGuraWjoZyZk4+LiISAe3dzcGxoZGBdWVVST0xJR0RCQT49Ozk3NTQyMC0rKSckIiAeHBoYFhQSEA4MCAYEAg==" type="audio/wav">
                </audio>
            """, unsafe_allow_html=True)
        
        st.session_state.last_signal = signal_type


# Main content
st.markdown(f"<h1 style='text-align: center; color: {preset['color']};'>📊 {strategy.upper()}</h1>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown(f"## {symbol}")

with col2:
    st.markdown(f"**Last Update:** {datetime.now().strftime('%H:%M:%S')}")

with col3:
    status = "🟢 LIVE" if st.session_state.auto_refresh else "⚪ PAUSED"
    st.markdown(f"**Status:** {status}")

# Fetch and process data
df = fetch_live_data(symbol, timeframe, api_key, api_provider)

if df is not None and len(df) > 0:
    df = calculate_indicators(df, rsi_period, ema_fast, ema_slow)
    signal_type, signals, latest, signal_strength = generate_signals(df, rsi_oversold, rsi_overbought)
    
    # Play alert sound
    play_alert_sound(signal_type)
    
    # Display main signal box
    st.markdown("---")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        signal_class = f"signal-{signal_type.lower().replace(' ', '-')}"
        st.markdown(f'<div class="{signal_class}">🎯 {signal_type}<br/>Strength: {signal_strength}</div>', unsafe_allow_html=True)
    
    with col2:
        change_pct = ((latest['close'] - df.iloc[-2]['close']) / df.iloc[-2]['close'] * 100)
        st.metric("Current Price", f"${latest['close']:.2f}", f"{change_pct:.2f}%")
    
    with col3:
        rsi_color = "🟢" if latest['rsi'] < rsi_oversold else ("🔴" if latest['rsi'] > rsi_overbought else "🟡")
        st.metric("RSI", f"{rsi_color} {latest['rsi']:.1f}")
    
    with col4:
        ema_status = "📈 Bullish" if latest['ema_fast'] > latest['ema_slow'] else "📉 Bearish"
        st.metric("EMA Status", ema_status)
    
    with col5:
        vol_color = "🔥" if latest['vol_ratio'] > 1.5 else "📊"
        st.metric("Volume", f"{vol_color} {latest['vol_ratio']:.2f}x")
    
    # Active signals
    st.markdown("### 🎯 Active Trading Signals")
    cols = st.columns(3)
    for idx, signal in enumerate(signals):
        with cols[idx % 3]:
            st.markdown(f"**{signal}**")
    
    # Trading recommendation
    st.markdown("---")
    st.markdown("### 💡 Trading Recommendation")
    
    if "STRONG BUY" in signal_type:
        st.success(f"""
        **🟢 STRONG BUY SIGNAL DETECTED!**
        
        - Entry: ${latest['close']:.2f}
        - Stop Loss: ${latest['close'] * 0.98:.2f} (-2%)
        - Target 1: ${latest['close'] * 1.02:.2f} (+2%)
        - Target 2: ${latest['close'] * 1.04:.2f} (+4%)
        
        Risk/Reward: 1:2
        """)
    elif "BUY" in signal_type:
        st.success(f"""
        **🟢 BUY SIGNAL**
        
        - Entry: ${latest['close']:.2f}
        - Stop Loss: ${latest['close'] * 0.985:.2f} (-1.5%)
        - Target: ${latest['close'] * 1.03:.2f} (+3%)
        """)
    elif "STRONG SELL" in signal_type:
        st.error(f"""
        **🔴 STRONG SELL SIGNAL DETECTED!**
        
        - Entry: ${latest['close']:.2f}
        - Stop Loss: ${latest['close'] * 1.02:.2f} (+2%)
        - Target 1: ${latest['close'] * 0.98:.2f} (-2%)
        - Target 2: ${latest['close'] * 0.96:.2f} (-4%)
        
        Risk/Reward: 1:2
        """)
    elif "SELL" in signal_type:
        st.error(f"""
        **🔴 SELL SIGNAL**
        
        - Entry: ${latest['close']:.2f}
        - Stop Loss: ${latest['close'] * 1.015:.2f} (+1.5%)
        - Target: ${latest['close'] * 0.97:.2f} (-3%)
        """)
    else:
        st.info("⚪ **NEUTRAL** - Wait for clearer signals. Avoid trading in sideways markets.")
    
    # Chart
    st.plotly_chart(create_advanced_chart(df, ema_fast, ema_slow, rsi_oversold, rsi_overbought, symbol, timeframe), use_container_width=True)
    
    # Market metrics
    st.markdown("---")
    st.markdown("### 📊 Market Statistics")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("High", f"${df['high'].max():.2f}")
    with col2:
        st.metric("Low", f"${df['low'].min():.2f}")
    with col3:
        volatility = ((df['high'] - df['low']) / df['close'] * 100).mean()
        st.metric("Volatility", f"{volatility:.2f}%")
    with col4:
        st.metric("Avg Volume", f"{df['volume'].mean()/1e6:.2f}M")
    with col5:
        st.metric("VWAP", f"${latest['vwap']:.2f}")
    with col6:
        st.metric("Momentum", f"{latest['momentum']:.2f}%")
    
    # Recent data table
    with st.expander("📋 Recent Price Action (Last 10 Candles)"):
        display_df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'rsi', 'ema_fast', 'ema_slow']].tail(10).iloc[::-1]
        display_df['rsi'] = display_df['rsi'].round(1)
        display_df['ema_fast'] = display_df['ema_fast'].round(2)
        display_df['ema_slow'] = display_df['ema_slow'].round(2)
        st.dataframe(display_df, use_container_width=True)

else:
    st.warning("⚠️ Unable to fetch data. Please check your settings.")

# Auto-refresh logic
if st.session_state.auto_refresh:
    time.sleep(30)
    st.rerun()

# Footer
st.markdown("---")
st.markdown(f"**⚠️ Disclaimer:** This tool is for educational purposes only. Always manage your risk and never invest more than you can afford to lose. Current Strategy: **{strategy}**")
