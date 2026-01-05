import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import time
import numpy as np

# Try to import required libraries
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import pytz
    NY_TZ = pytz.timezone('America/New_York')
    PYTZ_AVAILABLE = True
except ImportError:
    PYTZ_AVAILABLE = False
    NY_TZ = None

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
    "Indices": ["^GSPC", "^IXIC", "^NDX", "^DJI", "NQ=F", "ES=F", "SPY", "QQQ", "DIA", "IWM", "^VIX"],
    "Commodities": ["GC=F", "CL=F", "NG=F", "ZC=F", "ZS=F"],
    "Metals": ["GC=F", "SI=F", "GOLD", "PL=F", "PA=F", "HG=F"],
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
        
        # Convert to New York timezone if pytz is available
        if 'timestamp' in df.columns and PYTZ_AVAILABLE:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            if df['timestamp'].dt.tz is not None:
                df['timestamp'] = df['timestamp'].dt.tz_convert(NY_TZ)
            else:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(NY_TZ)
        
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        # Get last 50 candles for cleaner chart
        df = df.tail(50)
        
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
        # Generate demo data with timezone if available
        if PYTZ_AVAILABLE:
            dates = pd.date_range(end=datetime.now(NY_TZ), periods=50, freq='5min')
        else:
            dates = pd.date_range(end=datetime.now(), periods=50, freq='5min')
        
        base_price = np.random.uniform(100, 200)
        df = pd.DataFrame({
            'timestamp': dates,
            'open': base_price + np.random.uniform(-5, 5, 50),
            'high': base_price + np.random.uniform(0, 8, 50),
            'low': base_price + np.random.uniform(-8, 0, 50),
            'close': base_price + np.random.uniform(-5, 5, 50),
            'volume': np.random.randint(1000000, 10000000, 50)
        })
        return df


def calculate_indicators(df, rsi_period, ema_fast, ema_slow):
    """Calculate technical indicators with advanced volume analysis"""
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # EMAs
    df['ema_fast'] = df['close'].ewm(span=ema_fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=ema_slow, adjust=False).mean()
    
    # Basic volume analysis
    df['vol_ma'] = df['volume'].rolling(window=20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma']
    
    # VWAP (Volume Weighted Average Price)
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    
    # Price momentum
    df['momentum'] = df['close'].pct_change(periods=5) * 100
    
    # === ADVANCED VOLUME INDICATORS ===
    
    # 1. Volume Price Trend (VPT) - Tracks cumulative volume based on price direction
    df['vpt'] = (df['volume'] * ((df['close'] - df['close'].shift(1)) / df['close'].shift(1))).cumsum()
    
    # 2. Money Flow Index (MFI) - Volume-weighted RSI
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=14).sum()
    
    money_ratio = positive_flow / negative_flow
    df['mfi'] = 100 - (100 / (1 + money_ratio))
    
    # 3. On-Balance Volume (OBV) - Cumulative volume indicator
    obv = []
    obv_value = 0
    for i in range(len(df)):
        if i == 0:
            obv.append(df['volume'].iloc[i])
        else:
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv_value += df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv_value -= df['volume'].iloc[i]
            obv.append(obv_value)
    df['obv'] = obv
    df['obv_ma'] = df['obv'].rolling(window=20).mean()
    
    # 4. Volume Spike Detection (Institutional Activity)
    df['vol_std'] = df['volume'].rolling(window=20).std()
    df['vol_zscore'] = (df['volume'] - df['vol_ma']) / df['vol_std']
    
    # 5. Accumulation/Distribution Line (A/D Line)
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfm = mfm.fillna(0)  # Handle division by zero
    mfv = mfm * df['volume']
    df['ad_line'] = mfv.cumsum()
    df['ad_line_ma'] = df['ad_line'].rolling(window=20).mean()
    
    # 6. Chaikin Money Flow (CMF) - Measures buying/selling pressure
    df['cmf'] = mfv.rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    
    # 7. Volume-Weighted Moving Average (VWMA)
    df['vwma'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    
    # 8. Institutional Order Detection
    # Large candles + high volume = potential institutional activity
    df['candle_size'] = abs(df['close'] - df['open']) / df['open'] * 100
    df['is_large_candle'] = df['candle_size'] > df['candle_size'].rolling(window=20).mean() * 1.5
    df['institutional_buy'] = (df['vol_zscore'] > 2) & (df['close'] > df['open']) & df['is_large_candle']
    df['institutional_sell'] = (df['vol_zscore'] > 2) & (df['close'] < df['open']) & df['is_large_candle']
    
    # 9. Smart Money Index (SMI) - Tracks opening vs closing activity
    # Smart money buys at open, sells at close
    intraday_move = df['close'] - df['open']
    df['smi'] = intraday_move.cumsum()
    df['smi_ma'] = df['smi'].rolling(window=20).mean()
    
    return df


def generate_signals(df, rsi_oversold, rsi_overbought):
    """Generate advanced buy/sell signals with institutional detection"""
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    signals = []
    signal_strength = 0
    institutional_signals = []
    
    # === INSTITUTIONAL ORDER DETECTION ===
    
    # 1. Extreme Volume Spike (Z-score > 3) - BIG PLAYERS
    if latest['vol_zscore'] > 3:
        if latest['close'] > latest['open']:
            institutional_signals.append("🐋 WHALE BUY DETECTED! (Volume Spike 3σ)")
            signal_strength += 3
        else:
            institutional_signals.append("🐋 WHALE SELL DETECTED! (Volume Spike 3σ)")
            signal_strength -= 3
    elif latest['vol_zscore'] > 2:
        if latest['close'] > latest['open']:
            institutional_signals.append("🏦 INSTITUTIONAL BUYING (Volume Spike 2σ)")
            signal_strength += 2
        else:
            institutional_signals.append("🏦 INSTITUTIONAL SELLING (Volume Spike 2σ)")
            signal_strength -= 2
    
    # 2. Money Flow Index (MFI) - Volume-based momentum
    if latest['mfi'] < 20:
        institutional_signals.append("💰 SMART MONEY ACCUMULATING (MFI < 20)")
        signal_strength += 2
    elif latest['mfi'] > 80:
        institutional_signals.append("💰 SMART MONEY DISTRIBUTING (MFI > 80)")
        signal_strength -= 2
    
    # 3. On-Balance Volume (OBV) Divergence
    if latest['obv'] > latest['obv_ma'] * 1.1:
        institutional_signals.append("📊 STRONG ACCUMULATION (OBV Rising)")
        signal_strength += 1.5
    elif latest['obv'] < latest['obv_ma'] * 0.9:
        institutional_signals.append("📊 STRONG DISTRIBUTION (OBV Falling)")
        signal_strength -= 1.5
    
    # 4. Chaikin Money Flow (CMF) - Buying/Selling Pressure
    if latest['cmf'] > 0.2:
        institutional_signals.append("🔥 STRONG BUYING PRESSURE (CMF > 0.2)")
        signal_strength += 1.5
    elif latest['cmf'] < -0.2:
        institutional_signals.append("❄️ STRONG SELLING PRESSURE (CMF < -0.2)")
        signal_strength -= 1.5
    
    # 5. Accumulation/Distribution Line Divergence
    if latest['ad_line'] > latest['ad_line_ma'] * 1.05:
        institutional_signals.append("🏗️ ACCUMULATION PHASE (A/D Rising)")
        signal_strength += 1
    elif latest['ad_line'] < latest['ad_line_ma'] * 0.95:
        institutional_signals.append("🏚️ DISTRIBUTION PHASE (A/D Falling)")
        signal_strength -= 1
    
    # 6. Smart Money Index
    if latest['smi'] > latest['smi_ma']:
        institutional_signals.append("🧠 SMART MONEY BULLISH (SMI Up)")
        signal_strength += 1
    elif latest['smi'] < latest['smi_ma']:
        institutional_signals.append("🧠 SMART MONEY BEARISH (SMI Down)")
        signal_strength -= 1
    
    # 7. Direct Institutional Order Detection
    if latest['institutional_buy']:
        institutional_signals.append("🚨 INSTITUTIONAL BUY ORDER DETECTED!")
        signal_strength += 4
    elif latest['institutional_sell']:
        institutional_signals.append("🚨 INSTITUTIONAL SELL ORDER DETECTED!")
        signal_strength -= 4
    
    # === STANDARD TECHNICAL SIGNALS ===
    
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
    
    # Enhanced Volume Analysis
    if latest['vol_ratio'] > 2:
        signals.append(f"📊 MASSIVE VOLUME ({latest['vol_ratio']:.2f}x) - Big Players Active!")
        signal_strength += 2
    elif latest['vol_ratio'] > 1.5:
        signals.append(f"📊 High Volume ({latest['vol_ratio']:.2f}x)")
        signal_strength += 1
    
    # Price vs VWAP (Weight: 1)
    if latest['close'] > latest['vwap'] * 1.01:
        signals.append("📈 Price Above VWAP (Bullish)")
        signal_strength += 1
    elif latest['close'] < latest['vwap'] * 0.99:
        signals.append("📉 Price Below VWAP (Bearish)")
        signal_strength -= 1
    
    # Price vs VWMA
    if latest['close'] > latest['vwma']:
        signals.append("💹 Above Volume-Weighted MA")
        signal_strength += 0.5
    
    # Momentum
    if latest['momentum'] > 2:
        signals.append(f"🚀 Strong Upward Momentum (+{latest['momentum']:.2f}%)")
        signal_strength += 1
    elif latest['momentum'] < -2:
        signals.append(f"⚠️ Strong Downward Momentum ({latest['momentum']:.2f}%)")
        signal_strength -= 1
    
    # Determine overall signal
    if signal_strength >= 6:
        signal_type = "STRONG BUY"
    elif signal_strength >= 3:
        signal_type = "BUY"
    elif signal_strength <= -6:
        signal_type = "STRONG SELL"
    elif signal_strength <= -3:
        signal_type = "SELL"
    else:
        signal_type = "NEUTRAL"
    
    # Combine all signals
    all_signals = institutional_signals + signals
    
    return signal_type, all_signals, latest, signal_strength, institutional_signals


def create_advanced_chart(df, ema_fast, ema_slow, rsi_oversold, rsi_overbought, symbol, timeframe):
    """Create clean, spaced-out trading chart for real-time analysis"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.55, 0.25, 0.20],
        subplot_titles=(f'{symbol} - {timeframe} - LIVE PRICE ACTION', 'RSI INDICATOR', 'VOLUME')
    )
    
    # Candlestick with better visibility
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price',
        increasing_line_color='#00ff00',
        increasing_fillcolor='#00ff00',
        decreasing_line_color='#ff0000',
        decreasing_fillcolor='#ff0000',
        increasing_line_width=2,
        decreasing_line_width=2
    ), row=1, col=1)
    
    # EMAs with contrasting colors
    fig.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['ema_fast'],
        name=f'Fast EMA {ema_fast}',
        line=dict(color='#00ffff', width=3),
        mode='lines'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['ema_slow'],
        name=f'Slow EMA {ema_slow}',
        line=dict(color='#ffa500', width=3),
        mode='lines'
    ), row=1, col=1)
    
    # VWAP
    fig.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['vwap'],
        name='VWAP',
        line=dict(color='#ff00ff', width=2, dash='dash'),
        mode='lines'
    ), row=1, col=1)
    
    # Mark ONLY significant EMA crossovers (reduce clutter)
    crossovers = []
    for i in range(1, len(df)):
        # Bullish crossover
        if df['ema_fast'].iloc[i] > df['ema_slow'].iloc[i] and df['ema_fast'].iloc[i-1] <= df['ema_slow'].iloc[i-1]:
            # Check if volume supports the signal
            if df['vol_ratio'].iloc[i] > 1.2:
                fig.add_annotation(
                    x=df['timestamp'].iloc[i],
                    y=df['low'].iloc[i] * 0.998,
                    text="🟢 BUY",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="lime",
                    arrowsize=1.5,
                    arrowwidth=3,
                    ax=0,
                    ay=50,
                    font=dict(color="lime", size=14, family="Arial Black"),
                    bgcolor="rgba(0,255,0,0.3)",
                    bordercolor="lime",
                    borderwidth=2,
                    row=1, col=1
                )
        # Bearish crossover
        elif df['ema_fast'].iloc[i] < df['ema_slow'].iloc[i] and df['ema_fast'].iloc[i-1] >= df['ema_slow'].iloc[i-1]:
            if df['vol_ratio'].iloc[i] > 1.2:
                fig.add_annotation(
                    x=df['timestamp'].iloc[i],
                    y=df['high'].iloc[i] * 1.002,
                    text="🔴 SELL",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="red",
                    arrowsize=1.5,
                    arrowwidth=3,
                    ax=0,
                    ay=-50,
                    font=dict(color="red", size=14, family="Arial Black"),
                    bgcolor="rgba(255,0,0,0.3)",
                    bordercolor="red",
                    borderwidth=2,
                    row=1, col=1
                )
    
    # RSI with cleaner zones
    fig.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['rsi'],
        name='RSI',
        line=dict(color='#ffff00', width=4),
        fill='tonexty',
        mode='lines'
    ), row=2, col=1)
    
    # RSI reference lines
    fig.add_hline(y=rsi_overbought, line_dash="solid", line_color="red", line_width=3, 
                  annotation_text=f"Overbought {rsi_overbought}", annotation_position="right",
                  row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", line_width=1, row=2, col=1)
    fig.add_hline(y=rsi_oversold, line_dash="solid", line_color="lime", line_width=3,
                  annotation_text=f"Oversold {rsi_oversold}", annotation_position="right",
                  row=2, col=1)
    
    # Highlight RSI danger zones
    fig.add_hrect(y0=rsi_overbought, y1=100, fillcolor="red", opacity=0.15, 
                  annotation_text="SELL ZONE", annotation_position="top left",
                  row=2, col=1)
    fig.add_hrect(y0=0, y1=rsi_oversold, fillcolor="green", opacity=0.15,
                  annotation_text="BUY ZONE", annotation_position="bottom left",
                  row=2, col=1)
    
    # Volume with gradient colors
    volume_colors = []
    for i in range(len(df)):
        if df['close'].iloc[i] >= df['open'].iloc[i]:
            # Green candle - check strength
            if df['vol_ratio'].iloc[i] > 1.5:
                volume_colors.append('lime')  # High volume green
            else:
                volume_colors.append('green')  # Normal green
        else:
            # Red candle - check strength
            if df['vol_ratio'].iloc[i] > 1.5:
                volume_colors.append('#ff0000')  # High volume red
            else:
                volume_colors.append('#cc0000')  # Normal red
    
    fig.add_trace(go.Bar(
        x=df['timestamp'], 
        y=df['volume'],
        name='Volume',
        marker_color=volume_colors,
        showlegend=False,
        opacity=0.7
    ), row=3, col=1)
    
    # Volume MA line
    fig.add_trace(go.Scatter(
        x=df['timestamp'], 
        y=df['vol_ma'],
        name='Avg Volume',
        line=dict(color='yellow', width=2, dash='dash'),
        mode='lines'
    ), row=3, col=1)
    
    # Layout updates for better spacing
    fig.update_layout(
        title=dict(
            text=f'{symbol} - {timeframe} - Real-Time Trading Chart (NY Time)',
            font=dict(size=26, color='white', family='Arial Black'),
            x=0.5,
            xanchor='center'
        ),
        template='plotly_dark',
        height=1000,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12)
        ),
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        plot_bgcolor='#0a0a0a',
        paper_bgcolor='#0e1117',
        margin=dict(l=80, r=80, t=100, b=80)
    )
    
    # Better axis formatting - apply to all x-axes
    fig.update_xaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='#333333',
        tickformat='%I:%M %p'  # 12-hour format with AM/PM
    )
    
    # Update y-axes with proper formatting
    fig.update_yaxes(title_text="Price ($)", row=1, col=1, showgrid=True, gridwidth=1, gridcolor='#333333')
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100], showgrid=True, gridwidth=1, gridcolor='#333333')
    fig.update_yaxes(title_text="Volume", row=3, col=1, showgrid=True, gridwidth=1, gridcolor='#333333')
    
    return fig


def play_alert_sound(signal_type):
    """Display voice alert notification prominently"""
    if st.session_state.sound_enabled and signal_type != st.session_state.last_signal:
        
        if "STRONG BUY" in signal_type:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #00ff00 0%, #00cc00 100%); 
                        padding: 20px; 
                        border-radius: 15px; 
                        text-align: center;
                        animation: pulse 1s infinite;
                        box-shadow: 0 0 30px #00ff00;
                        margin: 20px 0;'>
                <div style='font-size: 32px; font-weight: bold; color: black;'>
                    🔊 VOICE ALERT 🔊
                </div>
                <div style='font-size: 24px; color: black; margin-top: 10px;'>
                    "STRONG BUY SIGNAL DETECTED!"
                </div>
                <div style='font-size: 20px; color: #003300; margin-top: 5px;'>
                    "ENTER LONG POSITION NOW!"
                </div>
            </div>
            <style>
            @keyframes pulse {
                0%, 100% { opacity: 1; transform: scale(1); }
                50% { opacity: 0.8; transform: scale(1.02); }
            }
            </style>
            """, unsafe_allow_html=True)
            
        elif "BUY" in signal_type:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #00ff00 0%, #66ff66 100%); 
                        padding: 20px; 
                        border-radius: 15px; 
                        text-align: center;
                        box-shadow: 0 0 20px #00ff00;
                        margin: 20px 0;'>
                <div style='font-size: 28px; font-weight: bold; color: black;'>
                    🔊 VOICE ALERT 🔊
                </div>
                <div style='font-size: 22px; color: black; margin-top: 10px;'>
                    "BUY SIGNAL!"
                </div>
                <div style='font-size: 18px; color: #003300; margin-top: 5px;'>
                    "GOOD ENTRY OPPORTUNITY!"
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        elif "STRONG SELL" in signal_type:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #ff0000 0%, #cc0000 100%); 
                        padding: 20px; 
                        border-radius: 15px; 
                        text-align: center;
                        animation: pulse 1s infinite;
                        box-shadow: 0 0 30px #ff0000;
                        margin: 20px 0;'>
                <div style='font-size: 32px; font-weight: bold; color: white;'>
                    🔊 VOICE ALERT 🔊
                </div>
                <div style='font-size: 24px; color: white; margin-top: 10px;'>
                    "STRONG SELL SIGNAL!"
                </div>
                <div style='font-size: 20px; color: #ffcccc; margin-top: 5px;'>
                    "EXIT POSITION IMMEDIATELY!"
                </div>
            </div>
            <style>
            @keyframes pulse {
                0%, 100% { opacity: 1; transform: scale(1); }
                50% { opacity: 0.8; transform: scale(1.02); }
            }
            </style>
            """, unsafe_allow_html=True)
            
        elif "SELL" in signal_type:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #ff0000 0%, #ff6666 100%); 
                        padding: 20px; 
                        border-radius: 15px; 
                        text-align: center;
                        box-shadow: 0 0 20px #ff0000;
                        margin: 20px 0;'>
                <div style='font-size: 28px; font-weight: bold; color: white;'>
                    🔊 VOICE ALERT 🔊
                </div>
                <div style='font-size: 22px; color: white; margin-top: 10px;'>
                    "SELL SIGNAL DETECTED!"
                </div>
                <div style='font-size: 18px; color: #ffcccc; margin-top: 5px;'>
                    "CONSIDER TAKING PROFITS!"
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Also play a browser beep sound
        st.markdown("""
            <audio autoplay>
                <source src="data:audio/wav;base64,UklGRpYAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YXIAAACAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIA=" type="audio/wav">
            </audio>
        """, unsafe_allow_html=True)
        
        st.session_state.last_signal = signal_type


# Main content
st.markdown(f"<h1 style='text-align: center; color: {preset['color']};'>📊 {strategy.upper()}</h1>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown(f"## {symbol}")

with col2:
    if PYTZ_AVAILABLE:
        ny_time = datetime.now(NY_TZ)
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 15px; 
                    border-radius: 10px; 
                    text-align: center;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.3);'>
            <div style='font-size: 24px; font-weight: bold; color: white;'>
                🕐 {ny_time.strftime('%I:%M:%S %p')}
            </div>
            <div style='font-size: 14px; color: #e0e0e0; margin-top: 5px;'>
                New York Time • {ny_time.strftime('%b %d, %Y')}
            </div>
            <div style='font-size: 12px; color: #ffd700; margin-top: 3px;'>
                {ny_time.strftime('%A')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        current_time = datetime.now()
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 15px; 
                    border-radius: 10px; 
                    text-align: center;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.3);'>
            <div style='font-size: 24px; font-weight: bold; color: white;'>
                🕐 {current_time.strftime('%I:%M:%S %p')}
            </div>
            <div style='font-size: 14px; color: #e0e0e0; margin-top: 5px;'>
                Local Time • {current_time.strftime('%b %d, %Y')}
            </div>
            <div style='font-size: 12px; color: #ff6b6b;'>
                ⚠️ Install pytz for NY time
            </div>
        </div>
        """, unsafe_allow_html=True)

with col3:
    status = "🟢 LIVE" if st.session_state.auto_refresh else "⚪ PAUSED"
    st.markdown(f"""
    <div style='background: {'linear-gradient(135deg, #11998e 0%, #38ef7d 100%)' if st.session_state.auto_refresh else 'linear-gradient(135deg, #606c88 0%, #3f4c6b 100%)'}; 
                padding: 15px; 
                border-radius: 10px; 
                text-align: center;
                box-shadow: 0 4px 6px rgba(0,0,0,0.3);'>
        <div style='font-size: 20px; font-weight: bold; color: white;'>
            {status}
        </div>
        <div style='font-size: 12px; color: #e0e0e0; margin-top: 5px;'>
            {'Auto-updating every 30s' if st.session_state.auto_refresh else 'Manual refresh only'}
        </div>
    </div>
    """, unsafe_allow_html=True)

# Fetch and process data
df = fetch_live_data(symbol, timeframe, api_key, api_provider)

if df is not None and len(df) > 0:
    df = calculate_indicators(df, rsi_period, ema_fast, ema_slow)
    signal_type, all_signals, latest, signal_strength, institutional_signals = generate_signals(df, rsi_oversold, rsi_overbought)
    
    # Play alert sound and show voice notification
    play_alert_sound(signal_type)
    
    # Display main signal box
    st.markdown("---")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        signal_class = f"signal-{signal_type.lower().replace(' ', '-')}"
        st.markdown(f'<div class="{signal_class}">🎯 {signal_type}<br/>Strength: {signal_strength:.1f}</div>', unsafe_allow_html=True)
    
    with col2:
        change_pct = ((latest['close'] - df.iloc[-2]['close']) / df.iloc[-2]['close'] * 100)
        st.metric("Current Price", f"${latest['close']:.2f}", f"{change_pct:.2f}%")
    
    with col3:
        rsi_color = "🟢" if latest['rsi'] < rsi_oversold else ("🔴" if latest['rsi'] > rsi_overbought else "🟡")
        st.metric("RSI", f"{rsi_color} {latest['rsi']:.1f}")
    
    with col4:
        # MFI indicator
        mfi_color = "🟢" if latest['mfi'] < 30 else ("🔴" if latest['mfi'] > 70 else "🟡")
        st.metric("MFI (Money Flow)", f"{mfi_color} {latest['mfi']:.1f}")
    
    with col5:
        vol_color = "🔥" if latest['vol_ratio'] > 2 else ("📊" if latest['vol_ratio'] > 1.5 else "📉")
        st.metric("Volume", f"{vol_color} {latest['vol_ratio']:.2f}x")
    
    # === INSTITUTIONAL ACTIVITY ALERTS ===
    if institutional_signals:
        st.markdown("---")
        st.markdown("""
        <div style='background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%); 
                    padding: 20px; 
                    border-radius: 15px; 
                    border: 3px solid #ff0000;
                    animation: glow 2s infinite;
                    margin: 20px 0;'>
            <div style='font-size: 28px; font-weight: bold; color: white; text-align: center;'>
                🏦 INSTITUTIONAL ACTIVITY DETECTED 🏦
            </div>
            <div style='font-size: 16px; color: white; text-align: center; margin-top: 10px;'>
                BIG PLAYERS (CENTRAL BANKS, HEDGE FUNDS, WHALES) ARE ACTIVE!
            </div>
        </div>
        <style>
        @keyframes glow {
            0%, 100% { box-shadow: 0 0 20px #ff6b6b; }
            50% { box-shadow: 0 0 40px #feca57, 0 0 60px #ff6b6b; }
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🐋 WHALE & INSTITUTIONAL SIGNALS:")
        inst_cols = st.columns(2)
        for idx, signal in enumerate(institutional_signals):
            with inst_cols[idx % 2]:
                st.markdown(f"**{signal}**")
    
    # Active signals
    st.markdown("---")
    st.markdown("### 🎯 All Active Trading Signals")
    cols = st.columns(3)
    for idx, signal in enumerate(all_signals):
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
    
    # Market metrics with institutional indicators
    st.markdown("---")
    st.markdown("### 📊 Market Statistics & Institutional Metrics")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("24h High", f"${df['high'].max():.2f}")
    with col2:
        st.metric("24h Low", f"${df['low'].min():.2f}")
    with col3:
        volatility = ((df['high'] - df['low']) / df['close'] * 100).mean()
        st.metric("Volatility", f"{volatility:.2f}%")
    with col4:
        st.metric("Avg Volume", f"{df['volume'].mean()/1e6:.2f}M")
    with col5:
        st.metric("VWAP", f"${latest['vwap']:.2f}")
    with col6:
        st.metric("Momentum", f"{latest['momentum']:.2f}%")
    
    # Institutional metrics row
    st.markdown("#### 🏦 Institutional Indicators")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        cmf_status = "🟢 Buying" if latest['cmf'] > 0 else "🔴 Selling"
        st.metric("Chaikin MF", f"{cmf_status}", f"{latest['cmf']:.3f}")
    with col2:
        obv_trend = "📈" if latest['obv'] > latest['obv_ma'] else "📉"
        st.metric("OBV Trend", obv_trend)
    with col3:
        ad_trend = "📈 Accumulation" if latest['ad_line'] > latest['ad_line_ma'] else "📉 Distribution"
        st.metric("A/D Line", ad_trend)
    with col4:
        smi_status = "🧠 Bullish" if latest['smi'] > latest['smi_ma'] else "🧠 Bearish"
        st.metric("Smart Money", smi_status)
    with col5:
        vol_zscore_status = "🐋 WHALE!" if abs(latest['vol_zscore']) > 3 else ("🏦 High" if abs(latest['vol_zscore']) > 2 else "📊 Normal")
        st.metric("Volume Z-Score", vol_zscore_status, f"{latest['vol_zscore']:.2f}σ")
    with col6:
        institutional_count = sum([latest['institutional_buy'], latest['institutional_sell']])
        inst_status = "🚨 ACTIVE!" if institutional_count > 0 else "😴 Quiet"
        st.metric("Institutional Orders", inst_status)
    
    # Recent data table with institutional indicators
    with st.expander("📋 Recent Price Action & Institutional Activity (Last 10 Candles)"):
        display_df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'rsi', 'mfi', 'cmf', 'vol_zscore', 'institutional_buy', 'institutional_sell']].tail(10).iloc[::-1]
        display_df['rsi'] = display_df['rsi'].round(1)
        display_df['mfi'] = display_df['mfi'].round(1)
        display_df['cmf'] = display_df['cmf'].round(3)
        display_df['vol_zscore'] = display_df['vol_zscore'].round(2)
        display_df['volume'] = (display_df['volume'] / 1e6).round(2)
        display_df = display_df.rename(columns={
            'volume': 'Vol (M)',
            'vol_zscore': 'Vol Z-Score',
            'institutional_buy': 'Inst Buy 🐋',
            'institutional_sell': 'Inst Sell 🐋'
        })
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
