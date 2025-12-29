import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="Gold Dashboard", layout="wide")
st.title("Gold Trading Edge Dashboard (XAUUSD)")
st.markdown("**Accurate live spot price** • Real-time signals (trigger instantly) • Refresh often")

@st.cache_data(ttl=60)  # Update every minute
def get_gold_data():
    ticker = "XAUUSD=X"  # True spot gold price (accurate real-time)
    try:
        df = yf.download(ticker, period="5d", interval="5m")  # 5min native for better speed
        if df.empty:
            return None
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        return df
    except:
        return None

df = get_gold_data()

if df is not None and not df.empty:
    latest_price = df['close'].iloc[-1]
    latest_time = df.index[-1].strftime('%Y-%m-%d %H:%M')  # Your local time
    st.success(f"**Live Gold Spot Price: ${latest_price:.2f}** at {latest_time}")

    # Use 5min data directly (faster, more accurate timing)
    df_5min = df.copy()

    # Resample to 15min for second chart
    df_15min = df.resample('15T').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()

    # Fixed real-time signals (trigger on current bar)
    def add_signals(data):
        data['vol_avg'] = data['volume'].rolling(20).mean()
        data['high_volume'] = data['volume'] > 1.5 * data['vol_avg']
        data['delta'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
        
        # Real-time: Check if high/low broken during current bar
        data['buy'] = (data['delta'] > 0.3) & data['high_volume'] & (data['high'] > data['high'].shift(1))
        data['sell'] = (data['delta'] < -0.3) & data['high_volume'] & (data['low'] < data['low'].shift(1))
        
        return data

    df_5min = add_signals(df_5min)
    df_15min = add_signals(df_15min)

    def plot_chart(df, title):
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                                     low=df['low'], close=df['close'], name="Price"))
        
        # Buy arrows on current bar
        buys = df[df['buy']]
        fig.add_trace(go.Scatter(x=buys.index, y=buys['low']*0.998, mode='markers',
                                 marker=dict(symbol='triangle-up', size=16, color='lime'), name='BUY NOW'))
        
        # Sell arrows on current bar
        sells = df[df['sell']]
        fig.add_trace(go.Scatter(x=sells.index, y=sells['high']*1.002, mode='markers',
                                 marker=dict(symbol='triangle-down', size=16, color='red'), name='SELL NOW'))
        
        fig.update_layout(title=title, template="plotly_dark", height=600,
                          xaxis_title="Time (Your Local)", yaxis_title="Price USD")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("5-Minute Chart (Real-Time Signals)")
    plot_chart(df_5min.tail(100), "XAUUSD 5min - Instant Volume Signals")

    st.subheader("15-Minute Chart")
    plot_chart(df_15min.tail(60), "XAUUSD 15min")

    # Instant alert for current bar
    current = df_5min.iloc[-1]
    if current['buy']:
        st.success("🟢 **REAL-TIME BUY SIGNAL ACTIVE NOW** on 5min!")
    elif current['sell']:
        st.warning("🔴 **REAL-TIME SELL SIGNAL ACTIVE NOW** on 5min!")
    else:
        st.info("Watching for volume spike...")

else:
    st.info("Loading live spot data... Refresh in 30 seconds")

st.caption("Accurate spot price via Yahoo Finance • Signals trigger instantly • Dec 29, 2025")
