import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="Gold Dashboard", layout="wide")
st.title("Gold Trading Edge Dashboard (XAUUSD)")

# Live countdown timer
now = datetime.now()
seconds_into_candle = (now.minute % 5) * 60 + now.second
seconds_left = 300 - seconds_into_candle
minutes = seconds_left // 60
seconds = seconds_left % 60
st.metric("Time to Next 5min Candle Close", f"{minutes:02d}:{seconds:02d}")

# Auto-refresh every 10 seconds
st.markdown("<script>setTimeout(function(){window.location.reload();}, 10000);</script>", unsafe_allow_html=True)

# Real live data
@st.cache_data(ttl=30)
def get_live_data():
    try:
        df = yf.download("XAUUSD=X", period="3d", interval="1m")
        if df.empty:
            return None
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        return df
    except:
        return None

df = get_live_data()

if df is not None and not df.empty:
    latest_price = df['close'].iloc[-1]
    st.success(f"**Live Gold Price: ${latest_price:.2f}** (real-time)")

    df_5min = df.resample('5T').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()

    df_15min = df.resample('15T').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()

    def add_signals(data):
        data['vol_avg'] = data['volume'].rolling(20).mean()
        data['high_volume'] = data['volume'] > 1.5 * data['vol_avg']
        data['very_high_volume'] = data['volume'] > 2.0 * data['vol_avg']
        data['delta'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
        data['buy'] = (data['delta'] > 0.3) & data['high_volume']
        data['sell'] = (data['delta'] < -0.3) & data['high_volume']
        data['strong_buy'] = (data['delta'] > 0.5) & data['very_high_volume']
        data['strong_sell'] = (data['delta'] < -0.5) & data['very_high_volume']
        return data

    df_5min = add_signals(df_5min)
    df_15min = add_signals(df_15min)

    current = df_5min.iloc[-1]
    if current['strong_buy'] or current['strong_sell']:
        st.audio("https://assets.mixkit.co/sfx/preview/mixkit-alarm-digital-clock-beep-989.mp3", autoplay=True)

    def plot_chart(df, title):
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                                     low=df['low'], close=df['close'], name="Price"))

        buys = df[df['buy'] & ~df['strong_buy']]
        sells = df[df['sell'] & ~df['strong_sell']]
        fig.add_trace(go.Scatter(x=buys.index, y=buys['low']*0.998, mode='markers',
                                 marker=dict(symbol='triangle-up', size=14, color='green'), name='Buy'))
        fig.add_trace(go.Scatter(x=sells.index, y=sells['high']*1.002, mode='markers',
                                 marker=dict(symbol='triangle-down', size=14, color='red'), name='Sell'))

        strong_buys = df[df['strong_buy']]
        strong_sells = df[df['strong_sell']]
        fig.add_trace(go.Scatter(x=strong_buys.index, y=strong_buys['low']*0.995, mode='markers+text',
                                 marker=dict(symbol='triangle-up', size=32, color='lime'),
                                 text=["STRONG BUY!"], textposition="bottom center", textfont=dict(size=16)))
        fig.add_trace(go.Scatter(x=strong_sells.index, y=strong_sells['high']*1.005, mode='markers+text',
                                 marker=dict(symbol='triangle-down', size=32, color='red'),
                                 text=["STRONG SELL!"], textposition="top center", textfont=dict(size=16)))

        fig.update_layout(title=title, template="plotly_dark", height=600)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("5-Minute Live Chart")
    plot_chart(df_5min.tail(120), "XAUUSD 5min (real-time)")

    st.subheader("15-Minute Live Chart")
    plot_chart(df_15min.tail(80), "XAUUSD 15min")

    if current['strong_buy']:
        st.success("🔊 VERY STRONG BUY!")
    elif current['strong_sell']:
        st.warning("🔊 VERY STRONG SELL!")
    elif current['buy']:
        st.success("🟢 Buy Signal")
    elif current['sell']:
        st.warning("🔴 Sell Signal")

else:
    st.info("Loading real-time data... (first load may take 10–20s, then fast)")

st.caption("Live countdown + real-time candles • Refresh or wait 10s for timer update • Gold ~$4341 today")
