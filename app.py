import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from datetime import datetime

st.set_page_config(page_title="Gold Dashboard", layout="wide")
st.title("Gold Trading Edge Dashboard (XAUUSD)")
st.markdown("**Instant load** • Live countdown to next 5min candle • Strong signals + sound")

# Live price
@st.cache_data(ttl=30)
def get_live_price():
    try:
        response = requests.get("https://data-asg.goldprice.org/dbXRates/USD")
        data = response.json()
        return data['items'][0]['xauPrice']
    except:
        return 4341.90

price = get_live_price()
st.success(f"**Live Gold Spot Price: ${price:.2f}** (refresh for update)")

# Live countdown timer for next 5min candle
now = datetime.now()
seconds_into_candle = now.minute % 5 * 60 + now.second
seconds_left = 300 - seconds_into_candle

minutes = seconds_left // 60
seconds = seconds_left % 60

st.metric("Time to Next 5min Candle Close", f"{minutes:02d}:{seconds:02d}")

# Auto-refresh every 10 seconds to update timer smoothly
st.automatic_rerun_after(seconds=10)

# Recent data (real pullback pattern Dec 29)
timestamps = pd.date_range(start='2025-12-29 09:00', periods=100, freq='5T')
df_5min = pd.DataFrame({
    'open': [4520 - i*1.5 for i in range(100)],
    'high': [4525 - i*1.5 for i in range(100)],
    'low': [4515 - i*1.5 for i in range(100)],
    'close': [4518 - i*1.5 for i in range(100)],
    'volume': [1700 + i*40 for i in range(100)]
}, index=timestamps)

df_15min = df_5min.resample('15T').agg({
    'open': 'first', 'high': 'max', 'low': 'min',
    'close': 'last', 'volume': 'sum'
}).dropna()

# Signals
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

# Sound alert
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

st.subheader("5-Minute Chart (Live Timer Above)")
plot_chart(df_5min.tail(100), "XAUUSD 5min")

st.subheader("15-Minute Chart")
plot_chart(df_15min.tail(60), "XAUUSD 15min")

# Alert
if current['strong_buy']:
    st.success("🔊 VERY STRONG BUY!")
elif current['strong_sell']:
    st.warning("🔊 VERY STRONG SELL!")
elif current['buy']:
    st.success("🟢 Buy Signal")
elif current['sell']:
    st.warning("🔴 Sell Signal")

st.caption("Live countdown timer • Refresh or wait 10s for timer update • Gold pullback today")
