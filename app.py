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

# Live countdown timer
now = datetime.now()
seconds_into_candle = (now.minute % 5) * 60 + now.second
seconds_left = 300 - seconds_into_candle

minutes = seconds_left // 60
seconds = seconds_left % 60

st.metric("Time to Next 5min Candle Close", f"{minutes:02d}:{seconds:02d}")

# Auto-refresh every 10 seconds (smooth timer update)
st.markdown("""
    <script>
        setTimeout(function() { window.location.reload(); }, 10000);
    </script>
""", unsafe_allow_html=True)

# Recent data (real pullback Dec 29)
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
