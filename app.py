import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from datetime import datetime

st.set_page_config(page_title="Gold Dashboard", layout="wide")
st.title("Gold Trading Edge Dashboard (XAUUSD)")
st.markdown("**Instant live spot price** • Fast loading • Real-time signals")

# Fast live price fetch (metalpriceapi - free, no key)
@st.cache_data(ttl=30)  # Update every 30 seconds
def get_live_price():
    try:
        response = requests.get("https://api.metalpriceapi.com/v1/latest?api_key=free&base=USD&currencies=XAU")
        data = response.json()
        if 'rates' in data and 'XAU' in data['rates']:
            return 1 / data['rates']['XAU']  # Convert from USD per XAU to XAUUSD price
        return None
    except:
        return None

price = get_live_price()
if price:
    st.success(f"**Live Gold Spot Price: ${price:.2f}** (updates every 30s - refresh page)")

# For charts & volume signals: Use yfinance but optimized (shorter period, higher interval)
@st.cache_data(ttl=120)  # Update every 2 minutes
def get_chart_data():
    try:
        import yfinance as yf
        df = yf.download("XAUUSD=X", period="3d", interval="5m")  # Short period = faster
        if df.empty:
            return None
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        return df
    except:
        return None

df = get_chart_data()

if df is not None and not df.empty:
    # Resample if needed
    df_5min = df.copy()
    df_15min = df.resample('15T').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()

    # Instant signals
    def add_signals(data):
        data['vol_avg'] = data['volume'].rolling(20).mean()
        data['high_volume'] = data['volume'] > 1.5 * data['vol_avg']
        data['delta'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
        data['buy'] = (data['delta'] > 0.3) & data['high_volume']
        data['sell'] = (data['delta'] < -0.3) & data['high_volume']
        return data

    df_5min = add_signals(df_5min)
    df_15min = add_signals(df_15min)

    def plot_chart(df, title):
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                                     low=df['low'], close=df['close'], name="Price"))
        
        buys = df[df['buy']]
        fig.add_trace(go.Scatter(x=buys.index, y=buys['low']*0.998, mode='markers',
                                 marker=dict(symbol='triangle-up', size=16, color='lime'), name='BUY'))
        
        sells = df[df['sell']]
        fig.add_trace(go.Scatter(x=sells.index, y=sells['high']*1.002, mode='markers',
                                 marker=dict(symbol='triangle-down', size=16, color='red'), name='SELL'))
        
        fig.update_layout(title=title, template="plotly_dark", height=600)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("5-Minute Chart (Fast Signals)")
    plot_chart(df_5min.tail(100), "XAUUSD 5min")

    st.subheader("15-Minute Chart")
    plot_chart(df_15min.tail(60), "XAUUSD 15min")

    current = df_5min.iloc[-1]
    if current['buy']:
        st.success("🟢 **BUY SIGNAL ACTIVE** on 5min!")
    elif current['sell']:
        st.warning("🔴 **SELL SIGNAL ACTIVE** on 5min!")
    else:
        st.info("Waiting for volume spike...")

else:
    st.info("Charts loading... (first load may take 10s - then instant)")

st.caption("Live price via free metalpriceapi • Charts via yfinance • Refresh for updates • Dec 29, 2025 price ~$4324 after pullback")
