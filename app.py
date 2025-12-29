import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from polygon import RESTClient
from datetime import datetime, timedelta
import numpy as np

st.title("Gold Trading Edge Dashboard (XAUUSD)")
st.markdown("""
### Volume & Delta Proxy Strategy (Footprint Approximation)
- Fetches live 1-min data from Polygon
- Resamples to 5min & 15min
- Signals: High volume + directional delta → Buy/Sell arrows
- Institutional news summary below
""")

# Fetch data function
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_data(days=3):
    client = RESTClient()  # API key auto-configured or from secrets
    ticker = "C:XAUUSD"
    end = int(datetime.now().timestamp() * 1_000_000)
    start = int((datetime.now() - timedelta(days=days)).timestamp() * 1_000_000)
    aggs = list(client.get_aggs(ticker, 1, "minute", start, end, limit=50000))
    if not aggs:
        st.error("No data fetched. Check Polygon API key.")
        return None
    df = pd.DataFrame(aggs)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume', 'transactions']]
    return df

df_min = fetch_data()

if df_min is not None and not df_min.empty:
    latest_price = df_min['close'].iloc[-1]
    st.success(f"Latest Gold Price: ${latest_price:.2f} (as of {df_min.index[-1]})")

    # Resample
    df_5 = df_min.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'transactions': 'sum'
    }).dropna()

    df_15 = df_min.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'transactions': 'sum'
    }).dropna()

    # Compute signals
    def add_signals(df, timeframe_name):
        df['delta_proxy'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-6)  # Normalized delta
        df['vol_avg'] = df['transactions'].rolling(20).mean()
        df['high_vol'] = df['transactions'] > 1.5 * df['vol_avg']
        
        df['buy_signal'] = (df['delta_proxy'] > 0.3) & df['high_vol'] & (df['close'] > df['high'].shift(1))
        df['sell_signal'] = (df['delta_proxy'] < -0.3) & df['high_vol'] & (df['close'] < df['low'].shift(1))
        
        return df

    df_5 = add_signals(df_5, "5min")
    df_15 = add_signals(df_15, "15min")

    # Plot function
    def plot_chart(df, title):
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index,
                                     open=df['open'], high=df['high'],
                                     low=df['low'], close=df['close'],
                                     name="Candles"))
        
        # Buy arrows
        buy = df[df['buy_signal']]
        fig.add_trace(go.Scatter(x=buy.index, y=buy['low'] * 0.99,
                                 mode='markers', marker_symbol='triangle-up',
                                 marker_color='green', marker_size=15, name='Buy Signal'))
        
        # Sell arrows
        sell = df[df['sell_signal']]
        fig.add_trace(go.Scatter(x=sell.index, y=sell['high'] * 1.01,
                                 mode='markers', marker_symbol='triangle-down',
                                 marker_color='red', marker_size=15, name='Sell Signal'))
        
        fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Price USD",
                          template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("5min Chart & Signals")
    plot_chart(df_5[-100:], "XAUUSD 5min - Volume Delta Signals")  # Last ~8 hours

    st.subheader("15min Chart & Signals")
    plot_chart(df_15[-100:], "XAUUSD 15min - Volume Delta Signals")

    # Current suggestions
    st.subheader("Current Quick Signals")
    last_5 = df_5.iloc[-1]
    if last_5['buy_signal']:
        st.success("🟢 Strong Long Signal on 5min - Consider entry above recent high")
    elif last_5['sell_signal']:
        st.warning("🔴 Strong Short Signal on 5min - Consider entry below recent low")
    else:
        st.info("No strong signal - Wait for volume spike + delta confirmation")

else:
    st.warning("Fetching data... Refresh page in a moment.")

st.subheader("Institutional Gold Activity Summary (Late 2025)")
st.markdown("""
- Central banks added ~53t in October alone, with YTD purchases tracking toward 800-1,000t despite high prices.
- Key buyers: Poland (leading YTD), China (consistent 25-30t/month), India, Kazakhstan.
- Hedge funds & ETFs: Record inflows ($45B+ into gold ETFs), supporting prices above $4,500.
- Overall bullish structural demand amid de-dollarization and geopolitical risks.
""")

st.caption("Data via Polygon (free tier limits apply). Refresh app for updates. Not financial advice - trade at your own risk!")
