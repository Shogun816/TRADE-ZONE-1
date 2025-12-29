import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta

st.set_page_config(page_title="Gold Dashboard", layout="wide")
st.title("Gold Trading Edge Dashboard (XAUUSD)")
st.markdown("**Real live price** from Yahoo Finance • 5min & 15min charts with volume signals")

@st.cache_data(ttl=60)  # Refresh every minute
def get_gold_data():
    ticker = "GC=F"  # Gold futures (very close to spot XAUUSD)
    try:
        df = yf.download(ticker, period="5d", interval="1m")
        if df.empty:
            st.warning("No data right now - try refreshing")
            return None
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        return df
    except Exception as e:
        st.error(f"Data error: {e}")
        return None

df = get_gold_data()

if df is not None and not df.empty:
    latest_price = df['close'].iloc[-1]
    st.success(f"**Current Real Gold Price: ${latest_price:.2f}** (updated live)")

    # Resample to 5min and 15min
    df_5min = df.resample('5T').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()

    df_15min = df.resample('15T').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()

    # Signals: high volume + strong candle direction
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
                                     low=df['low'], close=df['close'], name="Candles"))
        
        buys = df[df['buy']]
        fig.add_trace(go.Scatter(x=buys.index, y=buys['low']*0.998, mode='markers',
                                 marker=dict(symbol='triangle-up', size=15, color='lime'), name='Buy'))
        
        sells = df[df['sell']]
        fig.add_trace(go.Scatter(x=sells.index, y=sells['high']*1.002, mode='markers',
                                 marker=dict(symbol='triangle-down', size=15, color='red'), name='Sell'))
        
        fig.update_layout(title=title, template="plotly_dark", height=600)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("5-Minute Chart")
    plot_chart(df_5min.tail(120), "XAUUSD 5min - Volume Signals")

    st.subheader("15-Minute Chart")
    plot_chart(df_15min.tail(80), "XAUUSD 15min - Volume Signals")

    last_5 = df_5min.iloc[-1]
    if last_5['buy']:
        st.success("🟢 Strong BUY Signal on 5min!")
    elif last_5['sell']:
        st.warning("🔴 Strong SELL Signal on 5min!")
    else:
        st.info("No strong signal right now")

else:
    st.info("Loading real-time gold data... Refresh in 30 seconds.")

st.caption("Real-time price from Yahoo Finance • Not financial advice • Works 24/5 for gold")
