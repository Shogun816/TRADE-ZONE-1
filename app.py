import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from polygon.rest import RESTClient  # This is the correct import
from datetime import datetime

st.set_page_config(page_title="Gold Trading Dashboard", layout="wide")
st.title("Gold Trading Edge Dashboard (XAUUSD)")
st.markdown("Live 5min & 15min charts with volume-based buy/sell signals")

# Check for API key
if "POLYGON_API_KEY" not in st.secrets:
    st.error("⚠️ Please add your Polygon API key in Settings > Secrets as: POLYGON_API_KEY = \"your_key_here\"")
    st.info("Get a free key at: https://polygon.io")
    st.stop()

# Create client with your secret key
client = RESTClient(st.secrets["POLYGON_API_KEY"])

@st.cache_data(ttl=180)  # Refresh every 3 minutes
def get_gold_data():
    try:
        # Fetch recent 1-minute bars for gold
        aggs = client.get_aggs("C:XAUUSD", 1, "minute", limit=50000)
        if not aggs:
            st.warning("No data available right now (markets may be slow)")
            return None
        
        df = pd.DataFrame(aggs)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'transactions']]  # Use transactions as volume proxy
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = get_gold_data()

if df is not None and not df.empty:
    latest_price = df['close'].iloc[-1]
    st.success(f"Current Gold Price: ${latest_price:.2f}")

    # Resample to 5min and 15min
    df_5min = df.resample('5T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'transactions': 'sum'
    }).dropna()

    df_15min = df.resample('15T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'transactions': 'sum'
    }).dropna()

    # Add trading signals
    def add_signals(data):
        data['vol_avg'] = data['transactions'].rolling(20).mean()
        data['high_volume'] = data['transactions'] > 1.5 * data['vol_avg']
        data['delta'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
        data['buy'] = (data['delta'] > 0.3) & data['high_volume']
        data['sell'] = (data['delta'] < -0.3) & data['high_volume']
        return data

    df_5min = add_signals(df_5min)
    df_15min = add_signals(df_15min)

    # Plot function
    def plot_chart(df, title):
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Price"
        ))

        # Buy signals (green triangles)
        buys = df[df['buy']]
        fig.add_trace(go.Scatter(
            x=buys.index,
            y=buys['low'] * 0.998,
            mode='markers',
            marker=dict(symbol='triangle-up', size=15, color='lime'),
            name='Buy Signal'
        ))

        # Sell signals (red triangles)
        sells = df[df['sell']]
        fig.add_trace(go.Scatter(
            x=sells.index,
            y=sells['high'] * 1.002,
            mode='markers',
            marker=dict(symbol='triangle-down', size=15, color='red'),
            name='Sell Signal'
        ))

        fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Price (USD)", template="plotly_dark", height=600)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("5-Minute Chart")
    plot_chart(df_5min.tail(100), "XAUUSD 5min - Volume Signals")

    st.subheader("15-Minute Chart")
    plot_chart(df_15min.tail(60), "XAUUSD 15min - Volume Signals")

    # Latest signal alert
    last_5 = df_5min.iloc[-1]
    if last_5['buy']:
        st.success("🟢 Strong BUY Signal on 5min Chart!")
    elif last_5['sell']:
        st.warning("🔴 Strong SELL Signal on 5min Chart!")
    else:
        st.info("No strong signal right now — waiting for volume + momentum")

else:
    st.info("Loading live data... Refresh in 30 seconds.")

st.caption("Not financial advice • Data via Polygon.io • Gold spot price (XAUUSD)")
