import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from polygon.rest import RESTClient
from datetime import datetime, timedelta

st.set_page_config(page_title="Gold Dashboard", layout="wide")
st.title("Gold Trading Edge Dashboard (XAUUSD)")
st.markdown("Live 5min & 15min charts with volume-based buy/sell signals • Refresh for updates")

# API Key check
if "POLYGON_API_KEY" not in st.secrets:
    st.error("Add your Polygon API key in Settings > Secrets (POLYGON_API_KEY = \"your_key\")")
    st.stop()

client = RESTClient(st.secrets["POLYGON_API_KEY"])

@st.cache_data(ttl=180)  # Update every 3 minutes
def get_gold_data():
    ticker = "C:XAUUSD"
    try:
        # Set date range: last 3 days to now (forex/gold is 24/5, so plenty of data)
        to_date = datetime.now()
        from_date = to_date - timedelta(days=3)
        
        # Fetch aggregates - now with required from_ and to
        aggs = client.get_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="minute",
            from_=from_date.date(),  # Start date
            to=to_date.date(),       # End date
            limit=50000
        )
        
        if not aggs:
            st.warning("No data returned (markets may be quiet today - Dec 29)")
            return None
        
        df = pd.DataFrame(aggs)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'transactions']]  # Transactions = volume proxy
        return df
    except Exception as e:
        st.error(f"Data error: {e}")
        return None

df = get_gold_data()

if df is not None and not df.empty:
    latest_price = df['close'].iloc[-1]
    st.success(f"Current Gold Price: ${latest_price:.2f}")

    # Resample to 5min and 15min
    df_5min = df.resample('5T').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'transactions': 'sum'
    }).dropna()

    df_15min = df.resample('15T').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'transactions': 'sum'
    }).dropna()

    # Add signals (volume spike + strong directional move)
    def add_signals(data):
        data['vol_avg'] = data['transactions'].rolling(20).mean()
        data['high_volume'] = data['transactions'] > 1.5 * data['vol_avg']
        data['delta'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
        data['buy'] = (data['delta'] > 0.3) & data['high_volume']
        data['sell'] = (data['delta'] < -0.3) & data['high_volume']
        return data

    df_5min = add_signals(df_5min)
    df_15min = add_signals(df_15min)

    # Plot charts
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

    # Current signal
    last_5 = df_5min.iloc[-1]
    if last_5['buy']:
        st.success("🟢 Strong BUY Signal on 5min!")
    elif last_5['sell']:
        st.warning("🔴 Strong SELL Signal on 5min!")
    else:
        st.info("No strong signal right now")

else:
    st.info("Fetching live data... Refresh page in 30 seconds.")

st.caption("Not financial advice • Free Polygon tier • Data for XAUUSD (Gold spot)")
