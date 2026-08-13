import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from stable_baselines3 import PPO

# Alpaca Modern SDK (alpaca-py)
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# ---------------------------------------------------------
# 1. Page Config & Title
# ---------------------------------------------------------
st.set_page_config(page_title="RL Trading Engine & Terminal", layout="wide")
st.title("🤖 Autonomous RL Trading Engine & Live Demo Terminal")

# ---------------------------------------------------------
# 2. Sidebar Settings & Alpaca Paper Integration
# ---------------------------------------------------------
st.sidebar.header("⚙️ Simulation Settings")
symbol = st.sidebar.selectbox("Select Asset / Market", ["AAPL", "BTC-USD", "TSLA", "NVDA", "ETH-USD"])
period = st.sidebar.selectbox("Timeframe / Market Regime", ["1y", "6m", "2y"], index=0)
initial_balance = st.sidebar.number_input("Initial Balance ($)", value=10000.0, step=1000.0)
fee = st.sidebar.number_input("Transaction Fee (%)", value=0.1, step=0.01) / 100.0

st.sidebar.markdown("---")
st.sidebar.header("🔑 Alpaca Paper Trading API")
alpaca_key = st.sidebar.text_input("API Key ID", type="password")
alpaca_secret = st.sidebar.text_input("Secret Key", type="password")

# Initialize Alpaca Client (alpaca-py)
alpaca_connected = False
trading_client = None

if alpaca_key and alpaca_secret:
    try:
        trading_client = TradingClient(alpaca_key, alpaca_secret, paper=True)
        account = trading_client.get_account()
        st.sidebar.success(f"Connected! Paper Equity: ${float(account.equity):,.2f}")
        alpaca_connected = True
    except Exception as e:
        st.sidebar.error("Failed to connect to Alpaca.")

# ---------------------------------------------------------
# 3. Data Processing & Technical Indicators
# ---------------------------------------------------------
@st.cache_data
def load_and_preprocess(ticker, period_str):
    df = yf.download(ticker, period=period_str, interval="1h")
    
    if df.empty:
        st.error(f"❌ Could not download data for '{ticker}'.")
        st.stop()
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df.dropna()

    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    df['rsi_norm'] = (rsi - 50.0) / 50.0
    df['volatility'] = df['log_return'].rolling(20).std()
    
    df = df.dropna().reset_index()
    
    if len(df) == 0:
        st.error("❌ Not enough data points.")
        st.stop()
        
    return df

df = load_and_preprocess(symbol, period)

# ---------------------------------------------------------
# 4. Gymnasium Trading Simulator
# ---------------------------------------------------------
class TradingSimulator:
    def __init__(self, data, balance, fee_rate):
        self.df = data
        self.initial_balance = balance
        self.fee = fee_rate

    def run_backtest(self, model):
        balance = self.initial_balance
        shares = 0.0
        portfolio_history = []
        buy_x, buy_y, sell_x, sell_y = [], [], [], []

        for i in range(len(self.df)):
            row = self.df.iloc[i]
            price = float(row['Close'])
            
            pos_state = 1.0 if shares > 0 else 0.0
            obs = np.array([
                float(row['log_return']),
                float(row['rsi_norm']),
                float(row['volatility']),
                pos_state,
                float(balance / self.initial_balance)
            ], dtype=np.float32)

            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            if action == 1 and balance > 0:
                shares = (balance * (1 - self.fee)) / price
                balance = 0.0
                buy_x.append(row['Datetime'])
                buy_y.append(price)
            elif action == 2 and shares > 0:
                balance = shares * price * (1 - self.fee)
                shares = 0.0
                sell_x.append(row['Datetime'])
                sell_y.append(price)

            current_val = balance + (shares * price)
            portfolio_history.append(current_val)

        return (portfolio_history, buy_x, buy_y, sell_x, sell_y)

# ---------------------------------------------------------
# 5. Run Model & Compute Benchmarks
# ---------------------------------------------------------
try:
    model = PPO.load("ppo_trading_bot.zip")
    simulator = TradingSimulator(df, initial_balance, fee)
    bot_equity, buy_x, buy_y, sell_x, sell_y = simulator.run_backtest(model)
    
    # Latest Market Signal
    latest_row = df.iloc[-1]
    current_price = float(latest_row['Close'])
    volatility = float(latest_row['volatility'])
    rsi_norm = float(latest_row['rsi_norm'])
    
    latest_obs = np.array([
        float(latest_row['log_return']), rsi_norm, volatility, 1.0, 1.0
    ], dtype=np.float32)
    latest_action, _ = model.predict(latest_obs, deterministic=True)
    latest_action = int(latest_action)

    # ---------------------------------------------------------
    # 6. AI Expert Suggestion & Live Execution Panel
    # ---------------------------------------------------------
    st.markdown("---")
    col_signal, col_exec = st.columns([1, 1])

    with col_signal:
        st.subheader("💡 AI Expert Trade Suggestion")
        if latest_action == 1:
            st.success(f"### 🟢 RECOMMENDATION: STRONG BUY")
            st.write(f"**Action:** Enter Long Position at **${current_price:.2f}**")
        elif latest_action == 2:
            st.error(f"### 🔴 RECOMMENDATION: SELL / EXIT")
            st.write(f"**Action:** Liquidate Position at **${current_price:.2f}**")
        else:
            st.info(f"### 🟡 RECOMMENDATION: HOLD CASH")
            st.write(f"**Action:** Market neutral. Stand by.")

    with col_exec:
        st.subheader("🚀 Live Paper Execution")
        st.write("Trigger real-time execution via Alpaca Demo Broker API:")
        
        exec_button = st.button("⚡ Execute AI Signal on Alpaca Demo Account")
        if exec_button:
            if alpaca_connected and trading_client:
                try:
                    if latest_action == 0:
                        st.warning("Signal is HOLD. No trade dispatched.")
                    else:
                        clean_ticker = symbol.replace("-USD", "")
                        side = OrderSide.BUY if latest_action == 1 else OrderSide.SELL
                        
                        market_order_data = MarketOrderRequest(
                            symbol=clean_ticker,
                            qty=1,
                            side=side,
                            time_in_force=TimeInForce.GTC
                        )
                        
                        order = trading_client.submit_order(order_data=market_order_data)
                        st.balloons()
                        st.success(f"✅ Order Dispatched to Alpaca! Order ID: `{order.id}`")
                except Exception as e:
                    st.error(f"Order Execution Failed: {e}")
            else:
                st.warning("⚠️ Enter valid Alpaca API Keys in the sidebar to send live demo orders.")

    # ---------------------------------------------------------
    # 7. Embedded TradingView Live Chart
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader(f"📊 TradingView Terminal ({symbol})")
    
    tv_symbol = f"NASDAQ:{symbol}" if symbol in ["AAPL", "TSLA", "NVDA"] else f"BINANCE:{symbol.replace('-', '')}"

    tradingview_html = f"""
    <div class="tradingview-widget-container" style="height:500px;width:100%;">
      <div id="tradingview_chart" style="height:500px;width:100%;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "autosize": true,
        "symbol": "{tv_symbol}",
        "interval": "60",
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "container_id": "tradingview_chart"
      }});
      </script>
    </div>
    """
    components.html(tradingview_html, height=520)

    # ---------------------------------------------------------
    # 8. RL Backtest & Trade Execution Chart
    # ---------------------------------------------------------
    st.subheader("🎯 RL Execution Signal Overlay & Backtest")
    fig_candles = go.Figure()
    fig_candles.add_trace(go.Candlestick(
        x=df['Datetime'], open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name="Price Action"
    ))
    fig_candles.add_trace(go.Scatter(
        x=buy_x, y=buy_y, mode='markers',
        marker=dict(symbol='triangle-up', size=12, color='lime'),
        name="RL Buy Signal"
    ))
    fig_candles.add_trace(go.Scatter(
        x=sell_x, y=sell_y, mode='markers',
        marker=dict(symbol='triangle-down', size=12, color='red'),
        name="RL Sell Signal"
    ))
    fig_candles.update_layout(template="plotly_dark", height=450, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_candles, use_container_width=True)

except Exception as e:
    st.error(f"⚠️ Error initializing dashboard: {e}")
# ---------------------------------------------------------
# 9. Live Alpaca Portfolio Overview
# ---------------------------------------------------------
if alpaca_connected and trading_client:
    st.markdown("---")
    st.subheader("💼 Live Alpaca Paper Portfolio")
    
    acc = trading_client.get_account()
    positions = trading_client.get_all_positions()
    
    p1, p2, p3 = st.columns(3)
    p1.metric("Account Equity", f"${float(acc.equity):,.2f}")
    p2.metric("Buying Power", f"${float(acc.buying_power):,.2f}")
    p3.metric("Open Positions Count", len(positions))
    
    if positions:
        pos_data = []
        for p in positions:
            pos_data.append({
                "Asset": p.symbol,
                "Qty": p.qty,
                "Avg Entry": f"${float(p.avg_entry_price):,.2f}",
                "Current Price": f"${float(p.current_price):,.2f}",
                "Unrealized P&L": f"${float(p.unrealized_pl):,.2f}"
            })
        st.dataframe(pd.DataFrame(pos_data), use_container_width=True)
    else:
        st.info("No open positions currently held in paper account.")