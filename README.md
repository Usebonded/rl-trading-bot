# 🤖 Autonomous RL Trading Engine & Live Demo Terminal

An end-to-end algorithmic trading system powered by Reinforcement Learning (**Proximal Policy Optimization - PPO**). This interactive dashboard evaluates historical and dynamic market regimes, generates trade execution signals, renders embedded TradingView charting, and dispatches real-time demo paper orders via Alpaca's modern API.

---

## ✨ Key Features

* **🤖 Reinforcement Learning Policy:** Employs a PPO agent trained on key market indicators (*log returns, normalized RSI, and rolling volatility*) to execute adaptive trading strategies (`BUY`, `SELL`, `HOLD`).
* **📊 Backtesting & Signal Overlay:** Interactive backtest engine featuring Plotly candlestick charting with exact RL entry and exit signal markers.
* **📈 Embedded TradingView Terminal:** Live, interactive multi-asset technical charts built directly into the dashboard UI.
* **⚡ Live Alpaca Broker Integration:** Direct integration with Alpaca's modern Python SDK (`alpaca-py`) for real-time paper trading execution.
* **🎨 Interactive Analytics UI:** Clean Streamlit dashboard for real-time asset selection, timeframe switching, and market parameter tuning.

---

## 🛠️ Tech Stack

* **UI & Dashboard:** [Streamlit](https://streamlit.io/)
* **Reinforcement Learning:** [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
* **Market Data & Charting:** `yfinance`, [Plotly](https://plotly.com/python/), TradingView Widgets
* **Execution & Brokerage API:** [Alpaca Python SDK (`alpaca-py`)](https://github.com/alpacahq/alpaca-py)
* **Data Processing:** `pandas`, `numpy`

---

## 🚀 Getting Started Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/Usebonded/rl-trading-bot.git](https://github.com/Usebonded/rl-trading-bot.git)
cd rl-trading-bot
