"""
Dual-Purpose Investment Portfolio Dashboard
- SGX Income (Dividend) Portfolio
- US Momentum Trading Portfolio
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

# ──────────────────────────────────────────────
# Watchlist Configuration
# ──────────────────────────────────────────────
DEFAULT_TICKERS = {
    "D05.SI": "DBS Group",
    "O39.SI": "OCBC Bank",
    "U11.SI": "UOB",
    "CJLU.SI": "NetLink NBN Trust",
    "S68.SI": "SGX",
    "V03.SI": "Venture Corp",
    "U96.SI": "Sembcorp Industries",
    "AAPL": "Apple",
    "TSLA": "Tesla",
    "NVDA": "NVIDIA",
    "MSFT": "Microsoft",
    "AMD": "AMD",
}


@st.cache_data(ttl=86400)
def resolve_ticker_name(ticker: str) -> str:
    """Try to get the company name from yfinance, fallback to ticker."""
    try:
        info = yf.Ticker(ticker).info
        return info.get("shortName") or info.get("longName") or ticker
    except Exception:
        return ticker


def classify_ticker(ticker: str) -> str:
    """Classify ticker as SGX Income or US/Other Momentum."""
    if ticker.upper().endswith(".SI"):
        return "SGX (Income)"
    return "US/Other (Momentum)"

LOOKBACK_DAYS = 200


# ──────────────────────────────────────────────
# Data Fetching
# ──────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_price_data(ticker: str, period_days: int = LOOKBACK_DAYS) -> pd.DataFrame | None:
    """Fetch OHLCV data from yfinance with error handling."""
    try:
        end = datetime.today()
        start = end - timedelta(days=period_days + 60)  # extra buffer for indicator warm-up
        df = yf.download(ticker, start=start, end=end, progress=False, timeout=15)
        if df.empty:
            return None
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        st.warning(f"⚠️ Failed to fetch {ticker}: {e}")
        return None


@st.cache_data(ttl=1800)
def fetch_dividend_info(ticker: str) -> dict:
    """Pull last dividend and yield info from yfinance."""
    try:
        info = yf.Ticker(ticker).info
        return {
            "lastDividendValue": info.get("lastDividendValue", 0) or 0,
            "dividendYield": info.get("dividendYield", 0) or 0,
            "trailingAnnualDividendRate": info.get("trailingAnnualDividendRate", 0) or 0,
        }
    except Exception:
        return {"lastDividendValue": 0, "dividendYield": 0, "trailingAnnualDividendRate": 0}



# ──────────────────────────────────────────────
# Technical Analysis Engine
# ──────────────────────────────────────────────
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate EMAs, SMA, RSI, VWAP, and candlestick patterns."""
    df = df.copy()
    close = df["Close"]

    # Trend indicators (pure pandas)
    df["EMA_9"] = close.ewm(span=9, adjust=False).mean()
    df["EMA_20"] = close.ewm(span=20, adjust=False).mean()
    df["SMA_200"] = close.rolling(window=200).mean()

    # RSI (14-period)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # VWAP (cumulative approximation on daily bars)
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"] = (typical_price * df["Volume"]).cumsum() / df["Volume"].cumsum()

    # Candlestick pattern detection (manual implementation)
    df = detect_candlestick_patterns(df)

    return df


def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Detect Engulfing, Hammer, and Shooting Star patterns."""
    df["Bullish_Engulfing"] = False
    df["Bearish_Engulfing"] = False
    df["Hammer"] = False
    df["Shooting_Star"] = False

    opens = df["Open"].values
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values

    for i in range(1, len(df)):
        body_curr = abs(closes[i] - opens[i])
        body_prev = abs(closes[i - 1] - opens[i - 1])
        candle_range = highs[i] - lows[i]

        if candle_range == 0:
            continue

        # Bullish Engulfing: prev bearish, curr bullish, curr body engulfs prev body
        if (closes[i - 1] < opens[i - 1] and closes[i] > opens[i]
                and opens[i] <= closes[i - 1] and closes[i] >= opens[i - 1]):
            df.iloc[i, df.columns.get_loc("Bullish_Engulfing")] = True

        # Bearish Engulfing: prev bullish, curr bearish, curr body engulfs prev body
        if (closes[i - 1] > opens[i - 1] and closes[i] < opens[i]
                and opens[i] >= closes[i - 1] and closes[i] <= opens[i - 1]):
            df.iloc[i, df.columns.get_loc("Bearish_Engulfing")] = True

        # Hammer: small body at top, long lower shadow >= 2x body, tiny upper shadow
        upper_shadow = highs[i] - max(opens[i], closes[i])
        lower_shadow = min(opens[i], closes[i]) - lows[i]
        if body_curr > 0 and lower_shadow >= 2 * body_curr and upper_shadow <= body_curr * 0.3:
            df.iloc[i, df.columns.get_loc("Hammer")] = True

        # Shooting Star: small body at bottom, long upper shadow >= 2x body, tiny lower shadow
        if body_curr > 0 and upper_shadow >= 2 * body_curr and lower_shadow <= body_curr * 0.3:
            df.iloc[i, df.columns.get_loc("Shooting_Star")] = True

    return df



# ──────────────────────────────────────────────
# Alpha Signal Logic
# ──────────────────────────────────────────────
def generate_signal(row: pd.Series, prev_3_lows: float, holds_shares: bool = False) -> dict:
    """
    BUY: Price > 20-EMA AND 9-EMA > 20-EMA AND RSI < 65 AND (Bullish Engulfing OR Hammer)
    EXIT: RSI > 75 OR (Bearish Engulfing OR Shooting Star)
    STOP LOSS: Low of previous 3 candles
    """
    price = row["Close"]
    signal = {
        "action": "HOLD",
        "color": "gray",
        "stop_loss": round(prev_3_lows, 4),
        "entry_price": round(float(price), 4),
        "risk_reward": "—",
    }
    ema9 = row.get("EMA_9", np.nan)
    ema20 = row.get("EMA_20", np.nan)
    rsi = row.get("RSI_14", np.nan)
    bullish = row.get("Bullish_Engulfing", False)
    hammer = row.get("Hammer", False)
    bearish = row.get("Bearish_Engulfing", False)
    shooting = row.get("Shooting_Star", False)

    if pd.isna(ema20) or pd.isna(rsi):
        signal["action"] = "INSUFFICIENT DATA"
        return signal

    # EXIT conditions checked first
    if rsi > 75 or bearish or shooting:
        if holds_shares:
            signal["action"] = "🔴 TAKE PROFIT"
        else:
            signal["action"] = "⚠️ OVERBOUGHT"
        signal["color"] = "red"
        return signal

    # BUY conditions
    if price > ema20 and ema9 > ema20 and rsi < 65 and (bullish or hammer):
        signal["action"] = "🟢 STRONG BUY"
        signal["color"] = "green"
        signal["entry_price"] = round(float(price), 4)
        # Risk/Reward: risk = entry - stop, target = entry + 2*risk (2:1 R/R)
        risk = float(price) - prev_3_lows
        if risk > 0:
            target = float(price) + (2 * risk)
            signal["target_price"] = round(target, 4)
            signal["risk_reward"] = "2:1"
        return signal

    # Trend classification
    if pd.notna(ema9) and pd.notna(ema20):
        if ema9 > ema20:
            signal["action"] = "📈 BULLISH"
            signal["color"] = "lightgreen"
        else:
            signal["action"] = "📉 BEARISH"
            signal["color"] = "lightsalmon"

    return signal


def get_detected_patterns(row: pd.Series) -> str:
    """Return a comma-separated string of detected patterns."""
    patterns = []
    if row.get("Bullish_Engulfing", False):
        patterns.append("Bullish Engulfing")
    if row.get("Bearish_Engulfing", False):
        patterns.append("Bearish Engulfing")
    if row.get("Hammer", False):
        patterns.append("Hammer")
    if row.get("Shooting_Star", False):
        patterns.append("Shooting Star")
    return ", ".join(patterns) if patterns else "—"



# ──────────────────────────────────────────────
# Chart Builder
# ──────────────────────────────────────────────
def build_candlestick_chart(df: pd.DataFrame, ticker: str, name: str) -> go.Figure:
    """Create a professional candlestick chart with EMA overlays and pattern markers."""
    plot_df = df.tail(60).copy()  # last ~3 months of trading days

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
        subplot_titles=[f"{name} ({ticker})", "RSI (14)"],
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=plot_df.index, open=plot_df["Open"], high=plot_df["High"],
        low=plot_df["Low"], close=plot_df["Close"], name="Price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # EMAs & SMA
    for col, color, dash in [("EMA_9", "#ff9800", "dot"), ("EMA_20", "#2196f3", "dash"), ("SMA_200", "#9c27b0", "solid")]:
        if col in plot_df.columns:
            fig.add_trace(go.Scatter(
                x=plot_df.index, y=plot_df[col], name=col,
                line=dict(color=color, width=1.2, dash=dash),
            ), row=1, col=1)

    # Pattern markers
    bullish_mask = plot_df["Bullish_Engulfing"] | plot_df["Hammer"]
    bearish_mask = plot_df["Bearish_Engulfing"] | plot_df["Shooting_Star"]

    if bullish_mask.any():
        fig.add_trace(go.Scatter(
            x=plot_df.index[bullish_mask], y=plot_df["Low"][bullish_mask] * 0.995,
            mode="markers", name="Bullish Pattern",
            marker=dict(symbol="triangle-up", size=12, color="#00e676"),
        ), row=1, col=1)

    if bearish_mask.any():
        fig.add_trace(go.Scatter(
            x=plot_df.index[bearish_mask], y=plot_df["High"][bearish_mask] * 1.005,
            mode="markers", name="Bearish Pattern",
            marker=dict(symbol="triangle-down", size=12, color="#ff1744"),
        ), row=1, col=1)

    # RSI subplot
    if "RSI_14" in plot_df.columns:
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df["RSI_14"], name="RSI",
            line=dict(color="#7c4dff", width=1.5),
        ), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)

    fig.update_layout(
        height=550, xaxis_rangeslider_visible=False,
        template="plotly_dark", margin=dict(l=50, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(size=11),
    )
    fig.update_xaxes(type="category", nticks=15, row=1, col=1)
    fig.update_xaxes(type="category", nticks=15, row=2, col=1)

    return fig



# ──────────────────────────────────────────────
# Streamlit App
# ──────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Investment Portfolio Dashboard", page_icon="📊", layout="wide")
    st.title("📊 Investment Portfolio Dashboard")
    st.caption("Dual-Purpose: SGX Dividend Income & US Momentum Trading")

    # ── Auto-Refresh Controls ──
    st.sidebar.header("⚙️ Refresh Settings")
    auto_refresh = st.sidebar.toggle("Auto-Refresh", value=False, help="Automatically refresh data at the selected interval.")
    refresh_interval = st.sidebar.selectbox(
        "Refresh Interval",
        options=[1, 2, 5, 10, 15],
        index=2,
        format_func=lambda x: f"Every {x} min",
        help="How often to re-fetch market data.",
    )
    if st.sidebar.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()

    # ── Data Freshness Note ──
    st.sidebar.divider()
    st.sidebar.markdown(
        """
        📡 **Data Source & Freshness**
        - Price data: [Yahoo Finance](https://finance.yahoo.com) via `yfinance`
        - **US stocks**: ~15–20 min delay during NYSE/NASDAQ hours (9:30 AM–4 PM ET)
        - **SGX stocks**: ~15–20 min delay during SGX hours (9 AM–5 PM SGT)
        - Dividend data: Trailing 12-month from Yahoo Finance, refreshed every 30 min
        - Price cache: **5 minutes** (auto-cleared on refresh)
        - After market close, data reflects the final closing prices
        - ⚠️ *Not suitable for intraday scalping — use a broker terminal for live ticks*
        """
    )

    # ── Sidebar: Watchlist & Portfolio ──
    st.sidebar.divider()
    st.sidebar.header("📋 Watchlist")

    # Initialize watchlist in session state with defaults
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = dict(DEFAULT_TICKERS)

    # Add custom ticker
    with st.sidebar.expander("➕ Add a Ticker", expanded=False):
        new_ticker = st.text_input(
            "Ticker Symbol",
            placeholder="e.g. GOOG, 9988.HK, BN4.SI",
            help="Enter any valid Yahoo Finance ticker symbol. SGX tickers end with .SI",
            key="new_ticker_input",
        ).strip().upper()
        if st.button("Add to Watchlist") and new_ticker:
            if new_ticker in st.session_state.watchlist:
                st.warning(f"{new_ticker} is already in your watchlist.")
            else:
                with st.spinner(f"Validating {new_ticker}..."):
                    test_df = fetch_price_data(new_ticker, period_days=10)
                if test_df is not None and not test_df.empty:
                    name = resolve_ticker_name(new_ticker)
                    st.session_state.watchlist[new_ticker] = name
                    st.success(f"Added {name} ({new_ticker})")
                    st.rerun()
                else:
                    st.error(f"Could not find data for '{new_ticker}'. Check the symbol.")

    # Remove ticker
    if len(st.session_state.watchlist) > 0:
        with st.sidebar.expander("🗑️ Remove a Ticker", expanded=False):
            to_remove = st.selectbox(
                "Select ticker to remove",
                options=list(st.session_state.watchlist.keys()),
                format_func=lambda t: f"{st.session_state.watchlist[t]} ({t})",
                key="remove_ticker_select",
            )
            if st.button("Remove from Watchlist"):
                del st.session_state.watchlist[to_remove]
                if to_remove in st.session_state.get("portfolio", {}):
                    del st.session_state.portfolio[to_remove]
                st.rerun()

    # Show current watchlist count
    st.sidebar.caption(f"Tracking {len(st.session_state.watchlist)} tickers")

    # ── Portfolio Holdings ──
    st.sidebar.divider()
    st.sidebar.header("💼 My Portfolio")

    # Build ticker info from dynamic watchlist
    all_tickers = {}
    for ticker, name in st.session_state.watchlist.items():
        group = classify_ticker(ticker)
        all_tickers[ticker] = {"name": name, "group": group}

    # Initialize session state for portfolio
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = {}

    # Import portfolio from JSON
    st.sidebar.markdown("**Load / Save Portfolio**")
    uploaded = st.sidebar.file_uploader("📂 Import portfolio JSON", type=["json"], key="import_portfolio")
    if uploaded is not None:
        try:
            import json
            imported = json.load(uploaded)
            # Also add any tickers from the import that aren't in watchlist
            for t, data in imported.items():
                if t not in st.session_state.watchlist:
                    name = data.get("name") or resolve_ticker_name(t)
                    st.session_state.watchlist[t] = name
                    all_tickers[t] = {"name": name, "group": classify_ticker(t)}
            st.session_state.portfolio = imported
            st.sidebar.success(f"Loaded {len(imported)} holdings")
        except Exception as e:
            st.sidebar.error(f"Invalid JSON: {e}")

    st.sidebar.divider()

    # Select which stocks you hold
    selected_tickers = st.sidebar.multiselect(
        "Select stocks you own",
        options=list(all_tickers.keys()),
        default=[t for t in st.session_state.portfolio if t in all_tickers],
        format_func=lambda t: f"{all_tickers[t]['name']} ({t})",
        help="Pick the stocks in your portfolio. Then enter shares and cost below.",
    )

    # Input shares & avg cost for each selected ticker
    holdings: dict[str, dict] = {}
    for ticker in selected_tickers:
        info = all_tickers[ticker]
        saved = st.session_state.portfolio.get(ticker, {})
        with st.sidebar.expander(f"✏️ {info['name']} ({ticker})", expanded=True):
            shares = st.number_input(
                "Shares Owned", min_value=0, value=int(saved.get("shares", 0)),
                step=1, key=f"shares_{ticker}",
            )
            avg_cost = st.number_input(
                "Avg Cost ($)", min_value=0.0, value=float(saved.get("avg_cost", 0)),
                step=0.01, key=f"cost_{ticker}", format="%.4f",
            )
            holdings[ticker] = {"shares": shares, "avg_cost": avg_cost, "name": info["name"], "group": info["group"]}
            # Keep session state in sync
            st.session_state.portfolio[ticker] = {"shares": shares, "avg_cost": avg_cost}

    # Also include non-selected tickers with 0 shares for the action table
    for ticker, info in all_tickers.items():
        if ticker not in holdings:
            holdings[ticker] = {"shares": 0, "avg_cost": 0, "name": info["name"], "group": info["group"]}

    # Clean up removed tickers from session state
    for t in list(st.session_state.portfolio.keys()):
        if t not in selected_tickers:
            del st.session_state.portfolio[t]

    # Export portfolio as JSON download
    if st.session_state.portfolio:
        import json
        portfolio_json = json.dumps(st.session_state.portfolio, indent=2)
        st.sidebar.download_button(
            "💾 Export Portfolio JSON",
            data=portfolio_json,
            file_name="my_portfolio.json",
            mime="application/json",
            help="Download your portfolio as a JSON file. Import it next time to restore your holdings.",
        )

    # ── Fetch & Process Data ──
    with st.spinner("Fetching market data..."):
        all_data: dict[str, pd.DataFrame] = {}
        for ticker in st.session_state.watchlist:
            df = fetch_price_data(ticker)
            if df is not None and len(df) > 20:
                all_data[ticker] = compute_indicators(df)

    if not all_data:
        st.error("Could not fetch data for any ticker. Check your internet connection.")
        return

    # ── Income Summary Card ──
    st.header("💰 Income & P&L Summary")
    col1, col2, col3, col4 = st.columns(4)

    total_annual_div = 0.0
    total_invested = 0.0
    total_market_value = 0.0

    for ticker, df in all_data.items():
        h = holdings.get(ticker, {})
        shares = h.get("shares", 0)
        avg_cost = h.get("avg_cost", 0)
        current_price = float(df["Close"].iloc[-1])

        total_invested += shares * avg_cost
        total_market_value += shares * current_price

        if shares > 0:
            div_info = fetch_dividend_info(ticker)
            annual_div_rate = div_info["trailingAnnualDividendRate"]
            total_annual_div += shares * annual_div_rate

    total_pnl = total_market_value - total_invested
    pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0

    col1.metric("Total Invested", f"${total_invested:,.2f}")
    col2.metric("Market Value", f"${total_market_value:,.2f}")
    col3.metric("Unrealized P&L", f"${total_pnl:,.2f}", delta=f"{pnl_pct:+.2f}%")
    col4.metric("Projected Annual Dividends", f"${total_annual_div:,.2f}")

    # ── Daily Action Table ──
    st.header("📋 Daily Action Table")

    with st.expander("ℹ️ Action Legend — What does each recommendation mean?", expanded=False):
        st.markdown("""
**When you HOLD a position:**
| Action | Meaning |
|--------|---------|
| 💰 SELL — Take Profit | Overbought signal (RSI>75 or bearish pattern) AND you're up >10%. Strong sell signal. |
| 🟡 SELL — Lock Gains | Overbought signal AND you're in profit (<10%). Consider selling to protect gains. |
| 🟡 REDUCE — Protect Gains | Trend turned bearish BUT you're up >15%. Sell part of your position (30-50%) to lock in profit, keep the rest. |
| 🟡 TRIM — RSI High + Good Profit | RSI is elevated (>65) AND you're up >20%. Sell a small portion (20-30%) before a potential pullback. Preventive, not urgent. |
| 🟢 BUY MORE — Avg Down | Trend is bullish BUT you're down >5%. Good opportunity to lower your average cost. |
| 🟢 BUY MORE | Strong buy signal detected while you hold. Add to your position. |
| 🔴 CUT LOSS | Trend is bearish AND you're down >15%. Consider exiting to limit further losses. |
| ⏳ HOLD — Wait for Recovery | Overbought signal but you're still at a loss. Don't sell at a loss on a temporary spike. |
| ⏳ HOLD | No strong signal either way. Stay patient. |

**When you DON'T hold a position:**
| Action | Meaning |
|--------|---------|
| 🟢 OPEN POSITION | Strong buy signal — all conditions met. Good time to enter. |
| 👀 WATCHLIST — Wait for Entry | Trend is bullish but no strong buy trigger yet. Keep watching for a pullback entry. |
| 🚫 DON'T BUY — Overbought | RSI>75 or bearish pattern. Stock is overheated — wait for it to cool down. |
| ⏳ WAIT — Trend Down | Bearish trend. Don't catch a falling knife. |
| 👀 WATCH | Neutral. No clear signal. |

**Key Indicators:**
- **RSI**: <30 = oversold (buy zone), >70 = overbought (sell zone)
- **Trend**: Based on 9-EMA vs 20-EMA crossover
- **Pattern**: Candlestick patterns (Engulfing, Hammer, Shooting Star)
- **Stop Loss**: Lowest low of previous 3 trading days
- **Target (2:1)**: Take-profit at 2× the risk from entry to stop loss
        """)
    action_rows = []
    for ticker, df in all_data.items():
        latest = df.iloc[-1]
        prev_3_lows = float(df["Low"].iloc[-4:-1].min()) if len(df) >= 4 else float(latest["Low"])
        signal = generate_signal(latest, prev_3_lows, holds_shares=holdings.get(ticker, {}).get("shares", 0) > 0)
        pattern = get_detected_patterns(latest)
        h = holdings.get(ticker, {})
        name = h.get("name", ticker)
        shares = h.get("shares", 0)
        avg_cost = h.get("avg_cost", 0)
        current_price = float(latest["Close"])
        pnl = (current_price - avg_cost) * shares if shares > 0 and avg_cost > 0 else 0

        # Smart entry logic:
        # SGX Income → "Value Entry" = price where div yield hits 5yr avg (proxy: 20-EMA as support)
        # US Momentum → "Pullback Entry" = 20-EMA (buy the dip in an uptrend)
        ema20_val = float(latest.get("EMA_20", current_price))
        ema9_val = float(latest.get("EMA_9", current_price))
        group = holdings.get(ticker, {}).get("group", "")

        if "Income" in group:
            # Value approach: enter at 20-EMA or current price, whichever is lower
            # This approximates "buy on pullback to support"
            suggested_entry = round(min(ema20_val, current_price), 4)
            entry_label = "Value Entry"
        else:
            # Momentum approach: enter at 20-EMA pullback if bullish, else wait
            if ema9_val > ema20_val:
                suggested_entry = round(ema20_val, 4)  # buy the pullback
                entry_label = "Pullback Entry"
            else:
                suggested_entry = round(ema9_val, 4)  # tighter entry in weak trend
                entry_label = "Caution Entry"

        stop = signal["stop_loss"]
        risk = suggested_entry - stop
        if risk > 0:
            target = round(suggested_entry + 2 * risk, 4)
            rr = "2:1"
        else:
            target = None
            rr = "—"

        # Generate action recommendation based on cost basis + trend
        if shares > 0 and avg_cost > 0:
            pnl_pct = ((current_price - avg_cost) / avg_cost) * 100
            pnl_display = f"${pnl:,.2f}"
            pnl_pct_display = f"{pnl_pct:+.1f}%"
            avg_cost_display = f"${avg_cost:,.4f}"

            # Decision logic combining P&L with technical signals
            if signal["action"] in ("🔴 TAKE PROFIT",) and pnl_pct > 10:
                action = "💰 SELL — Take Profit"
            elif signal["action"] == "🔴 TAKE PROFIT" and pnl_pct > 0:
                action = "🟡 SELL — Lock Gains"
            elif signal["action"] == "🔴 TAKE PROFIT" and pnl_pct <= 0:
                action = "⏳ HOLD — Wait for Recovery"
            elif "BULLISH" in signal["action"] and pnl_pct < -5:
                action = "🟢 BUY MORE — Avg Down"
            elif "STRONG BUY" in signal["action"]:
                action = "🟢 BUY MORE"
            elif "BEARISH" in signal["action"] and pnl_pct < -15:
                action = "🔴 CUT LOSS"
            elif "BEARISH" in signal["action"] and pnl_pct > 15:
                action = "🟡 REDUCE — Protect Gains"
            elif pnl_pct > 20 and float(latest.get("RSI_14", 50)) > 65:
                action = "🟡 TRIM — RSI High + Good Profit"
            else:
                action = "⏳ HOLD"
        else:
            pnl_pct = 0
            pnl_display = "—"
            pnl_pct_display = "—"
            avg_cost_display = "—"
            # No position — should I buy?
            if "STRONG BUY" in signal["action"]:
                action = "🟢 OPEN POSITION"
            elif "BULLISH" in signal["action"]:
                action = "👀 WATCHLIST — Wait for Entry"
            elif "OVERBOUGHT" in signal["action"]:
                action = "🚫 DON'T BUY — Overbought"
            elif "BEARISH" in signal["action"]:
                action = "⏳ WAIT — Trend Down"
            else:
                action = "👀 WATCH"

        action_rows.append({
            "Ticker": ticker,
            "Name": name,
            "Price": f"${current_price:,.4f}",
            "Avg Cost": avg_cost_display,
            "P&L %": pnl_pct_display,
            "P&L": pnl_display,
            "RSI": f"{latest.get('RSI_14', 0):.1f}",
            "Trend": signal["action"],
            "Pattern": pattern,
            "Action": action,
            "Stop Loss": f"${stop:,.4f}",
            "Target (2:1)": f"${target:,.4f}" if target else "—",
            "Shares": shares,
        })

    action_df = pd.DataFrame(action_rows)
    st.dataframe(
        action_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", help="Stock ticker symbol."),
            "Name": st.column_config.TextColumn("Name", help="Company name."),
            "Price": st.column_config.TextColumn("Price", help="Latest market price from Yahoo Finance."),
            "Avg Cost": st.column_config.TextColumn("Avg Cost", help="Your average purchase price per share (from sidebar)."),
            "P&L %": st.column_config.TextColumn("P&L %", help="Percentage gain/loss vs your avg cost. Green = profit, Red = loss."),
            "P&L": st.column_config.TextColumn("P&L", help="Dollar profit/loss = (Price − Avg Cost) × Shares."),
            "RSI": st.column_config.TextColumn("RSI", help="Relative Strength Index (14). <30 = oversold (buy zone), >70 = overbought (sell zone)."),
            "Trend": st.column_config.TextColumn("Trend", help="Technical trend: 📈 BULLISH (uptrend), 📉 BEARISH (downtrend), ⚠️ OVERBOUGHT, 🟢 STRONG BUY."),
            "Pattern": st.column_config.TextColumn("Pattern", help="Candlestick patterns detected on the latest candle."),
            "Action": st.column_config.TextColumn("Action", help="Smart recommendation based on your cost basis + market trend. Click 'ℹ️ Action Legend' above the table for full details on each action type."),
            "Stop Loss": st.column_config.TextColumn("Stop Loss", help="Suggested stop loss = lowest low of previous 3 days. Sell if price drops below this."),
            "Target (2:1)": st.column_config.TextColumn("Target (2:1)", help="Take-profit target at 2:1 risk/reward ratio."),
            "Shares": st.column_config.NumberColumn("Shares", help="Number of shares you own."),
        },
    )

    # ── Charts ──
    st.header("📈 Technical Charts")
    # Group tickers by classification
    grouped: dict[str, list] = {}
    for ticker, name in st.session_state.watchlist.items():
        group = classify_ticker(ticker)
        grouped.setdefault(group, []).append((ticker, name))

    for group, ticker_list in grouped.items():
        st.subheader(group)
        for ticker, name in ticker_list:
            if ticker in all_data:
                fig = build_candlestick_chart(all_data[ticker], ticker, name)
                st.plotly_chart(fig, use_container_width=True)

    # ── Detailed Holdings Table ──
    st.header("📊 Detailed Holdings Breakdown")
    detail_rows = []
    for ticker, df in all_data.items():
        h = holdings.get(ticker, {})
        shares = h.get("shares", 0)
        avg_cost = h.get("avg_cost", 0)
        if shares <= 0:
            continue
        current_price = float(df["Close"].iloc[-1])
        div_info = fetch_dividend_info(ticker)
        annual_div = div_info["trailingAnnualDividendRate"]
        div_yield = div_info["dividendYield"]
        pnl = (current_price - avg_cost) * shares
        detail_rows.append({
            "Ticker": ticker,
            "Name": h.get("name", ticker),
            "Shares": shares,
            "Avg Cost": f"${avg_cost:,.4f}",
            "Current": f"${current_price:,.4f}",
            "P&L": f"${pnl:,.2f}",
            "P&L %": f"{((current_price / avg_cost - 1) * 100):+.2f}%" if avg_cost > 0 else "—",
            "Div/Share": f"${annual_div:,.4f}",
            "Yield": f"{(div_yield * 100):.2f}%" if div_yield else "—",
            "Annual Income": f"${shares * annual_div:,.2f}",
        })

    if detail_rows:
        st.dataframe(
            pd.DataFrame(detail_rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", help="Stock ticker symbol."),
                "Name": st.column_config.TextColumn("Name", help="Company name."),
                "Shares": st.column_config.NumberColumn("Shares", help="Number of shares you own."),
                "Avg Cost": st.column_config.TextColumn("Avg Cost", help="Your average purchase price per share (entered in sidebar)."),
                "Current": st.column_config.TextColumn("Current", help="Latest closing price from Yahoo Finance."),
                "P&L": st.column_config.TextColumn("P&L", help="Unrealized profit/loss = (Current Price − Avg Cost) × Shares."),
                "P&L %": st.column_config.TextColumn("P&L %", help="Percentage gain or loss relative to your average cost."),
                "Div/Share": st.column_config.TextColumn("Div/Share", help="Trailing annual dividend per share from Yahoo Finance."),
                "Yield": st.column_config.TextColumn("Yield", help="Dividend yield = Annual Dividend ÷ Current Price. Higher yield means more income per dollar invested."),
                "Annual Income": st.column_config.TextColumn("Annual Income", help="Projected yearly dividend income = Shares × Dividend per Share."),
            },
        )
    else:
        st.info("Enter your shares owned in the sidebar to see detailed holdings.")

    st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · Data from Yahoo Finance (15–20 min delay) · Not financial advice.")

    # ── Auto-Refresh Timer ──
    if auto_refresh:
        time.sleep(refresh_interval * 60)
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()
