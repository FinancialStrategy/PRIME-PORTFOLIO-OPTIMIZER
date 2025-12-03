import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import scipy.stats as stats
from scipy import optimize
import warnings

# PyPortfolioOpt imports - CORRECTED
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import objective_functions
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
# Fix: Import directly from submodules to avoid namespace errors
from pypfopt.cla import CLA
from pypfopt.hierarchical_portfolio import HRPOpt
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt import plotting

# ARCH for GARCH models
try:
    import arch
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION & ASSET UNIVERSES
# ============================================================================
st.set_page_config(page_title="Institutional Portfolio AI", layout="wide", page_icon="📈")

# Custom CSS for "Pro" look
st.markdown("""
<style>
    .metric-card {background-color: #1e1e1e; padding: 15px; border-radius: 5px; border-left: 5px solid #4CAF50;}
    .reportview-container {background: #0e1117;}
</style>
""", unsafe_allow_html=True)

# 1. BIST 30 (Turkish Major Stocks) - Tickers with .IS suffix
BIST_30 = [
    'AKBNK.IS', 'ARCLK.IS', 'ASELS.IS', 'BIMAS.IS', 'DOHOL.IS', 'EKGYO.IS', 
    'ENKAI.IS', 'EREGL.IS', 'FROTO.IS', 'GARAN.IS', 'GUBRF.IS', 'HALKB.IS', 
    'HEKTS.IS', 'ISCTR.IS', 'KCHOL.IS', 'KOZAA.IS', 'KOZAL.IS', 'KRDMD.IS', 
    'PETKM.IS', 'PGSUS.IS', 'SAHOL.IS', 'SASA.IS', 'SISE.IS', 'TCELL.IS', 
    'THYAO.IS', 'TKFEN.IS', 'TOASO.IS', 'TSKB.IS', 'TUPRS.IS', 'YKBNK.IS'
]

# 2. Global Major Indices (20)
GLOBAL_INDICES = [
    '^GSPC', '^DJI', '^IXIC', '^RUT', '^VIX', # US
    '^FTSE', '^GDAXI', '^FCHI', '^STOXX50E', # Europe
    '^N225', '^HSI', '000001.SS', '^STI', '^AXJO', # Asia/Pacific
    '^BVSP', '^MXX', '^MERV', # Latin America
    '^TA125.TA', '^CASE30', '^JN0U.JO' # Middle East / Others
]

# 3. US Tech/Blue Chip Defaults
US_DEFAULTS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V', 'WMT']

# ============================================================================
# DATA MANAGEMENT MODULE
# ============================================================================

class PortfolioDataManager:
    """Professional-grade financial data manager with Streamlit Caching"""
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def fetch_data(tickers, start_date, end_date):
        """Fetch financial data with robust MultiIndex and single-ticker handling"""
        if not tickers:
            return pd.DataFrame(), {}
            
        try:
            # Force list format
            if isinstance(tickers, str):
                tickers = [tickers]

            # 1. Download with threads=False (More stable on Streamlit Cloud)
            data = yf.download(tickers, start=start_date, end=end_date, progress=False, group_by='ticker', threads=False)
            
            prices = pd.DataFrame()
            ohlc_dict = {}

            # 2. Handle Case: Single Ticker
            if len(tickers) == 1:
                ticker = tickers[0]
                df = data
                
                # If yfinance returned a MultiIndex for a single ticker (level 0 is ticker)
                if isinstance(data.columns, pd.MultiIndex):
                    try:
                        df = data.xs(ticker, axis=1, level=0, drop_level=True)
                    except:
                        pass # Keep original if xs fails
                
                # Check for Adj Close, fallback to Close
                price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
                if price_col in df.columns:
                    prices[ticker] = df[price_col]
                    ohlc_dict[ticker] = df

            # 3. Handle Case: Multiple Tickers
            else:
                # Ensure we are working with a MultiIndex
                if not isinstance(data.columns, pd.MultiIndex):
                    # Edge case: yfinance returned flat frame for multiple tickers (rare failure mode)
                    return pd.DataFrame(), {}
                
                for ticker in tickers:
                    try:
                        # Extract dataframe for specific ticker
                        df = data.xs(ticker, axis=1, level=0, drop_level=True)
                        
                        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
                        if price_col in df.columns:
                            prices[ticker] = df[price_col]
                            ohlc_dict[ticker] = df
                    except KeyError:
                        # Ticker data missing in download
                        continue
            
            # 4. Final Data Cleaning
            prices = prices.ffill().bfill()
            
            # validation: Drop columns that are all NaN
            prices = prices.dropna(axis=1, how='all')
            
            return prices, ohlc_dict
            
        except Exception as e:
            st.error(f"Data Fetch Error: {str(e)}")
            return pd.DataFrame(), {}

    @staticmethod
    def calculate_returns(prices, method='log'):
        if method == 'log':
            returns = np.log(prices / prices.shift(1)).dropna()
        else:
            returns = prices.pct_change().dropna()
        return returns
        
    @staticmethod
    @st.cache_data(ttl=3600 * 24) # Cache market caps for 24 hours
    def get_market_caps(tickers):
        """Fetch real market caps for Black-Litterman"""
        mcaps = {}
        for t in tickers:
            try:
                info = yf.Ticker(t).info
                mcaps[t] = info.get('marketCap', 1e10) # Default to 10B if missing
            except:
                mcaps[t] = 1e10
        return pd.Series(mcaps)

# ============================================================================
# ADVANCED RISK METRICS MODULE
# ============================================================================

class AdvancedRiskMetrics:
    
    @staticmethod
    def calculate_metrics(returns, risk_free=0.02):
        """Calculate comprehensive risk metrics"""
        # Annualization factor
        ann_factor = 252
        
        # Basic
        total_return = (1 + returns).prod() - 1
        cagr = (1 + total_return) ** (ann_factor / len(returns)) - 1
        volatility = returns.std() * np.sqrt(ann_factor)
        
        # Sharpe
        excess_returns = returns - (risk_free / ann_factor)
        sharpe = np.sqrt(ann_factor) * excess_returns.mean() / returns.std()
        
        # Sortino (Downside Deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(ann_factor)
        sortino = np.sqrt(ann_factor) * excess_returns.mean() / downside_std if downside_std != 0 else 0
        
        # Max Drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        max_dd = drawdown.min()
        
        # Calmar Ratio
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        # VaR & CVaR (Historical)
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        return {
            "Total Return": total_return,
            "CAGR": cagr,
            "Volatility": volatility,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Max Drawdown": max_dd,
            "Calmar Ratio": calmar,
            "VaR 95%": var_95,
            "CVaR 95%": cvar_95
        }

    @staticmethod
    def calculate_garch_vol(returns):
        """GARCH(1,1) Volatility Forecasting"""
        if not HAS_ARCH:
            return None
        try:
            # Scale returns for numerical stability
            am = arch.arch_model(returns * 100, vol='Garch', p=1, q=1, dist='normal')
            res = am.fit(disp='off')
            forecast = res.forecast(horizon=1)
            return np.sqrt(forecast.variance.values[-1, 0]) / 100
        except:
            return returns.std()

# ============================================================================
# OPTIMIZATION ENGINE
# ============================================================================

class AdvancedPortfolioOptimizer:
    def __init__(self, returns, prices):
        self.returns = returns
        self.prices = prices
        self.mu = expected_returns.mean_historical_return(prices)
        self.S = risk_models.sample_cov(prices)
    
    def optimize(self, method, risk_free_rate=0.02, target_return=None):
        ef = EfficientFrontier(self.mu, self.S)
        
        if method == 'Max Sharpe':
            weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
        elif method == 'Min Volatility':
            weights = ef.min_volatility()
        elif method == 'Efficient Risk':
            # Target vol set to average asset vol as default
            avg_vol = np.sqrt(np.diag(self.S)).mean()
            weights = ef.efficient_risk(target_volatility=avg_vol)
        
        cleaned_weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
        return cleaned_weights, performance

    def optimize_hrp(self):
        hrp = HRPOpt(self.returns)
        weights = hrp.optimize()
        performance = hrp.portfolio_performance(verbose=False)
        return hrp.clean_weights(), performance
        
    def optimize_black_litterman(self, market_caps):
        # Delta: Risk aversion coefficient
        delta = risk_models.black_litterman.market_implied_risk_aversion(self.prices)
        
        # Prior: Market implied returns
        prior = risk_models.black_litterman.market_implied_prior_returns(market_caps, delta, self.S)
        
        # Run BL Model (using default views of 0 for simplicity in this UI, can be expanded)
        bl = BlackLittermanModel(self.S, pi=prior, absolute_views=None)
        
        ret_bl = bl.bl_returns()
        S_bl = bl.bl_cov()
        
        ef = EfficientFrontier(ret_bl, S_bl)
        weights = ef.max_sharpe()
        performance = ef.portfolio_performance(verbose=False)
        return ef.clean_weights(), performance

# ============================================================================
# STREAMLIT UI LAYOUT
# ============================================================================

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("🔧 Configuration")

# Asset Selection
st.sidebar.subheader("1. Asset Universe")
ticker_lists = {
    "US Defaults": US_DEFAULTS,
    "BIST 30 (Turkey)": BIST_30,
    "Global Indices": GLOBAL_INDICES
}

selected_list = st.sidebar.selectbox("Load Asset Group", list(ticker_lists.keys()))
available_tickers = ticker_lists[selected_list]

# Allow custom addition
custom_tickers = st.sidebar.text_input("Add Custom Tickers (comma sep)", value="")
if custom_tickers:
    available_tickers = list(set(available_tickers + [t.strip().upper() for t in custom_tickers.split(',')]))

selected_tickers = st.sidebar.multiselect("Select Assets for Portfolio", available_tickers, default=available_tickers[:5])

# Parameters
st.sidebar.subheader("2. Parameters")
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365*2))
end_date = st.sidebar.date_input("End Date", datetime.now())
rf_rate = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 10.0, 2.0, 0.1) / 100

method = st.sidebar.selectbox("Optimization Strategy", 
    ['Max Sharpe', 'Min Volatility', 'Hierarchical Risk Parity (HRP)', 'Black-Litterman', 'Equal Weight'])

run_btn = st.sidebar.button("🚀 RUN ANALYSIS", type="primary")

# --- MAIN APP LOGIC ---

if run_btn:
    with st.spinner('Fetching Institutional Data & Optimizing...'):
        # 1. Fetch Data
        data_manager = PortfolioDataManager()
        prices, ohlc_data = data_manager.fetch_data(selected_tickers, start_date, end_date)
        
        if prices.empty:
            st.error("No data found. Please check ticker symbols.")
        else:
            returns = data_manager.calculate_returns(prices)
            
            # 2. Optimization
            optimizer = AdvancedPortfolioOptimizer(returns, prices)
            
            if method == 'Hierarchical Risk Parity (HRP)':
                weights, perf = optimizer.optimize_hrp()
            elif method == 'Black-Litterman':
                mcaps = data_manager.get_market_caps(selected_tickers)
                weights, perf = optimizer.optimize_black_litterman(mcaps)
            elif method == 'Equal Weight':
                weights = {t: 1.0/len(selected_tickers) for t in selected_tickers}
                # Calc performance manually
                w_arr = np.array(list(weights.values()))
                ret = np.sum(returns.mean() * w_arr) * 252
                vol = np.sqrt(np.dot(w_arr.T, np.dot(returns.cov() * 252, w_arr)))
                sharpe = (ret - rf_rate) / vol
                perf = (ret, vol, sharpe)
            else:
                weights, perf = optimizer.optimize(method, risk_free_rate=rf_rate)

            # 3. TABS Layout
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Executive Summary", 
                "📈 Efficient Frontier", 
                "📉 Backtest & Risk", 
                "🕯️ OHLC Analysis",
                "🌪️ Stress Test"
            ])

            # --- TAB 1: EXECUTIVE SUMMARY ---
            with tab1:
                st.title("Portfolio Executive Summary")
                
                # Metrics Row
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Exp. Annual Return", f"{perf[0]:.2%}")
                col2.metric("Annual Volatility", f"{perf[1]:.2%}")
                col3.metric("Sharpe Ratio", f"{perf[2]:.2f}")
                
                # Calculate Portfolio VaR
                port_rets = returns.dot(pd.Series(weights))
                var_95 = np.percentile(port_rets, 5)
                col4.metric("Daily VaR (95%)", f"{var_95:.2%}", delta_color="inverse")

                # Weights Chart
                weights_df = pd.Series(weights).sort_values(ascending=False)
                weights_df = weights_df[weights_df > 0.001] # Filter tiny weights
                
                fig_pie = px.pie(
                    names=weights_df.index, 
                    values=weights_df.values, 
                    title="Optimized Asset Allocation",
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Prism
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            # --- TAB 2: EFFICIENT FRONTIER ---
            with tab2:
                st.subheader("Efficient Frontier & Monte Carlo Simulation")
                
                # Monte Carlo
                n_sims = 2000
                all_weights = np.zeros((n_sims, len(selected_tickers)))
                ret_arr = np.zeros(n_sims)
                vol_arr = np.zeros(n_sims)
                sharpe_arr = np.zeros(n_sims)
                
                mu = expected_returns.mean_historical_return(prices)
                S = risk_models.sample_cov(prices)
                
                for i in range(n_sims):
                    w = np.random.random(len(selected_tickers))
                    w /= np.sum(w)
                    all_weights[i,:] = w
                    ret_arr[i] = np.sum(mu * w)
                    vol_arr[i] = np.sqrt(np.dot(w.T, np.dot(S, w)))
                    sharpe_arr[i] = (ret_arr[i] - rf_rate) / vol_arr[i]
                
                # Plot
                fig_ef = go.Figure()
                fig_ef.add_trace(go.Scatter(
                    x=vol_arr, y=ret_arr, mode='markers',
                    marker=dict(color=sharpe_arr, colorscale='Viridis', showscale=True, size=5),
                    name='Random Portfolios'
                ))
                # Add Optimal Point
                fig_ef.add_trace(go.Scatter(
                    x=[perf[1]], y=[perf[0]], mode='markers',
                    marker=dict(color='red', size=20, symbol='star'),
                    name='Selected Portfolio'
                ))
                fig_ef.update_layout(xaxis_title="Volatility", yaxis_title="Return", height=600)
                st.plotly_chart(fig_ef, use_container_width=True)

            # --- TAB 3: BACKTEST & RISK ---
            with tab3:
                st.subheader("Historical Performance Backtest")
                
                # Cumulative Returns
                cum_ret = (1 + port_rets).cumprod()
                
                # Benchmark (Equal Weight)
                eq_weights = np.array([1/len(selected_tickers)] * len(selected_tickers))
                bench_rets = returns.dot(eq_weights)
                bench_cum = (1 + bench_rets).cumprod()
                
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=cum_ret.index, y=cum_ret, name="Optimized Strategy", line=dict(color='#00ff00')))
                fig_bt.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum, name="Equal Weight Benchmark", line=dict(dash='dash', color='gray')))
                
                st.plotly_chart(fig_bt, use_container_width=True)
                
                # Advanced Metrics Table
                st.subheader("🏆 Advanced Risk/Return Metrics")
                metrics = AdvancedRiskMetrics.calculate_metrics(port_rets, rf_rate)
                
                # Create a nice dataframe for metrics
                m_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
                m_df['Value'] = m_df['Value'].apply(lambda x: f"{x:.4f}" if abs(x) > 0.1 else f"{x:.2%}")
                st.table(m_df)
                
                # Drawdown Plot
                running_max = np.maximum.accumulate(cum_ret)
                drawdown = (cum_ret - running_max) / running_max
                
                fig_dd = px.area(drawdown, title="Underwater Plot (Drawdown)", color_discrete_sequence=['red'])
                st.plotly_chart(fig_dd, use_container_width=True)

            # --- TAB 4: OHLC ANALYSIS ---
            with tab4:
                st.subheader("Candlestick Analysis")
                chart_ticker = st.selectbox("Select Asset to View", selected_tickers)
                
                if chart_ticker in ohlc_data:
                    df_ohlc = ohlc_data[chart_ticker]
                    fig_candle = go.Figure(data=[go.Candlestick(
                        x=df_ohlc.index,
                        open=df_ohlc['Open'],
                        high=df_ohlc['High'],
                        low=df_ohlc['Low'],
                        close=df_ohlc['Close']
                    )])
                    fig_candle.update_layout(title=f"{chart_ticker} Price Action", height=600)
                    st.plotly_chart(fig_candle, use_container_width=True)
                else:
                    st.warning("OHLC data not available for this ticker.")

            # --- TAB 5: STRESS TESTING ---
            with tab5:
                st.subheader("Stress Test Scenarios")
                
                # Define Scenarios
                scenarios = {
                    '2008 Crash (-40% Equities)': -0.40,
                    'Covid Correction (-30%)': -0.30,
                    'Tech Bubble Burst (-20%)': -0.20,
                    'Rate Hike Shock (-10%)': -0.10,
                    'Bull Run (+20%)': 0.20
                }
                
                current_val = 100000 # Assume 100k portfolio
                stress_res = []
                
                for name, shock in scenarios.items():
                    # Calculate Beta of portfolio (simplified against equal weight benchmark)
                    covariance = np.cov(port_rets, bench_rets)[0][1]
                    variance = np.var(bench_rets)
                    beta = covariance / variance
                    
                    # Apply shock adjusted by Beta
                    est_impact = shock * beta
                    new_val = current_val * (1 + est_impact)
                    pnl = new_val - current_val
                    
                    stress_res.append({
                        "Scenario": name,
                        "Market Shock": f"{shock:.0%}",
                        "Est. Portfolio Impact": f"{est_impact:.2%}",
                        "PnL Impact ($100k Inv)": f"${pnl:,.2f}"
                    })
                
                st.table(pd.DataFrame(stress_res))

else:
    st.info("👈 Select assets and parameters in the sidebar, then click 'RUN ANALYSIS'")
