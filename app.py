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

# PyPortfolioOpt imports
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import objective_functions
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
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
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #0e1117; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { background-color: #1e1e1e; border-bottom: 2px solid #4CAF50; }
</style>
""", unsafe_allow_html=True)

# 1. BIST 30 (Turkish Major Stocks)
BIST_30 = [
    'AKBNK.IS', 'ARCLK.IS', 'ASELS.IS', 'BIMAS.IS', 'DOHOL.IS', 'EKGYO.IS', 
    'ENKAI.IS', 'EREGL.IS', 'FROTO.IS', 'GARAN.IS', 'GUBRF.IS', 'HALKB.IS', 
    'HEKTS.IS', 'ISCTR.IS', 'KCHOL.IS', 'KOZAA.IS', 'KOZAL.IS', 'KRDMD.IS', 
    'PETKM.IS', 'PGSUS.IS', 'SAHOL.IS', 'SASA.IS', 'SISE.IS', 'TCELL.IS', 
    'THYAO.IS', 'TKFEN.IS', 'TOASO.IS', 'TSKB.IS', 'TUPRS.IS', 'YKBNK.IS'
]

# 2. Global Major Indices
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
        if not tickers:
            return pd.DataFrame(), {}
        try:
            if isinstance(tickers, str):
                tickers = [tickers]
            
            # Stable download
            data = yf.download(tickers, start=start_date, end=end_date, progress=False, group_by='ticker', threads=False)
            
            prices = pd.DataFrame()
            ohlc_dict = {}

            if len(tickers) == 1:
                ticker = tickers[0]
                df = data
                if isinstance(data.columns, pd.MultiIndex):
                    try: df = data.xs(ticker, axis=1, level=0, drop_level=True)
                    except: pass
                price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
                if price_col in df.columns:
                    prices[ticker] = df[price_col]
                    ohlc_dict[ticker] = df
            else:
                if not isinstance(data.columns, pd.MultiIndex): return pd.DataFrame(), {}
                for ticker in tickers:
                    try:
                        df = data.xs(ticker, axis=1, level=0, drop_level=True)
                        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
                        if price_col in df.columns:
                            prices[ticker] = df[price_col]
                            ohlc_dict[ticker] = df
                    except KeyError: continue
            
            prices = prices.ffill().bfill().dropna(axis=1, how='all')
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
    @st.cache_data(ttl=3600 * 24)
    def get_market_caps(tickers):
        mcaps = {}
        for t in tickers:
            try:
                info = yf.Ticker(t).info
                mcaps[t] = info.get('marketCap', 1e10)
            except:
                mcaps[t] = 1e10
        return pd.Series(mcaps)

# ============================================================================
# ADVANCED RISK METRICS MODULE (UPGRADED)
# ============================================================================

class AdvancedRiskMetrics:
    
    @staticmethod
    def calculate_metrics(returns, risk_free=0.02):
        ann_factor = 252
        total_return = (1 + returns).prod() - 1
        cagr = (1 + total_return) ** (ann_factor / len(returns)) - 1
        volatility = returns.std() * np.sqrt(ann_factor)
        
        excess_returns = returns - (risk_free / ann_factor)
        sharpe = np.sqrt(ann_factor) * excess_returns.mean() / returns.std()
        
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(ann_factor)
        sortino = np.sqrt(ann_factor) * excess_returns.mean() / downside_std if downside_std != 0 else 0
        
        cum_returns = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        max_dd = drawdown.min()
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
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
    def calculate_comprehensive_risk_profile(returns, confidence_levels=[0.95, 0.99]):
        """
        Advanced Risk Engine calculating Parametric, Historical, and Modified (Cornish-Fisher) VaR
        """
        results = {}
        
        # Stats
        mu = returns.mean()
        sigma = returns.std()
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns) # Excess kurtosis
        
        for conf in confidence_levels:
            alpha = 1 - conf
            
            # 1. Parametric VaR (Normal Distribution)
            z_score = stats.norm.ppf(alpha)
            var_param = mu + z_score * sigma
            
            # 2. Historical VaR
            var_hist = np.percentile(returns, alpha * 100)
            
            # 3. Modified VaR (Cornish-Fisher Expansion)
            # Adjusts Z-score based on Skew and Kurtosis
            z_cf = z_score + (z_score**2 - 1)*skew/6 + (z_score**3 - 3*z_score)*kurt/24 - (2*z_score**3 - 5*z_score)*(skew**2)/36
            var_mod = mu + z_cf * sigma
            
            # 4. Expected Shortfall (CVaR) - using Historical for robustness
            cvar_hist = returns[returns <= var_hist].mean()
            
            tag = f"{int(conf*100)}%"
            results[f"Parametric VaR ({tag})"] = var_param
            results[f"Historical VaR ({tag})"] = var_hist
            results[f"Modified VaR ({tag})"] = var_mod
            results[f"CVaR ({tag})"] = cvar_hist
            
        return results, skew, kurt

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
        delta = risk_models.black_litterman.market_implied_risk_aversion(self.prices)
        prior = risk_models.black_litterman.market_implied_prior_returns(market_caps, delta, self.S)
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
                w_arr = np.array(list(weights.values()))
                ret = np.sum(returns.mean() * w_arr) * 252
                vol = np.sqrt(np.dot(w_arr.T, np.dot(returns.cov() * 252, w_arr)))
                sharpe = (ret - rf_rate) / vol
                perf = (ret, vol, sharpe)
            else:
                weights, perf = optimizer.optimize(method, risk_free_rate=rf_rate)

            # Calculate Portfolio Returns Series
            port_rets = returns.dot(pd.Series(weights))
            
            # 3. TABS Layout
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📊 Executive Summary", 
                "📈 Efficient Frontier", 
                "📉 Backtest & Risk", 
                "🕯️ OHLC Analysis",
                "🌪️ Stress Test",
                "⚠️ Advanced VaR Engine"
            ])

            # --- TAB 1: INSTITUTIONAL EXECUTIVE SUMMARY ---
            with tab1:
                st.markdown("## 🏛️ Strategy Tearsheet & Performance Attribution")
                
                # --- A. KPI DASHBOARD ---
                cum_ret = (1 + port_rets).cumprod()
                total_ret = cum_ret.iloc[-1] - 1
                ann_ret = perf[0]
                ann_vol = perf[1]
                sharpe = perf[2]
                
                metrics_basic = AdvancedRiskMetrics.calculate_metrics(port_rets, rf_rate)
                
                st.markdown("""
                <style>
                .kpi-card {background-color: #1e1e1e; border: 1px solid #333; border-radius: 8px; padding: 15px; text-align: center; box-shadow: 2px 2px 10px rgba(0,0,0,0.5);}
                .kpi-title { font-size: 14px; color: #aaa; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 1px;}
                .kpi-value { font-size: 26px; font-weight: bold; color: #fff; }
                .kpi-good { color: #00ff00; }
                .kpi-bad { color: #ff4444; }
                </style>
                """, unsafe_allow_html=True)

                c1, c2, c3, c4 = st.columns(4)
                with c1: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Total Return</div><div class='kpi-value'>{total_ret:.2%}</div></div>", unsafe_allow_html=True)
                with c2: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Annual CAGR</div><div class='kpi-value'>{ann_ret:.2%}</div></div>", unsafe_allow_html=True)
                with c3: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Volatility (Ann)</div><div class='kpi-value'>{ann_vol:.2%}</div></div>", unsafe_allow_html=True)
                with c4: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Sharpe Ratio</div><div class='kpi-value kpi-good'>{sharpe:.2f}</div></div>", unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                c5, c6, c7, c8 = st.columns(4)
                with c5: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Max Drawdown</div><div class='kpi-value kpi-bad'>{metrics_basic['Max Drawdown']:.2%}</div></div>", unsafe_allow_html=True)
                with c6: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Sortino Ratio</div><div class='kpi-value'>{metrics_basic['Sortino Ratio']:.2f}</div></div>", unsafe_allow_html=True)
                with c7: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Calmar Ratio</div><div class='kpi-value'>{metrics_basic['Calmar Ratio']:.2f}</div></div>", unsafe_allow_html=True)
                with c8: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>CVaR (95%)</div><div class='kpi-value kpi-bad'>{metrics_basic['CVaR 95%']:.2%}</div></div>", unsafe_allow_html=True)

                st.markdown("---")

                # --- B. GROWTH & ALLOCATION ---
                col_main_chart, col_alloc = st.columns([2, 1])
                with col_main_chart:
                    bench_series = returns.mean(axis=1) # Equal Weight Benchmark
                    bench_cum = (1 + bench_series).cumprod()
                    fig_growth = go.Figure()
                    fig_growth.add_trace(go.Scatter(x=cum_ret.index, y=cum_ret, name='Strategy', line=dict(color='#00ff00', width=2), fill='tozeroy', fillcolor='rgba(0, 255, 0, 0.05)'))
                    fig_growth.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum, name='Benchmark (Eq Wgt)', line=dict(color='#ffffff', width=1, dash='dot')))
                    fig_growth.update_layout(title="📈 Cumulative Wealth Growth ($1 Investment)", height=400, template="plotly_dark", legend=dict(y=1, x=0, bgcolor='rgba(0,0,0,0)'))
                    st.plotly_chart(fig_growth, use_container_width=True)

                with col_alloc:
                    weights_series = pd.Series(weights).sort_values(ascending=False)
                    weights_series = weights_series[weights_series > 0.001]
                    fig_alloc = px.pie(values=weights_series.values, names=weights_series.index, hole=0.6, title="Asset Allocation", color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig_alloc.update_layout(showlegend=False, height=400, annotations=[dict(text=f"{len(weights_series)}<br>Assets", x=0.5, y=0.5, font_size=20, showarrow=False)])
                    st.plotly_chart(fig_alloc, use_container_width=True)

                # --- C. HEATMAP & ATTRIBUTION ---
                c_heat, c_attr = st.columns(2)
                with c_heat:
                    st.subheader("📅 Monthly Returns Heatmap")
                    monthly_ret = port_rets.resample('M').apply(lambda x: (1 + x).prod() - 1)
                    monthly_ret_df = pd.DataFrame(monthly_ret, columns=['Return'])
                    monthly_ret_df['Year'] = monthly_ret_df.index.year
                    monthly_ret_df['Month'] = monthly_ret_df.index.strftime('%b')
                    heatmap_data = monthly_ret_df.pivot(index='Year', columns='Month', values='Return')
                    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    heatmap_data = heatmap_data.reindex(columns=month_order)
                    fig_heat = px.imshow(heatmap_data, labels=dict(x="Month", y="Year", color="Return"), x=month_order, y=heatmap_data.index, color_continuous_scale="RdYlGn", text_auto='.2%')
                    fig_heat.update_layout(height=400, template="plotly_dark")
                    st.plotly_chart(fig_heat, use_container_width=True)
                
                with c_attr:
                    st.subheader("🔎 Attribution (Top/Bottom)")
                    asset_total_ret = (1 + returns).prod() - 1
                    contribution = asset_total_ret * pd.Series(weights)
                    contribution = contribution.sort_values(ascending=False)
                    top_contrib = contribution.head(5)
                    bot_contrib = contribution.tail(5)
                    
                    fig_top = px.bar(x=top_contrib.values, y=top_contrib.index, orientation='h', title="Top Contributors", labels={'x':'Contribution', 'y':'Asset'}, color_discrete_sequence=['#00ff00'])
                    fig_top.update_layout(height=180, margin=dict(l=0,r=0,t=30,b=0))
                    st.plotly_chart(fig_top, use_container_width=True)
                    
                    fig_bot = px.bar(x=bot_contrib.values, y=bot_contrib.index, orientation='h', title="Top Detractors", labels={'x':'Contribution', 'y':'Asset'}, color_discrete_sequence=['#ff4444'])
                    fig_bot.update_layout(height=180, margin=dict(l=0,r=0,t=30,b=0))
                    st.plotly_chart(fig_bot, use_container_width=True)

            # --- TAB 2: EFFICIENT FRONTIER ---
            with tab2:
                st.subheader("Efficient Frontier & Monte Carlo Simulation")
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
                
                fig_ef = go.Figure()
                fig_ef.add_trace(go.Scatter(x=vol_arr, y=ret_arr, mode='markers', marker=dict(color=sharpe_arr, colorscale='Viridis', showscale=True, size=5), name='Random Portfolios'))
                fig_ef.add_trace(go.Scatter(x=[perf[1]], y=[perf[0]], mode='markers', marker=dict(color='red', size=20, symbol='star'), name='Selected Portfolio'))
                fig_ef.update_layout(xaxis_title="Volatility", yaxis_title="Return", height=600)
                st.plotly_chart(fig_ef, use_container_width=True)

            # --- TAB 3: BACKTEST & RISK ---
            with tab3:
                st.subheader("Historical Performance Backtest")
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=cum_ret.index, y=cum_ret, name="Optimized Strategy", line=dict(color='#00ff00')))
                fig_bt.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum, name="Equal Weight Benchmark", line=dict(dash='dash', color='gray')))
                st.plotly_chart(fig_bt, use_container_width=True)
                
                st.subheader("🏆 Advanced Risk/Return Metrics")
                m_df = pd.DataFrame.from_dict(metrics_basic, orient='index', columns=['Value'])
                m_df['Value'] = m_df['Value'].apply(lambda x: f"{x:.4f}" if abs(x) > 0.1 else f"{x:.2%}")
                st.table(m_df)
                
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
                    fig_candle = go.Figure(data=[go.Candlestick(x=df_ohlc.index, open=df_ohlc['Open'], high=df_ohlc['High'], low=df_ohlc['Low'], close=df_ohlc['Close'])])
                    fig_candle.update_layout(title=f"{chart_ticker} Price Action", height=600)
                    st.plotly_chart(fig_candle, use_container_width=True)
                else:
                    st.warning("OHLC data not available for this ticker.")

            # --- TAB 5: STRESS TESTING ---
            with tab5:
                st.subheader("Stress Test Scenarios")
                scenarios = {
                    '2008 Crash (-40% Equities)': -0.40,
                    'Covid Correction (-30%)': -0.30,
                    'Tech Bubble Burst (-20%)': -0.20,
                    'Rate Hike Shock (-10%)': -0.10,
                    'Bull Run (+20%)': 0.20
                }
                current_val = 100000 
                stress_res = []
                bench_rets_series = returns.mean(axis=1)
                for name, shock in scenarios.items():
                    covariance = np.cov(port_rets, bench_rets_series)[0][1]
                    variance = np.var(bench_rets_series)
                    beta = covariance / variance
                    est_impact = shock * beta
                    new_val = current_val * (1 + est_impact)
                    pnl = new_val - current_val
                    stress_res.append({"Scenario": name, "Market Shock": f"{shock:.0%}", "Est. Portfolio Impact": f"{est_impact:.2%}", "PnL Impact ($100k Inv)": f"${pnl:,.2f}"})
                st.table(pd.DataFrame(stress_res))

            # --- TAB 6: ADVANCED VAR ENGINE ---
            with tab6:
                st.markdown("## ⚠️ Advanced Value at Risk (VaR) Engine")
                st.info("This module calculates VaR using three distinct methodologies to account for 'Fat Tails' (extreme events) often missed by standard models.")
                
                # 1. Calculate Advanced Metrics
                var_metrics, skew, kurt = AdvancedRiskMetrics.calculate_comprehensive_risk_profile(port_rets)
                
                # 2. Distribution Plot
                col_dist, col_table = st.columns([2, 1])
                
                with col_dist:
                    st.subheader("Return Distribution Analysis")
                    fig_dist = go.Figure()
                    # Histogram
                    fig_dist.add_trace(go.Histogram(x=port_rets, nbinsx=100, name='Returns', histnorm='probability density', marker_color='#1f77b4', opacity=0.7))
                    # KDE (Approximate using Normal Curve for viz)
                    x_range = np.linspace(port_rets.min(), port_rets.max(), 100)
                    fig_dist.add_trace(go.Scatter(x=x_range, y=stats.norm.pdf(x_range, port_rets.mean(), port_rets.std()), mode='lines', name='Normal Dist', line=dict(color='white', dash='dash')))
                    
                    # Add VaR Lines (95%)
                    var_95_hist = var_metrics['Historical VaR (95%)']
                    var_95_mod = var_metrics['Modified VaR (95%)']
                    
                    fig_dist.add_vline(x=var_95_hist, line_width=2, line_dash="dash", line_color="orange", annotation_text="Hist VaR 95%")
                    fig_dist.add_vline(x=var_95_mod, line_width=2, line_dash="dot", line_color="red", annotation_text="Mod VaR 95%")
                    
                    fig_dist.update_layout(title="Portfolio Returns Distribution vs Normal", xaxis_title="Daily Return", yaxis_title="Density", height=450, template="plotly_dark")
                    st.plotly_chart(fig_dist, use_container_width=True)

                with col_table:
                    st.subheader("Risk Statistics")
                    st.metric("Skewness", f"{skew:.4f}", help="Negative skew means frequent small gains and few extreme losses.")
                    st.metric("Kurtosis (Excess)", f"{kurt:.4f}", help="High kurtosis (>0) indicates 'fat tails' - higher risk of extreme events.")
                    
                    st.markdown("### VaR Breakdown")
                    var_df = pd.DataFrame.from_dict(var_metrics, orient='index', columns=['Value'])
                    var_df['Value'] = var_df['Value'].apply(lambda x: f"{x:.4%}")
                    st.table(var_df)

                # 3. VaR Explanation
                st.markdown("### 📘 Methodology Guide")
                st.markdown("""
                * **Parametric VaR:** Assumes returns follow a perfect Bell Curve. Often underestimates risk in crises.
                * **Historical VaR:** Looks at the actual past worst days. Good, but limited by history length.
                * **Modified (Cornish-Fisher) VaR:** **The most advanced metric here.** It adjusts the Normal VaR using the Skewness and Kurtosis calculated above to account for "Black Swan" potential.
                * **CVaR (Expected Shortfall):** The average loss *if* the VaR threshold is breached. This is the "average worst case scenario".
                """)

else:
    st.info("👈 Select assets and parameters in the sidebar, then click 'RUN ANALYSIS'")
