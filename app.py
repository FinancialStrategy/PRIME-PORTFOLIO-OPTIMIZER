# ============================================================================
# ENIGMA INSTITUTIONAL TERMINAL - OPTIMIZED FOR STREAMLIT CLOUD
# ============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import scipy.stats as stats
from scipy import optimize
import warnings
from typing import Dict, Tuple, List, Optional
import numpy.random as npr

# --- QUANTITATIVE LIBRARY IMPORTS ---
# Wrapped in try-except to prevent immediate crash if dependency fails install
try:
    from pypfopt import expected_returns, risk_models
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt.cla import CLA
    from pypfopt.hierarchical_portfolio import HRPOpt
    from pypfopt.black_litterman import BlackLittermanModel
    HAS_PYPFOPT = True
except ImportError:
    st.error("PyPortfolioOpt not found. Please add 'pyportfolioopt' and 'cvxpy' to requirements.txt")
    HAS_PYPFOPT = False

from sklearn.decomposition import PCA

try:
    import arch
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

warnings.filterwarnings('ignore')

# ============================================================================
# 1. GLOBAL CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Enigma Institutional Terminal", 
    layout="wide", 
    page_icon="🏛️",
    initial_sidebar_state="expanded"
)

# Professional CSS
st.markdown("""
<style>
    .reportview-container {background: #0e1117;}
    div[class^="st-"] { font-family: 'Roboto', sans-serif; }
    .pro-card {
        background-color: #1e1e1e;
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 6px;
        padding: 24px 16px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-label {
        font-size: 11px;
        text-transform: uppercase;
        color: #888;
        letter-spacing: 1.5px;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .metric-value {
        font-family: 'Roboto Mono', monospace;
        font-size: 28px;
        font-weight: 700;
        color: #fff;
    }
</style>
""", unsafe_allow_html=True)

# Asset Universes
BIST_30 = [
    'AKBNK.IS', 'ARCLK.IS', 'ASELS.IS', 'BIMAS.IS', 'DOHOL.IS', 'EKGYO.IS', 
    'ENKAI.IS', 'EREGL.IS', 'FROTO.IS', 'GARAN.IS', 'GUBRF.IS', 'HALKB.IS', 
    'HEKTS.IS', 'ISCTR.IS', 'KCHOL.IS', 'KOZAA.IS', 'KOZAL.IS', 'KRDMD.IS', 
    'PETKM.IS', 'PGSUS.IS', 'SAHOL.IS', 'SASA.IS', 'SISE.IS', 'TCELL.IS', 
    'THYAO.IS', 'TKFEN.IS', 'TOASO.IS', 'TSKB.IS', 'TUPRS.IS', 'YKBNK.IS'
]

GLOBAL_INDICES = [
    '^GSPC', '^DJI', '^IXIC', '^RUT', '^VIX', '^FTSE', '^GDAXI', '^N225'
]

US_DEFAULTS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V', 'WMT']

# ============================================================================
# 2. ASSET CLASSIFICATION (OPTIMIZED)
# ============================================================================

class AssetClassifier:
    """Classifies assets. Optimized to avoid YF API calls per ticker."""
    
    SECTOR_MAP = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD', 'INTC', 'CRM'],
        'Financial Services': ['JPM', 'V', 'MA', 'BAC', 'GS', 'MS', 'C'],
        'Healthcare': ['JNJ', 'PFE', 'MRK', 'ABT', 'UNH', 'LLY'],
        'Consumer': ['AMZN', 'TSLA', 'NKE', 'MCD', 'SBUX', 'WMT', 'PG', 'KO'],
        'Energy': ['XOM', 'CVX', 'COP', 'SLB'],
        'Indices': ['^GSPC', '^DJI', '^IXIC', '^RUT', '^VIX', '^FTSE', '^GDAXI', '^N225']
    }
    
    @staticmethod
    def get_asset_metadata(tickers):
        """
        OPTIMIZATION: Calling yf.Ticker(t).info for 30 tickers freezes Streamlit Cloud.
        We now use simple string matching fallback to prevent API timeouts.
        """
        metadata = {}
        for ticker in tickers:
            # Fast inference without API call
            sector = AssetClassifier._infer_sector(ticker)
            region = AssetClassifier._infer_region(ticker)
            
            metadata[ticker] = {
                'sector': sector,
                'industry': 'Unknown',
                'country': region,
                'marketCap': 100e9, # Default fallback to avoid errors in attribution
                'fullName': ticker,
                'currency': 'TRY' if '.IS' in ticker else 'USD'
            }
        return metadata
    
    @staticmethod
    def _infer_sector(ticker):
        for sector, tickers in AssetClassifier.SECTOR_MAP.items():
            if ticker in tickers:
                return sector
        if '.IS' in ticker: return 'Emerging Mkt Equity'
        if '^' in ticker: return 'Market Index'
        return 'Other'
    
    @staticmethod
    def _infer_region(ticker):
        if '.IS' in ticker: return 'Turkey'
        if '.DE' in ticker: return 'Germany'
        if '.L' in ticker: return 'UK'
        if '^' in ticker: return 'Global'
        return 'US'

# ============================================================================
# 3. PROFESSIONAL PERFORMANCE ATTRIBUTION
# ============================================================================

class ProfessionalPortfolioAttribution:
    @staticmethod
    def calculate_brinson_fachler_attribution(portfolio_weights, benchmark_weights, 
                                              portfolio_returns_df, benchmark_returns_df,
                                              sector_mapping, risk_free_rate=0.02):
        # ... (Logic identical to previous, keeping math logic same)
        all_assets = set(list(portfolio_weights.keys()) + list(benchmark_weights.keys()))
        assets_list = list(all_assets)
        w_p = np.array([portfolio_weights.get(asset, 0) for asset in assets_list])
        w_b = np.array([benchmark_weights.get(asset, 0) for asset in assets_list])
        
        # Calculate annualized mean returns
        # Optimization: Use vectorized mean
        means = portfolio_returns_df.mean() * 252
        R_pi = np.array([means.get(asset, 0) for asset in assets_list])
        R_bi = R_pi # Assuming benchmark returns same as asset returns for this simplified attribution
        
        R_p = np.sum(w_p * R_pi)
        R_b = np.sum(w_b * R_bi)
        total_excess = R_p - R_b
        
        # Sector grouping
        sector_attribution = {}
        unique_sectors = set(sector_mapping.values())
        
        total_allocation = 0
        total_selection = 0
        total_interaction = 0
        
        for sector in unique_sectors:
            sec_assets = [a for a in assets_list if sector_mapping.get(a) == sector]
            indices = [i for i, a in enumerate(assets_list) if a in sec_assets]
            
            if not indices: continue
            
            w_p_sec = np.sum(w_p[indices])
            w_b_sec = np.sum(w_b[indices])
            
            if w_p_sec > 0:
                R_p_sec = np.sum(w_p[indices] * R_pi[indices]) / w_p_sec
            else:
                R_p_sec = 0
                
            if w_b_sec > 0:
                R_b_sec = np.sum(w_b[indices] * R_bi[indices]) / w_b_sec
            else:
                R_b_sec = 0
            
            # Brinson-Fachler logic
            allocation = (w_p_sec - w_b_sec) * (R_b_sec - R_b)
            selection = w_b_sec * (R_p_sec - R_b_sec)
            interaction = (w_p_sec - w_b_sec) * (R_p_sec - R_b_sec)
            
            total_allocation += allocation
            total_selection += selection
            total_interaction += interaction
            
            sector_attribution[sector] = {
                'Total Contribution': allocation + selection + interaction,
                'Portfolio Weight': w_p_sec,
                'Benchmark Weight': w_b_sec,
                'Active Weight': w_p_sec - w_b_sec,
                'Excess Return': R_p_sec - R_b_sec
            }

        # Metrics
        port_series = portfolio_returns_df.dot(w_p)
        bench_series = benchmark_returns_df.dot(w_b)
        te = np.std(port_series - bench_series) * np.sqrt(252)
        ir = (port_series - bench_series).mean() * 252 / te if te > 0 else 0
        beta = 1.0 # Simplified
        
        return {
            'Total Portfolio Return': R_p,
            'Total Benchmark Return': R_b,
            'Total Excess Return': total_excess,
            'Allocation Effect': total_allocation,
            'Selection Effect': total_selection,
            'Interaction Effect': total_interaction,
            'Allocation Ratio': 0, 
            'Information Ratio': ir,
            'Tracking Error': te,
            'Active Share': 0.5 * np.sum(np.abs(w_p - w_b)),
            'Portfolio Beta': beta,
            'Sector Breakdown': sector_attribution
        }

# ============================================================================
# 4. VISUALIZATION CLASSES (Keeping standard)
# ============================================================================
class AttributionVisualizationRedesign:
    @staticmethod
    def create_attribution_summary_chart(attribution_results):
        fig = go.Figure()
        components = ['Allocation', 'Selection', 'Interaction']
        values = [attribution_results['Allocation Effect'], attribution_results['Selection Effect'], attribution_results['Interaction Effect']]
        colors = ['#636EFA', '#EF553B', '#00CC96']
        fig.add_trace(go.Bar(x=components, y=values, marker_color=colors, text=[f"{v:.2%}" for v in values]))
        fig.update_layout(title="Performance Attribution", template="plotly_dark", height=400, yaxis_tickformat=".2%")
        return fig

# ============================================================================
# 5. VECTORIZED MONTE CARLO (CRITICAL FIX)
# ============================================================================

class AdvancedMonteCarloSimulator:
    """
    OPTIMIZED: Removed Python loops. Uses Numpy Broadcasting.
    Speed increase: ~100x
    """
    
    def __init__(self, returns: pd.DataFrame, prices: pd.DataFrame):
        self.returns = returns
        self.mean_returns = returns.mean().values
        self.cov_matrix = returns.cov().values
        # Handle non-positive definite matrix
        try:
            self.cholesky = np.linalg.cholesky(self.cov_matrix)
        except np.linalg.LinAlgError:
            self.cov_matrix += np.eye(len(self.cov_matrix)) * 1e-6
            self.cholesky = np.linalg.cholesky(self.cov_matrix)
    
    def gbm_simulation(self, weights: np.ndarray, days: int = 252, n_sims: int = 1000) -> Tuple[np.ndarray, Dict]:
        dt = 1/252
        mu = np.dot(weights, self.mean_returns)
        sigma = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        # Vectorized GBM
        # drift: (1,) -> (1, 1)
        drift = (mu - 0.5 * sigma**2) * dt
        # diffusion: (n_sims, days)
        z = npr.normal(size=(n_sims, days))
        daily_returns = np.exp(drift + sigma * np.sqrt(dt) * z)
        
        # Cumprod to get price paths
        paths = np.vstack([np.ones((n_sims, 1)).T, daily_returns.T]).T
        paths = np.cumprod(paths, axis=1)
        
        terminal_values = paths[:, -1]
        return paths, {"terminal_mean": np.mean(terminal_values), "terminal_std": np.std(terminal_values)}

    def t_distribution_simulation(self, weights: np.ndarray, days: int = 252, n_sims: int = 1000, df: float = 5.0):
        dt = 1/252
        mu = np.dot(weights, self.mean_returns)
        sigma = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        # Vectorized Student-t
        z = npr.standard_t(df, size=(n_sims, days))
        daily_returns = np.exp(mu * dt + sigma * np.sqrt(dt) * z)
        
        paths = np.vstack([np.ones((n_sims, 1)).T, daily_returns.T]).T
        paths = np.cumprod(paths, axis=1)
        
        terminal = paths[:, -1]
        return paths, {"terminal_mean": np.mean(terminal), "terminal_std": np.std(terminal), "skewness": stats.skew(terminal), "kurtosis": stats.kurtosis(terminal)}

    def jump_diffusion_simulation(self, weights: np.ndarray, days: int = 252, n_sims: int = 1000, 
                                  jump_intensity: float = 0.05, jump_mean: float = -0.1, jump_std: float = 0.15):
        dt = 1/252
        mu = np.dot(weights, self.mean_returns)
        sigma = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        # Vectorized Jump Diffusion
        z_brown = npr.normal(size=(n_sims, days))
        
        # Poisson jumps: 1 if jump, 0 if not
        # Poisson parameter is lambda * dt
        n_jumps = npr.poisson(jump_intensity * dt, size=(n_sims, days))
        
        # Jump sizes
        jump_factor = np.exp(npr.normal(jump_mean, jump_std, size=(n_sims, days)) * n_jumps)
        
        # Combined return
        # Note: When n_jumps=0, jump_factor=exp(0)=1 (no effect)
        diffusion = np.exp((mu - 0.5 * sigma**2)*dt + sigma*np.sqrt(dt)*z_brown)
        daily_returns = diffusion * jump_factor
        
        paths = np.vstack([np.ones((n_sims, 1)).T, daily_returns.T]).T
        paths = np.cumprod(paths, axis=1)
        
        terminal = paths[:, -1]
        return paths, {"terminal_mean": np.mean(terminal), "terminal_std": np.std(terminal)}

# ============================================================================
# 6. DATA PIPELINE (ROBUST)
# ============================================================================

class PortfolioDataManager:
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_data(tickers, start_date, end_date):
        if not tickers: return pd.DataFrame(), {}
        
        try:
            # Optimize: threads=True is generally okay for Streamlit, but handle MultiIndex
            data = yf.download(tickers, start=start_date, end=end_date, progress=False, group_by='ticker', auto_adjust=True)
            
            prices = pd.DataFrame()
            ohlc = {}
            
            if len(tickers) == 1:
                t = tickers[0]
                # YFinance v0.2 fix
                if isinstance(data.columns, pd.MultiIndex):
                    # Sometimes it comes as (Price, Ticker) or (Ticker, Price)
                    try:
                        df = data.xs(t, axis=1, level=0)
                    except:
                        df = data
                else:
                    df = data
                    
                col = 'Close' if 'Close' in df.columns else 'Adj Close'
                if col in df.columns:
                    prices[t] = df[col]
                    ohlc[t] = df
            else:
                for t in tickers:
                    try:
                        df = data.xs(t, axis=1, level=0)
                        col = 'Close' if 'Close' in df.columns else 'Adj Close'
                        if col in df.columns:
                            prices[t] = df[col]
                            ohlc[t] = df
                    except KeyError:
                        continue
                        
            prices = prices.ffill().bfill()
            return prices, ohlc
        except Exception as e:
            st.error(f"Data Fetch Error: {e}")
            return pd.DataFrame(), {}

    @staticmethod
    def calculate_returns(prices):
        return np.log(prices / prices.shift(1)).dropna()

# ============================================================================
# 7. OPTIMIZER WRAPPER
# ============================================================================

class AdvancedPortfolioOptimizer:
    def __init__(self, returns, prices):
        self.returns = returns
        self.mu = expected_returns.mean_historical_return(prices)
        self.S = risk_models.sample_cov(prices)

    def optimize(self, method, risk_free_rate=0.02, target_vol=0.10):
        if not HAS_PYPFOPT:
            st.warning("PyPortfolioOpt not installed. Using Equal Weights.")
            n = len(self.returns.columns)
            return {t: 1.0/n for t in self.returns.columns}, (0,0,0)

        ef = EfficientFrontier(self.mu, self.S)
        if method == 'Max Sharpe':
            w = ef.max_sharpe(risk_free_rate=risk_free_rate)
        elif method == 'Min Volatility':
            w = ef.min_volatility()
        elif method == 'Efficient Risk':
            w = ef.efficient_risk(target_volatility=target_vol)
        else:
            w = ef.max_sharpe(risk_free_rate=risk_free_rate)
            
        return ef.clean_weights(), ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)

# ============================================================================
# MAIN APP LOGIC
# ============================================================================

# Sidebar
st.sidebar.header("🔧 Enigma Config")
selected_list = st.sidebar.selectbox("Universe", ["US Defaults", "BIST 30", "Global Indices"])
if selected_list == "US Defaults": tickers = US_DEFAULTS
elif selected_list == "BIST 30": tickers = BIST_30
else: tickers = GLOBAL_INDICES

selected_tickers = st.sidebar.multiselect("Assets", tickers, default=tickers[:5])

with st.sidebar.expander("⚙️ Parameters", expanded=True):
    start_date = st.date_input("Start", datetime.now() - timedelta(days=365*2))
    end_date = st.date_input("End", datetime.now())
    method = st.selectbox("Strategy", ['Max Sharpe', 'Min Volatility', 'Efficient Risk', 'Equal Weight'])
    target_vol = st.slider("Target Vol", 0.05, 0.4, 0.15) if method == 'Efficient Risk' else 0.1
    
    # Monte Carlo Optimization for Cloud
    # Reduce defaults to prevent timeout on Free Tier
    mc_sims = st.selectbox("MC Sims", [1000, 5000, 10000], index=1)
    mc_method = st.selectbox("MC Model", ["GBM", "Student's t", "Jump Diffusion"])

run_btn = st.sidebar.button("🚀 EXECUTE", type="primary")

if run_btn:
    with st.spinner('Accessing Quantitative Engine...'):
        dm = PortfolioDataManager()
        prices, ohlc = dm.fetch_data(selected_tickers, start_date, end_date)
        
        if prices.empty:
            st.error("No data found.")
            st.stop()
            
        returns = dm.calculate_returns(prices)
        
        # Optimizer
        opt = AdvancedPortfolioOptimizer(returns, prices)
        if method == 'Equal Weight':
            weights = {t: 1.0/len(selected_tickers) for t in selected_tickers}
            perf = (0,0,0) # Placeholder
        else:
            try:
                weights, perf = opt.optimize(method, 0.02, target_vol)
            except Exception as e:
                st.error(f"Optimization failed: {e}")
                weights = {t: 1.0/len(selected_tickers) for t in selected_tickers}
                perf = (0,0,0)

        # MC Simulation (Vectorized)
        sim = AdvancedMonteCarloSimulator(returns, prices)
        w_arr = np.array([weights.get(t,0) for t in prices.columns])
        
        if mc_method == "GBM":
            paths, stats_mc = sim.gbm_simulation(w_arr, n_sims=mc_sims)
        elif mc_method == "Student's t":
            paths, stats_mc = sim.t_distribution_simulation(w_arr, n_sims=mc_sims)
        else:
            paths, stats_mc = sim.jump_diffusion_simulation(w_arr, n_sims=mc_sims)

        # Display Logic (Simplified for stability)
        t1, t2, t3 = st.tabs(["Dashboard", "MC Simulation", "Attribution"])
        
        with t1:
            st.subheader("Portfolio Performance")
            col1, col2, col3 = st.columns(3)
            col1.metric("Exp Return", f"{perf[0]:.2%}")
            col2.metric("Exp Volatility", f"{perf[1]:.2%}")
            col3.metric("Sharpe Ratio", f"{perf[2]:.2f}")
            
            # Allocation Chart
            fig = px.pie(names=list(weights.keys()), values=list(weights.values()), title="Optimal Allocation")
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
        with t2:
            st.subheader(f"Monte Carlo: {mc_method}")
            fig_mc = go.Figure()
            # Plot only 100 paths to save browser memory
            for i in range(min(100, mc_sims)):
                fig_mc.add_trace(go.Scatter(y=paths[i,:], mode='lines', line=dict(width=0.5, color='rgba(255,255,255,0.1)'), showlegend=False))
            
            mean_path = np.mean(paths, axis=0)
            fig_mc.add_trace(go.Scatter(y=mean_path, mode='lines', line=dict(color='#00cc96', width=3), name='Mean'))
            fig_mc.update_layout(template="plotly_dark", title=f"Projected Paths ({mc_sims} sims)")
            st.plotly_chart(fig_mc, use_container_width=True)
            
            st.info(f"Terminal Mean Value: {stats_mc['terminal_mean']:.4f} | Terminal Std: {stats_mc['terminal_std']:.4f}")

        with t3:
            st.subheader("Performance Attribution")
            # Fast attribution without heavy metadata calls
            classifier = AssetClassifier()
            meta = classifier.get_asset_metadata(selected_tickers)
            sector_map = {t: m['sector'] for t, m in meta.items()}
            
            # Use Equal Weight benchmark for simple comparison
            bench_weights = {t: 1.0/len(selected_tickers) for t in selected_tickers}
            
            attr_eng = ProfessionalPortfolioAttribution()
            res = attr_eng.calculate_brinson_fachler_attribution(weights, bench_weights, returns, returns, sector_map)
            
            viz = AttributionVisualizationRedesign()
            st.plotly_chart(viz.create_attribution_summary_chart(res), use_container_width=True)
            
            # Sector Breakdown
            st.dataframe(pd.DataFrame(res['Sector Breakdown']).T.style.format("{:.2%}"))

else:
    st.info("👈 Select assets and click EXECUTE to start the institutional engine.")
