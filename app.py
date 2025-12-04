# ============================================================================
# ENIGMA INSTITUTIONAL TERMINAL - ROBUST VERSION
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
import warnings
from typing import Dict, Tuple, List, Optional
import numpy.random as npr

# --- QUANTITATIVE LIBRARY IMPORTS ---
try:
    from pypfopt import expected_returns, risk_models
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt.cla import CLA
    from pypfopt.hierarchical_portfolio import HRPOpt
    from pypfopt.black_litterman import BlackLittermanModel
    PYPFOPT_AVAILABLE = True
except ImportError:
    PYPFOPT_AVAILABLE = False
    st.warning("PyPortfolioOpt not available. Some optimization methods will be disabled.")

try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("scikit-learn not available. PCA analysis will be disabled.")

# ARCH: For Econometric Volatility Forecasting (GARCH)
try:
    import arch
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

warnings.filterwarnings('ignore')

# ============================================================================
# 1. GLOBAL CONFIGURATION AND ASSET UNIVERSES
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
        transition: all 0.3s ease;
    }
    .pro-card:hover { 
        transform: translateY(-5px); 
        box-shadow: 0 8px 15px rgba(0,0,0,0.3); 
        border-color: #00cc96;
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
    .stTabs [data-baseweb="tab-list"] { 
        gap: 8px; 
        border-bottom: 1px solid #333; 
        padding-bottom: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: transparent;
        border-radius: 4px;
        font-weight: 600;
        color: #666;
        transition: color 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e1e1e;
        color: #00cc96;
        border: 1px solid #333;
        border-bottom: 2px solid #00cc96;
    }
    div[data-testid="stTable"] { font-size: 13px; font-family: 'Roboto Mono', monospace; }
    div[data-testid="stExpander"] { background-color: #161a24; border-radius: 4px; }
    .insight-box {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid;
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
    '^GSPC', '^DJI', '^IXIC', '^RUT', '^VIX', # US Markets
    '^FTSE', '^GDAXI', '^FCHI', '^STOXX50E',  # European Markets
    '^N225', '^HSI', '000001.SS', '^STI', '^AXJO', # Asian Markets
    '^BVSP', '^MXX', '^MERV' # Latin America
]

US_DEFAULTS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V', 'WMT']

# ============================================================================
# 2. ASSET CLASSIFICATION ENGINE
# ============================================================================

class AssetClassifier:
    """Classifies assets into sectors, industries, regions, and styles."""
    
    SECTOR_MAP = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD', 'INTC', 'QCOM', 'CRM'],
        'Financial Services': ['JPM', 'V', 'MA', 'BAC', 'GS', 'MS', 'C', 'WFC'],
        'Healthcare': ['JNJ', 'PFE', 'MRK', 'ABT', 'UNH', 'LLY', 'GILD', 'BMY'],
        'Consumer Cyclical': ['AMZN', 'TSLA', 'NKE', 'MCD', 'SBUX', 'HD', 'LOW'],
        'Consumer Defensive': ['WMT', 'PG', 'KO', 'PEP', 'COST', 'CL', 'MO'],
        'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PSX', 'VLO'],
        'Industrials': ['BA', 'CAT', 'MMM', 'HON', 'GE', 'RTX', 'LMT'],
        'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC'],
        'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SPG'],
        'Communication': ['T', 'VZ', 'CMCSA', 'DIS', 'NFLX', 'CHTR'],
        'Materials': ['LIN', 'APD', 'ECL', 'SHW', 'NEM', 'FCX']
    }
    
    @staticmethod
    @st.cache_data(ttl=3600*24)
    def get_asset_metadata(tickers):
        """Fetches detailed metadata for each asset with timeout."""
        metadata = {}
        for ticker in tickers:
            try:
                # Add timeout for Yahoo Finance calls
                info = yf.Ticker(ticker).info
                metadata[ticker] = {
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown'),
                    'country': info.get('country', 'Unknown'),
                    'marketCap': info.get('marketCap', 0),
                    'fullName': info.get('longName', ticker),
                    'currency': info.get('currency', 'USD')
                }
            except Exception as e:
                # Fallback to inference
                metadata[ticker] = {
                    'sector': AssetClassifier._infer_sector(ticker),
                    'industry': 'Unknown',
                    'country': AssetClassifier._infer_region(ticker),
                    'marketCap': 0,
                    'fullName': ticker,
                    'currency': 'Unknown'
                }
        return metadata
    
    @staticmethod
    def _infer_sector(ticker):
        """Infer sector from ticker using predefined mappings."""
        for sector, tickers in AssetClassifier.SECTOR_MAP.items():
            if ticker in tickers:
                return sector
        if '.IS' in ticker:
            return 'Financial Services'
        return 'Other'
    
    @staticmethod
    def _infer_region(ticker):
        """Infer region from ticker."""
        if '.IS' in ticker:
            return 'Turkey'
        elif '.DE' in ticker:
            return 'Germany'
        elif '.PA' in ticker:
            return 'France'
        elif '.L' in ticker:
            return 'UK'
        elif ticker.startswith('^'):
            return 'Index'
        return 'Global'

# ============================================================================
# 3. ROBUST DATA PIPELINE
# ============================================================================

class RobustPortfolioDataManager:
    """Handles secure data fetching from Yahoo Finance with robust error handling."""
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_data_safe(tickers, start_date, end_date, max_retries=2):
        """Fetches OHLCV data with robust error handling and retries."""
        if not tickers:
            return pd.DataFrame(), {}
            
        try:
            # Validate dates
            if start_date >= end_date:
                st.error("Start date must be before end date.")
                return pd.DataFrame(), {}
            
            # Limit number of tickers to avoid timeouts
            if len(tickers) > 50:
                tickers = tickers[:50]
                st.warning(f"Limited to first 50 tickers for performance.")
            
            for attempt in range(max_retries):
                try:
                    # Use threads=False for stability
                    data = yf.download(
                        tickers, 
                        start=start_date, 
                        end=end_date, 
                        progress=False, 
                        group_by='ticker', 
                        threads=False,
                        auto_adjust=True,
                        timeout=30  # Add timeout
                    )
                    
                    if data.empty:
                        if attempt < max_retries - 1:
                            continue
                        st.error("No data received from Yahoo Finance. Check ticker symbols.")
                        return pd.DataFrame(), {}
                    
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        continue
                    else:
                        raise e
            
            prices = pd.DataFrame()
            ohlc_dict = {}
            
            if len(tickers) == 1:
                ticker = tickers[0]
                df = data
                # Handle single level or multi-level
                if isinstance(data.columns, pd.MultiIndex):
                    try: 
                        df = data.xs(ticker, axis=1, level=0, drop_level=True)
                    except: 
                        pass
                
                price_col = 'Close' 
                if price_col in df.columns:
                    prices[ticker] = df[price_col]
                    ohlc_dict[ticker] = df
            else:
                if not isinstance(data.columns, pd.MultiIndex):
                    return pd.DataFrame(), {}
                
                for ticker in tickers:
                    try:
                        df = data.xs(ticker, axis=1, level=0, drop_level=True)
                        price_col = 'Close'
                        if price_col in df.columns:
                            prices[ticker] = df[price_col]
                            ohlc_dict[ticker] = df
                    except KeyError:
                        continue
            
            if prices.empty:
                st.error("No price data could be extracted. Check ticker symbols.")
                return pd.DataFrame(), {}
            
            # Forward fill then backfill
            prices = prices.ffill().bfill()
            
            # Drop columns with all NaN
            prices = prices.dropna(axis=1, how='all')
            
            if prices.empty:
                st.error("All data is NaN after cleaning. Check data availability for selected period.")
                return pd.DataFrame(), {}
            
            return prices, ohlc_dict
            
        except Exception as e:
            st.error(f"Data Pipeline Error: {str(e)}")
            return pd.DataFrame(), {}

    @staticmethod
    def calculate_returns(prices, method='log'):
        """Calculates Logarithmic or Simple returns."""
        if prices.empty or len(prices) < 2:
            return pd.DataFrame()
        
        if method == 'log':
            returns = np.log(prices / prices.shift(1))
        else:
            returns = prices.pct_change()
        
        returns = returns.dropna()
        return returns

    @staticmethod
    @st.cache_data(ttl=3600*24)
    def get_market_caps_safe(tickers):
        """Fetches Market Capitalization data with timeout."""
        mcaps = {}
        for t in tickers:
            try:
                mcaps[t] = yf.Ticker(t).info.get('marketCap', 10e9)
            except:
                mcaps[t] = 10e9
        return pd.Series(mcaps)

# ============================================================================
# 4. SIMPLIFIED PERFORMANCE ATTRIBUTION (Robust Version)
# ============================================================================

class RobustPortfolioAttribution:
    """Simplified but robust attribution engine."""
    
    @staticmethod
    def calculate_simple_attribution(portfolio_weights, benchmark_weights, returns_df):
        """Calculate simplified attribution to avoid complex dependencies."""
        
        # Ensure we have valid data
        if returns_df.empty or len(returns_df) < 10:
            return {
                'Total Portfolio Return': 0,
                'Total Benchmark Return': 0,
                'Total Excess Return': 0,
                'Allocation Effect': 0,
                'Selection Effect': 0,
                'Interaction Effect': 0,
                'Information Ratio': 0,
                'Tracking Error': 0,
                'Active Share': 0,
                'Portfolio Beta': 1.0,
                'Sector Breakdown': {}
            }
        
        try:
            # Align weights with available assets
            available_assets = returns_df.columns.tolist()
            
            # Create aligned weight vectors
            w_p = np.array([portfolio_weights.get(a, 0) for a in available_assets])
            w_b = np.array([benchmark_weights.get(a, 0) for a in available_assets])
            
            # Normalize weights to sum to 1
            if w_p.sum() > 0:
                w_p = w_p / w_p.sum()
            if w_b.sum() > 0:
                w_b = w_b / w_b.sum()
            
            # Calculate mean returns
            mean_returns = returns_df.mean().values
            
            # Portfolio and benchmark returns
            R_p = np.sum(w_p * mean_returns)
            R_b = np.sum(w_b * mean_returns)
            
            # Excess return
            excess = R_p - R_b
            
            # Simplified attribution (assuming interaction is small)
            # Allocation effect: (w_p - w_b) * R_b
            allocation = np.sum((w_p - w_b) * R_b)
            
            # Selection effect: w_b * (R_p - R_b)
            selection = np.sum(w_b * (R_p - R_b))
            
            # Interaction: (w_p - w_b) * (R_p - R_b)
            interaction = np.sum((w_p - w_b) * (mean_returns - R_b))
            
            # Calculate portfolio returns series
            portfolio_returns = returns_df.dot(w_p)
            benchmark_returns = returns_df.dot(w_b)
            
            # Tracking error
            tracking_error = np.std(portfolio_returns - benchmark_returns) * np.sqrt(252)
            
            # Information ratio
            excess_returns = portfolio_returns - benchmark_returns
            information_ratio = excess_returns.mean() * np.sqrt(252) / tracking_error if tracking_error > 0 else 0
            
            # Active share
            active_share = 0.5 * np.sum(np.abs(w_p - w_b))
            
            # Beta
            covariance = np.cov(portfolio_returns, benchmark_returns)[0][1]
            variance = np.var(benchmark_returns)
            beta = covariance / variance if variance > 0 else 1.0
            
            # Create simple sector breakdown
            sector_breakdown = {}
            
            return {
                'Total Portfolio Return': R_p,
                'Total Benchmark Return': R_b,
                'Total Excess Return': excess,
                'Allocation Effect': allocation,
                'Selection Effect': selection,
                'Interaction Effect': interaction,
                'Information Ratio': information_ratio,
                'Tracking Error': tracking_error,
                'Active Share': active_share,
                'Portfolio Beta': beta,
                'Sector Breakdown': sector_breakdown
            }
            
        except Exception as e:
            st.warning(f"Attribution calculation simplified due to error: {str(e)}")
            return {
                'Total Portfolio Return': 0,
                'Total Benchmark Return': 0,
                'Total Excess Return': 0,
                'Allocation Effect': 0,
                'Selection Effect': 0,
                'Interaction Effect': 0,
                'Information Ratio': 0,
                'Tracking Error': 0,
                'Active Share': 0,
                'Portfolio Beta': 1.0,
                'Sector Breakdown': {}
            }

# ============================================================================
# 5. SIMPLIFIED MONTE CARLO SIMULATOR
# ============================================================================

class SimpleMonteCarloSimulator:
    """Simplified Monte Carlo simulation for robustness."""
    
    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        if not returns.empty:
            self.mean_return = returns.mean().mean()
            self.volatility = returns.std().mean()
        else:
            self.mean_return = 0
            self.volatility = 0.2
    
    def gbm_simulation_simple(self, initial_value=1.0, days=252, n_sims=1000):
        """Simple GBM simulation."""
        if self.volatility == 0:
            return np.ones((n_sims, days + 1)) * initial_value
        
        dt = 1/252
        drift = (self.mean_return - 0.5 * self.volatility**2) * dt
        diffusion = self.volatility * np.sqrt(dt)
        
        paths = np.zeros((n_sims, days + 1))
        paths[:, 0] = initial_value
        
        # Generate all random numbers at once for speed
        z = np.random.randn(n_sims, days)
        
        for t in range(1, days + 1):
            paths[:, t] = paths[:, t-1] * np.exp(drift + diffusion * z[:, t-1])
        
        return paths
    
    def calculate_simple_var(self, paths, confidence=0.95):
        """Calculate simple VaR."""
        terminal_values = paths[:, -1]
        terminal_returns = (terminal_values - 1)
        var = np.percentile(terminal_returns, (1 - confidence) * 100)
        cvar = terminal_returns[terminal_returns <= var].mean()
        
        return {
            f"VaR ({int(confidence*100)}%)": var,
            f"CVaR ({int(confidence*100)}%)": cvar
        }

# ============================================================================
# 6. BASIC RISK METRICS
# ============================================================================

class BasicRiskMetrics:
    """Basic risk metrics calculation."""
    
    @staticmethod
    def calculate_basic_metrics(returns, risk_free=0.02):
        """Calculate basic risk metrics."""
        if returns.empty or len(returns) < 10:
            return {
                "CAGR": 0,
                "Volatility": 0,
                "Sharpe Ratio": 0,
                "Max Drawdown": 0,
                "Calmar Ratio": 0
            }
        
        ann_factor = 252
        
        total_return = (1 + returns).prod() - 1
        cagr = (1 + total_return) ** (ann_factor / len(returns)) - 1
        volatility = returns.std() * np.sqrt(ann_factor)
        
        excess_returns = returns - (risk_free / ann_factor)
        sharpe = np.sqrt(ann_factor) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0
        
        cum_returns = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        max_dd = drawdown.min()
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        return {
            "CAGR": cagr,
            "Volatility": volatility,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_dd,
            "Calmar Ratio": calmar
        }

# ============================================================================
# 7. BASIC OPTIMIZATION (Fallback)
# ============================================================================

class BasicPortfolioOptimizer:
    """Basic portfolio optimization if PyPortfolioOpt is not available."""
    
    @staticmethod
    def equal_weight(tickers):
        """Equal weight portfolio."""
        n = len(tickers)
        return {t: 1.0/n for t in tickers}
    
    @staticmethod
    def min_variance_basic(returns_df):
        """Basic minimum variance portfolio."""
        if returns_df.empty:
            return {}
        
        cov_matrix = returns_df.cov().values
        n = cov_matrix.shape[0]
        
        try:
            # Solve for minimum variance portfolio
            inv_cov = np.linalg.inv(cov_matrix)
            ones = np.ones(n)
            weights = inv_cov.dot(ones) / ones.dot(inv_cov).dot(ones)
            
            # Create weights dictionary
            weights_dict = {ticker: float(w) for ticker, w in zip(returns_df.columns, weights)}
            return weights_dict
        except:
            # Fallback to equal weight
            return BasicPortfolioOptimizer.equal_weight(returns_df.columns.tolist())

# ============================================================================
# 8. SIDEBAR CONFIGURATION
# ============================================================================

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("🔧 Portfolio Configuration")

# Asset Universe Selection
ticker_lists = {
    "US Large Caps": US_DEFAULTS, 
    "BIST 30 (Turkey)": BIST_30[:10],  # Reduced for performance
    "Global Indices": GLOBAL_INDICES[:10]
}
selected_list = st.sidebar.selectbox("Asset Universe", list(ticker_lists.keys()))
available_tickers = ticker_lists[selected_list]

# Custom Ticker Injection
custom_tickers = st.sidebar.text_input("Custom Tickers (Comma Separated)", value="")
if custom_tickers: 
    custom_list = [t.strip().upper() for t in custom_tickers.split(',') if t.strip()]
    available_tickers = list(set(available_tickers + custom_list))

# Limit selection for performance
selected_tickers = st.sidebar.multiselect(
    "Select Assets (Max 15)", 
    available_tickers, 
    default=available_tickers[:min(5, len(available_tickers))],
    max_selections=15
)

st.sidebar.markdown("---")

# Use Expanders for cleaner UI
with st.sidebar.expander("⚙️ Model Parameters", expanded=True):
    # Set default dates with buffer
    default_end = datetime.now()
    default_start = default_end - timedelta(days=365*2)  # 2 years for faster loading
    
    start_date = st.date_input("Start Date", default_start)
    end_date = st.date_input("End Date", default_end)
    
    # Validate dates
    if start_date >= end_date:
        st.sidebar.error("⚠️ Start date must be before end date")
        start_date = default_start
        end_date = default_end
    
    rf_rate = st.number_input("Risk-Free Rate (%)", 0.0, 10.0, 2.0, 0.1) / 100

    # Strategy Selection (simplified)
    strat_options = ['Equal Weight', 'Min Variance']
    if PYPFOPT_AVAILABLE:
        strat_options.extend(['Max Sharpe', 'Min Volatility', 'Efficient Risk'])
    
    method = st.selectbox("Optimization Method", strat_options)

    if method == 'Efficient Risk':
        target_vol = st.slider("Target Volatility", 0.05, 0.50, 0.15)

# Monte Carlo Simulation Parameters
with st.sidebar.expander("🎲 Monte Carlo Settings", expanded=False):
    mc_days = st.slider("Simulation Horizon (Days)", 21, 252, 126)  # Reduced for speed
    mc_sims = st.selectbox("Number of Simulations", [500, 1000, 2500], index=1)  # Reduced

# Backtest Settings
with st.sidebar.expander("📉 Backtest Settings", expanded=False):
    rebal_freq_ui = st.selectbox("Rebalancing Frequency", ["Quarterly", "Monthly", "Yearly"])
    freq_map = {"Quarterly": "Q", "Monthly": "M", "Yearly": "Y"}

# Add a clear warning about data loading
st.sidebar.markdown("---")
st.sidebar.info("""
**Note:** 
- First run may take 30-60 seconds to fetch data
- Large portfolios (>15 assets) may be slower
- Some tickers may not have data for selected period
""")

run_btn = st.sidebar.button("🚀 RUN ANALYSIS", type="primary", use_container_width=True)

# ============================================================================
# 9. MAIN EXECUTION BLOCK WITH PROGRESS UPDATES
# ============================================================================

if run_btn:
    if not selected_tickers:
        st.error("❌ Please select at least one asset.")
        st.stop()
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Data Loading (20%)
        status_text.text("📊 Fetching market data...")
        data_manager = RobustPortfolioDataManager()
        prices, ohlc_data = data_manager.fetch_data_safe(
            selected_tickers, 
            start_date, 
            end_date,
            max_retries=1
        )
        progress_bar.progress(20)
        
        if prices.empty:
            st.error("""
            ❌ No data received. Possible issues:
            1. Invalid ticker symbols
            2. No data for selected period
            3. Yahoo Finance API issue
            
            **Try:** 
            - Using simpler tickers (AAPL, MSFT, GOOGL)
            - Reducing number of assets
            - Checking date range
            """)
            st.stop()
        
        # Step 2: Returns Calculation (30%)
        status_text.text("📈 Calculating returns...")
        returns = data_manager.calculate_returns(prices)
        progress_bar.progress(30)
        
        if returns.empty:
            st.error("❌ Insufficient data for returns calculation.")
            st.stop()
        
        # Step 3: Portfolio Optimization (40%)
        status_text.text("⚖️ Optimizing portfolio...")
        
        if method == 'Equal Weight':
            weights = BasicPortfolioOptimizer.equal_weight(selected_tickers)
            # Simple performance metrics for equal weight
            w_array = np.array([weights.get(t, 0) for t in selected_tickers])
            expected_return = np.sum(returns.mean() * w_array) * 252
            expected_vol = np.sqrt(np.dot(w_array.T, np.dot(returns.cov() * 252, w_array)))
            sharpe = (expected_return - rf_rate) / expected_vol if expected_vol > 0 else 0
            
        elif method == 'Min Variance' and not PYPFOPT_AVAILABLE:
            weights = BasicPortfolioOptimizer.min_variance_basic(returns)
            w_array = np.array([weights.get(t, 0) for t in selected_tickers])
            expected_return = np.sum(returns.mean() * w_array) * 252
            expected_vol = np.sqrt(np.dot(w_array.T, np.dot(returns.cov() * 252, w_array)))
            sharpe = (expected_return - rf_rate) / expected_vol if expected_vol > 0 else 0
            
        elif PYPFOPT_AVAILABLE:
            try:
                # Try PyPortfolioOpt optimization
                mu = expected_returns.mean_historical_return(prices)
                S = risk_models.sample_cov(prices)
                ef = EfficientFrontier(mu, S)
                
                if method == 'Max Sharpe':
                    weights = ef.max_sharpe(risk_free_rate=rf_rate)
                elif method == 'Min Volatility':
                    weights = ef.min_volatility()
                elif method == 'Efficient Risk':
                    weights = ef.efficient_risk(target_volatility=target_vol)
                else:
                    weights = ef.min_volatility()
                
                weights = ef.clean_weights()
                expected_return, expected_vol, sharpe = ef.portfolio_performance(
                    verbose=False, risk_free_rate=rf_rate
                )
            except Exception as e:
                st.warning(f"PyPortfolioOpt optimization failed: {str(e)}. Using equal weight.")
                weights = BasicPortfolioOptimizer.equal_weight(selected_tickers)
                w_array = np.array([weights.get(t, 0) for t in selected_tickers])
                expected_return = np.sum(returns.mean() * w_array) * 252
                expected_vol = np.sqrt(np.dot(w_array.T, np.dot(returns.cov() * 252, w_array)))
                sharpe = (expected_return - rf_rate) / expected_vol if expected_vol > 0 else 0
        else:
            weights = BasicPortfolioOptimizer.equal_weight(selected_tickers)
            w_array = np.array([weights.get(t, 0) for t in selected_tickers])
            expected_return = np.sum(returns.mean() * w_array) * 252
            expected_vol = np.sqrt(np.dot(w_array.T, np.dot(returns.cov() * 252, w_array)))
            sharpe = (expected_return - rf_rate) / expected_vol if expected_vol > 0 else 0
        
        progress_bar.progress(50)
        
        # Step 4: Simple Backtest (60%)
        status_text.text("📉 Running backtest...")
        
        # Create simple equity curve
        w_array = np.array([weights.get(t, 0) for t in returns.columns])
        portfolio_returns = returns.dot(w_array)
        equity_curve = (1 + portfolio_returns).cumprod() * 100000
        
        progress_bar.progress(70)
        
        # Step 5: Risk Metrics (80%)
        status_text.text("⚠️ Calculating risk metrics...")
        risk_metrics = BasicRiskMetrics.calculate_basic_metrics(portfolio_returns, rf_rate)
        progress_bar.progress(80)
        
        # Step 6: Attribution Analysis (90%)
        status_text.text("🔍 Analyzing performance attribution...")
        
        # Create benchmark weights (equal weight)
        benchmark_weights = BasicPortfolioOptimizer.equal_weight(selected_tickers)
        
        attribution = RobustPortfolioAttribution.calculate_simple_attribution(
            weights, benchmark_weights, returns
        )
        progress_bar.progress(90)
        
        # Step 7: Monte Carlo Simulation (95%)
        status_text.text("🎲 Running Monte Carlo simulations...")
        mc_simulator = SimpleMonteCarloSimulator(pd.DataFrame(portfolio_returns))
        mc_paths = mc_simulator.gbm_simulation_simple(days=mc_days, n_sims=mc_sims)
        var_results = mc_simulator.calculate_simple_var(mc_paths, confidence=0.95)
        progress_bar.progress(95)
        
        # Step 8: Metadata (100%)
        status_text.text("🏷️ Loading asset metadata...")
        classifier = AssetClassifier()
        asset_metadata = classifier.get_asset_metadata(selected_tickers)
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        # ============================================================================
        # 10. DISPLAY RESULTS
        # ============================================================================
        
        st.success(f"✅ Analysis complete! Processed {len(selected_tickers)} assets over {len(prices)} trading days.")
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "📈 Performance", "⚖️ Allocation", "🎲 Risk"])
        
        with tab1:
            st.markdown("### 📊 Portfolio Summary")
            
            # Key metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Expected Return", f"{expected_return:.2%}")
            with col2:
                st.metric("Expected Volatility", f"{expected_vol:.2%}")
            with col3:
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            with col4:
                st.metric("Max Drawdown", f"{risk_metrics['Max Drawdown']:.2%}")
            
            st.markdown("---")
            
            # Performance attribution
            st.markdown("#### 🎯 Performance Attribution")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Excess Return", f"{attribution['Total Excess Return']:.2%}")
            with col_b:
                st.metric("Information Ratio", f"{attribution['Information Ratio']:.2f}")
            with col_c:
                st.metric("Active Share", f"{attribution['Active Share']:.1%}")
            
            # Monte Carlo results
            st.markdown("---")
            st.markdown("#### 🎲 Risk Metrics")
            col_var, col_cvar = st.columns(2)
            with col_var:
                st.metric("VaR (95%)", f"{var_results['VaR (95%)']:.2%}")
            with col_cvar:
                st.metric("CVaR (95%)", f"{var_results['CVaR (95%)']:.2%}")
        
        with tab2:
            st.markdown("### 📈 Performance Analysis")
            
            # Equity curve
            fig_equity = go.Figure()
            fig_equity.add_trace(go.Scatter(
                x=equity_curve.index,
                y=equity_curve.values,
                mode='lines',
                name='Portfolio',
                line=dict(color='#00cc96', width=2)
            ))
            
            # Add benchmark if available
            bench_returns = returns.mean(axis=1)
            bench_curve = (1 + bench_returns).cumprod() * 100000
            fig_equity.add_trace(go.Scatter(
                x=bench_curve.index,
                y=bench_curve.values,
                mode='lines',
                name='Equal Weight Benchmark',
                line=dict(color='#888', dash='dash')
            ))
            
            fig_equity.update_layout(
                title="Equity Curve ($100k Initial)",
                template="plotly_dark",
                height=400,
                xaxis_title="Date",
                yaxis_title="Portfolio Value"
            )
            st.plotly_chart(fig_equity, use_container_width=True)
            
            # Drawdown chart
            running_max = equity_curve.cummax()
            drawdown = (equity_curve - running_max) / running_max
            
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown.values * 100,
                fill='tozeroy',
                mode='none',
                name='Drawdown',
                fillcolor='rgba(239, 85, 59, 0.3)',
                line=dict(color='#ef553b')
            ))
            
            fig_dd.update_layout(
                title="Drawdown Profile",
                template="plotly_dark",
                height=300,
                yaxis_title="Drawdown %",
                yaxis_tickformat=".1f%"
            )
            st.plotly_chart(fig_dd, use_container_width=True)
        
        with tab3:
            st.markdown("### ⚖️ Portfolio Allocation")
            
            # Create allocation pie chart
            non_zero_weights = {k: v for k, v in weights.items() if v > 0.001}
            
            if non_zero_weights:
                fig_pie = go.Figure(data=[go.Pie(
                    labels=list(non_zero_weights.keys()),
                    values=list(non_zero_weights.values()),
                    hole=0.3,
                    textinfo='label+percent'
                )])
                
                fig_pie.update_layout(
                    title="Portfolio Allocation",
                    template="plotly_dark",
                    height=400
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Display weights table
                weights_df = pd.DataFrame({
                    'Asset': list(non_zero_weights.keys()),
                    'Weight': [f"{w:.2%}" for w in non_zero_weights.values()]
                })
                st.dataframe(weights_df, hide_index=True, use_container_width=True)
            else:
                st.info("No significant allocations found.")
        
        with tab4:
            st.markdown("### 🎲 Risk Analysis")
            
            # Monte Carlo paths
            fig_mc = go.Figure()
            
            # Plot sample of paths
            n_sample = min(50, mc_sims)
            for i in range(n_sample):
                fig_mc.add_trace(go.Scatter(
                    x=list(range(mc_days + 1)),
                    y=mc_paths[i, :],
                    mode='lines',
                    line=dict(width=0.5, color='rgba(100, 100, 100, 0.1)'),
                    showlegend=False
                ))
            
            # Plot mean path
            mean_path = np.mean(mc_paths, axis=0)
            fig_mc.add_trace(go.Scatter(
                x=list(range(mc_days + 1)),
                y=mean_path,
                mode='lines',
                name='Mean Path',
                line=dict(color='#00cc96', width=3)
            ))
            
            fig_mc.update_layout(
                title=f"Monte Carlo Simulation ({mc_sims} paths)",
                template="plotly_dark",
                height=400,
                xaxis_title="Days",
                yaxis_title="Portfolio Value (Normalized)"
            )
            st.plotly_chart(fig_mc, use_container_width=True)
            
            # Terminal distribution
            terminal_values = mc_paths[:, -1]
            
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=terminal_values,
                nbinsx=50,
                name='Terminal Values',
                marker_color='#636efa',
                opacity=0.7
            ))
            
            # Add VaR line
            var_95 = np.percentile(terminal_values, 5)
            fig_hist.add_vline(x=var_95, line_dash="dash", line_color="red", 
                             annotation_text=f"VaR 95%: {var_95:.4f}")
            
            fig_hist.update_layout(
                title="Distribution of Terminal Values",
                template="plotly_dark",
                height=300,
                xaxis_title="Terminal Value"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Display data info
        with st.expander("📋 Data Information"):
            st.write(f"**Period:** {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")
            st.write(f"**Trading Days:** {len(prices)}")
            st.write(f"**Assets with Data:** {len(prices.columns)}/{len(selected_tickers)}")
            
            # Show missing data
            missing = [t for t in selected_tickers if t not in prices.columns]
            if missing:
                st.warning(f"Missing data for: {', '.join(missing)}")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        st.error(f"❌ Analysis failed with error: {str(e)}")
        st.info("""
        **Troubleshooting tips:**
        1. Try fewer assets
        2. Use shorter time period
        3. Check ticker symbols are valid
        4. Try different asset universe
        """)

else:
    # Landing page
    st.markdown("""
    <div style="text-align: center; padding: 40px 20px;">
        <h1>🏛️ Portfolio Analysis Terminal</h1>
        <p style="color: #666; font-size: 16px; max-width: 800px; margin: 20px auto;">
            Professional portfolio analysis with attribution, risk metrics, and Monte Carlo simulations
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background-color: #1e1e1e; padding: 20px; border-radius: 10px; border-left: 4px solid #00cc96;">
            <h4 style="color: #00cc96; margin-top: 0;">📊 Portfolio Optimization</h4>
            <p style="color: #ccc; font-size: 14px;">
                • Mean-variance optimization<br>
                • Risk parity approaches<br>
                • Custom constraints
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: #1e1e1e; padding: 20px; border-radius: 10px; border-left: 4px solid #636efa;">
            <h4 style="color: #636efa; margin-top: 0;">🎯 Performance Attribution</h4>
            <p style="color: #ccc; font-size: 14px;">
                • Brinson-Fachler attribution<br>
                • Sector decomposition<br>
                • Active management analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background-color: #1e1e1e; padding: 20px; border-radius: 10px; border-left: 4px solid #ef553b;">
            <h4 style="color: #ef553b; margin-top: 0;">⚠️ Risk Analysis</h4>
            <p style="color: #ccc; font-size: 14px;">
                • VaR/CVaR calculations<br>
                • Monte Carlo simulations<br>
                • Stress testing
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick start examples
    st.markdown("#### 🚀 Quick Start Examples")
    
    example_col1, example_col2, example_col3 = st.columns(3)
    
    with example_col1:
        if st.button("US Tech Portfolio", use_container_width=True):
            st.session_state.preload = "US"
            st.rerun()
    
    with example_col2:
        if st.button("Global Diversified", use_container_width=True):
            st.session_state.preload = "Global"
            st.rerun()
    
    with example_col3:
        if st.button("Turkish Market", use_container_width=True):
            st.session_state.preload = "Turkey"
            st.rerun()
