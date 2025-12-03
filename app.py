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
from typing import Dict, Tuple, List, Optional
import numpy.random as npr

# --- QUANTITATIVE LIBRARY IMPORTS ---
# PyPortfolioOpt: The core engine for Mean-Variance Optimization
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.cla import CLA  # Critical Line Algorithm
from pypfopt.hierarchical_portfolio import HRPOpt
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt import objective_functions

# Scikit-Learn: For Principal Component Analysis (PCA)
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf

# ARCH: For Econometric Volatility Forecasting (GARCH)
try:
    import arch
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

warnings.filterwarnings('ignore')

# ============================================================================
# 1. GLOBAL SYSTEM CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Enigma Institutional Terminal", 
    layout="wide", 
    page_icon="🏛️",
    initial_sidebar_state="expanded"
)

# Professional CSS: Dark Mode, Financial Terminal Aesthetics
st.markdown("""
<style>
    /* Main Layout Tweaks */
    .reportview-container {background: #0e1117;}
    div[class^="st-"] { font-family: 'Roboto', sans-serif; }
    
    /* Metric Cards - Hedge Fund Style */
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
    
    /* Custom Tab Navigation */
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
    
    /* Data Tables */
    div[data-testid="stTable"] { font-size: 13px; font-family: 'Roboto Mono', monospace; }
</style>
""", unsafe_allow_html=True)

# --- ASSET UNIVERSES DEFINITIONS ---
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
    '^BVSP', '^MXX', '^MERV', # Latin America
    '^TA125.TA', '^CASE30', '^JN0U.JO' # Middle East
]

US_DEFAULTS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V', 'WMT']

# ============================================================================
# 2. ADVANCED MONTE CARLO SIMULATION ENGINE
# ============================================================================

class AdvancedMonteCarloSimulator:
    """
    Advanced Monte Carlo simulation engine for VaR/CVaR with multiple methodologies:
    1. Geometric Brownian Motion (GBM) - Standard
    2. Student's t-distribution - Fat tails
    3. Jump Diffusion (Merton Model) - Extreme events
    4. GARCH-MC - Conditional volatility
    5. Filtered Historical Simulation (FHS) - Non-parametric
    6. Copula-based - Dependence structure
    """
    
    def __init__(self, returns: pd.DataFrame, prices: pd.DataFrame):
        self.returns = returns
        self.prices = prices
        self.n_assets = len(returns.columns)
        self.mean_returns = returns.mean().values
        self.cov_matrix = returns.cov().values
        self.cholesky = None
        
        # Calculate Cholesky decomposition for correlated simulations
        try:
            self.cholesky = np.linalg.cholesky(self.cov_matrix)
        except np.linalg.LinAlgError:
            # Regularize if not positive definite
            self.cov_matrix = self.cov_matrix + np.eye(self.n_assets) * 1e-6
            self.cholesky = np.linalg.cholesky(self.cov_matrix)
    
    def gbm_simulation(self, weights: np.ndarray, days: int = 252, n_sims: int = 10000, 
                       antithetic: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Geometric Brownian Motion with correlation structure.
        Includes antithetic variates for variance reduction.
        """
        dt = 1/252  # Daily frequency
        mu = np.dot(weights, self.mean_returns)
        sigma = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        # Generate standard normal random variables
        if antithetic:
            z = npr.randn(n_sims//2, days)
            z = np.vstack([z, -z])  # Antithetic variates
        else:
            z = npr.randn(n_sims, days)
        
        # GBM simulation
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        
        paths = np.zeros((n_sims, days + 1))
        paths[:, 0] = 1  # Initial value
        
        for t in range(1, days + 1):
            paths[:, t] = paths[:, t-1] * np.exp(drift + diffusion * z[:, t-1])
        
        terminal_values = paths[:, -1]
        
        return paths, {
            "method": "GBM",
            "mu": mu,
            "sigma": sigma,
            "terminal_mean": np.mean(terminal_values),
            "terminal_std": np.std(terminal_values)
        }
    
    def t_distribution_simulation(self, weights: np.ndarray, days: int = 252, 
                                  n_sims: int = 10000, df: float = 5.0) -> Tuple[np.ndarray, Dict]:
        """
        Student's t-distribution simulation for fat tails.
        """
        dt = 1/252
        mu = np.dot(weights, self.mean_returns)
        sigma = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        # Generate correlated t-distributed returns
        paths = np.zeros((n_sims, days + 1))
        paths[:, 0] = 1
        
        for i in range(n_sims):
            for t in range(1, days + 1):
                # t-distributed innovation
                z = npr.standard_t(df)
                ret = mu * dt + sigma * np.sqrt(dt) * z
                paths[i, t] = paths[i, t-1] * np.exp(ret)
        
        terminal_values = paths[:, -1]
        
        return paths, {
            "method": "Student's t",
            "df": df,
            "terminal_mean": np.mean(terminal_values),
            "terminal_std": np.std(terminal_values),
            "skewness": stats.skew(terminal_values),
            "kurtosis": stats.kurtosis(terminal_values)
        }
    
    def jump_diffusion_simulation(self, weights: np.ndarray, days: int = 252, 
                                  n_sims: int = 10000, jump_intensity: float = 0.05, 
                                  jump_mean: float = -0.1, jump_std: float = 0.15) -> Tuple[np.ndarray, Dict]:
        """
        Merton Jump Diffusion model for capturing extreme events.
        """
        dt = 1/252
        mu = np.dot(weights, self.mean_returns)
        sigma = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        paths = np.zeros((n_sims, days + 1))
        paths[:, 0] = 1
        
        for i in range(n_sims):
            for t in range(1, days + 1):
                # Poisson jumps
                n_jumps = npr.poisson(jump_intensity * dt)
                jump_sum = 0
                for _ in range(n_jumps):
                    jump_sum += npr.normal(jump_mean, jump_std)
                
                # Brownian motion component
                z = npr.randn()
                ret = mu * dt + sigma * np.sqrt(dt) * z + jump_sum
                paths[i, t] = paths[i, t-1] * np.exp(ret)
        
        terminal_values = paths[:, -1]
        
        return paths, {
            "method": "Jump Diffusion",
            "jump_intensity": jump_intensity,
            "jump_mean": jump_mean,
            "jump_std": jump_std,
            "terminal_mean": np.mean(terminal_values),
            "terminal_std": np.std(terminal_values),
            "max_jump_impact": np.max(np.abs(np.diff(np.log(paths[:, 1:]), axis=1)))
        }
    
    def garch_monte_carlo(self, weights: np.ndarray, days: int = 252, 
                          n_sims: int = 5000, garch_model=None) -> Tuple[np.ndarray, Dict]:
        """
        Monte Carlo simulation using GARCH forecasted volatility.
        """
        if not HAS_ARCH or garch_model is None:
            return self.gbm_simulation(weights, days, n_sims)
        
        # Get GARCH volatility forecasts
        forecast = garch_model.forecast(horizon=days)
        cond_vol = forecast.variance.values[-1, :] ** 0.5
        
        mu = np.dot(weights, self.mean_returns)
        
        paths = np.zeros((n_sims, days + 1))
        paths[:, 0] = 1
        
        for i in range(n_sims):
            for t in range(1, days + 1):
                vol = cond_vol[t-1] if t-1 < len(cond_vol) else cond_vol[-1]
                z = npr.randn()
                ret = mu * (1/252) + vol * z
                paths[i, t] = paths[i, t-1] * np.exp(ret)
        
        terminal_values = paths[:, -1]
        
        return paths, {
            "method": "GARCH-MC",
            "terminal_mean": np.mean(terminal_values),
            "terminal_std": np.std(terminal_values),
            "avg_volatility": np.mean(cond_vol)
        }
    
    def filtered_historical_simulation(self, weights: np.ndarray, days: int = 252, 
                                       n_sims: int = 10000, block_size: int = 5) -> Tuple[np.ndarray, Dict]:
        """
        Filtered Historical Simulation - non-parametric approach using historical residuals.
        """
        # Calculate portfolio returns
        port_returns = self.returns.dot(weights)
        
        # Standardize returns
        standardized = (port_returns - port_returns.mean()) / port_returns.std()
        
        paths = np.zeros((n_sims, days + 1))
        paths[:, 0] = 1
        
        # Block bootstrap
        n_blocks = days // block_size + 1
        
        for i in range(n_sims):
            position = 0
            while position < days:
                # Randomly select a block
                start_idx = npr.randint(0, len(standardized) - block_size)
                block = standardized.iloc[start_idx:start_idx + block_size].values
                
                # Add block to path
                block_len = min(block_size, days - position)
                for j in range(block_len):
                    ret = port_returns.mean() + port_returns.std() * block[j]
                    paths[i, position + 1] = paths[i, position] * np.exp(ret)
                    position += 1
        
        terminal_values = paths[:, -1]
        
        return paths, {
            "method": "Filtered Historical Simulation",
            "block_size": block_size,
            "terminal_mean": np.mean(terminal_values),
            "terminal_std": np.std(terminal_values)
        }
    
    def calculate_var_cvar(self, paths: np.ndarray, confidence_levels: List[float] = [0.95, 0.99]) -> Dict:
        """
        Calculate VaR and CVaR from simulated paths.
        """
        terminal_returns = (paths[:, -1] - 1)  # Simple returns
        
        results = {}
        for conf in confidence_levels:
            alpha = 1 - conf
            
            # Historical VaR/CVaR from simulations
            var_hist = np.percentile(terminal_returns, alpha * 100)
            cvar_hist = terminal_returns[terminal_returns <= var_hist].mean()
            
            # Parametric assuming normality
            mu_sim = terminal_returns.mean()
            sigma_sim = terminal_returns.std()
            z_score = stats.norm.ppf(alpha)
            var_param = mu_sim + z_score * sigma_sim
            
            # Modified VaR (Cornish-Fisher)
            skew = stats.skew(terminal_returns)
            kurt = stats.kurtosis(terminal_returns)
            z_cf = z_score + (z_score**2 - 1) * skew / 6 + (z_score**3 - 3 * z_score) * kurt / 24
            var_mod = mu_sim + z_cf * sigma_sim
            
            results[f"MC VaR ({int(conf*100)}%)"] = var_hist
            results[f"MC CVaR ({int(conf*100)}%)"] = cvar_hist
            results[f"MC Parametric VaR ({int(conf*100)}%)"] = var_param
            results[f"MC Modified VaR ({int(conf*100)}%)"] = var_mod
        
        return results
    
    def stress_test_scenarios(self, weights: np.ndarray, scenarios: Dict[str, float], 
                              days: int = 21) -> Dict[str, float]:
        """
        Apply stress scenarios to portfolio.
        """
        initial_value = 1.0
        mu = np.dot(weights, self.mean_returns)
        sigma = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        results = {}
        for scenario_name, shock in scenarios.items():
            # Simple shock application (can be enhanced with copulas)
            stressed_return = mu * days/252 + shock
            final_value = initial_value * np.exp(stressed_return)
            results[scenario_name] = final_value - initial_value
        
        return results

# ============================================================================
# 3. ROBUST DATA PIPELINE & CACHING
# ============================================================================

class PortfolioDataManager:
    """
    Handles secure data fetching from Yahoo Finance with robust error handling,
    MultiIndex parsing, and local caching to prevent API rate limits.
    """
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_data(tickers, start_date, end_date):
        """
        Fetches OHLCV data for a list of tickers.
        Handles the complexity of yfinance returning different structures
        based on the number of tickers requested.
        """
        if not tickers:
            return pd.DataFrame(), {}
            
        try:
            # Normalize input
            if isinstance(tickers, str): tickers = [tickers]
            
            # Request Data
            data = yf.download(
                tickers, 
                start=start_date, 
                end=end_date, 
                progress=False, 
                group_by='ticker', 
                threads=False # Disable threads for cloud stability
            )
            
            prices = pd.DataFrame()
            ohlc_dict = {}

            # Case A: Single Ticker Requested
            if len(tickers) == 1:
                ticker = tickers[0]
                df = data
                # Unwrap MultiIndex if present
                if isinstance(data.columns, pd.MultiIndex):
                    try: 
                        df = data.xs(ticker, axis=1, level=0, drop_level=True)
                    except: 
                        pass # Keep original if extraction fails
                
                # Identify Price Column
                price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
                if price_col in df.columns:
                    prices[ticker] = df[price_col]
                    ohlc_dict[ticker] = df
            
            # Case B: Multiple Tickers Requested
            else:
                # Validation: Ensure we have a MultiIndex
                if not isinstance(data.columns, pd.MultiIndex):
                    return pd.DataFrame(), {}
                
                for ticker in tickers:
                    try:
                        df = data.xs(ticker, axis=1, level=0, drop_level=True)
                        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
                        if price_col in df.columns:
                            prices[ticker] = df[price_col]
                            ohlc_dict[ticker] = df
                    except KeyError:
                        # Skip tickers that failed to download
                        continue
            
            # Final Cleaning
            prices = prices.ffill().bfill()
            prices = prices.dropna(axis=1, how='all') # Drop empty columns
            
            return prices, ohlc_dict
            
        except Exception as e:
            st.error(f"Critical Data Pipeline Error: {str(e)}")
            return pd.DataFrame(), {}

    @staticmethod
    def calculate_returns(prices, method='log'):
        """Calculates Logarithmic or Simple returns"""
        if method == 'log':
            return np.log(prices / prices.shift(1)).dropna()
        else:
            return prices.pct_change().dropna()

    @staticmethod
    @st.cache_data(ttl=3600*24)
    def get_market_caps(tickers):
        """
        Fetches Market Capitalization data. 
        Crucial for Black-Litterman 'Market Implied' prior calculations.
        """
        mcaps = {}
        for t in tickers:
            try:
                # Default to 10B if API fails to prevent division by zero
                mcaps[t] = yf.Ticker(t).info.get('marketCap', 10e9)
            except:
                mcaps[t] = 10e9
        return pd.Series(mcaps)

# ============================================================================
# 4. ADVANCED RISK ENGINE (VaR, CVaR, GARCH, PCA)
# ============================================================================

class AdvancedRiskMetrics:
    """
    Institutional Risk Engine.
    Includes:
    1. Standard Metrics (Sharpe, Sortino)
    2. Tail Risk (VaR/CVaR)
    3. GARCH(1,1) Econometrics
    4. Component VaR (Risk Decomposition)
    5. PCA Analysis
    """

    @staticmethod
    def calculate_metrics(returns, risk_free=0.02):
        """Generates the standard KPI matrix for the Tearsheet."""
        ann_factor = 252
        
        # 1. Absolute Return Metrics
        total_return = (1 + returns).prod() - 1
        cagr = (1 + total_return) ** (ann_factor / len(returns)) - 1
        volatility = returns.std() * np.sqrt(ann_factor)
        
        # 2. Risk-Adjusted Metrics
        excess_returns = returns - (risk_free / ann_factor)
        sharpe = np.sqrt(ann_factor) * excess_returns.mean() / returns.std()
        
        # 3. Downside Risk Metrics
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(ann_factor)
        sortino = np.sqrt(ann_factor) * excess_returns.mean() / downside_std if downside_std != 0 else 0
        
        # 4. Drawdown Analytics
        cum_returns = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        max_dd = drawdown.min()
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        # 5. Tail Risk (Historical)
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
        The Core Engine for the 'Visual VaR' Tab.
        Computes VaR using 3 distinct methodologies to highlight Model Risk.
        """
        results = {}
        
        # Distribution Moments
        mu = returns.mean()
        sigma = returns.std()
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns) # Excess Kurtosis
        
        for conf in confidence_levels:
            alpha = 1 - conf
            
            # A. Parametric VaR (Normal Distribution Assumption)
            z_score = stats.norm.ppf(alpha)
            var_param = mu + z_score * sigma
            
            # B. Historical VaR (Empirical)
            var_hist = np.percentile(returns, alpha * 100)
            
            # C. Modified VaR (Cornish-Fisher Expansion)
            # Adjusts Z-score for Skewness and Kurtosis (Fat Tails)
            z_cf = z_score + (z_score**2 - 1)*skew/6 + (z_score**3 - 3*z_score)*kurt/24 - (2*z_score**3 - 5*z_score)*(skew**2)/36
            var_mod = mu + z_cf * sigma
            
            # D. Expected Shortfall (CVaR)
            tail_losses = returns[returns <= var_hist]
            cvar_hist = tail_losses.mean() if len(tail_losses) > 0 else var_hist
            
            # Store Results
            tag = f"{int(conf*100)}%"
            results[f"Parametric VaR ({tag})"] = var_param
            results[f"Historical VaR ({tag})"] = var_hist
            results[f"Modified VaR ({tag})"] = var_mod
            results[f"CVaR ({tag})"] = cvar_hist
            
        return results, skew, kurt

    @staticmethod
    def fit_garch_model(returns):
        """
        Fits a GARCH(1,1) model to the portfolio returns.
        Returns the model summary and the conditional volatility series.
        """
        if not HAS_ARCH:
            return None, None
        
        try:
            # Scale returns by 100 for better convergence
            scaled_returns = returns * 100
            am = arch.arch_model(scaled_returns, vol='Garch', p=1, q=1, dist='Normal')
            res = am.fit(disp='off')
            
            # Get Conditional Volatility (rescale back)
            cond_vol = res.conditional_volatility / 100
            return res, cond_vol
        except Exception as e:
            st.warning(f"GARCH Fit Failed: {e}")
            return None, None

    @staticmethod
    def calculate_component_var(returns_df, weights_dict, confidence=0.95):
        """
        Decomposes VaR into asset-level contributions (Component VaR)
        and performs Principal Component Analysis (PCA).
        """
        alpha = 1 - confidence
        
        # Align weights with dataframe columns
        assets = returns_df.columns
        w = np.array([weights_dict.get(a, 0) for a in assets])
        
        # Covariance Matrix
        cov_matrix = returns_df.cov()
        
        # Portfolio Std Dev
        port_var = np.dot(w.T, np.dot(cov_matrix, w))
        port_std = np.sqrt(port_var)
        
        # Marginal Contribution to Risk (MCR)
        # MCR = (Cov * w) / port_std
        mcr = np.dot(cov_matrix, w) / port_std
        
        # Component VaR = weight * MCR * Z_score (approximation)
        z_score = abs(stats.norm.ppf(alpha))
        component_var = w * mcr * z_score
        
        # PCA Analysis for Factor Strength
        pca = PCA()
        pca.fit(returns_df)
        explained_variance = pca.explained_variance_ratio_
        
        return pd.Series(component_var, index=assets), explained_variance, pca

# ============================================================================
# 5. PYPORTFOLIOOPT STRATEGY FACTORY
# ============================================================================

class AdvancedPortfolioOptimizer:
    """
    Wrapper for PyPortfolioOpt.
    Provides a unified interface for 8 different optimization strategies.
    """
    
    def __init__(self, returns, prices):
        self.returns = returns
        self.prices = prices
        # Calculate Expected Returns (Mean Historical)
        self.mu = expected_returns.mean_historical_return(prices)
        # Calculate Covariance Matrix (Sample Covariance)
        self.S = risk_models.sample_cov(prices)
    
    def optimize(self, method, risk_free_rate=0.02, target_vol=0.10, target_ret=0.20, risk_aversion=1.0):
        """
        Main routing function for Mean-Variance based strategies.
        """
        ef = EfficientFrontier(self.mu, self.S)
        
        # Strategy Selector
        if method == 'Max Sharpe':
            weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
            
        elif method == 'Min Volatility':
            weights = ef.min_volatility()
            
        elif method == 'Efficient Risk':
            # Maximize return for a given target volatility
            weights = ef.efficient_risk(target_volatility=target_vol)
            
        elif method == 'Efficient Return':
            # Minimize risk for a given target return
            weights = ef.efficient_return(target_return=target_ret)
            
        elif method == 'Max Quadratic Utility':
            # Maximize: ret - 0.5 * gamma * vol^2
            weights = ef.max_quadratic_utility(risk_aversion=risk_aversion)
            
        return ef.clean_weights(), ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)

    def optimize_cla(self):
        """Critical Line Algorithm (CLA) - The analytical solution to MVO"""
        cla = CLA(self.mu, self.S)
        # CLA computes the entire frontier, we pick Max Sharpe from it
        weights = cla.max_sharpe()
        return cla.clean_weights(), cla.portfolio_performance(verbose=False)

    def optimize_hrp(self):
        """
        Hierarchical Risk Parity (HRP).
        Uses clustering (graph theory) instead of matrix inversion. 
        Robust to shocks.
        """
        hrp = HRPOpt(self.returns)
        weights = hrp.optimize()
        return hrp.clean_weights(), hrp.portfolio_performance(verbose=False)
        
    def optimize_black_litterman(self, market_caps):
        """
        Black-Litterman Model.
        Combines Market Equilibrium (CAPM) with Investor Views.
        """
        # 1. Calculate Risk Aversion (Delta)
        delta = risk_models.black_litterman.market_implied_risk_aversion(self.prices)
        
        # 2. Calculate Market Prior
        prior = risk_models.black_litterman.market_implied_prior_returns(market_caps, delta, self.S)
        
        # 3. Posterior Estimate (No explicit views = Equilibrium Portfolio)
        bl = BlackLittermanModel(self.S, pi=prior, absolute_views=None)
        
        # 4. Optimization on Posterior
        ef = EfficientFrontier(bl.bl_returns(), bl.bl_cov())
        weights = ef.max_sharpe()
        
        return ef.clean_weights(), ef.portfolio_performance(verbose=False)

# ============================================================================
# 6. INSTITUTIONAL BACKTESTING ENGINE
# ============================================================================

class PortfolioBacktester:
    """
    Performs a realistic path-dependent simulation.
    Features:
    - Periodic Rebalancing (Monthly/Quarterly/Yearly)
    - Transaction Cost modeling
    - Drift calculation
    """
    
    def __init__(self, prices, returns):
        self.prices = prices
        self.returns = returns
        
    def run_rebalancing_backtest(self, weights, initial_capital=100000, rebalance_freq='Q', cost_bps=10):
        """
        Simulates the portfolio equity curve with transaction costs.
        """
        # 1. Setup Data Structures
        assets = self.returns.columns
        # Convert dictionary weights to aligned numpy array
        w_target = np.array([weights.get(a, 0) for a in assets])
        
        # 2. Identify Rebalance Dates
        if rebalance_freq == 'M': dates = self.returns.resample('M').last().index
        elif rebalance_freq == 'Q': dates = self.returns.resample('Q').last().index
        else: dates = self.returns.resample('Y').last().index
        
        # Create a boolean mask for fast lookup inside loop
        rebalance_mask = pd.Series(False, index=self.prices.index)
        valid_dates = [d for d in dates if d in self.prices.index]
        rebalance_mask.loc[valid_dates] = True
        
        # 3. Initialization
        cash = initial_capital
        current_prices = self.prices.iloc[0].values
        
        # Initial Allocation
        current_shares = (cash * w_target) / current_prices
        
        # Initial Transaction Cost
        initial_volume = cash
        cash -= initial_volume * (cost_bps / 10000)
        
        portfolio_history = []
        date_history = []
        
        # 4. Simulation Loop (Path Dependent)
        # We iterate day-by-day to capture drift and exact rebalancing timing
        for date, price_series in self.prices.iterrows():
            market_prices = price_series.values
            
            # A. Mark-to-Market
            portfolio_value = np.sum(current_shares * market_prices)
            
            # B. Rebalancing Logic
            if rebalance_mask.loc[date]:
                # 1. Calculate Target Holdings
                target_value = portfolio_value 
                target_exposure = target_value * w_target
                target_shares = target_exposure / market_prices
                
                # 2. Calculate Turnover
                diff_shares = target_shares - current_shares
                turnover_value = np.sum(np.abs(diff_shares) * market_prices)
                
                # 3. Deduct Costs
                cost = turnover_value * (cost_bps / 10000)
                portfolio_value -= cost
                
                # 4. Update Holdings
                current_shares = (portfolio_value * w_target) / market_prices
                
            # Append History
            portfolio_history.append(portfolio_value)
            date_history.append(date)
            
        return pd.Series(portfolio_history, index=date_history)

# ============================================================================
# 7. UI & APPLICATION LOGIC
# ============================================================================

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("🔧 Institutional Config")

# Asset Universe Selection
ticker_lists = {
    "US Defaults": US_DEFAULTS, 
    "BIST 30 (Turkey)": BIST_30, 
    "Global Indices": GLOBAL_INDICES
}
selected_list = st.sidebar.selectbox("Asset Universe", list(ticker_lists.keys()))
available_tickers = ticker_lists[selected_list]

# Custom Ticker Injection
custom_tickers = st.sidebar.text_input("Custom Tickers (Comma Separated)", value="")
if custom_tickers: 
    available_tickers = list(set(available_tickers + [t.strip().upper() for t in custom_tickers.split(',')]))

# Selection Widget
selected_tickers = st.sidebar.multiselect("Portfolio Assets", available_tickers, default=available_tickers[:5])

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Model Parameters")
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365*3))
end_date = st.sidebar.date_input("End Date", datetime.now())
rf_rate = st.sidebar.number_input("Risk-Free Rate (%)", 0.0, 10.0, 2.0, 0.1) / 100

# Strategy Selection
strat_options = [
    'Max Sharpe', 
    'Min Volatility', 
    'Efficient Risk', 
    'Efficient Return', 
    'Max Quadratic Utility', 
    'Critical Line Algorithm (CLA)', 
    'Hierarchical Risk Parity (HRP)', 
    'Black-Litterman', 
    'Equal Weight'
]
method = st.sidebar.selectbox("Optimization Objective", strat_options)

# Monte Carlo Simulation Parameters
st.sidebar.markdown("---")
st.sidebar.subheader("🎲 Monte Carlo Settings")
mc_days = st.sidebar.slider("Simulation Horizon (Days)", 21, 504, 252, 21)
mc_sims = st.sidebar.selectbox("Number of Simulations", [1000, 5000, 10000, 25000], index=2)
mc_method = st.sidebar.selectbox("Simulation Method", 
                                 ["GBM", "Student's t", "Jump Diffusion", "Filtered Historical"])

# Jump Diffusion Parameters (conditional)
if mc_method == "Jump Diffusion":
    st.sidebar.markdown("**Jump Diffusion Parameters**")
    jump_intensity = st.sidebar.slider("Jump Intensity (λ)", 0.01, 0.20, 0.05, 0.01)
    jump_mean = st.sidebar.slider("Jump Mean", -0.20, 0.00, -0.10, 0.01)
    jump_std = st.sidebar.slider("Jump Std Dev", 0.05, 0.30, 0.15, 0.01)

# Student's t Parameters
if mc_method == "Student's t":
    df_t = st.sidebar.slider("Degrees of Freedom", 3.0, 15.0, 5.0, 0.5)

# Backtest Settings
st.sidebar.markdown("---")
st.sidebar.subheader("📉 Backtest Settings")
rebal_freq_ui = st.sidebar.selectbox("Rebalancing Frequency", ["Quarterly", "Monthly", "Yearly"])
freq_map = {"Quarterly": "Q", "Monthly": "M", "Yearly": "Y"}

# Conditional Strategy Inputs
target_vol, target_ret, risk_aversion = 0.1, 0.2, 1.0
if method == 'Efficient Risk':
    target_vol = st.sidebar.slider("Target Volatility", 0.05, 0.50, 0.15)
elif method == 'Efficient Return':
    target_ret = st.sidebar.slider("Target Return", 0.05, 0.50, 0.20)
elif method == 'Max Quadratic Utility':
    risk_aversion = st.sidebar.slider("Risk Aversion (Delta)", 0.1, 10.0, 1.0)

run_btn = st.sidebar.button("🚀 EXECUTE STRATEGY", type="primary")

# --- MAIN EXECUTION BLOCK ---
if run_btn:
    with st.spinner('Initializing Quantitative Engine...'):
        # 1. Data Ingestion
        data_manager = PortfolioDataManager()
        prices, ohlc_data = data_manager.fetch_data(selected_tickers, start_date, end_date)
        
        if prices.empty:
            st.error("❌ Data Fetch Failed. Please check ticker validity.")
        else:
            # 2. Return Calculation
            returns = data_manager.calculate_returns(prices)
            optimizer = AdvancedPortfolioOptimizer(returns, prices)
            
            # 3. Optimization Routing
            try:
                if method == 'Critical Line Algorithm (CLA)':
                    weights, perf = optimizer.optimize_cla()
                elif method == 'Hierarchical Risk Parity (HRP)':
                    weights, perf = optimizer.optimize_hrp()
                elif method == 'Black-Litterman': 
                    mcaps = data_manager.get_market_caps(selected_tickers)
                    weights, perf = optimizer.optimize_black_litterman(mcaps)
                elif method == 'Equal Weight':
                    weights = {t: 1.0/len(selected_tickers) for t in selected_tickers}
                    # Manually calc performance for Eq Weight
                    w_vec = np.array(list(weights.values()))
                    r = np.sum(returns.mean()*w_vec)*252
                    v = np.sqrt(np.dot(w_vec.T, np.dot(returns.cov()*252, w_vec)))
                    perf = (r, v, (r-rf_rate)/v)
                else:
                    weights, perf = optimizer.optimize(method, rf_rate, target_vol, target_ret, risk_aversion)
            except Exception as e:
                st.error(f"Optimization Failed: {str(e)}")
                st.stop()

            # 4. Advanced Monte Carlo Simulations
            st.info(f"🌀 Running {mc_sims:,} Monte Carlo simulations using {mc_method} method...")
            mc_simulator = AdvancedMonteCarloSimulator(returns, prices)
            w_array = np.array([weights.get(t, 0) for t in selected_tickers])
            
            # Run selected simulation method
            if mc_method == "GBM":
                mc_paths, mc_stats = mc_simulator.gbm_simulation(w_array, days=mc_days, n_sims=mc_sims)
            elif mc_method == "Student's t":
                mc_paths, mc_stats = mc_simulator.t_distribution_simulation(w_array, days=mc_days, n_sims=mc_sims, df=df_t)
            elif mc_method == "Jump Diffusion":
                mc_paths, mc_stats = mc_simulator.jump_diffusion_simulation(
                    w_array, days=mc_days, n_sims=mc_sims, 
                    jump_intensity=jump_intensity, jump_mean=jump_mean, jump_std=jump_std
                )
            elif mc_method == "Filtered Historical":
                mc_paths, mc_stats = mc_simulator.filtered_historical_simulation(w_array, days=mc_days, n_sims=mc_sims)
            
            # Calculate VaR/CVaR from simulations
            mc_var_results = mc_simulator.calculate_var_cvar(mc_paths)
            
            # 5. Dynamic Backtesting
            backtester = PortfolioBacktester(prices, returns)
            equity_curve = backtester.run_rebalancing_backtest(
                weights, 
                rebalance_freq=freq_map[rebal_freq_ui]
            )
            port_ret_series = equity_curve.pct_change().dropna()
            
            # 6. Risk Profiling
            risk_metrics = AdvancedRiskMetrics.calculate_metrics(port_ret_series, rf_rate)
            var_profile, skew, kurt = AdvancedRiskMetrics.calculate_comprehensive_risk_profile(port_ret_series)
            
            # Combine historical and MC VaR results
            all_var_results = {**var_profile, **mc_var_results}
            
            # 7. GARCH & PCA CALCULATION
            garch_model, garch_vol = AdvancedRiskMetrics.fit_garch_model(port_ret_series)
            comp_var, pca_expl_var, pca_obj = AdvancedRiskMetrics.calculate_component_var(returns, weights)
            
            # 8. Visualization Layout
            t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs([
                "🏛️ Executive Tearsheet", 
                "📈 Efficient Frontier", 
                "📉 Dynamic Backtest", 
                "🕯️ OHLC Analysis", 
                "🌪️ Stress Test", 
                "⚠️ Comparative VaR",
                "🎲 Advanced MC Simulations",
                "🔬 Quant Lab (GARCH/PCA)"
            ])
            
            # --- TAB 1: EXECUTIVE TEARSHEET ---
            with t1:
                st.markdown("### 🏛️ Strategy Performance Attribution")
                st.markdown("---")
                
                # KPI Grid (Custom HTML/CSS)
                k1, k2, k3, k4, k5 = st.columns(5)
                def kpi(col, label, val, color="white"):
                    col.markdown(f"""
                    <div class="pro-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value" style="color:{color}">{val}</div>
                    </div>""", unsafe_allow_html=True)
                
                kpi(k1, "CAGR", f"{risk_metrics['CAGR']:.2%}", "#00cc96")
                kpi(k2, "Volatility", f"{risk_metrics['Volatility']:.2%}")
                kpi(k3, "Sharpe", f"{risk_metrics['Sharpe Ratio']:.2f}", "#00cc96" if risk_metrics['Sharpe Ratio']>1 else "white")
                kpi(k4, "Max Drawdown", f"{risk_metrics['Max Drawdown']:.2%}", "#ef553b")
                kpi(k5, "Sortino", f"{risk_metrics['Sortino Ratio']:.2f}")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Allocation & Rolling Metrics
                col_chart, col_wheel = st.columns([2, 1])
                
                with col_chart:
                    # Rolling Sharpe
                    roll_sharpe = port_ret_series.rolling(126).apply(lambda x: (x.mean()*252 - rf_rate)/(x.std()*np.sqrt(252)))
                    fig_roll = go.Figure()
                    fig_roll.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe, fill='tozeroy', line=dict(color='#00cc96', width=1.5), name='Rolling Sharpe (6M)'))
                    fig_roll.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig_roll.update_layout(title="Rolling Risk-Adjusted Return (6M)", height=380, template="plotly_dark", margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig_roll, use_container_width=True)
                
                with col_wheel:
                    # Optimized Wheel Chart
                    w_series = pd.Series(weights).sort_values(ascending=False)
                    w_series = w_series[w_series > 0.001] # Filter noise
                    
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=w_series.index, 
                        values=w_series.values, 
                        hole=0.6, 
                        textinfo='label+percent', 
                        textposition='outside',
                        marker=dict(colors=px.colors.qualitative.Bold)
                    )])
                    fig_pie.update_layout(title="Target Allocation", showlegend=False, height=380, template="plotly_dark", margin=dict(l=40, r=40, t=50, b=40))
                    st.plotly_chart(fig_pie, use_container_width=True)

            # --- TAB 2: EFFICIENT FRONTIER ---
            with t2:
                st.subheader("Efficient Frontier Simulation (25,000 Runs)")
                
                n_sims = 25000
                w_rand = np.random.dirichlet(np.ones(len(selected_tickers)), n_sims)
                
                # 1. FIX: Explicit Numpy Conversion to prevent Pandas Index mismatch error
                mu_np = optimizer.mu.to_numpy()
                S_np = optimizer.S.to_numpy()
                
                # 2. Vectorized Return Calculation: (25000, N) @ (N,) -> (25000,)
                r_arr = w_rand @ mu_np
                
                # 3. Vectorized Volatility Calculation: Variance = sum( (w @ S) * w, axis=1 )
                v_arr = np.sqrt(np.sum((w_rand @ S_np) * w_rand, axis=1))
                
                s_arr = (r_arr - rf_rate) / v_arr
                
                fig_ef = go.Figure()
                fig_ef.add_trace(go.Scatter(
                    x=v_arr, y=r_arr, mode='markers', 
                    marker=dict(color=s_arr, colorscale='Viridis', showscale=True, colorbar=dict(title="Sharpe")), 
                    name='Simulations'
                ))
                fig_ef.add_trace(go.Scatter(
                    x=[perf[1]], y=[perf[0]], mode='markers', 
                    marker=dict(color='red', size=25, symbol='star'), 
                    name='Optimal Portfolio'
                ))
                fig_ef.update_layout(xaxis_title="Expected Volatility", yaxis_title="Expected Return", height=600, template="plotly_dark")
                st.plotly_chart(fig_ef, use_container_width=True)

            # --- TAB 3: DYNAMIC BACKTEST ---
            with t3:
                st.subheader(f"Dynamic Backtest Analysis ({rebal_freq_ui} Rebalancing)")
                
                # Calculate Static Benchmark (Equal Weight)
                eq_weights = np.array([1/len(selected_tickers)] * len(selected_tickers))
                bench_ret = returns.dot(eq_weights)
                bench_curve = (1 + bench_ret).cumprod() * 100000
                
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, name="Active Strategy", line=dict(color='#00cc96', width=2)))
                fig_bt.add_trace(go.Scatter(x=bench_curve.index, y=bench_curve, name="Equal Weight Index", line=dict(color='#888', dash='dash')))
                fig_bt.update_layout(title="Equity Curve ($100k Initial)", template="plotly_dark", height=500, xaxis_title="Date", yaxis_title="Portfolio Value ($)")
                st.plotly_chart(fig_bt, use_container_width=True)
                
                # Underwater Plot
                running_max = equity_curve.cummax()
                drawdown = (equity_curve - running_max) / running_max
                
                fig_dd = px.area(drawdown, title="Drawdown Profile", color_discrete_sequence=['#ef553b'])
                fig_dd.update_layout(template="plotly_dark", height=300, yaxis_title="Drawdown %")
                st.plotly_chart(fig_dd, use_container_width=True)

            # --- TAB 4: OHLC ---
            with t4:
                tk_sel = st.selectbox("Inspect Asset", selected_tickers)
                if tk_sel in ohlc_data:
                    df_ohlc = ohlc_data[tk_sel]
                    fig_c = go.Figure(data=[go.Candlestick(
                        x=df_ohlc.index, 
                        open=df_ohlc['Open'], high=df_ohlc['High'], 
                        low=df_ohlc['Low'], close=df_ohlc['Close']
                    )])
                    fig_c.update_layout(height=600, template="plotly_dark", title=f"{tk_sel} Price Action")
                    st.plotly_chart(fig_c, use_container_width=True)
                else:
                    st.warning("Detailed OHLC data unavailable for this ticker.")

            # --- TAB 5: STRESS TEST ---
            with t5:
                st.subheader("Macro Scenario Analysis")
                
                # Calculate Beta
                bench_series = returns.mean(axis=1)
                covariance = np.cov(port_ret_series, bench_series)[0][1]
                variance = np.var(bench_series)
                beta = covariance / variance
                
                scenarios = {
                    '2008 Financial Crisis (-40%)': -0.40, 
                    'Covid-19 Crash (-30%)': -0.30, 
                    'Aggressive Rate Hike (-10%)': -0.10, 
                    'Tech Bubble Burst (-20%)': -0.20,
                    'Post-Recession Melt Up (+20%)': 0.20
                }
                
                res_stress = []
                for name, shock in scenarios.items():
                    imp = shock * beta
                    pnl = 100000 * imp
                    res_stress.append({
                        "Scenario": name, 
                        "Market Shock": f"{shock:.0%}", 
                        "Est. Portfolio Impact": f"{imp:.2%}", 
                        "Est. PnL ($100k)": f"${pnl:,.0f}"
                    })
                
                st.table(pd.DataFrame(res_stress))

            # --- TAB 6: COMPARATIVE VAR ---
            with t6:
                st.subheader("⚠️ Comparative VaR & CVaR Engine")
                
                # 1. Comparison Chart
                var_plot_data = []
                for k, v in all_var_results.items():
                    if "VaR" in k and "CVaR" not in k:
                        method_name = k.split("(")[0].strip()
                        conf_level = k.split("(")[1].strip(")")
                        var_plot_data.append({"Method": method_name, "Confidence": conf_level, "VaR": v})
                        
                df_var_plot = pd.DataFrame(var_plot_data)
                
                fig_bar = px.bar(
                    df_var_plot, x="Confidence", y="VaR", color="Method", barmode="group",
                    title="VaR Estimates by Methodology (Parametric vs Historical vs Modified vs MC)",
                    color_discrete_sequence=['#636EFA', '#EF553B', '#00CC96', '#FFA15A'],
                    text_auto='.2%'
                )
                fig_bar.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # 2. Distribution Visualizer
                c_dist, c_stat = st.columns([2, 1])
                
                with c_dist:
                    fig_d = go.Figure()
                    fig_d.add_trace(go.Histogram(
                        x=port_ret_series, nbinsx=100, histnorm='probability density', 
                        name='Returns', marker_color='#1f77b4', opacity=0.5
                    ))
                    x_rng = np.linspace(port_ret_series.min(), port_ret_series.max(), 100)
                    fig_d.add_trace(go.Scatter(
                        x=x_rng, y=stats.norm.pdf(x_rng, port_ret_series.mean(), port_ret_series.std()), 
                        mode='lines', name='Normal Dist', line=dict(color='white', dash='dash')
                    ))
                    cutoff = all_var_results['Historical VaR (95%)']
                    x_tail = x_rng[x_rng <= cutoff]
                    y_tail = stats.norm.pdf(x_tail, port_ret_series.mean(), port_ret_series.std())
                    fig_d.add_trace(go.Scatter(
                        x=x_tail, y=y_tail, fill='tozeroy', 
                        fillcolor='rgba(239, 85, 59, 0.5)', line=dict(width=0), 
                        name='CVaR (Expected Shortfall) Area'
                    ))
                    fig_d.add_vline(x=all_var_results['Modified VaR (95%)'], line_dash="dot", line_color="yellow", annotation_text="Mod VaR 95%")
                    fig_d.add_vline(x=all_var_results['MC VaR (95%)'], line_dash="dot", line_color="green", annotation_text="MC VaR 95%")
                    fig_d.update_layout(template="plotly_dark", height=450, title="Distribution with CVaR Shading")
                    st.plotly_chart(fig_d, use_container_width=True)
                
                with c_stat:
                    st.metric("Skewness", f"{skew:.3f}")
                    st.metric("Kurtosis", f"{kurt:.3f}")
                    var_df = pd.DataFrame.from_dict(all_var_results, orient='index', columns=['Value'])
                    var_df['Value'] = var_df['Value'].apply(lambda x: f"{x:.2%}")
                    st.table(var_df)

            # --- TAB 7: ADVANCED MC SIMULATIONS ---
            with t7:
                st.markdown(f"### 🎲 Advanced Monte Carlo Simulations ({mc_method} Method)")
                st.markdown(f"**Configuration:** {mc_sims:,} simulations, {mc_days}-day horizon")
                
                # Display simulation statistics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Terminal Mean", f"{mc_stats.get('terminal_mean', 0):.4f}")
                col2.metric("Terminal Std Dev", f"{mc_stats.get('terminal_std', 0):.4f}")
                if 'skewness' in mc_stats:
                    col3.metric("Skewness", f"{mc_stats['skewness']:.3f}")
                if 'kurtosis' in mc_stats:
                    col4.metric("Kurtosis", f"{mc_stats['kurtosis']:.3f}")
                
                # 1. Path Visualization
                st.subheader("1. Simulated Paths")
                fig_paths = go.Figure()
                
                # Plot sample of paths
                n_sample = min(100, mc_sims)
                for i in range(n_sample):
                    fig_paths.add_trace(go.Scatter(
                        x=list(range(mc_days + 1)), y=mc_paths[i, :],
                        mode='lines', line=dict(width=0.5, color='rgba(100, 100, 100, 0.1)'),
                        showlegend=False
                    ))
                
                # Plot mean path
                mean_path = np.mean(mc_paths, axis=0)
                fig_paths.add_trace(go.Scatter(
                    x=list(range(mc_days + 1)), y=mean_path,
                    mode='lines', line=dict(width=3, color='#00cc96'),
                    name='Mean Path'
                ))
                
                # Plot confidence bands
                std_path = np.std(mc_paths, axis=0)
                fig_paths.add_trace(go.Scatter(
                    x=list(range(mc_days + 1)), y=mean_path + 1.96 * std_path,
                    mode='lines', line=dict(width=0), showlegend=False
                ))
                fig_paths.add_trace(go.Scatter(
                    x=list(range(mc_days + 1)), y=mean_path - 1.96 * std_path,
                    mode='lines', line=dict(width=0), fill='tonexty',
                    fillcolor='rgba(0, 204, 150, 0.2)', name='95% Confidence Band'
                ))
                
                fig_paths.update_layout(
                    title=f"Monte Carlo Simulation Paths ({mc_method})",
                    xaxis_title="Days",
                    yaxis_title="Portfolio Value (Normalized)",
                    template="plotly_dark",
                    height=500
                )
                st.plotly_chart(fig_paths, use_container_width=True)
                
                # 2. Terminal Distribution
                st.subheader("2. Terminal Value Distribution")
                
                col_hist, col_stats = st.columns([2, 1])
                
                with col_hist:
                    terminal_values = mc_paths[:, -1]
                    fig_term = go.Figure()
                    fig_term.add_trace(go.Histogram(
                        x=terminal_values, nbinsx=100,
                        name='Terminal Values',
                        marker_color='#636efa',
                        opacity=0.7
                    ))
                    
                    # Add VaR/CVaR lines
                    var_95 = np.percentile(terminal_values, 5)
                    cvar_95 = terminal_values[terminal_values <= var_95].mean()
                    
                    fig_term.add_vline(x=var_95, line_dash="dash", line_color="red", 
                                      annotation_text=f"VaR 95%: {var_95:.4f}")
                    fig_term.add_vline(x=cvar_95, line_dash="dot", line_color="orange", 
                                      annotation_text=f"CVaR 95%: {cvar_95:.4f}")
                    
                    fig_term.update_layout(
                        title="Distribution of Terminal Values",
                        xaxis_title="Terminal Portfolio Value",
                        yaxis_title="Frequency",
                        template="plotly_dark",
                        height=400
                    )
                    st.plotly_chart(fig_term, use_container_width=True)
                
                with col_stats:
                    # Statistical moments
                    moments_data = {
                        "Statistic": ["Mean", "Std Dev", "Skewness", "Kurtosis", "VaR 95%", "CVaR 95%"],
                        "Value": [
                            f"{np.mean(terminal_values):.4f}",
                            f"{np.std(terminal_values):.4f}",
                            f"{stats.skew(terminal_values):.3f}",
                            f"{stats.kurtosis(terminal_values):.3f}",
                            f"{var_95:.4f}",
                            f"{cvar_95:.4f}"
                        ]
                    }
                    st.table(pd.DataFrame(moments_data))
                
                # 3. Convergence Analysis
                st.subheader("3. Simulation Convergence")
                
                # Calculate running statistics
                running_means = np.cumsum(terminal_values) / np.arange(1, len(terminal_values) + 1)
                running_stds = [np.std(terminal_values[:i+1]) for i in range(len(terminal_values))]
                
                fig_conv = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("Mean Convergence", "Std Dev Convergence"),
                    vertical_spacing=0.15
                )
                
                fig_conv.add_trace(
                    go.Scatter(x=np.arange(len(running_means)), y=running_means,
                              mode='lines', name='Running Mean', line=dict(color='#00cc96')),
                    row=1, col=1
                )
                fig_conv.add_hline(y=mean_path[-1], line_dash="dash", line_color="gray", row=1, col=1)
                
                fig_conv.add_trace(
                    go.Scatter(x=np.arange(len(running_stds)), y=running_stds,
                              mode='lines', name='Running Std Dev', line=dict(color='#ef553b')),
                    row=2, col=1
                )
                
                fig_conv.update_layout(
                    height=600,
                    template="plotly_dark",
                    showlegend=True
                )
                st.plotly_chart(fig_conv, use_container_width=True)

            # --- TAB 8: QUANT LAB (GARCH & PCA) ---
            with t8:
                st.markdown("### 🔬 Quant Lab: Advanced Risk Decomposition")
                
                # 1. GARCH SECTION
                st.subheader("1. Econometric Volatility Modeling (GARCH 1,1)")
                if HAS_ARCH and garch_vol is not None:
                    fig_g = go.Figure()
                    fig_g.add_trace(go.Scatter(
                        x=port_ret_series.index, y=np.abs(port_ret_series), 
                        mode='markers', name='Abs Returns', marker=dict(color='gray', opacity=0.3, size=3)
                    ))
                    fig_g.add_trace(go.Scatter(
                        x=garch_vol.index, y=garch_vol/100, 
                        mode='lines', name='Conditional Volatility (GARCH)', line=dict(color='#EF553B', width=2)
                    ))
                    fig_g.update_layout(title="Volatility Clustering Analysis", template="plotly_dark", height=400, yaxis_title="Volatility")
                    st.plotly_chart(fig_g, use_container_width=True)
                    st.info("The Red Line shows the 'Conditional Volatility' forecast by the GARCH model. Spikes indicate periods where market turbulence is statistically likely to persist.")
                else:
                    st.warning("GARCH model could not be fitted (Library missing or convergence error).")
                
                st.markdown("---")
                
                # 2. PCA & COMPONENT VAR SECTION
                st.subheader("2. Factor & Component Risk Analysis")
                c_pca, c_comp = st.columns(2)
                
                with c_pca:
                    st.markdown("**Principal Component Analysis (Market Factors)**")
                    expl_var_cum = np.cumsum(pca_expl_var)
                    fig_pca = go.Figure()
                    fig_pca.add_trace(go.Bar(
                        x=[f"PC{i+1}" for i in range(len(pca_expl_var))], y=pca_expl_var, name='Individual Variance'
                    ))
                    fig_pca.add_trace(go.Scatter(
                        x=[f"PC{i+1}" for i in range(len(pca_expl_var))], y=expl_var_cum, name='Cumulative Variance', line=dict(color='yellow')
                    ))
                    fig_pca.update_layout(title="PCA Scree Plot (Dimensionality of Risk)", template="plotly_dark", height=400)
                    st.plotly_chart(fig_pca, use_container_width=True)

                with c_comp:
                    st.markdown("**Component VaR (Risk Contribution by Asset)**")
                    comp_var_sorted = comp_var.sort_values(ascending=False)
                    fig_cvar = px.bar(
                        x=comp_var_sorted.values, y=comp_var_sorted.index, orientation='h',
                        title="Contribution to Total Portfolio VaR",
                        labels={'x': 'Risk Contribution amount', 'y': 'Asset'},
                        color=comp_var_sorted.values, color_continuous_scale='OrRd'
                    )
                    fig_cvar.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig_cvar, use_container_width=True)

else:
    st.info("👈 Please configure the portfolio parameters in the sidebar to launch the engine.")
