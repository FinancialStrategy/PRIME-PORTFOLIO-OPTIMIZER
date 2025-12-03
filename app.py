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
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.cla import CLA
from pypfopt.hierarchical_portfolio import HRPOpt
from pypfopt.black_litterman import BlackLittermanModel
from sklearn.decomposition import PCA

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
    '^BVSP', '^MXX', '^MERV', # Latin America
    '^TA125.TA', '^CASE30', '^JN0U.JO' # Middle East
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
    
    REGION_MAP = {
        'US': US_DEFAULTS,
        'Europe': ['^FTSE', '^GDAXI', '^FCHI', 'SAP.DE', 'ASML.AS'],
        'Asia': ['^N225', '^HSI', '000001.SS', '005930.KS'],
        'Emerging Markets': ['^BVSP', '^MXX', '^MERV'],
        'Turkey': BIST_30
    }
    
    @staticmethod
    @st.cache_data(ttl=3600*24)
    def get_asset_metadata(tickers):
        """Fetches detailed metadata for each asset."""
        metadata = {}
        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).info
                metadata[ticker] = {
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown'),
                    'country': info.get('country', 'Unknown'),
                    'marketCap': info.get('marketCap', 0),
                    'fullName': info.get('longName', ticker),
                    'currency': info.get('currency', 'USD')
                }
            except:
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
        for region, tickers in AssetClassifier.REGION_MAP.items():
            if ticker in tickers:
                return region
        if '.IS' in ticker:
            return 'Turkey'
        elif '.DE' in ticker:
            return 'Germany'
        elif '.PA' in ticker:
            return 'France'
        elif '.L' in ticker:
            return 'UK'
        return 'Global'

# ============================================================================
# 3. PORTFOLIO ATTRIBUTION ENGINE
# ============================================================================

class PortfolioAttribution:
    """Performs Brinson-Fachler attribution analysis."""
    
    @staticmethod
    def calculate_attribution(portfolio_returns, benchmark_returns, 
                              portfolio_weights, benchmark_weights, 
                              sector_map):
        """Calculates Brinson attribution analysis."""
        # Group by sectors
        portfolio_sector_returns = {}
        benchmark_sector_returns = {}
        
        for ticker in portfolio_returns.columns:
            sector = sector_map.get(ticker, 'Other')
            if sector not in portfolio_sector_returns:
                portfolio_sector_returns[sector] = []
                benchmark_sector_returns[sector] = []
            
            portfolio_sector_returns[sector].append(portfolio_returns[ticker].mean())
            benchmark_sector_returns[sector].append(benchmark_returns[ticker].mean())
        
        # Calculate sector averages
        portfolio_sector_avg = {s: np.mean(v) for s, v in portfolio_sector_returns.items()}
        benchmark_sector_avg = {s: np.mean(v) for s, v in benchmark_sector_returns.items()}
        
        # Calculate weights at sector level
        portfolio_sector_weights = {}
        benchmark_sector_weights = {}
        
        for ticker, weight in portfolio_weights.items():
            sector = sector_map.get(ticker, 'Other')
            portfolio_sector_weights[sector] = portfolio_sector_weights.get(sector, 0) + weight
        
        for ticker, weight in benchmark_weights.items():
            sector = sector_map.get(ticker, 'Other')
            benchmark_sector_weights[sector] = benchmark_sector_weights.get(sector, 0) + weight
        
        # Calculate attribution
        allocation_effect = 0
        selection_effect = 0
        interaction_effect = 0
        
        for sector in set(list(portfolio_sector_weights.keys()) + list(benchmark_sector_weights.keys())):
            w_p = portfolio_sector_weights.get(sector, 0)
            w_b = benchmark_sector_weights.get(sector, 0)
            r_p = portfolio_sector_avg.get(sector, 0)
            r_b = benchmark_sector_avg.get(sector, 0)
            R_b = np.mean(list(benchmark_sector_avg.values()))
            
            allocation_effect += (w_p - w_b) * (r_b - R_b)
            selection_effect += w_b * (r_p - r_b)
            interaction_effect += (w_p - w_b) * (r_p - r_b)
        
        total_excess = allocation_effect + selection_effect + interaction_effect
        
        return {
            'Total Excess Return': total_excess,
            'Allocation Effect': allocation_effect,
            'Selection Effect': selection_effect,
            'Interaction Effect': interaction_effect
        }

# ============================================================================
# 4. ENHANCED TEARSHEET COMPONENTS
# ============================================================================

class EnhancedTearsheet:
    """Creates professional institutional-grade tearsheet components."""
    
    @staticmethod
    def create_kpi_grid(risk_metrics, perf_metrics, attribution=None):
        """Creates enhanced KPI grid with 8 metrics in 4x2 layout."""
        metrics_config = [
            {"label": "CAGR", "value": risk_metrics['CAGR'], "format": ".2%", "color": "#00cc96"},
            {"label": "Volatility", "value": risk_metrics['Volatility'], "format": ".2%", "color": "white"},
            {"label": "Sharpe Ratio", "value": risk_metrics['Sharpe Ratio'], "format": ".2f", 
             "color": "#00cc96" if risk_metrics['Sharpe Ratio'] > 1 else "white"},
            {"label": "Sortino Ratio", "value": risk_metrics['Sortino Ratio'], "format": ".2f", "color": "white"},
            {"label": "Max Drawdown", "value": risk_metrics['Max Drawdown'], "format": ".2%", "color": "#ef553b"},
            {"label": "Calmar Ratio", "value": risk_metrics['Calmar Ratio'], "format": ".2f", "color": "white"},
            {"label": "VaR 95%", "value": risk_metrics['VaR 95%'], "format": ".2%", "color": "#ff9900"},
            {"label": "Beta", "value": perf_metrics.get('beta', 1.0), "format": ".2f", "color": "white"},
        ]
        
        if attribution:
            metrics_config.extend([
                {"label": "Excess Return", "value": attribution['Total Excess Return'], "format": ".2%", "color": "#00cc96"},
                {"label": "Allocation Effect", "value": attribution['Allocation Effect'], "format": ".2%", "color": "#636efa"},
            ])
        
        # Create columns
        cols = st.columns(5)
        for idx, metric in enumerate(metrics_config[:5]):
            with cols[idx]:
                EnhancedTearsheet._render_metric_card(metric)
        
        cols2 = st.columns(5)
        for idx, metric in enumerate(metrics_config[5:10], 1):
            with cols2[idx-1]:
                EnhancedTearsheet._render_metric_card(metric)
    
    @staticmethod
    def _render_metric_card(metric):
        """Renders a single metric card."""
        st.markdown(f"""
        <div class="pro-card">
            <div class="metric-label">{metric['label']}</div>
            <div class="metric-value" style="color:{metric['color']}">{metric['value']:{metric['format']}}</div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_sector_allocation_chart(weights, metadata):
        """Creates a sunburst chart showing allocation by sector, then by asset."""
        # Prepare data for sunburst
        sectors = {}
        for ticker, weight in weights.items():
            if ticker in metadata:
                sector = metadata[ticker]['sector']
                if sector not in sectors:
                    sectors[sector] = {'total': 0, 'assets': {}}
                sectors[sector]['total'] += weight
                sectors[sector]['assets'][ticker] = weight
        
        # Build sunburst data
        ids = ["portfolio"]
        labels = ["Portfolio"]
        parents = [""]
        values = [100]
        
        # Add sectors
        for sector, data in sectors.items():
            sector_id = f"sector_{sector}"
            ids.append(sector_id)
            labels.append(sector)
            parents.append("portfolio")
            values.append(data['total'] * 100)
            
            # Add assets within sector
            for asset, weight in data['assets'].items():
                asset_id = f"asset_{asset}"
                ids.append(asset_id)
                labels.append(asset)
                parents.append(sector_id)
                values.append(weight * 100)
        
        fig = go.Figure(go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            textinfo="label+percent parent",
            marker=dict(
                colors=px.colors.qualitative.Plotly * 3,
                line=dict(width=0.5, color='#1e1e1e')
            ),
            hovertemplate='<b>%{label}</b><br>Allocation: %{value:.2f}%<br>Percentage of Parent: %{percentParent:.1%}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Portfolio Allocation by Sector",
            template="plotly_dark",
            height=500,
            margin=dict(t=50, l=0, r=0, b=0)
        )
        
        return fig
    
    @staticmethod
    def create_risk_return_scatter(returns_df, weights, metadata):
        """Creates risk-return scatter plot with portfolio positioning."""
        # Calculate individual asset metrics
        ann_factor = 252
        asset_returns = returns_df.mean() * ann_factor
        asset_vols = returns_df.std() * np.sqrt(ann_factor)
        asset_sharpe = (asset_returns - 0.02) / asset_vols
        
        # Portfolio metrics
        port_return = np.dot(list(weights.values()), asset_returns)
        port_vol = np.sqrt(np.dot(list(weights.values()), np.dot(returns_df.cov() * ann_factor, list(weights.values()))))
        
        # Create scatter plot
        fig = go.Figure()
        
        # Add individual assets
        for idx, (ticker, ret) in enumerate(asset_returns.items()):
            vol = asset_vols[ticker]
            sector = metadata.get(ticker, {}).get('sector', 'Unknown')
            
            fig.add_trace(go.Scatter(
                x=[vol],
                y=[ret],
                mode='markers',
                name=ticker,
                text=[ticker],
                hovertemplate=f"<b>{ticker}</b><br>Sector: {sector}<br>Return: {ret:.2%}<br>Vol: {vol:.2%}<extra></extra>",
                marker=dict(
                    size=np.sqrt(weights.get(ticker, 0) * 500) + 8,
                    color=asset_sharpe[ticker],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Sharpe Ratio"),
                    line=dict(width=1, color='white')
                ),
                showlegend=False
            ))
        
        # Add portfolio
        fig.add_trace(go.Scatter(
            x=[port_vol],
            y=[port_return],
            mode='markers',
            name='Portfolio',
            marker=dict(
                size=30,
                color='red',
                symbol='star',
                line=dict(width=2, color='white')
            ),
            hovertemplate="<b>Portfolio</b><br>Return: %{y:.2%}<br>Vol: %{x:.2%}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Risk-Return Positioning",
            xaxis_title="Annualized Volatility",
            yaxis_title="Annualized Return",
            template="plotly_dark",
            height=450,
            hovermode='closest'
        )
        
        return fig
    
    @staticmethod
    def create_allocation_timeline(prices, weights, rebalance_freq='Q'):
        """Creates a stacked area chart showing allocation evolution."""
        # Simulate allocation changes
        dates = prices.index
        allocations = pd.DataFrame(index=dates, columns=list(weights.keys()))
        
        # Identify rebalance dates
        if rebalance_freq == 'M':
            rebalance_dates = prices.resample('M').last().index
        elif rebalance_freq == 'Q':
            rebalance_dates = prices.resample('Q').last().index
        else:
            rebalance_dates = prices.resample('Y').last().index
        
        current_weights = weights.copy()
        
        for date in dates:
            # Check if rebalance date
            if date in rebalance_dates:
                current_weights = weights.copy()
            
            # Calculate current values based on price movements
            allocations.loc[date] = current_weights
        
        # Create stacked area chart
        fig = go.Figure()
        
        for ticker in allocations.columns:
            fig.add_trace(go.Scatter(
                x=allocations.index,
                y=allocations[ticker] * 100,
                mode='none',
                name=ticker,
                stackgroup='one',
                hovertemplate=f"<b>{ticker}</b><br>Allocation: %{{y:.1f}}%<extra></extra>"
            ))
        
        fig.update_layout(
            title="Portfolio Allocation Evolution",
            xaxis_title="Date",
            yaxis_title="Allocation %",
            template="plotly_dark",
            height=400,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    @staticmethod
    def create_performance_attribution_chart(attribution_results):
        """Creates waterfall chart for performance attribution."""
        categories = list(attribution_results.keys())
        values = list(attribution_results.values())
        
        fig = go.Figure(go.Waterfall(
            name="Attribution",
            orientation="v",
            measure=["relative"] * (len(categories) - 1) + ["total"],
            x=categories,
            y=values,
            text=[f"{v:.2%}" for v in values],
            textposition="outside",
            connector=dict(line=dict(color="rgba(255,255,255,0.3)")),
            increasing=dict(marker=dict(color="#00cc96")),
            decreasing=dict(marker=dict(color="#ef553b")),
            totals=dict(marker=dict(color="#636efa"))
        ))
        
        fig.update_layout(
            title="Performance Attribution Breakdown",
            template="plotly_dark",
            height=400,
            showlegend=False,
            yaxis_tickformat=".2%"
        )
        
        return fig
    
    @staticmethod
    def create_style_exposure_chart(weights, metadata):
        """Creates radar chart showing portfolio style exposures."""
        # Define style factors (simplified)
        styles = ['Growth', 'Value', 'Momentum', 'Quality', 'Low Vol', 'Size']
        
        # Calculate exposures (simplified)
        exposures = []
        for style in styles:
            if style == 'Growth':
                exposure = sum(weights.get(t, 0) for t, meta in metadata.items() 
                             if 'Technology' in meta.get('sector', ''))
            elif style == 'Value':
                exposure = sum(weights.get(t, 0) for t, meta in metadata.items() 
                             if meta.get('sector') in ['Financial Services', 'Energy'])
            elif style == 'Momentum':
                exposure = 0.5
            elif style == 'Quality':
                exposure = sum(weights.get(t, 0) for t, meta in metadata.items() 
                             if meta.get('sector') in ['Healthcare', 'Consumer Defensive'])
            elif style == 'Low Vol':
                exposure = 0.3
            else:  # Size
                exposure = 0.5
            exposures.append(exposure)
        
        # Complete the radar
        styles = styles + [styles[0]]
        exposures = exposures + [exposures[0]]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=exposures,
            theta=styles,
            fill='toself',
            name='Portfolio',
            line=dict(color='#00cc96', width=2),
            fillcolor='rgba(0, 204, 150, 0.3)'
        ))
        
        # Add benchmark (equal weight)
        benchmark_exposures = [0.5] * 6 + [0.5]
        fig.add_trace(go.Scatterpolar(
            r=benchmark_exposures,
            theta=styles,
            fill='toself',
            name='Benchmark',
            line=dict(color='rgba(100, 100, 100, 0.5)', dash='dash'),
            fillcolor='rgba(100, 100, 100, 0.1)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Portfolio Style Exposures",
            template="plotly_dark",
            height=450,
            showlegend=True
        )
        
        return fig

# ============================================================================
# 5. ADVANCED MONTE CARLO SIMULATOR
# ============================================================================

class AdvancedMonteCarloSimulator:
    """Advanced Monte Carlo simulation engine for VaR/CVaR."""
    
    def __init__(self, returns: pd.DataFrame, prices: pd.DataFrame):
        self.returns = returns
        self.prices = prices
        self.n_assets = len(returns.columns)
        self.mean_returns = returns.mean().values
        self.cov_matrix = returns.cov().values
        try:
            self.cholesky = np.linalg.cholesky(self.cov_matrix)
        except np.linalg.LinAlgError:
            self.cov_matrix = self.cov_matrix + np.eye(self.n_assets) * 1e-6
            self.cholesky = np.linalg.cholesky(self.cov_matrix)
    
    def gbm_simulation(self, weights: np.ndarray, days: int = 252, n_sims: int = 10000, 
                       antithetic: bool = True) -> Tuple[np.ndarray, Dict]:
        """Geometric Brownian Motion with correlation structure."""
        dt = 1/252
        mu = np.dot(weights, self.mean_returns)
        sigma = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        # Generate standard normal random variables
        if antithetic:
            z = npr.randn(n_sims//2, days)
            z = np.vstack([z, -z])
        else:
            z = npr.randn(n_sims, days)
        
        # GBM simulation
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        
        paths = np.zeros((n_sims, days + 1))
        paths[:, 0] = 1
        
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
        """Student's t-distribution simulation for fat tails."""
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
        """Merton Jump Diffusion model for capturing extreme events."""
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
    
    def filtered_historical_simulation(self, weights: np.ndarray, days: int = 252, 
                                       n_sims: int = 10000, block_size: int = 5) -> Tuple[np.ndarray, Dict]:
        """Filtered Historical Simulation - non-parametric approach using historical residuals."""
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
        """Calculate VaR and CVaR from simulated paths."""
        terminal_returns = (paths[:, -1] - 1)
        
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

# ============================================================================
# 6. DATA PIPELINE
# ============================================================================

class PortfolioDataManager:
    """Handles secure data fetching from Yahoo Finance."""
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_data(tickers, start_date, end_date):
        """Fetches OHLCV data for a list of tickers."""
        if not tickers:
            return pd.DataFrame(), {}
            
        try:
            if isinstance(tickers, str): tickers = [tickers]
            
            data = yf.download(
                tickers, 
                start=start_date, 
                end=end_date, 
                progress=False, 
                group_by='ticker', 
                threads=False
            )
            
            prices = pd.DataFrame()
            ohlc_dict = {}

            if len(tickers) == 1:
                ticker = tickers[0]
                df = data
                if isinstance(data.columns, pd.MultiIndex):
                    try: 
                        df = data.xs(ticker, axis=1, level=0, drop_level=True)
                    except: 
                        pass
                
                price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
                if price_col in df.columns:
                    prices[ticker] = df[price_col]
                    ohlc_dict[ticker] = df
            else:
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
                        continue
            
            prices = prices.ffill().bfill()
            prices = prices.dropna(axis=1, how='all')
            
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
        """Fetches Market Capitalization data."""
        mcaps = {}
        for t in tickers:
            try:
                mcaps[t] = yf.Ticker(t).info.get('marketCap', 10e9)
            except:
                mcaps[t] = 10e9
        return pd.Series(mcaps)

# ============================================================================
# 7. RISK ENGINE
# ============================================================================

class AdvancedRiskMetrics:
    """Institutional Risk Engine."""
    
    @staticmethod
    def calculate_metrics(returns, risk_free=0.02):
        """Generates the standard KPI matrix."""
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
        """Computes VaR using 3 distinct methodologies to highlight Model Risk."""
        results = {}
        
        # Distribution Moments
        mu = returns.mean()
        sigma = returns.std()
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)
        
        for conf in confidence_levels:
            alpha = 1 - conf
            
            # A. Parametric VaR (Normal Distribution Assumption)
            z_score = stats.norm.ppf(alpha)
            var_param = mu + z_score * sigma
            
            # B. Historical VaR (Empirical)
            var_hist = np.percentile(returns, alpha * 100)
            
            # C. Modified VaR (Cornish-Fisher Expansion)
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
        """Fits a GARCH(1,1) model to the portfolio returns."""
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
        """Decomposes VaR into asset-level contributions."""
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
# 8. PORTFOLIO OPTIMIZER
# ============================================================================

class AdvancedPortfolioOptimizer:
    """Wrapper for PyPortfolioOpt."""
    
    def __init__(self, returns, prices):
        self.returns = returns
        self.prices = prices
        self.mu = expected_returns.mean_historical_return(prices)
        self.S = risk_models.sample_cov(prices)
    
    def optimize(self, method, risk_free_rate=0.02, target_vol=0.10, target_ret=0.20, risk_aversion=1.0):
        """Main routing function for Mean-Variance based strategies."""
        ef = EfficientFrontier(self.mu, self.S)
        
        if method == 'Max Sharpe':
            weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
        elif method == 'Min Volatility':
            weights = ef.min_volatility()
        elif method == 'Efficient Risk':
            weights = ef.efficient_risk(target_volatility=target_vol)
        elif method == 'Efficient Return':
            weights = ef.efficient_return(target_return=target_ret)
        elif method == 'Max Quadratic Utility':
            weights = ef.max_quadratic_utility(risk_aversion=risk_aversion)
        
        return ef.clean_weights(), ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)

    def optimize_cla(self):
        """Critical Line Algorithm (CLA)."""
        cla = CLA(self.mu, self.S)
        weights = cla.max_sharpe()
        return cla.clean_weights(), cla.portfolio_performance(verbose=False)

    def optimize_hrp(self):
        """Hierarchical Risk Parity (HRP)."""
        hrp = HRPOpt(self.returns)
        weights = hrp.optimize()
        return hrp.clean_weights(), hrp.portfolio_performance(verbose=False)
        
    def optimize_black_litterman(self, market_caps):
        """Black-Litterman Model."""
        delta = risk_models.black_litterman.market_implied_risk_aversion(self.prices)
        prior = risk_models.black_litterman.market_implied_prior_returns(market_caps, delta, self.S)
        bl = BlackLittermanModel(self.S, pi=prior, absolute_views=None)
        ef = EfficientFrontier(bl.bl_returns(), bl.bl_cov())
        weights = ef.max_sharpe()
        return ef.clean_weights(), ef.portfolio_performance(verbose=False)

# ============================================================================
# 9. BACKTESTER
# ============================================================================

class PortfolioBacktester:
    """Performs a realistic path-dependent simulation."""
    
    def __init__(self, prices, returns):
        self.prices = prices
        self.returns = returns
        
    def run_rebalancing_backtest(self, weights, initial_capital=100000, rebalance_freq='Q', cost_bps=10):
        """Simulates the portfolio equity curve with transaction costs."""
        assets = self.returns.columns
        w_target = np.array([weights.get(a, 0) for a in assets])
        
        if rebalance_freq == 'M': 
            dates = self.returns.resample('M').last().index
        elif rebalance_freq == 'Q': 
            dates = self.returns.resample('Q').last().index
        else: 
            dates = self.returns.resample('Y').last().index
        
        rebalance_mask = pd.Series(False, index=self.prices.index)
        valid_dates = [d for d in dates if d in self.prices.index]
        rebalance_mask.loc[valid_dates] = True
        
        cash = initial_capital
        current_prices = self.prices.iloc[0].values
        current_shares = (cash * w_target) / current_prices
        cash -= initial_capital * (cost_bps / 10000)
        
        portfolio_history = []
        date_history = []
        
        for date, price_series in self.prices.iterrows():
            market_prices = price_series.values
            portfolio_value = np.sum(current_shares * market_prices)
            
            if rebalance_mask.loc[date]:
                target_value = portfolio_value 
                target_exposure = target_value * w_target
                target_shares = target_exposure / market_prices
                diff_shares = target_shares - current_shares
                turnover_value = np.sum(np.abs(diff_shares) * market_prices)
                cost = turnover_value * (cost_bps / 10000)
                portfolio_value -= cost
                current_shares = (portfolio_value * w_target) / market_prices
            
            portfolio_history.append(portfolio_value)
            date_history.append(date)
            
        return pd.Series(portfolio_history, index=date_history)

# ============================================================================
# 10. MAIN APPLICATION
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

# Jump Diffusion Parameters
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
            
            # 5. Enhanced Tearsheet Analytics
            classifier = AssetClassifier()
            asset_metadata = classifier.get_asset_metadata(selected_tickers)
            
            # 6. Dynamic Backtesting
            backtester = PortfolioBacktester(prices, returns)
            equity_curve = backtester.run_rebalancing_backtest(
                weights, 
                rebalance_freq=freq_map[rebal_freq_ui]
            )
            port_ret_series = equity_curve.pct_change().dropna()
            
            # 7. Risk Profiling
            risk_metrics = AdvancedRiskMetrics.calculate_metrics(port_ret_series, rf_rate)
            var_profile, skew, kurt = AdvancedRiskMetrics.calculate_comprehensive_risk_profile(port_ret_series)
            
            # Combine historical and MC VaR results
            all_var_results = {**var_profile, **mc_var_results}
            
            # 8. Performance Attribution
            eq_weights = {t: 1/len(selected_tickers) for t in selected_tickers}
            sector_map = {t: meta['sector'] for t, meta in asset_metadata.items()}
            attribution = PortfolioAttribution.calculate_attribution(
                returns, returns, weights, eq_weights, sector_map
            )
            
            # Calculate Beta
            port_returns_series = returns.dot(list(weights.values()))
            bench_returns_series = returns.dot(list(eq_weights.values()))
            covariance = np.cov(port_returns_series, bench_returns_series)[0][1]
            variance = np.var(bench_returns_series)
            beta = covariance / variance
            
            perf_metrics = {
                'beta': beta,
                'tracking_error': np.std(port_returns_series - bench_returns_series) * np.sqrt(252),
                'active_share': 0.5 * np.sum(np.abs(np.array(list(weights.values())) - np.array(list(eq_weights.values()))))
            }
            
            # 9. GARCH & PCA Analysis
            garch_model, garch_vol = AdvancedRiskMetrics.fit_garch_model(port_ret_series)
            comp_var, pca_expl_var, pca_obj = AdvancedRiskMetrics.calculate_component_var(returns, weights)
            
            # 10. Visualization Layout
            t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs([
                "🏛️ Enhanced Tearsheet", 
                "📈 Efficient Frontier", 
                "📉 Dynamic Backtest", 
                "🕯️ OHLC Analysis", 
                "🌪️ Stress Test", 
                "⚠️ Comparative VaR",
                "🎲 Advanced MC Simulations",
                "🔬 Quant Lab (GARCH/PCA)"
            ])
            
            # --- TAB 1: ENHANCED TEARSHEET ---
            with t1:
                st.markdown("### 🏛️ Institutional Portfolio Analytics")
                st.markdown("---")
                
                # Enhanced KPI Dashboard
                st.markdown("#### 📊 Key Performance Indicators")
                EnhancedTearsheet.create_kpi_grid(risk_metrics, perf_metrics, attribution)
                
                st.markdown("---")
                
                # Allocation & Attribution Visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    # Sector Allocation Sunburst
                    fig_sunburst = EnhancedTearsheet.create_sector_allocation_chart(weights, asset_metadata)
                    st.plotly_chart(fig_sunburst, use_container_width=True)
                    
                    # Style Exposures Radar
                    st.markdown("#### 🎯 Style Exposures")
                    fig_radar = EnhancedTearsheet.create_style_exposure_chart(weights, asset_metadata)
                    st.plotly_chart(fig_radar, use_container_width=True)
                
                with col2:
                    # Risk-Return Scatter
                    fig_scatter = EnhancedTearsheet.create_risk_return_scatter(returns, weights, asset_metadata)
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    # Performance Attribution Waterfall
                    st.markdown("#### 📈 Performance Attribution")
                    fig_waterfall = EnhancedTearsheet.create_performance_attribution_chart(attribution)
                    st.plotly_chart(fig_waterfall, use_container_width=True)
                
                st.markdown("---")
                
                # Allocation Evolution
                st.markdown("#### 📅 Allocation Timeline")
                fig_timeline = EnhancedTearsheet.create_allocation_timeline(
                    prices, weights, freq_map[rebal_freq_ui]
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
                
                # Asset Class Details Table
                st.markdown("#### 🏢 Asset Class Details")
                asset_details = []
                for ticker, weight in weights.items():
                    if weight > 0.001:
                        meta = asset_metadata.get(ticker, {})
                        asset_ret = returns[ticker].mean() * 252
                        asset_vol = returns[ticker].std() * np.sqrt(252)
                        asset_details.append({
                            'Asset': ticker,
                            'Weight': f"{weight:.2%}",
                            'Sector': meta.get('sector', 'Unknown'),
                            'Region': meta.get('country', 'Unknown'),
                            'Ann. Return': f"{asset_ret:.2%}",
                            'Ann. Vol': f"{asset_vol:.2%}",
                            'Contribution': f"{weight * asset_ret:.2%}"
                        })
                
                df_details = pd.DataFrame(asset_details)
                st.dataframe(df_details, use_container_width=True, hide_index=True)
                
                # Portfolio Characteristics Summary
                st.markdown("#### 🏛️ Portfolio Characteristics")
                col_char1, col_char2, col_char3, col_char4 = st.columns(4)
                
                total_market_cap = sum(meta.get('marketCap', 0) * weights.get(ticker, 0) 
                                      for ticker, meta in asset_metadata.items())
                avg_market_cap = total_market_cap / sum(weights.values()) if sum(weights.values()) > 0 else 0
                
                sector_counts = {}
                for ticker, weight in weights.items():
                    if weight > 0.001:
                        sector = asset_metadata.get(ticker, {}).get('sector', 'Unknown')
                        sector_counts[sector] = sector_counts.get(sector, 0) + 1
                
                with col_char1:
                    st.metric("Number of Holdings", f"{sum(w > 0.001 for w in weights.values())}")
                    st.metric("Active Share", f"{perf_metrics['active_share']:.1%}")
                
                with col_char2:
                    st.metric("Number of Sectors", f"{len(sector_counts)}")
                    st.metric("Tracking Error", f"{perf_metrics['tracking_error']:.2%}")
                
                with col_char3:
                    st.metric("Avg Market Cap", f"${avg_market_cap/1e9:.1f}B" if avg_market_cap > 0 else "N/A")
                    st.metric("Beta to Benchmark", f"{beta:.2f}")
                
                with col_char4:
                    turnover_est = "15-20%" if rebal_freq_ui == "Quarterly" else "5-10%"
                    st.metric("Portfolio Turnover", f"Est. {turnover_est}")
                    currencies = len(set(m.get('currency', '') for m in asset_metadata.values()))
                    st.metric("Currency Exposure", "Mixed" if currencies > 1 else "Single")
            
            # --- TAB 2: EFFICIENT FRONTIER ---
            with t2:
                st.subheader("Efficient Frontier Simulation (25,000 Runs)")
                
                n_sims = 25000
                w_rand = np.random.dirichlet(np.ones(len(selected_tickers)), n_sims)
                
                mu_np = optimizer.mu.to_numpy()
                S_np = optimizer.S.to_numpy()
                
                r_arr = w_rand @ mu_np
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
                fig_ef.update_layout(xaxis_title="Expected Volatility", yaxis_title="Expected Return", 
                                    height=600, template="plotly_dark")
                st.plotly_chart(fig_ef, use_container_width=True)
            
            # --- TAB 3: DYNAMIC BACKTEST ---
            with t3:
                st.subheader(f"Dynamic Backtest Analysis ({rebal_freq_ui} Rebalancing)")
                
                eq_weights = np.array([1/len(selected_tickers)] * len(selected_tickers))
                bench_ret = returns.dot(eq_weights)
                bench_curve = (1 + bench_ret).cumprod() * 100000
                
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, name="Active Strategy", 
                                           line=dict(color='#00cc96', width=2)))
                fig_bt.add_trace(go.Scatter(x=bench_curve.index, y=bench_curve, name="Equal Weight Index", 
                                           line=dict(color='#888', dash='dash')))
                fig_bt.update_layout(title="Equity Curve ($100k Initial)", template="plotly_dark", 
                                    height=500, xaxis_title="Date", yaxis_title="Portfolio Value ($)")
                st.plotly_chart(fig_bt, use_container_width=True)
                
                # Underwater Plot
                running_max = equity_curve.cummax()
                drawdown = (equity_curve - running_max) / running_max
                
                fig_dd = px.area(drawdown, title="Drawdown Profile", color_discrete_sequence=['#ef553b'])
                fig_dd.update_layout(template="plotly_dark", height=300, yaxis_title="Drawdown %")
                st.plotly_chart(fig_dd, use_container_width=True)
            
            # --- TAB 4: OHLC ANALYSIS ---
            with t4:
                st.subheader("📊 Detailed OHLC Analysis")
                tk_sel = st.selectbox("Inspect Asset", selected_tickers, key="ohlc_select")
                if tk_sel in ohlc_data:
                    df_ohlc = ohlc_data[tk_sel]
                    
                    # Create candlestick chart
                    fig_c = go.Figure(data=[go.Candlestick(
                        x=df_ohlc.index, 
                        open=df_ohlc['Open'], 
                        high=df_ohlc['High'], 
                        low=df_ohlc['Low'], 
                        close=df_ohlc['Close'],
                        name='OHLC'
                    )])
                    
                    # Add volume as bar chart
                    fig_c.add_trace(go.Bar(
                        x=df_ohlc.index,
                        y=df_ohlc['Volume'],
                        name='Volume',
                        yaxis='y2',
                        marker_color='rgba(100, 100, 100, 0.5)'
                    ))
                    
                    fig_c.update_layout(
                        title=f"{tk_sel} - Price Action with Volume",
                        yaxis_title="Price ($)",
                        yaxis2=dict(
                            title="Volume",
                            overlaying='y',
                            side='right',
                            showgrid=False
                        ),
                        height=600,
                        template="plotly_dark",
                        xaxis_rangeslider_visible=False,
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_c, use_container_width=True)
                    
                    # Display OHLC statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Price", f"${df_ohlc['Close'].iloc[-1]:.2f}")
                    with col2:
                        st.metric("Daily Change", f"{(df_ohlc['Close'].iloc[-1]/df_ohlc['Close'].iloc[-2]-1)*100:.2f}%")
                    with col3:
                        st.metric("Avg Volume", f"{df_ohlc['Volume'].mean():,.0f}")
                    with col4:
                        st.metric("Volatility", f"{df_ohlc['Close'].pct_change().std()*np.sqrt(252)*100:.2f}%")
                else:
                    st.warning("Detailed OHLC data unavailable for this ticker.")
            
            # --- TAB 5: STRESS TEST ---
            with t5:
                st.subheader("🌪️ Macro Scenario Analysis")
                
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
                    'Post-Recession Melt Up (+20%)': 0.20,
                    'Inflation Spike (-15%)': -0.15,
                    'Geopolitical Crisis (-25%)': -0.25,
                    'Market Correction (-12%)': -0.12
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
                
                # Display stress test results
                df_stress = pd.DataFrame(res_stress)
                st.dataframe(df_stress, use_container_width=True, hide_index=True)
                
                # Create visualization
                fig_stress = go.Figure()
                fig_stress.add_trace(go.Bar(
                    x=df_stress['Scenario'],
                    y=df_stress['Est. Portfolio Impact'].str.rstrip('%').astype(float),
                    text=df_stress['Est. Portfolio Impact'],
                    textposition='auto',
                    marker_color=np.where(df_stress['Est. Portfolio Impact'].str.rstrip('%').astype(float) < 0, 
                                        '#ef553b', '#00cc96')
                ))
                
                fig_stress.update_layout(
                    title="Portfolio Impact Under Stress Scenarios",
                    xaxis_title="Scenario",
                    yaxis_title="Portfolio Impact (%)",
                    template="plotly_dark",
                    height=500,
                    xaxis_tickangle=45
                )
                st.plotly_chart(fig_stress, use_container_width=True)
            
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
                    title="VaR Estimates by Methodology",
                    color_discrete_sequence=['#636EFA', '#EF553B', '#00CC96', '#FFA15A', '#AB63FA'],
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
                    cutoff = all_var_results.get('Historical VaR (95%)', 0)
                    x_tail = x_rng[x_rng <= cutoff]
                    y_tail = stats.norm.pdf(x_tail, port_ret_series.mean(), port_ret_series.std())
                    fig_d.add_trace(go.Scatter(
                        x=x_tail, y=y_tail, fill='tozeroy', 
                        fillcolor='rgba(239, 85, 59, 0.5)', line=dict(width=0), 
                        name='CVaR (Expected Shortfall) Area'
                    ))
                    fig_d.add_vline(x=all_var_results.get('Modified VaR (95%)', 0), line_dash="dot", 
                                  line_color="yellow", annotation_text="Mod VaR 95%")
                    fig_d.add_vline(x=all_var_results.get('MC VaR (95%)', 0), line_dash="dot", 
                                  line_color="green", annotation_text="MC VaR 95%")
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
                    fig_g = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    # Add absolute returns
                    fig_g.add_trace(
                        go.Scatter(
                            x=port_ret_series.index, 
                            y=np.abs(port_ret_series), 
                            mode='markers', 
                            name='Absolute Returns',
                            marker=dict(color='gray', opacity=0.3, size=3)
                        ),
                        secondary_y=False
                    )
                    
                    # Add GARCH volatility
                    fig_g.add_trace(
                        go.Scatter(
                            x=garch_vol.index, 
                            y=garch_vol, 
                            mode='lines', 
                            name='Conditional Volatility (GARCH)', 
                            line=dict(color='#EF553B', width=2)
                        ),
                        secondary_y=True
                    )
                    
                    fig_g.update_layout(
                        title="Volatility Clustering Analysis",
                        template="plotly_dark",
                        height=500,
                        xaxis_title="Date"
                    )
                    
                    fig_g.update_yaxes(title_text="Absolute Returns", secondary_y=False)
                    fig_g.update_yaxes(title_text="Conditional Volatility", secondary_y=True)
                    
                    st.plotly_chart(fig_g, use_container_width=True)
                    
                    # Display GARCH parameters
                    if garch_model is not None:
                        st.markdown("**GARCH(1,1) Parameters:**")
                        params = garch_model.params
                        col_g1, col_g2, col_g3 = st.columns(3)
                        with col_g1:
                            st.metric("Alpha (ARCH)", f"{params['alpha[1]']:.4f}")
                        with col_g2:
                            st.metric("Beta (GARCH)", f"{params['beta[1]']:.4f}")
                        with col_g3:
                            st.metric("Omega", f"{params['omega']:.6f}")
                    
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
                        x=[f"PC{i+1}" for i in range(len(pca_expl_var))], 
                        y=pca_expl_var, 
                        name='Individual Variance',
                        marker_color='#636efa'
                    ))
                    fig_pca.add_trace(go.Scatter(
                        x=[f"PC{i+1}" for i in range(len(pca_expl_var))], 
                        y=expl_var_cum, 
                        name='Cumulative Variance', 
                        line=dict(color='yellow', width=3),
                        mode='lines+markers'
                    ))
                    fig_pca.update_layout(
                        title="PCA Scree Plot (Dimensionality of Risk)",
                        xaxis_title="Principal Component",
                        yaxis_title="Variance Explained",
                        template="plotly_dark",
                        height=400
                    )
                    st.plotly_chart(fig_pca, use_container_width=True)
                    
                    # Display PCA insights
                    st.markdown("**PCA Insights:**")
                    st.info(f"First 3 PCs explain {expl_var_cum[2]*100:.1f}% of total variance")
                    st.info(f"First 5 PCs explain {expl_var_cum[4]*100:.1f}% of total variance")

                with c_comp:
                    st.markdown("**Component VaR (Risk Contribution by Asset)**")
                    comp_var_sorted = comp_var.sort_values(ascending=False)
                    fig_cvar = px.bar(
                        x=comp_var_sorted.values, 
                        y=comp_var_sorted.index, 
                        orientation='h',
                        title="Contribution to Total Portfolio VaR",
                        labels={'x': 'Risk Contribution', 'y': 'Asset'},
                        color=comp_var_sorted.values, 
                        color_continuous_scale='OrRd'
                    )
                    fig_cvar.update_layout(
                        template="plotly_dark", 
                        height=400,
                        xaxis_title="Risk Contribution Amount"
                    )
                    st.plotly_chart(fig_cvar, use_container_width=True)
                    
                    # Display top risk contributors
                    st.markdown("**Top Risk Contributors:**")
                    top_5 = comp_var_sorted.head(5)
                    for asset, risk in top_5.items():
                        st.metric(asset, f"{risk:.4f}")

else:
    st.info("👈 Please configure the portfolio parameters in the sidebar to launch the engine.")
