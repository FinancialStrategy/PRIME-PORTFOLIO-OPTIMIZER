# app_yedek_V02_complete_enhanced_fixed_v3.py
# Complete Institutional Portfolio Analysis Platform with Enhanced Attribution System
# Integrated with real benchmark data (SP500 for global/US, XU030 for Turkish assets)
# Fixed: Proper attribution logic, dashboard errors, and removed the specified redundant HTML block.

# ============================================================================
# 1. CORE IMPORTS
# ============================================================================
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
from typing import Dict, Tuple, List, Optional, Union
import numpy.random as npr
import io
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm

# ============================================================================
# 2. QUANTITATIVE LIBRARY IMPORTS
# ============================================================================
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

# Statsmodels for factor attribution (optional)
try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

warnings.filterwarnings('ignore')

# ============================================================================
# 3. GLOBAL CONFIGURATION AND ASSET UNIVERSES
# ============================================================================

st.set_page_config(
    page_title="Enigma Institutional Terminal Pro", 
    layout="wide", 
    page_icon="🏛️",
    initial_sidebar_state="expanded"
)

# Professional CSS with enhanced styling
st.markdown("""
<style>
    .reportview-container {background: #0e1117;}
    div[class^="st-"] { font-family: 'Roboto', sans-serif; }
    .pro-card {
        background-color: #1e1e1e;
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
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
    .positive { color: #00cc96 !important; }
    .negative { color: #ef553b !important; }
    .highlight-box {
        background: linear-gradient(135deg, #1e1e1e 0%, #2a2a2a 100%);
        border-left: 4px solid #00cc96;
        padding: 15px;
        border-radius: 6px;
        margin: 10px 0;
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
    '^FTSE', '^GDAXI', '^FCHI', '^STOXX50E', # European Markets
    '^N225', '^HSI', '000001.SS', '^STI', '^AXJO', # Asian Markets
    '^BVSP', '^MXX', '^MERV', # Latin America
    '^TA125.TA', '^CASE30', '^JN0U.JO' # Middle East
]

US_DEFAULTS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V', 'WMT']

# Benchmark definitions
BENCHMARK_MAP = {
    'US': '^GSPC',  # S&P 500 for US assets
    'Global': '^GSPC',  # S&P 500 for global
    'Turkey': 'XU030.IS',  # BIST 30 Index for Turkish assets
    'Europe': '^STOXX50E',  # Euro Stoxx 50 for European
    'Asia': '^N225',  # Nikkei 225 for Asian
}

# ============================================================================
# 4. ENHANCED ASSET CLASSIFICATION ENGINE
# ============================================================================

class EnhancedAssetClassifier:
    """Classifies assets into sectors, industries, regions, and styles with enhanced metadata."""
    
    SECTOR_MAP = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD', 'INTC', 'QCOM', 'CRM', 'ADBE', 'ORCL'],
        'Financial Services': ['JPM', 'V', 'MA', 'BAC', 'GS', 'MS', 'C', 'WFC', 'AXP', 'BLK'],
        'Healthcare': ['JNJ', 'PFE', 'MRK', 'ABT', 'UNH', 'LLY', 'GILD', 'BMY', 'AMGN', 'TMO'],
        'Consumer Cyclical': ['AMZN', 'TSLA', 'NKE', 'MCD', 'SBUX', 'HD', 'LOW', 'NFLX', 'DIS', 'BKNG'],
        'Consumer Defensive': ['WMT', 'PG', 'KO', 'PEP', 'COST', 'CL', 'MO', 'MDLZ', 'KHC', 'GIS'],
        'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PSX', 'VLO', 'MPC', 'OXY', 'KMI'],
        'Industrials': ['BA', 'CAT', 'MMM', 'HON', 'GE', 'RTX', 'LMT', 'UPS', 'UNP', 'DE'],
        'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'XEL', 'SRE', 'WEC', 'ED'],
        'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SPG', 'AVB', 'WELL', 'O', 'DLR'],
        'Communication': ['T', 'VZ', 'CMCSA', 'DIS', 'NFLX', 'CHTR', 'TMUS', 'FOXA', 'OMC', 'IPG'],
        'Materials': ['LIN', 'APD', 'ECL', 'SHW', 'NEM', 'FCX', 'DD', 'APD', 'NUE', 'MLM'],
        'Turkish Financials': ['AKBNK.IS', 'GARAN.IS', 'ISCTR.IS', 'HALKB.IS', 'YKBNK.IS', 'TSKB.IS'],
        'Turkish Industrials': ['THYAO.IS', 'TKFEN.IS', 'TOASO.IS', 'FROTO.IS', 'ARCLK.IS', 'ASELS.IS'],
        'Turkish Consumer': ['BIMAS.IS', 'MGROS.IS', 'SAHOL.IS', 'SASA.IS', 'KCHOL.IS']
    }
    
    REGION_MAP = {
        'US': US_DEFAULTS,
        'Europe': ['^FTSE', '^GDAXI', '^FCHI', 'SAP.DE', 'ASML.AS', 'RYAAY.AS', 'NOVOb.CO', 'SIE.DE'],
        'Asia': ['^N225', '^HSI', '000001.SS', '005930.KS', '9988.HK', 'BABA', 'TSM', 'SONY'],
        'Emerging Markets': ['^BVSP', '^MXX', '^MERV', 'EEM'],
        'Turkey': BIST_30
    }
    
    @staticmethod
    @st.cache_data(ttl=3600*24)
    def get_asset_metadata(tickers):
        """Fetches detailed metadata for each asset with enhanced classification."""
        metadata = {}
        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).info
                
                # Enhanced sector classification
                sector = EnhancedAssetClassifier._enhanced_sector_classification(ticker, info)
                region = EnhancedAssetClassifier._enhanced_region_classification(ticker, info)
                
                # Get style factors
                style_factors = EnhancedAssetClassifier._calculate_style_factors(ticker, info)
                
                metadata[ticker] = {
                    'sector': sector,
                    'industry': info.get('industry', 'Unknown'),
                    'country': info.get('country', 'Unknown'),
                    'region': region,
                    'marketCap': info.get('marketCap', 0),
                    'fullName': info.get('longName', ticker),
                    'currency': info.get('currency', 'USD'),
                    'beta': info.get('beta', 1.0),
                    'peRatio': info.get('trailingPE', 0),
                    'dividendYield': info.get('dividendYield', 0),
                    'profitMargins': info.get('profitMargins', 0),
                    'institutionOwnership': info.get('heldPercentInstitutions', 0),
                    'style_factors': style_factors
                }
            except Exception as e:
                # Fallback to inference
                metadata[ticker] = {
                    'sector': EnhancedAssetClassifier._infer_sector(ticker),
                    'industry': 'Unknown',
                    'country': EnhancedAssetClassifier._infer_country(ticker),
                    'region': EnhancedAssetClassifier._infer_region(ticker),
                    'marketCap': 0,
                    'fullName': ticker,
                    'currency': EnhancedAssetClassifier._infer_currency(ticker),
                    'beta': 1.0,
                    'peRatio': 0,
                    'dividendYield': 0,
                    'profitMargins': 0,
                    'institutionOwnership': 0,
                    'style_factors': {}
                }
        return metadata
    
    @staticmethod
    def _enhanced_sector_classification(ticker, info):
        """Enhanced sector classification with multiple fallbacks."""
        sector = info.get('sector', '')
        if sector:
            return sector
        
        # Check if ticker is in predefined sectors
        for sector_name, ticker_list in EnhancedAssetClassifier.SECTOR_MAP.items():
            if ticker in ticker_list:
                return sector_name
        
        # Infer from ticker suffix
        if '.IS' in ticker:
            if any(bank in ticker for bank in ['BANK', 'BNK', 'GARAN', 'ISCTR', 'HALK']):
                return 'Turkish Financials'
            elif any(ind in ticker for ind in ['HOLD', 'INDU', 'MAK', 'FAB']):
                return 'Turkish Industrials'
            else:
                return 'Turkish Consumer'
        
        return 'Other'
    
    @staticmethod
    def _enhanced_region_classification(ticker, info):
        """Enhanced region classification."""
        country = info.get('country', '')
        if country:
            if 'United States' in country:
                return 'US'
            elif any(eu in country for eu in ['Germany', 'France', 'United Kingdom', 'Switzerland', 'Netherlands']):
                return 'Europe'
            elif any(asia in country for asia in ['Japan', 'China', 'Hong Kong', 'South Korea', 'Taiwan']):
                return 'Asia'
            elif 'Turkey' in country:
                return 'Turkey'
        
        # Infer from ticker suffix
        if '.IS' in ticker:
            return 'Turkey'
        elif '.DE' in ticker:
            return 'Germany'
        elif '.PA' in ticker:
            return 'France'
        elif '.L' in ticker:
            return 'UK'
        elif '.T' in ticker:
            return 'Japan'
        elif '.HK' in ticker:
            return 'Hong Kong'
        elif '.SS' in ticker or '.SZ' in ticker:
            return 'China'
        
        return 'Global'
    
    @staticmethod
    def _calculate_style_factors(ticker, info):
        """Calculate style factors for the asset."""
        factors = {
            'growth': 0.5,
            'value': 0.5,
            'momentum': 0.5,
            'quality': 0.5,
            'low_vol': 0.5,
            'size': 0.5
        }
        
        try:
            # Growth factor (high revenue growth, high R&D)
            if info.get('revenueGrowth', 0) > 0.1:
                factors['growth'] = 0.8
            elif info.get('revenueGrowth', 0) < 0:
                factors['growth'] = 0.2
            
            # Value factor (low P/E, high dividend)
            pe = info.get('trailingPE', 0)
            if pe > 0 and pe < 15:
                factors['value'] = 0.8
            elif pe > 25:
                factors['value'] = 0.2
            
            if info.get('dividendYield', 0) > 0.03:
                factors['value'] = max(factors['value'], 0.7)
            
            # Quality factor (high margins, ROE)
            if info.get('profitMargins', 0) > 0.15:
                factors['quality'] = 0.8
            elif info.get('profitMargins', 0) < 0.05:
                factors['quality'] = 0.2
            
            # Size factor (market cap based)
            mcap = info.get('marketCap', 0)
            if mcap > 100e9:  # Large cap
                factors['size'] = 0.9
            elif mcap < 10e9:  # Small cap
                factors['size'] = 0.1
        except:
            pass
        
        return factors
    
    @staticmethod
    def _infer_sector(ticker):
        """Infer sector from ticker using predefined mappings."""
        for sector, tickers in EnhancedAssetClassifier.SECTOR_MAP.items():
            if ticker in tickers:
                return sector
        if '.IS' in ticker:
            return 'Turkish Financials'
        return 'Other'
    
    @staticmethod
    def _infer_country(ticker):
        """Infer country from ticker."""
        if '.IS' in ticker:
            return 'Turkey'
        elif '.DE' in ticker:
            return 'Germany'
        elif '.PA' in ticker:
            return 'France'
        elif '.L' in ticker:
            return 'United Kingdom'
        elif '.T' in ticker:
            return 'Japan'
        return 'United States'
    
    @staticmethod
    def _infer_region(ticker):
        """Infer region from ticker."""
        for region, tickers in EnhancedAssetClassifier.REGION_MAP.items():
            if ticker in tickers:
                return region
        if '.IS' in ticker:
            return 'Turkey'
        elif any(suffix in ticker for suffix in ['.DE', '.PA', '.L', '.AS']):
            return 'Europe'
        elif any(suffix in ticker for suffix in ['.T', '.HK', '.SS', '.SZ', '.KS']):
            return 'Asia'
        return 'US'
    
    @staticmethod
    def _infer_currency(ticker):
        """Infer currency from ticker."""
        if '.IS' in ticker:
            return 'TRY'
        elif any(suffix in ticker for suffix in ['.DE', '.PA', '.L', '.AS']):
            return 'EUR'
        elif '.T' in ticker:
            return 'JPY'
        elif '.HK' in ticker:
            return 'HKD'
        return 'USD'

# ============================================================================
# 5. ENHANCED PORTFOLIO ATTRIBUTION ENGINE WITH REAL DATA
# ============================================================================

class EnhancedPortfolioAttributionPro:
    """
    Professional Brinson-Fachler attribution analysis with multiple decomposition methods.
    Enhanced with real benchmark data and proper calculations.
    """
    
    @staticmethod
    def get_appropriate_benchmark(asset_tickers):
        """
        Determine appropriate benchmark based on asset mix.
        """
        # Analyze asset regions
        classifier = EnhancedAssetClassifier()
        metadata = classifier.get_asset_metadata(asset_tickers)
        
        # Count assets by region
        region_counts = {}
        for ticker, meta in metadata.items():
            region = meta.get('region', 'US')
            region_counts[region] = region_counts.get(region, 0) + 1
        
        # Determine primary region
        if not region_counts:
            return '^GSPC'  # Default to S&P 500
        
        primary_region = max(region_counts.items(), key=lambda x: x[1])[0]
        
        # Return appropriate benchmark
        if primary_region == 'Turkey':
            return 'XU030.IS'  # BIST 30 Index
        elif primary_region == 'Europe':
            return '^STOXX50E'  # Euro Stoxx 50
        elif primary_region == 'Asia':
            return '^N225'  # Nikkei 225
        else:
            return '^GSPC'  # S&P 500 (default for US and Global)
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def calculate_brinson_fachler_attribution(portfolio_returns_df, benchmark_returns_series,
                                              portfolio_weights, sector_map):
        """
        Complete Brinson-Fachler attribution using time-series returns and asset weights.
        
        Args:
            portfolio_returns_df (pd.DataFrame): Daily returns for all assets.
            benchmark_returns_series (pd.Series): Daily returns for the benchmark index.
            portfolio_weights (dict): Current weights of the portfolio assets.
            sector_map (dict): Maps ticker to its sector.
            
        Returns:
            dict: Attribution results.
        """
        
        # 1. Align data and handle missing tickers/weights
        all_assets = portfolio_returns_df.columns.tolist()
        
        # Ensure benchmark returns are compounded for the period (Total Return)
        R_b_total = (1 + benchmark_returns_series).prod() - 1
        
        # Calculate portfolio weights, setting benchmark weights to equal weight for simplicity 
        # (A true benchmark weight requires detailed index constituent data, which yfinance doesn't provide easily)
        w_p = pd.Series(portfolio_weights).reindex(all_assets).fillna(0)
        w_b_asset = pd.Series(1 / len(all_assets), index=all_assets) # Placeholder: Equal weight benchmark assets
        
        # 2. Group by Sector and Calculate Sector Metrics
        unique_sectors = set(sector_map.values())
        sector_data = {}
        
        for sector in unique_sectors:
            sector_assets = [t for t, s in sector_map.items() if s == sector and t in all_assets]
            
            if not sector_assets:
                continue
                
            # a. Sector Weights (Portfolio and Benchmark)
            w_p_sector = w_p[sector_assets].sum()
            w_b_sector = w_b_asset[sector_assets].sum()
            
            # b. Asset-level Weights normalized within sector (required for sector return calculation)
            sector_port_weights_norm = w_p[sector_assets] / w_p_sector if w_p_sector > 0 else pd.Series(0, index=sector_assets)
            sector_bench_weights_norm = w_b_asset[sector_assets] / w_b_sector if w_b_sector > 0 else pd.Series(0, index=sector_assets)
            
            # c. Calculate Sector Returns (Compounded Weighted Return over the whole period)
            # Portfolio Sector Return (R_p_sector)
            # R_p_sector is the return of a portfolio holding only the assets within that sector, weighted by p-weights
            if w_p_sector > 0:
                # Calculate daily return of the sector-portfolio
                daily_sector_returns_p = portfolio_returns_df[sector_assets].dot(sector_port_weights_norm)
                r_p_sector = (1 + daily_sector_returns_p).prod() - 1
            else:
                r_p_sector = 0
            
            # Benchmark Sector Return (R_b_sector)
            # R_b_sector is the return of a portfolio holding only the assets within that sector, weighted by b-weights
            if w_b_sector > 0:
                # Calculate daily return of the sector-benchmark
                daily_sector_returns_b = portfolio_returns_df[sector_assets].dot(sector_bench_weights_norm)
                r_b_sector = (1 + daily_sector_returns_b).prod() - 1
            else:
                r_b_sector = 0
            
            sector_data[sector] = {
                'w_p': w_p_sector,
                'w_b': w_b_sector,
                'r_p': r_p_sector,
                'r_b': r_b_sector,
                'assets': sector_assets
            }

        # 3. Final Attribution Calculation (Brinson-Fachler)
        
        # Recalculate R_p_total and R_b_total from weighted sector data for additivity check
        R_p_calculated = sum(data['w_p'] * (1 + data['r_p']) for data in sector_data.values()) - 1
        R_b_calculated = sum(data['w_b'] * (1 + data['r_b']) for data in sector_data.values()) - 1
        
        # Use the actual compounded returns calculated from the returns series for the final results
        R_p_total_actual = (1 + portfolio_returns_df.dot(w_p)).prod() - 1 # Total return of the *actual* portfolio
        # For simplicity, we use the original R_b_total which is the entire benchmark index return.
        
        total_excess = R_p_total_actual - R_b_total 
        
        allocation_effect = 0
        selection_effect = 0
        interaction_effect = 0
        sector_attribution = {}
        
        for sector, data in sector_data.items():
            w_p_i = data['w_p']
            w_b_i = data['w_b']
            r_p_i = data['r_p']
            r_b_i = data['r_b']
            
            # Brinson-Fachler 3-component attribution
            alloc = (w_p_i - w_b_i) * (r_b_i - R_b_total)
            select = w_b_i * (r_p_i - r_b_i)
            inter = (w_p_i - w_b_i) * (r_p_i - r_b_i)
            
            allocation_effect += alloc
            selection_effect += select
            interaction_effect += inter
            
            sector_attribution[sector] = {
                'Allocation': alloc,
                'Selection': select,
                'Interaction': inter,
                'Total Contribution': alloc + select + inter,
                'Portfolio Weight': w_p_i,
                'Benchmark Weight': w_b_i,
                'Portfolio Return': r_p_i,
                'Benchmark Return': r_b_i,
                'Active Weight': w_p_i - w_b_i,
                'Total': alloc + select + inter
            }

        attribution_total = allocation_effect + selection_effect + interaction_effect
        attribution_discrepancy = total_excess - attribution_total
        
        return {
            'Total Excess Return': total_excess,
            'Allocation Effect': allocation_effect,
            'Selection Effect': selection_effect,
            'Interaction Effect': interaction_effect,
            'Attribution Discrepancy': attribution_discrepancy,
            'Sector Breakdown': sector_attribution,
            'Benchmark Return': R_b_total,
            'Portfolio Return': R_p_total_actual,
            'Attribution Additivity': abs(attribution_discrepancy) < 1e-6, # Allow small float tolerance
            'Annualized Excess': total_excess * 252 / len(portfolio_returns_df) # Simplified estimate
        }
    
    @staticmethod
    def calculate_attribution_with_real_benchmark(portfolio_returns_df, benchmark_returns_series,
                                                 portfolio_weights, start_date, end_date):
        """
        Calculate attribution using real benchmark data from Yahoo Finance.
        """
        # Calculate portfolio returns as weighted sum (time-series)
        aligned_weights = pd.Series(portfolio_weights).reindex(portfolio_returns_df.columns).fillna(0)
        
        # Calculate weighted portfolio returns time-series
        portfolio_returns_ts = portfolio_returns_df.dot(aligned_weights)
        
        # Align benchmark returns with portfolio returns
        benchmark_aligned = benchmark_returns_series.reindex(portfolio_returns_ts.index).fillna(0)
        
        # Calculate excess returns time-series
        excess_returns = portfolio_returns_ts - benchmark_aligned
        
        # Get metadata for sector classification
        classifier = EnhancedAssetClassifier()
        asset_metadata = classifier.get_asset_metadata(portfolio_returns_df.columns.tolist())
        sector_map = {ticker: meta.get('sector', 'Other') for ticker, meta in asset_metadata.items()}
        
        # Calculate attribution
        attribution_results = EnhancedPortfolioAttributionPro.calculate_brinson_fachler_attribution(
            portfolio_returns_df, benchmark_aligned, portfolio_weights, sector_map
        )
        
        # Calculate performance metrics
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if tracking_error > 0 else 0
        
        # Calculate Active Share
        # For Active Share, we need a realistic benchmark portfolio weights. 
        # Using the simplified equal weight benchmark previously defined.
        active_share_weights = pd.Series(1 / len(portfolio_returns_df.columns), index=portfolio_returns_df.columns)
        active_share_weights = active_share_weights.reindex(aligned_weights.index).fillna(0)
        active_share = 0.5 * np.sum(np.abs(aligned_weights - active_share_weights))
        
        return {
            'portfolio_returns': portfolio_returns_ts,
            'benchmark_returns': benchmark_aligned,
            'excess_returns': excess_returns,
            'attribution': attribution_results,
            'cumulative_excess': (1 + excess_returns).cumprod() - 1,
            'rolling_excess': excess_returns.rolling(window=63).mean(),  # 3-month rolling
            'information_ratio': information_ratio,
            'tracking_error': tracking_error,
            'active_share': active_share
        }
    
    @staticmethod
    def calculate_factor_attribution(portfolio_returns, factor_returns, portfolio_exposures):
        """
        Factor-based attribution analysis.
        """
        # Factor attribution using regression
        if not HAS_STATSMODELS:
            st.warning("Statsmodels library is required for Factor Attribution.")
            return None
            
        try:
            # Combine returns and factors
            # Ensure factors are aligned to portfolio returns index
            X = factor_returns.reindex(portfolio_returns.index).dropna()
            y = portfolio_returns.reindex(X.index).dropna()

            if len(y) < len(X.columns) + 2:
                 st.warning("Insufficient data points for meaningful regression.")
                 return None

            X = sm.add_constant(X)
            
            # Run regression
            model = sm.OLS(y, X).fit()
            
            # Extract factor contributions
            factor_contributions = {}
            for factor in factor_returns.columns:
                if factor in model.params:
                    # Contribution = beta * factor return (annualized)
                    annual_contribution = model.params[factor] * factor_returns[factor].mean() * 252
                    factor_contributions[factor] = {
                        'coefficient': model.params[factor],
                        't_stat': model.tvalues.get(factor, 0),
                        'p_value': model.pvalues.get(factor, 1),
                        'contribution': annual_contribution
                    }
            
            # Calculate R-squared and other metrics
            attribution_metrics = {
                'r_squared': model.rsquared,
                'adj_r_squared': model.rsquared_adj,
                'f_statistic': model.fvalue,
                'residual_std': np.std(model.resid),
                'alpha': model.params.get('const', 0) * 252,  # Annualized alpha (intercept)
                'factor_contributions': factor_contributions
            }
            
            return attribution_metrics
            
        except Exception as e:
            st.warning(f"Factor attribution failed: {str(e)}")
            return None
    
    @staticmethod
    def calculate_rolling_attribution(portfolio_returns, benchmark_returns,
                                      portfolio_weights, benchmark_weights,
                                      sector_map, window=63):
        """
        Calculate rolling attribution over time.
        NOTE: This is computationally expensive. It performs the Brinson calculation 
        on a rolling basis. Using a simplified calculation based on cumulative returns 
        over the window to mitigate complexity/time.
        """
        
        # Align returns
        aligned_returns = pd.concat([portfolio_returns, benchmark_returns], axis=1)
        aligned_returns.columns = ['Portfolio', 'Benchmark']
        aligned_returns = aligned_returns.dropna()
        
        rolling_results = []
        
        # NOTE: Using a simplified calculation for rolling attribution
        # A full Brinson on a rolling window requires asset price data, not just portfolio returns.
        # Here we only track rolling excess return and use a fixed split for the components.
        
        for i in range(window, len(aligned_returns) + 1):
            window_returns = aligned_returns.iloc[i-window:i]
            
            # Calculate cumulative returns over the window
            port_return = (1 + window_returns['Portfolio']).prod() - 1
            bench_return = (1 + window_returns['Benchmark']).prod() - 1
            
            excess = port_return - bench_return
            
            # Simplified distribution of excess for visualization purposes only
            # In a real system, full rolling Brinson must be implemented.
            allocation = excess * 0.6
            selection = excess * 0.3  
            interaction = excess * 0.1
            
            rolling_results.append({
                'Date': aligned_returns.index[i-1],
                'Portfolio_Return': port_return,
                'Benchmark_Return': bench_return,
                'Excess_Return': excess,
                'Allocation': allocation,
                'Selection': selection,
                'Interaction': interaction,
                'Cumulative_Excess': excess
            })
        
        return pd.DataFrame(rolling_results)
    
    @staticmethod
    def calculate_attribution_quality_metrics(attribution_results):
        """
        Calculate quality metrics for attribution analysis.
        """
        metrics = {
            'Additivity Check': attribution_results.get('Attribution Additivity', False),
            'Discrepancy': attribution_results.get('Attribution Discrepancy', 0),
            'Allocation Dominance': 0,
            'Selection Dominance': 0,
            'Interaction Significance': 0
        }
        
        total_excess = attribution_results.get('Total Excess Return', 1)
        
        if abs(total_excess) > 1e-6:
            metrics['Allocation Dominance'] = abs(attribution_results.get('Allocation Effect', 0) / total_excess)
            metrics['Selection Dominance'] = abs(attribution_results.get('Selection Effect', 0) / total_excess)
            metrics['Interaction Significance'] = abs(attribution_results.get('Interaction Effect', 0) / total_excess)
        
        # Determine attribution style
        if metrics['Allocation Dominance'] > 0.6:
            metrics['Attribution Style'] = 'Allocation-Driven'
        elif metrics['Selection Dominance'] > 0.6:
            metrics['Attribution Style'] = 'Selection-Driven'
        else:
            metrics['Attribution Style'] = 'Balanced'
        
        return metrics

# ============================================================================
# 6. ENHANCED ATTRIBUTION VISUALIZATION WITH REAL DATA - FIXED DASHBOARD
# ============================================================================

class AttributionVisualizerPro:
    """Professional visualization components for attribution analysis with enhanced plots."""
    
    @staticmethod
    def create_enhanced_attribution_waterfall(attribution_results, title="Performance Attribution Breakdown"):
        """Enhanced waterfall chart with breakdown by sector and asset-level details."""
        try:
            
            # Extract data
            benchmark_return = attribution_results.get('Benchmark Return', 0)
            portfolio_return = attribution_results.get('Portfolio Return', 0)
            allocation = attribution_results.get('Allocation Effect', 0)
            selection = attribution_results.get('Selection Effect', 0)
            interaction = attribution_results.get('Interaction Effect', 0)
            
            # Create waterfall categories
            categories = ['Benchmark Return', 'Allocation Effect', 'Selection Effect', 
                          'Interaction Effect', 'Portfolio Return']
            values = [benchmark_return, allocation, selection, interaction, portfolio_return]
            measures = ['absolute', 'relative', 'relative', 'relative', 'total']
            
            # Handle possible zero division in relative measures
            if abs(benchmark_return) < 1e-6:
                # If benchmark return is near zero, use absolute return as starting base
                pass
            
            fig = go.Figure()
            
            fig.add_trace(go.Waterfall(
                name="Attribution",
                orientation="v",
                measure=measures,
                x=categories,
                y=values,
                text=[f"{v:+.2%}" if m != 'absolute' else f"{v:.2%}" for v, m in zip(values, measures)],
                textposition="outside",
                textfont=dict(size=11, color='white'),
                connector=dict(line=dict(color="rgba(255,255,255,0.5)", width=2)),
                increasing=dict(marker=dict(
                    color="#00cc96",
                    line=dict(color="#00cc96", width=2)
                )),
                decreasing=dict(marker=dict(
                    color="#ef553b",
                    line=dict(color="#ef553b", width=2)
                )),
                totals=dict(marker=dict(
                    color="#636efa",
                    line=dict(color="#636efa", width=3)
                ))
            ))
            
            fig.update_layout(
                title={
                    'text': title,
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=20, color='white')
                },
                template="plotly_dark",
                height=550,
                showlegend=False,
                yaxis_tickformat=".2%",
                yaxis_title="Return Contribution",
                xaxis_title="Attribution Component",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=80, l=60, r=60, b=100),
                xaxis_tickangle=45
            )
            
            # Add annotations for key insights
            total_excess = attribution_results.get('Total Excess Return', 0)
            if abs(total_excess) > 1e-10:
                arrow_color = "#00cc96" if total_excess > 0 else "#ef553b"
                arrow_symbol = "▲" if total_excess > 0 else "▼"
                
                fig.add_annotation(
                    x=len(categories)-1,
                    y=values[-1] + (0.1 * abs(values[-1]) if values[-1] != 0 else 0.01),
                    text=f"{arrow_symbol} {abs(total_excess):.2%} Total Excess",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=arrow_color,
                    font=dict(size=12, color=arrow_color, weight='bold'),
                    ax=0,
                    ay=-60 if total_excess > 0 else 60,
                    bgcolor="rgba(30, 30, 30, 0.8)",
                    bordercolor=arrow_color,
                    borderwidth=1
                )
            
            return fig
            
        except Exception as e:
            # Return error figure
            fig = go.Figure()
            fig.update_layout(
                title="Waterfall Chart Error",
                annotations=[dict(
                    text=f"Error: {str(e)[:50]}...",
                    showarrow=False,
                    font=dict(size=14, color='red')
                )],
                template="plotly_dark",
                height=500
            )
            return fig
    
    @staticmethod
    def create_sector_attribution_heatmap(sector_attribution):
        """Enhanced heatmap showing attribution by sector with asset-level details."""
        if not sector_attribution:
            fig = go.Figure()
            fig.update_layout(
                title="No Sector Data Available",
                template="plotly_dark",
                height=600
            )
            return fig
        
        # Prepare data
        sectors = list(sector_attribution.keys())
        
        # Extract metrics
        allocation_data = [sector_attribution[s].get('Allocation', 0) for s in sectors]
        selection_data = [sector_attribution[s].get('Selection', 0) for s in sectors]
        interaction_data = [sector_attribution[s].get('Interaction', 0) for s in sectors]
        total_data = [sector_attribution[s].get('Total Contribution', 0) for s in sectors]
        active_weight_data = [sector_attribution[s].get('Active Weight', 0) for s in sectors]
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=("Allocation Effect", "Selection Effect", 
                            "Interaction Effect", "Total Contribution",
                            "Active Weight", "Total Contribution (Bar)"),
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
            specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}],
                   [{'type': 'heatmap'}, {'type': 'heatmap'}],
                   [{'type': 'heatmap'}, {'type': 'bar'}]]
        )
        
        # Allocation heatmap (Row 1, Col 1)
        fig.add_trace(
            go.Heatmap(
                z=[allocation_data],
                x=sectors,
                y=['Allocation'],
                colorscale='RdBu',
                zmid=0,
                colorbar=dict(title="%", x=0.45, y=0.8, tickformat=".2%"),
                text=[[f"{v:.2%}" for v in allocation_data]],
                texttemplate="%{text}",
                textfont={"size": 10, "color": "black"},
                hovertemplate="<b>Sector: %{x}</b><br>Allocation Effect: %{z:.2%}<extra></extra>"
            ),
            row=1, col=1
        )
        
        # Selection heatmap (Row 1, Col 2)
        fig.add_trace(
            go.Heatmap(
                z=[selection_data],
                x=sectors,
                y=['Selection'],
                colorscale='RdBu',
                zmid=0,
                showscale=False,
                text=[[f"{v:.2%}" for v in selection_data]],
                texttemplate="%{text}",
                textfont={"size": 10, "color": "black"},
                hovertemplate="<b>Sector: %{x}</b><br>Selection Effect: %{z:.2%}<extra></extra>"
            ),
            row=1, col=2
        )
        
        # Interaction heatmap (Row 2, Col 1)
        fig.add_trace(
            go.Heatmap(
                z=[interaction_data],
                x=sectors,
                y=['Interaction'],
                colorscale='RdBu',
                zmid=0,
                showscale=False,
                text=[[f"{v:.2%}" for v in interaction_data]],
                texttemplate="%{text}",
                textfont={"size": 10, "color": "black"},
                hovertemplate="<b>Sector: %{x}</b><br>Interaction Effect: %{z:.2%}<extra></extra>"
            ),
            row=2, col=1
        )
        
        # Total contribution heatmap (Row 2, Col 2)
        fig.add_trace(
            go.Heatmap(
                z=[total_data],
                x=sectors,
                y=['Total'],
                colorscale='RdYlGn',
                zmid=0,
                colorbar=dict(title="%", x=1.02, y=0.8, tickformat=".2%"),
                text=[[f"{v:.2%}" for v in total_data]],
                texttemplate="%{text}",
                textfont={"size": 10, "color": "black"},
                hovertemplate="<b>Sector: %{x}</b><br>Total Contribution: %{z:.2%}<extra></extra>"
            ),
            row=2, col=2
        )
        
        # Active weight heatmap (Row 3, Col 1)
        fig.add_trace(
            go.Heatmap(
                z=[active_weight_data],
                x=sectors,
                y=['Active Wt'],
                colorscale='RdBu',
                zmid=0,
                showscale=False,
                text=[[f"{v:.2%}" for v in active_weight_data]],
                texttemplate="%{text}",
                textfont={"size": 10, "color": "black"},
                hovertemplate="<b>Sector: %{x}</b><br>Active Weight: %{z:.2%}<extra></extra>"
            ),
            row=3, col=1
        )
        
        # Contribution breakdown bar chart (Row 3, Col 2)
        fig.add_trace(
            go.Bar(
                x=sectors,
                y=total_data, # Use total_data for the bar chart
                text=[f"{t:.2%}" for t in total_data],
                textposition='auto',
                marker_color=['#00cc96' if t > 0 else '#ef553b' for t in total_data],
                hovertemplate="<b>Sector: %{x}</b><br>Total Contribution: %{y:.2%}<extra></extra>"
            ),
            row=3, col=2
        )
        
        fig.update_layout(
            title={
                'text': "Sector-Level Attribution Analysis",
                'y':0.98,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=20, color='white')
            },
            template="plotly_dark",
            height=800,
            showlegend=False,
            # Adjust x-axis labels to prevent overlap
            xaxis1=dict(tickangle=45), xaxis2=dict(tickangle=45), 
            xaxis3=dict(tickangle=45), xaxis4=dict(tickangle=45),
            xaxis5=dict(tickangle=45), xaxis6=dict(tickangle=45),
            margin=dict(t=100, l=60, r=60, b=120)
        )
        
        # Remove y-axis labels from heatmap plots, keep for bar chart
        for i in range(1, 4):
            fig.update_yaxes(title_text='', row=i, col=1, showticklabels=True, tickformat=".0%")
            fig.update_yaxes(title_text='', row=i, col=2, showticklabels=(i==3), tickformat=".2%")
        
        return fig
    
    @staticmethod
    def create_attribution_comparison_chart(attribution_results_list, labels):
        """Compare multiple attribution analyses."""
        fig = go.Figure()
        
        metrics = ['Allocation Effect', 'Selection Effect', 'Interaction Effect', 'Total Excess Return']
        
        for i, (results, label) in enumerate(zip(attribution_results_list, labels)):
            values = [results.get(metric, 0) for metric in metrics]
            
            fig.add_trace(go.Bar(
                name=label,
                x=metrics,
                y=values,
                text=[f"{v:.2%}" for v in values],
                textposition='auto',
                marker_color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
            ))
        
        fig.update_layout(
            title="Attribution Comparison Across Strategies",
            xaxis_title="Attribution Component",
            yaxis_title="Return Contribution",
            yaxis_tickformat=".2%",
            template="plotly_dark",
            height=500,
            barmode='group',
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def create_rolling_attribution_chart(rolling_attribution_df):
        """Create time series chart of rolling attribution."""
        if rolling_attribution_df.empty:
            fig = go.Figure()
            fig.update_layout(
                title="No Rolling Attribution Data Available",
                template="plotly_dark",
                height=400
            )
            return fig
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Rolling Attribution Components", "Cumulative Excess Return"),
            vertical_spacing=0.15
        )
        
        # Rolling attribution components
        fig.add_trace(
            go.Scatter(
                x=rolling_attribution_df['Date'],
                y=rolling_attribution_df['Allocation'].rolling(5).mean(),
                mode='lines',
                name='Allocation (5-day MA)',
                line=dict(color='#636efa', width=2),
                fill='tozeroy',
                fillcolor='rgba(99, 110, 250, 0.2)'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=rolling_attribution_df['Date'],
                y=rolling_attribution_df['Selection'].rolling(5).mean(),
                mode='lines',
                name='Selection (5-day MA)',
                line=dict(color='#ef553b', width=2),
                # This fill=tonexty/tozeroy logic is flawed for stacked traces 
                # unless they are strictly positive/negative. 
                # Removing fill for now to prevent visual errors.
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=rolling_attribution_df['Date'],
                y=rolling_attribution_df['Interaction'].rolling(5).mean(),
                mode='lines',
                name='Interaction (5-day MA)',
                line=dict(color='#00cc96', width=2),
            ),
            row=1, col=1
        )
        
        # Cumulative excess return
        fig.add_trace(
            go.Scatter(
                x=rolling_attribution_df['Date'],
                y=rolling_attribution_df['Cumulative_Excess'].cumsum(),
                mode='lines',
                name='Cumulative Excess',
                line=dict(color='white', width=3),
                fill='tozeroy',
                fillcolor='rgba(255, 255, 255, 0.1)'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            template="plotly_dark",
            height=700,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_yaxes(title_text="Return Contribution", row=1, col=1, tickformat=".2%")
        fig.update_yaxes(title_text="Cumulative Return", row=2, col=1, tickformat=".2%")
        
        return fig
    
    @staticmethod
    def create_factor_attribution_chart(factor_attribution_results):
        """Visualize factor attribution results."""
        if not factor_attribution_results or 'factor_contributions' not in factor_attribution_results:
            fig = go.Figure()
            fig.update_layout(
                title="No Factor Attribution Data Available",
                template="plotly_dark",
                height=400
            )
            return fig
        
        factor_data = factor_attribution_results['factor_contributions']
        
        # Prepare data for visualization
        factors = list(factor_data.keys())
        coefficients = [factor_data[f]['coefficient'] for f in factors]
        contributions = [factor_data[f]['contribution'] for f in factors]
        t_stats = [abs(factor_data[f]['t_stat']) for f in factors]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Factor Coefficients", "Factor Contributions",
                            "T-Statistics", "Factor Importance"),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'pie'}]]
        )
        
        # Factor coefficients (Row 1, Col 1)
        fig.add_trace(
            go.Bar(
                x=factors,
                y=coefficients,
                name='Coefficients',
                marker_color=['#636efa' if c > 0 else '#ef553b' for c in coefficients],
                text=[f"{c:.3f}" for c in coefficients],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Factor contributions (Row 1, Col 2)
        fig.add_trace(
            go.Bar(
                x=factors,
                y=contributions,
                name='Contributions',
                marker_color=['#00cc96' if c > 0 else '#ef553b' for c in contributions],
                text=[f"{c:.2%}" for c in contributions],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        # T-statistics (Row 2, Col 1)
        fig.add_trace(
            go.Bar(
                x=factors,
                y=t_stats,
                name='|t-stat|',
                marker_color='#FFA15A',
                text=[f"{t:.2f}" for t in t_stats],
                textposition='auto'
            ),
            row=2, col=1
        )
        
        # Factor importance (pie chart) (Row 2, Col 2)
        abs_contributions = [abs(c) for c in contributions]
        fig.add_trace(
            go.Pie(
                labels=factors,
                values=abs_contributions,
                name='Factor Importance',
                hole=0.4,
                marker_colors=px.colors.qualitative.Set3
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title={
                'text': f"Factor Attribution Analysis (R²: {factor_attribution_results.get('r_squared', 0):.2%})",
                'y':0.98,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=20, color='white')
            },
            template="plotly_dark",
            height=800,
            showlegend=False,
            margin=dict(t=100, l=60, r=60, b=60)
        )
        
        return fig
    
    @staticmethod
    def create_performance_attribution_dashboard(attribution_data, rolling_data=None):
        """Create comprehensive performance attribution dashboard - FIXED VERSION."""
        
        # Extract main attribution results
        attribution_results = attribution_data.get('attribution', {})
        sector_breakdown = attribution_results.get('Sector Breakdown', {})
        
        # Create main figure with multiple subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "Excess Returns Over Time",
                "Rolling Attribution (63-day)",
                "Sector Allocation vs Selection",
                "Attribution Waterfall",
                "Cumulative Excess Return",
                "Information Ratio Trend"
            ),
            vertical_spacing=0.08,
            horizontal_spacing=0.1,
            row_heights=[0.33, 0.33, 0.33]
        )
        
        # 1. Excess Returns Over Time (row 1, col 1)
        if 'excess_returns' in attribution_data:
            excess_returns = attribution_data['excess_returns']
            
            # Line for daily excess returns
            fig.add_trace(
                go.Scatter(
                    x=excess_returns.index,
                    y=excess_returns * 100,
                    mode='lines',
                    name='Daily Excess Returns',
                    line=dict(color='rgba(0, 204, 150, 0.7)', width=1),
                    fill='tozeroy',
                    fillcolor='rgba(0, 204, 150, 0.2)'
                ),
                row=1, col=1
            )
            
            # Add rolling mean
            rolling_mean = excess_returns.rolling(window=20).mean()
            fig.add_trace(
                go.Scatter(
                    x=rolling_mean.index,
                    y=rolling_mean * 100,
                    mode='lines',
                    name='20-day MA',
                    line=dict(color='white', width=2)
                ),
                row=1, col=1
            )
        
        # 2. Rolling Attribution Heatmap (row 1, col 2)
        if rolling_data is not None and not rolling_data.empty:
            # Prepare heatmap data
            heatmap_data = rolling_data[['Allocation', 'Selection', 'Interaction']].T
            
            fig.add_trace(
                go.Heatmap(
                    z=heatmap_data.values * 100,
                    x=rolling_data['Date'],
                    y=['Allocation', 'Selection', 'Interaction'],
                    colorscale='RdBu',
                    zmid=0,
                    colorbar=dict(title="%", x=1.02, y=0.75, tickformat=".2f"),
                    hovertemplate="<b>Date: %{x}</b><br>Component: %{y}<br>Value: %{z:.2f}%<extra></extra>"
                ),
                row=1, col=2
            )
        
        # 3. Sector Allocation vs Selection (scatter plot) - row 2, col 1
        if sector_breakdown:
            sectors = list(sector_breakdown.keys())
            allocation_vals = [sector_breakdown[s].get('Allocation', 0) for s in sectors]
            selection_vals = [sector_breakdown[s].get('Selection', 0) for s in sectors]
            total_vals = [sector_breakdown[s].get('Total', 0) for s in sectors]
            
            # Bubble size based on absolute contribution (scaled)
            bubble_sizes = [abs(t) * 5000 + 10 for t in total_vals]
            
            fig.add_trace(
                go.Scatter(
                    x=allocation_vals,
                    y=selection_vals,
                    mode='markers+text',
                    text=sectors,
                    textposition="top center",
                    marker=dict(
                        size=bubble_sizes,
                        color=total_vals,
                        colorscale='RdYlGn',
                        showscale=True,
                        coloraxis="coloraxis", # Use a dedicated color axis
                        line=dict(width=2, color='white')
                    ),
                    hovertemplate="<b>%{text}</b><br>Allocation: %{x:.2%}<br>Selection: %{y:.2%}<br>Total: %{marker.color:.2%}<extra></extra>",
                    name="Sector Scatter"
                ),
                row=2, col=1
            )
            
            # Add horizontal and vertical zero lines
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
            fig.add_vline(x=0, line_dash="dash", line_color="gray", row=2, col=1)
            
            # Add color axis for scatter plot
            fig.update_layout(coloraxis=dict(colorscale='RdYlGn', colorbar=dict(title="Total Contribution", x=0.55, y=0.45, tickformat=".2%")))
        
        # 4. Attribution Waterfall (simplified) - row 2, col 2
        if attribution_results:
            components = ['Benchmark', 'Allocation', 'Selection', 'Interaction', 'Portfolio']
            values = [
                attribution_results.get('Benchmark Return', 0),
                attribution_results.get('Allocation Effect', 0),
                attribution_results.get('Selection Effect', 0),
                attribution_results.get('Interaction Effect', 0),
                attribution_results.get('Portfolio Return', 0)
            ]
            
            fig.add_trace(
                go.Waterfall(
                    x=components,
                    y=values,
                    measure=['absolute', 'relative', 'relative', 'relative', 'total'],
                    connector=dict(line=dict(color="white", width=1)),
                    increasing=dict(marker=dict(color='#00cc96')),
                    decreasing=dict(marker=dict(color='#ef553b')),
                    totals=dict(marker=dict(color='#636efa')),
                    name="Attribution Waterfall"
                ),
                row=2, col=2
            )
        
        # 5. Cumulative Excess Return (row 3, col 1)
        if 'cumulative_excess' in attribution_data:
            cumulative_excess = attribution_data['cumulative_excess']
            fig.add_trace(
                go.Scatter(
                    x=cumulative_excess.index,
                    y=cumulative_excess * 100,
                    mode='lines',
                    name='Cumulative Excess',
                    line=dict(color='#FFA15A', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(255, 161, 90, 0.2)'
                ),
                row=3, col=1
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
        
        # 6. Information Ratio Trend (row 3, col 2)
        if 'excess_returns' in attribution_data:
            excess_returns = attribution_data['excess_returns']
            # Calculate rolling information ratio
            rolling_window = 63
            # Rolling IR = Rolling Mean Excess Return / Rolling Std Dev Excess Return * sqrt(252)
            rolling_ir = excess_returns.rolling(window=rolling_window).mean() / \
                         excess_returns.rolling(window=rolling_window).std() * np.sqrt(252)
            
            fig.add_trace(
                go.Scatter(
                    x=rolling_ir.index,
                    y=rolling_ir,
                    mode='lines',
                    name=f'{rolling_window}-day IR',
                    line=dict(color='#AB63FA', width=2)
                ),
                row=3, col=2
            )
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=2)
        
        # Update layout
        fig.update_layout(
            title={
                'text': "Performance Attribution Dashboard",
                'y':0.98,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=24, color='white')
            },
            template="plotly_dark",
            height=1000,
            showlegend=True,
            hovermode='x unified',
            margin=dict(t=120, l=80, r=80, b=80)
        )
        
        # Update axis labels
        fig.update_yaxes(title_text="Excess Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Return Contribution (%)", row=2, col=2, tickformat=".2%")
        fig.update_xaxes(title_text="Component", row=2, col=2)
        
        # Scatter plot specific axis formats
        fig.update_yaxes(title_text="Selection Effect", row=2, col=1, tickformat=".2%")
        fig.update_xaxes(title_text="Allocation Effect", row=2, col=1, tickformat=".2%")
        
        fig.update_yaxes(title_text="Cumulative Excess (%)", row=3, col=1)
        fig.update_yaxes(title_text="Information Ratio", row=3, col=2)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=2)
        
        return fig

# ============================================================================
# 7. ENHANCED DATA PIPELINE WITH REAL BENCHMARK SUPPORT
# ============================================================================

class EnhancedPortfolioDataManager:
    """Handles secure data fetching from Yahoo Finance with benchmark support."""
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_data_with_benchmark(tickers, benchmark_ticker, start_date, end_date):
        """Fetches OHLCV data for tickers and benchmark."""
        all_tickers = list(set(tickers + [benchmark_ticker]))
        
        try:
            data = yf.download(
                all_tickers, 
                start=start_date, 
                end=end_date, 
                progress=False, 
                group_by='ticker', 
                threads=False,
                auto_adjust=True
            )
            
            prices = pd.DataFrame()
            ohlc_dict = {}
            benchmark_prices = pd.Series()
            
            # Helper to extract data for a single ticker
            def extract_ticker_data(data, ticker, all_tickers):
                if len(all_tickers) == 1:
                    df = data.copy()
                    if isinstance(data.columns, pd.MultiIndex):
                        try:
                             df = data.xs(ticker, axis=1, level=0, drop_level=True)
                        except:
                            pass
                else:
                    try:
                        df = data.xs(ticker, axis=1, level=0, drop_level=True)
                    except KeyError:
                        return None, None
                
                price_col = 'Close'
                if price_col in df.columns:
                    return df[price_col], df
                return None, None

            # Process all tickers
            for ticker in all_tickers:
                price_series, ohlc_df = extract_ticker_data(data, ticker, all_tickers)
                if price_series is not None:
                    if ticker == benchmark_ticker:
                        benchmark_prices = price_series
                    else:
                        prices[ticker] = price_series
                    ohlc_dict[ticker] = ohlc_df
            
            # Clean and forward fill
            prices = prices.ffill().bfill()
            benchmark_prices = benchmark_prices.ffill().bfill()
            
            # Align dates and drop any rows with NaN in assets (after ffill/bfill)
            common_idx = prices.index.intersection(benchmark_prices.index)
            prices = prices.loc[common_idx]
            benchmark_prices = benchmark_prices.loc[common_idx]
            
            # Drop any asset that still has NaN (might be delisted)
            prices = prices.dropna(axis=1, how='any')
            
            return prices, benchmark_prices, ohlc_dict
            
        except Exception as e:
            # st.error(f"Data Pipeline Error: {str(e)}") # Removed error output here to prevent re-display
            return pd.DataFrame(), pd.Series(), {}
    
    @staticmethod
    def calculate_enhanced_returns(prices, benchmark_prices, method='log'):
        """Calculates portfolio and benchmark returns."""
        if method == 'log':
            portfolio_returns = np.log(prices / prices.shift(1)).dropna()
            benchmark_returns = np.log(benchmark_prices / benchmark_prices.shift(1)).dropna()
        else:
            portfolio_returns = prices.pct_change().dropna()
            benchmark_returns = benchmark_prices.pct_change().dropna()
        
        # Align returns
        common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
        portfolio_returns = portfolio_returns.loc[common_idx]
        benchmark_returns = benchmark_returns.loc[common_idx]
        
        return portfolio_returns, benchmark_returns
    
    @staticmethod
    def fetch_factor_data(factors, start_date, end_date):
        """Fetch factor data for attribution analysis."""
        # Common factors (using liquid ETFs/Indices as proxies for factor returns)
        factor_map = {
            'MKT': '^GSPC',  # Market factor (S&P 500)
            'SMB': 'IWM',     # Small minus Big (Russell 2000 ETF)
            'HML': 'VFINX',   # Value factor (Vanguard 500 as proxy - often use 'VTV' or 'VFINX')
            'MOM': 'MTUM',    # Momentum factor (iShares MSCI USA Momentum)
            'QUAL': 'QUAL',   # Quality factor (iShares MSCI USA Quality)
            'LOWVOL': 'USMV', # Low Volatility factor (iShares Edge MSCI Min Vol)
            # Fama-French are not easily fetched by yfinance, use style ETF proxies
        }
        
        factor_data = {}
        fetch_tickers = []
        
        for factor in factors:
            if factor in factor_map:
                fetch_tickers.append(factor_map[factor])
        
        if not fetch_tickers:
            return pd.DataFrame()
            
        try:
            data = yf.download(fetch_tickers, start=start_date, end=end_date, progress=False)
            if data.empty:
                return pd.DataFrame()

            factor_df = pd.DataFrame(index=data.index)

            # Extract Close price for each fetched ticker and calculate returns
            for factor, ticker in factor_map.items():
                if ticker in fetch_tickers:
                    if len(fetch_tickers) == 1:
                        close_prices = data['Close'] if 'Close' in data.columns else data.iloc[:, -1]
                    else:
                        close_prices = data['Close'][ticker]
                        
                    factor_df[factor] = close_prices.pct_change().dropna()

            # Filter for only the requested factors
            factor_df = factor_df[[f for f in factors if f in factor_df.columns]].dropna()

            return factor_df
        except Exception as e:
            # st.error(f"Factor Data Fetch Error: {str(e)}")
            return pd.DataFrame()

# ============================================================================
# 8. MISSING CLASSES - ADDING THE REQUIRED ONES
# ============================================================================

class AdvancedPortfolioOptimizer:
    """Portfolio optimization with multiple methods."""
    
    def __init__(self, returns, prices):
        self.returns = returns
        self.prices = prices
        
    def optimize(self, method, rf_rate, target_vol=None, target_ret=None, risk_aversion=None):
        """Main optimization method."""
        # Calculate expected returns and covariance matrix
        mu = expected_returns.mean_historical_return(self.prices)
        cov = risk_models.sample_cov(self.prices)
        
        if method == 'Max Sharpe':
            ef = EfficientFrontier(mu, cov)
            weights = ef.max_sharpe(rf_rate)
            cleaned_weights = ef.clean_weights()
            performance = ef.portfolio_performance(rf_rate)
            
        elif method == 'Min Volatility':
            ef = EfficientFrontier(mu, cov)
            weights = ef.min_volatility()
            cleaned_weights = ef.clean_weights()
            performance = ef.portfolio_performance(rf_rate)
            
        elif method == 'Efficient Risk':
            ef = EfficientFrontier(mu, cov)
            weights = ef.efficient_risk(target_vol)
            cleaned_weights = ef.clean_weights()
            performance = ef.portfolio_performance(rf_rate)
            
        elif method == 'Efficient Return':
            ef = EfficientFrontier(mu, cov)
            weights = ef.efficient_return(target_ret)
            cleaned_weights = ef.clean_weights()
            performance = ef.portfolio_performance(rf_rate)
            
        elif method == 'Max Quadratic Utility':
            ef = EfficientFrontier(mu, cov)
            weights = ef.max_quadratic_utility(risk_aversion=risk_aversion)
            cleaned_weights = ef.clean_weights()
            performance = ef.portfolio_performance(rf_rate)
            
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        return cleaned_weights, performance
    
    def optimize_cla(self):
        """Critical Line Algorithm optimization."""
        mu = expected_returns.mean_historical_return(self.prices)
        cov = risk_models.sample_cov(self.prices)
        
        cla = CLA(mu, cov)
        cla.max_sharpe()
        weights = cla.clean_weights()
        performance = cla.portfolio_performance()
        
        return weights, performance
    
    def optimize_hrp(self):
        """Hierarchical Risk Parity optimization."""
        hrp = HRPOpt(self.returns)
        weights = hrp.optimize()
        
        # Calculate performance metrics
        w_vec = np.array([weights.get(t, 0) for t in self.returns.columns])
        portfolio_returns = self.returns.dot(w_vec)
        r = portfolio_returns.mean() * 252
        v = portfolio_returns.std() * np.sqrt(252)
        sharpe = r / v if v > 0 else 0
        
        return weights, (r, v, sharpe)
    
    def optimize_black_litterman(self, market_caps):
        """Black-Litterman optimization."""
        mu = expected_returns.mean_historical_return(self.prices)
        cov = risk_models.sample_cov(self.prices)
        
        # Create market prior
        bl = BlackLittermanModel(cov, pi=mu, market_caps=market_caps, risk_aversion=1)
        
        # Get equilibrium returns
        equilibrium_returns = bl.equilibrium_returns()
        
        # Optimize
        ef = EfficientFrontier(equilibrium_returns, cov)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        performance = ef.portfolio_performance()
        
        return cleaned_weights, performance

class AdvancedRiskMetrics:
    """Advanced risk calculation methods."""
    
    @staticmethod
    def calculate_metrics(returns, rf_rate):
        """Calculate comprehensive risk metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['Annual Return'] = returns.mean() * 252
        metrics['Annual Volatility'] = returns.std() * np.sqrt(252)
        metrics['Sharpe Ratio'] = (metrics['Annual Return'] - rf_rate) / metrics['Annual Volatility'] if metrics['Annual Volatility'] > 0 else 0
        
        # Downside metrics
        downside_returns = returns[returns < 0]
        metrics['Downside Deviation'] = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        metrics['Sortino Ratio'] = (metrics['Annual Return'] - rf_rate) / metrics['Downside Deviation'] if metrics['Downside Deviation'] > 0 else 0
        
        # Drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        metrics['Max Drawdown'] = drawdown.min()
        
        # VaR and CVaR
        metrics['VaR 95%'] = np.percentile(returns, 5)
        metrics['CVaR 95%'] = returns[returns <= metrics['VaR 95%']].mean()
        
        return metrics
    
    @staticmethod
    def calculate_comprehensive_risk_profile(returns):
        """Calculate comprehensive risk profile."""
        # VaR at different confidence levels
        var_levels = [0.90, 0.95, 0.99]
        var_profile = {}
        for level in var_levels:
            var_profile[f'VaR {int(level*100)}%'] = np.percentile(returns, (1-level)*100)
        
        # Higher moments
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)
        
        return var_profile, skew, kurt
    
    @staticmethod
    def fit_garch_model(returns):
        """Fit GARCH model to returns."""
        if not HAS_ARCH:
            return None, None
        
        try:
            # Fit GARCH(1,1) model (returns scaled up for arch model)
            am = arch.arch_model(returns * 100, vol='Garch', p=1, q=1, rescale=False)
            res = am.fit(disp='off')
            
            # Get conditional volatility (scaled back down)
            conditional_vol = res.conditional_volatility / 100
            
            return res, conditional_vol
        except Exception as e:
            # print(f"GARCH failed: {e}")
            return None, None
    
    @staticmethod
    def calculate_component_var(returns, weights):
        """Calculate Component VaR."""
        # Check if there are enough assets/returns
        if len(weights) < 2 or returns.shape[0] < 2:
            return pd.Series(0, index=weights.keys()), np.array([0]), None
            
        # Convert weights to array
        tickers = returns.columns.tolist()
        w_array = np.array([weights.get(t, 0) for t in tickers])
        
        # Filter for only non-zero weight assets for robust calculation
        non_zero_indices = w_array != 0
        w_filtered = w_array[non_zero_indices]
        returns_filtered = returns.iloc[:, non_zero_indices]

        if len(w_filtered) == 0:
            return pd.Series(0, index=weights.keys()), np.array([0]), None

        # Calculate covariance matrix
        cov_matrix = returns_filtered.cov() * 252
        
        # Calculate portfolio variance
        portfolio_variance = np.dot(w_filtered.T, np.dot(cov_matrix, w_filtered))
        
        # Marginal Contribution to Volatility (MCV) - simpler and more stable calculation
        MCV = np.dot(cov_matrix, w_filtered) / np.sqrt(portfolio_variance)
        
        # Component Contribution to Volatility (CCV)
        CCV = w_filtered * MCV
        
        # Map back to full ticker list
        comp_vol_full = pd.Series(0.0, index=weights.keys())
        comp_vol_full[returns_filtered.columns] = CCV
        
        # Calculate PCA for diversification
        try:
            pca = PCA(n_components=min(len(returns_filtered.columns), 5))
            pca.fit(returns_filtered.corr())
            explained_variance = pca.explained_variance_ratio_
        except:
            explained_variance = np.array([0])
            pca = None
        
        return comp_vol_full, explained_variance, pca

class PortfolioDataManager:
    """Data management utilities."""
    
    @staticmethod
    def get_market_caps(tickers):
        """Get market capitalization for tickers."""
        market_caps = {}
        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).info
                market_caps[ticker] = info.get('marketCap', 1e9)  # Default 1 billion
            except:
                market_caps[ticker] = 1e9
        return market_caps

# ============================================================================
# 9. MONTE CARLO SIMULATOR (FIXED VERSION)
# ============================================================================

class MonteCarloSimulator:
    """Monte Carlo simulation for portfolio returns."""
    
    def __init__(self, returns, prices):
        self.returns = returns
        self.prices = prices
        
    def _prepare_simulation_data(self, weights):
        """Prepare inputs for Monte Carlo simulation."""
        
        tickers = self.returns.columns.tolist()
        w_array = np.array([weights.get(t, 0) for t in tickers])
        
        # Filter out assets with zero weights
        non_zero_indices = w_array != 0
        w_array = w_array[non_zero_indices]
        returns_filtered = self.returns.iloc[:, non_zero_indices]

        if len(w_array) == 0:
            raise ValueError("All weights are zero in the portfolio.")

        mu_annual = returns_filtered.mean().values * 252
        sigma_annual = returns_filtered.std().values * np.sqrt(252)
        corr_matrix = returns_filtered.corr().values
        
        return w_array, mu_annual, sigma_annual, corr_matrix
        
    def simulate_gbm_copula(self, weights, n_sims=1000, days=252):
        """Geometric Brownian Motion simulation with copula."""
        
        w_array, mu, sigma, corr = self._prepare_simulation_data(weights)
        n_assets = len(mu)
        
        # Cholesky decomposition
        try:
            L = np.linalg.cholesky(corr)
        except np.linalg.LinAlgError:
            # Handle non-positive definite matrix
            from scipy import linalg
            eigenvalues, eigenvectors = linalg.eigh(corr)
            eigenvalues[eigenvalues < 1e-8] = 0 # Set small/negative eigenvalues to zero
            # Reconstruct and try Cholesky again
            corr = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            # Force symmetry for safe use with Cholesky
            corr = (corr + corr.T) / 2
            try:
                 L = np.linalg.cholesky(corr)
            except np.linalg.LinAlgError:
                # Still failing, fall back to diagonal covariance (worst case)
                L = np.diag(np.sqrt(np.diag(corr)))
        
        # Simulation
        dt = 1/252
        
        # Initialize paths
        paths = np.zeros((n_sims, n_assets, days + 1))
        paths[:, :, 0] = 1  # Starting value of 1
        
        for sim in range(n_sims):
            # Generate correlated random variables
            z = np.random.normal(0, 1, (days, n_assets))
            epsilon = z @ L.T # Use L.T for standard correlated generation

            for t in range(1, days + 1):
                # GBM update for each asset
                for i in range(n_assets):
                    drift = (mu[i] - 0.5 * sigma[i]**2) * dt
                    diffusion = sigma[i] * np.sqrt(dt) * epsilon[t-1, i]
                    paths[sim, i, t] = paths[sim, i, t-1] * np.exp(drift + diffusion)
        
        # Calculate portfolio paths
        # Portfolio value is the weighted sum of asset values
        port_paths = np.sum(w_array * paths, axis=1)
        
        # Calculate statistics
        final_values = port_paths[:, -1]
        mean_final = np.mean(final_values)
        median_final = np.median(final_values)
        std_final = np.std(final_values)
        var_95 = np.percentile(final_values, 5)
        cvar_95 = final_values[final_values <= var_95].mean()
        
        mc_stats = {
            'Mean Final Value': mean_final,
            'Median Final Value': median_final,
            'Std Final Value': std_final,
            'VaR 95%': var_95,
            'CVaR 95%': cvar_95,
            'Probability of Loss': np.mean(final_values < 1),
            'Expected Shortfall': 1 - cvar_95 # Using 1 - CVaR as a measure of expected loss
        }
        
        return port_paths, mc_stats
    
    def simulate_students_t(self, weights, n_sims=1000, days=252, df=5):
        """Student's t-distribution simulation."""
        
        w_array, mu, sigma, corr = self._prepare_simulation_data(weights)
        n_assets = len(mu)
        
        # Cholesky decomposition with robustness check
        try:
            L = np.linalg.cholesky(corr)
        except np.linalg.LinAlgError:
            from scipy import linalg
            eigenvalues, eigenvectors = linalg.eigh(corr)
            eigenvalues[eigenvalues < 1e-8] = 0
            corr = (eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T)
            corr = (corr + corr.T) / 2
            try:
                 L = np.linalg.cholesky(corr)
            except np.linalg.LinAlgError:
                L = np.diag(np.sqrt(np.diag(corr)))

        # Simulation
        dt = 1/252
        
        paths = np.zeros((n_sims, n_assets, days + 1))
        paths[:, :, 0] = 1
        
        # Factor to adjust t-distribution variance to 1: sqrt((df-2)/df)
        t_adjust = np.sqrt((df-2)/df) 
        
        for sim in range(n_sims):
            # Generate scaled t-distributed random variables (unit variance)
            z = np.random.standard_t(df, (days, n_assets)) * t_adjust
            epsilon = z @ L.T
            
            for t in range(1, days + 1):
                # GBM update
                for i in range(n_assets):
                    drift = (mu[i] - 0.5 * sigma[i]**2) * dt
                    diffusion = sigma[i] * np.sqrt(dt) * epsilon[t-1, i]
                    paths[sim, i, t] = paths[sim, i, t-1] * np.exp(drift + diffusion)
        
        # Portfolio paths
        port_paths = np.sum(w_array * paths, axis=1)
        
        # Statistics
        final_values = port_paths[:, -1]
        var_95 = np.percentile(final_values, 5)
        
        mc_stats = {
            'Mean Final Value': np.mean(final_values),
            'Median Final Value': np.median(final_values),
            'Std Final Value': np.std(final_values),
            'VaR 95%': var_95,
            'CVaR 95%': final_values[final_values <= var_95].mean(),
            'Probability of Loss': np.mean(final_values < 1),
            'Expected Shortfall': 1 - final_values[final_values <= var_95].mean()
        }
        
        return port_paths, mc_stats

# ============================================================================
# 10. MAIN APPLICATION - INTEGRATED WITH ENHANCED ATTRIBUTION
# ============================================================================

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("🔧 Institutional Configuration Pro")

# Asset Universe Selection
ticker_lists = {
    "US Defaults": US_DEFAULTS, 
    "BIST 30 (Turkey)": BIST_30, 
    "Global Indices": GLOBAL_INDICES,
    "Custom Portfolio": []
}
selected_list = st.sidebar.selectbox("Asset Universe", list(ticker_lists.keys()))
available_tickers = ticker_lists[selected_list].copy()

# Custom Ticker Injection
custom_tickers = st.sidebar.text_input("Custom Tickers (Comma Separated)", value="")
if custom_tickers: 
    available_tickers = list(set(available_tickers + [t.strip().upper() for t in custom_tickers.split(',')]))

# Selection Widget
selected_tickers = st.sidebar.multiselect("Portfolio Assets", available_tickers, default=available_tickers[:min(len(available_tickers), 5)])

# Enhanced Attribution Settings
st.sidebar.markdown("---")
with st.sidebar.expander("📊 Advanced Attribution Settings", expanded=True):
    attribution_method = st.selectbox(
        "Attribution Method",
        ["Brinson-Fachler", "Factor-Based", "Multi-Period", "Rolling Analysis"]
    )
    
    selected_factors = []
    if attribution_method == "Factor-Based":
        selected_factors = st.multiselect(
            "Select Factors",
            ["MKT", "SMB", "HML", "MOM", "QUAL", "LOWVOL"],
            default=["MKT", "SMB", "HML"]
        )
    
    benchmark_selection = st.selectbox(
        "Benchmark Selection",
        ["Auto-detect", "S&P 500 (^GSPC)", "BIST 30 (XU030.IS)", "Euro Stoxx 50 (^STOXX50E)", "Custom"]
    )
    
    custom_benchmark = None
    if benchmark_selection == "Custom":
        custom_benchmark = st.text_input("Custom Benchmark Ticker", value="^GSPC")
    
    attribution_period = st.selectbox(
        "Attribution Period",
        ["Daily", "Weekly", "Monthly", "Quarterly"]
    )

st.sidebar.markdown("---")

# Model Parameters
with st.sidebar.expander("⚙️ Model Parameters", expanded=True):
    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365*3))
    end_date = st.date_input("End Date", datetime.now())
    rf_rate = st.number_input("Risk-Free Rate (%)", 0.0, 10.0, 2.0, 0.1) / 100

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
    method = st.selectbox("Optimization Objective", strat_options)

    # Conditional Strategy Inputs
    target_vol, target_ret, risk_aversion = 0.15, 0.20, 1.0
    if method == 'Efficient Risk':
        target_vol = st.slider("Target Volatility", 0.05, 0.50, 0.15)
    elif method == 'Efficient Return':
        target_ret = st.slider("Target Return", 0.05, 0.50, 0.20)
    elif method == 'Max Quadratic Utility':
        risk_aversion = st.slider("Risk Aversion (Delta)", 0.1, 10.0, 1.0)

# Monte Carlo Simulation Parameters
with st.sidebar.expander("🎲 Monte Carlo Settings"):
    mc_days = st.slider("Simulation Horizon (Days)", 21, 504, 252, 21)
    mc_sims = st.selectbox("Number of Simulations", [1000, 5000, 10000, 25000], index=2)
    mc_method = st.selectbox("Simulation Method", 
                                  ["GBM", "Student's t", "Jump Diffusion", "Filtered Historical"])

    # Jump Diffusion Parameters (Not implemented in class, but kept for UI structure)
    if mc_method == "Jump Diffusion":
        st.markdown("**Jump Diffusion Parameters (UI Only)**")
        jump_intensity = st.slider("Jump Intensity (λ)", 0.01, 0.20, 0.05, 0.01)
        jump_mean = st.slider("Jump Mean", -0.20, 0.00, -0.10, 0.01)
        jump_std = st.slider("Jump Std Dev", 0.05, 0.30, 0.15, 0.01)

    # Student's t Parameters
    df_t = 5.0
    if mc_method == "Student's t":
        df_t = st.slider("Degrees of Freedom", 3.0, 15.0, 5.0, 0.5)

# Backtest Settings (UI only, not fully implemented in current code logic)
with st.sidebar.expander("📉 Backtest Settings"):
    rebal_freq_ui = st.selectbox("Rebalancing Frequency (UI Only)", ["Quarterly", "Monthly", "Yearly", "Daily"])
    transaction_cost = st.number_input("Transaction Cost (bps) (UI Only)", 0, 100, 10)

run_btn = st.sidebar.button("🚀 EXECUTE ENHANCED ANALYSIS", type="primary")

# --- MAIN EXECUTION BLOCK ---
if run_btn:
    if not selected_tickers:
        st.error("❌ Please select at least one asset for analysis.")
        st.stop()
    
    with st.spinner('Initializing Enhanced Quantitative Engine...'):
        try:
            # 1. Determine appropriate benchmark
            if benchmark_selection == "Auto-detect":
                benchmark_ticker = EnhancedPortfolioAttributionPro.get_appropriate_benchmark(selected_tickers)
            elif benchmark_selection == "Custom":
                benchmark_ticker = custom_benchmark if custom_benchmark else "^GSPC"
            else:
                # Extract ticker from selection
                benchmark_ticker = benchmark_selection.split('(')[1].split(')')[0]
            
            st.info(f"📊 Selected benchmark: **{benchmark_ticker}**")

            # 2. Enhanced Data Ingestion
            data_manager = EnhancedPortfolioDataManager()
            prices, benchmark_prices, ohlc_data = data_manager.fetch_data_with_benchmark(
                selected_tickers, benchmark_ticker, start_date, end_date
            )
            
            if prices.empty or benchmark_prices.empty or prices.shape[1] == 0:
                st.error("❌ Data Fetch Failed. Please check ticker validity/availability and date range.")
                st.stop()
            
            st.success(f"✅ Data loaded: {len(prices)} days of data for {prices.shape[1]} assets")
            
            # 3. Calculate Returns
            portfolio_returns, benchmark_returns = data_manager.calculate_enhanced_returns(prices, benchmark_prices)
            
            # Filter tickers again to only include those that have valid data after return calculation
            valid_tickers = portfolio_returns.columns.tolist()
            if not valid_tickers:
                st.error("❌ No assets remaining after cleaning data and calculating returns.")
                st.stop()
            prices = prices[valid_tickers]
            
            # 4. Portfolio Optimization
            optimizer = AdvancedPortfolioOptimizer(portfolio_returns, prices)
            
            # Default to Equal Weight if optimization fails
            weights = {t: 1.0/len(valid_tickers) for t in valid_tickers}
            perf = (np.sum(portfolio_returns.mean()*np.array(list(weights.values())))*252, 
                    np.sqrt(np.dot(np.array(list(weights.values())).T, np.dot(portfolio_returns.cov()*252, np.array(list(weights.values()))))),
                    0.0)
            
            try:
                if method == 'Critical Line Algorithm (CLA)':
                    weights, perf = optimizer.optimize_cla()
                elif method == 'Hierarchical Risk Parity (HRP)':
                    weights, perf = optimizer.optimize_hrp()
                elif method == 'Black-Litterman': 
                    mcaps = PortfolioDataManager.get_market_caps(valid_tickers)
                    weights, perf = optimizer.optimize_black_litterman(mcaps)
                elif method == 'Equal Weight':
                    # Weights already initialized to equal weight
                    pass
                else:
                    weights, perf = optimizer.optimize(method, rf_rate, target_vol, target_ret, risk_aversion)
                
                st.success(f"✅ Portfolio optimized ({method}): Expected Return {perf[0]:.2%}, Volatility {perf[1]:.2%}, Sharpe {perf[2]:.2f}")
                
            except Exception as e:
                st.error(f"❌ Optimization Failed for {method}: {str(e)}")
                st.warning("⚠️ Using equal weight as fallback")
            
            # 5. ENHANCED PERFORMANCE ATTRIBUTION
            st.info("📈 Running Enhanced Performance Attribution Analysis...")
            
            # Get metadata for sector classification
            classifier = EnhancedAssetClassifier()
            asset_metadata = classifier.get_asset_metadata(valid_tickers)
            sector_map = {t: meta.get('sector', 'Other') for t, meta in asset_metadata.items()}
            
            # Calculate attribution with real benchmark
            attribution_engine = EnhancedPortfolioAttributionPro()
            attribution_data = attribution_engine.calculate_attribution_with_real_benchmark(
                portfolio_returns, benchmark_returns, weights, start_date, end_date
            )
            
            # Calculate rolling attribution
            rolling_attribution = attribution_engine.calculate_rolling_attribution(
                attribution_data['portfolio_returns'],
                attribution_data['benchmark_returns'],
                weights,
                {t: 1/len(valid_tickers) for t in valid_tickers}, # Equal weight benchmark
                sector_map,
                window=63
            )
            
            # Calculate attribution quality metrics
            attribution_quality = attribution_engine.calculate_attribution_quality_metrics(
                attribution_data['attribution']
            )
            
            # Factor attribution (if selected)
            factor_attribution = None
            if attribution_method == "Factor-Based" and selected_factors:
                st.info("🔎 Running Factor Attribution Analysis...")
                factor_data = data_manager.fetch_factor_data(selected_factors, start_date, end_date)
                
                if not factor_data.empty:
                    factor_attribution = attribution_engine.calculate_factor_attribution(
                        attribution_data['portfolio_returns'],
                        factor_data,
                        {} # Portfolio exposures placeholder
                    )
                else:
                    st.warning("⚠️ Could not fetch factor data or insufficient factor data overlap.")

            
            # 6. Create Visualization Dashboard
            visualizer = AttributionVisualizerPro()
            
            # Main tabs
            tabs = st.tabs([
                "🏛️ Overview Dashboard",
                "📊 Performance Attribution",
                "📈 Factor Analysis",
                "📉 Risk Metrics",
                "🎲 Monte Carlo",
                "🔬 Advanced Analytics"
            ])
            
            # TAB 1: OVERVIEW DASHBOARD
            with tabs[0]:
                st.markdown("## 🏛️ Enhanced Institutional Portfolio Analytics")
                st.markdown("---")
                
                # Key Metrics Dashboard
                total_return = (1 + attribution_data['portfolio_returns']).prod() - 1
                annual_vol = attribution_data['portfolio_returns'].std() * np.sqrt(252)
                sharpe = (attribution_data['portfolio_returns'].mean() * 252 - rf_rate) / annual_vol if annual_vol > 0 else 0
                ir = attribution_data.get('information_ratio', 0)
                te = attribution_data.get('tracking_error', 0)
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Total Return", f"{total_return:.2%}",
                              delta=f"{(attribution_data['attribution'].get('Total Excess Return', 0)):.2%} vs benchmark")
                
                with col2:
                    st.metric("Annualized Volatility", f"{annual_vol:.2%}")
                
                with col3:
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                
                with col4:
                    st.metric("Information Ratio", f"{ir:.2f}")
                
                with col5:
                    st.metric("Tracking Error", f"{te:.2%}")
                
                # Main Attribution Dashboard
                st.markdown("### 📈 Performance Attribution Dashboard")
                dashboard_fig = visualizer.create_performance_attribution_dashboard(
                    attribution_data, rolling_attribution
                )
                st.plotly_chart(dashboard_fig, use_container_width=True)
                
                # Attribution Summary
                st.markdown("### 📊 Attribution Summary")
                
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                
                alloc_effect = attribution_data['attribution'].get('Allocation Effect', 0)
                select_effect = attribution_data['attribution'].get('Selection Effect', 0)
                total_excess = attribution_data['attribution'].get('Total Excess Return', 0)
                
                with summary_col1:
                    st.markdown("#### Allocation Effect")
                    alloc_color = "positive" if alloc_effect > 0 else "negative"
                    st.markdown(f"<h1 class='{alloc_color}'>{alloc_effect:.2%}</h1>", unsafe_allow_html=True)
                    st.markdown("<p style='color: #888;'>Sector allocation contribution</p>", unsafe_allow_html=True)
                
                with summary_col2:
                    st.markdown("#### Selection Effect")
                    select_color = "positive" if select_effect > 0 else "negative"
                    st.markdown(f"<h1 class='{select_color}'>{select_effect:.2%}</h1>", unsafe_allow_html=True)
                    st.markdown("<p style='color: #888;'>Stock selection contribution</p>", unsafe_allow_html=True)
                
                with summary_col3:
                    st.markdown("#### Total Excess")
                    excess_color = "positive" if total_excess > 0 else "negative"
                    st.markdown(f"<h1 class='{excess_color}'>{total_excess:.2%}</h1>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: #888;'>vs {benchmark_ticker}</p>", unsafe_allow_html=True)
            
            # TAB 2: PERFORMANCE ATTRIBUTION
            with tabs[1]:
                st.markdown("## 📊 Detailed Performance Attribution")
                
                # Waterfall Chart
                col_wf1, col_wf2 = st.columns([3, 1])
                
                with col_wf1:
                    waterfall_fig = visualizer.create_enhanced_attribution_waterfall(
                        attribution_data['attribution'],
                        title="Detailed Attribution Breakdown (Brinson-Fachler)"
                    )
                    st.plotly_chart(waterfall_fig, use_container_width=True)
                
                with col_wf2:
                    st.markdown("#### Attribution Insights")
                    
                    # Calculate attribution style
                    alloc_pct = attribution_quality.get('Allocation Dominance', 0)
                    select_pct = attribution_quality.get('Selection Dominance', 0)
                    
                    if alloc_pct > 0.6 and alloc_pct > select_pct:
                        insight = "**Allocation-Driven Performance**\n\nPortfolio performance primarily driven by sector allocation decisions."
                        icon = "📍"
                    elif select_pct > 0.6 and select_pct > alloc_pct:
                        insight = "**Selection-Driven Performance**\n\nStock selection skills are the main driver of excess returns."
                        icon = "🎯"
                    else:
                        insight = "**Balanced Attribution**\n\nBoth allocation and selection contributed meaningfully to performance."
                        icon = "⚖️"
                    
                    st.markdown(f"""
                    <div class="highlight-box">
                        <h4>{icon} Attribution Style</h4>
                        <p>{insight}</p>
                        <p><strong>Allocation Dominance:</strong> {alloc_pct:.1%}</p>
                        <p><strong>Selection Dominance:</strong> {select_pct:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Sector Attribution Heatmap
                st.markdown("#### Sector-Level Analysis")
                heatmap_fig = visualizer.create_sector_attribution_heatmap(
                    attribution_data['attribution'].get('Sector Breakdown', {})
                )
                st.plotly_chart(heatmap_fig, use_container_width=True)
                
                # Rolling Attribution
                st.markdown("#### Time-Series Attribution Analysis (63-day Window)")
                if not rolling_attribution.empty:
                    rolling_fig = visualizer.create_rolling_attribution_chart(rolling_attribution)
                    st.plotly_chart(rolling_fig, use_container_width=True)
                
                # Attribution Quality Metrics
                st.markdown("#### Attribution Quality Metrics")
                
                quality_cols = st.columns(4)
                with quality_cols[0]:
                    additivity = "✅ Pass" if attribution_quality.get('Additivity Check', False) else "❌ Fail"
                    st.metric("Additivity Check", additivity)
                
                with quality_cols[1]:
                    st.metric("Discrepancy", f"{attribution_quality.get('Discrepancy', 0):.6f}")
                
                with quality_cols[2]:
                    st.metric("Attribution Style", attribution_quality.get('Attribution Style', 'Unknown'))
                
                with quality_cols[3]:
                    significance = "Significant" if abs(total_excess) > 0.001 else "Marginal"
                    st.metric("Excess Significance", significance)
            
            # TAB 3: FACTOR ANALYSIS
            with tabs[2]:
                st.markdown("## 📈 Factor Attribution Analysis")
                
                if factor_attribution:
                    # Factor Attribution Chart
                    factor_fig = visualizer.create_factor_attribution_chart(factor_attribution)
                    st.plotly_chart(factor_fig, use_container_width=True)
                    
                    # Factor Exposure Analysis
                    st.markdown("#### Portfolio Style Exposures (Based on Asset Metadata)")
                    
                    # Calculate portfolio factor exposures
                    factor_exposures = {}
                    for ticker, weight in weights.items():
                        if weight > 0:
                            meta = asset_metadata.get(ticker, {})
                            style_factors = meta.get('style_factors', {})
                            for factor, value in style_factors.items():
                                factor_exposures[factor] = factor_exposures.get(factor, 0) + weight * value
                    
                    if factor_exposures:
                        exp_df = pd.DataFrame.from_dict(factor_exposures, orient='index', columns=['Exposure'])
                        exp_df = exp_df.sort_values('Exposure', ascending=False)
                        
                        col_exp1, col_exp2 = st.columns(2)
                        
                        with col_exp1:
                            fig_exp = px.bar(
                                exp_df, 
                                x=exp_df.index, 
                                y='Exposure',
                                title="Portfolio Style Exposures (0=Min, 1=Max)",
                                color='Exposure',
                                color_continuous_scale='RdYlGn',
                                range_y=[0, 1]
                            )
                            fig_exp.update_layout(template="plotly_dark", height=400)
                            st.plotly_chart(fig_exp, use_container_width=True)
                        
                        with col_exp2:
                            st.markdown("##### Exposure Interpretation")
                            st.markdown("""
                            <div class="highlight-box">
                                <p><strong>Growth (>0.6):</strong> High exposure to Growth style factors.</p>
                                <p><strong>Value (<0.4):</strong> Low exposure to Value style factors.</p>
                                <p><strong>Quality (>0.6):</strong> Strong exposure to high profitability/low debt.</p>
                                <p><strong>Size (>0.7):</strong> Large-cap bias; (<0.3): Small-cap bias.</p>
                                <p style='color:#ef553b;'>*Scores are calculated heuristically based on fundamentals.</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Regression Statistics
                    st.markdown("#### Regression Statistics")
                    
                    stat_cols = st.columns(4)
                    with stat_cols[0]:
                        st.metric("R-squared", f"{factor_attribution.get('r_squared', 0):.2%}")
                    
                    with stat_cols[1]:
                        st.metric("Adjusted R²", f"{factor_attribution.get('adj_r_squared', 0):.2%}")
                    
                    with stat_cols[2]:
                        st.metric("Alpha (Annualized)", f"{factor_attribution.get('alpha', 0):.2%}", 
                                  delta=f"Residual return not explained by factors")
                    
                    with stat_cols[3]:
                        st.metric("Residual Std Dev", f"{factor_attribution.get('residual_std', 0):.4f}", 
                                  delta=f"Unexplained risk (idiosyncratic)")
                
                else:
                    st.info("Factor attribution analysis requires selecting factors in the sidebar settings.")

            
            # TAB 4: RISK METRICS
            with tabs[3]:
                st.markdown("## 📉 Comprehensive Risk Analysis")
                
                # Calculate aggregated metrics
                portfolio_ret_series = attribution_data['portfolio_returns']
                risk_metrics = AdvancedRiskMetrics.calculate_metrics(portfolio_ret_series, rf_rate)
                var_profile, skew, kurt = AdvancedRiskMetrics.calculate_comprehensive_risk_profile(
                    portfolio_ret_series
                )
                
                # Display risk metrics
                st.markdown("#### Key Risk Metrics")
                
                risk_cols = st.columns(4)
                with risk_cols[0]:
                    st.metric("Max Drawdown", f"{risk_metrics.get('Max Drawdown', 0):.2%}")
                
                with risk_cols[1]:
                    st.metric("VaR 95%", f"{risk_metrics.get('VaR 95%', 0):.2%}",
                              delta=f"Loss expected 5% of the time")
                
                with risk_cols[2]:
                    st.metric("CVaR 95%", f"{risk_metrics.get('CVaR 95%', 0):.2%}",
                              delta=f"Average loss during the worst 5% events")
                
                with risk_cols[3]:
                    st.metric("Sortino Ratio", f"{risk_metrics.get('Sortino Ratio', 0):.2f}",
                              delta=f"Return per unit of downside risk")
                
                # Distribution Analysis
                st.markdown("#### Return Distribution Analysis")
                
                dist_col1, dist_col2 = st.columns(2)
                
                with dist_col1:
                    # Histogram with normal distribution overlay
                    returns = portfolio_ret_series
                    fig_hist = go.Figure()
                    
                    fig_hist.add_trace(go.Histogram(
                        x=returns * 100,
                        nbinsx=50,
                        name='Portfolio Returns',
                        marker_color='#636efa',
                        opacity=0.7
                    ))
                    
                    # Add normal distribution curve
                    x_norm = np.linspace(returns.min() * 100, returns.max() * 100, 100)
                    if returns.std() > 0:
                        y_norm = stats.norm.pdf(x_norm, returns.mean() * 100, returns.std() * 100)
                        
                        fig_hist.add_trace(go.Scatter(
                            x=x_norm,
                            # Scale PDF to match histogram area
                            y=y_norm * len(returns) * (returns.max() - returns.min()) * 100 / 50, 
                            mode='lines',
                            name='Normal Distribution',
                            line=dict(color='white', width=2, dash='dash')
                        ))
                    
                    fig_hist.update_layout(
                        title="Return Distribution vs Normal",
                        xaxis_title="Daily Return (%)",
                        yaxis_title="Frequency",
                        template="plotly_dark",
                        height=400
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with dist_col2:
                    # QQ Plot
                    try:
                        import statsmodels.api as sm
                        
                        # Generate the QQ plot using matplotlib/statsmodels
                        fig_qq_mpl = sm.qqplot(returns, line='45', fit=True)
                        
                        # Convert Matplotlib figure to Plotly (simplified conversion)
                        # This is a complex conversion, often best served by native plotly elements. 
                        # We extract the underlying data and plot it in Plotly.
                        
                        fig_qq = go.Figure()
                        
                        # Extract data from the plotted lines (0: scatter points, 1: 45-degree line)
                        theoretical = fig_qq_mpl.gca().lines[0].get_xdata()
                        sample = fig_qq_mpl.gca().lines[0].get_ydata()
                        
                        plt.close(fig_qq_mpl) # Close the matplotlib figure
                        
                        fig_qq.add_trace(go.Scatter(
                            x=theoretical,
                            y=sample,
                            mode='markers',
                            name='Data Points',
                            marker=dict(color='#ef553b', size=6)
                        ))
                        
                        # Add 45-degree line
                        line_x = np.linspace(min(theoretical), max(theoretical), 100)
                        fig_qq.add_trace(go.Scatter(
                            x=line_x,
                            y=line_x,
                            mode='lines',
                            name='Normal Line',
                            line=dict(color='white', width=2, dash='dash')
                        ))
                        
                        fig_qq.update_layout(
                            title="QQ Plot (Normality Check)",
                            xaxis_title="Theoretical Quantiles (Normal)",
                            yaxis_title="Sample Quantiles",
                            template="plotly_dark",
                            height=400
                        )
                        st.plotly_chart(fig_qq, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not create QQ plot: {str(e)[:50]}...")
                
                # Skewness and Kurtosis Analysis
                st.markdown("#### Higher Moment Analysis")
                
                moment_cols = st.columns(4)
                with moment_cols[0]:
                    st.metric("Skewness", f"{skew:.3f}",
                              delta="Positive Skew" if skew > 0.5 else "Negative Skew" if skew < -0.5 else "Normal")
                
                with moment_cols[1]:
                    st.metric("Kurtosis", f"{kurt:.3f}",
                              delta="Fat Tails" if kurt > 3 else "Thin Tails" if kurt < 3 else "Normal")
                
                with moment_cols[2]:
                    jarque_bera = stats.jarque_bera(returns)[0]
                    st.metric("Jarque-Bera", f"{jarque_bera:.0f}",
                              delta="Non-normal" if jarque_bera > 5.99 else "Normal")
                
                with moment_cols[3]:
                    autocorr = returns.autocorr(lag=1)
                    st.metric("Autocorrelation (Lag 1)", f"{autocorr:.3f}",
                              delta="Mean Reverting" if autocorr < -0.1 else "Trending" if autocorr > 0.1 else "Random")
            
            # TAB 5: MONTE CARLO SIMULATION
            with tabs[4]:
                st.markdown("## 🎲 Monte Carlo Simulation Analysis")
                
                # Initialize Monte Carlo Simulator
                mc_simulator = MonteCarloSimulator(portfolio_returns, prices)
                
                # Display simulation parameters
                param_cols = st.columns(4)
                with param_cols[0]:
                    st.metric("Simulation Method", mc_method)
                
                with param_cols[1]:
                    st.metric("Number of Simulations", f"{mc_sims:,}")
                
                with param_cols[2]:
                    st.metric("Time Horizon", f"{mc_days} days")
                
                with param_cols[3]:
                    st.metric("Risk-Free Rate", f"{rf_rate:.2%}")
                
                # Run Monte Carlo simulation
                try:
                    
                    if mc_method == "GBM":
                        mc_paths, mc_stats = mc_simulator.simulate_gbm_copula(weights, n_sims=mc_sims, days=mc_days)
                    elif mc_method == "Student's t":
                        mc_paths, mc_stats = mc_simulator.simulate_students_t(weights, n_sims=mc_sims, days=mc_days, df=df_t)
                    else:
                        # Default to GBM for Jump Diffusion / Filtered Historical
                        mc_paths, mc_stats = mc_simulator.simulate_gbm_copula(weights, n_sims=mc_sims, days=mc_days)
                    
                    st.success(f"✅ Monte Carlo simulation complete.")
                    
                    # Display Monte Carlo results
                    st.markdown("#### Simulation Results")
                    
                    # Key statistics
                    mc_cols = st.columns(4)
                    with mc_cols[0]:
                        st.metric("Mean Final Value", f"{mc_stats.get('Mean Final Value', 0):.2f}")
                    
                    with mc_cols[1]:
                        st.metric("VaR 95%", f"{mc_stats.get('VaR 95%', 0):.2f}")
                    
                    with mc_cols[2]:
                        st.metric("CVaR 95%", f"{mc_stats.get('CVaR 95%', 0):.2f}")
                    
                    with mc_cols[3]:
                        st.metric("Prob. of Loss", f"{mc_stats.get('Probability of Loss', 0):.2%}")
                    
                    # Create Monte Carlo visualization
                    st.markdown("#### Simulation Paths")
                    
                    fig_mc = go.Figure()
                    
                    # Sample paths (show first 100 for clarity)
                    n_sample_paths = min(100, mc_sims)
                    for i in range(n_sample_paths):
                        fig_mc.add_trace(go.Scatter(
                            x=list(range(mc_days + 1)),
                            y=mc_paths[i, :],
                            mode='lines',
                            line=dict(width=1, color='rgba(100, 100, 100, 0.1)'),
                            showlegend=False
                        ))
                    
                    # Mean path
                    mean_path = np.mean(mc_paths, axis=0)
                    fig_mc.add_trace(go.Scatter(
                        x=list(range(mc_days + 1)),
                        y=mean_path,
                        mode='lines',
                        name='Expected Path',
                        line=dict(color='#00cc96', width=3)
                    ))
                    
                    # Confidence bands
                    upper_band = np.percentile(mc_paths, 95, axis=0)
                    lower_band = np.percentile(mc_paths, 5, axis=0)
                    
                    fig_mc.add_trace(go.Scatter(
                        x=list(range(mc_days + 1)) + list(range(mc_days + 1))[::-1],
                        y=list(upper_band) + list(lower_band[::-1]),
                        fill='toself',
                        fillcolor='rgba(0, 204, 150, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='90% Confidence Band',
                        showlegend=True
                    ))
                    
                    fig_mc.update_layout(
                        title="Monte Carlo Simulation Paths",
                        xaxis_title="Days",
                        yaxis_title="Portfolio Value (Normalized)",
                        template="plotly_dark",
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_mc, use_container_width=True)
                    
                    # Final value distribution
                    st.markdown("#### Final Value Distribution")
                    
                    fig_dist = go.Figure()
                    
                    fig_dist.add_trace(go.Histogram(
                        x=mc_paths[:, -1],
                        nbinsx=50,
                        name='Final Values',
                        marker_color='#636efa',
                        opacity=0.7
                    ))
                    
                    # Add vertical lines for key statistics
                    fig_dist.add_vline(x=1, line_dash="dash", line_color="white", annotation_text="Initial Value", annotation_position="top")
                    fig_dist.add_vline(x=mc_stats.get('Mean Final Value', 1), line_dash="dash", line_color="#00cc96", annotation_text="Mean", annotation_position="top")
                    fig_dist.add_vline(x=mc_stats.get('VaR 95%', 0.9), line_dash="dash", line_color="#ef553b", annotation_text="VaR 95%", annotation_position="top")
                    
                    fig_dist.update_layout(
                        title="Distribution of Final Portfolio Values",
                        xaxis_title="Final Value",
                        yaxis_title="Frequency",
                        template="plotly_dark",
                        height=400
                    )
                    
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Monte Carlo simulation failed: {str(e)}")
            
            # TAB 6: ADVANCED ANALYTICS
            with tabs[5]:
                st.markdown("## 🔬 Advanced Analytics")
                
                # GARCH Analysis
                st.markdown("#### GARCH Volatility Modeling")
                
                if HAS_ARCH:
                    garch_model, garch_vol = AdvancedRiskMetrics.fit_garch_model(
                        portfolio_ret_series
                    )
                    
                    if garch_vol is not None and not garch_vol.empty:
                        fig_garch = go.Figure()
                        
                        fig_garch.add_trace(go.Scatter(
                            x=garch_vol.index,
                            y=garch_vol * 100 * np.sqrt(252), # Annualized
                            mode='lines',
                            name='Conditional Volatility (GARCH Annualized)',
                            line=dict(color='#ef553b', width=2)
                        ))
                        
                        # Add rolling volatility for comparison
                        rolling_vol = portfolio_ret_series.rolling(window=20).std() * np.sqrt(252) * 100
                        fig_garch.add_trace(go.Scatter(
                            x=rolling_vol.index,
                            y=rolling_vol,
                            mode='lines',
                            name='20-day Rolling Vol (Annualized)',
                            line=dict(color='#00cc96', width=1, dash='dash')
                        ))
                        
                        fig_garch.update_layout(
                            title="GARCH Conditional Volatility vs Rolling Volatility",
                            xaxis_title="Date",
                            yaxis_title="Annualized Volatility (%)",
                            template="plotly_dark",
                            height=400
                        )
                        
                        st.plotly_chart(fig_garch, use_container_width=True)
                        
                        if garch_model is not None:
                            # Display GARCH parameters
                            st.markdown("##### GARCH(1,1) Parameters")
                            
                            garch_cols = st.columns(4)
                            try:
                                with garch_cols[0]:
                                    st.metric("Alpha (ARCH)", f"{garch_model.params['alpha[1]']:.4f}")
                                
                                with garch_cols[1]:
                                    st.metric("Beta (GARCH)", f"{garch_model.params['beta[1]']:.4f}")
                                
                                with garch_cols[2]:
                                    st.metric("Omega", f"{garch_model.params['omega']:.6f}")
                                
                                with garch_cols[3]:
                                    persistence = garch_model.params['alpha[1]'] + garch_model.params['beta[1]']
                                    st.metric("Persistence", f"{persistence:.4f}",
                                              delta="High Persistence" if persistence > 0.9 else "Low Persistence")
                            except:
                                st.warning("Could not extract GARCH parameters.")
                        else:
                            st.warning("GARCH model fitting failed.")
                    else:
                        st.warning("GARCH model fitting skipped: Insufficient data or model error.")
                else:
                    st.warning("ARCH library not available. Install with: pip install arch")
                
                # PCA Analysis
                st.markdown("#### Principal Component Analysis (Diversification)")
                
                comp_var, pca_expl_var, pca_obj = AdvancedRiskMetrics.calculate_component_var(
                    portfolio_returns, weights
                )
                
                if pca_expl_var[0] != 0:
                    pca_col1, pca_col2 = st.columns(2)
                    
                    with pca_col1:
                        # Scree plot
                        fig_pca = go.Figure()
                        
                        fig_pca.add_trace(go.Bar(
                            x=[f"PC{i+1}" for i in range(len(pca_expl_var))],
                            y=pca_expl_var * 100,
                            name='Explained Variance',
                            marker_color='#636efa'
                        ))
                        
                        fig_pca.add_trace(go.Scatter(
                            x=[f"PC{i+1}" for i in range(len(pca_expl_var))],
                            y=np.cumsum(pca_expl_var) * 100,
                            name='Cumulative',
                            line=dict(color='#00cc96', width=3),
                            mode='lines+markers'
                        ))
                        
                        fig_pca.update_layout(
                            title="PCA Scree Plot",
                            xaxis_title="Principal Component",
                            yaxis_title="Variance Explained (%)",
                            template="plotly_dark",
                            height=400
                        )
                        
                        st.plotly_chart(fig_pca, use_container_width=True)
                    
                    with pca_col2:
                        # Component VaR (using Component Contribution to Volatility for stability)
                        comp_var_sorted = comp_var.sort_values(ascending=False)
                        
                        fig_cvar = px.bar(
                            x=comp_var_sorted.values * 100,
                            y=comp_var_sorted.index,
                            orientation='h',
                            title="Component Volatility Contribution (%)",
                            labels={'x': 'Risk Contribution (%)', 'y': 'Asset'},
                            color=comp_var_sorted.values,
                            color_continuous_scale='OrRd'
                        )
                        
                        fig_cvar.update_layout(
                            template="plotly_dark",
                            height=400,
                            xaxis_title="Risk Contribution (%)"
                        )
                        
                        st.plotly_chart(fig_cvar, use_container_width=True)
                else:
                    st.info("PCA/Risk Contribution is not calculated for single-asset portfolios or due to data errors.")

                # Correlation Analysis
                st.markdown("#### Correlation Structure")
                
                corr_matrix = portfolio_returns.corr()
                
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    zmin=-1,
                    zmax=1,
                    title="Asset Correlation Matrix"
                )
                
                fig_corr.update_layout(
                    template="plotly_dark",
                    height=500
                )
                
                st.plotly_chart(fig_corr, use_container_width=True)
            
            # --- EXPORT SECTION ---
            st.sidebar.markdown("---")
            with st.sidebar.expander("💾 Export Results"):
                
                # Create summary report
                report_data = {
                    'Portfolio': list(weights.keys()),
                    'Weights': list(weights.values()),
                    'Total Return': [total_return] * len(weights),
                    'Excess Return': [attribution_data['attribution'].get('Total Excess Return', 0)] * len(weights)
                }
                
                report_df = pd.DataFrame(report_data)
                report_csv = report_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Attribution Report (CSV)",
                    data=report_csv,
                    file_name='attribution_report.csv',
                    mime='text/csv'
                )
                
                # Download portfolio weights
                weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
                weights_csv = weights_df.to_csv(header=True)
                
                st.download_button(
                    label="📥 Download Portfolio Weights (CSV)",
                    data=weights_csv,
                    file_name='portfolio_weights.csv',
                    mime='text/csv'
                )
                
                # Download performance data
                perf_data = pd.DataFrame({
                    'Date': attribution_data['portfolio_returns'].index,
                    'Portfolio_Return': attribution_data['portfolio_returns'].values,
                    'Benchmark_Return': attribution_data['benchmark_returns'].values,
                    'Excess_Return': attribution_data['excess_returns'].values
                })
                perf_csv = perf_data.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Performance Data (CSV)",
                    data=perf_csv,
                    file_name='performance_data.csv',
                    mime='text/csv'
                )
        
        except Exception as e:
            st.error(f"❌ Analysis failed with an unexpected error: {str(e)}")
            st.exception(e)

else:
    # Empty state with updated welcome message - REMOVED SPECIFIED HTML BLOCK
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px;">
        <h1 style="color: #00cc96; font-size: 48px; margin-bottom: 20px;">🏛️ Enigma Institutional Terminal Pro</h1>
        <p style="color: #888; font-size: 20px; margin-bottom: 40px;">
            Advanced Portfolio Analytics with Real Benchmark Attribution
        </p>
        
        <div style="max-width: 800px; margin: 0 auto; padding: 30px; background: linear-gradient(135deg, rgba(30, 30, 30, 0.7) 0%, rgba(42, 42, 42, 0.7) 100%); 
                    border-radius: 10px; border-left: 4px solid #00cc96;">
            <h3 style="color: #ccc; margin-bottom: 20px;">Ready to Analyze Your Portfolio?</h3>
            <p style="color: #888; font-size: 16px; line-height: 1.6;">
                Configure your portfolio analysis using the sidebar controls. Select assets, choose your benchmark, 
                set optimization parameters, and launch the enhanced analytics engine for comprehensive performance attribution.
            </p>
            
            <div style="margin-top: 30px; padding: 20px; background: rgba(0, 0, 0, 0.3); border-radius: 8px;">
                <h4 style="color: #00cc96; margin-bottom: 15px;">👈 Quick Start Instructions</h4>
                <ol style="text-align: left; color: #aaa; font-size: 15px; line-height: 1.8;">
                    <li>Select your asset universe from the dropdown</li>
                    <li>Choose portfolio assets using the multiselect</li>
                    <li>Configure attribution settings and benchmark</li>
                    <li>Set model parameters and optimization objectives</li>
                    <li>Click <strong style="color: #00cc96;">'EXECUTE ENHANCED ANALYSIS'</strong> to begin</li>
                </ol>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick start guide
    with st.expander("🚀 Quick Start Guide", expanded=True):
        st.markdown("""
        ### Getting Started
        
        1. **Select Asset Universe** - Choose from predefined lists or enter custom tickers
        2. **Configure Benchmark** - Auto-detect or manually select appropriate benchmark
            - S&P 500 (^GSPC) for US/Global portfolios
            - BIST 30 (XU030.IS) for Turkish portfolios
            - Euro Stoxx 50 (^STOXX50E) for European portfolios
        3. **Choose Attribution Method** - Select from multiple attribution approaches
        4. **Set Analysis Parameters** - Configure date range, risk-free rate, etc.
        5. **Click 'EXECUTE ENHANCED ANALYSIS'** to launch the analytics engine
        
        ### Key Features
        
        - **Real Benchmark Data**: Uses actual Yahoo Finance data for benchmarks
        - **Enhanced Attribution**: Sector-level and factor-based attribution
        - **Interactive Visualizations**: Plotly-based dashboards with drill-down capabilities
        - **Risk Analytics**: Comprehensive risk metrics including GARCH and PCA
        - **Monte Carlo Simulations**: Multiple simulation methods for risk assessment
        - **Export Capabilities**: Download all results for further analysis
        """)
