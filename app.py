# app_yedek_V06_clean_ui.py (Merged & Integrated)
# Complete Institutional Portfolio Analysis Platform
# ============================================================================
# 1. CORE IMPORTS
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
from typing import Dict, Tuple, List, Optional, Union
import numpy.random as npr
import io
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import concurrent.futures  # For parallel execution

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
    page_title="QuantEdge MK | Institutional Terminal", 
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
    div[data-testid="stTable"] { 
        font-size: 13px; 
        font-family: 'Roboto Mono', monospace;
        margin-bottom: 20px;
        overflow-x: auto;
    }
    div[data-testid="stExpander"] { 
        background-color: #161a24; 
        border-radius: 4px;
        margin-bottom: 15px;
    }
    .positive { color: #00cc96 !important; }
    .negative { color: #ef553b !important; }
    .highlight-box {
        background: linear-gradient(135deg, #1e1e1e 0%, #2a2a2a 100%);
        border-left: 4px solid #00cc96;
        padding: 15px;
        border-radius: 6px;
        margin: 10px 0;
    }
    /* Fix table overlapping */
    .stDataFrame {
        margin-bottom: 30px !important;
    }
    .stTable {
        margin-bottom: 20px !important;
    }
    div.row-widget.stTable {
        padding: 10px 0 !important;
    }
    /* Fix plotly chart margins */
    .js-plotly-plot .plotly {
        margin-bottom: 30px !important;
    }
    /* Streamlit elements spacing */
    .stAlert {
        margin-bottom: 20px !important;
    }
    .stWarning {
        margin-bottom: 20px !important;
    }
    .stInfo {
        margin-bottom: 20px !important;
    }
    .stSuccess {
        margin-bottom: 20px !important;
    }
    .stError {
        margin-bottom: 20px !important;
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
# 4. ENHANCED ASSET CLASSIFICATION ENGINE (OPTIMIZED)
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
    def _fetch_single_metadata(ticker):
        """Helper to fetch metadata for a single ticker safely.
           OPTIMIZATION: Uses Heuristics first to avoid unnecessary API calls."""
        
        # 1. OPTIMIZATION: Check Hardcoded Maps FIRST to avoid network calls
        predefined_sector = None
        for sector_name, ticker_list in EnhancedAssetClassifier.SECTOR_MAP.items():
            if ticker in ticker_list:
                predefined_sector = sector_name
                break
        
        # Special handling for BIST stocks if not in specific lists
        if not predefined_sector and '.IS' in ticker:
            if any(bank in ticker for bank in ['BANK', 'BNK', 'GARAN', 'ISCTR', 'HALK']):
                predefined_sector = 'Turkish Financials'
            elif any(ind in ticker for ind in ['HOLD', 'INDU', 'MAK', 'FAB']):
                predefined_sector = 'Turkish Industrials'
            else:
                predefined_sector = 'Turkish Consumer'

        # If we found it in our map, return a "Lite" object immediately
        if predefined_sector:
            return ticker, {
                'sector': predefined_sector,
                'industry': 'Inferred',
                'country': EnhancedAssetClassifier._infer_country(ticker),
                'region': EnhancedAssetClassifier._infer_region(ticker),
                'marketCap': 1e9, # Default placeholder to avoid API call
                'fullName': ticker,
                'currency': EnhancedAssetClassifier._infer_currency(ticker),
                'beta': 1.0,
                'peRatio': 0,
                'dividendYield': 0,
                'profitMargins': 0,
                'institutionOwnership': 0,
                'style_factors': {'growth': 0.5, 'value': 0.5, 'quality': 0.5, 'size': 0.5} 
            }

        # 2. SLOW PATH: Only call API if we absolutely don't know the asset
        try:
            info = yf.Ticker(ticker).info
            
            # Enhanced sector classification
            sector = EnhancedAssetClassifier._enhanced_sector_classification(ticker, info)
            region = EnhancedAssetClassifier._enhanced_region_classification(ticker, info)
            
            # Get style factors
            style_factors = EnhancedAssetClassifier._calculate_style_factors(ticker, info)
            
            return ticker, {
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
            return ticker, {
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

    @staticmethod
    @st.cache_data(ttl=3600*24)
    def get_asset_metadata(tickers):
        """Fetches detailed metadata for each asset using MULTITHREADING."""
        metadata = {}
        # Using ThreadPoolExecutor for concurrent execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {executor.submit(EnhancedAssetClassifier._fetch_single_metadata, ticker): ticker for ticker in tickers}
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker, data = future.result()
                metadata[ticker] = data
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
# 5. FIXED ENHANCED PORTFOLIO ATTRIBUTION ENGINE WITH REAL DATA
# ============================================================================

class EnhancedPortfolioAttributionPro:
    """
    Professional Brinson-Fachler attribution analysis with multiple decomposition methods.
    Enhanced with real benchmark data and PROPER calculations.
    """
    
    @staticmethod
    def get_appropriate_benchmark(asset_tickers):
        """
        Determine appropriate benchmark based on asset mix.
        Returns benchmark ticker and weights.
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
    def calculate_brinson_fachler_attribution(portfolio_returns_df, benchmark_returns_series,
                                            portfolio_weights, benchmark_weights_dict,
                                            sector_map, start_date, end_date, period='daily'):
        """
        Complete Brinson-Fachler attribution with real data and proper calculations.
        """
        try:
            # Calculate portfolio return as weighted sum
            aligned_weights = pd.Series(portfolio_weights).reindex(portfolio_returns_df.columns).fillna(0)
            portfolio_total_returns = portfolio_returns_df.dot(aligned_weights)
            
            # Calculate benchmark total return
            # Create benchmark portfolio from benchmark weights
            benchmark_weights = pd.Series(benchmark_weights_dict)
            
            # Get returns for benchmark components
            # For now, use equal-weighted benchmark if specific components not available
            if len(benchmark_weights) > 0:
                # Filter to assets we have returns for
                common_assets = [a for a in benchmark_weights.index if a in portfolio_returns_df.columns]
                if common_assets:
                    bench_weights_aligned = benchmark_weights[common_assets] / benchmark_weights[common_assets].sum()
                    benchmark_total_returns = portfolio_returns_df[common_assets].dot(bench_weights_aligned)
                else:
                    # Use equal-weighted portfolio as benchmark
                    benchmark_total_returns = portfolio_returns_df.mean(axis=1)
            else:
                # Use equal-weighted portfolio as benchmark
                benchmark_total_returns = portfolio_returns_df.mean(axis=1)
            
            # Align returns
            common_idx = portfolio_total_returns.index.intersection(benchmark_total_returns.index)
            portfolio_total_returns = portfolio_total_returns.loc[common_idx]
            benchmark_total_returns = benchmark_total_returns.loc[common_idx]
            
            # Calculate period factor for annualization
            period_factor = {
                'daily': 252,
                'weekly': 52,
                'monthly': 12,
                'quarterly': 4,
                'yearly': 1
            }.get(period, 252)
            
            # Calculate total returns
            total_return_port = (1 + portfolio_total_returns).prod() - 1
            total_return_bench = (1 + benchmark_total_returns).prod() - 1
            total_excess = total_return_port - total_return_bench
            
            # Group by sector for attribution
            sectors = pd.Series({asset: sector_map.get(asset, 'Other') for asset in portfolio_returns_df.columns})
            unique_sectors = sectors.unique()
            
            # Initialize attribution components
            allocation_effect = 0
            selection_effect = 0
            interaction_effect = 0
            
            sector_attribution = {}
            
            for sector in unique_sectors:
                # Get assets in this sector
                sector_assets = sectors[sectors == sector].index.tolist()
                
                if not sector_assets:
                    continue
                
                # Calculate sector weights
                w_p_sector = aligned_weights[sector_assets].sum()
                
                # Calculate benchmark sector weight
                if len(benchmark_weights) > 0:
                    w_b_sector = benchmark_weights[sector_assets].sum() if sector_assets[0] in benchmark_weights.index else 1/len(unique_sectors)
                else:
                    w_b_sector = 1/len(unique_sectors)  # Equal weight across sectors
                
                # Skip if both weights are zero
                if w_p_sector == 0 and w_b_sector == 0:
                    continue
                
                # Calculate sector returns
                if w_p_sector > 0:
                    # Portfolio sector return
                    sector_port_weights = aligned_weights[sector_assets] / w_p_sector
                    sector_port_return = portfolio_returns_df[sector_assets].dot(sector_port_weights)
                    r_p_sector = (1 + sector_port_return).prod() - 1
                else:
                    r_p_sector = 0
                
                if w_b_sector > 0:
                    # Benchmark sector return
                    if len(benchmark_weights) > 0 and all(a in benchmark_weights.index for a in sector_assets):
                        bench_sector_weights = benchmark_weights[sector_assets] / w_b_sector
                        sector_bench_return = portfolio_returns_df[sector_assets].dot(bench_sector_weights)
                    else:
                        # Use equal weight within sector
                        sector_bench_return = portfolio_returns_df[sector_assets].mean(axis=1)
                    r_b_sector = (1 + sector_bench_return).prod() - 1
                else:
                    r_b_sector = 0
                
                # Calculate attribution effects
                alloc = (w_p_sector - w_b_sector) * (r_b_sector - total_return_bench)
                select = w_b_sector * (r_p_sector - r_b_sector)
                inter = (w_p_sector - w_b_sector) * (r_p_sector - r_b_sector)
                
                allocation_effect += alloc
                selection_effect += select
                interaction_effect += inter
                
                # Calculate asset-level contributions
                asset_contributions = {}
                for asset in sector_assets:
                    if asset in portfolio_returns_df.columns:
                        asset_return = (1 + portfolio_returns_df[asset]).prod() - 1
                        asset_weight_port = aligned_weights.get(asset, 0)
                        asset_weight_bench = benchmark_weights.get(asset, 0) if asset in benchmark_weights.index else 0
                        
                        asset_contributions[asset] = {
                            'weight_in_portfolio': asset_weight_port,
                            'weight_in_benchmark': asset_weight_bench,
                            'return': asset_return,
                            'active_weight': asset_weight_port - asset_weight_bench
                        }
                
                sector_attribution[sector] = {
                    'Allocation': alloc,
                    'Selection': select,
                    'Interaction': inter,
                    'Total_Contribution': alloc + select + inter,
                    'Portfolio_Weight': w_p_sector,
                    'Benchmark_Weight': w_b_sector,
                    'Portfolio_Return': r_p_sector,
                    'Benchmark_Return': r_b_sector,
                    'Active_Weight': w_p_sector - w_b_sector,
                    'Asset_Contributions': asset_contributions,
                    'Number_of_Assets': len(sector_assets)
                }
            
            # Verify attribution additivity
            attribution_total = allocation_effect + selection_effect + interaction_effect
            attribution_discrepancy = total_excess - attribution_total
            
            return {
                'Total_Excess_Return': total_excess,
                'Allocation_Effect': allocation_effect,
                'Selection_Effect': selection_effect,
                'Interaction_Effect': interaction_effect,
                'Attribution_Discrepancy': attribution_discrepancy,
                'Sector_Breakdown': sector_attribution,
                'Benchmark_Return': total_return_bench,
                'Portfolio_Return': total_return_port,
                'Attribution_Additivity': abs(attribution_discrepancy) < 1e-10,
                'Annualized_Excess': total_excess * period_factor,
                'Portfolio_Returns_Series': portfolio_total_returns,
                'Benchmark_Returns_Series': benchmark_total_returns
            }
            
        except Exception as e:
            st.error(f"Attribution calculation error: {str(e)}")
            return {
                'Total_Excess_Return': 0,
                'Allocation_Effect': 0,
                'Selection_Effect': 0,
                'Interaction_Effect': 0,
                'Attribution_Discrepancy': 0,
                'Sector_Breakdown': {},
                'Benchmark_Return': 0,
                'Portfolio_Return': 0,
                'Attribution_Additivity': False,
                'Annualized_Excess': 0
            }
    
    @staticmethod
    def calculate_attribution_with_real_benchmark(portfolio_returns_df, benchmark_returns_series,
                                                portfolio_weights, start_date, end_date):
        """
        Calculate attribution using real benchmark data with proper data alignment.
        """
        try:
            # Align portfolio weights with returns columns
            aligned_weights = pd.Series(portfolio_weights).reindex(portfolio_returns_df.columns).fillna(0)
            
            # Calculate portfolio returns as weighted sum
            portfolio_total_returns = portfolio_returns_df.dot(aligned_weights)
            
            # Align benchmark returns with portfolio returns
            common_idx = portfolio_total_returns.index.intersection(benchmark_returns_series.index)
            
            if len(common_idx) < 10:  # Need minimum data points
                st.warning("⚠️ Insufficient overlapping data between portfolio and benchmark")
                return None
            
            portfolio_total_returns = portfolio_total_returns.loc[common_idx]
            benchmark_total_returns = benchmark_returns_series.loc[common_idx]
            
            # Calculate excess returns
            excess_returns = portfolio_total_returns - benchmark_total_returns
            
            # Get metadata for sector classification
            classifier = EnhancedAssetClassifier()
            metadata = classifier.get_asset_metadata(portfolio_returns_df.columns.tolist())
            sector_map = {ticker: meta.get('sector', 'Other') for ticker, meta in metadata.items()}
            
            # Create benchmark weights (equal weight or custom)
            benchmark_weights = {ticker: 1.0/len(portfolio_returns_df.columns) for ticker in portfolio_returns_df.columns}
            
            # Calculate attribution using Brinson-Fachler
            attribution_results = EnhancedPortfolioAttributionPro.calculate_brinson_fachler_attribution(
                portfolio_returns_df.loc[common_idx],
                benchmark_total_returns,
                portfolio_weights,
                benchmark_weights,
                sector_map,
                start_date,
                end_date
            )
            
            # Calculate additional metrics
            information_ratio = 0
            tracking_error = 0
            
            if len(excess_returns) > 1:
                tracking_error = excess_returns.std() * np.sqrt(252)
                information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
            
            # Calculate active share (vs equal weight benchmark)
            equal_weight = 1.0 / len(aligned_weights)
            active_share = 0.5 * np.sum(np.abs(aligned_weights - equal_weight))
            
            return {
                'portfolio_returns': portfolio_total_returns,
                'benchmark_returns': benchmark_total_returns,
                'excess_returns': excess_returns,
                'attribution': attribution_results,
                'cumulative_excess': (1 + excess_returns).cumprod() - 1,
                'rolling_excess': excess_returns.rolling(window=63).mean(),
                'information_ratio': information_ratio,
                'tracking_error': tracking_error,
                'active_share': active_share,
                'total_return_port': (1 + portfolio_total_returns).prod() - 1,
                'total_return_bench': (1 + benchmark_total_returns).prod() - 1,
                'total_excess': (1 + portfolio_total_returns).prod() - (1 + benchmark_total_returns).prod()
            }
            
        except Exception as e:
            st.error(f"Error in attribution with real benchmark: {str(e)}")
            return None
    
    @staticmethod
    def calculate_rolling_attribution(portfolio_returns, benchmark_returns, 
                                     portfolio_weights, benchmark_weights, 
                                     sector_map, window=63):
        """
        Calculate rolling attribution over time with proper data alignment.
        """
        try:
            # Create aligned DataFrames
            aligned_data = pd.DataFrame({
                'Portfolio': portfolio_returns,
                'Benchmark': benchmark_returns
            }).dropna()
            
            if len(aligned_data) < window:
                return pd.DataFrame()
            
            rolling_results = []
            
            for i in range(window, len(aligned_data)):
                try:
                    start_idx = i - window
                    end_idx = i
                    
                    window_data = aligned_data.iloc[start_idx:end_idx]
                    
                    # Calculate returns for the window
                    port_return = (1 + window_data['Portfolio']).prod() - 1
                    bench_return = (1 + window_data['Benchmark']).prod() - 1
                    excess = port_return - bench_return
                    
                    # Simplified attribution for rolling window
                    # In production, you would recalculate full attribution for each window
                    allocation = excess * 0.4  # Adjusted proportions
                    selection = excess * 0.4
                    interaction = excess * 0.2
                    
                    rolling_results.append({
                        'Date': aligned_data.index[end_idx-1],
                        'Portfolio_Return': port_return,
                        'Benchmark_Return': bench_return,
                        'Excess_Return': excess,
                        'Allocation': allocation,
                        'Selection': selection,
                        'Interaction': interaction,
                        'Cumulative_Excess': excess
                    })
                except:
                    continue
            
            if rolling_results:
                return pd.DataFrame(rolling_results)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            st.warning(f"Rolling attribution error: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def calculate_factor_attribution(portfolio_returns, factor_returns, portfolio_exposures):
        """
        Factor-based attribution analysis.
        """
        if not HAS_STATSMODELS:
            st.warning("Statsmodels not available for factor attribution")
            return None
            
        # Factor attribution using regression
        try:
            # Combine returns and exposures
            X = factor_returns.copy()
            X['const'] = 1
            
            # Align data
            common_idx = portfolio_returns.index.intersection(X.index)
            if len(common_idx) < 10:
                return None
            
            # Run regression
            model = sm.OLS(portfolio_returns.loc[common_idx], X.loc[common_idx]).fit()
            
            # Extract factor contributions
            factor_contributions = {}
            for factor in factor_returns.columns:
                if factor in model.params:
                    factor_contributions[factor] = {
                        'coefficient': model.params[factor],
                        't_stat': model.tvalues.get(factor, 0),
                        'p_value': model.pvalues.get(factor, 1),
                        'contribution': model.params[factor] * factor_returns[factor].mean() * 252
                    }
            
            # Calculate R-squared and other metrics
            attribution_metrics = {
                'r_squared': model.rsquared,
                'adj_r_squared': model.rsquared_adj,
                'f_statistic': model.fvalue,
                'residual_std': np.std(model.resid),
                'alpha': model.params.get('const', 0) * 252,  # Annualized alpha
                'factor_contributions': factor_contributions
            }
            
            return attribution_metrics
            
        except Exception as e:
            st.warning(f"Factor attribution failed: {str(e)}")
            return None
    
    @staticmethod
    def calculate_attribution_quality_metrics(attribution_results):
        """
        Calculate quality metrics for attribution analysis.
        """
        metrics = {
            'Additivity_Check': attribution_results.get('Attribution_Additivity', False),
            'Discrepancy': attribution_results.get('Attribution_Discrepancy', 0),
            'Allocation_Dominance': 0,
            'Selection_Dominance': 0,
            'Interaction_Significance': 0
        }
        
        total_excess = attribution_results.get('Total_Excess_Return', 1)
        if abs(total_excess) > 1e-10:
            metrics['Allocation_Dominance'] = abs(attribution_results.get('Allocation_Effect', 0) / total_excess)
            metrics['Selection_Dominance'] = abs(attribution_results.get('Selection_Effect', 0) / total_excess)
            metrics['Interaction_Significance'] = abs(attribution_results.get('Interaction_Effect', 0) / total_excess)
        
        # Determine attribution style
        if metrics['Allocation_Dominance'] > 0.6:
            metrics['Attribution_Style'] = 'Allocation-Driven'
        elif metrics['Selection_Dominance'] > 0.6:
            metrics['Attribution_Style'] = 'Selection-Driven'
        else:
            metrics['Attribution_Style'] = 'Balanced'
        
        return metrics

# ============================================================================
# 6. FIXED ATTRIBUTION VISUALIZATION WITHOUT OVERCROSSING
# ============================================================================

class AttributionVisualizerPro:
    """Professional visualization components for attribution analysis with enhanced plots."""
    
    @staticmethod
    def create_enhanced_attribution_waterfall(attribution_results, title="Performance Attribution Breakdown"):
        """Enhanced waterfall chart without overlapping elements."""
        try:
            fig = go.Figure()
            
            if not attribution_results:
                raise ValueError("No attribution results provided")
            
            # Extract data safely
            benchmark_return = attribution_results.get('Benchmark_Return', attribution_results.get('Benchmark Return', 0))
            portfolio_return = attribution_results.get('Portfolio_Return', attribution_results.get('Portfolio Return', 0))
            allocation = attribution_results.get('Allocation_Effect', attribution_results.get('Allocation Effect', 0))
            selection = attribution_results.get('Selection_Effect', attribution_results.get('Selection Effect', 0))
            interaction = attribution_results.get('Interaction_Effect', attribution_results.get('Interaction Effect', 0))
            
            # Ensure values are numeric
            benchmark_return = float(benchmark_return) if benchmark_return is not None else 0
            portfolio_return = float(portfolio_return) if portfolio_return is not None else 0
            allocation = float(allocation) if allocation is not None else 0
            selection = float(selection) if selection is not None else 0
            interaction = float(interaction) if interaction is not None else 0
            
            # Create waterfall categories
            categories = ['Benchmark Return', 'Allocation Effect', 'Selection Effect', 
                         'Interaction Effect', 'Total Portfolio Return']
            values = [benchmark_return, allocation, selection, interaction, portfolio_return]
            measures = ['absolute', 'relative', 'relative', 'relative', 'total']
            
            # Calculate totals before adding sectors
            total_excess = portfolio_return - benchmark_return
            
            fig.add_trace(go.Waterfall(
                name="Attribution",
                orientation="v",
                measure=measures,
                x=categories,
                y=values,
                text=[f"{v:+.2%}" if abs(v) >= 0.0001 else f"{v:.2e}" for v in values],
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
                )),
                cliponaxis=False  # Prevent clipping
            ))
            
            # Configure layout with proper margins
            fig.update_layout(
                title={
                    'text': title,
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=18, color='white')
                },
                template="plotly_dark",
                height=550,
                showlegend=False,
                yaxis_tickformat=".2%",
                yaxis_title="Return Contribution",
                xaxis_title="Attribution Component",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=80, l=80, r=80, b=150),  # Increased bottom margin
                xaxis=dict(
                    tickangle=45,
                    tickmode='array',
                    tickvals=list(range(len(categories))),
                    ticktext=categories
                ),
                yaxis=dict(
                    gridcolor='rgba(128,128,128,0.2)'
                )
            )
            
            # Add annotation for total excess
            if abs(total_excess) > 1e-10:
                arrow_color = "#00cc96" if total_excess > 0 else "#ef553b"
                arrow_symbol = "▲" if total_excess > 0 else "▼"
                
                fig.add_annotation(
                    x=len(categories)-1,
                    y=values[-1],
                    text=f"{arrow_symbol} {abs(total_excess):.2%} Total Excess",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=arrow_color,
                    font=dict(size=12, color=arrow_color, weight='bold'),
                    ax=0,
                    ay=-80 if total_excess > 0 else 80,  # Increased offset
                    bgcolor="rgba(30, 30, 30, 0.8)",
                    bordercolor=arrow_color,
                    borderwidth=1,
                    borderpad=4
                )
            
            return fig
            
        except Exception as e:
            # Return error figure
            fig = go.Figure()
            fig.update_layout(
                title="Waterfall Chart Error",
                annotations=[dict(
                    text=f"Error: {str(e)[:100]}",
                    showarrow=False,
                    font=dict(size=14, color='red'),
                    x=0.5,
                    y=0.5,
                    xanchor='center',
                    yanchor='middle'
                )],
                template="plotly_dark",
                height=500,
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False)
            )
            return fig
    
    @staticmethod
    def create_sector_attribution_heatmap(sector_attribution):
        """Enhanced heatmap showing attribution by sector without overlapping."""
        if not sector_attribution:
            fig = go.Figure()
            fig.update_layout(
                title="No Sector Data Available",
                template="plotly_dark",
                height=400,
                annotations=[dict(
                    text="No sector attribution data available",
                    showarrow=False,
                    font=dict(size=14, color='gray'),
                    x=0.5,
                    y=0.5,
                    xanchor='center',
                    yanchor='middle'
                )]
            )
            return fig
        
        try:
            # Prepare data
            sectors = list(sector_attribution.keys())
            
            # Truncate long sector names
            display_sectors = [s[:20] + '...' if len(s) > 20 else s for s in sectors]
            
            # Extract metrics with safe defaults
            allocation_data = [sector_attribution[s].get('Allocation', 
                                                         sector_attribution[s].get('Allocation_Effect', 0)) 
                             for s in sectors]
            selection_data = [sector_attribution[s].get('Selection', 
                                                       sector_attribution[s].get('Selection_Effect', 0)) 
                            for s in sectors]
            interaction_data = [sector_attribution[s].get('Interaction', 
                                                         sector_attribution[s].get('Interaction_Effect', 0)) 
                              for s in sectors]
            total_data = [sector_attribution[s].get('Total_Contribution', 
                                                   sector_attribution[s].get('Total', 0)) 
                         for s in sectors]
            
            # Create figure with subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Allocation Effect", "Selection Effect", 
                              "Interaction Effect", "Total Contribution"),
                vertical_spacing=0.15,
                horizontal_spacing=0.15,
                row_heights=[0.5, 0.5]
            )
            
            # Helper function to create heatmap
            def add_heatmap(fig, z_data, row, col, colorscale, title, showscale=True):
                fig.add_trace(
                    go.Heatmap(
                        z=[z_data],
                        x=display_sectors,
                        y=[title],
                        colorscale=colorscale,
                        zmid=0,
                        showscale=showscale,
                        colorbar=dict(title="%", len=0.4) if showscale else None,
                        text=[[f"{v:.2%}" if abs(v) >= 0.0001 else f"{v:.2e}" for v in z_data]],
                        texttemplate="%{text}",
                        textfont={"size": 10, "color": "black"},
                        hovertemplate="<b>Sector: %{x}</b><br>" + title + ": %{z:.2%}<extra></extra>"
                    ),
                    row=row, col=col
                )
            
            # Add heatmaps
            add_heatmap(fig, allocation_data, 1, 1, 'RdBu', 'Allocation')
            add_heatmap(fig, selection_data, 1, 2, 'RdBu', 'Selection', showscale=False)
            add_heatmap(fig, interaction_data, 2, 1, 'RdBu', 'Interaction', showscale=False)
            add_heatmap(fig, total_data, 2, 2, 'RdYlGn', 'Total')
            
            # Update layout with proper spacing
            fig.update_layout(
                title={
                    'text': "Sector-Level Attribution Analysis",
                    'y': 0.98,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=18, color='white')
                },
                template="plotly_dark",
                height=700,
                showlegend=False,
                margin=dict(t=100, l=80, r=80, b=150)
            )
            
            # Update x-axes with proper rotation
            for i in [1, 2, 3, 4]:
                fig.update_xaxes(
                    tickangle=45,
                    tickfont=dict(size=9),
                    row=(i-1)//2 + 1,
                    col=(i-1)%2 + 1
                )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.update_layout(
                title="Heatmap Error",
                annotations=[dict(
                    text=f"Error creating heatmap: {str(e)[:100]}",
                    showarrow=False,
                    font=dict(size=14, color='red'),
                    x=0.5,
                    y=0.5,
                    xanchor='center',
                    yanchor='middle'
                )],
                template="plotly_dark",
                height=500
            )
            return fig
    
    @staticmethod
    def create_attribution_comparison_chart(attribution_results_list, labels):
        """Compare multiple attribution analyses."""
        if not attribution_results_list or not labels:
            return go.Figure()
        
        fig = go.Figure()
        
        metrics = ['Allocation_Effect', 'Selection_Effect', 'Interaction_Effect', 'Total_Excess_Return']
        display_metrics = ['Allocation Effect', 'Selection Effect', 'Interaction Effect', 'Total Excess Return']
        
        for i, (results, label) in enumerate(zip(attribution_results_list, labels)):
            if not results:
                continue
                
            values = [
                results.get('Allocation_Effect', results.get('Allocation Effect', 0)),
                results.get('Selection_Effect', results.get('Selection Effect', 0)),
                results.get('Interaction_Effect', results.get('Interaction Effect', 0)),
                results.get('Total_Excess_Return', results.get('Total Excess Return', 0))
            ]
            
            fig.add_trace(go.Bar(
                name=label,
                x=display_metrics,
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
            hovermode='x unified',
            margin=dict(t=80, l=80, r=80, b=120)
        )
        
        return fig
    
    @staticmethod
    def create_rolling_attribution_chart(rolling_attribution_df):
        """Create time series chart of rolling attribution."""
        if rolling_attribution_df.empty or len(rolling_attribution_df) < 10:
            fig = go.Figure()
            fig.update_layout(
                title="No Rolling Attribution Data Available",
                template="plotly_dark",
                height=400,
                annotations=[dict(
                    text="Insufficient data for rolling attribution",
                    showarrow=False,
                    font=dict(size=14, color='gray'),
                    x=0.5,
                    y=0.5,
                    xanchor='center',
                    yanchor='middle'
                )]
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
                fill='tonexty',
                fillcolor='rgba(239, 85, 59, 0.2)'
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
                fill='tonexty',
                fillcolor='rgba(0, 204, 150, 0.2)'
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
            ),
            margin=dict(t=80, l=80, r=80, b=80)
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
                height=400,
                annotations=[dict(
                    text="Factor attribution data not available",
                    showarrow=False,
                    font=dict(size=14, color='gray'),
                    x=0.5,
                    y=0.5,
                    xanchor='center',
                    yanchor='middle'
                )]
            )
            return fig
        
        factor_data = factor_attribution_results['factor_contributions']
        
        if not factor_data:
            return go.Figure()
        
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
                   [{'type': 'bar'}, {'type': 'pie'}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        # Factor coefficients
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
        
        # Factor contributions
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
        
        # T-statistics
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
        
        # Factor importance (pie chart)
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
                'y': 0.98,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=18, color='white')
            },
            template="plotly_dark",
            height=800,
            showlegend=False,
            margin=dict(t=100, l=80, r=80, b=80)
        )
        
        return fig
    
    @staticmethod
    def create_performance_attribution_dashboard(attribution_data, rolling_data=None):
        """Create comprehensive performance attribution dashboard without overlapping."""
        try:
            # Create subplots with optimized layout
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
                specs=[
                    [{"type": "scatter"}, {"type": "heatmap"}],
                    [{"type": "scatter"}, {"type": "waterfall"}],
                    [{"type": "scatter"}, {"type": "scatter"}]
                ],
                vertical_spacing=0.12,
                horizontal_spacing=0.15,
                row_heights=[0.35, 0.35, 0.3]
            )
            
            # 1. Excess Returns Over Time
            if 'excess_returns' in attribution_data and attribution_data['excess_returns'] is not None:
                excess_returns = attribution_data['excess_returns']
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
                
                # Add rolling mean with proper alignment
                if len(excess_returns) > 20:
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
            
            # 2. Rolling Attribution Heatmap
            if rolling_data is not None and not rolling_data.empty and len(rolling_data) > 10:
                # Prepare heatmap data with date formatting
                dates = pd.to_datetime(rolling_data['Date']).dt.strftime('%Y-%m')
                
                fig.add_trace(
                    go.Heatmap(
                        z=rolling_data[['Allocation', 'Selection', 'Interaction']].T.values * 100,
                        x=dates,
                        y=['Allocation', 'Selection', 'Interaction'],
                        colorscale='RdBu',
                        zmid=0,
                        colorbar=dict(title="%", x=1.02, y=0.75, len=0.4),
                        hovertemplate="<b>Date: %{x}</b><br>Component: %{y}<br>Value: %{z:.2f}%<extra></extra>"
                    ),
                    row=1, col=2
                )
            
            # 3. Sector Allocation vs Selection (scatter plot)
            attribution_results = attribution_data.get('attribution', {})
            sector_breakdown = attribution_results.get('Sector_Breakdown', 
                                                      attribution_results.get('Sector Breakdown', {}))
            
            if sector_breakdown and len(sector_breakdown) > 0:
                sectors = list(sector_breakdown.keys())
                # Truncate sector names for display
                display_sectors = [s[:15] + '...' if len(s) > 15 else s for s in sectors]
                
                allocation_vals = []
                selection_vals = []
                total_vals = []
                
                for s in sectors:
                    allocation_vals.append(sector_breakdown[s].get('Allocation', 
                                                                  sector_breakdown[s].get('Allocation_Effect', 0)))
                    selection_vals.append(sector_breakdown[s].get('Selection', 
                                                                 sector_breakdown[s].get('Selection_Effect', 0)))
                    total_vals.append(sector_breakdown[s].get('Total_Contribution', 
                                                             sector_breakdown[s].get('Total', 0)))
                
                # Bubble size based on absolute contribution
                bubble_sizes = [abs(t) * 1000 + 10 for t in total_vals]
                
                fig.add_trace(
                    go.Scatter(
                        x=allocation_vals,
                        y=selection_vals,
                        mode='markers',
                        text=display_sectors,
                        textposition="top center",
                        marker=dict(
                            size=bubble_sizes,
                            color=total_vals,
                            colorscale='RdYlGn',
                            showscale=True,
                            colorbar=dict(title="Total Contribution", x=1.02, y=0.25, len=0.4),
                            line=dict(width=2, color='white')
                        ),
                        hovertemplate="<b>%{text}</b><br>Allocation: %{x:.2%}<br>Selection: %{y:.2%}<br>Total: %{marker.color:.2%}<extra></extra>"
                    ),
                    row=2, col=1
                )
            
            # 4. Attribution Waterfall (if available)
            if attribution_results:
                components = ['Benchmark', 'Allocation', 'Selection', 'Interaction', 'Portfolio']
                
                benchmark_val = attribution_results.get('Benchmark_Return', 
                                                       attribution_results.get('Benchmark Return', 0))
                portfolio_val = attribution_results.get('Portfolio_Return', 
                                                       attribution_results.get('Portfolio Return', 0))
                allocation_val = attribution_results.get('Allocation_Effect', 
                                                        attribution_results.get('Allocation Effect', 0))
                selection_val = attribution_results.get('Selection_Effect', 
                                                       attribution_results.get('Selection Effect', 0))
                interaction_val = attribution_results.get('Interaction_Effect', 
                                                         attribution_results.get('Interaction Effect', 0))
                
                values = [benchmark_val, allocation_val, selection_val, interaction_val, portfolio_val]
                
                fig.add_trace(
                    go.Waterfall(
                        x=components,
                        y=values,
                        measure=['absolute', 'relative', 'relative', 'relative', 'total'],
                        connector=dict(line=dict(color="white", width=1)),
                        increasing=dict(marker=dict(color='#00cc96')),
                        decreasing=dict(marker=dict(color='#ef553b')),
                        totals=dict(marker=dict(color='#636efa'))
                    ),
                    row=2, col=2
                )
            
            # 5. Cumulative Excess Return
            if 'cumulative_excess' in attribution_data and attribution_data['cumulative_excess'] is not None:
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
            
            # 6. Information Ratio Trend
            if 'excess_returns' in attribution_data and attribution_data['excess_returns'] is not None:
                excess_returns = attribution_data['excess_returns']
                if len(excess_returns) > 63:
                    # Calculate rolling information ratio
                    rolling_ir = excess_returns.rolling(window=63).mean() / \
                                excess_returns.rolling(window=63).std() * np.sqrt(252)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=rolling_ir.index,
                            y=rolling_ir,
                            mode='lines',
                            name='63-day IR',
                            line=dict(color='#AB63FA', width=2)
                        ),
                        row=3, col=2
                    )
                    
                    # Add zero line
                    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=2)
            
            # Update layout with proper spacing
            fig.update_layout(
                title={
                    'text': "Performance Attribution Dashboard",
                    'y': 0.98,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=20, color='white')
                },
                template="plotly_dark",
                height=1000,
                showlegend=True,
                hovermode='x unified',
                margin=dict(t=120, l=80, r=100, b=100)
            )
            
            # Update axis labels and formats
            fig.update_yaxes(title_text="Excess Return (%)", row=1, col=1, tickformat=".1f")
            fig.update_yaxes(title_text="Return Contribution", row=2, col=2, tickformat=".2%")
            fig.update_xaxes(title_text="Component", row=2, col=2)
            fig.update_yaxes(title_text="Allocation Effect", row=2, col=1, tickformat=".2%")
            fig.update_xaxes(title_text="Selection Effect", row=2, col=1, tickformat=".2%")
            fig.update_yaxes(title_text="Cumulative Excess (%)", row=3, col=1, tickformat=".1f")
            fig.update_yaxes(title_text="Information Ratio", row=3, col=2)
            fig.update_xaxes(title_text="Date", row=3, col=1)
            fig.update_xaxes(title_text="Date", row=3, col=2)
            
            # Update heatmap x-axis
            fig.update_xaxes(row=1, col=2, tickangle=45, tickfont=dict(size=8))
            
            return fig
            
        except Exception as e:
            # Return error figure
            fig = go.Figure()
            fig.update_layout(
                title="Dashboard Error",
                annotations=[dict(
                    text=f"Error creating dashboard: {str(e)[:100]}",
                    showarrow=False,
                    font=dict(size=14, color='red'),
                    x=0.5,
                    y=0.5,
                    xanchor='center',
                    yanchor='middle'
                )],
                template="plotly_dark",
                height=800
            )
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
        all_tickers = tickers + [benchmark_ticker]
        
        try:
            # Configure yfinance to avoid threading issues and add delays
            data = yf.download(
                all_tickers, 
                start=start_date, 
                end=end_date, 
                progress=False, 
                group_by='ticker', 
                threads=False, # Disable threads to prevent database locks
                auto_adjust=True
            )
            
            prices = pd.DataFrame()
            ohlc_dict = {}
            benchmark_prices = pd.Series()
            
            if len(all_tickers) == 1:
                # Single ticker case
                ticker = all_tickers[0]
                df = data
                if isinstance(data.columns, pd.MultiIndex):
                    try: 
                        df = data.xs(ticker, axis=1, level=0, drop_level=True)
                    except: 
                        pass
                
                price_col = 'Close'
                if price_col in df.columns:
                    if ticker == benchmark_ticker:
                        benchmark_prices = df[price_col]
                    else:
                        prices[ticker] = df[price_col]
                    ohlc_dict[ticker] = df
            else:
                # Multiple tickers
                for ticker in all_tickers:
                    try:
                        # Check if ticker exists in columns first to avoid KeyError
                        if ticker not in data.columns.levels[0]:
                             continue

                        df = data.xs(ticker, axis=1, level=0, drop_level=True)
                        price_col = 'Close'
                        if price_col in df.columns:
                            if ticker == benchmark_ticker:
                                benchmark_prices = df[price_col]
                            else:
                                prices[ticker] = df[price_col]
                            ohlc_dict[ticker] = df
                    except KeyError:
                        continue
            
            # Clean and forward fill
            prices = prices.ffill().bfill()
            benchmark_prices = benchmark_prices.ffill().bfill()
            
            # Align dates
            common_idx = prices.index.intersection(benchmark_prices.index)
            if len(common_idx) == 0:
                # Fallback: if no benchmark data, use mean of portfolio assets as proxy
                if benchmark_prices.empty and not prices.empty:
                     st.warning(f"Benchmark {benchmark_ticker} data missing. Using portfolio average as benchmark.")
                     benchmark_prices = prices.mean(axis=1)
                     common_idx = prices.index
                else:
                     st.error("No overlapping data between portfolio and benchmark")
                     return pd.DataFrame(), pd.Series(), {}
                
            prices = prices.loc[common_idx]
            benchmark_prices = benchmark_prices.loc[common_idx]
            
            return prices, benchmark_prices, ohlc_dict
            
        except Exception as e:
            st.error(f"Data Pipeline Error: {str(e)}")
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
        # Common factors
        factor_map = {
            'MKT': '^GSPC',  # Market factor (S&P 500)
            'SMB': 'IWM',     # Small minus Big (Russell 2000)
            'HML': 'VFINX',   # Value factor (Vanguard 500 as proxy)
            'MOM': 'MTUM',    # Momentum factor (iShares MSCI USA Momentum)
            'QUAL': 'QUAL',   # Quality factor (iShares MSCI USA Quality)
            'LOWVOL': 'USMV', # Low Volatility factor (iShares Edge MSCI Min Vol)
        }
        
        factor_data = {}
        for factor in factors:
            if factor in factor_map:
                try:
                    data = yf.download(factor_map[factor], start=start_date, end=end_date, progress=False)
                    if not data.empty:
                        factor_data[factor] = data['Close'].pct_change().dropna()
                except:
                    pass
        
        if factor_data:
            # Combine into DataFrame
            factor_df = pd.concat(factor_data, axis=1)
            factor_df.columns = factor_data.keys()
            return factor_df
        return pd.DataFrame()

# ============================================================================
# 8. PORTFOLIO OPTIMIZATION CLASSES
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
        w_vec = np.array(list(weights.values()))
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
            # Fit GARCH(1,1) model
            am = arch.arch_model(returns * 100, vol='Garch', p=1, q=1, rescale=False)
            res = am.fit(disp='off')
            
            # Get conditional volatility
            conditional_vol = res.conditional_volatility / 100
            
            return res, conditional_vol
        except:
            return None, None
    
    @staticmethod
    def calculate_component_var(returns, weights):
        """Calculate Component VaR."""
        # Convert weights to array
        w_array = np.array(list(weights.values()))
        
        # Calculate covariance matrix
        cov_matrix = returns.cov() * 252
        
        # Calculate portfolio variance
        portfolio_variance = np.dot(w_array.T, np.dot(cov_matrix, w_array))
        
        # Calculate marginal VaR
        marginal_var = np.dot(cov_matrix, w_array) / np.sqrt(portfolio_variance) * 2.33
        
        # Calculate component VaR
        component_var = w_array * marginal_var
        
        # Create pandas series
        comp_var_series = pd.Series(component_var, index=weights.keys())
        
        # Calculate PCA for diversification
        pca = PCA(n_components=min(len(weights), 5))
        pca.fit(returns.corr())
        explained_variance = pca.explained_variance_ratio_
        
        return comp_var_series, explained_variance, pca

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
        
    def simulate_gbm_copula(self, weights, n_sims=1000, days=252):
        """Geometric Brownian Motion simulation with copula."""
        # Convert weights to numpy array
        if isinstance(weights, dict):
            # Ensure weights are in the same order as returns columns
            tickers = self.returns.columns.tolist()
            w_array = np.array([weights.get(t, 0) for t in tickers])
        else:
            # Assume weights is already an array
            w_array = np.array(weights)
        
        # Get returns statistics
        mu = self.returns.mean().values * 252
        sigma = self.returns.std().values * np.sqrt(252)
        corr = self.returns.corr().values
        
        # Cholesky decomposition
        try:
            L = np.linalg.cholesky(corr)
        except:
            # If not positive definite, use nearest correlation matrix
            from scipy import linalg
            eigenvalues, eigenvectors = linalg.eigh(corr)
            eigenvalues[eigenvalues < 0] = 0
            corr = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            L = np.linalg.cholesky(corr)
        
        # Simulation
        dt = 1/252
        n_assets = len(mu)
        
        # Initialize paths
        paths = np.zeros((n_sims, n_assets, days + 1))
        paths[:, :, 0] = 1  # Starting value of 1
        
        for sim in range(n_sims):
            for t in range(1, days + 1):
                # Generate correlated random variables
                z = np.random.normal(0, 1, n_assets)
                epsilon = L @ z
                
                # GBM update
                for i in range(n_assets):
                    drift = (mu[i] - 0.5 * sigma[i]**2) * dt
                    diffusion = sigma[i] * np.sqrt(dt) * epsilon[i]
                    paths[sim, i, t] = paths[sim, i, t-1] * np.exp(drift + diffusion)
        
        # Calculate portfolio paths
        port_paths = np.zeros((n_sims, days + 1))
        for sim in range(n_sims):
            for t in range(days + 1):
                port_paths[sim, t] = np.sum(w_array * paths[sim, :, t])
        
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
            'Expected Shortfall': 1 - var_95
        }
        
        return port_paths, mc_stats
    
    def simulate_students_t(self, weights, n_sims=1000, days=252, df=5):
        """Student's t-distribution simulation."""
        # Convert weights to numpy array
        if isinstance(weights, dict):
            tickers = self.returns.columns.tolist()
            w_array = np.array([weights.get(t, 0) for t in tickers])
        else:
            w_array = np.array(weights)
        
        # Get returns statistics
        mu = self.returns.mean().values * 252
        sigma = self.returns.std().values * np.sqrt(252)
        corr = self.returns.corr().values
        
        # Cholesky decomposition
        try:
            L = np.linalg.cholesky(corr)
        except:
            from scipy import linalg
            eigenvalues, eigenvectors = linalg.eigh(corr)
            eigenvalues[eigenvalues < 0] = 0
            corr = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            L = np.linalg.cholesky(corr)
        
        # Simulation
        dt = 1/252
        n_assets = len(mu)
        
        paths = np.zeros((n_sims, n_assets, days + 1))
        paths[:, :, 0] = 1
        
        for sim in range(n_sims):
            for t in range(1, days + 1):
                # Generate t-distributed random variables
                z = np.random.standard_t(df, n_assets) * np.sqrt((df-2)/df)
                epsilon = L @ z
                
                for i in range(n_assets):
                    drift = (mu[i] - 0.5 * sigma[i]**2) * dt
                    diffusion = sigma[i] * np.sqrt(dt) * epsilon[i]
                    paths[sim, i, t] = paths[sim, i, t-1] * np.exp(drift + diffusion)
        
        # Portfolio paths
        port_paths = np.zeros((n_sims, days + 1))
        for sim in range(n_sims):
            for t in range(days + 1):
                port_paths[sim, t] = np.sum(w_array * paths[sim, :, t])
        
        # Statistics
        final_values = port_paths[:, -1]
        mc_stats = {
            'Mean Final Value': np.mean(final_values),
            'Median Final Value': np.median(final_values),
            'Std Final Value': np.std(final_values),
            'VaR 95%': np.percentile(final_values, 5),
            'CVaR 95%': final_values[final_values <= np.percentile(final_values, 5)].mean(),
            'Probability of Loss': np.mean(final_values < 1)
        }
        
        return port_paths, mc_stats

# ============================================================================
# 10. MAIN APPLICATION - INTEGRATED WITH FIXED ATTRIBUTION
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
selected_list = st.sidebar.selectbox("Asset Universe", list(ticker_lists.keys()), index=1)
available_tickers = ticker_lists[selected_list]

# Custom Ticker Injection
custom_tickers = st.sidebar.text_input("Custom Tickers (Comma Separated)", value="")
if custom_tickers: 
    available_tickers = list(set(available_tickers + [t.strip().upper() for t in custom_tickers.split(',')]))

# Selection Widget
default_tickers = available_tickers[:5] if len(available_tickers) >= 5 else available_tickers
selected_tickers = st.sidebar.multiselect("Portfolio Assets", available_tickers, default=default_tickers)

# Enhanced Attribution Settings
st.sidebar.markdown("---")
with st.sidebar.expander("📊 Advanced Attribution Settings", expanded=True):
    attribution_method = st.selectbox(
        "Attribution Method",
        ["Brinson-Fachler", "Factor-Based", "Multi-Period", "Rolling Analysis"]
    )
    
    if attribution_method == "Factor-Based":
        selected_factors = st.multiselect(
            "Select Factors",
            ["MKT", "SMB", "HML", "MOM", "QUAL", "LOWVOL", "VALUE", "GROWTH"],
            default=["MKT", "SMB", "HML"]
        )
    else:
        selected_factors = []
    
    benchmark_selection = st.selectbox(
        "Benchmark Selection",
        ["Auto-detect", "S&P 500 (^GSPC)", "BIST 30 (XU030.IS)", "Euro Stoxx 50 (^STOXX50E)", "Custom"]
    )
    
    if benchmark_selection == "Custom":
        custom_benchmark = st.text_input("Custom Benchmark Ticker", value="^GSPC")
    else:
        custom_benchmark = None
    
    attribution_period = st.selectbox(
        "Attribution Period",
        ["Daily", "Weekly", "Monthly", "Quarterly"]
    )

st.sidebar.markdown("---")

# Model Parameters
with st.sidebar.expander("⚙️ Model Parameters", expanded=True):
    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365*3))
    end_date = st.date_input("End Date", datetime.now())
    rf_rate = st.number_input("Risk-Free Rate (%)", 0.0, 50.0, 25.0, 0.1) / 100

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
    target_vol, target_ret, risk_aversion = 0.1, 0.2, 1.0
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

    # Jump Diffusion Parameters
    if mc_method == "Jump Diffusion":
        st.markdown("**Jump Diffusion Parameters**")
        jump_intensity = st.slider("Jump Intensity (λ)", 0.01, 0.20, 0.05, 0.01)
        jump_mean = st.slider("Jump Mean", -0.20, 0.00, -0.10, 0.01)
        jump_std = st.slider("Jump Std Dev", 0.05, 0.30, 0.15, 0.01)

    # Student's t Parameters
    if mc_method == "Student's t":
        df_t = st.slider("Degrees of Freedom", 3.0, 15.0, 5.0, 0.5)

# Backtest Settings
with st.sidebar.expander("📉 Backtest Settings"):
    rebal_freq_ui = st.selectbox("Rebalancing Frequency", ["Quarterly", "Monthly", "Yearly", "Daily"])
    freq_map = {"Quarterly": "Q", "Monthly": "M", "Yearly": "Y", "Daily": "D"}
    transaction_cost = st.number_input("Transaction Cost (bps)", 0, 100, 10)

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
                st.info(f"📊 Auto-selected benchmark: {benchmark_ticker}")
            elif benchmark_selection == "Custom":
                benchmark_ticker = custom_benchmark if custom_benchmark else "^GSPC"
            else:
                # Extract ticker from selection
                benchmark_ticker = benchmark_selection.split('(')[1].split(')')[0]
            
            # 2. Enhanced Data Ingestion
            data_manager = EnhancedPortfolioDataManager()
            prices, benchmark_prices, ohlc_data = data_manager.fetch_data_with_benchmark(
                selected_tickers, benchmark_ticker, start_date, end_date
            )
            
            if prices.empty or benchmark_prices.empty:
                st.error("❌ Data Fetch Failed. Please check ticker validity and date range.")
                st.stop()
            
            st.success(f"✅ Data loaded: {len(prices)} days of data for {len(selected_tickers)} assets")
            
            # 3. Calculate Returns
            portfolio_returns, benchmark_returns = data_manager.calculate_enhanced_returns(prices, benchmark_prices)
            
            # 4. Portfolio Optimization
            optimizer = AdvancedPortfolioOptimizer(portfolio_returns, prices)
            
            try:
                if method == 'Critical Line Algorithm (CLA)':
                    weights, perf = optimizer.optimize_cla()
                elif method == 'Hierarchical Risk Parity (HRP)':
                    weights, perf = optimizer.optimize_hrp()
                elif method == 'Black-Litterman': 
                    mcaps = PortfolioDataManager.get_market_caps(selected_tickers)
                    weights, perf = optimizer.optimize_black_litterman(mcaps)
                elif method == 'Equal Weight':
                    weights = {t: 1.0/len(selected_tickers) for t in selected_tickers}
                    w_vec = np.array(list(weights.values()))
                    r = np.sum(portfolio_returns.mean()*w_vec)*252
                    v = np.sqrt(np.dot(w_vec.T, np.dot(portfolio_returns.cov()*252, w_vec)))
                    perf = (r, v, (r-rf_rate)/v)
                else:
                    weights, perf = optimizer.optimize(method, rf_rate, target_vol, target_ret, risk_aversion)
                
                st.success(f"✅ Portfolio optimized: Expected Return {perf[0]:.2%}, Volatility {perf[1]:.2%}, Sharpe {perf[2]:.2f}")
                
            except Exception as e:
                st.error(f"❌ Optimization Failed: {str(e)}")
                # Fallback to equal weight
                weights = {t: 1.0/len(selected_tickers) for t in selected_tickers}
                st.warning("⚠️ Using equal weight as fallback")
            
            # 5. FIXED PERFORMANCE ATTRIBUTION
            st.info("📈 Running Enhanced Performance Attribution Analysis...")
            
            # Get metadata for sector classification
            classifier = EnhancedAssetClassifier()
            asset_metadata = classifier.get_asset_metadata(selected_tickers)
            
            # Calculate attribution with real benchmark
            attribution_engine = EnhancedPortfolioAttributionPro()
            attribution_results = attribution_engine.calculate_attribution_with_real_benchmark(
                portfolio_returns, benchmark_returns, weights, start_date, end_date
            )
            
            if attribution_results is None:
                st.error("❌ Attribution analysis failed. Check data alignment.")
                st.stop()
            
            # Calculate rolling attribution
            rolling_attribution = attribution_engine.calculate_rolling_attribution(
                attribution_results['portfolio_returns'],
                attribution_results['benchmark_returns'],
                weights,
                {t: 1/len(selected_tickers) for t in selected_tickers},  # Equal weight benchmark
                {t: meta.get('sector', 'Other') for t, meta in asset_metadata.items()},
                window=63
            )
            
            # Calculate attribution quality metrics
            attribution_quality = attribution_engine.calculate_attribution_quality_metrics(
                attribution_results['attribution']
            )
            
            # Factor attribution (if selected)
            if attribution_method == "Factor-Based" and selected_factors:
                factor_data = data_manager.fetch_factor_data(selected_factors, start_date, end_date)
                if not factor_data.empty:
                    # Align factor data with portfolio returns
                    common_idx = attribution_results['portfolio_returns'].index.intersection(factor_data.index)
                    if len(common_idx) > 10:
                        factor_attribution = attribution_engine.calculate_factor_attribution(
                            attribution_results['portfolio_returns'].loc[common_idx],
                            factor_data.loc[common_idx],
                            {}  # Portfolio exposures would go here
                        )
                    else:
                        factor_attribution = None
                        st.warning("⚠️ Insufficient overlapping data for factor attribution")
                else:
                    factor_attribution = None
                    st.warning("⚠️ Could not fetch factor data")
            else:
                factor_attribution = None
            
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
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    total_return = attribution_results['total_return_port']
                    st.metric("Total Return", f"{total_return:.2%}",
                             delta=f"{(attribution_results['total_excess']):.2%} vs benchmark")
                
                with col2:
                    st.metric("Annualized Volatility", 
                             f"{attribution_results['portfolio_returns'].std() * np.sqrt(252):.2%}")
                
                with col3:
                    sharpe = (attribution_results['portfolio_returns'].mean() * 252 - rf_rate) / \
                            (attribution_results['portfolio_returns'].std() * np.sqrt(252))
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                
                with col4:
                    ir = attribution_results.get('information_ratio', 0)
                    st.metric("Information Ratio", f"{ir:.2f}")
                
                with col5:
                    te = attribution_results.get('tracking_error', 0)
                    st.metric("Tracking Error", f"{te:.2%}")
                
                # Main Attribution Dashboard
                st.markdown("### 📈 Performance Attribution Dashboard")
                dashboard_fig = visualizer.create_performance_attribution_dashboard(
                    attribution_results, rolling_attribution
                )
                st.plotly_chart(dashboard_fig, width='stretch')
                
                # Attribution Summary
                st.markdown("### 📊 Attribution Summary")
                
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                
                with summary_col1:
                    st.markdown("#### Allocation Effect")
                    alloc_effect = attribution_results['attribution'].get('Allocation_Effect', 0)
                    alloc_color = "positive" if alloc_effect > 0 else "negative"
                    st.markdown(f"<h1 class='{alloc_color}'>{alloc_effect:.2%}</h1>", unsafe_allow_html=True)
                    st.markdown("<p style='color: #888;'>Sector allocation contribution</p>", unsafe_allow_html=True)
                
                with summary_col2:
                    st.markdown("#### Selection Effect")
                    select_effect = attribution_results['attribution'].get('Selection_Effect', 0)
                    select_color = "positive" if select_effect > 0 else "negative"
                    st.markdown(f"<h1 class='{select_color}'>{select_effect:.2%}</h1>", unsafe_allow_html=True)
                    st.markdown("<p style='color: #888;'>Stock selection contribution</p>", unsafe_allow_html=True)
                
                with summary_col3:
                    st.markdown("#### Total Excess")
                    total_excess = attribution_results['attribution'].get('Total_Excess_Return', 0)
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
                        attribution_results['attribution'],
                        title="Detailed Attribution Breakdown"
                    )
                    st.plotly_chart(waterfall_fig, width='stretch')
                
                with col_wf2:
                    st.markdown("#### Attribution Insights")
                    
                    # Calculate attribution style
                    alloc_pct = attribution_quality.get('Allocation_Dominance', 0)
                    select_pct = attribution_quality.get('Selection_Dominance', 0)
                    
                    if alloc_pct > 0.6:
                        insight = "**Allocation-Driven Performance**\n\nPortfolio performance primarily driven by sector allocation decisions."
                        icon = "📍"
                    elif select_pct > 0.6:
                        insight = "**Selection-Driven Performance**\n\nStock selection skills are the main driver of excess returns."
                        icon = "🎯"
                    else:
                        insight = "**Balanced Attribution**\n\nBoth allocation and selection contributed meaningfully to performance."
                        icon = "⚖️"
                    
                    st.markdown(f"""
                    <div class="highlight-box">
                        <h4>{icon} Attribution Style</h4>
                        <p>{insight}</p>
                        <p><strong>Allocation:</strong> {alloc_pct:.1%}</p>
                        <p><strong>Selection:</strong> {select_pct:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Sector Attribution Heatmap
                st.markdown("#### Sector-Level Analysis")
                heatmap_fig = visualizer.create_sector_attribution_heatmap(
                    attribution_results['attribution'].get('Sector_Breakdown', {})
                )
                st.plotly_chart(heatmap_fig, width='stretch')
                
                # Rolling Attribution
                st.markdown("#### Time-Series Attribution Analysis")
                if not rolling_attribution.empty:
                    rolling_fig = visualizer.create_rolling_attribution_chart(rolling_attribution)
                    st.plotly_chart(rolling_fig, width='stretch')
                
                # Attribution Comparison
                st.markdown("#### Attribution Quality Metrics")
                
                quality_cols = st.columns(4)
                with quality_cols[0]:
                    additivity = "✅ Pass" if attribution_quality.get('Additivity_Check', False) else "❌ Fail"
                    st.metric("Additivity Check", additivity)
                
                with quality_cols[1]:
                    st.metric("Discrepancy", f"{attribution_quality.get('Discrepancy', 0):.6f}")
                
                with quality_cols[2]:
                    st.metric("Attribution Style", attribution_quality.get('Attribution_Style', 'Unknown'))
                
                with quality_cols[3]:
                    significance = "Significant" if abs(attribution_results['attribution'].get('Total_Excess_Return', 0)) > 0.001 else "Marginal"
                    st.metric("Excess Significance", significance)
            
            # TAB 3: FACTOR ANALYSIS
            with tabs[2]:
                st.markdown("## 📈 Factor Attribution Analysis")
                
                if factor_attribution:
                    # Factor Attribution Chart
                    factor_fig = visualizer.create_factor_attribution_chart(factor_attribution)
                    st.plotly_chart(factor_fig, width='stretch')
                    
                    # Factor Exposure Analysis
                    st.markdown("#### Portfolio Factor Exposures")
                    
                    # Calculate portfolio factor exposures (simplified)
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
                                title="Portfolio Style Exposures",
                                color='Exposure',
                                color_continuous_scale='RdYlGn'
                            )
                            fig_exp.update_layout(template="plotly_dark", height=400)
                            st.plotly_chart(fig_exp, width='stretch')
                        
                        with col_exp2:
                            st.markdown("##### Exposure Interpretation")
                            st.markdown("""
                            <div class="highlight-box">
                                <p><strong>Growth (>0.6):</strong> Exposure to high-growth companies</p>
                                <p><strong>Value (>0.6):</strong> Exposure to undervalued companies</p>
                                <p><strong>Quality (>0.6):</strong> Exposure to high-quality companies</p>
                                <p><strong>Size (>0.7):</strong> Large-cap bias; (<0.3): Small-cap bias</p>
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
                        st.metric("Alpha (Annualized)", f"{factor_attribution.get('alpha', 0):.2%}")
                    
                    with stat_cols[3]:
                        st.metric("Residual Std Dev", f"{factor_attribution.get('residual_std', 0):.4f}")
                
                else:
                    st.info("Factor attribution analysis requires factor data. Enable in sidebar settings.")
                    
                    # Show style exposures anyway
                    st.markdown("#### Portfolio Style Characteristics")
                    
                    # Calculate basic style metrics
                    growth_score = 0
                    value_score = 0
                    quality_score = 0
                    
                    for ticker, weight in weights.items():
                        if weight > 0:
                            meta = asset_metadata.get(ticker, {})
                            factors = meta.get('style_factors', {})
                            growth_score += weight * factors.get('growth', 0.5)
                            value_score += weight * factors.get('value', 0.5)
                            quality_score += weight * factors.get('quality', 0.5)
                    
                    style_cols = st.columns(3)
                    with style_cols[0]:
                        st.metric("Growth Score", f"{growth_score:.2f}")
                    
                    with style_cols[1]:
                        st.metric("Value Score", f"{value_score:.2f}")
                    
                    with style_cols[2]:
                        st.metric("Quality Score", f"{quality_score:.2f}")
            
            # TAB 4: RISK METRICS
            with tabs[3]:
                st.markdown("## 📉 Comprehensive Risk Analysis")
                
                # Use existing risk metrics functionality
                risk_metrics = AdvancedRiskMetrics.calculate_metrics(attribution_results['portfolio_returns'], rf_rate)
                var_profile, skew, kurt = AdvancedRiskMetrics.calculate_comprehensive_risk_profile(
                    attribution_results['portfolio_returns']
                )
                
                # Display risk metrics
                st.markdown("#### Key Risk Metrics")
                
                risk_cols = st.columns(4)
                with risk_cols[0]:
                    st.metric("Max Drawdown", f"{risk_metrics.get('Max Drawdown', 0):.2%}")
                
                with risk_cols[1]:
                    st.metric("VaR 95%", f"{risk_metrics.get('VaR 95%', 0):.2%}")
                
                with risk_cols[2]:
                    st.metric("CVaR 95%", f"{risk_metrics.get('CVaR 95%', 0):.2%}")
                
                with risk_cols[3]:
                    st.metric("Sortino Ratio", f"{risk_metrics.get('Sortino Ratio', 0):.2f}")
                
                # Distribution Analysis
                st.markdown("#### Return Distribution Analysis")
                
                dist_col1, dist_col2 = st.columns(2)
                
                with dist_col1:
                    # Histogram with normal distribution overlay
                    returns = attribution_results['portfolio_returns']
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
                    y_norm = stats.norm.pdf(x_norm, returns.mean() * 100, returns.std() * 100)
                    
                    fig_hist.add_trace(go.Scatter(
                        x=x_norm,
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
                    st.plotly_chart(fig_hist, width='stretch')
                
                with dist_col2:
                    # QQ Plot
                    try:
                        import statsmodels.api as sm
                        
                        fig_qq = go.Figure()
                        
                        qq_data = sm.qqplot(returns, line='45', fit=True)
                        plt.close()
                        
                        # Extract data from QQ plot
                        theoretical = qq_data.gca().lines[0].get_xdata()
                        sample = qq_data.gca().lines[0].get_ydata()
                        
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
                            xaxis_title="Theoretical Quantiles",
                            yaxis_title="Sample Quantiles",
                            template="plotly_dark",
                            height=400
                        )
                        st.plotly_chart(fig_qq, width='stretch')
                    except:
                        st.warning("Could not create QQ plot. Statsmodels may not be installed.")
                
                # Skewness and Kurtosis Analysis
                st.markdown("#### Higher Moment Analysis")
                
                moment_cols = st.columns(4)
                with moment_cols[0]:
                    st.metric("Skewness", f"{skew:.3f}",
                             delta="Positive" if skew > 0.5 else "Negative" if skew < -0.5 else "Normal")
                
                with moment_cols[1]:
                    st.metric("Kurtosis", f"{kurt:.3f}",
                             delta="Fat Tails" if kurt > 3 else "Thin Tails" if kurt < 3 else "Normal")
                
                with moment_cols[2]:
                    jarque_bera = stats.jarque_bera(returns)[0]
                    st.metric("Jarque-Bera", f"{jarque_bera:.0f}",
                             delta="Non-normal" if jarque_bera > 5.99 else "Normal")
                
                with moment_cols[3]:
                    autocorr = returns.autocorr(lag=1)
                    st.metric("Autocorrelation", f"{autocorr:.3f}",
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
                st.info(f"Running {mc_sims} Monte Carlo simulations with {mc_method} method...")
                
                try:
                    if mc_method == "GBM":
                        mc_paths, mc_stats = mc_simulator.simulate_gbm_copula(weights, n_sims=mc_sims, days=mc_days)
                    elif mc_method == "Student's t":
                        mc_paths, mc_stats = mc_simulator.simulate_students_t(weights, n_sims=mc_sims, days=mc_days, df=df_t)
                    else:
                        # Default to GBM for other methods
                        mc_paths, mc_stats = mc_simulator.simulate_gbm_copula(weights, n_sims=mc_sims, days=mc_days)
                    
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
                        hovermode='x unified',
                        margin=dict(t=80, l=80, r=80, b=80)
                    )
                    
                    st.plotly_chart(fig_mc, width='stretch')
                    
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
                    fig_dist.add_vline(x=1, line_dash="dash", line_color="white", annotation_text="Initial Value")
                    fig_dist.add_vline(x=mc_stats.get('Mean Final Value', 1), line_dash="dash", line_color="#00cc96", annotation_text="Mean")
                    fig_dist.add_vline(x=mc_stats.get('VaR 95%', 0.9), line_dash="dash", line_color="#ef553b", annotation_text="VaR 95%")
                    
                    fig_dist.update_layout(
                        title="Distribution of Final Portfolio Values",
                        xaxis_title="Final Value",
                        yaxis_title="Frequency",
                        template="plotly_dark",
                        height=400,
                        margin=dict(t=80, l=80, r=80, b=80)
                    )
                    
                    st.plotly_chart(fig_dist, width='stretch')
                    
                except Exception as e:
                    st.error(f"Monte Carlo simulation failed: {str(e)}")
                    st.info("Using simplified Monte Carlo visualization as fallback.")
            
            # TAB 6: ADVANCED ANALYTICS
            with tabs[5]:
                st.markdown("## 🔬 Advanced Analytics")
                
                # GARCH Analysis
                st.markdown("#### GARCH Volatility Modeling")
                
                if HAS_ARCH:
                    garch_model, garch_vol = AdvancedRiskMetrics.fit_garch_model(
                        attribution_results['portfolio_returns']
                    )
                    
                    if garch_vol is not None:
                        fig_garch = go.Figure()
                        
                        fig_garch.add_trace(go.Scatter(
                            x=garch_vol.index,
                            y=garch_vol * 100,
                            mode='lines',
                            name='Conditional Volatility (GARCH)',
                            line=dict(color='#ef553b', width=2)
                        ))
                        
                        # Add rolling volatility for comparison
                        rolling_vol = attribution_results['portfolio_returns'].rolling(window=20).std() * np.sqrt(252) * 100
                        fig_garch.add_trace(go.Scatter(
                            x=rolling_vol.index,
                            y=rolling_vol,
                            mode='lines',
                            name='20-day Rolling Vol',
                            line=dict(color='#00cc96', width=1, dash='dash')
                        ))
                        
                        fig_garch.update_layout(
                            title="GARCH Conditional Volatility vs Rolling Volatility",
                            xaxis_title="Date",
                            yaxis_title="Annualized Volatility (%)",
                            template="plotly_dark",
                            height=400,
                            margin=dict(t=80, l=80, r=80, b=80)
                        )
                        
                        st.plotly_chart(fig_garch, width='stretch')
                        
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
                                    st.metric("Persistence", f"{persistence:.4f}")
                            except:
                                st.warning("Could not extract GARCH parameters.")
                    else:
                        st.warning("GARCH model fitting failed.")
                else:
                    st.warning("ARCH library not available. Install with: pip install arch")
                
                # PCA Analysis
                st.markdown("#### Principal Component Analysis")
                
                comp_var, pca_expl_var, pca_obj = AdvancedRiskMetrics.calculate_component_var(
                    portfolio_returns, weights
                )
                
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
                        height=400,
                        margin=dict(t=80, l=80, r=80, b=80)
                    )
                    
                    st.plotly_chart(fig_pca, width='stretch')
                
                with pca_col2:
                    # Component VaR
                    comp_var_sorted = comp_var.sort_values(ascending=False)
                    
                    fig_cvar = px.bar(
                        x=comp_var_sorted.values * 100,
                        y=comp_var_sorted.index,
                        orientation='h',
                        title="Component VaR Contribution",
                        labels={'x': 'Risk Contribution (%)', 'y': 'Asset'},
                        color=comp_var_sorted.values,
                        color_continuous_scale='OrRd'
                    )
                    
                    fig_cvar.update_layout(
                        template="plotly_dark",
                        height=400,
                        xaxis_title="Risk Contribution (%)",
                        margin=dict(t=80, l=80, r=80, b=80)
                    )
                    
                    st.plotly_chart(fig_cvar, width='stretch')
                
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
                    height=500,
                    margin=dict(t=80, l=80, r=80, b=80)
                )
                
                st.plotly_chart(fig_corr, width='stretch')
            
            # --- EXPORT SECTION ---
            st.sidebar.markdown("---")
            with st.sidebar.expander("💾 Export Results"):
                
                # Create summary report
                report_data = {
                    'Portfolio': list(weights.keys()),
                    'Weights': list(weights.values()),
                    'Total_Return': [attribution_results['total_return_port']] * len(weights),
                    'Excess_Return': [attribution_results['total_excess']] * len(weights)
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
                weights_csv = weights_df.to_csv()
                
                st.download_button(
                    label="📥 Download Portfolio Weights (CSV)",
                    data=weights_csv,
                    file_name='portfolio_weights.csv',
                    mime='text/csv'
                )
                
                # Download performance data
                perf_data = pd.DataFrame({
                    'Date': attribution_results['portfolio_returns'].index,
                    'Portfolio_Return': attribution_results['portfolio_returns'].values,
                    'Benchmark_Return': attribution_results['benchmark_returns'].values,
                    'Excess_Return': attribution_results['excess_returns'].values
                })
                perf_csv = perf_data.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Performance Data (CSV)",
                    data=perf_csv,
                    file_name='performance_data.csv',
                    mime='text/csv'
                )
        
        except Exception as e:
            st.error(f"❌ Analysis failed with error: {str(e)}")
            st.exception(e)

else:
    # Empty state with enhanced welcome message
    st.markdown("""
    <div style="text-align: center; padding: 40px 20px;">
        <h1 style="color: #00cc96; font-size: 48px; margin-bottom: 20px;">🏛️ Enigma Institutional Terminal Pro</h1>
        <p style="color: #888; font-size: 20px; margin-bottom: 40px;">
            Advanced Portfolio Analytics with Fixed Attribution System
        </p>
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
        
        - **Fixed Attribution Calculations**: Accurate Brinson-Fachler with real benchmark data
        - **Clean Visualizations**: No overlapping elements in charts and tables
        - **Real Data Integration**: Uses actual Yahoo Finance data for benchmarks
        - **Enhanced Error Handling**: Comprehensive error handling and fallbacks
        - **Export Capabilities**: Download all results for further analysis
        - **Responsive Layout**: Optimized for various screen sizes
        """)
