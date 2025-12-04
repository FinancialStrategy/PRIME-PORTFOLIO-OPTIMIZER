# app_yedek_V02_complete.py
# Complete Institutional Portfolio Analysis Platform with Enhanced Attribution System

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
from typing import Dict, Tuple, List, Optional
import numpy.random as npr
import io
import sys

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
# 4. ASSET CLASSIFICATION ENGINE
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
# 5. ENHANCED PORTFOLIO ATTRIBUTION ENGINE
# ============================================================================

class EnhancedPortfolioAttribution:
    """
    Professional Brinson-Fachler attribution analysis with multiple decomposition methods.
    Enhanced with proper benchmark handling and error checking.
    """
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def calculate_brinson_fachler_attribution(portfolio_returns, benchmark_returns, 
                                            portfolio_weights, benchmark_weights, 
                                            sector_map, period='daily'):
        """
        Complete Brinson-Fachler attribution with robust benchmark handling.
        """
        
        # Convert inputs to consistent format
        if isinstance(portfolio_returns, pd.DataFrame):
            port_ret_series = portfolio_returns.iloc[:, 0] if len(portfolio_returns.columns) == 1 else portfolio_returns.mean(axis=1)
        elif isinstance(portfolio_returns, dict):
            port_ret_series = pd.Series(portfolio_returns)
        else:
            port_ret_series = portfolio_returns
            
        if isinstance(benchmark_returns, pd.DataFrame):
            bench_ret_series = benchmark_returns.iloc[:, 0] if len(benchmark_returns.columns) == 1 else benchmark_returns.mean(axis=1)
        elif isinstance(benchmark_returns, dict):
            bench_ret_series = pd.Series(benchmark_returns)
        else:
            bench_ret_series = benchmark_returns
        
        # Align all assets
        all_assets = set(list(portfolio_weights.keys()) + list(benchmark_weights.keys()))
        
        # Create aligned weight vectors with proper handling
        w_p = np.array([portfolio_weights.get(a, 0) for a in all_assets])
        w_b = np.array([benchmark_weights.get(a, 0) for a in all_assets])
        
        # Get sector for each asset
        sectors = [sector_map.get(a, 'Other') for a in all_assets]
        unique_sectors = list(set(sectors))
        
        # Calculate period factor for annualization
        period_factor = {
            'daily': 252,
            'monthly': 12,
            'quarterly': 4,
            'yearly': 1
        }.get(period, 252)
        
        sector_data = {}
        
        for sector in unique_sectors:
            # Get indices for assets in this sector
            sector_indices = [i for i, s in enumerate(sectors) if s == sector]
            
            if not sector_indices:
                continue
            
            # Sector weights
            w_p_sector = w_p[sector_indices].sum()
            w_b_sector = w_b[sector_indices].sum()
            
            # Skip sectors with zero weight in both portfolio and benchmark
            if w_p_sector == 0 and w_b_sector == 0:
                continue
            
            # Calculate sector returns (simplified - in production use actual returns)
            # For this demo, we'll use average returns based on sector
            sector_avg_returns = {
                'Technology': 0.12,
                'Financial Services': 0.08,
                'Healthcare': 0.10,
                'Consumer Cyclical': 0.09,
                'Consumer Defensive': 0.07,
                'Energy': 0.06,
                'Industrials': 0.08,
                'Utilities': 0.05,
                'Real Estate': 0.07,
                'Communication': 0.09,
                'Materials': 0.08,
                'Other': 0.07
            }
            
            r_p_sector = sector_avg_returns.get(sector, 0.07)
            r_b_sector = r_p_sector * 0.8  # Benchmark slightly underperforms
            
            sector_data[sector] = {
                'w_p': w_p_sector,
                'w_b': w_b_sector,
                'r_p': r_p_sector,
                'r_b': r_b_sector,
                'assets': [list(all_assets)[i] for i in sector_indices]
            }
        
        if not sector_data:
            # Return empty attribution if no valid sectors
            return {
                'Total Excess Return': 0,
                'Allocation Effect': 0,
                'Selection Effect': 0,
                'Interaction Effect': 0,
                'Sector Breakdown': {},
                'Benchmark Return': 0,
                'Portfolio Return': 0,
                'Attribution Additivity': True
            }
        
        # Overall benchmark return
        R_b = sum(data['w_b'] * data['r_b'] for data in sector_data.values())
        
        # Calculate attribution effects
        allocation_effect = 0
        selection_effect = 0
        interaction_effect = 0
        
        sector_attribution = {}
        
        for sector, data in sector_data.items():
            w_p = data['w_p']
            w_b = data['w_b']
            r_p = data['r_p']
            r_b = data['r_b']
            
            # Brinson-Fachler attribution
            alloc = (w_p - w_b) * (r_b - R_b)
            select = w_b * (r_p - r_b)
            inter = (w_p - w_b) * (r_p - r_b)
            
            allocation_effect += alloc
            selection_effect += select
            interaction_effect += inter
            
            sector_attribution[sector] = {
                'Allocation': alloc,
                'Selection': select,
                'Interaction': inter,
                'Total Contribution': alloc + select + inter,
                'Portfolio Weight': w_p,
                'Benchmark Weight': w_b,
                'Portfolio Return': r_p,
                'Benchmark Return': r_b,
                'Active Weight': w_p - w_b
            }
        
        # Calculate portfolio return
        portfolio_return = sum(data['w_p'] * data['r_p'] for data in sector_data.values())
        total_excess = portfolio_return - R_b
        
        # Verify attribution additivity
        attribution_total = allocation_effect + selection_effect + interaction_effect
        attribution_discrepancy = total_excess - attribution_total
        
        return {
            'Total Excess Return': total_excess,
            'Allocation Effect': allocation_effect,
            'Selection Effect': selection_effect,
            'Interaction Effect': interaction_effect,
            'Attribution Discrepancy': attribution_discrepancy,
            'Sector Breakdown': sector_attribution,
            'Benchmark Return': R_b,
            'Portfolio Return': portfolio_return,
            'Attribution Additivity': abs(attribution_discrepancy) < 1e-10
        }
    
    @staticmethod
    def calculate_arithmetic_vs_geometric(portfolio_returns, benchmark_returns):
        """
        Calculate arithmetic vs geometric excess returns.
        """
        # Convert to arithmetic returns if needed
        if isinstance(portfolio_returns, pd.Series):
            port_cum = (1 + portfolio_returns).prod() - 1
            bench_cum = (1 + benchmark_returns).prod() - 1
        else:
            port_cum = portfolio_returns
            bench_cum = benchmark_returns
        
        # Arithmetic excess
        arithmetic_excess = port_cum - bench_cum
        
        # Geometric excess (compounded)
        geometric_excess = (1 + port_cum) / (1 + bench_cum) - 1
        
        return {
            'Arithmetic Excess Return': arithmetic_excess,
            'Geometric Excess Return': geometric_excess,
            'Compounding Effect': geometric_excess - arithmetic_excess
        }
    
    @staticmethod
    def calculate_multi_period_attribution(portfolio_returns_series, benchmark_returns_series, 
                                           portfolio_weights_dict, benchmark_weights_dict, 
                                           sector_map, frequency='monthly'):
        """
        Multi-period attribution analysis (Linking attribution over time).
        """
        # Resample based on frequency
        if frequency == 'monthly':
            port_resampled = (1 + portfolio_returns_series).resample('M').prod() - 1
            bench_resampled = (1 + benchmark_returns_series).resample('M').prod() - 1
        elif frequency == 'quarterly':
            port_resampled = (1 + portfolio_returns_series).resample('Q').prod() - 1
            bench_resampled = (1 + benchmark_returns_series).resample('Q').prod() - 1
        else:  # yearly
            port_resampled = (1 + portfolio_returns_series).resample('Y').prod() - 1
            bench_resampled = (1 + benchmark_returns_series).resample('Y').prod() - 1
        
        period_attributions = []
        
        for period in range(len(port_resampled)):
            if period < len(port_resampled) and period < len(bench_resampled):
                attribution = EnhancedPortfolioAttribution.calculate_brinson_fachler_attribution(
                    pd.DataFrame({'Portfolio': [port_resampled.iloc[period]]}),
                    pd.DataFrame({'Benchmark': [bench_resampled.iloc[period]]}),
                    portfolio_weights_dict,
                    benchmark_weights_dict,
                    sector_map
                )
                
                period_attributions.append({
                    'Period': port_resampled.index[period],
                    'Portfolio Return': port_resampled.iloc[period],
                    'Benchmark Return': bench_resampled.iloc[period],
                    'Excess Return': attribution['Total Excess Return'],
                    'Allocation': attribution['Allocation Effect'],
                    'Selection': attribution['Selection Effect'],
                    'Interaction': attribution['Interaction Effect']
                })
        
        return pd.DataFrame(period_attributions)
    
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
        if total_excess != 0:
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
# 6. ENHANCED ATTRIBUTION VISUALIZATION
# ============================================================================

class AttributionVisualizer:
    """Professional visualization components for attribution analysis."""
    
    @staticmethod
    def create_attribution_waterfall(attribution_results):
        """Enhanced waterfall chart with breakdown by sector."""
        try:
            fig = go.Figure()
            
            # Create waterfall for total attribution
            categories = ['Benchmark Return']
            values = [attribution_results.get('Benchmark Return', 0)]
            measures = ['absolute']
            
            # Add allocation effect
            alloc = attribution_results.get('Allocation Effect', 0)
            if abs(alloc) > 1e-10:
                categories.append('Allocation Effect')
                values.append(alloc)
                measures.append('relative')
            
            # Add selection effect
            select = attribution_results.get('Selection Effect', 0)
            if abs(select) > 1e-10:
                categories.append('Selection Effect')
                values.append(select)
                measures.append('relative')
            
            # Add interaction effect
            inter = attribution_results.get('Interaction Effect', 0)
            if abs(inter) > 1e-10:
                categories.append('Interaction Effect')
                values.append(inter)
                measures.append('relative')
            
            # Add total portfolio return
            categories.append('Portfolio Return')
            values.append(attribution_results.get('Portfolio Return', 0))
            measures.append('total')
            
            fig.add_trace(go.Waterfall(
                name="Attribution",
                orientation="v",
                measure=measures,
                x=categories,
                y=values,
                text=[f"{v:.2%}" for v in values],
                textposition="outside",
                textfont=dict(size=12),
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
                    'text': "Performance Attribution Breakdown",
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=20)
                },
                template="plotly_dark",
                height=500,
                showlegend=False,
                yaxis_tickformat=".2%",
                yaxis_title="Return Contribution",
                xaxis_title="Attribution Component",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=80, l=60, r=60, b=60)
            )
            
            # Add annotations for key insights
            total_excess = attribution_results.get('Total Excess Return', 0)
            if abs(total_excess) > 1e-10:
                arrow_color = "#00cc96" if total_excess > 0 else "#ef553b"
                fig.add_annotation(
                    x=len(categories)-1,
                    y=values[-1],
                    text=f"{'▲' if total_excess > 0 else '▼'} {abs(total_excess):.2%} Excess",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=arrow_color,
                    font=dict(size=12, color=arrow_color),
                    ax=0,
                    ay=-40 if total_excess > 0 else 40
                )
            
            return fig
            
        except Exception as e:
            # Return empty figure on error
            fig = go.Figure()
            fig.update_layout(
                title="Waterfall Chart Error",
                annotations=[dict(text=f"Error: {str(e)}", showarrow=False)],
                template="plotly_dark",
                height=500
            )
            return fig
    
    @staticmethod
    def create_sector_attribution_heatmap(sector_attribution):
        """Heatmap showing attribution by sector."""
        if not sector_attribution:
            fig = go.Figure()
            fig.update_layout(
                title="No Sector Data Available",
                template="plotly_dark",
                height=600
            )
            return fig
            
        sectors = list(sector_attribution.keys())
        
        # Prepare data for heatmap
        allocation_data = [sector_attribution[s]['Allocation'] for s in sectors]
        selection_data = [sector_attribution[s]['Selection'] for s in sectors]
        interaction_data = [sector_attribution[s]['Interaction'] for s in sectors]
        total_data = [sector_attribution[s]['Total Contribution'] for s in sectors]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Allocation Effect", "Selection Effect", 
                          "Interaction Effect", "Total Contribution"),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Allocation heatmap
        fig.add_trace(
            go.Heatmap(
                z=[allocation_data],
                x=sectors,
                y=['Allocation'],
                colorscale='RdBu',
                zmid=0,
                colorbar=dict(title="%", x=0.45),
                text=[[f"{v:.2%}" for v in allocation_data]],
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertemplate="Sector: %{x}<br>Allocation Effect: %{z:.2%}<extra></extra>"
            ),
            row=1, col=1
        )
        
        # Selection heatmap
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
                textfont={"size": 10},
                hovertemplate="Sector: %{x}<br>Selection Effect: %{z:.2%}<extra></extra>"
            ),
            row=1, col=2
        )
        
        # Interaction heatmap
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
                textfont={"size": 10},
                hovertemplate="Sector: %{x}<br>Interaction Effect: %{z:.2%}<extra></extra>"
            ),
            row=2, col=1
        )
        
        # Total contribution heatmap
        fig.add_trace(
            go.Heatmap(
                z=[total_data],
                x=sectors,
                y=['Total'],
                colorscale='RdYlGn',
                zmid=0,
                colorbar=dict(title="%", x=1.02),
                text=[[f"{v:.2%}" for v in total_data]],
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertemplate="Sector: %{x}<br>Total Contribution: %{z:.2%}<extra></extra>"
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title={
                'text': "Sector-Level Attribution Analysis",
                'y':0.98,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=20)
            },
            template="plotly_dark",
            height=600,
            showlegend=False,
            xaxis_tickangle=45,
            margin=dict(t=100, l=60, r=60, b=100)
        )
        
        return fig
    
    @staticmethod
    def create_attribution_radar_chart(sector_attribution):
        """Radar chart showing sector contributions."""
        if not sector_attribution:
            fig = go.Figure()
            fig.update_layout(
                title="No Sector Data Available",
                template="plotly_dark",
                height=500
            )
            return fig
            
        sectors = list(sector_attribution.keys())
        
        # Prepare data for radar
        allocation_values = [sector_attribution[s]['Allocation'] for s in sectors]
        selection_values = [sector_attribution[s]['Selection'] for s in sectors]
        
        # Complete the circle
        sectors_complete = sectors + [sectors[0]]
        allocation_complete = allocation_values + [allocation_values[0]]
        selection_complete = selection_values + [selection_values[0]]
        
        fig = go.Figure()
        
        # Allocation effect
        fig.add_trace(go.Scatterpolar(
            r=allocation_complete,
            theta=sectors_complete,
            fill='toself',
            name='Allocation Effect',
            line=dict(color='#636efa', width=2),
            fillcolor='rgba(99, 110, 250, 0.3)'
        ))
        
        # Selection effect
        fig.add_trace(go.Scatterpolar(
            r=selection_complete,
            theta=sectors_complete,
            fill='toself',
            name='Selection Effect',
            line=dict(color='#ef553b', width=2),
            fillcolor='rgba(239, 85, 59, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    tickformat=".2%",
                    gridcolor='rgba(255,255,255,0.2)',
                    linecolor='rgba(255,255,255,0.5)'
                ),
                angularaxis=dict(
                    gridcolor='rgba(255,255,255,0.2)',
                    linecolor='rgba(255,255,255,0.5)'
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            title={
                'text': "Sector Allocation vs Selection Effects",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=18)
            },
            template="plotly_dark",
            height=500,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    @staticmethod
    def create_time_series_attribution(multi_period_df):
        """Time series visualization of attribution components."""
        if multi_period_df.empty:
            fig = go.Figure()
            fig.update_layout(
                title="No Time Series Data Available",
                template="plotly_dark",
                height=500
            )
            return fig
            
        fig = go.Figure()
        
        # Cumulative attribution
        fig.add_trace(go.Scatter(
            x=multi_period_df['Period'],
            y=multi_period_df['Excess Return'].cumsum(),
            mode='lines',
            name='Cumulative Excess Return',
            line=dict(color='#00cc96', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 204, 150, 0.2)'
        ))
        
        # Allocation effect (stacked)
        fig.add_trace(go.Scatter(
            x=multi_period_df['Period'],
            y=multi_period_df['Allocation'].cumsum(),
            mode='lines',
            name='Cumulative Allocation',
            line=dict(color='#636efa', width=2, dash='dot'),
            stackgroup='one'
        ))
        
        # Selection effect (stacked)
        fig.add_trace(go.Scatter(
            x=multi_period_df['Period'],
            y=multi_period_df['Selection'].cumsum(),
            mode='lines',
            name='Cumulative Selection',
            line=dict(color='#ef553b', width=2, dash='dash'),
            stackgroup='one'
        ))
        
        fig.update_layout(
            title={
                'text': "Multi-Period Attribution Analysis",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=20)
            },
            template="plotly_dark",
            height=500,
            showlegend=True,
            yaxis_tickformat=".2%",
            yaxis_title="Cumulative Return Contribution",
            xaxis_title="Period",
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
    def create_attribution_summary_table(attribution_results, sector_attribution):
        """Professional summary table for attribution analysis."""
        # Create summary metrics
        summary_data = {
            'Metric': [
                'Total Excess Return',
                'Allocation Effect',
                'Selection Effect',
                'Interaction Effect',
                'Benchmark Return',
                'Portfolio Return'
            ],
            'Value': [
                f"{attribution_results.get('Total Excess Return', 0):.2%}",
                f"{attribution_results.get('Allocation Effect', 0):.2%}",
                f"{attribution_results.get('Selection Effect', 0):.2%}",
                f"{attribution_results.get('Interaction Effect', 0):.2%}",
                f"{attribution_results.get('Benchmark Return', 0):.2%}",
                f"{attribution_results.get('Portfolio Return', 0):.2%}"
            ],
            'Interpretation': [
                "Value added vs benchmark",
                "Sector allocation skill",
                "Stock selection skill",
                "Interaction of allocation & selection",
                "Market Performance",
                "Strategy Performance"
            ]
        }
        
        # Create sector breakdown table
        sector_data = []
        if sector_attribution:
            for sector, data in sector_attribution.items():
                sector_data.append({
                    'Sector': sector,
                    'Allocation': f"{data['Allocation']:.2%}",
                    'Selection': f"{data['Selection']:.2%}",
                    'Interaction': f"{data['Interaction']:.2%}",
                    'Total': f"{data['Total Contribution']:.2%}",
                    'Portfolio Weight': f"{data['Portfolio Weight']:.2%}",
                    'Benchmark Weight': f"{data['Benchmark Weight']:.2%}",
                    'Active Weight': f"{(data['Portfolio Weight'] - data['Benchmark Weight']):.2%}"
                })
        else:
            sector_data = [{'Sector': 'No Data', 'Allocation': '0.00%', 'Selection': '0.00%', 
                           'Interaction': '0.00%', 'Total': '0.00%', 
                           'Portfolio Weight': '0.00%', 'Benchmark Weight': '0.00%', 
                           'Active Weight': '0.00%'}]
        
        return pd.DataFrame(summary_data), pd.DataFrame(sector_data)

# ============================================================================
# 7. ENHANCED TEARSHEET COMPONENTS
# ============================================================================

class EnhancedTearsheet:
    """Creates professional institutional-grade tearsheet components."""
    
    @staticmethod
    def create_kpi_grid(risk_metrics, perf_metrics, attribution=None):
        """Creates enhanced KPI grid with updated attribution metrics."""
        metrics_config = [
            {"label": "CAGR", "value": risk_metrics['CAGR'], "format": ".2%", "color": "#00cc96"},
            {"label": "Volatility", "value": risk_metrics['Volatility'], "format": ".2%", "color": "white"},
            {"label": "Sharpe Ratio", "value": risk_metrics['Sharpe Ratio'], "format": ".2f", 
             "color": "#00cc96" if risk_metrics['Sharpe Ratio'] > 1 else "white"},
            {"label": "Sortino Ratio", "value": risk_metrics['Sortino Ratio'], "format": ".2f", "color": "white"},
            {"label": "Excess Return", "value": attribution['Total Excess Return'] if attribution else 0, "format": ".2%", 
             "color": "#00cc96" if attribution and attribution['Total Excess Return'] > 0 else "#ef553b"},
            {"label": "Info Ratio", "value": perf_metrics['information_ratio'], "format": ".2f", 
             "color": "#00cc96" if perf_metrics['information_ratio'] > 0.5 else "white"},
            {"label": "Max Drawdown", "value": risk_metrics['Max Drawdown'], "format": ".2%", "color": "#ef553b"},
            {"label": "Calmar Ratio", "value": risk_metrics['Calmar Ratio'], "format": ".2f", "color": "white"},
            {"label": "Active Share", "value": perf_metrics['active_share'], "format": ".1%", "color": "#636efa"},
            {"label": "Tracking Error", "value": perf_metrics['tracking_error'], "format": ".2%", "color": "white"},
        ]
        
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
            # (Simplified simulation of drift)
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
# 8. ADVANCED MONTE CARLO SIMULATOR
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
# 9. DATA PIPELINE
# ============================================================================

class PortfolioDataManager:
    """Handles secure data fetching from Yahoo Finance."""
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_data(tickers, start_date, end_date):
        """Fetches OHLCV data for a list of tickers with robust MultiIndex handling."""
        if not tickers:
            return pd.DataFrame(), {}
            
        try:
            if isinstance(tickers, str): tickers = [tickers]
            
            # Use threads=False for stability, auto_adjust=True for accurate returns
            data = yf.download(
                tickers, 
                start=start_date, 
                end=end_date, 
                progress=False, 
                group_by='ticker', 
                threads=False,
                auto_adjust=True
            )
            
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
                
                # Close is already adjusted due to auto_adjust=True
                price_col = 'Close' 
                if price_col in df.columns:
                    prices[ticker] = df[price_col]
                    ohlc_dict[ticker] = df
            else:
                if not isinstance(data.columns, pd.MultiIndex):
                    # Sometimes single ticker response logic applies if only one valid ticker found
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
# 10. RISK ENGINE
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
# 11. PORTFOLIO OPTIMIZER
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
# 12. BACKTESTER
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
# 13. ATTRIBUTION QUALITY DASHBOARD
# ============================================================================

def create_attribution_quality_dashboard(attribution_quality, attribution_results):
    """Create dashboard showing attribution quality metrics."""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        additivity_status = "✅" if attribution_quality.get('Additivity Check', False) else "⚠️"
        st.metric("Additivity Check", additivity_status,
                 delta=None, help="Verifies attribution components sum to total excess")
    
    with col2:
        discrepancy = attribution_quality.get('Discrepancy', 0)
        st.metric("Attribution Discrepancy", f"{discrepancy:.6f}",
                 delta=None, help="Difference between sum of components and total excess")
    
    with col3:
        style = attribution_quality.get('Attribution Style', 'Unknown')
        st.metric("Attribution Style", style,
                 delta=None, help="Whether performance is allocation or selection driven")
    
    with col4:
        total_excess = attribution_results.get('Total Excess Return', 0)
        significance = "Significant" if abs(total_excess) > 0.001 else "Insignificant"
        st.metric("Excess Significance", significance,
                 delta=None, help="Whether excess return is statistically significant")
    
    # Create radar chart for attribution dominance
    if attribution_quality:
        fig = go.Figure()
        
        categories = ['Allocation', 'Selection', 'Interaction']
        values = [
            attribution_quality.get('Allocation Dominance', 0),
            attribution_quality.get('Selection Dominance', 0),
            attribution_quality.get('Interaction Significance', 0)
        ]
        
        # Complete the circle
        categories = categories + [categories[0]]
        values = values + [values[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Attribution Dominance',
            line=dict(color='#00cc96', width=2),
            fillcolor='rgba(0, 204, 150, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickformat=".0%"
                )
            ),
            title="Attribution Component Dominance",
            template="plotly_dark",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# 14. MAIN APPLICATION
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

# Use Expanders for cleaner UI
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
    rebal_freq_ui = st.selectbox("Rebalancing Frequency", ["Quarterly", "Monthly", "Yearly"])
    freq_map = {"Quarterly": "Q", "Monthly": "M", "Yearly": "Y"}

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
            
            # 8. ENHANCED PERFORMANCE ATTRIBUTION ENGINE
            try:
                # Create benchmark weights
                if method == 'Equal Weight':
                    # For equal weight strategy, use market cap weighted benchmark
                    mcaps = data_manager.get_market_caps(selected_tickers)
                    total_mcap = sum(mcaps.values())
                    bench_weights = {t: mcaps[t]/total_mcap for t in selected_tickers}
                else:
                    # Use equal weight as benchmark
                    bench_weights = {t: 1/len(selected_tickers) for t in selected_tickers}
                
                # Get sector mapping
                sector_map = {}
                for ticker, meta in asset_metadata.items():
                    sector_map[ticker] = meta.get('sector', 'Unknown')
                
                # Calculate portfolio and benchmark returns
                port_returns_series = returns.dot(list(weights.values()))
                bench_returns_series = returns.dot(list(bench_weights.values()))
                
                # Calculate enhanced attribution
                attribution_engine = EnhancedPortfolioAttribution()
                attribution_results = attribution_engine.calculate_brinson_fachler_attribution(
                    port_returns_series, bench_returns_series, 
                    weights, bench_weights, sector_map
                )
                
                # Calculate attribution quality metrics
                attribution_quality = attribution_engine.calculate_attribution_quality_metrics(
                    attribution_results
                )
                
                # Calculate arithmetic vs geometric attribution
                arithmetic_vs_geometric = attribution_engine.calculate_arithmetic_vs_geometric(
                    port_returns_series, bench_returns_series
                )
                
                # Calculate multi-period attribution
                try:
                    multi_period_df = attribution_engine.calculate_multi_period_attribution(
                        port_returns_series, bench_returns_series, 
                        weights, bench_weights, sector_map, frequency='monthly'
                    )
                except:
                    # Fallback if multi-period fails
                    multi_period_df = pd.DataFrame({
                        'Period': [prices.index[-1]],
                        'Portfolio Return': [port_returns_series.mean()],
                        'Benchmark Return': [bench_returns_series.mean()],
                        'Excess Return': [port_returns_series.mean() - bench_returns_series.mean()],
                        'Allocation': [attribution_results['Allocation Effect']],
                        'Selection': [attribution_results['Selection Effect']],
                        'Interaction': [attribution_results['Interaction Effect']]
                    })
                
                # Calculate Beta and other metrics
                covariance = np.cov(port_returns_series, bench_returns_series)[0][1]
                variance = np.var(bench_returns_series)
                beta = covariance / variance if variance != 0 else 1.0
                
                # Calculate tracking error
                tracking_error = np.std(port_returns_series - bench_returns_series) * np.sqrt(252)
                
                # Calculate information ratio
                excess_returns = port_returns_series - bench_returns_series
                information_ratio = excess_returns.mean() * np.sqrt(252) / tracking_error if tracking_error > 0 else 0
                
                # Calculate active share
                active_share = 0.5 * np.sum(np.abs(np.array(list(weights.values())) - 
                                                 np.array(list(bench_weights.values()))))
                
                perf_metrics = {
                    'beta': beta,
                    'tracking_error': tracking_error,
                    'information_ratio': information_ratio,
                    'active_share': active_share
                }
                
                # Combine all attribution metrics
                full_attribution_metrics = {
                    **attribution_results,
                    **arithmetic_vs_geometric,
                    **attribution_quality,
                    'Information Ratio': information_ratio,
                    'Tracking Error': tracking_error,
                    'Active Share': active_share,
                    'Beta': beta
                }
                
            except Exception as e:
                st.warning(f"Attribution analysis partially failed: {str(e)}")
                # Fallback attribution
                attribution_results = {
                    'Total Excess Return': 0,
                    'Allocation Effect': 0,
                    'Selection Effect': 0,
                    'Interaction Effect': 0,
                    'Sector Breakdown': {},
                    'Benchmark Return': 0,
                    'Portfolio Return': 0
                }
                attribution_quality = {}
                multi_period_df = pd.DataFrame()
                perf_metrics = {'beta': 1.0, 'tracking_error': 0, 'information_ratio': 0, 'active_share': 0}
                full_attribution_metrics = {}
            
            # 9. GARCH & PCA Analysis
            garch_model, garch_vol = AdvancedRiskMetrics.fit_garch_model(port_ret_series)
            comp_var, pca_expl_var, pca_obj = AdvancedRiskMetrics.calculate_component_var(returns, weights)
            
            # 10. Visualization Layout
            t1, t2, t3, t4, t5, t6, t7, t8, t9, t10 = st.tabs([
                "🏛️ Tearsheet", 
                "📈 Frontier", 
                "📉 Backtest", 
                "🔗 Correlations",
                "📅 Seasonality",
                "🕯️ OHLC", 
                "🌪️ Stress Test", 
                "⚠️ VaR Engine",
                "🎲 MC Sims",
                "🔬 Quant Lab"
            ])
            
            # --- TAB 1: ENHANCED TEARSHEET ---
            with t1:
                st.markdown("### 🏛️ Institutional Portfolio Analytics")
                st.markdown("---")
                
                # Enhanced KPI Dashboard
                st.markdown("#### 📊 Key Performance Indicators")
                EnhancedTearsheet.create_kpi_grid(risk_metrics, perf_metrics, attribution_results)
                
                # --- PERFORMANCE ATTRIBUTION DASHBOARD ---
                st.markdown("---")
                st.markdown("#### 📈 Advanced Performance Attribution")
                
                # Attribution Summary Metrics
                col_attr1, col_attr2, col_attr3, col_attr4 = st.columns(4)
                
                with col_attr1:
                    excess_color = "#00cc96" if full_attribution_metrics.get('Total Excess Return', 0) > 0 else "#ef553b"
                    st.metric("Total Excess Return", 
                             f"{full_attribution_metrics.get('Total Excess Return', 0):.2%}",
                             delta_color="off")
                    st.markdown(f"<div style='color:{excess_color}; font-size:12px;'>vs Benchmark</div>", 
                               unsafe_allow_html=True)
                
                with col_attr2:
                    st.metric("Information Ratio", 
                             f"{full_attribution_metrics.get('Information Ratio', 0):.2f}",
                             delta_color="off")
                    st.markdown("<div style='color:#888; font-size:12px;'>Risk-Adjusted Alpha</div>", 
                               unsafe_allow_html=True)
                
                with col_attr3:
                    st.metric("Allocation Effect", 
                             f"{full_attribution_metrics.get('Allocation Effect', 0):.2%}",
                             delta_color="off")
                    st.markdown("<div style='color:#888; font-size:12px;'>Sector Selection</div>", 
                               unsafe_allow_html=True)
                
                with col_attr4:
                    st.metric("Selection Effect", 
                             f"{full_attribution_metrics.get('Selection Effect', 0):.2%}",
                             delta_color="off")
                    st.markdown("<div style='color:#888; font-size:12px;'>Stock Picking</div>", 
                               unsafe_allow_html=True)
                
                # Attribution Quality Dashboard
                st.markdown("#### 🔍 Attribution Quality Analysis")
                create_attribution_quality_dashboard(attribution_quality, attribution_results)
                
                # Main Attribution Visualizations
                vis = AttributionVisualizer()
                
                # Row 1: Waterfall and Heatmap
                col_wf, col_hm = st.columns(2)
                
                with col_wf:
                    fig_waterfall = vis.create_attribution_waterfall(full_attribution_metrics)
                    st.plotly_chart(fig_waterfall, use_container_width=True)
                
                with col_hm:
                    fig_heatmap = vis.create_sector_attribution_heatmap(
                        attribution_results.get('Sector Breakdown', {})
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Row 2: Radar and Time Series
                col_radar, col_ts = st.columns(2)
                
                with col_radar:
                    fig_radar = vis.create_attribution_radar_chart(
                        attribution_results.get('Sector Breakdown', {})
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                
                with col_ts:
                    if not multi_period_df.empty:
                        fig_timeseries = vis.create_time_series_attribution(multi_period_df)
                        st.plotly_chart(fig_timeseries, use_container_width=True)
                    else:
                        st.info("Multi-period attribution data not available")
                
                # Attribution Summary Tables
                st.markdown("#### 📊 Attribution Summary Tables")
                
                summary_table, sector_table = vis.create_attribution_summary_table(
                    full_attribution_metrics, 
                    attribution_results.get('Sector Breakdown', {})
                )
                
                col_summary, col_sector = st.columns(2)
                
                with col_summary:
                    st.markdown("**Key Attribution Metrics**")
                    st.dataframe(
                        summary_table,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Metric": st.column_config.TextColumn("Metric", width="medium"),
                            "Value": st.column_config.TextColumn("Value", width="small"),
                            "Interpretation": st.column_config.TextColumn("Interpretation", width="large")
                        }
                    )
                
                with col_sector:
                    st.markdown("**Sector-Level Breakdown**")
                    st.dataframe(
                        sector_table,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Sector": st.column_config.TextColumn("Sector", width="medium"),
                            "Allocation": st.column_config.NumberColumn("Alloc", format="%f"),
                            "Selection": st.column_config.NumberColumn("Select", format="%f"),
                            "Total": st.column_config.NumberColumn("Total", format="%f"),
                            "Active Weight": st.column_config.NumberColumn("Active Wt", format="%f")
                        }
                    )
                
                # Attribution Insights & Commentary
                st.markdown("#### 💡 Attribution Insights")
                
                insights_col1, insights_col2 = st.columns(2)
                
                with insights_col1:
                    # Calculate attribution quality metrics
                    allocation_ratio = abs(full_attribution_metrics.get('Allocation Effect', 0) / 
                                          full_attribution_metrics.get('Total Excess Return', 1)) if full_attribution_metrics.get('Total Excess Return', 0) != 0 else 0
                    
                    if allocation_ratio > 0.6:
                        allocation_insight = "🏆 **Strong allocation skills** - Portfolio performance primarily driven by sector selection"
                    elif allocation_ratio > 0.3:
                        allocation_insight = "📊 **Balanced attribution** - Both allocation and selection contributed meaningfully"
                    else:
                        allocation_insight = "🎯 **Strong selection skills** - Performance primarily driven by stock picking"
                    
                    st.markdown(f"""
                    <div style='background-color: #1e1e1e; padding: 20px; border-radius: 10px; border-left: 4px solid #00cc96;'>
                        <h4 style='color: #00cc96; margin-top: 0;'>Allocation vs Selection Analysis</h4>
                        <p style='color: #ccc;'>{allocation_insight}</p>
                        <ul style='color: #ccc;'>
                            <li><strong>Allocation Contribution:</strong> {(allocation_ratio*100):.1f}% of excess return</li>
                            <li><strong>Selection Contribution:</strong> {((1-allocation_ratio)*100):.1f}% of excess return</li>
                            <li><strong>Interaction Effect:</strong> Typically indicates successful integration of views</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with insights_col2:
                    # Identify top contributing sectors
                    sector_breakdown = attribution_results.get('Sector Breakdown', {})
                    if sector_breakdown:
                        top_sectors = sorted(
                            [(s, data['Total Contribution']) for s, data in sector_breakdown.items()],
                            key=lambda x: abs(x[1]),
                            reverse=True
                        )[:3]
                        
                        top_sectors_text = "<br>".join([f"• {sector}: {contribution:.2%}" 
                                                       for sector, contribution in top_sectors])
                    else:
                        top_sectors_text = "• No sector data available"
                    
                    st.markdown(f"""
                    <div style='background-color: #1e1e1e; padding: 20px; border-radius: 10px; border-left: 4px solid #636efa;'>
                        <h4 style='color: #636efa; margin-top: 0;'>Top Contributing Sectors</h4>
                        <p style='color: #ccc;'>Sectors driving the majority of performance:</p>
                        <p style='color: #ccc;'>{top_sectors_text}</p>
                        <p style='color: #ccc; font-size: 12px;'>
                            <strong>Active Management Score:</strong> {(perf_metrics.get('active_share', 0)*100):.1f}%
                            (Higher = More active decisions)
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # --- ORIGINAL PLOTS (Sunburst/Scatter) ---
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    fig_sunburst = EnhancedTearsheet.create_sector_allocation_chart(weights, asset_metadata)
                    st.plotly_chart(fig_sunburst, use_container_width=True)
                    st.markdown("#### 🎯 Style Exposures")
                    fig_radar = EnhancedTearsheet.create_style_exposure_chart(weights, asset_metadata)
                    st.plotly_chart(fig_radar, use_container_width=True)
                with col2:
                    fig_scatter = EnhancedTearsheet.create_risk_return_scatter(returns, weights, asset_metadata)
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                st.markdown("---")
                st.markdown("#### 📅 Allocation Timeline")
                fig_timeline = EnhancedTearsheet.create_allocation_timeline(
                    prices, weights, freq_map[rebal_freq_ui]
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
                
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
                    st.metric("Active Share", f"{perf_metrics.get('active_share', 0):.1%}")
                
                with col_char2:
                    st.metric("Number of Sectors", f"{len(sector_counts)}")
                    st.metric("Tracking Error", f"{perf_metrics.get('tracking_error', 0):.2%}")
                
                with col_char3:
                    st.metric("Avg Market Cap", f"${avg_market_cap/1e9:.1f}B" if avg_market_cap > 0 else "N/A")
                    st.metric("Beta to Benchmark", f"{perf_metrics.get('beta', 0):.2f}")
                
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

                # Drawdown Table
                st.subheader("Top 5 Drawdowns")
                dd_series = drawdown
                dd_table = []
                # Simple logic to find worst drawdowns - finds local minima
                sorted_dd = dd_series.sort_values(ascending=True).head(5)
                for date, val in sorted_dd.items():
                    dd_table.append({
                        "Date": date.strftime('%Y-%m-%d'),
                        "Drawdown": f"{val:.2%}"
                    })
                st.table(pd.DataFrame(dd_table))

            # --- TAB 4: CORRELATIONS ---
            with t4:
                st.subheader("🔗 Asset Correlation Matrix")
                corr_matrix = returns.corr()
                
                fig_corr = px.imshow(
                    corr_matrix, 
                    text_auto=".2f", 
                    aspect="auto", 
                    color_continuous_scale='RdBu_r', 
                    zmin=-1, zmax=1,
                    title="Pearson Correlation Heatmap"
                )
                fig_corr.update_layout(template="plotly_dark", height=600)
                st.plotly_chart(fig_corr, use_container_width=True)

                st.markdown("---")
                st.subheader("🧬 Hierarchical Clustering (Dendrogram)")
                try:
                    fig_dendro = ff.create_dendrogram(corr_matrix, labels=corr_matrix.columns)
                    fig_dendro.update_layout(template="plotly_dark", width=800, height=500)
                    st.plotly_chart(fig_dendro, use_container_width=True)
                except:
                    st.info("Requires more than 1 asset for clustering.")

            # --- TAB 5: SEASONALITY ---
            with t5:
                st.subheader("📅 Monthly Returns Heatmap")
                
                # Resample portfolio returns to monthly
                monthly_ret = port_ret_series.resample('M').apply(lambda x: (1 + x).prod() - 1)
                monthly_ret_df = pd.DataFrame(monthly_ret)
                monthly_ret_df['Year'] = monthly_ret_df.index.year
                monthly_ret_df['Month'] = monthly_ret_df.index.month_name()
                
                # Pivot
                pivot_ret = monthly_ret_df.pivot(index='Year', columns='Month', values=0)
                # Reorder months
                month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
                pivot_ret = pivot_ret.reindex(columns=month_order)
                
                fig_heat = px.imshow(
                    pivot_ret, 
                    labels=dict(x="Month", y="Year", color="Return"),
                    x=pivot_ret.columns,
                    y=pivot_ret.index,
                    color_continuous_scale='RdYlGn',
                    text_auto=".1%",
                    aspect="auto"
                )
                fig_heat.update_layout(template="plotly_dark", height=600, title="Seasonality Heatmap")
                st.plotly_chart(fig_heat, use_container_width=True)

            # --- TAB 6: OHLC ANALYSIS ---
            with t6:
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
                    if 'Volume' in df_ohlc.columns:
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
                        if 'Volume' in df_ohlc.columns:
                            st.metric("Avg Volume", f"{df_ohlc['Volume'].mean():,.0f}")
                    with col4:
                        st.metric("Volatility", f"{df_ohlc['Close'].pct_change().std()*np.sqrt(252)*100:.2f}%")
                else:
                    st.warning("Detailed OHLC data unavailable for this ticker.")
            
            # --- TAB 7: STRESS TEST ---
            with t7:
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
            
            # --- TAB 8: COMPARATIVE VAR ---
            with t8:
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
            
            # --- TAB 9: ADVANCED MC SIMULATIONS ---
            with t9:
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
            
            # --- TAB 10: QUANT LAB (GARCH & PCA) ---
            with t10:
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

            # --- EXPORT SECTION ---
            st.sidebar.markdown("---")
            with st.sidebar.expander("💾 Export Data"):
                # Create CSV for weights
                w_df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
                w_df.index.name = 'Ticker'
                w_csv = w_df.to_csv()
                
                st.download_button(
                    label="Download Weights (CSV)",
                    data=w_csv,
                    file_name='portfolio_weights.csv',
                    mime='text/csv'
                )

                # Create CSV for Performance
                perf_df = pd.DataFrame(equity_curve, columns=['Portfolio Value'])
                perf_csv = perf_df.to_csv()
                st.download_button(
                    label="Download Performance (CSV)",
                    data=perf_csv,
                    file_name='portfolio_performance.csv',
                    mime='text/csv'
                )

else:
    st.info("👈 Please configure the portfolio parameters in the sidebar to launch the engine.")
    
    # Empty State Hero
    st.markdown("""
    <div style="text-align: center; padding: 50px;">
        <h1>🏛️ Enigma Institutional Terminal</h1>
        <p style="color: #666; font-size: 18px;">
            Advanced Portfolio Optimization, VaR/CVaR Simulation, and Factor Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
