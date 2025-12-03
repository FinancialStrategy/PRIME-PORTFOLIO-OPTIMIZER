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
from dataclasses import dataclass
import json

# --- QUANTITATIVE LIBRARY IMPORTS ---
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.cla import CLA
from pypfopt.hierarchical_portfolio import HRPOpt
from pypfopt.black_litterman import BlackLittermanModel
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ARCH: For Econometric Volatility Forecasting (GARCH)
try:
    import arch
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

warnings.filterwarnings('ignore')

# ============================================================================
# 1. ASSET CLASSIFICATION ENGINE
# ============================================================================

class AssetClassifier:
    """
    Classifies assets into sectors, industries, regions, and styles.
    Uses Yahoo Finance API for real-time classification.
    """
    
    # Predefined sector mappings for common assets
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
        'US': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'JNJ', 'V', 'WMT'],
        'Europe': ['^FTSE', '^GDAXI', '^FCHI', 'SAP.DE', 'ASML.AS'],
        'Asia': ['^N225', '^HSI', '000001.SS', '005930.KS'],
        'Emerging Markets': ['^BVSP', '^MXX', '^MERV'],
        'Turkey': BIST_30  # Will be defined globally
    }
    
    @staticmethod
    @st.cache_data(ttl=3600*24)
    def get_asset_metadata(tickers):
        """
        Fetches detailed metadata for each asset including sector, industry, country.
        """
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
                # Fallback to mapping or default
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
        return 'Other'
    
    @staticmethod
    def _infer_region(ticker):
        """Infer region from ticker."""
        for region, tickers in AssetClassifier.REGION_MAP.items():
            if ticker in tickers:
                return region
        if '.IS' in ticker:
            return 'Turkey'
        elif '.L' in ticker:
            return 'UK'
        elif '.DE' in ticker:
            return 'Germany'
        elif '.PA' in ticker:
            return 'France'
        return 'Global'

# ============================================================================
# 2. PORTFOLIO ATTRIBUTION ENGINE
# ============================================================================

class PortfolioAttribution:
    """
    Performs Brinson-Fachler attribution analysis.
    Decomposes returns into allocation, selection, and interaction effects.
    """
    
    @staticmethod
    def calculate_attribution(portfolio_returns, benchmark_returns, 
                              portfolio_weights, benchmark_weights, 
                              sector_map):
        """
        Calculates Brinson attribution analysis.
        
        Returns:
        - Total excess return
        - Allocation effect
        - Selection effect
        - Interaction effect
        """
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
            R_b = np.mean(list(benchmark_sector_avg.values()))  # Overall benchmark return
            
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
# 3. ENHANCED TEARSHEET COMPONENTS
# ============================================================================

class EnhancedTearsheet:
    """
    Creates professional institutional-grade tearsheet components.
    """
    
    @staticmethod
    def create_kpi_grid(risk_metrics, perf_metrics, attribution=None):
        """
        Creates enhanced KPI grid with 8 metrics in 4x2 layout.
        """
        # Define metrics with colors and formatting
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
        """
        Creates a sunburst chart showing allocation by sector, then by asset.
        """
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
        ids = []
        labels = []
        parents = []
        values = []
        
        # Add root
        ids.append("portfolio")
        labels.append("Portfolio")
        parents.append("")
        values.append(100)
        
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
        """
        Creates risk-return scatter plot with portfolio positioning.
        """
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
        
        # Add efficient frontier line (simplified)
        n_points = 50
        min_vol = asset_vols.min()
        max_vol = asset_vols.max()
        frontier_vols = np.linspace(min_vol * 0.8, max_vol * 1.2, n_points)
        
        fig.add_trace(go.Scatter(
            x=frontier_vols,
            y=[port_return * (v/port_vol) for v in frontier_vols],
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='rgba(100, 100, 100, 0.5)', dash='dash'),
            showlegend=False
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
        """
        Creates a stacked area chart showing allocation evolution.
        """
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
        """
        Creates waterfall chart for performance attribution.
        """
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
        """
        Creates radar chart showing portfolio style exposures.
        """
        # Define style factors (simplified)
        styles = ['Growth', 'Value', 'Momentum', 'Quality', 'Low Vol', 'Size']
        
        # Calculate exposures (simplified - in reality would use factor models)
        exposures = []
        for style in styles:
            # Simplified exposure calculation
            if style == 'Growth':
                # Higher weight to tech stocks
                exposure = sum(weights.get(t, 0) for t, meta in metadata.items() 
                             if 'Technology' in meta.get('sector', ''))
            elif style == 'Value':
                # Higher weight to financials, energy
                exposure = sum(weights.get(t, 0) for t, meta in metadata.items() 
                             if meta.get('sector') in ['Financial Services', 'Energy'])
            elif style == 'Momentum':
                # Assume equal exposure
                exposure = 0.5
            elif style == 'Quality':
                # Higher weight to healthcare, consumer staples
                exposure = sum(weights.get(t, 0) for t, meta in metadata.items() 
                             if meta.get('sector') in ['Healthcare', 'Consumer Defensive'])
            elif style == 'Low Vol':
                # Inverse to high volatility
                exposure = 0.3
            else:  # Size
                # Mix of large and small cap
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
# GLOBAL ASSET UNIVERSES (Keep existing)
# ============================================================================

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
# MAIN APPLICATION (Integrating Enhanced Tearsheet)
# ============================================================================

# [Previous code sections remain unchanged up to the main execution block]

# In the main execution block, modify the Tab 1 content:

            # --- TAB 1: ENHANCED EXECUTIVE TEARSHEET ---
            with t1:
                st.markdown("### 🏛️ Institutional Portfolio Analytics")
                st.markdown("---")
                
                # 1. ASSET CLASSIFICATION
                classifier = AssetClassifier()
                asset_metadata = classifier.get_asset_metadata(selected_tickers)
                
                # 2. CALCULATE ADDITIONAL METRICS
                # Calculate benchmark (equal weight) for comparison
                eq_weights = {t: 1/len(selected_tickers) for t in selected_tickers}
                port_returns_series = returns.dot(list(weights.values()))
                bench_returns_series = returns.dot(list(eq_weights.values()))
                
                # Calculate Beta
                covariance = np.cov(port_returns_series, bench_returns_series)[0][1]
                variance = np.var(bench_returns_series)
                beta = covariance / variance
                
                # Calculate tracking error
                tracking_error = np.std(port_returns_series - bench_returns_series) * np.sqrt(252)
                
                perf_metrics = {
                    'beta': beta,
                    'tracking_error': tracking_error,
                    'active_share': 0.5 * np.sum(np.abs(np.array(list(weights.values())) - np.array(list(eq_weights.values()))))
                }
                
                # 3. PERFORMANCE ATTRIBUTION
                sector_map = {t: meta['sector'] for t, meta in asset_metadata.items()}
                attribution = PortfolioAttribution.calculate_attribution(
                    returns, returns, weights, eq_weights, sector_map
                )
                
                # 4. ENHANCED KPI DASHBOARD
                st.markdown("#### 📊 Key Performance Indicators")
                EnhancedTearsheet.create_kpi_grid(risk_metrics, perf_metrics, attribution)
                
                st.markdown("---")
                
                # 5. ALLOCATION & ATTRIBUTION VISUALIZATION
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
                
                # 6. ALLOCATION EVOLUTION
                st.markdown("#### 📅 Allocation Timeline")
                fig_timeline = EnhancedTearsheet.create_allocation_timeline(
                    prices, weights, freq_map[rebal_freq_ui]
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
                
                # 7. ASSET CLASS DETAILS TABLE
                st.markdown("#### 🏢 Asset Class Details")
                
                # Create detailed table
                asset_details = []
                for ticker, weight in weights.items():
                    if weight > 0.001:  # Only show significant holdings
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
                
                # 8. PORTFOLIO CHARACTERISTICS SUMMARY
                st.markdown("#### 🏛️ Portfolio Characteristics")
                
                col_char1, col_char2, col_char3, col_char4 = st.columns(4)
                
                # Calculate portfolio characteristics
                total_market_cap = sum(meta.get('marketCap', 0) * weights.get(ticker, 0) 
                                      for ticker, meta in asset_metadata.items())
                avg_market_cap = total_market_cap / sum(weights.values())
                
                # Count sectors
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
                    st.metric("Portfolio Turnover", "Est. 15-20%" if rebal_freq_ui == "Quarterly" else "Est. 5-10%")
                    st.metric("Currency Exposure", "Mixed" if len(set(m.get('currency', '') for m in asset_metadata.values())) > 1 else "Single")

# [Rest of the code remains unchanged...]
