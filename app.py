import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# PyPortfolioOpt imports
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import CLA, HRPOpt
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt import plotting

# Additional financial libraries
import scipy.stats as stats
from scipy import optimize
import arch
from streamlit_option_menu import option_menu

# Page configuration
st.set_page_config(
    page_title="Advanced Portfolio Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# 1. DATA MANAGEMENT MODULE
# ============================================================================

class PortfolioDataManager:
    """Professional-grade financial data manager with caching"""
    
    def __init__(self):
        self.data_cache = {}
        self.benchmarks = {
            'SPY': 'S&P 500',
            'QQQ': 'NASDAQ 100',
            'IWM': 'Russell 2000',
            'AGG': 'US Aggregate Bond',
            'GLD': 'Gold'
        }
    
    @st.cache_data(ttl=3600)
    def fetch_data(_self, tickers, start_date, end_date):
        """Fetch and cache financial data"""
        try:
            data = yf.download(tickers, start=start_date, end=end_date, progress=False)
            
            if len(tickers) == 1:
                data = pd.DataFrame(data['Adj Close']).rename(columns={'Adj Close': tickers[0]})
            else:
                data = data['Adj Close']
            
            data = data.ffill().bfill()
            return data
            
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def calculate_returns(self, prices, method='log'):
        """Calculate returns with multiple methodologies"""
        if method == 'log':
            returns = np.log(prices / prices.shift(1)).dropna()
        else:
            returns = prices.pct_change().dropna()
        return returns
    
    def get_benchmark_data(self, start_date, end_date):
        """Fetch benchmark data for comparison"""
        benchmark_tickers = list(self.benchmarks.keys())
        return self.fetch_data(benchmark_tickers, start_date, end_date)

# ============================================================================
# 2. RISK METRICS MODULE
# ============================================================================

class AdvancedRiskMetrics:
    """Advanced risk calculation module"""
    
    @staticmethod
    def calculate_var(returns, confidence_level=0.95, method='historical'):
        """Calculate Value at Risk using multiple methods"""
        if method == 'historical':
            var = np.percentile(returns, (1 - confidence_level) * 100)
        elif method == 'parametric':
            mean = returns.mean()
            std = returns.std()
            var = stats.norm.ppf(1 - confidence_level, mean, std)
        elif method == 'modified':
            z = stats.norm.ppf(1 - confidence_level)
            s = stats.skew(returns)
            k = stats.kurtosis(returns)
            z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3*z) * k / 24 - (2*z**3 - 5*z) * s**2 / 36
            var = returns.mean() + z_cf * returns.std()
        return var
    
    @staticmethod
    def calculate_cvar(returns, confidence_level=0.95):
        """Calculate Conditional Value at Risk"""
        var = AdvancedRiskMetrics.calculate_var(returns, confidence_level, 'historical')
        cvar = returns[returns <= var].mean()
        return cvar
    
    @staticmethod
    def calculate_garch_var(returns, confidence_level=0.95):
        """Calculate VaR using GARCH model"""
        try:
            am = arch.arch_model(returns * 100, vol='Garch', p=1, q=1, dist='normal')
            res = am.fit(disp='off')
            forecast = res.forecast(horizon=1)
            vol_forecast = np.sqrt(forecast.variance.values[-1, 0]) / 100
            var = stats.norm.ppf(1 - confidence_level) * vol_forecast
            return var
        except:
            return AdvancedRiskMetrics.calculate_var(returns, confidence_level, 'parametric')
    
    @staticmethod
    def stress_test_portfolio(weights, returns, scenarios=None):
        """Perform stress testing on portfolio"""
        if scenarios is None:
            scenarios = {
                'Market Crash': -0.10,
                'High Volatility': returns.std() * 2,
                'Bear Market': -0.05,
                'Flash Crash': -0.15,
            }
        
        portfolio_returns = returns.dot(weights)
        results = {}
        
        for scenario, shock in scenarios.items():
            if scenario == 'High Volatility':
                shocked_returns = portfolio_returns * (shock / portfolio_returns.std())
            else:
                shocked_returns = portfolio_returns + shock
            
            results[scenario] = {
                'Mean Return': shocked_returns.mean(),
                'Std Dev': shocked_returns.std(),
                'VaR 95%': AdvancedRiskMetrics.calculate_var(shocked_returns, 0.95),
                'CVaR 95%': AdvancedRiskMetrics.calculate_cvar(shocked_returns, 0.95),
                'Max Drawdown': AdvancedRiskMetrics.calculate_max_drawdown(shocked_returns.cumsum())
            }
        
        return pd.DataFrame(results).T
    
    @staticmethod
    def calculate_max_drawdown(cumulative_returns):
        """Calculate maximum drawdown"""
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()
    
    @staticmethod
    def calculate_risk_decomposition(weights, covariance_matrix):
        """Decompose portfolio risk by asset"""
        portfolio_variance = weights.T @ covariance_matrix @ weights
        marginal_contrib = covariance_matrix @ weights
        risk_contrib = weights * marginal_contrib / portfolio_variance
        return risk_contrib

# ============================================================================
# 3. PORTFOLIO OPTIMIZATION MODULE
# ============================================================================

class AdvancedPortfolioOptimizer:
    """Professional portfolio optimizer"""
    
    def __init__(self, returns, prices):
        self.returns = returns
        self.prices = prices
        self.mu = expected_returns.mean_historical_return(prices)
        self.S = risk_models.sample_cov(prices)
    
    def optimize_mean_variance(self, objective='sharpe', target_return=None, 
                              risk_free_rate=0.02, gamma=1.0):
        """Mean-variance optimization"""
        ef = EfficientFrontier(self.mu, self.S)
        
        if objective == 'sharpe':
            weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
        elif objective == 'min_volatility':
            weights = ef.min_volatility()
        elif objective == 'max_quadratic_utility':
            weights = ef.max_quadratic_utility(risk_aversion=gamma)
        elif objective == 'efficient_risk':
            weights = ef.efficient_risk(target_volatility=target_return)
        elif objective == 'efficient_return':
            weights = ef.efficient_return(target_return=target_return)
        
        cleaned_weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
        
        return cleaned_weights, performance
    
    def optimize_cvar(self, confidence_level=0.95, target_return=None):
        """CVaR optimization"""
        from pypfopt import EfficientCVaR
        
        scenarios = self.returns.values
        ec = EfficientCVaR(self.mu, scenarios, confidence_level=confidence_level)
        
        if target_return:
            weights = ec.efficient_return(target_return)
        else:
            weights = ec.min_cvar()
        
        cleaned_weights = ec.clean_weights()
        performance = ec.portfolio_performance(verbose=False)
        
        return cleaned_weights, performance
    
    def optimize_hrp(self):
        """Hierarchical Risk Parity optimization"""
        hrp = HRPOpt(self.returns)
        weights = hrp.optimize()
        cleaned_weights = hrp.clean_weights()
        performance = hrp.portfolio_performance(verbose=False)
        
        return cleaned_weights, performance
    
    def optimize_black_litterman(self, market_caps, views=None, view_confidences=None):
        """Black-Litterman model optimization"""
        if views is None:
            views = pd.Series(0.05, index=self.mu.index)
            view_confidences = [0.5] * len(views)
        
        bl = BlackLittermanModel(self.S, pi=self.mu, absolute_views=views)
        bl_mu = bl.bl_returns()
        bl_S = bl.bl_cov()
        
        ef = EfficientFrontier(bl_mu, bl_S)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False)
        
        return cleaned_weights, performance
    
    def monte_carlo_optimization(self, n_portfolios=10000):
        """Monte Carlo simulation for efficient frontier"""
        n_assets = len(self.mu)
        results = np.zeros((n_portfolios, 3))
        weights_record = []
        
        for i in range(n_portfolios):
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)
            weights_record.append(weights)
            
            portfolio_return = np.sum(self.mu.values * weights)
            portfolio_volatility = np.sqrt(weights.T @ self.S.values @ weights)
            sharpe_ratio = (portfolio_return - 0.02) / portfolio_volatility
            
            results[i, :] = [portfolio_return, portfolio_volatility, sharpe_ratio]
        
        return results, np.array(weights_record)

# ============================================================================
# 4. BACKTESTING MODULE
# ============================================================================

class PortfolioBacktester:
    """Advanced portfolio backtesting system"""
    
    def __init__(self, prices, returns):
        self.prices = prices
        self.returns = returns
    
    def run_backtest(self, weights, rebalance_freq='M', transaction_cost=0.001):
        """Run backtest with rebalancing"""
        initial_value = 10000
        portfolio_value = initial_value
        portfolio_values = [portfolio_value]
        dates = []
        
        if isinstance(weights, dict):
            weight_array = np.array([weights[ticker] for ticker in self.returns.columns])
        else:
            weight_array = weights
        
        if rebalance_freq == 'M':
            rebalance_dates = self.returns.resample('M').last().index
        elif rebalance_freq == 'Q':
            rebalance_dates = self.returns.resample('Q').last().index
        else:
            rebalance_dates = self.returns.resample('Y').last().index
        
        holdings = weight_array * portfolio_value / self.prices.iloc[0].values
        
        for date in self.returns.index[1:]:
            current_prices = self.prices.loc[date].values
            portfolio_value = np.sum(holdings * current_prices)
            portfolio_values.append(portfolio_value)
            dates.append(date)
            
            if date in rebalance_dates:
                target_value = weight_array * portfolio_value
                target_holdings = target_value / current_prices
                
                trades = target_holdings - holdings
                trade_costs = np.sum(np.abs(trades) * current_prices) * transaction_cost
                portfolio_value -= trade_costs
                
                holdings = target_holdings
        
        results = pd.DataFrame({
            'Date': dates,
            'Portfolio Value': portfolio_values[1:],
            'Returns': pd.Series(portfolio_values[1:]).pct_change().fillna(0)
        })
        results.set_index('Date', inplace=True)
        
        return results
    
    def calculate_backtest_metrics(self, portfolio_returns, benchmark_returns=None):
        """Calculate comprehensive backtest metrics"""
        metrics = {}
        
        metrics['Total Return'] = (portfolio_returns + 1).prod() - 1
        metrics['Annual Return'] = portfolio_returns.mean() * 252
        metrics['Annual Volatility'] = portfolio_returns.std() * np.sqrt(252)
        metrics['Sharpe Ratio'] = metrics['Annual Return'] / metrics['Annual Volatility'] if metrics['Annual Volatility'] > 0 else 0
        
        metrics['Max Drawdown'] = self.calculate_max_drawdown((portfolio_returns + 1).cumprod())
        metrics['VaR 95%'] = AdvancedRiskMetrics.calculate_var(portfolio_returns, 0.95)
        metrics['CVaR 95%'] = AdvancedRiskMetrics.calculate_cvar(portfolio_returns, 0.95)
        
        if benchmark_returns is not None:
            metrics['Alpha'] = self.calculate_alpha(portfolio_returns, benchmark_returns)
            metrics['Beta'] = self.calculate_beta(portfolio_returns, benchmark_returns)
            metrics['Tracking Error'] = (portfolio_returns - benchmark_returns).std() * np.sqrt(252)
            metrics['Information Ratio'] = metrics['Alpha'] / metrics['Tracking Error'] if metrics['Tracking Error'] > 0 else 0
        
        return metrics
    
    @staticmethod
    def calculate_max_drawdown(cumulative_values):
        running_max = np.maximum.accumulate(cumulative_values)
        drawdown = (cumulative_values - running_max) / running_max
        return drawdown.min()
    
    @staticmethod
    def calculate_alpha(portfolio_returns, benchmark_returns, risk_free=0.02):
        excess_portfolio = portfolio_returns - risk_free/252
        excess_benchmark = benchmark_returns - risk_free/252
        beta = np.cov(excess_portfolio, excess_benchmark)[0, 1] / np.var(excess_benchmark)
        alpha = excess_portfolio.mean() - beta * excess_benchmark.mean()
        return alpha * 252
    
    @staticmethod
    def calculate_beta(portfolio_returns, benchmark_returns):
        cov_matrix = np.cov(portfolio_returns, benchmark_returns)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
        return beta

# ============================================================================
# 5. STREAMLIT APP LAYOUT
# ============================================================================

def main():
    # Initialize data manager
    data_manager = PortfolioDataManager()
    
    # Sidebar configuration
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # Ticker input
        st.subheader("Portfolio Assets")
        default_tickers = "AAPL, MSFT, GOOGL, AMZN, TSLA, JPM, JNJ, V, WMT, DIS"
        tickers_input = st.text_area("Enter tickers (comma-separated):", 
                                   value=default_tickers,
                                   height=100)
        
        # Date range
        st.subheader("Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", 
                                      value=datetime.now() - timedelta(days=365*3),
                                      max_value=datetime.now() - timedelta(days=1))
        with col2:
            end_date = st.date_input("End Date", 
                                    value=datetime.now(),
                                    max_value=datetime.now())
        
        # Optimization settings
        st.subheader("Optimization Settings")
        optimization_method = st.selectbox(
            "Optimization Method",
            ["Mean-Variance (Sharpe)", 
             "Mean-Variance (Min Volatility)",
             "CVaR Optimization",
             "Hierarchical Risk Parity",
             "Black-Litterman",
             "Monte Carlo Simulation"]
        )
        
        risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 2.0, 0.1)
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            confidence_level = st.slider("Confidence Level for VaR/CVaR", 0.90, 0.99, 0.95, 0.01)
            transaction_cost = st.slider("Transaction Cost (%)", 0.0, 2.0, 0.1, 0.05)
            rebalance_freq = st.selectbox("Rebalancing Frequency", 
                                         ["Monthly", "Quarterly", "Annually"])
        
        # Optimize button
        if st.button("🚀 Optimize Portfolio", type="primary", use_container_width=True):
            st.session_state.optimize_clicked = True
        else:
            if 'optimize_clicked' not in st.session_state:
                st.session_state.optimize_clicked = False
    
    # Main content area
    st.title("📊 Advanced Portfolio Optimizer")
    st.markdown("---")
    
    # Parse tickers
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    
    if not tickers:
        st.warning("Please enter at least one ticker symbol.")
        return
    
    # Fetch data
    with st.spinner("Fetching market data..."):
        prices = data_manager.fetch_data(tickers, start_date, end_date)
    
    if prices.empty:
        st.error("Failed to fetch data. Please check ticker symbols and date range.")
        return
    
    # Display price chart
    st.subheader("Asset Price Performance")
    fig = go.Figure()
    for ticker in tickers:
        fig.add_trace(go.Scatter(
            x=prices.index,
            y=prices[ticker],
            mode='lines',
            name=ticker,
            hovertemplate=f'{ticker}<br>Date: %{{x}}<br>Price: $%{{y:.2f}}<extra></extra>'
        ))
    
    fig.update_layout(
        template='plotly_dark',
        height=400,
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    if st.session_state.optimize_clicked:
        # Calculate returns
        returns = data_manager.calculate_returns(prices)
        
        # Initialize optimizer
        optimizer = AdvancedPortfolioOptimizer(returns, prices)
        risk_free_decimal = risk_free_rate / 100
        
        # Perform optimization
        with st.spinner("Optimizing portfolio..."):
            if optimization_method == "Mean-Variance (Sharpe)":
                weights, performance = optimizer.optimize_mean_variance(
                    objective='sharpe', risk_free_rate=risk_free_decimal
                )
            elif optimization_method == "Mean-Variance (Min Volatility)":
                weights, performance = optimizer.optimize_mean_variance(objective='min_volatility')
            elif optimization_method == "CVaR Optimization":
                weights, performance = optimizer.optimize_cvar(confidence_level)
            elif optimization_method == "Hierarchical Risk Parity":
                weights, performance = optimizer.optimize_hrp()
            elif optimization_method == "Black-Litterman":
                market_caps = pd.Series(np.random.random(len(tickers)) * 1e9, index=tickers)
                weights, performance = optimizer.optimize_black_litterman(market_caps)
            else:  # Monte Carlo Simulation
                results, mc_weights = optimizer.monte_carlo_optimization()
                sharpe_ratios = results[:, 2]
                optimal_idx = np.argmax(sharpe_ratios)
                weights = dict(zip(tickers, mc_weights[optimal_idx]))
                portfolio_return = np.sum(optimizer.mu.values * mc_weights[optimal_idx])
                portfolio_vol = np.sqrt(mc_weights[optimal_idx].T @ optimizer.S.values @ mc_weights[optimal_idx])
                performance = (portfolio_return, portfolio_vol, sharpe_ratios[optimal_idx])
        
        # Display results in tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Portfolio Summary", 
            "🎯 Risk Analysis", 
            "🔄 Backtesting",
            "🔥 Stress Testing",
            "📊 Statistics"
        ])
        
        with tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Optimal Portfolio Allocation")
                
                # Create pie chart
                fig_pie = go.Figure(data=[go.Pie(
                    labels=list(weights.keys()),
                    values=list(weights.values()),
                    hole=0.3,
                    textinfo='label+percent',
                    marker=dict(colors=px.colors.qualitative.Set3)
                )])
                fig_pie.update_layout(
                    template='plotly_dark',
                    height=400
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.subheader("Performance Metrics")
                
                # Display performance metrics
                metrics_df = pd.DataFrame({
                    'Metric': ['Expected Annual Return', 
                              'Annual Volatility', 
                              'Sharpe Ratio'],
                    'Value': [f"{performance[0]*252:.2%}", 
                             f"{performance[1]*np.sqrt(252):.2%}", 
                             f"{performance[2]:.2f}"]
                })
                
                # Add VaR/CVaR
                weight_array = np.array([weights[ticker] for ticker in tickers])
                portfolio_returns = returns.dot(weight_array)
                
                var_95 = AdvancedRiskMetrics.calculate_var(portfolio_returns, 0.95)
                cvar_95 = AdvancedRiskMetrics.calculate_cvar(portfolio_returns, 0.95)
                
                var_row = pd.DataFrame({
                    'Metric': ['VaR (95%)', 'CVaR (95%)'],
                    'Value': [f"{var_95:.2%}", f"{cvar_95:.2%}"]
                })
                metrics_df = pd.concat([metrics_df, var_row], ignore_index=True)
                
                # Display as table
                st.dataframe(
                    metrics_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download weights
                weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
                csv = weights_df.to_csv()
                st.download_button(
                    label="📥 Download Portfolio Weights",
                    data=csv,
                    file_name="portfolio_weights.csv",
                    mime="text/csv"
                )
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Efficient Frontier")
                
                # Monte Carlo simulation
                mc_results, mc_weights = optimizer.monte_carlo_optimization(n_portfolios=5000)
                
                fig_ef = go.Figure()
                fig_ef.add_trace(go.Scatter(
                    x=mc_results[:, 1] * np.sqrt(252),
                    y=mc_results[:, 0] * 252,
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=mc_results[:, 2],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Sharpe Ratio")
                    ),
                    name='Random Portfolios',
                    hovertemplate='Risk: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: %{marker.color:.2f}'
                ))
                
                # Optimal portfolio
                portfolio_return = np.sum(optimizer.mu.values * weight_array) * 252
                portfolio_vol = np.sqrt(weight_array.T @ optimizer.S.values @ weight_array) * np.sqrt(252)
                
                fig_ef.add_trace(go.Scatter(
                    x=[portfolio_vol],
                    y=[portfolio_return],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='star'),
                    name='Optimized Portfolio'
                ))
                
                fig_ef.update_layout(
                    template='plotly_dark',
                    height=400,
                    xaxis_title="Annualized Volatility",
                    yaxis_title="Annualized Return",
                    title="Efficient Frontier with Monte Carlo Simulation"
                )
                st.plotly_chart(fig_ef, use_container_width=True)
            
            with col2:
                st.subheader("Risk Decomposition")
                
                # Calculate risk decomposition
                covariance_matrix = returns.cov() * 252
                risk_contrib = AdvancedRiskMetrics.calculate_risk_decomposition(
                    weight_array, covariance_matrix.values
                )
                
                fig_risk = go.Figure(data=[
                    go.Bar(
                        x=tickers,
                        y=risk_contrib * 100,
                        text=[f'{v:.1f}%' for v in risk_contrib * 100],
                        textposition='auto',
                        marker_color='lightblue'
                    )
                ])
                
                fig_risk.update_layout(
                    template='plotly_dark',
                    height=400,
                    title='Risk Contribution by Asset',
                    yaxis_title='Risk Contribution (%)'
                )
                st.plotly_chart(fig_risk, use_container_width=True)
                
                # VaR/CVaR distribution
                st.subheader("VaR/CVaR Analysis")
                
                var_99 = AdvancedRiskMetrics.calculate_var(portfolio_returns, 0.99)
                cvar_99 = AdvancedRiskMetrics.calculate_cvar(portfolio_returns, 0.99)
                
                var_data = pd.DataFrame({
                    'Confidence Level': ['95%', '99%'],
                    'VaR': [var_95, var_99],
                    'CVaR': [cvar_95, cvar_99]
                })
                var_data['VaR'] = var_data['VaR'].apply(lambda x: f"{x:.2%}")
                var_data['CVaR'] = var_data['CVaR'].apply(lambda x: f"{x:.2%}")
                
                st.dataframe(var_data, use_container_width=True, hide_index=True)
        
        with tab3:
            st.subheader("Backtest Results")
            
            # Run backtest
            backtester = PortfolioBacktester(prices, returns)
            rebalance_map = {"Monthly": "M", "Quarterly": "Q", "Annually": "Y"}
            backtest_results = backtester.run_backtest(
                weights, 
                rebalance_freq=rebalance_map[rebalance_freq],
                transaction_cost=transaction_cost/100
            )
            
            # Get benchmark
            benchmark_prices = data_manager.fetch_data(['SPY'], start_date, end_date)
            if not benchmark_prices.empty:
                benchmark_returns = data_manager.calculate_returns(benchmark_prices)
                benchmark_returns = benchmark_returns.reindex(backtest_results.index).fillna(0)
            else:
                benchmark_returns = None
            
            # Create backtest plot
            fig_backtest = go.Figure()
            
            fig_backtest.add_trace(go.Scatter(
                x=backtest_results.index,
                y=backtest_results['Portfolio Value'],
                mode='lines',
                name='Portfolio',
                line=dict(color='lightblue', width=2)
            ))
            
            if benchmark_returns is not None:
                benchmark_cumulative = (1 + benchmark_returns['SPY']).cumprod() * 10000
                fig_backtest.add_trace(go.Scatter(
                    x=benchmark_cumulative.index,
                    y=benchmark_cumulative.values,
                    mode='lines',
                    name='Benchmark (SPY)',
                    line=dict(color='gray', width=2, dash='dash')
                ))
            
            fig_backtest.update_layout(
                template='plotly_dark',
                height=400,
                title='Portfolio Backtest Performance',
                xaxis_title='Date',
                yaxis_title='Portfolio Value ($)'
            )
            st.plotly_chart(fig_backtest, use_container_width=True)
            
            # Performance metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Performance Metrics")
                metrics = backtester.calculate_backtest_metrics(
                    backtest_results['Returns'], 
                    benchmark_returns['SPY'] if benchmark_returns is not None else None
                )
                
                metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
                metrics_df['Value'] = metrics_df['Value'].apply(
                    lambda x: f"{x:.2%}" if isinstance(x, float) and abs(x) < 1 else f"{x:.2f}"
                )
                st.dataframe(metrics_df, use_container_width=True)
            
            with col2:
                st.subheader("Drawdown Analysis")
                
                # Calculate drawdown
                cumulative_returns = (1 + backtest_results['Returns']).cumprod()
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdown = (cumulative_returns - running_max) / running_max
                
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=drawdown.index,
                    y=drawdown * 100,
                    fill='tozeroy',
                    fillcolor='rgba(255, 0, 0, 0.3)',
                    line=dict(color='red', width=1),
                    name='Drawdown'
                ))
                
                fig_dd.update_layout(
                    template='plotly_dark',
                    height=300,
                    title='Portfolio Drawdown',
                    xaxis_title='Date',
                    yaxis_title='Drawdown (%)'
                )
                st.plotly_chart(fig_dd, use_container_width=True)
        
        with tab4:
            st.subheader("Stress Test Scenarios")
            
            # Perform stress testing
            stress_results = AdvancedRiskMetrics.stress_test_portfolio(weight_array, returns)
            
            # Display results
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Create radar chart
                metrics = ['Mean Return', 'Std Dev', 'VaR 95%', 'CVaR 95%']
                
                fig_radar = go.Figure()
                
                for scenario in stress_results.index:
                    values = [abs(stress_results.loc[scenario, metric]) for metric in metrics]
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values,
                        theta=metrics,
                        fill='toself',
                        name=scenario
                    ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, max([abs(stress_results[metric].max()) for metric in metrics])]
                        )),
                    template='plotly_dark',
                    height=500,
                    title='Stress Test Scenarios Comparison'
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            
            with col2:
                st.subheader("Scenario Details")
                
                # Format stress test results for display
                display_results = stress_results.copy()
                for col in display_results.columns:
                    display_results[col] = display_results[col].apply(
                        lambda x: f"{x:.2%}" if abs(x) < 1 else f"{x:.4f}"
                    )
                
                st.dataframe(
                    display_results,
                    use_container_width=True
                )
                
                st.info("""
                **Scenarios Explained:**
                - **Market Crash**: -10% returns
                - **High Volatility**: 2x normal volatility
                - **Bear Market**: -5% steady decline
                - **Flash Crash**: -15% sudden drop
                """)
        
        with tab5:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Asset Statistics")
                
                # Calculate statistics for each asset
                stats_data = []
                for asset in returns.columns:
                    asset_returns = returns[asset].dropna()
                    
                    stats_data.append({
                        'Asset': asset,
                        'Mean Return': asset_returns.mean() * 252,
                        'Annual Vol': asset_returns.std() * np.sqrt(252),
                        'Sharpe': (asset_returns.mean() * 252) / (asset_returns.std() * np.sqrt(252)) if asset_returns.std() > 0 else 0,
                        'Skewness': stats.skew(asset_returns),
                        'Kurtosis': stats.kurtosis(asset_returns),
                        'VaR 95%': AdvancedRiskMetrics.calculate_var(asset_returns, 0.95)
                    })
                
                stats_df = pd.DataFrame(stats_data)
                
                # Format for display
                display_stats = stats_df.copy()
                display_stats['Mean Return'] = display_stats['Mean Return'].apply(lambda x: f"{x:.2%}")
                display_stats['Annual Vol'] = display_stats['Annual Vol'].apply(lambda x: f"{x:.2%}")
                display_stats['Sharpe'] = display_stats['Sharpe'].apply(lambda x: f"{x:.2f}")
                display_stats['Skewness'] = display_stats['Skewness'].apply(lambda x: f"{x:.2f}")
                display_stats['Kurtosis'] = display_stats['Kurtosis'].apply(lambda x: f"{x:.2f}")
                display_stats['VaR 95%'] = display_stats['VaR 95%'].apply(lambda x: f"{x:.2%}")
                
                st.dataframe(
                    display_stats,
                    use_container_width=True,
                    hide_index=True
                )
            
            with col2:
                st.subheader("Correlation Matrix")
                
                # Calculate correlation matrix
                corr_matrix = returns.corr()
                
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='RdBu',
                    zmid=0,
                    text=np.round(corr_matrix.values, 2),
                    texttemplate='%{text}',
                    textfont={"size": 10}
                ))
                
                fig_corr.update_layout(
                    template='plotly_dark',
                    height=400,
                    title='Asset Correlation Matrix'
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Returns distribution
                st.subheader("Returns Distribution")
                
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=portfolio_returns,
                    nbinsx=50,
                    name='Portfolio Returns',
                    marker_color='lightblue',
                    opacity=0.7
                ))
                
                # Add normal distribution
                x_range = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 100)
                pdf = stats.norm.pdf(x_range, portfolio_returns.mean(), portfolio_returns.std())
                pdf = pdf / pdf.max() * len(portfolio_returns) / 10
                
                fig_dist.add_trace(go.Scatter(
                    x=x_range,
                    y=pdf,
                    mode='lines',
                    line=dict(color='red', width=2),
                    name='Normal Distribution'
                ))
                
                fig_dist.update_layout(
                    template='plotly_dark',
                    height=300,
                    title='Portfolio Returns Distribution',
                    xaxis_title='Daily Returns',
                    yaxis_title='Frequency'
                )
                st.plotly_chart(fig_dist, use_container_width=True)
    
    else:
        # Display initial instructions
        st.info("""
        ## 📋 Instructions
        
        1. **Enter ticker symbols** in the sidebar (comma-separated, e.g., AAPL, MSFT, GOOGL)
        2. **Select date range** for historical data
        3. **Choose optimization method** based on your risk preference
        4. **Adjust risk-free rate** and other parameters as needed
        5. **Click 'Optimize Portfolio'** to generate optimal allocation
        
        ### Available Optimization Methods:
        - **Mean-Variance (Sharpe)**: Maximizes risk-adjusted returns
        - **Mean-Variance (Min Volatility)**: Minimizes portfolio volatility
        - **CVaR Optimization**: Focuses on tail risk minimization
        - **Hierarchical Risk Parity**: Uses clustering for risk diversification
        - **Black-Litterman**: Incorporates market views and equilibrium
        - **Monte Carlo Simulation**: Random portfolio generation for frontier
        """)
        
        # Show sample portfolios
        st.subheader("🎯 Sample Portfolios")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### Conservative
            - **Bonds**: 60%
            - **Stocks**: 30%
            - **Gold**: 10%
            *Low risk, stable returns*
            """)
        
        with col2:
            st.markdown("""
            ### Balanced
            - **Stocks**: 60%
            - **Bonds**: 30%
            - **REITs**: 10%
            *Moderate risk/return balance*
            """)
        
        with col3:
            st.markdown("""
            ### Aggressive
            - **Tech Stocks**: 70%
            - **Growth Stocks**: 20%
            - **Emerging Markets**: 10%
            *High risk, high potential returns*
            """)

if __name__ == "__main__":
    main()
