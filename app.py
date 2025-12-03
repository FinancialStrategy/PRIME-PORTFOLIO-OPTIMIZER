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
from pypfopt.hierarchical_portfolio import HRPOpt
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt import objective_functions

# ARCH for GARCH models
try:
    import arch
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

warnings.filterwarnings('ignore')

# ============================================================================
# 1. GLOBAL CONFIGURATION & STYLING
# ============================================================================
st.set_page_config(page_title="Enigma Institutional Terminal", layout="wide", page_icon="🏛️")

# Professional CSS: Dark Mode, Clean Lines, Financial Terminal Look
st.markdown("""
<style>
    /* Main Background */
    .reportview-container {background: #0e1117;}
    
    /* Metrics Cards */
    .metric-card-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 15px;
        margin-bottom: 20px;
    }
    .pro-card {
        background-color: #1e1e1e;
        border: 1px solid #333;
        border-radius: 4px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .metric-label {
        font-family: 'Roboto', sans-serif;
        font-size: 11px;
        text-transform: uppercase;
        color: #888;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    .metric-value {
        font-family: 'Roboto Mono', monospace;
        font-size: 22px;
        font-weight: 600;
        color: #eee;
    }
    .metric-pos { color: #00cc96; }
    .metric-neg { color: #ef553b; }
    
    /* Tables */
    div[data-testid="stTable"] {
        font-family: 'Roboto Mono', monospace;
        font-size: 12px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        font-size: 14px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e1e1e;
        border-top: 2px solid #00cc96;
        color: #00cc96;
    }
</style>
""", unsafe_allow_html=True)

# --- ASSET UNIVERSES ---

# 1. BIST 30 (Turkish Major Stocks)
BIST_30 = [
    'AKBNK.IS', 'ARCLK.IS', 'ASELS.IS', 'BIMAS.IS', 'DOHOL.IS', 'EKGYO.IS', 
    'ENKAI.IS', 'EREGL.IS', 'FROTO.IS', 'GARAN.IS', 'GUBRF.IS', 'HALKB.IS', 
    'HEKTS.IS', 'ISCTR.IS', 'KCHOL.IS', 'KOZAA.IS', 'KOZAL.IS', 'KRDMD.IS', 
    'PETKM.IS', 'PGSUS.IS', 'SAHOL.IS', 'SASA.IS', 'SISE.IS', 'TCELL.IS', 
    'THYAO.IS', 'TKFEN.IS', 'TOASO.IS', 'TSKB.IS', 'TUPRS.IS', 'YKBNK.IS'
]

# 2. Global Major Indices (20)
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
# 2. DATA MANAGEMENT MODULE (ROBUST)
# ============================================================================

class PortfolioDataManager:
    """Professional-grade financial data manager with error handling & caching"""
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def fetch_data(tickers, start_date, end_date):
        if not tickers:
            return pd.DataFrame(), {}
        try:
            if isinstance(tickers, str):
                tickers = [tickers]
            
            # Robust Download
            data = yf.download(tickers, start=start_date, end=end_date, progress=False, group_by='ticker', threads=False)
            
            prices = pd.DataFrame()
            ohlc_dict = {}

            # Handle Single vs Multi Ticker Structures
            if len(tickers) == 1:
                ticker = tickers[0]
                df = data
                if isinstance(data.columns, pd.MultiIndex):
                    try: df = data.xs(ticker, axis=1, level=0, drop_level=True)
                    except: pass
                
                # Check for Adj Close, then Close
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
            
            # Cleaning
            prices = prices.ffill().bfill()
            prices = prices.dropna(axis=1, how='all')
            return prices, ohlc_dict
            
        except Exception as e:
            st.error(f"Data Pipeline Error: {str(e)}")
            return pd.DataFrame(), {}

    @staticmethod
    def calculate_returns(prices, method='log'):
        if method == 'log':
            return np.log(prices / prices.shift(1)).dropna()
        else:
            return prices.pct_change().dropna()
        
    @staticmethod
    @st.cache_data(ttl=3600 * 24)
    def get_market_caps(tickers):
        """Fetch Real Market Caps for Black-Litterman Priors"""
        mcaps = {}
        for t in tickers:
            try:
                mcaps[t] = yf.Ticker(t).info.get('marketCap', 10e9) # Default 10B
            except:
                mcaps[t] = 10e9
        return pd.Series(mcaps)

# ============================================================================
# 3. ADVANCED RISK METRICS ENGINE
# ============================================================================

class AdvancedRiskMetrics:
    """Institutional Risk Calculation Engine"""

    @staticmethod
    def calculate_metrics(returns, risk_free=0.02):
        ann_factor = 252
        
        # Absolute Metrics
        total_return = (1 + returns).prod() - 1
        cagr = (1 + total_return) ** (ann_factor / len(returns)) - 1
        volatility = returns.std() * np.sqrt(ann_factor)
        
        # Risk-Adjusted Metrics
        excess_returns = returns - (risk_free / ann_factor)
        sharpe = np.sqrt(ann_factor) * excess_returns.mean() / returns.std()
        
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(ann_factor)
        sortino = np.sqrt(ann_factor) * excess_returns.mean() / downside_std if downside_std != 0 else 0
        
        # Drawdown Dynamics
        cum_returns = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        max_dd = drawdown.min()
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        # Tail Risk
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        return {
            "Total Return": total_return, "CAGR": cagr, "Volatility": volatility,
            "Sharpe Ratio": sharpe, "Sortino Ratio": sortino, "Max Drawdown": max_dd,
            "Calmar Ratio": calmar, "VaR 95%": var_95, "CVaR 95%": cvar_95
        }

    @staticmethod
    def calculate_garch_vol(returns):
        """GARCH(1,1) Volatility Forecasting"""
        if not HAS_ARCH: return None
        try:
            am = arch.arch_model(returns * 100, vol='Garch', p=1, q=1, dist='normal')
            res = am.fit(disp='off')
            return np.sqrt(res.forecast(horizon=1).variance.values[-1, 0]) / 100
        except:
            return returns.std()

    @staticmethod
    def calculate_comprehensive_risk_profile(returns, confidence_levels=[0.95, 0.99]):
        """Parametric, Historical, and Modified VaR/CVaR"""
        results = {}
        mu, sigma = returns.mean(), returns.std()
        skew, kurt = stats.skew(returns), stats.kurtosis(returns)
        
        for conf in confidence_levels:
            alpha = 1 - conf
            z = stats.norm.ppf(alpha)
            
            # Cornish-Fisher Expansion for Modified VaR (Fat Tail Adjustment)
            z_cf = z + (z**2 - 1)*skew/6 + (z**3 - 3*z)*kurt/24 - (2*z**3 - 5*z)*(skew**2)/36
            
            results[f"Parametric VaR ({int(conf*100)}%)"] = mu + z * sigma
            results[f"Historical VaR ({int(conf*100)}%)"] = np.percentile(returns, alpha * 100)
            results[f"Modified VaR ({int(conf*100)}%)"] = mu + z_cf * sigma
            results[f"CVaR ({int(conf*100)}%)"] = returns[returns <= np.percentile(returns, alpha * 100)].mean()
            
        return results, skew, kurt

# ============================================================================
# 4. PORTFOLIO OPTIMIZATION ENGINE
# ============================================================================

class AdvancedPortfolioOptimizer:
    def __init__(self, returns, prices):
        self.returns = returns
        self.prices = prices
        self.mu = expected_returns.mean_historical_return(prices)
        self.S = risk_models.sample_cov(prices)
    
    def optimize(self, method, risk_free_rate=0.02):
        ef = EfficientFrontier(self.mu, self.S)
        
        if method == 'Max Sharpe':
            weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
        elif method == 'Min Volatility':
            weights = ef.min_volatility()
        elif method == 'Efficient Risk':
            avg_vol = np.sqrt(np.diag(self.S)).mean()
            weights = ef.efficient_risk(target_volatility=avg_vol)
            
        return ef.clean_weights(), ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)

    def optimize_hrp(self):
        hrp = HRPOpt(self.returns)
        hrp.optimize()
        return hrp.clean_weights(), hrp.portfolio_performance(verbose=False)
        
    def optimize_black_litterman(self, market_caps):
        delta = risk_models.black_litterman.market_implied_risk_aversion(self.prices)
        prior = risk_models.black_litterman.market_implied_prior_returns(market_caps, delta, self.S)
        bl = BlackLittermanModel(self.S, pi=prior, absolute_views=None)
        
        ef = EfficientFrontier(bl.bl_returns(), bl.bl_cov())
        weights = ef.max_sharpe()
        return ef.clean_weights(), ef.portfolio_performance(verbose=False)

# ============================================================================
# 5. DYNAMIC BACKTESTING ENGINE (RESTORED)
# ============================================================================

class PortfolioBacktester:
    """Simulates realistic portfolio performance with Rebalancing"""
    
    def __init__(self, prices, returns):
        self.prices = prices
        self.returns = returns
        
    def run_rebalancing_backtest(self, weights, initial_capital=100000, rebalance_freq='Q', cost_bps=10):
        """
        Runs a backtest where the portfolio is rebalanced to target weights 
        at the specified frequency (M=Monthly, Q=Quarterly, Y=Yearly).
        """
        # Align weights with available assets
        assets = self.returns.columns
        w_vector = np.array([weights.get(a, 0) for a in assets])
        
        # Setup Timeline
        if rebalance_freq == 'M': dates = self.returns.resample('M').last().index
        elif rebalance_freq == 'Q': dates = self.returns.resample('Q').last().index
        else: dates = self.returns.resample('Y').last().index
        
        # Simulation Loop
        portfolio_value = [initial_capital]
        current_capital = initial_capital
        current_weights = w_vector.copy()
        idx_dates = [self.returns.index[0]]
        
        # Iterate through daily returns
        cumulative_ret_series = []
        
        # Simplified vector approach for speed in Streamlit
        # 1. Calculate Daily Returns of the strategy (Drifting weights)
        # This is an approximation. For full path dependency, we need a slow loop.
        # Given Streamlit constraints, we use a hybrid approach.
        
        # Create a series of target weights re-aligned at rebalance dates
        reb_weights = pd.DataFrame(index=self.returns.index, columns=assets)
        
        # Fill target weights at rebalance dates
        for d in dates:
            if d in reb_weights.index:
                reb_weights.loc[d] = w_vector
        
        # Forward fill weights (Drift logic handled by returns compounding)
        # Actually, standard vector backtest:
        strat_returns = self.returns.dot(w_vector) # This is daily rebalancing (too optimistic)
        
        # Let's do the proper iterative loop for accuracy
        cash = initial_capital
        holdings = (cash * w_vector) / self.prices.iloc[0].values
        
        history_val = []
        history_dates = []
        
        for date, price_row in self.prices.iterrows():
            # Mark to Market
            val = np.sum(holdings * price_row.values)
            
            # Rebalance Check
            if date in dates:
                # Transaction Costs
                target_holdings = (val * w_vector) / price_row.values
                turnover = np.sum(np.abs(target_holdings - holdings)) * price_row.values
                cost = np.sum(turnover) * (cost_bps / 10000)
                val -= cost
                
                # Re-set holdings
                holdings = (val * w_vector) / price_row.values
            
            history_val.append(val)
            history_dates.append(date)
            
        return pd.Series(history_val, index=history_dates)

# ============================================================================
# 6. STREAMLIT APPLICATION LOGIC
# ============================================================================

# --- SIDEBAR ---
st.sidebar.header("🔧 Configuration")
ticker_lists = {"US Defaults": US_DEFAULTS, "BIST 30 (Turkey)": BIST_30, "Global Indices": GLOBAL_INDICES}
selected_list = st.sidebar.selectbox("Asset Universe", list(ticker_lists.keys()))
available_tickers = ticker_lists[selected_list]
custom_tickers = st.sidebar.text_input("Custom Tickers (comma sep)", value="")
if custom_tickers: available_tickers = list(set(available_tickers + [t.strip().upper() for t in custom_tickers.split(',')]))
selected_tickers = st.sidebar.multiselect("Select Assets", available_tickers, default=available_tickers[:5])

st.sidebar.subheader("Parameters")
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365*3))
end_date = st.sidebar.date_input("End Date", datetime.now())
rf_rate = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 10.0, 2.0, 0.1) / 100
method = st.sidebar.selectbox("Optimization Model", ['Max Sharpe', 'Min Volatility', 'Hierarchical Risk Parity (HRP)', 'Black-Litterman', 'Equal Weight'])
rebal_freq = st.sidebar.selectbox("Backtest Rebalancing", ['Quarterly', 'Monthly', 'Yearly'])

run_btn = st.sidebar.button("🚀 INITIATE ANALYSIS", type="primary")

# --- MAIN EXECUTION ---
if run_btn:
    with st.spinner('Initializing Quantitative Engines...'):
        # A. Data Ingestion
        data_mgr = PortfolioDataManager()
        prices, ohlc_data = data_mgr.fetch_data(selected_tickers, start_date, end_date)
        
        if prices.empty:
            st.error("No valid data found. Check Tickers.")
        else:
            returns = data_mgr.calculate_returns(prices)
            optimizer = AdvancedPortfolioOptimizer(returns, prices)
            
            # B. Optimization Logic
            if method == 'Hierarchical Risk Parity (HRP)': weights, perf = optimizer.optimize_hrp()
            elif method == 'Black-Litterman': 
                mcaps = data_mgr.get_market_caps(selected_tickers)
                weights, perf = optimizer.optimize_black_litterman(mcaps)
            elif method == 'Equal Weight':
                weights = {t: 1.0/len(selected_tickers) for t in selected_tickers}
                w_arr = np.array(list(weights.values()))
                ret = np.sum(returns.mean() * w_arr) * 252
                vol = np.sqrt(np.dot(w_arr.T, np.dot(returns.cov() * 252, w_arr)))
                perf = (ret, vol, (ret - rf_rate)/vol)
            else: 
                weights, perf = optimizer.optimize(method, rf_rate)

            # C. Dynamic Backtesting (The Heavy Lifting)
            backtester = PortfolioBacktester(prices, returns)
            freq_map = {'Quarterly':'Q', 'Monthly':'M', 'Yearly':'Y'}
            portfolio_equity_curve = backtester.run_rebalancing_backtest(weights, rebalance_freq=freq_map[rebal_freq])
            port_rets = portfolio_equity_curve.pct_change().dropna()
            
            # D. Risk Calculation
            risk_metrics = AdvancedRiskMetrics.calculate_metrics(port_rets, rf_rate)
            var_profile, skew, kurt = AdvancedRiskMetrics.calculate_comprehensive_risk_profile(port_rets)

            # --- VISUALIZATION TABS ---
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "🏛️ Executive Tearsheet", "📈 Efficient Frontier", "📉 Dynamic Backtest", 
                "🕯️ OHLC Analysis", "🌪️ Stress Test", "⚠️ Advanced VaR"
            ])

            # TAB 1: EXECUTIVE SUMMARY (Hedge Fund Style)
            with tab1:
                st.markdown("### 🏛️ Strategy Performance Attribution")
                
                # KPI Grid
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
                kpi(k4, "Max DD", f"{risk_metrics['Max Drawdown']:.2%}", "#ef553b")
                kpi(k5, "Sortino", f"{risk_metrics['Sortino Ratio']:.2f}")

                # Charts: Equity Curve & Allocation
                c_main, c_pie = st.columns([2, 1])
                with c_main:
                    # Rolling Sharpe
                    roll_sharpe = port_rets.rolling(126).apply(lambda x: (x.mean()*252 - rf_rate)/(x.std()*np.sqrt(252)))
                    fig_roll = go.Figure()
                    fig_roll.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe, fill='tozeroy', line=dict(color='#00cc96', width=1), name='Rolling Sharpe (6M)'))
                    fig_roll.update_layout(title="Rolling Risk-Adjusted Return (6M)", height=350, template="plotly_dark")
                    st.plotly_chart(fig_roll, use_container_width=True)
                
                with c_pie:
                    # Separated Labels Wheel Chart
                    w_s = pd.Series(weights).sort_values(ascending=False)
                    w_s = w_s[w_s > 0.001]
                    fig_pie = go.Figure(data=[go.Pie(labels=w_s.index, values=w_s.values, hole=0.6, textinfo='label+percent', textposition='outside', marker=dict(colors=px.colors.qualitative.Bold))])
                    fig_pie.update_layout(title="Target Allocation", height=350, showlegend=False, template="plotly_dark")
                    st.plotly_chart(fig_pie, use_container_width=True)

                # Heatmap & Drawdown
                c_heat, c_dd = st.columns([1.5, 1])
                with c_heat:
                    st.markdown("**Monthly Returns Heatmap**")
                    m_ret = port_rets.resample('M').apply(lambda x: (1+x).prod()-1)
                    m_df = pd.DataFrame(m_ret, columns=['Ret'])
                    m_df['Y'] = m_df.index.year; m_df['M'] = m_df.index.strftime('%b')
                    piv = m_df.pivot(index='Y', columns='M', values='Ret')
                    piv = piv.reindex(columns=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
                    fig_heat = px.imshow(piv, color_continuous_scale="RdYlGn", text_auto='.1%', aspect="auto")
                    fig_heat.update_layout(height=300, template="plotly_dark")
                    st.plotly_chart(fig_heat, use_container_width=True)

                with c_dd:
                    st.markdown("**Worst 5 Daily Losses**")
                    worst = port_rets.nsmallest(5)
                    dd_df = pd.DataFrame({"Date": worst.index.strftime('%Y-%m-%d'), "Loss": worst.values})
                    dd_df['Loss'] = dd_df['Loss'].apply(lambda x: f"{x:.2%}")
                    st.table(dd_df)

            # TAB 2: EFFICIENT FRONTIER
            with tab2:
                st.subheader("Efficient Frontier & Monte Carlo")
                n_sims = 1500
                all_weights = np.zeros((n_sims, len(selected_tickers)))
                ret_arr, vol_arr, sharpe_arr = np.zeros(n_sims), np.zeros(n_sims), np.zeros(n_sims)
                
                for i in range(n_sims):
                    w = np.random.random(len(selected_tickers)); w /= np.sum(w)
                    all_weights[i,:] = w
                    ret_arr[i] = np.sum(optimizer.mu * w)
                    vol_arr[i] = np.sqrt(np.dot(w.T, np.dot(optimizer.S, w)))
                    sharpe_arr[i] = (ret_arr[i] - rf_rate) / vol_arr[i]
                
                fig_ef = go.Figure()
                fig_ef.add_trace(go.Scatter(x=vol_arr, y=ret_arr, mode='markers', marker=dict(color=sharpe_arr, colorscale='Viridis', showscale=True), name='Simulations'))
                fig_ef.add_trace(go.Scatter(x=[perf[1]], y=[perf[0]], mode='markers', marker=dict(color='red', size=15, symbol='star'), name='Optimal'))
                fig_ef.update_layout(xaxis_title="Volatility", yaxis_title="Return", height=600, template="plotly_dark")
                st.plotly_chart(fig_ef, use_container_width=True)

            # TAB 3: BACKTEST
            with tab3:
                st.subheader(f"Dynamic Backtest ({rebal_freq} Rebalancing)")
                
                # Benchmark Calculation
                eq_weights = np.array([1/len(selected_tickers)] * len(selected_tickers))
                # Simple static benchmark for comparison
                bench_ret = returns.dot(eq_weights)
                bench_curve = (1 + bench_ret).cumprod() * 100000
                
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=portfolio_equity_curve.index, y=portfolio_equity_curve, name="Active Strategy", line=dict(color='#00cc96', width=2)))
                fig_bt.add_trace(go.Scatter(x=bench_curve.index, y=bench_curve, name="Equal Weight Index", line=dict(color='#888', dash='dash')))
                fig_bt.update_layout(title="Portfolio Value ($100k Initial)", template="plotly_dark", height=500)
                st.plotly_chart(fig_bt, use_container_width=True)
                
                # Underwater Plot
                running_max = portfolio_equity_curve.cummax()
                drawdown = (portfolio_equity_curve - running_max) / running_max
                fig_dd = px.area(drawdown, title="Drawdown Profile", color_discrete_sequence=['#ef553b'])
                fig_dd.update_layout(template="plotly_dark", height=300)
                st.plotly_chart(fig_dd, use_container_width=True)

            # TAB 4: OHLC
            with tab4:
                t_sel = st.selectbox("View Asset", selected_tickers)
                if t_sel in ohlc_data:
                    d = ohlc_data[t_sel]
                    fig = go.Figure(data=[go.Candlestick(x=d.index, open=d['Open'], high=d['High'], low=d['Low'], close=d['Close'])])
                    fig.update_layout(title=f"{t_sel} Price Action", template="plotly_dark", height=600)
                    st.plotly_chart(fig, use_container_width=True)

            # TAB 5: STRESS TEST
            with tab5:
                st.subheader("Macro Stress Test Scenarios")
                scenarios = {'2008 GFC (-40%)': -0.40, 'Covid-19 (-30%)': -0.30, 'Rate Shock (-10%)': -0.10, 'Melt Up (+20%)': 0.20}
                res_stress = []
                
                # Calculate Beta against Equal Weight Benchmark
                bench_daily = returns.mean(axis=1)
                beta = np.cov(port_rets, bench_daily)[0][1] / np.var(bench_daily)
                
                for n, s in scenarios.items():
                    imp = s * beta
                    pnl = 100000 * imp
                    res_stress.append({"Scenario": n, "Market": f"{s:.0%}", "Portfolio Impact": f"{imp:.2%}", "Est PnL": f"${pnl:,.0f}"})
                st.table(pd.DataFrame(res_stress))

            # TAB 6: ADVANCED VAR
            with tab6:
                st.subheader("⚠️ Comprehensive Risk Profile")
                col_dist, col_stat = st.columns([2, 1])
                
                with col_dist:
                    fig_v = go.Figure()
                    fig_v.add_trace(go.Histogram(x=port_rets, nbinsx=100, histnorm='probability density', name='Returns', marker_color='#1f77b4', opacity=0.6))
                    x = np.linspace(port_rets.min(), port_rets.max(), 100)
                    fig_v.add_trace(go.Scatter(x=x, y=stats.norm.pdf(x, port_rets.mean(), port_rets.std()), mode='lines', name='Normal Dist', line=dict(color='white', dash='dash')))
                    fig_v.add_vline(x=var_profile['Modified VaR (95%)'], line_dash="dot", line_color="red", annotation_text="Mod VaR 95%")
                    fig_v.update_layout(title="Return Distribution vs Normal Curve", template="plotly_dark", height=450)
                    st.plotly_chart(fig_v, use_container_width=True)
                
                with col_stat:
                    st.markdown("### Tail Risk Metrics")
                    st.metric("Skewness", f"{skew:.3f}")
                    st.metric("Kurtosis", f"{kurt:.3f}")
                    st.markdown("---")
                    v_df = pd.DataFrame.from_dict(var_profile, orient='index', columns=['Value'])
                    v_df['Value'] = v_df['Value'].apply(lambda x: f"{x:.2%}")
                    st.table(v_df)

else:
    st.info("👈 Configure your Institutional Portfolio in the Sidebar.")
