import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
from pytz import timezone
import logging

# Import custom modules
from config import *
from dhan_client import DhanClient, DhanWebSocket
from indicators import TechnicalIndicators, VolumeOrderBlocks, PivotDetector
from market_bias import MarketBiasCalculator, RangeDetector
from options_analyzer import OptionsAnalyzer
from database import SupabaseDB
from telegram_bot import TelegramBot
from trade_manager import TradeManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Nifty Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .signal-bull {
        background-color: #089981;
        padding: 10px;
        border-radius: 5px;
        color: white;
        font-weight: bold;
    }
    .signal-bear {
        background-color: #f23645;
        padding: 10px;
        border-radius: 5px;
        color: white;
        font-weight: bold;
    }
    .signal-neutral {
        background-color: #787B86;
        padding: 10px;
        border-radius: 5px;
        color: white;
        font-weight: bold;
    }
    .stAlert {
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# INITIALIZATION
# ============================================================================

@st.cache_resource
def init_clients():
    """Initialize API clients and connections"""
    try:
        # Initialize DhanHQ client
        dhan = DhanClient(DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN)
        
        # Initialize database
        db = SupabaseDB(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL else None
        
        # Initialize Telegram bot
        telegram = TelegramBot(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        
        # Initialize trade manager
        config_dict = {k: v for k, v in globals().items() if k.isupper()}
        trade_manager = TradeManager(dhan, db, telegram, config_dict) if db else None
        
        return dhan, db, telegram, trade_manager
    
    except Exception as e:
        st.error(f"Initialization error: {e}")
        return None, None, None, None


# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================

@st.cache_data(ttl=60)
def fetch_stock_data(dhan_client, stocks_config):
    """Fetch real-time stock data"""
    try:
        stock_ids = [data['security_id'] for data in stocks_config.values() if data['weight'] > 0]
        
        # Fetch current prices
        response = dhan_client.get_market_quote_ohlc("NSE_EQ", stock_ids)
        
        if 'error' in response:
            return {}
        
        stock_data = {}
        for symbol, config in stocks_config.items():
            if config['weight'] == 0:
                continue
            
            sec_id = config['security_id']
            if 'NSE_EQ' in response.get('data', {}) and sec_id in response['data']['NSE_EQ']:
                data = response['data']['NSE_EQ'][sec_id]
                
                current_price = data.get('last_price', 0)
                open_price = data['ohlc'].get('open', current_price)
                close_price = data['ohlc'].get('close', current_price)
                
                daily_change = current_price - close_price
                daily_change_pct = (daily_change / close_price * 100) if close_price > 0 else 0
                
                stock_data[symbol] = {
                    'security_id': sec_id,
                    'weight': config['weight'],
                    'price': current_price,
                    'open': open_price,
                    'close': close_price,
                    'daily_change': daily_change,
                    'daily_change_pct': daily_change_pct,
                    'tf1_change_pct': 0,  # Would need intraday data
                    'tf2_change_pct': 0   # Would need intraday data
                }
        
        return stock_data
    
    except Exception as e:
        logger.error(f"Error fetching stock data: {e}")
        return {}


@st.cache_data(ttl=300)
def fetch_historical_data(dhan_client, symbol, security_id, days_back=5):
    """Fetch historical intraday data"""
    try:
        to_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d %H:%M:%S')
        
        response = dhan_client.get_historical_intraday(
            security_id=security_id,
            exchange_segment="NSE_FNO" if "NIFTY" in symbol else "IDX_I",
            instrument="INDEX" if symbol == "NIFTY50" else "EQUITY",
            interval="5",
            from_date=from_date,
            to_date=to_date,
            oi=True
        )
        
        if 'error' in response or not response.get('open'):
            return None
        
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(response['timestamp'], unit='s'),
            'open': response['open'],
            'high': response['high'],
            'low': response['low'],
            'close': response['close'],
            'volume': response['volume']
        })
        
        df.set_index('timestamp', inplace=True)
        return df
    
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        return None


@st.cache_data(ttl=180)
def fetch_option_chain(dhan_client, underlying_scrip, underlying_seg, expiry):
    """Fetch option chain data"""
    try:
        response = dhan_client.get_option_chain(underlying_scrip, underlying_seg, expiry)
        return response
    except Exception as e:
        logger.error(f"Error fetching option chain: {e}")
        return None


@st.cache_data(ttl=300)
def fetch_expiry_list(dhan_client, underlying_scrip, underlying_seg):
    """Fetch expiry list"""
    try:
        response = dhan_client.get_expiry_list(underlying_scrip, underlying_seg)
        return response
    except Exception as e:
        logger.error(f"Error fetching expiry list: {e}")
        return None


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def calculate_all_indicators(df):
    """Calculate all technical indicators"""
    indicators = {}
    
    # VWAP
    indicators['vwap'] = TechnicalIndicators.calculate_vwap(df)
    
    # RSI
    indicators['rsi'] = TechnicalIndicators.calculate_rsi(df['close'], RSI_PERIOD)
    
    # MFI
    indicators['mfi'] = TechnicalIndicators.calculate_mfi(df, MFI_PERIOD)
    
    # DMI
    di_plus, di_minus, adx = TechnicalIndicators.calculate_dmi(df, DMI_PERIOD, DMI_SMOOTHING)
    indicators['di_plus'] = di_plus
    indicators['di_minus'] = di_minus
    indicators['adx'] = adx
    
    # ATR
    indicators['atr'] = TechnicalIndicators.calculate_atr(df, ATR_PERIOD)
    
    # EMAs (for order blocks and VOB)
    indicators['ema_fast'] = TechnicalIndicators.calculate_ema(df['close'], VOB_SENSITIVITY)
    indicators['ema_slow'] = TechnicalIndicators.calculate_ema(df['close'], VOB_LENGTH2)
    
    # VIDYA
    indicators['vidya'] = TechnicalIndicators.calculate_vidya(df['close'], VIDYA_LENGTH, VIDYA_MOMENTUM)
    
    # OBV
    indicators['obv'] = TechnicalIndicators.calculate_obv(df)
    indicators['obv_ma'] = TechnicalIndicators.calculate_sma(indicators['obv'], OBV_SMOOTHING)
    
    # Force Index
    indicators['force_index'] = TechnicalIndicators.calculate_force_index(df, FORCE_INDEX_LENGTH, FORCE_INDEX_SMOOTHING)
    
    # Volatility Ratio
    indicators['volatility_ratio'] = TechnicalIndicators.calculate_volatility_ratio(df, VOLATILITY_RATIO_LENGTH)
    
    # Volume ROC
    indicators['volume_roc'] = TechnicalIndicators.calculate_volume_roc(df['volume'], VOLUME_ROC_LENGTH)
    
    # Price ROC
    indicators['price_roc'] = TechnicalIndicators.calculate_price_roc(df['close'], 12)
    
    # Choppiness Index
    indicators['choppiness_index'] = TechnicalIndicators.calculate_choppiness_index(df, 14)
    
    return indicators


def analyze_market_bias(df, indicators, vob_data, stock_data, market_condition):
    """Perform complete market bias analysis"""
    config_dict = {k: v for k, v in globals().items() if k.isupper()}
    bias_calc = MarketBiasCalculator(config_dict)
    
    bias_result = bias_calc.calculate_bias(
        df=df,
        indicators=indicators,
        vob_data=vob_data,
        stock_data=stock_data,
        market_condition=market_condition
    )
    
    return bias_result


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_candlestick_chart(df, indicators, vob_data, range_data, title="NIFTY 50"):
    """Create interactive candlestick chart with indicators"""
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(title, 'Volume', 'RSI')
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#089981',
            decreasing_line_color='#f23645'
        ),
        row=1, col=1
    )
    
    # VWAP
    if 'vwap' in indicators:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=indicators['vwap'],
                name='VWAP',
                line=dict(color='#2962FF', width=2)
            ),
            row=1, col=1
        )
    
    # VIDYA
    if 'vidya' in indicators:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=indicators['vidya'],
                name='VIDYA',
                line=dict(color='#FF6D00', width=1, dash='dash')
            ),
            row=1, col=1
        )
    
    # VOB zones (bullish)
    if vob_data.get('bullish_vobs'):
        for vob in vob_data['bullish_vobs'][-5:]:  # Last 5 VOBs
            fig.add_shape(
                type="rect",
                x0=vob['timestamp'], x1=df.index[-1],
                y0=vob['lower'], y1=vob['upper'],
                fillcolor="green", opacity=0.2,
                line=dict(width=0),
                row=1, col=1
            )
    
    # VOB zones (bearish)
    if vob_data.get('bearish_vobs'):
        for vob in vob_data['bearish_vobs'][-5:]:  # Last 5 VOBs
            fig.add_shape(
                type="rect",
                x0=vob['timestamp'], x1=df.index[-1],
                y0=vob['lower'], y1=vob['upper'],
                fillcolor="red", opacity=0.2,
                line=dict(width=0),
                row=1, col=1
            )
    
    # Range boundaries
    if range_data['condition'] == "RANGE-BOUND":
        if range_data['range_high']:
            fig.add_hline(
                y=range_data['range_high'],
                line_dash="dash",
                line_color="red",
                annotation_text="Range High",
                row=1, col=1
            )
        
        if range_data['range_low']:
            fig.add_hline(
                y=range_data['range_low'],
                line_dash="dash",
                line_color="green",
                annotation_text="Range Low",
                row=1, col=1
            )
    
    # Volume
    colors = ['#089981' if df['close'].iloc[i] >= df['open'].iloc[i] else '#f23645' 
              for i in range(len(df))]
    
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            name='Volume',
            marker_color=colors
        ),
        row=2, col=1
    )
    
    # RSI
    if 'rsi' in indicators:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=indicators['rsi'],
                name='RSI',
                line=dict(color='purple', width=2)
            ),
            row=3, col=1
        )
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        hovermode='x unified'
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.2)')
    
    return fig


def create_bias_gauge(bias_pct, title="Market Bias"):
    """Create gauge chart for bias percentage"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=bias_pct,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#089981" if bias_pct >= 60 else "#f23645" if bias_pct <= 40 else "#787B86"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(242, 54, 69, 0.3)'},
                {'range': [40, 60], 'color': 'rgba(120, 123, 134, 0.3)'},
                {'range': [60, 100], 'color': 'rgba(8, 153, 129, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': bias_pct
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "white", 'family': "Arial"}
    )
    
    return fig


def display_stock_correlation_table(stock_data):
    """Display stock correlation table"""
    if not stock_data:
        st.warning("No stock data available")
        return
    
    # Prepare data
    table_data = []
    for symbol, data in stock_data.items():
        table_data.append({
            'Symbol': symbol,
            'Price': f"₹{data['price']:.2f}",
            'Change': f"{data['daily_change']:+.2f}",
            'Change %': f"{data['daily_change_pct']:+.2f}%",
            'Weight': f"{data['weight']:.2f}%"
        })
    
    df_display = pd.DataFrame(table_data)
    
    # Style the dataframe
    def color_change(val):
        if '+' in str(val):
            return 'background-color: rgba(8, 153, 129, 0.3)'
        elif '-' in str(val):
            return 'background-color: rgba(242, 54, 69, 0.3)'
        return ''
    
    styled_df = df_display.style.applymap(color_change, subset=['Change', 'Change %'])
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)


def display_options_chain_analysis(option_chain_data, expiry):
    """Display options chain analysis"""
    options_analyzer = OptionsAnalyzer(OPTIONS_WEIGHTS)
    
    underlying, df_summary, summary_stats = options_analyzer.analyze_option_chain(
        option_chain_data, expiry
    )
    
    if underlying is None or df_summary is None:
        st.error("Failed to analyze option chain")
        return
    
    # Display summary metrics
    st.subheader("📊 Options Chain Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Underlying Price", f"₹{underlying:.2f}")
    
    with col2:
        pcr_color = "normal" if summary_stats['overall_pcr'] > 1.2 else "inverse"
        st.metric("Overall PCR", f"{summary_stats['overall_pcr']:.2f}", 
                 delta_color=pcr_color)
    
    with col3:
        ce_change = summary_stats['total_ce_change'] / 100000
        st.metric("CALL ΔOI", f"{ce_change:+.1f}L", delta_color="inverse")
    
    with col4:
        pe_change = summary_stats['total_pe_change'] / 100000
        st.metric("PUT ΔOI", f"{pe_change:+.1f}L", delta_color="normal")
    
    # Display bias
    bias_color = "#089981" if summary_stats['options_bias'] == "BULLISH" else \
                 "#f23645" if summary_stats['options_bias'] == "BEARISH" else "#787B86"
    
    st.markdown(f"""
    <div style="background-color: {bias_color}; padding: 15px; border-radius: 5px; text-align: center;">
        <h3 style="color: white; margin: 0;">Options Bias: {summary_stats['options_bias']}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Display detailed table
    st.subheader("📋 Strike-wise Analysis")
    
    # Style function for PCR
    def color_pcr(val):
        try:
            v = float(val)
            if v > 1.5:
                return 'background-color: rgba(8, 153, 129, 0.5)'
            elif v < 0.7:
                return 'background-color: rgba(242, 54, 69, 0.5)'
            return ''
        except:
            return ''
    
    # Style function for verdict
    def color_verdict(val):
        if 'Strong Bullish' in str(val):
            return 'background-color: rgba(8, 153, 129, 0.7); color: white; font-weight: bold'
        elif 'Bullish' in str(val):
            return 'background-color: rgba(8, 153, 129, 0.4)'
        elif 'Strong Bearish' in str(val):
            return 'background-color: rgba(242, 54, 69, 0.7); color: white; font-weight: bold'
        elif 'Bearish' in str(val):
            return 'background-color: rgba(242, 54, 69, 0.4)'
        return ''
    
    # Style function for ATM
    def highlight_atm(row):
        return ['background-color: rgba(255, 215, 0, 0.3)' if row['Zone'] == 'ATM' else '' 
                for _ in row]
    
    # Display columns
    display_cols = ['Strike', 'Zone', 'Level', 'PCR', 'PCR_Signal', 'BiasScore', 'Verdict',
                   'lastPrice_CE', 'lastPrice_PE', 'openInterest_CE', 'openInterest_PE']
    
    df_display = df_summary[display_cols].copy()
    df_display.columns = ['Strike', 'Zone', 'Level', 'PCR', 'PCR Signal', 'Bias Score', 
                         'Verdict', 'CE Price', 'PE Price', 'CE OI', 'PE OI']
    
    styled_df = df_display.style\
        .applymap(color_pcr, subset=['PCR'])\
        .applymap(color_verdict, subset=['Verdict'])\
        .apply(highlight_atm, axis=1)
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Download button
    csv = df_summary.to_csv(index=False)
    st.download_button(
        label="📥 Download Full Analysis",
        data=csv,
        file_name=f"options_analysis_{expiry}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application"""
    
    # Title
    st.title("📈 NIFTY Trading System - Enhanced Pro")
    st.markdown("*Real-time market analysis with VOB detection, enhanced bias calculation, and options chain analysis*")
    
    # Initialize clients
    dhan, db, telegram, trade_manager = init_clients()
    
    if not dhan:
        st.error("❌ Failed to initialize DhanHQ client. Please check credentials.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Symbol selection
        symbol = st.selectbox(
            "Select Symbol",
            ["NIFTY50", "BANKNIFTY"],
            index=0
        )
        
        security_id = NIFTY_SECURITY_ID if symbol == "NIFTY50" else BANKNIFTY_SECURITY_ID
        
        # Timeframe
        timeframe = st.selectbox(
            "Timeframe",
            ["5", "15", "60"],
            index=0,
            format_func=lambda x: f"{x} minutes"
        )
        
        # Auto-refresh
        auto_refresh = st.checkbox("Auto Refresh", value=True)
        refresh_interval = st.slider("Refresh Interval (seconds)", 10, 300, 60) if auto_refresh else None
        
        # Trading mode
        st.markdown("---")
        st.subheader("🎮 Trading Mode")
        trading_mode = st.radio(
            "Mode",
            ["Analysis Only", "Paper Trading", "Live Trading"],
            index=0
        )
        
        if trading_mode == "Live Trading":
            st.warning("⚠️ Live trading enabled. Orders will be placed automatically.")
            
            quantity = st.number_input("Quantity per trade", min_value=1, value=50, step=1)
            use_super_order = st.checkbox("Use Super Order (Entry+SL+Target)", value=True)
        
        # Manual refresh button
        st.markdown("---")
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Market Analysis",
        "📈 Charts",
        "🎯 Options Chain",
        "💼 Positions",
        "📉 Performance"
    ])
    
    # ========================================================================
    # TAB 1: MARKET ANALYSIS
    # ========================================================================
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"🎯 {symbol} Analysis")
            
            with st.spinner("Fetching market data..."):
                # Fetch data
                df = fetch_historical_data(dhan, symbol, security_id, days_back=5)
                
                if df is None or df.empty:
                    st.error("Failed to fetch historical data")
                    st.stop()
                
                stock_data = fetch_stock_data(dhan, STOCKS_CONFIG)
                
                # Calculate indicators
                indicators = calculate_all_indicators(df)
                
                # VOB detection
                vob_detector = VolumeOrderBlocks(VOB_SENSITIVITY)
                bullish_vobs, bearish_vobs = vob_detector.detect_vobs(df)
                
                current_price = df['close'].iloc[-1]
                current_low = df['low'].iloc[-1]
                current_high = df['high'].iloc[-1]
                atr = indicators['atr'].iloc[-1]
                
                vob_detector.update_active_vobs(bullish_vobs, bearish_vobs, current_price, atr)
                
                bull_touch, bear_touch, bull_level, bear_level = vob_detector.check_vob_touch(
                    current_price, current_low, current_high
                )
                
                vob_data = {
                    'bullish_vobs': vob_detector.lower_vobs,
                    'bearish_vobs': vob_detector.upper_vobs,
                    'bull_touch': bull_touch,
                    'bear_touch': bear_touch,
                    'bull_level': bull_level,
                    'bear_level': bear_level
                }
                
                # Range detection
                config_dict = {k: v for k, v in globals().items() if k.isupper()}
                range_detector = RangeDetector(config_dict)
                range_data = range_detector.detect_condition(
                    df,
                    indicators['ema_fast'],
                    indicators['ema_slow'],
                    indicators['atr']
                )
                
                # Market bias calculation
                bias_result = analyze_market_bias(
                    df, indicators, vob_data, stock_data, range_data['condition']
                )
            
            # Display current price
            prev_close = df['close'].iloc[-2]
            price_change = current_price - prev_close
            price_change_pct = (price_change / prev_close * 100)
            
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                st.metric(
                    "Current Price",
                    f"₹{current_price:.2f}",
                    f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
                )
            
            with col_b:
                st.metric(
                    "Market Condition",
                    range_data['condition'],
                    f"{range_data['movement_quality']}"
                )
            
            with col_c:
                st.metric(
                    "ATR",
                    f"₹{atr:.2f}",
                    f"Vol: {range_data['volatility_regime']}"
                )
            
            with col_d:
                st.metric(
                    "Mode",
                    bias_result['mode'],
                    "🔄" if bias_result['divergence_detected'] else "✓"
                )
            
            # Market bias display
            st.markdown("### 🎯 Enhanced Market Bias")
            
            bias_color_map = {
                "BULLISH": "#089981",
                "BEARISH": "#f23645",
                "NEUTRAL": "#787B86"
            }
            
            bias_color = bias_color_map.get(bias_result['bias'], "#787B86")
            
            st.markdown(f"""
            <div style="background-color: {bias_color}; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="color: white; margin: 0;">{bias_result['bias']}</h2>
                <p style="color: white; margin: 5px 0;">Confidence: {bias_result['confidence']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            col_x, col_y = st.columns(2)
            
            with col_x:
                fig_bull = create_bias_gauge(bias_result['bullish_pct'], "Bullish Strength")
                st.plotly_chart(fig_bull, use_container_width=True)
            
            with col_y:
                fig_bear = create_bias_gauge(bias_result['bearish_pct'], "Bearish Strength")
                st.plotly_chart(fig_bear, use_container_width=True)
            
            # Signal breakdown
            st.markdown("### 📊 Signal Breakdown")
            
            col_i, col_ii, col_iii = st.columns(3)
            
            with col_i:
                st.markdown("**Fast Signals**")
                fast = bias_result['fast_signals']
                st.write(f"🟢 Bull: {fast['bull']}/{fast['total']}")
                st.write(f"🔴 Bear: {fast['bear']}/{fast['total']}")
            
            with col_ii:
                st.markdown("**Medium Signals**")
                medium = bias_result['medium_signals']
                st.write(f"🟢 Bull: {medium['bull']}/{medium['total']}")
                st.write(f"🔴 Bear: {medium['bear']}/{medium['total']}")
            
            with col_iii:
                st.markdown("**Slow Signals**")
                slow = bias_result['slow_signals']
                st.write(f"🟢 Bull: {slow['bull']}/{slow['total']}")
                st.write(f"🔴 Bear: {slow['bear']}/{slow['total']}")
            
            # VOB status
            if bull_touch or bear_touch:
                st.markdown("### 🔷 VOB Status")
                
                if bull_touch:
                    st.success(f"🔷 **Bullish VOB Touched** at ₹{bull_level:.2f}")
                
                if bear_touch:
                    st.error(f"🔶 **Bearish VOB Touched** at ₹{bear_level:.2f}")
            
            # Alerts
            if bias_result['divergence_detected']:
                st.warning("⚠️ **DIVERGENCE DETECTED** - Potential reversal warning!")
            
            if range_data['range_breakout_long']:
                st.success("📈 **RANGE BREAKOUT LONG** - Strong upward momentum!")
            
            if range_data['range_breakout_short']:
                st.error("📉 **RANGE BREAKOUT SHORT** - Strong downward momentum!")
        
        with col2:
            st.subheader("📋 Stock Correlations")
            display_stock_correlation_table(stock_data)
            
            # Key indicators
            st.markdown("### 📈 Key Indicators")
            
            st.metric("RSI", f"{indicators['rsi'].iloc[-1]:.1f}")
            st.metric("ADX", f"{indicators['adx'].iloc[-1]:.1f}")
            st.metric("MFI", f"{indicators['mfi'].iloc[-1]:.1f}")
            
            # Market breadth
            if stock_data:
                bullish_stocks = sum(1 for s in stock_data.values() if s['daily_change_pct'] > 0)
                total_stocks = len([s for s in stock_data.values() if s['weight'] > 0])
                breadth_pct = (bullish_stocks / total_stocks * 100) if total_stocks > 0 else 0
                
                st.metric(
                    "Market Breadth",
                    f"{breadth_pct:.0f}%",
                    f"{bullish_stocks}/{total_stocks} stocks up"
                )
    
    # ========================================================================
    # TAB 2: CHARTS
    # ========================================================================
    
    with tab2:
        st.subheader(f"📈 {symbol} Chart Analysis")
        
        if 'df' in locals() and df is not None:
            fig = create_candlestick_chart(df, indicators, vob_data, range_data, symbol)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Load data from Market Analysis tab first")
    
    # ========================================================================
    # TAB 3: OPTIONS CHAIN
    # ========================================================================
    
    with tab3:
        st.subheader("🎯 NIFTY Options Chain Analysis")
        
        # Fetch expiry list
        expiry_data = fetch_expiry_list(dhan, NIFTY_UNDERLYING_SCRIP, NIFTY_UNDERLYING_SEG)
        
        if expiry_data and 'data' in expiry_data:
            expiry_dates = expiry_data['data']
            
            # Expiry selector
            selected_expiry = st.selectbox(
                "Select Expiry",
                expiry_dates,
                index=0
            )
            
            # Fetch and analyze
            with st.spinner("Analyzing option chain..."):
                option_chain_data = fetch_option_chain(
                    dhan, NIFTY_UNDERLYING_SCRIP, NIFTY_UNDERLYING_SEG, selected_expiry
                )
                
                if option_chain_data:
                    display_options_chain_analysis(option_chain_data, selected_expiry)
                else:
                    st.error("Failed to fetch option chain data")
        else:
            st.error("Failed to fetch expiry list")
    
    # ========================================================================
    # TAB 4: POSITIONS
    # ========================================================================
    
    with tab4:
        st.subheader("💼 Open Positions")
        
        if db and trade_manager:
            open_positions = trade_manager.get_open_positions_summary()
            
            if open_positions['count'] > 0:
                for position in open_positions['positions']:
                    with st.expander(f"{position['symbol']} - {position['direction']} ({position['quantity']} qty)"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Entry Price", f"₹{position['entry_price']:.2f}")
                        
                        with col2:
                            st.metric("Stop Loss", f"₹{position['stop_loss']:.2f}")
                        
                        with col3:
                            st.metric("Target", f"₹{position['target']:.2f}")
                        
                        # Manual close button
                        if st.button(f"Close Position {position['order_id']}", key=position['order_id']):
                            # Close logic here
                            st.success("Position closed (manual)")
            else:
                st.info("No open positions")
        else:
            st.warning("Database not configured. Cannot track positions.")
    
    # ========================================================================
    # TAB 5: PERFORMANCE
    # ========================================================================
    
    with tab5:
        st.subheader("📉 Performance Analytics")
        
        if db:
            # Date range selector
            col1, col2 = st.columns(2)
            
            with col1:
                days_back = st.slider("Days to show", 7, 90, 30)
            
            # Fetch trade history
            trade_history = db.get_trade_history(symbol, days_back=days_back)
            
            if not trade_history.empty:
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                closed_trades = trade_history[trade_history['status'] == 'CLOSED']
                
                if not closed_trades.empty:
                    total_trades = len(closed_trades)
                    winning_trades = len(closed_trades[closed_trades['pnl'] > 0])
                    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                    total_pnl = closed_trades['pnl'].sum()
                    
                    with col1:
                        st.metric("Total Trades", total_trades)
                    
                    with col2:
                        st.metric("Win Rate", f"{win_rate:.1f}%")
                    
                    with col3:
                        st.metric("Total P&L", f"₹{total_pnl:,.2f}")
                    
                    with col4:
                        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
                        st.metric("Avg P&L", f"₹{avg_pnl:,.2f}")
                    
                    # P&L chart
                    st.markdown("### 📊 P&L Over Time")
                    
                    closed_trades_sorted = closed_trades.sort_values('entry_time')
                    closed_trades_sorted['cumulative_pnl'] = closed_trades_sorted['pnl'].cumsum()
                    
                    fig_pnl = go.Figure()
                    
                    fig_pnl.add_trace(go.Scatter(
                        x=pd.to_datetime(closed_trades_sorted['entry_time']),
                        y=closed_trades_sorted['cumulative_pnl'],
                        mode='lines+markers',
                        name='Cumulative P&L',
                        line=dict(color='#089981', width=2)
                    ))
                    
                    fig_pnl.update_layout(
                        title="Cumulative P&L",
                        template='plotly_dark',
                        height=400,
                        xaxis_title="Date",
                        yaxis_title="P&L (₹)"
                    )
                    
                    st.plotly_chart(fig_pnl, use_container_width=True)
                    
                    # Trade details table
                    st.markdown("### 📋 Trade History")
                    
                    display_cols = ['entry_time', 'symbol', 'direction', 'entry_price', 
                                   'exit_price', 'quantity', 'pnl', 'pnl_pct', 'exit_reason']
                    
                    df_trades = closed_trades[display_cols].copy()
                    df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time']).dt.strftime('%Y-%m-%d %H:%M')
                    
                    st.dataframe(df_trades, use_container_width=True, hide_index=True)
                else:
                    st.info("No closed trades in selected period")
            else:
                st.info("No trade history available")
        else:
            st.warning("Database not configured. Cannot show performance.")
    
    # Auto-refresh
    if auto_refresh and refresh_interval:
        time.sleep(refresh_interval)
        st.rerun()


# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
