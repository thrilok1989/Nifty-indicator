"""
Configuration file for Nifty Trading System
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# DHAN API CREDENTIALS
# ============================================================================
DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID", "your_client_id")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN", "your_access_token")

# ============================================================================
# NIFTY & STOCK CONFIGURATION
# ============================================================================
NIFTY_SECURITY_ID = "13"  # NIFTY 50 Index
BANKNIFTY_SECURITY_ID = "25"  # Bank Nifty Index

# Stock configuration matching your Pine Script
STOCKS_CONFIG = {
    "RELIANCE": {"security_id": "2885", "weight": 9.98},
    "HDFCBANK": {"security_id": "1333", "weight": 9.67},
    "BHARTIARTL": {"security_id": "14366", "weight": 9.97},
    "TCS": {"security_id": "11536", "weight": 8.54},
    "ICICIBANK": {"security_id": "4963", "weight": 8.01},
    "INFY": {"security_id": "1594", "weight": 8.55},
    "HINDUNILVR": {"security_id": "1394", "weight": 1.98},
    "ITC": {"security_id": "1660", "weight": 2.44},
    "MARUTI": {"security_id": "10999", "weight": 0.0}
}

# Market indicators
US30_SECURITY_ID = "CAPITALCOM:US30"  # US Dow Jones
USDINR_SECURITY_ID = "1"  # USD/INR

# ============================================================================
# OPTIONS CHAIN CONFIGURATION
# ============================================================================
NIFTY_UNDERLYING_SCRIP = "13"
NIFTY_UNDERLYING_SEG = "IDX_I"

# ============================================================================
# INDICATOR SETTINGS (From Pine Script)
# ============================================================================
# VWAP
SHOW_VWAP = True

# RSI/MFI/DMI
RSI_PERIOD = 14
MFI_PERIOD = 10
DMI_PERIOD = 13
DMI_SMOOTHING = 8

# VIDYA
VIDYA_LENGTH = 10
VIDYA_MOMENTUM = 20
BAND_DISTANCE = 2.0

# Volume Order Blocks (VOB)
VOB_SENSITIVITY = 5
VOB_LENGTH2 = VOB_SENSITIVITY + 13
SHOW_VOB = True
SHOW_VOB_ENTRY_SIGNALS = True

# Order Blocks
OB_SENSITIVITY = 5

# HTF Pivots (Note: Using 5min instead of 3min due to DhanHQ limitation)
PIVOT_TIMEFRAMES = {
    "5min": {"length": 5, "color": "#2962FF"},
    "15min": {"length": 5, "color": "#9C27B0"}
}

# Range Detection
ENABLE_RANGE_DETECTION = True
RANGE_PCT_THRESHOLD = 2.0
RANGE_MIN_BARS = 20
EMA_SPREAD_THRESHOLD = 0.5

# Enhanced Indicators
VOLATILITY_RATIO_LENGTH = 14
VOLATILITY_THRESHOLD = 1.5
VOLUME_ROC_LENGTH = 14
VOLUME_THRESHOLD = 1.2
OBV_SMOOTHING = 21
FORCE_INDEX_LENGTH = 13
FORCE_INDEX_SMOOTHING = 2

# ============================================================================
# BIAS CALCULATION SETTINGS
# ============================================================================
BIAS_STRENGTH = 60  # Minimum bias strength %
DIVERGENCE_THRESHOLD = 60

# Normal mode weights
NORMAL_FAST_WEIGHT = 2.0
NORMAL_MEDIUM_WEIGHT = 3.0
NORMAL_SLOW_WEIGHT = 5.0

# Reversal mode weights
REVERSAL_FAST_WEIGHT = 5.0
REVERSAL_MEDIUM_WEIGHT = 3.0
REVERSAL_SLOW_WEIGHT = 2.0

# ============================================================================
# TRADE MANAGEMENT
# ============================================================================
USE_ATR_STOP_LOSS = True
ATR_MULTIPLIER = 1.5
ATR_PERIOD = 14

USE_FIXED_STOP_LOSS = False
FIXED_STOP_LOSS_PCT = 1.0

EXIT_ON_BIAS_CHANGE = True
EXIT_ON_PIVOT_TOUCH = True
PIVOT_TOUCH_TOLERANCE = 0.15
PIVOT_EXIT_TOLERANCE = 0.2

RISK_REWARD_RATIO = 2.0

# ============================================================================
# OPTIONS CHAIN WEIGHTS (For Bias Calculation)
# ============================================================================
OPTIONS_WEIGHTS = {
    "ChgOI_Bias": 3.0,
    "Volume_Bias": 2.5,
    "AskQty_Bias": 2.0,
    "BidQty_Bias": 2.0,
    "DVP_Bias": 3.5,
    "PCR_Signal": 4.0
}

# ============================================================================
# SUPABASE CONFIGURATION
# ============================================================================
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# ============================================================================
# TELEGRAM CONFIGURATION
# ============================================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ============================================================================
# WEBSOCKET SETTINGS
# ============================================================================
WEBSOCKET_URL = "wss://api-feed.dhan.co"
WEBSOCKET_VERSION = 2

# ============================================================================
# TIMEFRAME SETTINGS
# ============================================================================
TIMEFRAMES = {
    "TF1": "15",  # 15 minutes
    "TF2": "60"   # 1 hour
}

# ============================================================================
# VOLUME SETTINGS
# ============================================================================
AVERAGE_VOLUME_BANKNIFTY = 40000  # in contracts
AVERAGE_VOLUME_NIFTY = 100000     # in contracts

# ============================================================================
# STREAMLIT UI SETTINGS
# ============================================================================
DASHBOARD_LOCATION = "Top Right"
TABLE_TEXT_SIZE = "Small"
SHOW_TICKERS_TABLE = True
SHOW_INDICATOR_TABLE = True
SHOW_BIAS_TABLE = True

# Colors
BULL_COLOR = "#089981"
BEAR_COLOR = "#f23645"
NEUTRAL_COLOR = "#787B86"
