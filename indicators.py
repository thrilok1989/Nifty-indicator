import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
import talib


class TechnicalIndicators:
    """Calculate technical indicators from OHLCV data"""
    
    @staticmethod
    def calculate_vwap(df: pd.DataFrame) -> pd.Series:
        """
        Calculate Volume Weighted Average Price
        Formula: VWAP = Σ(Price * Volume) / Σ(Volume)
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        return (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    
    @staticmethod
    def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        return talib.RSI(close, timeperiod=period)
    
    @staticmethod
    def calculate_mfi(df: pd.DataFrame, period: int = 10) -> pd.Series:
        """Money Flow Index"""
        return talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=period)
    
    @staticmethod
    def calculate_dmi(df: pd.DataFrame, period: int = 13, smoothing: int = 8) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Directional Movement Index
        Returns: (DI+, DI-, ADX)
        """
        # Calculate True Range
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        dm_plus = np.where((df['high'] - df['high'].shift()) > (df['low'].shift() - df['low']),
                          df['high'] - df['high'].shift(), 0)
        dm_plus = np.where(dm_plus < 0, 0, dm_plus)
        
        dm_minus = np.where((df['low'].shift() - df['low']) > (df['high'] - df['high'].shift()),
                           df['low'].shift() - df['low'], 0)
        dm_minus = np.where(dm_minus < 0, 0, dm_minus)
        
        # Smooth with Wilder's smoothing
        tr_smooth = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean()
        dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False).mean()
        dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False).mean()
        
        # Calculate DI+ and DI-
        di_plus = 100 * dm_plus_smooth / tr_smooth
        di_minus = 100 * dm_minus_smooth / tr_smooth
        
        # Calculate ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.ewm(alpha=1/smoothing, adjust=False).mean()
        
        return di_plus, di_minus, adx
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range"""
        return talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)
    
    @staticmethod
    def calculate_ema(close: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return talib.EMA(close, timeperiod=period)
    
    @staticmethod
    def calculate_sma(close: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return talib.SMA(close, timeperiod=period)
    
    @staticmethod
    def calculate_vidya(close: pd.Series, length: int = 10, momentum: int = 20) -> pd.Series:
        """
        Variable Index Dynamic Average (VIDYA)
        Adaptive moving average based on price momentum
        """
        m = close.diff()
        p = m.clip(lower=0).rolling(momentum).sum()
        n = (-m.clip(upper=0)).rolling(momentum).sum()
        
        abs_cmo = abs(100 * (p - n) / (p + n))
        alpha = 2 / (length + 1)
        
        vidya = pd.Series(index=close.index, dtype=float)
        vidya.iloc[0] = close.iloc[0]
        
        for i in range(1, len(close)):
            if pd.notna(abs_cmo.iloc[i]):
                vidya.iloc[i] = alpha * abs_cmo.iloc[i] / 100 * close.iloc[i] + \
                               (1 - alpha * abs_cmo.iloc[i] / 100) * vidya.iloc[i-1]
            else:
                vidya.iloc[i] = vidya.iloc[i-1]
        
        return vidya.rolling(15).mean()
    
    @staticmethod
    def calculate_obv(df: pd.DataFrame) -> pd.Series:
        """On Balance Volume"""
        return talib.OBV(df['close'], df['volume'])
    
    @staticmethod
    def calculate_force_index(df: pd.DataFrame, length: int = 13, smoothing: int = 2) -> pd.Series:
        """
        Force Index = (Close - Previous Close) * Volume
        """
        force = (df['close'] - df['close'].shift(1)) * df['volume']
        force_ema = force.ewm(span=length, adjust=False).mean()
        return force_ema.ewm(span=smoothing, adjust=False).mean()
    
    @staticmethod
    def calculate_volatility_ratio(df: pd.DataFrame, length: int = 14) -> pd.Series:
        """
        Volatility Ratio = StdDev(close) / ATR
        """
        std_dev = df['close'].rolling(length).std()
        atr = TechnicalIndicators.calculate_atr(df, length)
        return (std_dev / atr) * 100
    
    @staticmethod
    def calculate_volume_roc(volume: pd.Series, length: int = 14) -> pd.Series:
        """Volume Rate of Change"""
        return ((volume - volume.shift(length)) / volume.shift(length)) * 100
    
    @staticmethod
    def calculate_price_roc(close: pd.Series, length: int = 12) -> pd.Series:
        """Price Rate of Change"""
        return ((close - close.shift(length)) / close.shift(length)) * 100
    
    @staticmethod
    def calculate_choppiness_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Choppiness Index
        CI > 61.8: Choppy market
        CI < 38.2: Trending market
        """
        tr = TechnicalIndicators.calculate_atr(df, 1)
        sum_tr = tr.rolling(period).sum()
        highest_high = df['high'].rolling(period).max()
        lowest_low = df['low'].rolling(period).min()
        
        ci = 100 * np.log(sum_tr / (highest_high - lowest_low)) / np.log(period)
        return ci


class VolumeOrderBlocks:
    """
    Volume Order Block (VOB) Detection
    Replicates Pine Script VOB logic
    """
    
    def __init__(self, length1: int = 5):
        self.length1 = length1
        self.length2 = length1 + 13
        self.upper_vobs = []  # Bearish VOBs (resistance)
        self.lower_vobs = []  # Bullish VOBs (support)
    
    def detect_vobs(self, df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """
        Detect Volume Order Blocks from price data
        Returns: (bullish_vobs, bearish_vobs)
        """
        # Calculate EMAs
        ema1 = TechnicalIndicators.calculate_ema(df['close'], self.length1)
        ema2 = TechnicalIndicators.calculate_ema(df['close'], self.length2)
        
        # Detect crossovers
        cross_up = (ema1 > ema2) & (ema1.shift(1) <= ema2.shift(1))
        cross_down = (ema1 < ema2) & (ema1.shift(1) >= ema2.shift(1))
        
        bullish_vobs = []
        bearish_vobs = []
        
        # Find bullish VOBs (support zones)
        for i in range(len(df)):
            if cross_up.iloc[i]:
                # Look back to find the lowest low
                lookback_start = max(0, i - self.length2)
                lookback_period = df.iloc[lookback_start:i+1]
                
                lowest_idx = lookback_period['low'].idxmin()
                lowest_low = lookback_period.loc[lowest_idx, 'low']
                
                # Calculate block boundaries
                open_val = df.loc[lowest_idx, 'open']
                close_val = df.loc[lowest_idx, 'close']
                upper = min(open_val, close_val)
                
                # Calculate volume in block
                volume_sum = lookback_period['volume'].sum()
                
                atr = TechnicalIndicators.calculate_atr(df.iloc[:i+1], 200).iloc[-1]
                if (upper - lowest_low) < atr * 2 * 0.5:
                    upper = lowest_low + atr * 2 * 0.5
                
                mid = (upper + lowest_low) / 2
                
                bullish_vobs.append({
                    'index': i,
                    'upper': upper,
                    'lower': lowest_low,
                    'mid': mid,
                    'volume': volume_sum,
                    'timestamp': df.index[i]
                })
        
        # Find bearish VOBs (resistance zones)
        for i in range(len(df)):
            if cross_down.iloc[i]:
                # Look back to find the highest high
                lookback_start = max(0, i - self.length2)
                lookback_period = df.iloc[lookback_start:i+1]
                
                highest_idx = lookback_period['high'].idxmax()
                highest_high = lookback_period.loc[highest_idx, 'high']
                
                # Calculate block boundaries
                open_val = df.loc[highest_idx, 'open']
                close_val = df.loc[highest_idx, 'close']
                lower = max(open_val, close_val)
                
                # Calculate volume in block
                volume_sum = lookback_period['volume'].sum()
                
                atr = TechnicalIndicators.calculate_atr(df.iloc[:i+1], 200).iloc[-1]
                if (highest_high - lower) < atr * 2 * 0.5:
                    lower = highest_high - atr * 2 * 0.5
                
                mid = (highest_high + lower) / 2
                
                bearish_vobs.append({
                    'index': i,
                    'upper': highest_high,
                    'lower': lower,
                    'mid': mid,
                    'volume': volume_sum,
                    'timestamp': df.index[i]
                })
        
        return bullish_vobs, bearish_vobs
    
    def check_vob_touch(self, current_price: float, current_low: float, 
                       current_high: float) -> Tuple[bool, bool, float, float]:
        """
        Check if price is touching any active VOB
        Returns: (bull_touch, bear_touch, bull_level, bear_level)
        """
        bull_touch = False
        bear_touch = False
        bull_level = None
        bear_level = None
        
        # Check bullish VOBs (support)
        for vob in self.lower_vobs:
            if current_low <= vob['upper'] and current_low >= vob['lower']:
                bull_touch = True
                bull_level = vob['mid']
                break
        
        # Check bearish VOBs (resistance)
        for vob in self.upper_vobs:
            if current_high >= vob['lower'] and current_high <= vob['upper']:
                bear_touch = True
                bear_level = vob['mid']
                break
        
        return bull_touch, bear_touch, bull_level, bear_level
    
    def update_active_vobs(self, bullish_vobs: List[Dict], bearish_vobs: List[Dict], 
                          current_price: float, atr: float):
        """
        Update active VOB levels, removing invalidated ones
        """
        # Filter out invalidated bullish VOBs
        self.lower_vobs = [
            vob for vob in bullish_vobs
            if current_price >= vob['lower']  # Price hasn't broken below
        ]
        
        # Filter out invalidated bearish VOBs
        self.upper_vobs = [
            vob for vob in bearish_vobs
            if current_price <= vob['upper']  # Price hasn't broken above
        ]
        
        # Limit to most recent 15 VOBs
        self.lower_vobs = self.lower_vobs[-15:]
        self.upper_vobs = self.upper_vobs[-15:]


class PivotDetector:
    """
    HTF Pivot Point Detection
    """
    
    @staticmethod
    def detect_pivots(df: pd.DataFrame, length: int = 5) -> Tuple[pd.Series, pd.Series]:
        """
        Detect pivot highs and lows
        Returns: (pivot_highs, pivot_lows)
        """
        pivot_highs = pd.Series(index=df.index, dtype=float)
        pivot_lows = pd.Series(index=df.index, dtype=float)
        
        for i in range(length, len(df) - length):
            # Check pivot high
            is_pivot_high = True
            for j in range(1, length + 1):
                if df['high'].iloc[i] <= df['high'].iloc[i - j] or \
                   df['high'].iloc[i] <= df['high'].iloc[i + j]:
                    is_pivot_high = False
                    break
            
            if is_pivot_high:
                pivot_highs.iloc[i] = df['high'].iloc[i]
            
            # Check pivot low
            is_pivot_low = True
            for j in range(1, length + 1):
                if df['low'].iloc[i] >= df['low'].iloc[i - j] or \
                   df['low'].iloc[i] >= df['low'].iloc[i + j]:
                    is_pivot_low = False
                    break
            
            if is_pivot_low:
                pivot_lows.iloc[i] = df['low'].iloc[i]
        
        return pivot_highs.fillna(method='ffill'), pivot_lows.fillna(method='ffill')
