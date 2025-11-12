import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from dataclasses import dataclass
from indicators import TechnicalIndicators, VolumeOrderBlocks
import logging

logger = logging.getLogger(__name__)


@dataclass
class BiasSignals:
    """Container for bias calculation signals"""
    # Fast signals (8 total)
    volume_bullish: bool = False
    volume_bearish: bool = False
    hvp_bullish: bool = False
    hvp_bearish: bool = False
    vob_touched_bull: bool = False
    vob_touched_bear: bool = False
    ob_bullish: bool = False
    ob_bearish: bool = False
    rsi_bullish: bool = False
    rsi_bearish: bool = False
    dmi_bullish: bool = False
    dmi_bearish: bool = False
    vidya_bullish: bool = False
    vidya_bearish: bool = False
    mfi_bullish: bool = False
    mfi_bearish: bool = False
    
    # Medium signals (2 total)
    price_above_vwap: bool = False
    price_below_vwap: bool = False
    
    # Slow signals (3 total)
    stocks_daily_bullish: bool = False
    stocks_daily_bearish: bool = False
    stocks_tf1_bullish: bool = False
    stocks_tf1_bearish: bool = False
    stocks_tf2_bullish: bool = False
    stocks_tf2_bearish: bool = False
    
    # Enhanced signals
    obv_bullish: bool = False
    obv_bearish: bool = False
    force_bullish: bool = False
    force_bearish: bool = False
    momentum_bullish: bool = False
    momentum_bearish: bool = False
    volatility_high: bool = False
    volatility_low: bool = False
    volume_strong: bool = False
    volume_weak: bool = False
    breadth_bullish: bool = False
    breadth_bearish: bool = False
    divergence_bullish: bool = False
    divergence_bearish: bool = False
    choppy_market: bool = False
    trending_market: bool = False


class MarketBiasCalculator:
    """
    Calculate comprehensive market bias using multi-factor analysis
    Implements Pine Script enhanced bias logic
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.bias_strength = config.get('BIAS_STRENGTH', 60)
        self.divergence_threshold = config.get('DIVERGENCE_THRESHOLD', 60)
        
        # Normal mode weights
        self.normal_fast_weight = config.get('NORMAL_FAST_WEIGHT', 2.0)
        self.normal_medium_weight = config.get('NORMAL_MEDIUM_WEIGHT', 3.0)
        self.normal_slow_weight = config.get('NORMAL_SLOW_WEIGHT', 5.0)
        
        # Reversal mode weights
        self.reversal_fast_weight = config.get('REVERSAL_FAST_WEIGHT', 5.0)
        self.reversal_medium_weight = config.get('REVERSAL_MEDIUM_WEIGHT', 3.0)
        self.reversal_slow_weight = config.get('REVERSAL_SLOW_WEIGHT', 2.0)
        
        self.reversal_mode = False
        self.market_condition = "NEUTRAL"
        
    def calculate_fast_signals(self, df: pd.DataFrame, vob_data: dict,
                              indicators: dict) -> Tuple[int, int]:
        """
        Calculate fast signals (volume, VOB, OB, RSI, DMI, VIDYA, MFI)
        Returns: (bull_count, bear_count)
        """
        signals = BiasSignals()
        bull_count = 0
        bear_count = 0
        
        # 1. Volume Analysis
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        
        if current_volume > avg_volume * 1.5:
            if df['close'].iloc[-1] > df['open'].iloc[-1]:
                signals.volume_bullish = True
                bull_count += 1
            else:
                signals.volume_bearish = True
                bear_count += 1
        
        # 2. VOB Touch Detection
        if vob_data.get('bull_touch', False):
            signals.vob_touched_bull = True
            bull_count += 1
        
        if vob_data.get('bear_touch', False):
            signals.vob_touched_bear = True
            bear_count += 1
        
        # 3. Order Block Signals (EMA crossover)
        ema_fast = indicators.get('ema_fast')
        ema_slow = indicators.get('ema_slow')
        
        if ema_fast is not None and ema_slow is not None:
            if ema_fast.iloc[-1] > ema_slow.iloc[-1] and ema_fast.iloc[-2] <= ema_slow.iloc[-2]:
                signals.ob_bullish = True
                bull_count += 1
            elif ema_fast.iloc[-1] < ema_slow.iloc[-1] and ema_fast.iloc[-2] >= ema_slow.iloc[-2]:
                signals.ob_bearish = True
                bear_count += 1
        
        # 4. RSI
        rsi = indicators.get('rsi')
        if rsi is not None:
            if rsi.iloc[-1] > 50:
                signals.rsi_bullish = True
                bull_count += 1
            else:
                signals.rsi_bearish = True
                bear_count += 1
        
        # 5. DMI (Directional Movement Index)
        di_plus = indicators.get('di_plus')
        di_minus = indicators.get('di_minus')
        
        if di_plus is not None and di_minus is not None:
            if di_plus.iloc[-1] > di_minus.iloc[-1]:
                signals.dmi_bullish = True
                bull_count += 1
            else:
                signals.dmi_bearish = True
                bear_count += 1
        
        # 6. VIDYA
        vidya = indicators.get('vidya')
        if vidya is not None:
            if df['close'].iloc[-1] > vidya.iloc[-1]:
                signals.vidya_bullish = True
                bull_count += 1
            else:
                signals.vidya_bearish = True
                bear_count += 1
        
        # 7. MFI (Money Flow Index)
        mfi = indicators.get('mfi')
        if mfi is not None:
            if mfi.iloc[-1] > 50:
                signals.mfi_bullish = True
                bull_count += 1
            else:
                signals.mfi_bearish = True
                bear_count += 1
        
        # 8. HVP (High Volume Pivot) - simplified
        # Using volume spike at pivot points
        if df['volume'].iloc[-1] > avg_volume * 2:
            pivot_highs = df['high'].rolling(15).max()
            pivot_lows = df['low'].rolling(15).min()
            
            if df['low'].iloc[-1] == pivot_lows.iloc[-1]:
                signals.hvp_bullish = True
                bull_count += 1
            elif df['high'].iloc[-1] == pivot_highs.iloc[-1]:
                signals.hvp_bearish = True
                bear_count += 1
        
        return bull_count, bear_count
    
    def calculate_medium_signals(self, df: pd.DataFrame, 
                                indicators: dict) -> Tuple[int, int]:
        """
        Calculate medium signals (VWAP, price position)
        Returns: (bull_count, bear_count)
        """
        bull_count = 0
        bear_count = 0
        
        # 1. Price vs VWAP
        vwap = indicators.get('vwap')
        if vwap is not None:
            if df['close'].iloc[-1] > vwap.iloc[-1]:
                bull_count += 1
            else:
                bear_count += 1
        
        # 2. Additional VWAP confirmation
        if vwap is not None:
            if df['close'].iloc[-1] > vwap.iloc[-1]:
                bull_count += 1
            else:
                bear_count += 1
        
        return bull_count, bear_count
    
    def calculate_slow_signals(self, stock_data: Dict[str, dict]) -> Tuple[int, int]:
        """
        Calculate slow signals (stock correlations, weighted averages)
        Returns: (bull_count, bear_count)
        """
        bull_count = 0
        bear_count = 0
        
        # Calculate weighted averages across stocks
        daily_changes = []
        tf1_changes = []
        tf2_changes = []
        weights = []
        
        for stock_name, data in stock_data.items():
            if 'weight' in data and data['weight'] > 0:
                weights.append(data['weight'])
                daily_changes.append(data.get('daily_change_pct', 0))
                tf1_changes.append(data.get('tf1_change_pct', 0))
                tf2_changes.append(data.get('tf2_change_pct', 0))
        
        if weights:
            total_weight = sum(weights)
            
            # Weighted average daily
            weighted_daily = sum(d * w for d, w in zip(daily_changes, weights)) / total_weight
            if weighted_daily > 0:
                bull_count += 1
            else:
                bear_count += 1
            
            # Weighted average TF1
            weighted_tf1 = sum(d * w for d, w in zip(tf1_changes, weights)) / total_weight
            if weighted_tf1 > 0:
                bull_count += 1
            else:
                bear_count += 1
            
            # Weighted average TF2
            weighted_tf2 = sum(d * w for d, w in zip(tf2_changes, weights)) / total_weight
            if weighted_tf2 > 0:
                bull_count += 1
            else:
                bear_count += 1
        
        return bull_count, bear_count
    
    def calculate_enhanced_signals(self, df: pd.DataFrame, 
                                  indicators: dict,
                                  stock_data: Dict[str, dict]) -> Dict[str, int]:
        """
        Calculate enhanced signals (OBV, Force Index, Breadth, etc.)
        Returns: dict with bull/bear counts for each category
        """
        enhanced = {'bull': 0, 'bear': 0}
        
        # 1. OBV Analysis
        obv = indicators.get('obv')
        obv_ma = indicators.get('obv_ma')
        
        if obv is not None and obv_ma is not None:
            obv_rising = obv.iloc[-1] > obv.iloc[-2]
            if obv.iloc[-1] > obv_ma.iloc[-1] and obv_rising:
                enhanced['bull'] += 1
            elif obv.iloc[-1] < obv_ma.iloc[-1] and not obv_rising:
                enhanced['bear'] += 1
        
        # 2. Force Index
        force_index = indicators.get('force_index')
        if force_index is not None:
            force_rising = force_index.iloc[-1] > force_index.iloc[-2]
            if force_index.iloc[-1] > 0 and force_rising:
                enhanced['bull'] += 1
            elif force_index.iloc[-1] < 0 and not force_rising:
                enhanced['bear'] += 1
        
        # 3. Price Momentum (ROC)
        price_roc = indicators.get('price_roc')
        if price_roc is not None:
            if price_roc.iloc[-1] > 0:
                enhanced['bull'] += 1
            else:
                enhanced['bear'] += 1
        
        # 4. Market Breadth (% of stocks bullish)
        bullish_stocks = sum(1 for stock, data in stock_data.items() 
                           if data.get('daily_change_pct', 0) > 0)
        total_stocks = len([s for s in stock_data.values() if s.get('weight', 0) > 0])
        
        if total_stocks > 0:
            breadth_pct = (bullish_stocks / total_stocks) * 100
            if breadth_pct > 60:
                enhanced['bull'] += 2  # Double weight
            elif breadth_pct < 40:
                enhanced['bear'] += 2
        
        # 5. Volatility Analysis
        volatility_ratio = indicators.get('volatility_ratio')
        if volatility_ratio is not None:
            if volatility_ratio.iloc[-1] < 1.0:  # Low volatility
                # Low volatility with trend continuation
                if df['close'].iloc[-1] > df['close'].iloc[-5]:
                    enhanced['bull'] += 1
                else:
                    enhanced['bear'] += 1
        
        # 6. Volume Strength
        volume_roc = indicators.get('volume_roc')
        if volume_roc is not None:
            if volume_roc.iloc[-1] > 20:  # Strong volume increase
                if df['close'].iloc[-1] > df['open'].iloc[-1]:
                    enhanced['bull'] += 1
                else:
                    enhanced['bear'] += 1
        
        # 7. Choppiness Index
        ci = indicators.get('choppiness_index')
        if ci is not None:
            if ci.iloc[-1] < 38.2:  # Trending market
                if df['close'].iloc[-1] > df['close'].iloc[-5]:
                    enhanced['bull'] += 1
                else:
                    enhanced['bear'] += 1
        
        return enhanced
    
    def detect_divergence(self, fast_bull_pct: float, fast_bear_pct: float,
                         slow_bull_pct: float, slow_bear_pct: float) -> Tuple[bool, bool]:
        """
        Detect divergence between fast and slow indicators
        Returns: (bullish_divergence, bearish_divergence)
        """
        bullish_divergence = (slow_bull_pct >= 66 and 
                             fast_bear_pct >= self.divergence_threshold)
        
        bearish_divergence = (slow_bear_pct >= 66 and 
                             fast_bull_pct >= self.divergence_threshold)
        
        return bullish_divergence, bearish_divergence
    
    def calculate_bias(self, df: pd.DataFrame, indicators: dict,
                      vob_data: dict, stock_data: Dict[str, dict],
                      market_condition: str = "NEUTRAL") -> Dict:
        """
        Main bias calculation function
        Returns comprehensive bias analysis
        """
        self.market_condition = market_condition
        
        # Calculate signal groups
        fast_bull, fast_bear = self.calculate_fast_signals(df, vob_data, indicators)
        medium_bull, medium_bear = self.calculate_medium_signals(df, indicators)
        slow_bull, slow_bear = self.calculate_slow_signals(stock_data)
        
        fast_total = 8  # Total fast indicators
        medium_total = 2  # Total medium indicators
        slow_total = 3  # Total slow indicators
        
        # Calculate percentages
        fast_bull_pct = (fast_bull / fast_total) * 100
        fast_bear_pct = (fast_bear / fast_total) * 100
        slow_bull_pct = (slow_bull / slow_total) * 100
        slow_bear_pct = (slow_bear / slow_total) * 100
        
        # Detect divergence
        bullish_div, bearish_div = self.detect_divergence(
            fast_bull_pct, fast_bear_pct, slow_bull_pct, slow_bear_pct
        )
        
        divergence_detected = bullish_div or bearish_div
        
        # Update reversal mode
        if divergence_detected and not self.reversal_mode:
            self.reversal_mode = True
            logger.info("🚨 REVERSAL MODE ACTIVATED")
        
        # Exit reversal mode when alignment occurs
        if self.reversal_mode:
            if (fast_bull_pct >= 60 and slow_bull_pct >= 60) or \
               (fast_bear_pct >= 60 and slow_bear_pct >= 60):
                self.reversal_mode = False
                logger.info("✅ NORMAL MODE ACTIVATED")
        
        # Select weights based on mode
        if self.reversal_mode:
            fast_weight = self.reversal_fast_weight
            medium_weight = self.reversal_medium_weight
            slow_weight = self.reversal_slow_weight
        else:
            fast_weight = self.normal_fast_weight
            medium_weight = self.normal_medium_weight
            slow_weight = self.normal_slow_weight
        
        # Calculate weighted bias
        bullish_signals = (fast_bull * fast_weight) + \
                         (medium_bull * medium_weight) + \
                         (slow_bull * slow_weight)
        
        bearish_signals = (fast_bear * fast_weight) + \
                         (medium_bear * medium_weight) + \
                         (slow_bear * slow_weight)
        
        total_signals = (fast_total * fast_weight) + \
                       (medium_total * medium_weight) + \
                       (slow_total * slow_weight)
        
        bullish_bias_pct = (bullish_signals / total_signals) * 100
        bearish_bias_pct = (bearish_signals / total_signals) * 100
        
        # Calculate enhanced signals
        enhanced = self.calculate_enhanced_signals(df, indicators, stock_data)
        enhanced_total = enhanced['bull'] + enhanced['bear']
        
        if enhanced_total > 0:
            enhanced_bull_pct = (enhanced['bull'] / enhanced_total) * 100
            enhanced_bear_pct = (enhanced['bear'] / enhanced_total) * 100
        else:
            enhanced_bull_pct = 0
            enhanced_bear_pct = 0
        
        # Combined bias (60% base + 40% enhanced)
        combined_bull_pct = (bullish_bias_pct * 0.6) + (enhanced_bull_pct * 0.4)
        combined_bear_pct = (bearish_bias_pct * 0.6) + (enhanced_bear_pct * 0.4)
        
        # Adjust bias strength based on market condition
        adjusted_bias_strength = self.bias_strength
        if market_condition == "RANGE-BOUND":
            adjusted_bias_strength += 10
        
        # Determine final bias
        if combined_bull_pct >= adjusted_bias_strength:
            final_bias = "BULLISH"
            bias_color = "#089981"
        elif combined_bear_pct >= adjusted_bias_strength:
            final_bias = "BEARISH"
            bias_color = "#f23645"
        else:
            final_bias = "NEUTRAL"
            bias_color = "#787B86"
        
        # Calculate confidence level
        max_bias = max(combined_bull_pct, combined_bear_pct)
        if max_bias > 80:
            confidence = "HIGH"
        elif max_bias > 60:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        return {
            'bias': final_bias,
            'bias_color': bias_color,
            'bullish_pct': round(combined_bull_pct, 1),
            'bearish_pct': round(combined_bear_pct, 1),
            'confidence': confidence,
            'mode': 'REVERSAL' if self.reversal_mode else 'NORMAL',
            'divergence_detected': divergence_detected,
            'bullish_divergence': bullish_div,
            'bearish_divergence': bearish_div,
            'market_condition': market_condition,
            'fast_signals': {'bull': fast_bull, 'bear': fast_bear, 'total': fast_total},
            'medium_signals': {'bull': medium_bull, 'bear': medium_bear, 'total': medium_total},
            'slow_signals': {'bull': slow_bull, 'bear': slow_bear, 'total': slow_total},
            'enhanced_signals': enhanced,
            'adjusted_bias_strength': adjusted_bias_strength
        }


class RangeDetector:
    """
    Market condition detection (Range-bound vs Trending)
    """
    
    def __init__(self, config: dict):
        self.range_pct_threshold = config.get('RANGE_PCT_THRESHOLD', 2.0)
        self.range_min_bars = config.get('RANGE_MIN_BARS', 20)
        self.ema_spread_threshold = config.get('EMA_SPREAD_THRESHOLD', 0.5)
        self.current_condition = "NEUTRAL"
        self.range_high = None
        self.range_low = None
        self.range_mid = None
        self.bars_in_range = 0
    
    def detect_condition(self, df: pd.DataFrame, ema1: pd.Series, 
                        ema2: pd.Series, atr: pd.Series) -> Dict:
        """
        Detect market condition: RANGE-BOUND, TRENDING UP, TRENDING DOWN
        """
        current_close = df['close'].iloc[-1]
        
        # Calculate rolling range
        lookback = self.range_min_bars
        rolling_high = df['high'].iloc[-lookback:].max()
        rolling_low = df['low'].iloc[-lookback:].min()
        rolling_range_pct = ((rolling_high - rolling_low) / current_close) * 100
        
        # Calculate EMA spread
        ema_spread_pct = abs(ema1.iloc[-1] - ema2.iloc[-1]) / current_close * 100
        
        # ATR analysis
        atr_current = atr.iloc[-1]
        atr_avg = atr.iloc[-50:].mean()
        low_volatility = atr_current < atr_avg
        
        # Check for recent crossovers
        cross_up_recent = any((ema1.iloc[-i] > ema2.iloc[-i] and 
                              ema1.iloc[-i-1] <= ema2.iloc[-i-1]) 
                             for i in range(1, min(self.range_min_bars, len(ema1))))
        
        cross_dn_recent = any((ema1.iloc[-i] < ema2.iloc[-i] and 
                              ema1.iloc[-i-1] >= ema2.iloc[-i-1]) 
                             for i in range(1, min(self.range_min_bars, len(ema1))))
        
        # Determine condition
        tight_range = rolling_range_pct < self.range_pct_threshold
        low_ema_spread = ema_spread_pct < self.ema_spread_threshold
        no_recent_cross = not (cross_up_recent or cross_dn_recent)
        
        if tight_range and low_ema_spread and low_volatility:
            self.current_condition = "RANGE-BOUND"
            if self.range_high is None:
                self.range_high = rolling_high
                self.range_low = rolling_low
                self.range_mid = (self.range_high + self.range_low) / 2
            self.bars_in_range += 1
            
        elif cross_up_recent or (ema1.iloc[-1] > ema2.iloc[-1] and ema_spread_pct > 1.0):
            self.current_condition = "TRENDING UP"
            self.range_high = None
            self.range_low = None
            self.range_mid = None
            self.bars_in_range = 0
            
        elif cross_dn_recent or (ema1.iloc[-1] < ema2.iloc[-1] and ema_spread_pct > 1.0):
            self.current_condition = "TRENDING DOWN"
            self.range_high = None
            self.range_low = None
            self.range_mid = None
            self.bars_in_range = 0
            
        else:
            self.current_condition = "TRANSITION"
            if self.bars_in_range > 0:
                self.bars_in_range -= 1
            if self.bars_in_range <= 0:
                self.range_high = None
                self.range_low = None
                self.range_mid = None
        
        # Movement analysis
        movement_strength = ema_spread_pct * 10
        if movement_strength > 5:
            movement_quality = "STRONG"
        elif movement_strength > 2:
            movement_quality = "MODERATE"
        elif movement_strength > 0.5:
            movement_quality = "WEAK"
        else:
            movement_quality = "CHOPPY"
        
        # Breakout detection
        breakout_potential = (self.current_condition == "RANGE-BOUND" and 
                            self.bars_in_range > 50)
        
        if breakout_potential and self.range_mid:
            breakout_direction = "BULLISH" if current_close > self.range_mid else "BEARISH"
        else:
            breakout_direction = None
        
        # Volume analysis for expansion
        volume_trend = df['volume'].iloc[-20:].mean()
        volume_increasing = df['volume'].iloc[-1] > volume_trend * 1.2
        expansion_coming = (self.current_condition == "RANGE-BOUND" and 
                          volume_increasing)
        
        # Range breakout signals
        range_breakout_long = (self.current_condition == "RANGE-BOUND" and 
                              self.range_high and 
                              current_close > self.range_high and 
                              volume_increasing)
        
        range_breakout_short = (self.current_condition == "RANGE-BOUND" and 
                               self.range_low and 
                               current_close < self.range_low and 
                               volume_increasing)
        
        return {
            'condition': self.current_condition,
            'range_high': self.range_high,
            'range_low': self.range_low,
            'range_mid': self.range_mid,
            'bars_in_range': self.bars_in_range,
            'rolling_range_pct': round(rolling_range_pct, 2),
            'ema_spread_pct': round(ema_spread_pct, 2),
            'movement_strength': round(movement_strength, 2),
            'movement_quality': movement_quality,
            'breakout_potential': breakout_potential,
            'breakout_direction': breakout_direction,
            'expansion_coming': expansion_coming,
            'range_breakout_long': range_breakout_long,
            'range_breakout_short': range_breakout_short,
            'volatility_regime': 'LOW' if low_volatility else 'NORMAL'
        }
