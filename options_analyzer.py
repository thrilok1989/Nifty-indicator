import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from datetime import datetime, timedelta
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)


class OptionsAnalyzer:
    """
    Complete options chain analysis
    Implements Greeks calculation, PCR analysis, bias detection
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            "ChgOI_Bias": 3.0,
            "Volume_Bias": 2.5,
            "AskQty_Bias": 2.0,
            "BidQty_Bias": 2.0,
            "DVP_Bias": 3.5,
            "PCR_Signal": 4.0
        }
    
    def calculate_time_to_expiry(self, expiry_date: str) -> float:
        """
        Calculate exact time to expiry in years
        Accounts for market hours (9:15 AM to 3:30 PM IST)
        """
        from pytz import timezone
        
        ist = timezone('Asia/Kolkata')
        now = datetime.now(ist)
        
        # Parse expiry date
        expiry = datetime.strptime(expiry_date, '%Y-%m-%d')
        expiry = ist.localize(expiry.replace(hour=15, minute=30))  # 3:30 PM expiry
        
        # Calculate time difference
        time_diff = expiry - now
        
        # Convert to years (trading hours basis)
        # Assuming 252 trading days, 6.25 hours per day
        total_hours = time_diff.total_seconds() / 3600
        trading_hours_per_year = 252 * 6.25
        
        T = max(total_hours / trading_hours_per_year, 0.0001)  # Minimum 0.0001 to avoid division by zero
        
        return T
    
    def calculate_greeks(self, option_type: str, underlying: float, 
                        strike: float, T: float, r: float, 
                        sigma: float) -> Tuple[float, float, float, float, float]:
        """
        Calculate Black-Scholes Greeks
        Returns: (Delta, Gamma, Vega, Theta, Rho)
        """
        if T <= 0:
            T = 0.0001
        
        if sigma <= 0:
            sigma = 0.15  # Default 15% IV
        
        # Black-Scholes d1 and d2
        d1 = (np.log(underlying / strike) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Delta
        if option_type == 'CE':
            delta = norm.cdf(d1)
        else:  # PE
            delta = -norm.cdf(-d1)
        
        # Gamma (same for both CE and PE)
        gamma = norm.pdf(d1) / (underlying * sigma * np.sqrt(T))
        
        # Vega (same for both CE and PE, divided by 100 for 1% change)
        vega = (underlying * norm.pdf(d1) * np.sqrt(T)) / 100
        
        # Theta
        if option_type == 'CE':
            theta = (-(underlying * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - 
                    r * strike * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:  # PE
            theta = (-(underlying * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + 
                    r * strike * np.exp(-r * T) * norm.cdf(-d2)) / 365
        
        # Rho (divided by 100 for 1% change in interest rate)
        if option_type == 'CE':
            rho = (strike * T * np.exp(-r * T) * norm.cdf(d2)) / 100
        else:  # PE
            rho = (-strike * T * np.exp(-r * T) * norm.cdf(-d2)) / 100
        
        return delta, gamma, vega, theta, rho
    
    def get_iv_fallback(self, df: pd.DataFrame, strike: float) -> Tuple[float, float]:
        """
        Get IV fallback using nearest strike average
        """
        # Find nearest strikes
        strikes_sorted = df['strikePrice'].sort_values()
        nearest_idx = (strikes_sorted - strike).abs().argsort()[:3]
        nearest_strikes = strikes_sorted.iloc[nearest_idx]
        
        # Average IV from nearest strikes
        iv_ce_list = []
        iv_pe_list = []
        
        for s in nearest_strikes:
            row = df[df['strikePrice'] == s]
            if not row.empty:
                iv_ce = row['implied_volatility_CE'].iloc[0]
                iv_pe = row['implied_volatility_PE'].iloc[0]
                
                if pd.notna(iv_ce) and iv_ce > 0:
                    iv_ce_list.append(iv_ce)
                if pd.notna(iv_pe) and iv_pe > 0:
                    iv_pe_list.append(iv_pe)
        
        avg_iv_ce = np.mean(iv_ce_list) if iv_ce_list else 15
        avg_iv_pe = np.mean(iv_pe_list) if iv_pe_list else 15
        
        return avg_iv_ce, avg_iv_pe
    
    def calculate_pcr(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Put-Call Ratio (PCR)
        PCR = PUT OI / CALL OI
        """
        pcr = df['openInterest_PE'] / df['openInterest_CE']
        pcr = np.where(df['openInterest_CE'] == 0, 0, pcr)
        return pcr
    
    def calculate_bid_ask_pressure(self, bid_qty_ce: float, ask_qty_ce: float,
                                  bid_qty_pe: float, ask_qty_pe: float) -> Tuple[str, str]:
        """
        Calculate bid-ask pressure
        """
        call_pressure = bid_qty_ce - ask_qty_ce
        put_pressure = bid_qty_pe - ask_qty_pe
        
        net_pressure = put_pressure - call_pressure
        
        if net_pressure > 0:
            pressure_bias = "Bullish"
            if net_pressure > 1000:
                pressure_strength = "Strong"
            elif net_pressure > 500:
                pressure_strength = "Moderate"
            else:
                pressure_strength = "Weak"
        elif net_pressure < 0:
            pressure_bias = "Bearish"
            if abs(net_pressure) > 1000:
                pressure_strength = "Strong"
            elif abs(net_pressure) > 500:
                pressure_strength = "Moderate"
            else:
                pressure_strength = "Weak"
        else:
            pressure_bias = "Neutral"
            pressure_strength = "Neutral"
        
        return f"{pressure_strength} {pressure_bias}", pressure_bias
    
    def delta_volume_bias(self, price_diff: float, volume_diff: float, 
                         oi_diff: float) -> str:
        """
        Calculate Delta-Volume-Price bias
        """
        # Scoring system
        score = 0
        
        # Price differential
        if price_diff > 0:
            score -= 1  # Bearish (calls more expensive)
        else:
            score += 1  # Bullish (puts more expensive)
        
        # Volume differential
        if volume_diff > 0:
            score -= 1  # Bearish (more call volume)
        else:
            score += 1  # Bullish (more put volume)
        
        # OI differential
        if oi_diff > 0:
            score -= 1  # Bearish (more call OI buildup)
        else:
            score += 1  # Bullish (more put OI buildup)
        
        if score >= 2:
            return "Bullish"
        elif score <= -2:
            return "Bearish"
        else:
            return "Neutral"
    
    def determine_level(self, row: pd.Series) -> str:
        """
        Determine support/resistance level based on OI
        """
        total_oi = row.get('openInterest_CE', 0) + row.get('openInterest_PE', 0)
        
        if total_oi > 1000000:  # 10 lakh
            return "Strong"
        elif total_oi > 500000:  # 5 lakh
            return "Moderate"
        else:
            return "Weak"
    
    def analyze_option_chain(self, option_chain_data: Dict, 
                            expiry: str) -> Tuple[float, pd.DataFrame, Dict]:
        """
        Comprehensive option chain analysis
        Returns: (underlying_price, analysis_df, summary_stats)
        """
        if not option_chain_data or 'data' not in option_chain_data:
            logger.error("Invalid option chain data")
            return None, None, None
        
        data = option_chain_data['data']
        underlying = data['last_price']
        oc_data = data['oc']
        
        # Parse calls and puts
        calls, puts = [], []
        for strike, strike_data in oc_data.items():
            strike_price = float(strike)
            
            if 'ce' in strike_data:
                ce_data = strike_data['ce'].copy()
                ce_data['strikePrice'] = strike_price
                calls.append(ce_data)
            
            if 'pe' in strike_data:
                pe_data = strike_data['pe'].copy()
                pe_data['strikePrice'] = strike_price
                puts.append(pe_data)
        
        # Create dataframes
        df_ce = pd.DataFrame(calls)
        df_pe = pd.DataFrame(puts)
        
        if df_ce.empty or df_pe.empty:
            logger.error("Empty call or put data")
            return underlying, None, None
        
        # Merge on strike price
        df = pd.merge(df_ce, df_pe, on='strikePrice', suffixes=('_CE', '_PE'))
        df = df.sort_values('strikePrice')
        
        # Rename columns to match expected format
        column_mapping = {
            'last_price': 'lastPrice',
            'oi': 'openInterest',
            'previous_oi': 'previousOpenInterest',
            'top_ask_quantity': 'askQty',
            'top_bid_quantity': 'bidQty',
            'volume': 'totalTradedVolume',
            'implied_volatility': 'impliedVolatility'
        }
        
        for old_col, new_col in column_mapping.items():
            if f"{old_col}_CE" in df.columns:
                df.rename(columns={f"{old_col}_CE": f"{new_col}_CE"}, inplace=True)
            if f"{old_col}_PE" in df.columns:
                df.rename(columns={f"{old_col}_PE": f"{new_col}_PE"}, inplace=True)
        
        # Calculate OI changes
        df['changeinOpenInterest_CE'] = df['openInterest_CE'] - df['previousOpenInterest_CE']
        df['changeinOpenInterest_PE'] = df['openInterest_PE'] - df['previousOpenInterest_PE']
        
        # Calculate Greeks
        T = self.calculate_time_to_expiry(expiry)
        r = 0.06  # Risk-free rate
        
        for idx, row in df.iterrows():
            strike = row['strikePrice']
            
            # Get IV with fallback
            iv_ce = row.get('impliedVolatility_CE', 0)
            iv_pe = row.get('impliedVolatility_PE', 0)
            
            if pd.isna(iv_ce) or iv_ce == 0:
                iv_ce, _ = self.get_iv_fallback(df, strike)
            if pd.isna(iv_pe) or iv_pe == 0:
                _, iv_pe = self.get_iv_fallback(df, strike)
            
            iv_ce = iv_ce if iv_ce > 0 else 15
            iv_pe = iv_pe if iv_pe > 0 else 15
            
            # Calculate Greeks
            greeks_ce = self.calculate_greeks('CE', underlying, strike, T, r, iv_ce / 100)
            greeks_pe = self.calculate_greeks('PE', underlying, strike, T, r, iv_pe / 100)
            
            df.at[idx, 'Delta_CE'], df.at[idx, 'Gamma_CE'], df.at[idx, 'Vega_CE'], \
            df.at[idx, 'Theta_CE'], df.at[idx, 'Rho_CE'] = greeks_ce
            
            df.at[idx, 'Delta_PE'], df.at[idx, 'Gamma_PE'], df.at[idx, 'Vega_PE'], \
            df.at[idx, 'Theta_PE'], df.at[idx, 'Rho_PE'] = greeks_pe
        
        # Determine ATM
        atm_strike = min(df['strikePrice'], key=lambda x: abs(x - underlying))
        
        # Add zone classification
        df['Zone'] = df['strikePrice'].apply(
            lambda x: 'ATM' if x == atm_strike else 
                     'ITM' if x < underlying else 'OTM'
        )
        
        df['Level'] = df.apply(self.determine_level, axis=1)
        
        # Calculate PCR
        df['PCR'] = self.calculate_pcr(df)
        df['PCR'] = df['PCR'].round(2)
        df['PCR_Signal'] = np.where(
            df['PCR'] > 1.2, "Bullish",
            np.where(df['PCR'] < 0.7, "Bearish", "Neutral")
        )
        
        # Calculate bias for each strike
        bias_results = []
        for _, row in df.iterrows():
            bid_ask_pressure, pressure_bias = self.calculate_bid_ask_pressure(
                row.get('bidQty_CE', 0), row.get('askQty_CE', 0),
                row.get('bidQty_PE', 0), row.get('askQty_PE', 0)
            )
            
            row_data = {
                "Strike": row['strikePrice'],
                "Zone": row['Zone'],
                "Level": row['Level'],
                "ChgOI_Bias": "Bullish" if row.get('changeinOpenInterest_CE', 0) < row.get('changeinOpenInterest_PE', 0) else "Bearish",
                "Volume_Bias": "Bullish" if row.get('totalTradedVolume_CE', 0) < row.get('totalTradedVolume_PE', 0) else "Bearish",
                "AskQty_Bias": "Bullish" if row.get('askQty_PE', 0) > row.get('askQty_CE', 0) else "Bearish",
                "BidQty_Bias": "Bearish" if row.get('bidQty_PE', 0) > row.get('bidQty_CE', 0) else "Bullish",
                "DVP_Bias": self.delta_volume_bias(
                    row.get('lastPrice_CE', 0) - row.get('lastPrice_PE', 0),
                    row.get('totalTradedVolume_CE', 0) - row.get('totalTradedVolume_PE', 0),
                    row.get('changeinOpenInterest_CE', 0) - row.get('changeinOpenInterest_PE', 0)
                ),
                "BidAskPressure": bid_ask_pressure,
                "PressureBias": pressure_bias,
                "PCR": row['PCR'],
                "PCR_Signal": row['PCR_Signal']
            }
            
            # Calculate weighted bias score
            score = 0
            for k in row_data:
                if "_Bias" in k or k == "PCR_Signal":
                    bias = row_data[k]
                    weight = self.weights.get(k, 1)
                    score += weight if bias == "Bullish" else -weight
            
            row_data["BiasScore"] = round(score, 2)
            row_data["Verdict"] = self.final_verdict(score)
            
            bias_results.append(row_data)
        
        df_summary = pd.DataFrame(bias_results)
        
        # Merge with original data
        df_summary = pd.merge(
            df_summary,
            df[['strikePrice', 'openInterest_CE', 'openInterest_PE', 'lastPrice_CE', 'lastPrice_PE']],
            left_on='Strike', right_on='strikePrice', how='left'
        )
        
        # Calculate summary statistics
        total_ce_oi = df['openInterest_CE'].sum()
        total_pe_oi = df['openInterest_PE'].sum()
        total_ce_change = df['changeinOpenInterest_CE'].sum()
        total_pe_change = df['changeinOpenInterest_PE'].sum()
        total_ce_volume = df['totalTradedVolume_CE'].sum()
        total_pe_volume = df['totalTradedVolume_PE'].sum()
        
        overall_pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
        
        # Overall bias from option chain
        bullish_verdicts = len(df_summary[df_summary['Verdict'] == 'Strong Bullish']) + \
                          len(df_summary[df_summary['Verdict'] == 'Bullish'])
        bearish_verdicts = len(df_summary[df_summary['Verdict'] == 'Strong Bearish']) + \
                          len(df_summary[df_summary['Verdict'] == 'Bearish'])
        
        if bullish_verdicts > bearish_verdicts:
            options_bias = "BULLISH"
        elif bearish_verdicts > bullish_verdicts:
            options_bias = "BEARISH"
        else:
            options_bias = "NEUTRAL"
        
        summary_stats = {
            'total_ce_oi': total_ce_oi,
            'total_pe_oi': total_pe_oi,
            'total_ce_change': total_ce_change,
            'total_pe_change': total_pe_change,
            'total_ce_volume': total_ce_volume,
            'total_pe_volume': total_pe_volume,
            'overall_pcr': round(overall_pcr, 2),
            'options_bias': options_bias,
            'atm_strike': atm_strike
        }
        
        return underlying, df_summary, summary_stats
    
    def final_verdict(self, score: float) -> str:
        """
        Determine final verdict from bias score
        """
        if score >= 8:
            return "Strong Bullish"
        elif score >= 4:
            return "Bullish"
        elif score <= -8:
            return "Strong Bearish"
        elif score <= -4:
            return "Bearish"
        else:
            return "Neutral"
