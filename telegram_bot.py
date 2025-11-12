import requests
import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class TelegramBot:
    """
    Telegram bot for sending trading alerts
    """
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        
        if not bot_token or not chat_id:
            logger.warning("Telegram credentials not configured")
            self.enabled = False
        else:
            self.enabled = True
            logger.info("Telegram bot initialized")
    
    def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """Send text message"""
        if not self.enabled:
            logger.warning("Telegram not enabled, skipping message")
            return False
        
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True
            }
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info("Telegram message sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False
    
    def send_get_ready_alert(self, direction: str, symbol: str, 
                            bias_pct: float, condition: str) -> bool:
        """Send GET READY signal alert"""
        emoji = "🐂" if direction == "BULL" else "🐻"
        
        message = f"""
<b>{emoji} ⚠️ GET READY - {direction} Signal</b>

<b>Symbol:</b> {symbol}
<b>Bias Strength:</b> {bias_pct}%
<b>Market Condition:</b> {condition}

<i>Wait for price to touch support/resistance pivot level</i>

<code>Time: {datetime.now().strftime('%H:%M:%S')}</code>
"""
        return self.send_message(message)
    
    def send_entry_alert(self, direction: str, symbol: str, entry_price: float,
                        stop_loss: float, target: float, entry_type: str,
                        bias_pct: float, vob_level: Optional[float] = None) -> bool:
        """Send trade entry alert"""
        emoji = "🐂" if direction == "BULL" else "🐻"
        
        risk = abs(entry_price - stop_loss)
        reward = abs(target - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0
        
        message = f"""
<b>{emoji} 🎯 TRADE ENTRY - {direction}</b>

<b>Symbol:</b> {symbol}
<b>Entry Type:</b> {entry_type}
<b>Entry Price:</b> ₹{entry_price:.2f}
<b>Stop Loss:</b> ₹{stop_loss:.2f} (-{risk:.2f})
<b>Target:</b> ₹{target:.2f} (+{reward:.2f})
<b>Risk:Reward:</b> 1:{rr_ratio:.1f}

<b>Bias Strength:</b> {bias_pct}%
"""
        
        if vob_level:
            message += f"<b>VOB Level:</b> ₹{vob_level:.2f}\n"
        
        message += f"\n<code>Time: {datetime.now().strftime('%H:%M:%S')}</code>"
        
        return self.send_message(message)
    
    def send_exit_alert(self, direction: str, symbol: str, entry_price: float,
                       exit_price: float, exit_reason: str, pnl: float,
                       pnl_pct: float) -> bool:
        """Send trade exit alert"""
        emoji = "✅" if pnl > 0 else "❌"
        
        message = f"""
<b>{emoji} TRADE CLOSED - {direction}</b>

<b>Symbol:</b> {symbol}
<b>Entry:</b> ₹{entry_price:.2f}
<b>Exit:</b> ₹{exit_price:.2f}
<b>Exit Reason:</b> {exit_reason}

<b>P&L:</b> ₹{pnl:.2f} ({pnl_pct:+.2f}%)

<code>Time: {datetime.now().strftime('%H:%M:%S')}</code>
"""
        return self.send_message(message)
    
    def send_vob_touch_alert(self, symbol: str, vob_type: str, level: float,
                            current_price: float, bias: str) -> bool:
        """Send VOB touch alert"""
        emoji = "🔷" if vob_type == "BULL" else "🔶"
        
        message = f"""
<b>{emoji} VOB Zone Touch</b>

<b>Symbol:</b> {symbol}
<b>Type:</b> {vob_type} Order Block
<b>VOB Level:</b> ₹{level:.2f}
<b>Current Price:</b> ₹{current_price:.2f}
<b>Market Bias:</b> {bias}

<i>Watch for entry confirmation</i>

<code>Time: {datetime.now().strftime('%H:%M:%S')}</code>
"""
        return self.send_message(message)
    
    def send_divergence_alert(self, symbol: str, divergence_type: str,
                             fast_pct: float, slow_pct: float) -> bool:
        """Send divergence detection alert"""
        message = f"""
<b>⚠️ DIVERGENCE DETECTED</b>

<b>Symbol:</b> {symbol}
<b>Type:</b> {divergence_type}
<b>Fast Indicators:</b> {fast_pct:.1f}%
<b>Slow Indicators:</b> {slow_pct:.1f}%

<i>Potential reversal warning - watch for confirmation</i>

<code>Time: {datetime.now().strftime('%H:%M:%S')}</code>
"""
        return self.send_message(message)
    
    def send_range_breakout_alert(self, symbol: str, direction: str,
                                  range_high: float, range_low: float,
                                  breakout_price: float, volume_surge: bool) -> bool:
        """Send range breakout alert"""
        emoji = "📈" if direction == "LONG" else "📉"
        
        message = f"""
<b>{emoji} RANGE BREAKOUT - {direction}</b>

<b>Symbol:</b> {symbol}
<b>Range High:</b> ₹{range_high:.2f}
<b>Range Low:</b> ₹{range_low:.2f}
<b>Breakout Price:</b> ₹{breakout_price:.2f}
<b>Volume Surge:</b> {'Yes' if volume_surge else 'No'}

<i>Strong momentum breakout detected</i>

<code>Time: {datetime.now().strftime('%H:%M:%S')}</code>
"""
        return self.send_message(message)
    
    def send_options_bias_alert(self, underlying_price: float, pcr: float,
                               options_bias: str, ce_change: float,
                               pe_change: float) -> bool:
        """Send options chain bias alert"""
        emoji = "📊"
        
        message = f"""
<b>{emoji} OPTIONS CHAIN UPDATE</b>

<b>NIFTY:</b> ₹{underlying_price:.2f}
<b>PCR:</b> {pcr:.2f}
<b>Options Bias:</b> {options_bias}

<b>CALL OI Change:</b> {ce_change/100000:+.1f}L
<b>PUT OI Change:</b> {pe_change/100000:+.1f}L

<code>Time: {datetime.now().strftime('%H:%M:%S')}</code>
"""
        return self.send_message(message)
    
    def send_daily_summary(self, summary_data: Dict) -> bool:
        """Send end-of-day summary"""
        message = f"""
<b>📈 DAILY TRADING SUMMARY</b>

<b>Date:</b> {summary_data.get('date')}
<b>Symbol:</b> {summary_data.get('symbol')}

<b>Total Trades:</b> {summary_data.get('total_trades', 0)}
<b>Winning Trades:</b> {summary_data.get('winning_trades', 0)}
<b>Losing Trades:</b> {summary_data.get('losing_trades', 0)}
<b>Win Rate:</b> {summary_data.get('win_rate', 0)}%

<b>Total P&L:</b> ₹{summary_data.get('total_pnl', 0):.2f}
<b>Avg Win:</b> ₹{summary_data.get('avg_win', 0):.2f}
<b>Avg Loss:</b> ₹{summary_data.get('avg_loss', 0):.2f}

<b>Max Drawdown:</b> ₹{summary_data.get('max_drawdown', 0):.2f}
<b>Sharpe Ratio:</b> {summary_data.get('sharpe_ratio', 0):.2f}
"""
        return self.send_message(message)
    
    def send_error_alert(self, error_type: str, error_message: str) -> bool:
        """Send error notification"""
        message = f"""
<b>❌ ERROR ALERT</b>

<b>Type:</b> {error_type}
<b>Message:</b> {error_message}

<code>Time: {datetime.now().strftime('%H:%M:%S')}</code>
"""
        return self.send_message(message)
    
    def send_system_status(self, status: str, details: str) -> bool:
        """Send system status update"""
        message = f"""
<b>🔧 SYSTEM STATUS</b>

<b>Status:</b> {status}
<b>Details:</b> {details}

<code>Time: {datetime.now().strftime('%H:%M:%S')}</code>
"""
        return self.send_message(message)
