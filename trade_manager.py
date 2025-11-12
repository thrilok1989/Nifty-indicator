import logging
from typing import Dict, Optional, Tuple
from datetime import datetime
from dhan_client import DhanClient
from database import SupabaseDB
from telegram_bot import TelegramBot

logger = logging.getLogger(__name__)


class TradeManager:
    """
    Manages trade execution and position tracking
    """
    
    def __init__(self, dhan: DhanClient, db: SupabaseDB, 
                 telegram: TelegramBot, config: dict):
        self.dhan = dhan
        self.db = db
        self.telegram = telegram
        self.config = config
        
        # Trade settings
        self.use_atr_sl = config.get('USE_ATR_STOP_LOSS', True)
        self.atr_multiplier = config.get('ATR_MULTIPLIER', 1.5)
        self.use_fixed_sl = config.get('USE_FIXED_STOP_LOSS', False)
        self.fixed_sl_pct = config.get('FIXED_STOP_LOSS_PCT', 1.0)
        self.rr_ratio = config.get('RISK_REWARD_RATIO', 2.0)
        
        # Track open positions
        self.open_positions = {}
        
    def calculate_stop_loss(self, entry_price: float, direction: str,
                           atr: float, pivot_level: Optional[float] = None) -> float:
        """
        Calculate stop loss based on configured method
        """
        if self.use_atr_sl:
            # ATR-based stop loss
            if direction == "BULL":
                sl = entry_price - (atr * self.atr_multiplier)
            else:  # BEAR
                sl = entry_price + (atr * self.atr_multiplier)
        
        elif self.use_fixed_sl:
            # Fixed percentage stop loss
            if direction == "BULL":
                sl = entry_price * (1 - self.fixed_sl_pct / 100)
            else:  # BEAR
                sl = entry_price * (1 + self.fixed_sl_pct / 100)
        
        elif pivot_level:
            # Pivot-based stop loss
            if direction == "BULL":
                sl = pivot_level - 10  # 10 points below support
            else:  # BEAR
                sl = pivot_level + 10  # 10 points above resistance
        
        else:
            # Default: 1% stop loss
            if direction == "BULL":
                sl = entry_price * 0.99
            else:  # BEAR
                sl = entry_price * 1.01
        
        return round(sl, 2)
    
    def calculate_target(self, entry_price: float, stop_loss: float,
                        direction: str, range_target: Optional[float] = None) -> float:
        """
        Calculate target price based on risk-reward ratio
        """
        if range_target:
            # Use range boundary as target
            return range_target
        
        # Calculate target based on R:R ratio
        risk = abs(entry_price - stop_loss)
        reward = risk * self.rr_ratio
        
        if direction == "BULL":
            target = entry_price + reward
        else:  # BEAR
            target = entry_price - reward
        
        return round(target, 2)
    
    def check_entry_conditions(self, signal_data: Dict, current_price: float,
                              current_low: float, current_high: float,
                              bias: str) -> bool:
        """
        Check if entry conditions are met
        """
        direction = signal_data['direction']
        
        # Check bias alignment
        if direction == "BULL" and bias != "BULLISH":
            logger.info(f"Entry blocked: Bias is {bias}, not BULLISH")
            return False
        
        if direction == "BEAR" and bias != "BEARISH":
            logger.info(f"Entry blocked: Bias is {bias}, not BEARISH")
            return False
        
        # Check pivot/VOB touch
        if 'vob_level' in signal_data and signal_data['vob_level']:
            vob_level = signal_data['vob_level']
            tolerance = vob_level * 0.002  # 0.2% tolerance
            
            if direction == "BULL":
                # For bull entry, price should touch VOB support
                if not (current_low <= vob_level + tolerance):
                    return False
            else:  # BEAR
                # For bear entry, price should touch VOB resistance
                if not (current_high >= vob_level - tolerance):
                    return False
        
        # Check candle confirmation
        if direction == "BULL" and current_price < signal_data.get('entry_price', current_price) * 0.999:
            return False  # Wait for bullish candle
        
        if direction == "BEAR" and current_price > signal_data.get('entry_price', current_price) * 1.001:
            return False  # Wait for bearish candle
        
        return True
    
    def place_entry_order(self, signal_data: Dict, quantity: int,
                         use_super_order: bool = True) -> Optional[Dict]:
        """
        Place entry order (regular or super order)
        """
        symbol = signal_data['symbol']
        direction = signal_data['direction']
        entry_price = signal_data['entry_price']
        stop_loss = signal_data['stop_loss']
        target = signal_data['target']
        
        # Get security ID
        security_id = self.config['STOCKS_CONFIG'].get(symbol, {}).get('security_id')
        if not security_id:
            if symbol == "NIFTY50":
                security_id = self.config['NIFTY_SECURITY_ID']
            else:
                logger.error(f"Security ID not found for {symbol}")
                return None
        
        transaction_type = "BUY" if direction == "BULL" else "SELL"
        
        try:
            if use_super_order:
                # Place Super Order (Entry + SL + Target in one)
                logger.info(f"Placing Super Order: {transaction_type} {quantity} {symbol}")
                
                response = self.dhan.place_super_order(
                    transaction_type=transaction_type,
                    exchange_segment="NSE_FNO" if "NIFTY" in symbol else "NSE_EQ",
                    product_type="INTRADAY",
                    order_type="LIMIT",
                    security_id=security_id,
                    quantity=quantity,
                    price=entry_price,
                    target_price=target,
                    stop_loss_price=stop_loss,
                    trailing_jump=5.0,  # Trail SL by 5 points
                    correlation_id=f"signal_{signal_data.get('signal_id', 'manual')}"
                )
            else:
                # Place regular order
                logger.info(f"Placing Regular Order: {transaction_type} {quantity} {symbol}")
                
                response = self.dhan.place_order(
                    transaction_type=transaction_type,
                    exchange_segment="NSE_FNO" if "NIFTY" in symbol else "NSE_EQ",
                    product_type="INTRADAY",
                    order_type="LIMIT",
                    security_id=security_id,
                    quantity=quantity,
                    price=entry_price,
                    validity="DAY",
                    correlation_id=f"signal_{signal_data.get('signal_id', 'manual')}"
                )
            
            if 'error' not in response:
                order_id = response.get('orderId')
                logger.info(f"Order placed successfully: {order_id}")
                
                # Save trade to database
                trade_data = {
                    'signal_id': signal_data.get('signal_id'),
                    'order_id': order_id,
                    'symbol': symbol,
                    'direction': direction,
                    'entry_price': entry_price,
                    'entry_time': datetime.now().isoformat(),
                    'stop_loss': stop_loss,
                    'target': target,
                    'quantity': quantity,
                    'status': 'OPEN'
                }
                
                trade_id = self.db.save_trade(trade_data)
                
                # Track position
                self.open_positions[order_id] = {
                    'trade_id': trade_id,
                    'signal_data': signal_data,
                    'quantity': quantity,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'target': target
                }
                
                # Send Telegram alert
                self.telegram.send_entry_alert(
                    direction=direction,
                    symbol=symbol,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    target=target,
                    entry_type=signal_data.get('signal_type', 'MANUAL'),
                    bias_pct=signal_data.get('bias_pct', 0),
                    vob_level=signal_data.get('vob_level')
                )
                
                return {'order_id': order_id, 'trade_id': trade_id}
            else:
                logger.error(f"Order placement failed: {response.get('error')}")
                self.telegram.send_error_alert(
                    error_type="Order Placement Failed",
                    error_message=response.get('error')
                )
                return None
                
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            self.telegram.send_error_alert(
                error_type="Order Placement Exception",
                error_message=str(e)
            )
            return None
    
    def check_exit_conditions(self, order_id: str, current_price: float,
                             current_high: float, current_low: float,
                             bias: str) -> Tuple[bool, str]:
        """
        Check if exit conditions are met
        Returns: (should_exit, exit_reason)
        """
        if order_id not in self.open_positions:
            return False, None
        
        position = self.open_positions[order_id]
        direction = position['signal_data']['direction']
        stop_loss = position['stop_loss']
        target = position['target']
        
        # Check stop loss hit
        if direction == "BULL" and current_low <= stop_loss:
            return True, "SL_HIT"
        
        if direction == "BEAR" and current_high >= stop_loss:
            return True, "SL_HIT"
        
        # Check target hit
        if direction == "BULL" and current_high >= target:
            return True, "TARGET_HIT"
        
        if direction == "BEAR" and current_low <= target:
            return True, "TARGET_HIT"
        
        # Check bias change exit
        if self.config.get('EXIT_ON_BIAS_CHANGE', True):
            if direction == "BULL" and bias == "BEARISH":
                return True, "BIAS_CHANGE"
            
            if direction == "BEAR" and bias == "BULLISH":
                return True, "BIAS_CHANGE"
        
        return False, None
    
    def close_position(self, order_id: str, exit_price: float, exit_reason: str):
        """
        Close an open position
        """
        if order_id not in self.open_positions:
            logger.warning(f"Position {order_id} not found")
            return
        
        position = self.open_positions[order_id]
        trade_id = position['trade_id']
        direction = position['signal_data']['direction']
        entry_price = position['entry_price']
        quantity = position['quantity']
        
        # Calculate P&L
        if direction == "BULL":
            pnl = (exit_price - entry_price) * quantity
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        else:  # BEAR
            pnl = (entry_price - exit_price) * quantity
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100
        
        # Update trade in database
        update_data = {
            'exit_price': exit_price,
            'exit_time': datetime.now().isoformat(),
            'pnl': round(pnl, 2),
            'pnl_pct': round(pnl_pct, 2),
            'status': 'CLOSED',
            'exit_reason': exit_reason
        }
        
        self.db.update_trade(trade_id, update_data)
        
        # Send Telegram alert
        self.telegram.send_exit_alert(
            direction=direction,
            symbol=position['signal_data']['symbol'],
            entry_price=entry_price,
            exit_price=exit_price,
            exit_reason=exit_reason,
            pnl=pnl,
            pnl_pct=pnl_pct
        )
        
        # Remove from open positions
        del self.open_positions[order_id]
        
        logger.info(f"Position closed: {order_id}, P&L: ₹{pnl:.2f} ({pnl_pct:+.2f}%)")
    
    def get_open_positions_summary(self) -> Dict:
        """
        Get summary of open positions
        """
        if not self.open_positions:
            return {'count': 0, 'positions': []}
        
        positions_list = []
        for order_id, position in self.open_positions.items():
            positions_list.append({
                'order_id': order_id,
                'symbol': position['signal_data']['symbol'],
                'direction': position['signal_data']['direction'],
                'entry_price': position['entry_price'],
                'stop_loss': position['stop_loss'],
                'target': position['target'],
                'quantity': position['quantity']
            })
        
        return {
            'count': len(self.open_positions),
            'positions': positions_list
        }
