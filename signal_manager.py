import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from enum import Enum
import json

logger = logging.getLogger(__name__)


class SignalStatus(Enum):
    """Signal status states"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    EXECUTED = "executed"


class TradingSignal:
    """
    Trading signal object with approval workflow
    """
    
    def __init__(self, signal_data: Dict):
        self.signal_id = signal_data.get('signal_id', f"sig_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.timestamp = signal_data.get('timestamp', datetime.now())
        self.symbol = signal_data['symbol']
        self.direction = signal_data['direction']  # BULL or BEAR
        self.signal_type = signal_data['signal_type']  # GET_READY, ENTRY, EXIT
        
        # Price levels
        self.current_price = signal_data['current_price']
        self.entry_price = signal_data.get('entry_price', self.current_price)
        self.stop_loss = signal_data['stop_loss']
        self.target = signal_data['target']
        
        # Context
        self.bias_pct = signal_data.get('bias_pct', 0)
        self.confidence = signal_data.get('confidence', 'MEDIUM')
        self.market_condition = signal_data.get('market_condition', 'NEUTRAL')
        self.vob_level = signal_data.get('vob_level')
        self.pivot_level = signal_data.get('pivot_level')
        
        # Additional metadata
        self.indicators = signal_data.get('indicators', {})
        self.reason = signal_data.get('reason', '')
        
        # Status tracking
        self.status = SignalStatus.PENDING
        self.created_at = datetime.now()
        self.expires_at = self.created_at + timedelta(minutes=15)  # 15 min expiry
        self.approved_at = None
        self.rejected_at = None
        self.executed_at = None
        
        # Order details (filled after approval)
        self.order_id = None
        self.trade_id = None
        self.quantity = None
    
    def is_expired(self) -> bool:
        """Check if signal has expired"""
        return datetime.now() > self.expires_at and self.status == SignalStatus.PENDING
    
    def approve(self, quantity: int):
        """Approve signal for execution"""
        if self.status != SignalStatus.PENDING:
            logger.warning(f"Cannot approve signal {self.signal_id} - status is {self.status}")
            return False
        
        if self.is_expired():
            self.status = SignalStatus.EXPIRED
            logger.warning(f"Signal {self.signal_id} has expired")
            return False
        
        self.status = SignalStatus.APPROVED
        self.approved_at = datetime.now()
        self.quantity = quantity
        logger.info(f"Signal {self.signal_id} approved for {quantity} quantity")
        return True
    
    def reject(self, reason: str = ""):
        """Reject signal"""
        if self.status != SignalStatus.PENDING:
            logger.warning(f"Cannot reject signal {self.signal_id} - status is {self.status}")
            return False
        
        self.status = SignalStatus.REJECTED
        self.rejected_at = datetime.now()
        logger.info(f"Signal {self.signal_id} rejected: {reason}")
        return True
    
    def mark_executed(self, order_id: str, trade_id: int):
        """Mark signal as executed"""
        self.status = SignalStatus.EXECUTED
        self.executed_at = datetime.now()
        self.order_id = order_id
        self.trade_id = trade_id
        logger.info(f"Signal {self.signal_id} executed - Order: {order_id}")
    
    def to_dict(self) -> Dict:
        """Convert signal to dictionary"""
        return {
            'signal_id': self.signal_id,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'direction': self.direction,
            'signal_type': self.signal_type,
            'current_price': self.current_price,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'target': self.target,
            'bias_pct': self.bias_pct,
            'confidence': self.confidence,
            'market_condition': self.market_condition,
            'vob_level': self.vob_level,
            'pivot_level': self.pivot_level,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'approved_at': self.approved_at.isoformat() if self.approved_at else None,
            'quantity': self.quantity,
            'order_id': self.order_id,
            'trade_id': self.trade_id,
            'reason': self.reason
        }


class SignalManager:
    """
    Manages trading signals with approval workflow
    """
    
    def __init__(self, db, telegram, trade_manager, config: dict):
        self.db = db
        self.telegram = telegram
        self.trade_manager = trade_manager
        self.config = config
        
        # Active signals waiting for approval
        self.pending_signals: Dict[str, TradingSignal] = {}
        
        # Historical signals
        self.signal_history: List[TradingSignal] = []
        
        # Trading mode
        self.trading_mode = config.get('TRADING_MODE', 'ANALYSIS_ONLY')
        # ANALYSIS_ONLY, SEMI_AUTOMATIC, FULLY_AUTOMATIC
        
        logger.info(f"SignalManager initialized in {self.trading_mode} mode")
    
    def generate_signal(self, signal_data: Dict) -> TradingSignal:
        """
        Generate a new trading signal
        """
        signal = TradingSignal(signal_data)
        
        # Save to database if available
        if self.db:
            db_signal_data = {
                'timestamp': signal.timestamp.isoformat(),
                'symbol': signal.symbol,
                'signal_type': signal.signal_type,
                'direction': signal.direction,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'target': signal.target,
                'bias_pct': signal.bias_pct,
                'market_condition': signal.market_condition,
                'vob_level': signal.vob_level,
                'pivot_level': signal.pivot_level,
                'confidence': signal.confidence
            }
            
            signal_id_db = self.db.save_signal(db_signal_data)
            if signal_id_db:
                signal.signal_id = f"sig_{signal_id_db}"
        
        # Send Telegram notification based on signal type
        if signal.signal_type == "GET_READY":
            self.telegram.send_get_ready_alert(
                direction=signal.direction,
                symbol=signal.symbol,
                bias_pct=signal.bias_pct,
                condition=signal.market_condition
            )
        
        elif signal.signal_type == "ENTRY":
            # Add to pending signals (waiting for approval)
            self.pending_signals[signal.signal_id] = signal
            
            # Send approval request alert
            self._send_approval_request(signal)
            
            logger.info(f"New {signal.direction} entry signal generated: {signal.signal_id}")
        
        # Add to history
        self.signal_history.append(signal)
        
        return signal
    
    def _send_approval_request(self, signal: TradingSignal):
        """
        Send Telegram message requesting approval
        """
        risk = abs(signal.entry_price - signal.stop_loss)
        reward = abs(signal.target - signal.entry_price)
        rr_ratio = reward / risk if risk > 0 else 0
        
        emoji = "🐂" if signal.direction == "BULL" else "🐻"
        
        message = f"""
<b>{emoji} 🚨 TRADE SIGNAL - APPROVAL REQUIRED</b>

<b>Signal ID:</b> <code>{signal.signal_id}</code>
<b>Symbol:</b> {signal.symbol}
<b>Direction:</b> {signal.direction}

<b>Entry Price:</b> ₹{signal.entry_price:.2f}
<b>Stop Loss:</b> ₹{signal.stop_loss:.2f} (-{risk:.2f})
<b>Target:</b> ₹{signal.target:.2f} (+{reward:.2f})
<b>Risk:Reward:</b> 1:{rr_ratio:.1f}

<b>Bias Strength:</b> {signal.bias_pct:.1f}%
<b>Confidence:</b> {signal.confidence}
<b>Market Condition:</b> {signal.market_condition}
"""
        
        if signal.vob_level:
            message += f"<b>VOB Level:</b> ₹{signal.vob_level:.2f}\n"
        
        message += f"\n<b>⏰ Expires in 15 minutes</b>"
        message += f"\n\n<i>Go to dashboard to approve/reject this signal</i>"
        message += f"\n<code>Time: {datetime.now().strftime('%H:%M:%S')}</code>"
        
        self.telegram.send_message(message)
    
    def approve_signal(self, signal_id: str, quantity: int) -> bool:
        """
        Approve a pending signal and execute trade
        """
        if signal_id not in self.pending_signals:
            logger.error(f"Signal {signal_id} not found in pending signals")
            return False
        
        signal = self.pending_signals[signal_id]
        
        # Approve signal
        if not signal.approve(quantity):
            return False
        
        # Execute trade based on mode
        if self.trading_mode == "SEMI_AUTOMATIC":
            # Place order
            order_result = self.trade_manager.place_entry_order(
                signal_data=signal.to_dict(),
                quantity=quantity,
                use_super_order=True
            )
            
            if order_result:
                signal.mark_executed(
                    order_id=order_result['order_id'],
                    trade_id=order_result['trade_id']
                )
                
                # Remove from pending
                del self.pending_signals[signal_id]
                
                # Send confirmation
                self.telegram.send_message(f"""
<b>✅ TRADE EXECUTED</b>

<b>Signal ID:</b> <code>{signal_id}</code>
<b>Order ID:</b> <code>{order_result['order_id']}</code>
<b>Quantity:</b> {quantity}
<b>Status:</b> Order placed successfully

<code>Time: {datetime.now().strftime('%H:%M:%S')}</code>
""")
                
                logger.info(f"Signal {signal_id} approved and executed")
                return True
            else:
                # Order failed - revert to pending
                signal.status = SignalStatus.PENDING
                
                self.telegram.send_error_alert(
                    error_type="Order Execution Failed",
                    error_message=f"Failed to place order for signal {signal_id}"
                )
                
                return False
        
        return True
    
    def reject_signal(self, signal_id: str, reason: str = "") -> bool:
        """
        Reject a pending signal
        """
        if signal_id not in self.pending_signals:
            logger.error(f"Signal {signal_id} not found in pending signals")
            return False
        
        signal = self.pending_signals[signal_id]
        
        if signal.reject(reason):
            # Remove from pending
            del self.pending_signals[signal_id]
            
            # Send notification
            self.telegram.send_message(f"""
<b>❌ SIGNAL REJECTED</b>

<b>Signal ID:</b> <code>{signal_id}</code>
<b>Reason:</b> {reason if reason else 'User rejected'}

<code>Time: {datetime.now().strftime('%H:%M:%S')}</code>
""")
            
            logger.info(f"Signal {signal_id} rejected: {reason}")
            return True
        
        return False
    
    def cleanup_expired_signals(self):
        """
        Clean up expired signals
        """
        expired_ids = []
        
        for signal_id, signal in self.pending_signals.items():
            if signal.is_expired():
                signal.status = SignalStatus.EXPIRED
                expired_ids.append(signal_id)
                
                # Send expiry notification
                self.telegram.send_message(f"""
<b>⏰ SIGNAL EXPIRED</b>

<b>Signal ID:</b> <code>{signal_id}</code>
<b>Symbol:</b> {signal.symbol}
<b>Direction:</b> {signal.direction}

<i>Signal expired without approval</i>

<code>Time: {datetime.now().strftime('%H:%M:%S')}</code>
""")
        
        # Remove expired signals
        for signal_id in expired_ids:
            del self.pending_signals[signal_id]
        
        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired signals")
    
    def get_pending_signals(self) -> List[TradingSignal]:
        """Get all pending signals"""
        return list(self.pending_signals.values())
    
    def get_signal_by_id(self, signal_id: str) -> Optional[TradingSignal]:
        """Get signal by ID"""
        return self.pending_signals.get(signal_id)
    
    def get_signal_history(self, hours_back: int = 24) -> List[TradingSignal]:
        """Get signal history"""
        cutoff = datetime.now() - timedelta(hours=hours_back)
        return [s for s in self.signal_history if s.created_at >= cutoff]
