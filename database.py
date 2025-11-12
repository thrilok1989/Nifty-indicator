import pandas as pd
from supabase import create_client, Client
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class SupabaseDB:
    """
    Database handler for storing trading data
    """
    
    def __init__(self, url: str, key: str):
        self.client: Client = create_client(url, key)
        
    def create_tables(self):
        """
        Create necessary tables (run this once)
        Note: Execute these SQL commands in Supabase SQL editor
        """
        tables_sql = """
        -- Market data table
        CREATE TABLE IF NOT EXISTS market_data (
            id BIGSERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            open DECIMAL(10, 2),
            high DECIMAL(10, 2),
            low DECIMAL(10, 2),
            close DECIMAL(10, 2),
            volume BIGINT,
            vwap DECIMAL(10, 2),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        CREATE INDEX idx_market_data_symbol_time ON market_data(symbol, timestamp DESC);
        
        -- Trading signals table
        CREATE TABLE IF NOT EXISTS trading_signals (
            id BIGSERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            signal_type VARCHAR(20) NOT NULL, -- 'GET_READY', 'ENTRY', 'EXIT'
            direction VARCHAR(10) NOT NULL, -- 'BULL', 'BEAR'
            entry_price DECIMAL(10, 2),
            stop_loss DECIMAL(10, 2),
            target DECIMAL(10, 2),
            bias_pct DECIMAL(5, 2),
            market_condition VARCHAR(20),
            vob_level DECIMAL(10, 2),
            pivot_level DECIMAL(10, 2),
            confidence VARCHAR(10),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        CREATE INDEX idx_signals_symbol_time ON trading_signals(symbol, timestamp DESC);
        
        -- Trades table
        CREATE TABLE IF NOT EXISTS trades (
            id BIGSERIAL PRIMARY KEY,
            signal_id BIGINT REFERENCES trading_signals(id),
            order_id VARCHAR(50),
            symbol VARCHAR(20) NOT NULL,
            direction VARCHAR(10) NOT NULL,
            entry_price DECIMAL(10, 2) NOT NULL,
            entry_time TIMESTAMPTZ NOT NULL,
            exit_price DECIMAL(10, 2),
            exit_time TIMESTAMPTZ,
            stop_loss DECIMAL(10, 2),
            target DECIMAL(10, 2),
            quantity INTEGER NOT NULL,
            pnl DECIMAL(10, 2),
            pnl_pct DECIMAL(5, 2),
            status VARCHAR(20), -- 'OPEN', 'CLOSED', 'SL_HIT', 'TARGET_HIT'
            exit_reason VARCHAR(50),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        CREATE INDEX idx_trades_symbol_time ON trades(symbol, entry_time DESC);
        
        -- Market bias history
        CREATE TABLE IF NOT EXISTS market_bias (
            id BIGSERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            bias VARCHAR(20) NOT NULL,
            bullish_pct DECIMAL(5, 2),
            bearish_pct DECIMAL(5, 2),
            confidence VARCHAR(10),
            mode VARCHAR(20), -- 'NORMAL', 'REVERSAL'
            market_condition VARCHAR(20),
            fast_bull INT,
            fast_bear INT,
            medium_bull INT,
            medium_bear INT,
            slow_bull INT,
            slow_bear INT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        CREATE INDEX idx_bias_symbol_time ON market_bias(symbol, timestamp DESC);
        
        -- Options chain snapshot
        CREATE TABLE IF NOT EXISTS options_snapshot (
            id BIGSERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL,
            expiry DATE NOT NULL,
            underlying_price DECIMAL(10, 2),
            total_ce_oi BIGINT,
            total_pe_oi BIGINT,
            total_ce_change BIGINT,
            total_pe_change BIGINT,
            overall_pcr DECIMAL(4, 2),
            options_bias VARCHAR(20),
            atm_strike DECIMAL(10, 2),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        CREATE INDEX idx_options_time ON options_snapshot(timestamp DESC);
        
        -- Stock correlations
        CREATE TABLE IF NOT EXISTS stock_correlations (
            id BIGSERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL,
            nifty_change DECIMAL(5, 2),
            weighted_avg_daily DECIMAL(5, 2),
            weighted_avg_tf1 DECIMAL(5, 2),
            weighted_avg_tf2 DECIMAL(5, 2),
            bullish_stocks INT,
            total_stocks INT,
            market_breadth DECIMAL(5, 2),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        CREATE INDEX idx_correlations_time ON stock_correlations(timestamp DESC);
        
        -- Performance metrics
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id BIGSERIAL PRIMARY KEY,
            date DATE NOT NULL UNIQUE,
            symbol VARCHAR(20) NOT NULL,
            total_trades INT,
            winning_trades INT,
            losing_trades INT,
            win_rate DECIMAL(5, 2),
            total_pnl DECIMAL(10, 2),
            avg_win DECIMAL(10, 2),
            avg_loss DECIMAL(10, 2),
            max_drawdown DECIMAL(10, 2),
            sharpe_ratio DECIMAL(5, 2),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        CREATE INDEX idx_performance_date ON performance_metrics(date DESC);
        """
        
        logger.info("Tables SQL schema created. Execute in Supabase SQL editor.")
        return tables_sql
    
    # ========================================================================
    # MARKET DATA METHODS
    # ========================================================================
    
    def save_market_data(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Save OHLCV market data"""
        try:
            records = []
            for idx, row in data.iterrows():
                records.append({
                    'timestamp': idx.isoformat() if isinstance(idx, datetime) else str(idx),
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': int(row['volume']),
                    'vwap': float(row.get('vwap', 0))
                })
            
            if records:
                self.client.table('market_data').insert(records).execute()
                logger.info(f"Saved {len(records)} market data records for {symbol}")
        except Exception as e:
            logger.error(f"Error saving market data: {e}")
    
    def get_market_data(self, symbol: str, timeframe: str, 
                       hours_back: int = 24) -> pd.DataFrame:
        """Retrieve market data"""
        try:
            cutoff = (datetime.now() - timedelta(hours=hours_back)).isoformat()
            
            response = self.client.table('market_data')\
                .select('*')\
                .eq('symbol', symbol)\
                .eq('timeframe', timeframe)\
                .gte('timestamp', cutoff)\
                .order('timestamp', desc=False)\
                .execute()
            
            if response.data:
                df = pd.DataFrame(response.data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                return df
            
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error retrieving market data: {e}")
            return pd.DataFrame()
    
    # ========================================================================
    # TRADING SIGNALS METHODS
    # ========================================================================
    
    def save_signal(self, signal_data: Dict) -> int:
        """Save trading signal and return signal ID"""
        try:
            response = self.client.table('trading_signals').insert(signal_data).execute()
            
            if response.data and len(response.data) > 0:
                signal_id = response.data[0]['id']
                logger.info(f"Saved signal ID: {signal_id}")
                return signal_id
            
            return None
        except Exception as e:
            logger.error(f"Error saving signal: {e}")
            return None
    
    def get_recent_signals(self, symbol: str, hours_back: int = 24) -> pd.DataFrame:
        """Get recent signals"""
        try:
            cutoff = (datetime.now() - timedelta(hours=hours_back)).isoformat()
            
            response = self.client.table('trading_signals')\
                .select('*')\
                .eq('symbol', symbol)\
                .gte('timestamp', cutoff)\
                .order('timestamp', desc=True)\
                .execute()
            
            if response.data:
                return pd.DataFrame(response.data)
            
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error retrieving signals: {e}")
            return pd.DataFrame()
    
    # ========================================================================
    # TRADES METHODS
    # ========================================================================
    
    def save_trade(self, trade_data: Dict) -> int:
        """Save trade and return trade ID"""
        try:
            response = self.client.table('trades').insert(trade_data).execute()
            
            if response.data and len(response.data) > 0:
                trade_id = response.data[0]['id']
                logger.info(f"Saved trade ID: {trade_id}")
                return trade_id
            
            return None
        except Exception as e:
            logger.error(f"Error saving trade: {e}")
            return None
    
    def update_trade(self, trade_id: int, update_data: Dict):
        """Update existing trade"""
        try:
            self.client.table('trades')\
                .update(update_data)\
                .eq('id', trade_id)\
                .execute()
            
            logger.info(f"Updated trade ID: {trade_id}")
        except Exception as e:
            logger.error(f"Error updating trade: {e}")
    
    def get_open_trades(self, symbol: str) -> pd.DataFrame:
        """Get open trades"""
        try:
            response = self.client.table('trades')\
                .select('*')\
                .eq('symbol', symbol)\
                .eq('status', 'OPEN')\
                .order('entry_time', desc=True)\
                .execute()
            
            if response.data:
                return pd.DataFrame(response.data)
            
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error retrieving open trades: {e}")
            return pd.DataFrame()
    
    def get_trade_history(self, symbol: str, days_back: int = 30) -> pd.DataFrame:
        """Get trade history"""
        try:
            cutoff = (datetime.now() - timedelta(days=days_back)).isoformat()
            
            response = self.client.table('trades')\
                .select('*')\
                .eq('symbol', symbol)\
                .gte('entry_time', cutoff)\
                .order('entry_time', desc=True)\
                .execute()
            
            if response.data:
                return pd.DataFrame(response.data)
            
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error retrieving trade history: {e}")
            return pd.DataFrame()
    
    # ========================================================================
    # MARKET BIAS METHODS
    # ========================================================================
    
    def save_market_bias(self, bias_data: Dict):
        """Save market bias snapshot"""
        try:
            self.client.table('market_bias').insert(bias_data).execute()
            logger.debug("Saved market bias")
        except Exception as e:
            logger.error(f"Error saving market bias: {e}")
    
    def get_bias_history(self, symbol: str, hours_back: int = 24) -> pd.DataFrame:
        """Get bias history"""
        try:
            cutoff = (datetime.now() - timedelta(hours=hours_back)).isoformat()
            
            response = self.client.table('market_bias')\
                .select('*')\
                .eq('symbol', symbol)\
                .gte('timestamp', cutoff)\
                .order('timestamp', desc=False)\
                .execute()
            
            if response.data:
                return pd.DataFrame(response.data)
            
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error retrieving bias history: {e}")
            return pd.DataFrame()
    
    # ========================================================================
    # OPTIONS METHODS
    # ========================================================================
    
    def save_options_snapshot(self, snapshot_data: Dict):
        """Save options chain snapshot"""
        try:
            self.client.table('options_snapshot').insert(snapshot_data).execute()
            logger.debug("Saved options snapshot")
        except Exception as e:
            logger.error(f"Error saving options snapshot: {e}")
    
    def get_options_history(self, hours_back: int = 24) -> pd.DataFrame:
        """Get options history"""
        try:
            cutoff = (datetime.now() - timedelta(hours=hours_back)).isoformat()
            
            response = self.client.table('options_snapshot')\
                .select('*')\
                .gte('timestamp', cutoff)\
                .order('timestamp', desc=False)\
                .execute()
            
            if response.data:
                return pd.DataFrame(response.data)
            
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error retrieving options history: {e}")
            return pd.DataFrame()
    
    # ========================================================================
    # ANALYTICS METHODS
    # ========================================================================
    
    def calculate_performance_metrics(self, symbol: str, date: datetime.date) -> Dict:
        """Calculate daily performance metrics"""
        try:
            # Get trades for the day
            day_start = datetime.combine(date, datetime.min.time()).isoformat()
            day_end = datetime.combine(date, datetime.max.time()).isoformat()
            
            response = self.client.table('trades')\
                .select('*')\
                .eq('symbol', symbol)\
                .gte('entry_time', day_start)\
                .lte('entry_time', day_end)\
                .eq('status', 'CLOSED')\
                .execute()
            
            if not response.data:
                return None
            
            trades_df = pd.DataFrame(response.data)
            
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] < 0])
            
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            total_pnl = trades_df['pnl'].sum()
            
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 0
            
            # Calculate max drawdown
            cumulative_pnl = trades_df['pnl'].cumsum()
            running_max = cumulative_pnl.cummax()
            drawdown = running_max - cumulative_pnl
            max_drawdown = drawdown.max() if len(drawdown) > 0 else 0
            
            # Simple Sharpe ratio calculation
            returns = trades_df['pnl_pct']
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if len(returns) > 1 else 0
            
            metrics = {
                'date': date.isoformat(),
                'symbol': symbol,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': round(win_rate, 2),
                'total_pnl': round(total_pnl, 2),
                'avg_win': round(avg_win, 2),
                'avg_loss': round(avg_loss, 2),
                'max_drawdown': round(max_drawdown, 2),
                'sharpe_ratio': round(sharpe_ratio, 2)
            }
            
            # Save metrics
            self.client.table('performance_metrics')\
                .upsert(metrics, on_conflict='date,symbol')\
                .execute()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return None
    
    def get_performance_summary(self, symbol: str, days_back: int = 30) -> pd.DataFrame:
        """Get performance summary"""
        try:
            cutoff = (datetime.now() - timedelta(days=days_back)).date().isoformat()
            
            response = self.client.table('performance_metrics')\
                .select('*')\
                .eq('symbol', symbol)\
                .gte('date', cutoff)\
                .order('date', desc=False)\
                .execute()
            
            if response.data:
                return pd.DataFrame(response.data)
            
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error retrieving performance summary: {e}")
            return pd.DataFrame()
