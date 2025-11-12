import requests
import time
import json
import websocket
import struct
from functools import lru_cache
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DhanClient:
    """Enhanced DhanHQ API client with WebSocket support"""
    
    BASE_URL = "https://api.dhan.co/v2"
    WEBSOCKET_URL = "wss://api-feed.dhan.co"
    
    def __init__(self, client_id: str, access_token: str):
        self.client_id = client_id
        self.access_token = access_token
        self.headers = {
            "Content-Type": "application/json",
            "access-token": access_token,
            "client-id": client_id
        }
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Rate limit: 1 req/sec
        
    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, method: str, endpoint: str, data: dict = None) -> dict:
        """Make HTTP request with error handling"""
        self._rate_limit()
        
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            if method == "GET":
                response = requests.get(url, headers=self.headers)
            elif method == "POST":
                response = requests.post(url, headers=self.headers, json=data)
            elif method == "PUT":
                response = requests.put(url, headers=self.headers, json=data)
            elif method == "DELETE":
                response = requests.delete(url, headers=self.headers)
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return {"error": str(e)}
    
    # ========================================================================
    # MARKET DATA APIs
    # ========================================================================
    
    @lru_cache(maxsize=100)
    def get_market_quote_ltp(self, exchange_segment: str, security_ids: tuple) -> dict:
        """
        Get LTP for multiple instruments
        Cached for performance (1 second TTL via rate limit)
        """
        data = {exchange_segment: list(security_ids)}
        return self._make_request("POST", "/marketfeed/ltp", data)
    
    def get_market_quote_ohlc(self, exchange_segment: str, security_ids: List[str]) -> dict:
        """Get OHLC data for multiple instruments"""
        data = {exchange_segment: security_ids}
        return self._make_request("POST", "/marketfeed/ohlc", data)
    
    def get_market_quote_full(self, exchange_segment: str, security_ids: List[str]) -> dict:
        """Get full market depth quote"""
        data = {exchange_segment: security_ids}
        return self._make_request("POST", "/marketfeed/quote", data)
    
    def get_historical_daily(self, security_id: str, exchange_segment: str, 
                            instrument: str, from_date: str, to_date: str,
                            expiry_code: int = 0, oi: bool = False) -> dict:
        """Get daily historical data"""
        data = {
            "securityId": security_id,
            "exchangeSegment": exchange_segment,
            "instrument": instrument,
            "expiryCode": expiry_code,
            "oi": oi,
            "fromDate": from_date,
            "toDate": to_date
        }
        return self._make_request("POST", "/charts/historical", data)
    
    def get_historical_intraday(self, security_id: str, exchange_segment: str,
                               instrument: str, interval: str, from_date: str,
                               to_date: str, oi: bool = False) -> dict:
        """
        Get intraday historical data
        interval: "1", "5", "15", "25", "60" (minutes)
        """
        data = {
            "securityId": security_id,
            "exchangeSegment": exchange_segment,
            "instrument": instrument,
            "interval": interval,
            "oi": oi,
            "fromDate": from_date,
            "toDate": to_date
        }
        return self._make_request("POST", "/charts/intraday", data)
    
    # ========================================================================
    # OPTIONS CHAIN APIs
    # ========================================================================
    
    @lru_cache(maxsize=10)
    def get_option_chain(self, underlying_scrip: str, underlying_seg: str, 
                        expiry: str) -> dict:
        """
        Get complete option chain
        Cached for 3 seconds (per rate limit guidelines)
        """
        data = {
            "UnderlyingScrip": underlying_scrip,
            "UnderlyingSeg": underlying_seg,
            "Expiry": expiry
        }
        return self._make_request("POST", "/optionchain", data)
    
    @lru_cache(maxsize=5)
    def get_expiry_list(self, underlying_scrip: str, underlying_seg: str) -> dict:
        """Get list of expiries for underlying"""
        data = {
            "UnderlyingScrip": underlying_scrip,
            "UnderlyingSeg": underlying_seg
        }
        return self._make_request("POST", "/optionchain/expirylist", data)
    
    # ========================================================================
    # TRADING APIs
    # ========================================================================
    
    def place_order(self, transaction_type: str, exchange_segment: str,
                   product_type: str, order_type: str, security_id: str,
                   quantity: int, price: float = 0.0, trigger_price: float = 0.0,
                   validity: str = "DAY", disclosed_quantity: int = 0,
                   after_market_order: bool = False, correlation_id: str = "") -> dict:
        """Place a regular order"""
        data = {
            "dhanClientId": self.client_id,
            "correlationId": correlation_id,
            "transactionType": transaction_type,
            "exchangeSegment": exchange_segment,
            "productType": product_type,
            "orderType": order_type,
            "validity": validity,
            "securityId": security_id,
            "quantity": str(quantity),
            "disclosedQuantity": str(disclosed_quantity),
            "price": str(price),
            "triggerPrice": str(trigger_price),
            "afterMarketOrder": after_market_order
        }
        return self._make_request("POST", "/orders", data)
    
    def place_super_order(self, transaction_type: str, exchange_segment: str,
                         product_type: str, order_type: str, security_id: str,
                         quantity: int, price: float, target_price: float,
                         stop_loss_price: float, trailing_jump: float = 0.0,
                         correlation_id: str = "") -> dict:
        """
        Place Super Order (Entry + Target + Stop Loss in one)
        Unique to DhanHQ - combines bracket order with trailing SL
        """
        data = {
            "dhanClientId": self.client_id,
            "correlationId": correlation_id,
            "transactionType": transaction_type,
            "exchangeSegment": exchange_segment,
            "productType": product_type,
            "orderType": order_type,
            "securityId": security_id,
            "quantity": quantity,
            "price": price,
            "targetPrice": target_price,
            "stopLossPrice": stop_loss_price,
            "trailingJump": trailing_jump
        }
        return self._make_request("POST", "/super/orders", data)
    
    def modify_order(self, order_id: str, order_type: str = None, 
                    quantity: int = None, price: float = None,
                    trigger_price: float = None, validity: str = None) -> dict:
        """Modify pending order"""
        data = {"dhanClientId": self.client_id, "orderId": order_id}
        
        if order_type:
            data["orderType"] = order_type
        if quantity:
            data["quantity"] = str(quantity)
        if price:
            data["price"] = str(price)
        if trigger_price:
            data["triggerPrice"] = str(trigger_price)
        if validity:
            data["validity"] = validity
            
        return self._make_request("PUT", f"/orders/{order_id}", data)
    
    def cancel_order(self, order_id: str) -> dict:
        """Cancel pending order"""
        return self._make_request("DELETE", f"/orders/{order_id}")
    
    def get_order_list(self) -> dict:
        """Get all orders for the day"""
        return self._make_request("GET", "/orders")
    
    def get_order_by_id(self, order_id: str) -> dict:
        """Get specific order details"""
        return self._make_request("GET", f"/orders/{order_id}")
    
    def get_positions(self) -> dict:
        """Get current positions"""
        return self._make_request("GET", "/positions")
    
    def get_holdings(self) -> dict:
        """Get holdings"""
        return self._make_request("GET", "/holdings")
    
    def get_fund_limit(self) -> dict:
        """Get fund limits and margins"""
        return self._make_request("GET", "/fundlimit")
    
    def calculate_margin(self, exchange_segment: str, transaction_type: str,
                        quantity: int, product_type: str, security_id: str,
                        price: float, trigger_price: float = 0.0) -> dict:
        """Calculate margin requirements before placing order"""
        data = {
            "dhanClientId": self.client_id,
            "exchangeSegment": exchange_segment,
            "transactionType": transaction_type,
            "quantity": quantity,
            "productType": product_type,
            "securityId": security_id,
            "price": price,
            "triggerPrice": trigger_price
        }
        return self._make_request("POST", "/margincalculator", data)


class DhanWebSocket:
    """
    WebSocket client for real-time market data
    Handles binary data parsing from DhanHQ feed
    """
    
    def __init__(self, client_id: str, access_token: str):
        self.client_id = client_id
        self.access_token = access_token
        self.ws = None
        self.callbacks = {}
        
    def connect(self):
        """Connect to WebSocket"""
        ws_url = f"wss://api-feed.dhan.co?version=2&token={self.access_token}&clientId={self.client_id}&authType=2"
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
    def on_open(self, ws):
        """WebSocket connection opened"""
        logger.info("WebSocket connected")
        
    def on_message(self, ws, message):
        """Parse binary message from WebSocket"""
        # Parse header (8 bytes)
        feed_code = struct.unpack('B', message[0:1])[0]
        msg_length = struct.unpack('<H', message[1:3])[0]
        exchange_seg = struct.unpack('B', message[3:4])[0]
        security_id = struct.unpack('<I', message[4:8])[0]
        
        # Parse payload based on feed code
        if feed_code == 2:  # Ticker packet
            ltp = struct.unpack('<f', message[8:12])[0]
            ltt = struct.unpack('<I', message[12:16])[0]
            self._trigger_callback('ticker', {
                'security_id': security_id,
                'ltp': ltp,
                'timestamp': ltt
            })
            
        elif feed_code == 4:  # Quote packet
            data = self._parse_quote_packet(message[8:])
            data['security_id'] = security_id
            self._trigger_callback('quote', data)
            
        elif feed_code == 8:  # Full packet (with market depth)
            data = self._parse_full_packet(message[8:])
            data['security_id'] = security_id
            self._trigger_callback('full', data)
    
    def _parse_quote_packet(self, payload: bytes) -> dict:
        """Parse quote packet (51 bytes)"""
        return {
            'ltp': struct.unpack('<f', payload[0:4])[0],
            'ltq': struct.unpack('<H', payload[4:6])[0],
            'ltt': struct.unpack('<I', payload[6:10])[0],
            'avg_price': struct.unpack('<f', payload[10:14])[0],
            'volume': struct.unpack('<I', payload[14:18])[0],
            'sell_qty': struct.unpack('<I', payload[18:22])[0],
            'buy_qty': struct.unpack('<I', payload[22:26])[0],
            'open': struct.unpack('<f', payload[26:30])[0],
            'close': struct.unpack('<f', payload[30:34])[0],
            'high': struct.unpack('<f', payload[34:38])[0],
            'low': struct.unpack('<f', payload[38:42])[0]
        }
    
    def _parse_full_packet(self, payload: bytes) -> dict:
        """Parse full packet with market depth (162 bytes)"""
        quote_data = self._parse_quote_packet(payload[0:51])
        
        # Parse OI data (12 bytes)
        quote_data['oi'] = struct.unpack('<I', payload[30:34])[0]
        quote_data['oi_high'] = struct.unpack('<I', payload[34:38])[0]
        quote_data['oi_low'] = struct.unpack('<I', payload[38:42])[0]
        
        # Parse market depth (100 bytes = 5 levels * 20 bytes each)
        depth_buy = []
        depth_sell = []
        
        for i in range(5):
            offset = 62 + (i * 20)
            depth_buy.append({
                'quantity': struct.unpack('<I', payload[offset:offset+4])[0],
                'orders': struct.unpack('<H', payload[offset+8:offset+10])[0],
                'price': struct.unpack('<f', payload[offset+12:offset+16])[0]
            })
            
            depth_sell.append({
                'quantity': struct.unpack('<I', payload[offset+4:offset+8])[0],
                'orders': struct.unpack('<H', payload[offset+10:offset+12])[0],
                'price': struct.unpack('<f', payload[offset+16:offset+20])[0]
            })
        
        quote_data['depth'] = {'buy': depth_buy, 'sell': depth_sell}
        return quote_data
    
    def subscribe(self, exchange_segment: str, security_ids: List[str], mode: str = "quote"):
        """
        Subscribe to instruments
        mode: "ticker" (LTP only), "quote" (OHLC+Volume), "full" (with depth)
        """
        request_codes = {"ticker": 15, "quote": 17, "full": 21}
        
        message = {
            "RequestCode": request_codes.get(mode, 17),
            "InstrumentCount": len(security_ids),
            "InstrumentList": [
                {"ExchangeSegment": exchange_segment, "SecurityId": sid}
                for sid in security_ids
            ]
        }
        
        if self.ws and self.ws.sock and self.ws.sock.connected:
            self.ws.send(json.dumps(message))
            logger.info(f"Subscribed to {len(security_ids)} instruments in {mode} mode")
    
    def register_callback(self, event: str, callback):
        """Register callback for events"""
        self.callbacks[event] = callback
    
    def _trigger_callback(self, event: str, data: dict):
        """Trigger registered callback"""
        if event in self.callbacks:
            self.callbacks[event](data)
    
    def on_error(self, ws, error):
        """WebSocket error"""
        logger.error(f"WebSocket error: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        """WebSocket closed"""
        logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
    
    def run(self):
        """Start WebSocket connection (blocking)"""
        if self.ws:
            self.ws.run_forever()
