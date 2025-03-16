import ccxt
import time
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import json

action_status = "No Action"

# =========================================
#  USER INPUTS
# =========================================
choice = input("Enter 1 for Spot, or 2 for Futures: ").strip()
if choice not in ["1", "2"]:
    print("❌ Invalid selection. Please re-run and choose 1 or 2.")
    exit()

allocated_usd = float(input("Enter how much USDT to invest/trade with: ").strip())
trade_type = "spot" if choice == "1" else "futures"
trade_type_label = "SPOT" if trade_type == "spot" else "FUTURES"

print(f"\nTrade type selected: {trade_type_label}")
print(f"Bot will trade up to ${allocated_usd:.2f} USDT on each BUY.\n")

# -----------------------------------------
#  Select Trading Account (Real or Demo)
# -----------------------------------------
account_type = input("Select trading account: 1 for Real, 2 for Demo: ").strip()

# Load API keys from .env file
load_dotenv()

if account_type == "1":
    print("✅ Trading in REAL account")
    use_demo = False
    API_KEY = os.getenv("Real_API_KEY")
    SECRET_KEY = os.getenv("Real_SECRET_KEY")
    PASSPHRASE = os.getenv("Real_PASSPHRASE")
elif account_type == "2":
    print("✅ Trading in DEMO account")
    use_demo = True
    API_KEY = os.getenv("Demo_API_KEY")
    SECRET_KEY = os.getenv("Demo_SECRET_KEY")
    PASSPHRASE = os.getenv("Demo_PASSPHRASE")
else:
    print("❌ Invalid selection for trading account. Exiting.")
    exit()

# -----------------------------------------
#  IMPORTANT: DEMO MODE CHECK FOR SPOT
# -----------------------------------------
if trade_type == "spot" and use_demo:
    print("❌ Demo mode is not supported for Spot trading on Bitget. Please select Real account for Spot trading.")
    exit()

# =========================================
#  CONFIG & CREDENTIALS: Create Bitget Client
# =========================================
bitget = ccxt.bitget({
    'apiKey': API_KEY,
    'secret': SECRET_KEY,
    'password': PASSPHRASE,  # Bitget expects your passphrase as "password"
    'enableRateLimit': True,
})

# (Optional) Enable verbose mode to see raw request/response details:
# bitget.verbose = True

# For demo mode (Futures only), add required headers and reload markets.
if use_demo and trade_type == "futures":
    bitget.headers = {
        "X-Bitget-Environment": "demo",
        "paptrading": "1",
        "locale": "en-US",
        "Content-Type": "application/json"
    }
    bitget.load_markets()  # Ensure these headers are used for all requests.
    print("🔄 Using Bitget DEMO environment.")
else:
    bitget.load_markets()
    print("🔄 Using Bitget REAL environment.")

# -----------------------------------------
#  SYMBOL & TIMEFRAME CONFIGURATION
# -----------------------------------------
if trade_type == "spot":
    CCXT_SYMBOL = "BTC/USDT"
    TIMEFRAME = '5m'
else:
    # For futures (demo or real) use the standard symbol.
    CCXT_SYMBOL = "BTC/USDT:USDT"
    TIMEFRAME = '5m'

SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 1
LOOP_INTERVAL = 10  # seconds
LEVERAGE = 10

# =========================================
#  FETCH OHLCV USING CCXT
# =========================================
def fetch_ohlcv_ccxt(symbol: str, timeframe: str = "5m", limit: int = 100) -> pd.DataFrame:
    try:
        data = bitget.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df["timestamp"] = df["timestamp"].astype(float)
        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        print("❌ Error fetching OHLCV:", e)
        return pd.DataFrame()

# =========================================
#  CALCULATE SUPER TREND
# =========================================
def calculate_supertrend(df: pd.DataFrame, period=10, multiplier=1.0) -> pd.DataFrame:
    if df.empty:
        return df
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = (df['high'] - df['close'].shift(1)).abs()
    df['tr3'] = (df['low'] - df['close'].shift(1)).abs()
    df['true_range'] = df[['tr1','tr2','tr3']].max(axis=1)
    df['atr'] = df['true_range'].ewm(alpha=1/period, adjust=False).mean()
    df['src'] = (df['high'] + df['low']) / 2
    df['up'] = df['src'] - multiplier * df['atr']
    df['dn'] = df['src'] + multiplier * df['atr']
    up_series = np.zeros(len(df))
    dn_series = np.zeros(len(df))
    trend_series = np.zeros(len(df))
    up_series[0] = df['up'].iloc[0]
    dn_series[0] = df['dn'].iloc[0]
    trend_series[0] = 1
    for i in range(1, len(df)):
        prev_close = df['close'].iloc[i-1]
        up_series[i] = max(df['up'].iloc[i], up_series[i-1]) if prev_close > up_series[i-1] else df['up'].iloc[i]
        dn_series[i] = min(df['dn'].iloc[i], dn_series[i-1]) if prev_close < dn_series[i-1] else df['dn'].iloc[i]
        if trend_series[i-1] == -1 and df['close'].iloc[i] > dn_series[i-1]:
            trend_series[i] = 1
        elif trend_series[i-1] == 1 and df['close'].iloc[i] < up_series[i-1]:
            trend_series[i] = -1
        else:
            trend_series[i] = trend_series[i-1]
    df['trend'] = trend_series
    df['supertrend'] = np.where(df['trend'] == 1, up_series, dn_series)
    df.drop(columns=["tr1", "tr2", "tr3", "true_range", "atr", "src", "up", "dn"], inplace=True, errors='ignore')
    return df

# =========================================
#  SPOT BALANCE (REAL ONLY)
# =========================================
def fetch_spot_balance(coin: str) -> float:
    if use_demo:
        return 0.0
    try:
        b = bitget.fetch_balance({'type': 'spot'})
        return float(b['free'].get(coin.upper(), 0.0))
    except Exception as e:
        print("❌ Spot balance error:", e)
        return 0.0

# =========================================
#  FUTURES BALANCE via GET /account/accounts
# =========================================
def fetch_futures_balance(coin: str = "USDT") -> float:
    """
    Calls Bitget's Get Account List endpoint:
      GET /api/mix/v1/account/accounts?productType=umcbl
    and returns the 'available' balance for the specified coin.
    """
    response = None
    try:
        # Use the endpoint key expected by CCXT (the base URL is preconfigured).
        endpoint = "account/accounts"  
        method = "GET"
        params = {"productType": "umcbl"}
        response = bitget.fetch2(endpoint, 'private', method, params)
        # If response is a string, parse it.
        if isinstance(response, str):
            response = json.loads(response)
        print("Raw response from /account/accounts =>", response)
        data = response.get("data", [])
        if not isinstance(data, list):
            print("Data field is not a list:", data)
            return 0.0
        for acc_info in data:
            print("Processing account info:", acc_info)
            margin_coin = acc_info.get("marginCoin", "").upper()
            if margin_coin == coin.upper():
                available_str = acc_info.get("available", "0")
                return float(available_str)
        return 0.0
    except KeyError as ke:
        print("KeyError encountered:", ke, "Response:", response)
        return 0.0
    except Exception as e:
        print(f"❌ Error fetching futures balance via /account/accounts: {e}")
        return 0.0

# =========================================
#  SPOT ORDER FUNCTIONS
# =========================================
def place_spot_buy(symbol, cost_usd):
    try:
        params = {'quoteOrderQty': cost_usd}
        return bitget.create_market_buy_order(symbol, None, params=params)
    except Exception as e:
        print("❌ Error placing spot buy:", e)
        return None

def place_spot_sell(symbol, amount):
    try:
        return bitget.create_market_sell_order(symbol, amount)
    except Exception as e:
        print("❌ Error placing spot sell:", e)
        return None

def get_spot_position_btc() -> float:
    if use_demo:
        return 0.0
    try:
        b = bitget.fetch_balance({'type': 'spot'})
        return float(b['free'].get('BTC', 0.0))
    except Exception as e:
        print("❌ Spot position error:", e)
        return 0.0

# =========================================
#  FUTURES ORDER FUNCTIONS
# =========================================
def open_long(symbol, size: float):
    try:
        params = {
            'productType': 'umcbl',
            'marginMode': 'cross',
            'marginCoin': 'USDT',
            'tradeSide': 'open',
            'orderType': 'market',
            'force': 'gtc'
        }
        order = bitget.create_order(symbol, 'market', 'buy', size, None, params)
        if order and 'id' in order:
            print(f"✅ LONG Opened: Order ID {order['id']}")
        else:
            print(f"⚠️ Order may not have been placed correctly: {order}")
        return order
    except Exception as e:
        print("❌ Error opening long:", e)
        return None

def close_long(symbol, size: float):
    try:
        params = {
            'productType': 'USDT-FUTURES',
            'marginMode': 'cross',
            'marginCoin': 'USDT',
            'tradeSide': 'close',
            'orderType': 'market',
            'force': 'gtc'
        }
        return bitget.create_order(symbol, 'market', 'sell', size, None, params)
    except Exception as e:
        print("❌ Error closing long:", e)
        return None

def open_short(symbol, size: float):
    try:
        params = {
            'productType': 'USDT-FUTURES',
            'marginMode': 'cross',
            'marginCoin': 'USDT',
            'tradeSide': 'open',
            'orderType': 'market',
            'force': 'gtc'
        }
        return bitget.create_order(symbol, 'market', 'sell', size, None, params)
    except Exception as e:
        print("❌ Error opening short:", e)
        return None

def close_short(symbol, size: float):
    try:
        params = {
            'productType': 'USDT-FUTURES',
            'marginMode': 'cross',
            'marginCoin': 'USDT',
            'tradeSide': 'close',
            'orderType': 'market',
            'force': 'gtc'
        }
        return bitget.create_order(symbol, 'market', 'buy', size, None, params)
    except Exception as e:
        print("❌ Error closing short:", e)
        return None

# =========================================
#  FUTURES POSITION HELPER
# =========================================
def get_futures_position():
    try:
        positions = bitget.fetch_positions()
        pos_data = {"long": 0.0, "short": 0.0}
        for p in positions:
            if p.get("symbol") == CCXT_SYMBOL:
                size = float(p.get("contracts", 0))
                side = p.get("side")
                if side == "long":
                    pos_data["long"] += size
                elif side == "short":
                    pos_data["short"] += size
        return pos_data
    except Exception as e:
        print("❌ Error fetching futures positions:", e)
        return {"long": 0.0, "short": 0.0}

def parse_fill_price(order_resp) -> str:
    if not order_resp:
        return "N/A"
    data = order_resp.get("info", {}).get("data", [])
    if data and isinstance(data, list) and len(data) > 0:
        return str(data[0].get("fillPrice", "N/A"))
    return "N/A"

# =========================================
#  STRATEGY: Spot
# =========================================
def manage_spot_position(signal: str):
    btc_bal = get_spot_position_btc()
    have_btc = btc_bal > 0
    if have_btc and signal == "SELL":
        close_res = place_spot_sell(CCXT_SYMBOL, btc_bal)
        if close_res:
            fill_price = parse_fill_price(close_res)
            print(f"Executed SELL (Spot) @ {fill_price}")
        else:
            print("Failed to SELL (Spot).")
    elif (not have_btc) and signal == "BUY":
        buy_res = place_spot_buy(CCXT_SYMBOL, allocated_usd)
        if buy_res:
            fill_price = parse_fill_price(buy_res)
            print(f"Executed BUY (Spot) @ {fill_price}")
        else:
            print("Failed to BUY (Spot).")

# =========================================
#  STRATEGY: Futures (Hedge Mode Logic)
# =========================================
def manage_futures_position(signal: str):
    pos = get_futures_position()
    long_sz = pos.get("long", 0.0)
    short_sz = pos.get("short", 0.0)
    print("Fetched Futures Position:", pos)
    df = fetch_ohlcv_ccxt(CCXT_SYMBOL, TIMEFRAME, limit=1)
    if df.empty:
        print("No data => no futures action possible.")
        return
    price = df["close"].iloc[-1]
    futures_usdt = fetch_futures_balance()
    allocated_for_trade = min(allocated_usd, futures_usdt)
    if allocated_for_trade <= 0:
        print("❌ Not enough USDT in futures account to open new position.")
        return
    new_amt = (allocated_for_trade * LEVERAGE) / price
    action = "No Action"
    if signal == "BUY" and long_sz > 0:
        action = "No Action"
    elif signal == "SELL" and long_sz > 0:
        close_res = close_long(CCXT_SYMBOL, long_sz)
        if close_res and close_res.get("id"):
            action = "LONG Closed"
            print(f"✅ Closed LONG @ {parse_fill_price(close_res)}")
            open_res = open_short(CCXT_SYMBOL, new_amt)
            if open_res and open_res.get("id"):
                action = "Reentry SHORT Successful"
                print(f"✅ Reentered SHORT @ {parse_fill_price(open_res)}")
            else:
                print(f"Failed to open SHORT. Response: {open_res}")
    elif signal == "SELL" and short_sz > 0:
        action = "No Action"
    elif signal == "BUY" and short_sz > 0:
        close_res = close_short(CCXT_SYMBOL, short_sz)
        if close_res and close_res.get("id"):
            action = "SHORT Closed"
            print(f"✅ Closed SHORT @ {parse_fill_price(close_res)}")
            open_res = open_long(CCXT_SYMBOL, new_amt)
            if open_res and open_res.get("id"):
                action = "Reentry LONG Successful"
                print(f"✅ Reentered LONG @ {parse_fill_price(open_res)}")
            else:
                print(f"Failed to open LONG. Response: {open_res}")
    elif signal == "BUY" and long_sz == 0 and short_sz == 0:
        open_res = open_long(CCXT_SYMBOL, new_amt)
        if open_res and open_res.get("id"):
            action = "Reentry LONG Successful"
            print(f"✅ Opened LONG @ {parse_fill_price(open_res)}")
        else:
            print(f"Failed to open LONG. Response: {open_res}")
    elif signal == "SELL" and long_sz == 0 and short_sz == 0:
        open_res = open_short(CCXT_SYMBOL, new_amt)
        if open_res and open_res.get("id"):
            action = "Reentry SHORT Successful"
            print(f"✅ Opened SHORT @ {parse_fill_price(open_res)}")
        else:
            print(f"Failed to open SHORT. Response: {open_res}")
    global action_status
    action_status = action

# =========================================
#  PRINT STATUS TABLE
# =========================================
def print_status_table(trade_type_label, spot_usdt, futures_usdt, coin, ltp, last_close, supertrend, sentiment, signal):
    if trade_type_label == "FUTURES":
        pos_data = get_futures_position()
        open_positions_count = sum(1 for v in pos_data.values() if v > 0)
    else:
        pos_data = {"long": 0.0, "short": 0.0}
        open_positions_count = 0
    open_pos_msg = f"Yes ({open_positions_count})" if open_positions_count > 0 else "No (0)"
    print("=" * 100)
    print(f"| Trade Type : {trade_type_label:<6} | SPOT Bal: {spot_usdt:.2f}  FUT Bal: {futures_usdt:.2f} | Open Position: {open_pos_msg}")
    print("=" * 100)
    print("| Coin   |   LTP       | Last Close    | Supertrend    | Sentiment   | Signal   | Action                      |")
    print("=" * 100)
    print(f"| {coin:<6} | ${ltp:<10.2f}| ${last_close:<13.2f}| ${supertrend:<13.2f}| {sentiment:<11} | {signal:<7} | {action_status:<25} |")
    print("=" * 100)

# =========================================
#  MAIN LOOP
# =========================================
def main():
    print("\nStarting CCXT-based Bitget Supertrend Bot...\n")
    while True:
        if trade_type == "spot":
            spot_usdt = fetch_spot_balance("USDT")
            futures_usdt = 0.0
        elif trade_type == "futures":
            if use_demo:
                spot_usdt = 0.0
                futures_usdt = fetch_futures_balance()
            else:
                spot_usdt = fetch_spot_balance("USDT")
                futures_usdt = fetch_futures_balance()

        df = fetch_ohlcv_ccxt(CCXT_SYMBOL, TIMEFRAME, limit=100)
        if df.empty or len(df) < SUPERTREND_PERIOD:
            print("❌ Not enough data or fetch error. Sleeping...\n")
            time.sleep(LOOP_INTERVAL)
            continue

        df = calculate_supertrend(df, SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER)
        if "supertrend" not in df.columns:
            print("❌ Supertrend calc fail. Sleeping...\n")
            time.sleep(LOOP_INTERVAL)
            continue

        last_close = df["close"].iloc[-1]
        st_val = df["supertrend"].iloc[-1]
        tr_val = df["trend"].iloc[-1]

        if tr_val == 1:
            sentiment = "BULLISH"
            signal = "BUY"
        else:
            sentiment = "BEARISH"
            signal = "SELL"

        if trade_type == "spot":
            manage_spot_position(signal)
        else:
            manage_futures_position(signal)

        print_status_table(
            trade_type_label,
            spot_usdt,
            futures_usdt,
            "BTC",
            last_close,
            last_close,
            st_val,
            sentiment,
            signal
        )

        time.sleep(LOOP_INTERVAL)

if __name__ == "__main__":
    main()
