import sys
import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from urllib.parse import quote

import pandas as pd
import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# ============================================================
# CONFIG
# ============================================================

SCRIPT_FOLDER = Path(__file__).resolve().parent

INPUT_FILE = SCRIPT_FOLDER / "My-Stocks.csv"
SETUP_HISTORY_FILE = SCRIPT_FOLDER / "Setup_History.csv"

# India Standard Time (IST)
IST = ZoneInfo("Asia/Kolkata")

REPORT_TIMESTAMP = datetime.now(IST).strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_FILE = SCRIPT_FOLDER / f"RSI_Scanner_{REPORT_TIMESTAMP}.csv"

# Files consumed by GitHub Pages
LATEST_CSV_FILE = SCRIPT_FOLDER / "latest_results.csv"
LATEST_JSON_FILE = SCRIPT_FOLDER / "latest_results.json"

# Persistent history
HISTORY_FILE = SCRIPT_FOLDER / "RSI_History.csv"
MONITORING_STATE_FILE = SCRIPT_FOLDER / "RSI_Monitoring_State.csv"
SCAN_CYCLE_FILE = SCRIPT_FOLDER / "RSI_Scan_Cycle.json"
MONITORING_ENTRY_RSI = 50.0
MONITORING_EXIT_RSI = 50.0

RSI_PERIOD = 14
WEEKLY_LOOKBACK_DAYS = 730
HOURLY_LOOKBACK_DAYS = 30

UPSTOX_API = "https://api.upstox.com"

ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN", "").strip()

if not ACCESS_TOKEN:
    raise SystemExit("ERROR: UPSTOX_ACCESS_TOKEN is not configured.")


HEADERS = {
    "Accept": "application/json",
    "Authorization": f"Bearer {ACCESS_TOKEN}",
}


# ============================================================
# RSI
# ============================================================

def calculate_rsi(close, period=14):
    close = pd.to_numeric(close, errors="coerce").reset_index(drop=True)

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    rsi = pd.Series(index=close.index, dtype=float)

    if len(close) <= period:
        return rsi

    avg_gain = gain.iloc[1:period + 1].mean()
    avg_loss = loss.iloc[1:period + 1].mean()

    if avg_loss == 0:
        rsi.iloc[period] = 100.0 if avg_gain > 0 else 50.0
    else:
        rsi.iloc[period] = 100 - (
            100 / (1 + avg_gain / avg_loss)
        )

    for i in range(period + 1, len(close)):
        avg_gain = (
            (avg_gain * (period - 1)) + gain.iloc[i]
        ) / period

        avg_loss = (
            (avg_loss * (period - 1)) + loss.iloc[i]
        ) / period

        if avg_loss == 0:
            rsi.iloc[i] = 100.0 if avg_gain > 0 else 50.0
        else:
            rsi.iloc[i] = 100 - (
                100 / (1 + avg_gain / avg_loss)
            )

    return rsi


# ============================================================
# STOCK LIST
# ============================================================

def load_stocks():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"My-Stocks.csv not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    if "Symbol" not in df.columns:
        raise ValueError(
            "My-Stocks.csv must contain a column named 'Symbol'."
        )

    df["Symbol"] = (
        df["Symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace("\\", "", regex=False)
    )

    # Keep Tag from My-Stocks.csv.
    # Tag is optional, but if the column is present it is carried
    # forward into the scanner result and Setup_History.csv.
    if "Tag" not in df.columns:
        df["Tag"] = ""

    df["Tag"] = (
        df["Tag"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return (
        df[df["Symbol"] != ""]
        .drop_duplicates("Symbol")
        .copy()
    )


# ============================================================
# UPSTOX
# ============================================================

def find_instrument_key(symbol):
    trading_symbol = (
        symbol.replace("NSE:", "")
        .strip()
        .upper()
    )

    response = requests.get(
        f"{UPSTOX_API}/v2/instruments/search",
        headers=HEADERS,
        params={
            "query": trading_symbol,
            "exchanges": "NSE",
            "segments": "EQ",
            "page_number": 1,
            "records": 30,
        },
        timeout=20,
    )

    if response.status_code != 200:
        print(
            f"  Instrument search failed: "
            f"{response.status_code}"
        )
        return None

    instruments = response.json().get("data", [])

    for item in instruments:
        if (
            str(item.get("trading_symbol", "")).upper()
            == trading_symbol
            and item.get("segment") == "NSE_EQ"
        ):
            return item.get("instrument_key")

    for item in instruments:
        if item.get("segment") == "NSE_EQ":
            return item.get("instrument_key")

    return None


def get_candles(
    instrument_key,
    unit,
    interval,
    lookback_days
):
    """
    Upstox Historical Candle V3.
    """

    to_date = datetime.now().strftime("%Y-%m-%d")

    from_date = (
        datetime.now()
        - timedelta(days=lookback_days)
    ).strftime("%Y-%m-%d")

    encoded_key = quote(instrument_key, safe="")

    response = requests.get(
        f"{UPSTOX_API}/v3/historical-candle/"
        f"{encoded_key}/{unit}/{interval}/"
        f"{to_date}/{from_date}",
        headers=HEADERS,
        timeout=20,
    )

    if response.status_code != 200:
        print(
            f"  {unit}/{interval} candle request failed: "
            f"{response.status_code} "
            f"{response.text[:300]}"
        )
        return []

    return (
        response.json()
        .get("data", {})
        .get("candles", [])
    )


def get_current_ltp(instrument_key):
    encoded_key = quote(instrument_key, safe="")

    response = requests.get(
        f"{UPSTOX_API}/v3/market-quote/ltp"
        f"?instrument_key={encoded_key}",
        headers=HEADERS,
        timeout=20,
    )

    if response.status_code != 200:
        print(
            f"  LTP request failed: "
            f"{response.status_code}"
        )
        return None

    quote_data = response.json().get("data", {})

    if not quote_data:
        return None

    first_quote = next(iter(quote_data.values()))

    last_price = first_quote.get("last_price")

    return (
        float(last_price)
        if last_price is not None
        else None
    )


# ============================================================
# CANDLE DATAFRAME
# ============================================================

def candle_dataframe(candles):
    rows = [
        {
            "timestamp": c[0],
            "high": c[2],
            "close": c[4],
        }
        for c in candles
        if len(c) >= 5
    ]

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )

    df["high"] = pd.to_numeric(
        df["high"],
        errors="coerce",
    )

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce",
    )

    return (
        df.dropna(
            subset=["timestamp", "high", "close"]
        )
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )


# ============================================================
# WEEKLY RSI
# ============================================================

def get_current_weekly_rsi(
    instrument_key,
    ltp
):
    df = candle_dataframe(
        get_candles(
            instrument_key,
            "weeks",
            "1",
            WEEKLY_LOOKBACK_DAYS,
        )
    )

    if df.empty:
        return None

    now = pd.Timestamp.now(
        tz="Asia/Kolkata"
    )

    week_start = (
        now
        - pd.Timedelta(days=now.weekday())
    ).normalize()

    local_time = (
        df["timestamp"]
        .dt.tz_convert("Asia/Kolkata")
    )

    completed = df[
        local_time < week_start
    ].copy()

    if len(completed) < RSI_PERIOD + 1:
        return None

    current = pd.DataFrame([
        {
            "timestamp": week_start,
            "close": ltp,
        }
    ])

    calc = pd.concat(
        [
            completed[["timestamp", "close"]],
            current,
        ],
        ignore_index=True,
    )

    calc["RSI"] = calculate_rsi(
        calc["close"],
        RSI_PERIOD,
    )

    valid = calc.dropna(
        subset=["RSI"]
    )

    if len(valid) < 2:
        return None

    return {
        "current": float(
            valid.iloc[-1]["RSI"]
        ),
        "previous": float(
            valid.iloc[-2]["RSI"]
        ),
    }


# ============================================================
# HOURLY RSI
# ============================================================

def get_current_hourly_rsi(
    instrument_key,
    ltp
):
    df = candle_dataframe(
        get_candles(
            instrument_key,
            "hours",
            "1",
            HOURLY_LOOKBACK_DAYS,
        )
    )

    if df.empty:
        return None

    now = pd.Timestamp.now(
        tz="Asia/Kolkata"
    )

    current_hour = now.floor("h")

    local_time = (
        df["timestamp"]
        .dt.tz_convert("Asia/Kolkata")
    )

    completed = df[
        local_time < current_hour
    ].copy()

    # PHASE 4: price confirmation.
    # Current completed hourly candle CLOSE >
    # previous completed hourly candle HIGH.
    price_confirmation = False
    current_completed_close = None
    previous_completed_high = None

    if len(completed) >= 2:
        previous_completed = completed.iloc[-2]
        current_completed = completed.iloc[-1]

        current_completed_close = float(
            current_completed["close"]
        )
        previous_completed_high = float(
            previous_completed["high"]
        )

        price_confirmation = (
            current_completed_close >
            previous_completed_high
        )

    if len(completed) < RSI_PERIOD + 1:
        return None

    current = pd.DataFrame([
        {
            "timestamp": current_hour,
            "close": ltp,
        }
    ])

    calc = pd.concat(
        [
            completed[["timestamp", "close"]],
            current,
        ],
        ignore_index=True,
    )

    calc["RSI"] = calculate_rsi(
        calc["close"],
        RSI_PERIOD,
    )

    valid = calc.dropna(
        subset=["RSI"]
    )

    if len(valid) < 2:
        return None

    return {
        "current": float(
            valid.iloc[-1]["RSI"]
        ),
        "previous": float(
            valid.iloc[-2]["RSI"]
        ),
        "hour": current_hour.strftime(
            "%Y-%m-%d %H:%M"
        ),
        "price_confirmation": price_confirmation,
        "current_completed_close": (
            round(current_completed_close, 2)
            if current_completed_close is not None
            else None
        ),
        "previous_completed_high": (
            round(previous_completed_high, 2)
            if previous_completed_high is not None
            else None
        ),
    }


# ============================================================
# 15-MINUTE PRICE CONFIRMATION
# ============================================================

def get_intraday_candles(
    instrument_key,
    unit,
    interval
):
    """
    Upstox Historical Candle V3 - current trading day.

    The intraday V3 endpoint is specifically intended to retrieve
    OHLC candles for the current trading day.
    """

    encoded_key = quote(
        instrument_key,
        safe=""
    )

    response = requests.get(
        f"{UPSTOX_API}/v3/historical-candle/intraday/"
        f"{encoded_key}/{unit}/{interval}",
        headers=HEADERS,
        timeout=20,
    )

    if response.status_code != 200:
        print(
            f"  Intraday {unit}/{interval} candle request failed: "
            f"{response.status_code} "
            f"{response.text[:300]}"
        )
        return []

    return (
        response.json()
        .get("data", {})
        .get("candles", [])
    )


def get_current_15m_price_confirmation(instrument_key):
    """
    Price confirmation on the latest COMPLETED 15-minute candle.

    Rule:
        Latest completed 15m CLOSE >
        Previous completed 15m HIGH

    IMPORTANT:
    Upstox can return only a limited number of candles for the
    requested interval. We therefore fetch enough recent candles
    and select the latest completed candle using IST time.
    """

    # IMPORTANT:
    # Use the CURRENT-DAY intraday endpoint, not the historical endpoint.
    # This prevents yesterday's last 15m candle from being selected.
    df = candle_dataframe(
        get_intraday_candles(
            instrument_key,
            "minutes",
            "15",
        )
    )

    if df.empty:
        return {
            "confirmed": False,
            "current_close": None,
            "previous_high": None,
            "candle": None,
        }

    now = pd.Timestamp.now(tz="Asia/Kolkata")

    # Start of the currently forming 15-minute candle.
    current_15m = now.floor("15min")

    local_time = (
        df["timestamp"]
        .dt.tz_convert("Asia/Kolkata")
    )

    # Keep ONLY candles whose 15-minute period has fully completed.
    completed = df[
        local_time < current_15m
    ].copy()

    completed = completed.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    if len(completed) < 2:
        return {
            "confirmed": False,
            "current_close": None,
            "previous_high": None,
            "candle": None,
        }

    # Latest completed candle = immediately before
    # the currently forming 15-minute candle.
    current_completed = completed.iloc[-1]

    # Candle immediately before that.
    previous_completed = completed.iloc[-2]

    current_close = float(
        current_completed["close"]
    )

    previous_high = float(
        previous_completed["high"]
    )

    candle_time = (
        pd.Timestamp(
            current_completed["timestamp"]
        )
        .tz_convert("Asia/Kolkata")
        .strftime("%Y-%m-%d %H:%M")
    )

    return {
        "confirmed": (
            current_close > previous_high
        ),
        "current_close": round(
            current_close,
            2
        ),
        "previous_high": round(
            previous_high,
            2
        ),
        "candle": candle_time,
    }


# ============================================================
# ============================================================
# HOURLY RSI TOUCH-30 STATE
# ============================================================

def hourly_rsi_touched_30(symbol, current_hourly_rsi):
    """
    True when this stock has touched hourly RSI <= 30.

    The current RSI is checked first. Then RSI_History.csv is checked
    for previous hourly RSI observations for the same symbol.
    """

    try:
        if float(current_hourly_rsi) <= 30:
            return True
    except (TypeError, ValueError):
        pass

    if not HISTORY_FILE.exists():
        return False

    try:
        history = pd.read_csv(
            HISTORY_FILE,
            encoding="utf-8-sig",
        )

        if history.empty:
            return False

        if "Symbol" not in history.columns:
            return False

        if "Current Hourly RSI" not in history.columns:
            return False

        stock_history = history[
            history["Symbol"].astype(str).str.strip()
            == str(symbol).strip()
        ]

        if stock_history.empty:
            return False

        hourly_values = pd.to_numeric(
            stock_history["Current Hourly RSI"],
            errors="coerce",
        ).dropna()

        return bool(
            (hourly_values <= 30).any()
        )

    except Exception as error:
        print(
            f"  ⚠️ Could not check hourly RSI touch history: "
            f"{error}"
        )
        return False


# ============================================================
# 15-MINUTE MONITORING STATE
# ============================================================
MONITORING_COLUMNS = ["Symbol","Tag","Monitoring Status","Monitoring Started","Last Hourly RSI","Last Hourly Check","Last 15m Check"]

def load_monitoring_state():
    if not MONITORING_STATE_FILE.exists():
        return pd.DataFrame(columns=MONITORING_COLUMNS)
    try:
        s=pd.read_csv(MONITORING_STATE_FILE, encoding="utf-8-sig")
        for c in MONITORING_COLUMNS:
            if c not in s.columns: s[c]=""
        s=s.reindex(columns=MONITORING_COLUMNS)
        s["Symbol"]=s["Symbol"].fillna("").astype(str).str.strip().str.upper()
        return s[s["Symbol"]!=""].drop_duplicates("Symbol",keep="last").reset_index(drop=True)
    except Exception as e:
        print(f"  WARNING: Could not read monitoring state: {e}")
        return pd.DataFrame(columns=MONITORING_COLUMNS)

def save_monitoring_state(state):
    if state is None: state=pd.DataFrame(columns=MONITORING_COLUMNS)
    for c in MONITORING_COLUMNS:
        if c not in state.columns: state[c]=""
    state.reindex(columns=MONITORING_COLUMNS).to_csv(MONITORING_STATE_FILE,index=False,encoding="utf-8-sig")

def monitoring_symbols(state):
    if state is None or state.empty: return set()
    return set(state["Symbol"].astype(str).str.strip().str.upper())

def get_last_hourly_cycle():
    """Return the last completed full-universe hourly scan cycle."""
    if not SCAN_CYCLE_FILE.exists():
        return ""

    try:
        with open(SCAN_CYCLE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        return str(data.get("last_hourly_cycle", "")).strip()

    except Exception as error:
        print(f"  WARNING: Could not read scan cycle file: {error}")
        return ""


def save_hourly_cycle(now=None):
    """Record that the full stock universe has been scanned for this hour."""
    if now is None:
        now = datetime.now(IST)

    cycle = now.strftime("%Y-%m-%d %H")

    data = {
        "last_hourly_cycle": cycle,
        "updated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(SCAN_CYCLE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"  HOURLY CYCLE SAVED: {cycle}")


def hourly_cycle_due(now):
    """
    Return True only when this clock-hour has not had a full-universe scan.

    IMPORTANT:
    This is deliberately independent of RSI_Monitoring_State.csv.
    Last Hourly Check in the monitoring state is stock-level information;
    it must not be used to decide whether the whole universe is due.
    """
    current_cycle = now.strftime("%Y-%m-%d %H")
    last_cycle = get_last_hourly_cycle()

    return current_cycle != last_cycle

def update_monitoring_state(state, result, now):
    if result is None: return state
    symbol=str(result.get("Symbol","")).strip().upper()
    try: rsi=float(result.get("Current Hourly RSI"))
    except (TypeError,ValueError): return state
    if state is None: state=pd.DataFrame(columns=MONITORING_COLUMNS)
    state=state.copy()
    for c in MONITORING_COLUMNS:
        if c not in state.columns: state[c]=""
    state=state.reindex(columns=MONITORING_COLUMNS)
    state["Symbol"]=state["Symbol"].fillna("").astype(str).str.strip().str.upper()
    idx=state.index[state["Symbol"]==symbol].tolist()
    stamp=now.strftime("%Y-%m-%d %H:%M:%S")
    if rsi < MONITORING_ENTRY_RSI:
        if idx:
            i=idx[-1]; state.at[i,"Tag"]=result.get("Tag",""); state.at[i,"Monitoring Status"]="MONITORING"; state.at[i,"Last Hourly RSI"]=round(rsi,2); state.at[i,"Last Hourly Check"]=stamp
        else:
            state=pd.concat([state,pd.DataFrame([{
                "Symbol":symbol,"Tag":result.get("Tag",""),"Monitoring Status":"MONITORING",
                "Monitoring Started":stamp,"Last Hourly RSI":round(rsi,2),"Last Hourly Check":stamp,"Last 15m Check":""
            }])],ignore_index=True)
            print(f"  ENTERED 15M MONITORING: {symbol} | Hourly RSI {rsi:.2f}")
    elif idx:
        state=state.drop(index=idx).reset_index(drop=True)
        print(f"  EXITED 15M MONITORING: {symbol} | Hourly RSI {rsi:.2f}")
    return state

def mark_15m_check(state, symbols, now):
    if state is None or state.empty: return state
    state=state.copy(); syms={str(x).strip().upper() for x in symbols}
    mask=state["Symbol"].astype(str).str.strip().str.upper().isin(syms)
    state.loc[mask,"Last 15m Check"]=now.strftime("%Y-%m-%d %H:%M:%S")
    return state

# SIGNAL LOGIC
# ============================================================

def classify(
    weekly,
    hourly,
    fifteen_minute,
    hourly_touch_30,
):
    """
    PHASE 4 FINAL LOGIC

    1. Weekly RSI > 50
    2. Hourly RSI must TOUCH 30 or below
    3. After that touch, check 15m:
           Latest completed 15m CLOSE >
           Previous completed 15m HIGH
    4. If all conditions are satisfied = SETUP

    IMPORTANT:
    The current hourly RSI does NOT have to remain below 30.
    Once it has touched <= 30, the touch state is remembered
    through RSI_History.csv.
    """

    weekly_rsi = float(weekly["current"])
    hourly_rsi = float(hourly["current"])

    fifteen_price_confirmed = bool(
        fifteen_minute.get("confirmed", False)
    )

    if weekly_rsi <= 50:
        return (
            "IGNORE",
            "❌ IGNORE",
            f"Weekly RSI {weekly_rsi:.2f} <= 50",
        )

    if not hourly_touch_30:
        return (
            "WATCH",
            "👀 WATCH",
            f"Weekly RSI {weekly_rsi:.2f} > 50, "
            f"but Hourly RSI has not touched 30 "
            f"(current {hourly_rsi:.2f})",
        )

    if not fifteen_price_confirmed:
        current_close = fifteen_minute.get("current_close")
        previous_high = fifteen_minute.get("previous_high")

        return (
            "WATCH",
            "👀 WATCH",
            f"Weekly RSI {weekly_rsi:.2f} > 50 + "
            f"Hourly RSI touched 30 + "
            f"15m price confirmation failed "
            f"(Close {current_close} <= "
            f"Previous High {previous_high})",
        )

    return (
        "SETUP",
        "🔥 SETUP",
        f"Weekly RSI {weekly_rsi:.2f} > 50 + "
        f"Hourly RSI touched 30 + "
        f"15m Close > Previous 15m High",
    )



# ============================================================
# PROCESS STOCK
# ============================================================

def process_stock(symbol, tag="", check_15m=True):
    print(f"\nChecking {symbol} ...")

    instrument_key = find_instrument_key(symbol)

    if not instrument_key:
        print("  ❌ Instrument not found")
        return None

    print(
        f"  Instrument: {instrument_key}"
    )

    ltp = get_current_ltp(
        instrument_key
    )

    if ltp is None:
        print("  ❌ LTP unavailable")
        return None

    weekly = get_current_weekly_rsi(
        instrument_key,
        ltp,
    )

    hourly = get_current_hourly_rsi(
        instrument_key,
        ltp,
    )

    if weekly is None or hourly is None:
        print(
            "  ❌ Could not calculate "
            "Weekly/Hourly RSI"
        )
        return None

    if check_15m:
        fifteen_minute = get_current_15m_price_confirmation(instrument_key)
    else:
        fifteen_minute = {"confirmed": False, "current_close": None, "previous_high": None, "candle": None}

    hourly_touch_30 = hourly_rsi_touched_30(
        symbol,
        hourly["current"],
    )

    if check_15m:
        category, signal, reason = classify(weekly, hourly, fifteen_minute, hourly_touch_30)
    else:
        wr=float(weekly["current"]); hr=float(hourly["current"])
        if wr <= 50:
            category,signal,reason="IGNORE","❌ IGNORE",f"Weekly RSI {wr:.2f} <= 50"
        elif hourly_touch_30:
            category,signal,reason="WATCH","👀 WATCH",f"Hourly RSI {hr:.2f}; touch <=30 recorded; waiting for 15m monitoring confirmation"
        else:
            category,signal,reason="WATCH","👀 WATCH",f"Weekly RSI {wr:.2f} > 50; Hourly RSI {hr:.2f}; 15m confirmation checked only while monitored"

    hourly_change = (
        hourly["current"]
        - hourly["previous"]
    )

    scan_time = datetime.now(IST).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    print(
        f"  LTP: ₹{ltp:.2f}"
    )

    print(
        f"  Weekly RSI: "
        f"{weekly['current']:.2f}"
    )

    print(
        f"  Hourly RSI: "
        f"{hourly['current']:.2f}"
    )

    print(
        f"  Hourly Change: "
        f"{hourly_change:+.2f}"
    )

    print(
        f"  Price Confirmation: "
        f"{'YES' if hourly.get('price_confirmation', False) else 'NO'}"
    )

    if (
        hourly.get("current_completed_close") is not None
        and hourly.get("previous_completed_high") is not None
    ):
        print(
            f"  Completed Hourly Close: "
            f"₹{hourly['current_completed_close']:.2f}"
        )
        print(
            f"  Previous Hourly High: "
            f"₹{hourly['previous_completed_high']:.2f}"
        )

    print(
        f"  Hourly RSI Touched 30: "
        f"{'YES' if hourly_touch_30 else 'NO'}"
    )

    print(
        f"  Current IST Time: "
        f"{pd.Timestamp.now(tz='Asia/Kolkata').strftime('%Y-%m-%d %H:%M:%S')}"
    )

    print(
        f"  15m Confirmation Candle: "
        f"{fifteen_minute.get('candle', 'N/A')}"
    )

    print(
        f"  15m Price Confirmation: "
        f"{'YES' if fifteen_minute.get('confirmed', False) else 'NO'}"
    )

    if (
        fifteen_minute.get("current_close") is not None
        and fifteen_minute.get("previous_high") is not None
    ):
        print(
            f"  Completed 15m Close: "
            f"₹{fifteen_minute['current_close']:.2f}"
        )

        print(
            f"  Previous 15m High: "
            f"₹{fifteen_minute['previous_high']:.2f}"
        )

    print(
        "  15m RSI logic: REMOVED"
    )

    print(
        f"  Signal: {signal}"
    )

    print(
        f"  Reason: {reason}"
    )

    return {
        "Scan Time": scan_time,
        "Symbol": symbol,
        "Tag": tag,
        "Instrument Key": instrument_key,
        "Current LTP": round(ltp, 2),

        "Current Week RSI": round(
            weekly["current"],
            2,
        ),

        "Previous Week RSI": round(
            weekly["previous"],
            2,
        ),

        "Weekly RSI Change": round(
            weekly["current"]
            - weekly["previous"],
            2,
        ),

        "Current Hour": hourly["hour"],

        "Current Hourly RSI": round(
            hourly["current"],
            2,
        ),

        "Previous Hourly RSI": round(
            hourly["previous"],
            2,
        ),

        "Hourly RSI Change": round(
            hourly_change,
            2,
        ),

        "Hourly RSI Rising": (
            "YES"
            if hourly_change > 0
            else "NO"
        ),

        "Price Confirmation": (
            "YES"
            if hourly.get("price_confirmation", False)
            else "NO"
        ),

        "Completed Hourly Close": (
            hourly.get("current_completed_close", "")
        ),

        "Previous Hourly High": (
            hourly.get("previous_completed_high", "")
        ),

        "Hourly RSI Touched 30": (
            "YES"
            if hourly_touch_30
            else "NO"
        ),

        "15m Price Confirmation": (
            "YES"
            if fifteen_minute.get("confirmed", False)
            else "NO"
        ),

        "Completed 15m Close": (
            fifteen_minute.get("current_close", "")
        ),

        "Previous 15m High": (
            fifteen_minute.get("previous_high", "")
        ),

        "15m Confirmation Candle": (
            fifteen_minute.get("candle", "")
        ),

        # Kept blank for backward compatibility
        # with the existing dashboard/history files.
        "Current 15m Candle": "",
        "Current 15m RSI": "",
        "Previous 15m Candle RSI": "",
        "15m RSI Change": "",
        "15m RSI Rising": "",
        "15m Rising Count": "",
        "History Transition": ("15m logic removed"),
        "Scan Mode": "15M MONITOR" if check_15m else "HOURLY",
        "Monitoring Status": "MONITORING" if check_15m else "NORMAL",
        "Monitoring Started": "",
        "Last 15m Check": "",
        "Category": category,
        "Signal": signal,
        "Reason": reason,
    }


# ============================================================
# LATEST RESULTS
# ============================================================

def save_latest_results(output):
    """
    These two files are the ONLY stable files
    required by GitHub Pages.
    """

    output.to_csv(
        LATEST_CSV_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    output.to_json(
        LATEST_JSON_FILE,
        orient="records",
        date_format="iso",
        force_ascii=False,
        indent=2,
    )

    print(
        f"Latest CSV saved: "
        f"{LATEST_CSV_FILE}"
    )

    print(
        f"Latest JSON saved: "
        f"{LATEST_JSON_FILE}"
    )


# ============================================================
# RSI HISTORY
# ============================================================

HISTORY_COLUMNS = [
    "Scan Time",
    "Symbol",
    "Current LTP",
    "Current Week RSI",
    "Previous Week RSI",
    "Weekly RSI Change",
    "Current Hour",
    "Current Hourly RSI",
    "Previous Hourly RSI",
    "Hourly RSI Change",
    "Hourly RSI Rising",
    "Current 15m Candle",
    "Current 15m RSI",
    "Previous 15m Candle RSI",
    "15m RSI Change",
    "15m RSI Rising",
    "15m Rising Count",
    "History Transition",
    "Scan Mode",
    "Monitoring Status",
    "Monitoring Started",
    "Last 15m Check",
    "Category",
    "Signal",
    "Reason",
]


def save_rsi_history(results):
    """
    Append one row per stock to RSI_History.csv.

    This function is intentionally independent from
    the dashboard output so a history problem cannot
    prevent latest_results.json from being generated.
    """

    history_rows = []

    for item in results:
        row = {
            column: item.get(
                column,
                "",
            )
            for column in HISTORY_COLUMNS
        }

        history_rows.append(row)

    new_history = pd.DataFrame(
        history_rows,
        columns=HISTORY_COLUMNS,
    )

    if HISTORY_FILE.exists():
        try:
            existing = pd.read_csv(
                HISTORY_FILE,
                encoding="utf-8-sig",
            )

            # Preserve older history while normalizing
            # it to the current column structure.
            existing = existing.reindex(
                columns=HISTORY_COLUMNS,
                fill_value="",
            )

            combined = pd.concat(
                [existing, new_history],
                ignore_index=True,
            )

        except Exception as error:
            print(
                "  ⚠️ Could not read existing "
                f"RSI_History.csv: {error}"
            )
            combined = new_history

    else:
        combined = new_history

    combined.to_csv(
        HISTORY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"History saved/appended: "
        f"{HISTORY_FILE}"
    )



# ============================================================
# PHASE 5A — RECORD NEW SETUPS
# ============================================================

SETUP_HISTORY_COLUMNS = [
    "Setup ID",
    "Setup Time",
    "Symbol",
    "Tag",
    "Entry Price",
    "Weekly RSI",
    "Hourly RSI",
    "15m Confirmation Candle",
    "15m Close",
    "Previous 15m High",
    "Status",
]


def record_new_setups(results):
    """Record each new SETUP once; do not change scanner logic."""

    if SETUP_HISTORY_FILE.exists():
        try:
            history = pd.read_csv(
                SETUP_HISTORY_FILE,
                encoding="utf-8-sig",
            )
        except Exception:
            history = pd.DataFrame()
    else:
        history = pd.DataFrame()

    for column in SETUP_HISTORY_COLUMNS:
        if column not in history.columns:
            history[column] = ""

    history = history.reindex(
        columns=SETUP_HISTORY_COLUMNS
    )

    active_symbols = set(
        history.loc[
            history["Status"].astype(str).str.upper().eq("ACTIVE"),
            "Symbol",
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    added = 0

    for result in results:
        if str(result.get("Category", "")).upper() != "SETUP":
            continue

        symbol = str(result.get("Symbol", "")).strip().upper()

        if not symbol or symbol in active_symbols:
            continue

        try:
            entry_price = float(result.get("Completed 15m Close"))
        except (TypeError, ValueError):
            try:
                entry_price = float(result.get("Current LTP"))
            except (TypeError, ValueError):
                print(f"  ⚠️ Cannot determine entry price for {symbol}")
                continue

        setup_time = result.get(
            "Scan Time",
            datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        )

        setup_id = (
            f"{symbol.replace(':', '_')}_"
            f"{datetime.now(IST).strftime('%Y%m%d_%H%M%S')}"
        )

        row = {
            "Setup ID": setup_id,
            "Setup Time": setup_time,
            "Symbol": symbol,
            "Tag": result.get("Tag", ""),
            "Entry Price": round(entry_price, 2),
            "Weekly RSI": result.get("Current Week RSI", ""),
            "Hourly RSI": result.get("Current Hourly RSI", ""),
            "15m Confirmation Candle": result.get(
                "15m Confirmation Candle", ""
            ),
            "15m Close": result.get("Completed 15m Close", ""),
            "Previous 15m High": result.get("Previous 15m High", ""),
            "Status": "ACTIVE",
        }

        history = pd.concat(
            [
                history,
                pd.DataFrame(
                    [row],
                    columns=SETUP_HISTORY_COLUMNS,
                ),
            ],
            ignore_index=True,
        )

        active_symbols.add(symbol)
        added += 1

        print(
            f"  🆕 NEW SETUP RECORDED: "
            f"{symbol} @ ₹{entry_price:.2f}"
        )

    history.to_csv(
        SETUP_HISTORY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Setup History: {len(history)} total records | "
        f"{added} new setup(s) recorded"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 90)
    print("RSI SCANNER - TWO SPEED MONITORING")
    print("Hourly RSI < 50 -> 15m monitoring")
    print("SETUP = Weekly RSI > 50 + Hourly RSI TOUCH <= 30 + 15m confirmation")
    print("=" * 90)

    now=datetime.now(IST)
    stocks=load_stocks()
    state=load_monitoring_state()
    full_hourly=hourly_cycle_due(now)
    results=[]

    print(f"Current IST: {now:%Y-%m-%d %H:%M:%S}")
    print(f"Master stocks: {len(stocks)}")
    print(f"15m monitoring stocks: {len(monitoring_symbols(state))}")
    print(f"Scan type: {'FULL HOURLY' if full_hourly else '15M MONITORING'}")

    if full_hourly:
        # Once per clock hour, calculate Weekly + Hourly RSI for ALL stocks.
        for n,(_,row) in enumerate(stocks.iterrows(),1):
            symbol=row["Symbol"]; tag=row.get("Tag","")
            print(f"\n[{n}/{len(stocks)}] {symbol} [HOURLY]")
            try:
                result=process_stock(symbol,tag,check_15m=False)
                if result is not None:
                    results.append(result)
                    state=update_monitoring_state(state,result,now)
            except Exception as e:
                print(f"  ERROR: {e}")
            time.sleep(0.3)

        save_hourly_cycle(now)

        # Immediately run 15m confirmation only for stocks that now qualify for monitoring.
        monitored=monitoring_symbols(state)
        monitored_rows=stocks[stocks["Symbol"].isin(monitored)]
        result_map={str(r.get("Symbol","")).strip().upper():r for r in results}
        for n,(_,row) in enumerate(monitored_rows.iterrows(),1):
            symbol=row["Symbol"]; tag=row.get("Tag","")
            print(f"\n[{n}/{len(monitored_rows)}] {symbol} [15M MONITOR]")
            try:
                result=process_stock(symbol,tag,check_15m=True)
                if result is not None:
                    result_map[symbol.upper()]=result
                    state=update_monitoring_state(state,result,now)
            except Exception as e:
                print(f"  ERROR: {e}")
            time.sleep(0.3)
        results=list(result_map.values())
    else:
        # Between hourly scans, only monitored stocks are processed.
        monitored=monitoring_symbols(state)
        monitored_rows=stocks[stocks["Symbol"].isin(monitored)]
        for n,(_,row) in enumerate(monitored_rows.iterrows(),1):
            symbol=row["Symbol"]; tag=row.get("Tag","")
            print(f"\n[{n}/{len(monitored_rows)}] {symbol} [15M MONITOR]")
            try:
                result=process_stock(symbol,tag,check_15m=True)
                if result is not None:
                    results.append(result)
                    # Do not remove monitoring between hourly checkpoints.
            except Exception as e:
                print(f"  ERROR: {e}")
            time.sleep(0.3)
        state=mark_15m_check(state,monitored,now)

    save_monitoring_state(state)

    if not results:
        print("\nNo results generated.")
        return 0

    output=pd.DataFrame(results)
    priority={"SETUP":1,"WATCH":2,"WAIT":3,"IGNORE":4}
    output["_priority"]=output["Category"].map(priority).fillna(99)
    output=output.sort_values(["_priority","Current Hourly RSI"],ascending=[True,True]).drop(columns=["_priority"]).reset_index(drop=True)

    output.to_csv(OUTPUT_FILE,index=False,encoding="utf-8-sig")
    save_latest_results(output)

    try: save_rsi_history(results)
    except Exception as e: print(f"  WARNING: RSI history update failed: {e}")
    try: record_new_setups(results)
    except Exception as e: print(f"  WARNING: Setup history update failed: {e}")

    counts={c:int((output["Category"]==c).sum()) for c in ["SETUP","WATCH","WAIT","IGNORE"]}
    print("\n"+"="*90)
    print("FINAL RSI SCANNER RESULT")
    print("="*90)
    cols=["Scan Time","Symbol","Current LTP","Current Week RSI","Current Hourly RSI","Hourly RSI Change","Hourly RSI Rising","Scan Mode","Monitoring Status","Category","Signal","Reason"]
    print(output[[c for c in cols if c in output.columns]].to_string(index=False))
    print("\n"+"="*90)
    print(f"🔥 SETUP : {counts['SETUP']}")
    print(f"👀 WATCH : {counts['WATCH']}")
    print(f"⏳ WAIT  : {counts['WAIT']}")
    print(f"❌ IGNORE: {counts['IGNORE']}")
    print(f"🔴 15M MONITORING: {len(monitoring_symbols(state))}")
    print(f"Scan Type: {'FULL HOURLY + 15M MONITORING' if full_hourly else '15M MONITORING'}")
    print(f"Report: {OUTPUT_FILE}")
    print(f"Latest JSON: {LATEST_JSON_FILE}")
    print(f"Monitoring State: {MONITORING_STATE_FILE}")
    print("="*90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
