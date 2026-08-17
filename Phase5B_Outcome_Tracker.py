import os
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import quote

import pandas as pd
import requests


# ============================================================
# PHASE 5B — OUTCOME TRACKER
# ============================================================
#
# Independent from the existing RSI dashboard/project.
#
# Reads:
#   Setup_History.csv
#
# Updates:
#   Setup_History.csv
#
# Logic:
#   Entry price = setup Entry Price
#
#   Target / stop levels are checked using INTRADAY HIGH/LOW
#   after the setup time:
#
#       +1%, +2%, +3%, +5%
#       -1%, -2%, -3%
#
#   1D / 3D / 5D return:
#       Close of the 1st / 3rd / 5th trading day AFTER setup date
#       compared with Entry Price.
#
#   Setup becomes COMPLETED when the 5th trading-day close is
#   available.
#
# IMPORTANT:
#   If target and stop are both reached in the same candle,
#   both flags can be YES. This version does not assume which
#   happened first.
# ============================================================


SCRIPT_FOLDER = Path(__file__).resolve().parent
SETUP_HISTORY_FILE = SCRIPT_FOLDER / "Setup_History.csv"

UPSTOX_API = "https://api.upstox.com"

ACCESS_TOKEN = os.getenv(
    "UPSTOX_ACCESS_TOKEN",
    ""
).strip()

if not ACCESS_TOKEN:
    raise SystemExit(
        "ERROR: UPSTOX_ACCESS_TOKEN is not configured."
    )

HEADERS = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Accept": "application/json",
}

IST = "Asia/Kolkata"

TARGET_COLUMNS = [
    "Plus 1% Hit",
    "Plus 2% Hit",
    "Plus 3% Hit",
    "Plus 5% Hit",
    "Minus 1% Hit",
    "Minus 2% Hit",
    "Minus 3% Hit",
    "Max Gain %",
    "Max Loss %",
    "1D Return %",
    "3D Return %",
    "5D Return %",
    "Status",
    "Completed Time",
]

REQUIRED_COLUMNS = [
    "Setup ID",
    "Setup Time",
    "Symbol",
    "Tag",
    "Entry Price",
    "Status",
]


# ============================================================
# HELPERS
# ============================================================

def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_symbol(symbol):
    return (
        str(symbol)
        .strip()
        .upper()
        .replace("\\", "")
    )


def get_setup_date(value):
    timestamp = pd.to_datetime(
        value,
        errors="coerce"
    )

    if pd.isna(timestamp):
        return None

    return timestamp.date()


def get_setup_timestamp(value):
    timestamp = pd.to_datetime(
        value,
        errors="coerce"
    )

    if pd.isna(timestamp):
        return None

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(
            IST
        )
    else:
        timestamp = timestamp.tz_convert(
            IST
        )

    return timestamp


def format_pct(value):
    if value is None:
        return ""

    return round(
        float(value),
        2
    )


# ============================================================
# UPSTOX
# ============================================================

def find_instrument_key(symbol):
    trading_symbol = normalize_symbol(
        symbol
    ).replace(
        "NSE:",
        ""
    )

    try:
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
    except requests.RequestException as error:
        print(
            f"  ❌ Instrument search error: {error}"
        )
        return None

    if response.status_code != 200:
        print(
            f"  ❌ Instrument search failed: "
            f"{response.status_code} "
            f"{response.text[:200]}"
        )
        return None

    instruments = (
        response.json()
        .get("data", [])
    )

    for item in instruments:
        if (
            str(
                item.get(
                    "trading_symbol",
                    ""
                )
            ).upper()
            == trading_symbol
            and item.get("segment") == "NSE_EQ"
        ):
            return item.get(
                "instrument_key"
            )

    for item in instruments:
        if item.get("segment") == "NSE_EQ":
            return item.get(
                "instrument_key"
            )

    return None


def get_intraday_15m(
    instrument_key
):
    encoded_key = quote(
        instrument_key,
        safe=""
    )

    try:
        response = requests.get(
            f"{UPSTOX_API}/v3/historical-candle/"
            f"intraday/{encoded_key}/minutes/15",
            headers=HEADERS,
            timeout=20,
        )
    except requests.RequestException as error:
        print(
            f"  ❌ 15m request error: {error}"
        )
        return []

    if response.status_code != 200:
        print(
            f"  ❌ 15m request failed: "
            f"{response.status_code} "
            f"{response.text[:200]}"
        )
        return []

    return (
        response.json()
        .get("data", {})
        .get("candles", [])
    )


def get_daily_candles(
    instrument_key,
    from_date,
    to_date
):
    encoded_key = quote(
        instrument_key,
        safe=""
    )

    try:
        response = requests.get(
            f"{UPSTOX_API}/v3/historical-candle/"
            f"{encoded_key}/days/1/"
            f"{to_date}/{from_date}",
            headers=HEADERS,
            timeout=20,
        )
    except requests.RequestException as error:
        print(
            f"  ❌ Daily candle request error: {error}"
        )
        return []

    if response.status_code != 200:
        print(
            f"  ❌ Daily candle request failed: "
            f"{response.status_code} "
            f"{response.text[:200]}"
        )
        return []

    return (
        response.json()
        .get("data", {})
        .get("candles", [])
    )


# ============================================================
# CANDLE DATAFRAME
# ============================================================

def candles_to_dataframe(
    candles
):
    if not candles:
        return pd.DataFrame()

    rows = []

    for candle in candles:
        if len(candle) < 6:
            continue

        rows.append(
            {
                "timestamp": candle[0],
                "open": safe_float(candle[1]),
                "high": safe_float(candle[2]),
                "low": safe_float(candle[3]),
                "close": safe_float(candle[4]),
                "volume": safe_float(candle[5]),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )

    df = df.dropna(
        subset=["timestamp"]
    )

    df["timestamp_ist"] = (
        df["timestamp"]
        .dt.tz_convert(IST)
    )

    return (
        df.sort_values("timestamp_ist")
        .reset_index(drop=True)
    )


# ============================================================
# TARGET / STOP CALCULATION
# ============================================================

def update_intraday_outcome(
    row,
    candles
):
    entry = safe_float(
        row["Entry Price"]
    )

    if entry is None or entry <= 0:
        return row

    setup_time = get_setup_timestamp(
        row["Setup Time"]
    )

    if setup_time is None:
        return row

    df = candles_to_dataframe(
        candles
    )

    if df.empty:
        return row

    # Only candles AFTER the setup time.
    #
    # This is important:
    # We must never count price movement that happened
    # before the setup existed.
    after_setup = df[
        df["timestamp_ist"] > setup_time
    ].copy()

    if after_setup.empty:
        return row

    max_high = pd.to_numeric(
        after_setup["high"],
        errors="coerce"
    ).max()

    min_low = pd.to_numeric(
        after_setup["low"],
        errors="coerce"
    ).min()

    if pd.notna(max_high):
        max_gain = (
            (max_high - entry)
            / entry
            * 100
        )

        row["Max Gain %"] = format_pct(
            max_gain
        )

        row["Plus 1% Hit"] = (
            "YES"
            if max_gain >= 1
            else row.get(
                "Plus 1% Hit",
                ""
            )
        )

        row["Plus 2% Hit"] = (
            "YES"
            if max_gain >= 2
            else row.get(
                "Plus 2% Hit",
                ""
            )
        )

        row["Plus 3% Hit"] = (
            "YES"
            if max_gain >= 3
            else row.get(
                "Plus 3% Hit",
                ""
            )
        )

        row["Plus 5% Hit"] = (
            "YES"
            if max_gain >= 5
            else row.get(
                "Plus 5% Hit",
                ""
            )
        )

    if pd.notna(min_low):
        max_loss = (
            (min_low - entry)
            / entry
            * 100
        )

        row["Max Loss %"] = format_pct(
            max_loss
        )

        row["Minus 1% Hit"] = (
            "YES"
            if max_loss <= -1
            else row.get(
                "Minus 1% Hit",
                ""
            )
        )

        row["Minus 2% Hit"] = (
            "YES"
            if max_loss <= -2
            else row.get(
                "Minus 2% Hit",
                ""
            )
        )

        row["Minus 3% Hit"] = (
            "YES"
            if max_loss <= -3
            else row.get(
                "Minus 3% Hit",
                ""
            )
        )

    return row


# ============================================================
# DAILY RETURNS
# ============================================================

def update_daily_returns(
    row,
    daily_candles
):
    entry = safe_float(
        row["Entry Price"]
    )

    setup_date = get_setup_date(
        row["Setup Time"]
    )

    if entry is None or entry <= 0:
        return row

    if setup_date is None:
        return row

    df = candles_to_dataframe(
        daily_candles
    )

    if df.empty:
        return row

    # Daily candles after the setup date.
    df["trade_date"] = (
        df["timestamp_ist"].dt.date
    )

    future = df[
        df["trade_date"] > setup_date
    ].copy()

    if future.empty:
        return row

    # The first 5 trading days AFTER setup date.
    future = (
        future.sort_values("trade_date")
        .drop_duplicates(
            "trade_date",
            keep="last"
        )
        .head(5)
    )

    closes = list(
        pd.to_numeric(
            future["close"],
            errors="coerce"
        ).dropna()
    )

    returns = [
        "1D Return %",
        "3D Return %",
        "5D Return %",
    ]

    positions = [
        0,
        2,
        4,
    ]

    for column, position in zip(
        returns,
        positions
    ):
        if len(closes) > position:
            close_price = closes[position]

            row[column] = format_pct(
                (
                    (close_price - entry)
                    / entry
                    * 100
                )
            )

    # Completed once the 5th trading-day close exists.
    if len(closes) >= 5:
        row["Status"] = "COMPLETED"

        setup_time = get_setup_timestamp(
            row["Setup Time"]
        )

        if setup_time is not None:
            row["Completed Time"] = (
                future.iloc[4]["timestamp_ist"]
                .strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )

    return row


# ============================================================
# MAIN TRACKER
# ============================================================

def main():
    print("=" * 90)
    print("PHASE 5B — OUTCOME TRACKER")
    print("=" * 90)

    if not SETUP_HISTORY_FILE.exists():
        raise SystemExit(
            f"Setup_History.csv not found: "
            f"{SETUP_HISTORY_FILE}"
        )

    history = pd.read_csv(
        SETUP_HISTORY_FILE,
        encoding="utf-8-sig",
    )

    if history.empty:
        print("No setups found.")
        return

    # Ensure all Phase 5B columns exist.
    for column in TARGET_COLUMNS:
        if column not in history.columns:
            history[column] = ""

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in history.columns
    ]

    if missing:
        raise SystemExit(
            "Setup_History.csv is missing columns: "
            + ", ".join(missing)
        )

    active_mask = (
        history["Status"]
        .astype(str)
        .str.upper()
        .eq("ACTIVE")
    )

    active_indexes = list(
        history.index[active_mask]
    )

    print(
        f"Total setups: {len(history)}"
    )
    print(
        f"Active setups: {len(active_indexes)}"
    )
    print()

    today = datetime.now().date()

    for count, index in enumerate(
        active_indexes,
        start=1
    ):
        row = history.loc[index]

        symbol = normalize_symbol(
            row["Symbol"]
        )

        print(
            f"[{count}/{len(active_indexes)}] "
            f"{symbol}"
        )

        entry = safe_float(
            row["Entry Price"]
        )

        setup_date = get_setup_date(
            row["Setup Time"]
        )

        print(
            f"  Entry: ₹{entry:.2f}"
            if entry is not None
            else "  Entry: INVALID"
        )

        print(
            f"  Setup date: {setup_date}"
        )

        instrument_key = find_instrument_key(
            symbol
        )

        if not instrument_key:
            print(
                "  ❌ Instrument not found"
            )
            print()
            continue

        # ----------------------------------------------------
        # Intraday target/stop tracking
        # ----------------------------------------------------

        intraday = get_intraday_15m(
            instrument_key
        )

        history.loc[index] = (
            update_intraday_outcome(
                history.loc[index].copy(),
                intraday
            )
        )

        # ----------------------------------------------------
        # Daily 1D/3D/5D tracking
        # ----------------------------------------------------

        if setup_date is not None:
            from_date = (
                setup_date
                - timedelta(days=2)
            ).strftime("%Y-%m-%d")

            to_date = (
                today
                + timedelta(days=2)
            ).strftime("%Y-%m-%d")

            daily = get_daily_candles(
                instrument_key,
                from_date,
                to_date
            )

            history.loc[index] = (
                update_daily_returns(
                    history.loc[index].copy(),
                    daily
                )
            )

        row = history.loc[index]

        print(
            f"  Max Gain: {row['Max Gain %']}"
        )
        print(
            f"  Max Loss: {row['Max Loss %']}"
        )
        print(
            f"  +1/+2/+3/+5: "
            f"{row['Plus 1% Hit']}/"
            f"{row['Plus 2% Hit']}/"
            f"{row['Plus 3% Hit']}/"
            f"{row['Plus 5% Hit']}"
        )
        print(
            f"  -1/-2/-3: "
            f"{row['Minus 1% Hit']}/"
            f"{row['Minus 2% Hit']}/"
            f"{row['Minus 3% Hit']}"
        )
        print(
            f"  1D/3D/5D: "
            f"{row['1D Return %']}/"
            f"{row['3D Return %']}/"
            f"{row['5D Return %']}"
        )
        print(
            f"  Status: {row['Status']}"
        )
        print()

    # Save with the same UTF-8 format used by Phase 5A.
    history.to_csv(
        SETUP_HISTORY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 90)
    print(
        "Setup_History.csv updated successfully."
    )
    print("=" * 90)


if __name__ == "__main__":
    main()
