import csv
import json
import os
import time
from datetime import datetime, timezone

import requests

BASE_URLS = [
    "https://external-api.kalshi.com/trade-api/v2",
    "https://api.elections.kalshi.com/trade-api/v2",
]
SERIES_TICKER = "KXBTC15M"
POLL_SECONDS = 5
OUTPUT_FILE = "kalshi_btc_15m_snapshots.csv"

FIELDNAMES = [
    "timestamp_utc", "market_ticker", "event_ticker", "market_title",
    "window_start_utc", "window_end_utc", "seconds_left", "status",
    "yes_bid", "yes_ask", "yes_midpoint", "yes_spread",
    "no_bid", "no_ask", "no_midpoint", "no_spread",
    "yes_bid_size", "yes_ask_size", "no_bid_size", "no_ask_size",
    "volume", "volume_24h", "liquidity", "last_price", "open_interest",
    "floor_strike", "market_result", "settlement_value",
    "resolution_source", "raw_orderbook_json",
]

session = requests.Session()
session.headers.update({"User-Agent": "polyshi-kalshi-research/1.1"})


def utc_now():
    return datetime.now(timezone.utc)


def parse_utc(s):
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def api_get(path, params=None):
    errors = []
    for attempt in range(2):
        for base in BASE_URLS:
            try:
                r = session.get(base + path, params=params, timeout=15)
                r.raise_for_status()
                return r.json()
            except requests.RequestException as e:
                errors.append(f"{base}: {e}")
        time.sleep(1)
    raise requests.RequestException(" | ".join(errors[-2:]))


def pick_live_market():
    payload = api_get("/markets", {"series_ticker": SERIES_TICKER, "status": "open", "limit": 100})
    markets = payload.get("markets", [])
    if not markets:
        return None
    now = utc_now()

    def score(m):
        close = parse_utc(m.get("close_time") or m.get("expected_expiration_time"))
        if close is None or close < now:
            return float("inf")
        return (close - now).total_seconds()

    return min(markets, key=score)


def best_bid(levels):
    parsed = []
    for lvl in levels or []:
        try:
            parsed.append((float(lvl[0]), float(lvl[1])))
        except (TypeError, ValueError, IndexError):
            continue
    if not parsed:
        return None, None
    return max(parsed, key=lambda x: x[0])


def get_book_prices(orderbook):
    book = orderbook.get("orderbook_fp", orderbook.get("orderbook", {})) or {}
    yes_bid, yes_bid_size = best_bid(book.get("yes_dollars", book.get("yes")))
    no_bid, no_bid_size = best_bid(book.get("no_dollars", book.get("no")))

    yes_ask = round(1 - no_bid, 4) if no_bid is not None else None
    no_ask = round(1 - yes_bid, 4) if yes_bid is not None else None

    def mid(b, a):
        return round((b + a) / 2, 4) if b is not None and a is not None else None

    def spr(b, a):
        return round(a - b, 4) if b is not None and a is not None else None

    return {
        "yes_bid": yes_bid, "yes_ask": yes_ask,
        "yes_midpoint": mid(yes_bid, yes_ask), "yes_spread": spr(yes_bid, yes_ask),
        "no_bid": no_bid, "no_ask": no_ask,
        "no_midpoint": mid(no_bid, no_ask), "no_spread": spr(no_bid, no_ask),
        "yes_bid_size": yes_bid_size, "yes_ask_size": no_bid_size,
        "no_bid_size": no_bid_size, "no_ask_size": yes_bid_size,
    }


def make_row(market, orderbook):
    now = utc_now()
    close = parse_utc(market.get("close_time") or market.get("expected_expiration_time"))
    row = {f: None for f in FIELDNAMES}
    row.update({
        "timestamp_utc": now.isoformat(),
        "market_ticker": market.get("ticker"),
        "event_ticker": market.get("event_ticker"),
        "market_title": market.get("title") or market.get("yes_sub_title"),
        "window_start_utc": market.get("open_time"),
        "window_end_utc": close.isoformat() if close else None,
        "seconds_left": round((close - now).total_seconds(), 3) if close else None,
        "status": market.get("status"),
        "volume": market.get("volume_fp", market.get("volume")),
        "volume_24h": market.get("volume_24h_fp", market.get("volume_24h")),
        "liquidity": market.get("liquidity_dollars"),
        "last_price": market.get("last_price_dollars"),
        "open_interest": market.get("open_interest_fp", market.get("open_interest")),
        "floor_strike": market.get("floor_strike"),
        "market_result": market.get("result"),
        "settlement_value": market.get("settlement_value"),
        "resolution_source": "CF Benchmarks Bitcoin Real Time Index",
        "raw_orderbook_json": json.dumps(orderbook, separators=(",", ":")),
    })
    row.update(get_book_prices(orderbook))
    return row


def append_row(row):
    new = not os.path.exists(OUTPUT_FILE) or os.path.getsize(OUTPUT_FILE) == 0
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if new:
            w.writeheader()
        w.writerow(row)


def fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "None"


def main():
    print("Kalshi BTC 15-minute collector started.")
    print(f"Writing snapshots to: {os.path.abspath(OUTPUT_FILE)}")
    print("Stop with Ctrl+C.\n")
    last_ticker = None

    while True:
        start = time.time()
        try:
            market = pick_live_market()
            if market is None:
                print(f"{utc_now().isoformat()} | No open {SERIES_TICKER} market found.")
            else:
                ticker = market["ticker"]
                if ticker != last_ticker:
                    print(f"\nNow tracking: {ticker}")
                    print(f"Window: {market.get('open_time')} to {market.get('close_time')}")
                    print(f"Strike: {market.get('floor_strike')}")
                    last_ticker = ticker

                ob = api_get(f"/markets/{ticker}/orderbook", {"depth": 100})
                row = make_row(market, ob)
                append_row(row)
                print(
                    f"{row['timestamp_utc'][11:19]} | {ticker} | "
                    f"{row['seconds_left']:6.1f}s left | "
                    f"bid/ask/mid: {fmt(row['yes_bid'])} / {fmt(row['yes_ask'])} / "
                    f"{fmt(row['yes_midpoint'])} | spread: {fmt(row['yes_spread'])} | "
                    f"vol: {row['volume']}"
                )
        except requests.RequestException as e:
            print(f"{utc_now().isoformat()} | API/network error: {str(e)[:200]}")
        except Exception as e:
            print(f"{utc_now().isoformat()} | Unexpected error: {type(e).__name__}: {e}")

        time.sleep(max(0.1, POLL_SECONDS - (time.time() - start)))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCollector stopped safely.")