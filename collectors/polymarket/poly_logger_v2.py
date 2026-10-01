import json
import sqlite3
import time
from datetime import datetime, timezone

import requests

DB_FILE = "polymarket_btc15m_v2.db"
EVENT_BASE_URL = "https://gamma-api.polymarket.com/events/slug"
BOOK_URL = "https://clob.polymarket.com/book"
MIDPOINT_URL = "https://clob.polymarket.com/midpoint"
COINBASE_SPOT_URL = "https://api.coinbase.com/v2/prices/BTC-USD/spot"
POLL_SECONDS = 5

session = requests.Session()
session.headers.update({"User-Agent": "polyshi-polymarket-research/2.0"})

market_cache = {}


def utc_now():
    return datetime.now(timezone.utc)


def current_btc_15m_slug():
    ts = int(time.time() // 900) * 900
    return f"btc-updown-15m-{ts}"


def get_json(url, params=None, allow_404=False):
    last_error = None
    for _ in range(2):
        try:
            r = session.get(url, params=params, timeout=15)
            if allow_404 and r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last_error = e
            time.sleep(1)
    raise last_error


def setup_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS markets (
        slug TEXT PRIMARY KEY,
        market_id TEXT,
        condition_id TEXT,
        question TEXT,
        start_iso TEXT,
        end_iso TEXT,
        up_token_id TEXT,
        down_token_id TEXT,
        resolution_source TEXT,
        event_metadata_json TEXT,
        created_at TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS ticks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_iso TEXT,
        slug TEXT,
        seconds_left REAL,
        up_bid REAL,
        up_ask REAL,
        up_midpoint REAL,
        up_spread REAL,
        up_bid_size REAL,
        up_ask_size REAL,
        down_bid REAL,
        down_ask REAL,
        down_midpoint REAL,
        down_spread REAL,
        down_bid_size REAL,
        down_ask_size REAL,
        up_api_midpoint REAL,
        up_last_trade REAL,
        down_last_trade REAL,
        up_book_hash TEXT,
        down_book_hash TEXT,
        btc_spot REAL,
        FOREIGN KEY (slug) REFERENCES markets(slug)
    )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ticks_slug ON ticks(slug)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ticks_ts ON ticks(ts_iso)")
    conn.commit()
    return conn


def get_event(slug):
    return get_json(f"{EVENT_BASE_URL}/{slug}", allow_404=True)


def extract_token_ids(market):
    outcomes = market.get("outcomes", "[]")
    token_ids = market.get("clobTokenIds", "[]")
    outcomes = json.loads(outcomes) if isinstance(outcomes, str) else outcomes
    token_ids = json.loads(token_ids) if isinstance(token_ids, str) else token_ids

    up_token, down_token = None, None
    for outcome, token_id in zip(outcomes, token_ids):
        name = str(outcome).strip().lower()
        if name == "up":
            up_token = str(token_id)
        elif name == "down":
            down_token = str(token_id)

    if not up_token or not down_token:
        raise ValueError(f"Could not map outcomes to Up/Down: {outcomes}")
    return up_token, down_token


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_levels(levels):
    parsed = []
    for lvl in levels or []:
        p = to_float(lvl.get("price"))
        s = to_float(lvl.get("size"))
        if p is not None and s is not None and s > 0:
            parsed.append((p, s))
    return parsed


def book_summary(token_id):
    book = get_json(BOOK_URL, {"token_id": token_id}, allow_404=True)
    if not book:
        return {"bid": None, "ask": None, "mid": None, "spread": None,
                "bid_size": None, "ask_size": None, "last": None, "hash": None}

    bids = parse_levels(book.get("bids"))
    asks = parse_levels(book.get("asks"))

    bid, bid_size = max(bids, key=lambda x: x[0]) if bids else (None, None)
    ask, ask_size = min(asks, key=lambda x: x[0]) if asks else (None, None)

    mid = round((bid + ask) / 2, 4) if bid is not None and ask is not None else None
    spread = round(ask - bid, 4) if bid is not None and ask is not None else None

    return {"bid": bid, "ask": ask, "mid": mid, "spread": spread,
            "bid_size": bid_size, "ask_size": ask_size,
            "last": to_float(book.get("last_trade_price")),
            "hash": book.get("hash")}


def get_api_midpoint(token_id):
    data = get_json(MIDPOINT_URL, {"token_id": token_id}, allow_404=True)
    if isinstance(data, dict):
        return to_float(data.get("mid", data.get("midpoint")))
    return to_float(data)


def get_btc_spot():
    try:
        data = get_json(COINBASE_SPOT_URL)
        return float(data["data"]["amount"])
    except Exception:
        return None


def upsert_market(conn, event, market, up_token, down_token):
    meta = market.get("eventMetadata") or event.get("eventMetadata") or event.get("metadata")
    conn.execute("""
    INSERT OR IGNORE INTO markets (
        slug, market_id, condition_id, question, start_iso, end_iso,
        up_token_id, down_token_id, resolution_source, event_metadata_json, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        market.get("slug") or event.get("slug"),
        str(market.get("id")),
        market.get("conditionId"),
        market.get("question"),
        market.get("eventStartTime") or market.get("startDate"),
        market.get("endDate"),
        up_token,
        down_token,
        market.get("resolutionSource") or event.get("resolutionSource"),
        json.dumps(meta) if meta else None,
        utc_now().isoformat(),
    ))
    conn.commit()


def insert_tick(conn, row):
    cols = ", ".join(row.keys())
    qs = ", ".join("?" for _ in row)
    conn.execute(f"INSERT INTO ticks ({cols}) VALUES ({qs})", tuple(row.values()))
    conn.commit()


def fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "None"


def main():
    print("Polymarket BTC 15m logger v2 started.")
    print(f"Database: {DB_FILE}")
    print("Stop with Ctrl+C.\n")
    conn = setup_db()
    last_slug = None

    while True:
        start = time.time()
        try:
            slug = current_btc_15m_slug()

            if slug not in market_cache:
                event = get_event(slug)
                if not event or not event.get("markets"):
                    print(f"[{utc_now():%H:%M:%S}] No market found for slug: {slug}")
                    time.sleep(POLL_SECONDS)
                    continue
                market = event["markets"][0]
                up_token, down_token = extract_token_ids(market)
                end_dt = datetime.fromisoformat(market["endDate"].replace("Z", "+00:00"))
                upsert_market(conn, event, market, up_token, down_token)
                market_cache.clear()
                market_cache[slug] = (up_token, down_token, end_dt)

            up_token, down_token, end_dt = market_cache[slug]

            if slug != last_slug:
                print(f"\nNow tracking: {slug}")
                print(f"Ends: {end_dt.isoformat()}")
                last_slug = slug

            up = book_summary(up_token)
            down = book_summary(down_token)
            up_api_mid = get_api_midpoint(up_token)
            btc = get_btc_spot()
            now = utc_now()

            row = {
                "ts_iso": now.isoformat(),
                "slug": slug,
                "seconds_left": round((end_dt - now).total_seconds(), 3),
                "up_bid": up["bid"], "up_ask": up["ask"],
                "up_midpoint": up["mid"], "up_spread": up["spread"],
                "up_bid_size": up["bid_size"], "up_ask_size": up["ask_size"],
                "down_bid": down["bid"], "down_ask": down["ask"],
                "down_midpoint": down["mid"], "down_spread": down["spread"],
                "down_bid_size": down["bid_size"], "down_ask_size": down["ask_size"],
                "up_api_midpoint": up_api_mid,
                "up_last_trade": up["last"], "down_last_trade": down["last"],
                "up_book_hash": up["hash"], "down_book_hash": down["hash"],
                "btc_spot": btc,
            }
            insert_tick(conn, row)

            flag = ""
            if up["mid"] is not None and up_api_mid is not None and abs(up["mid"] - up_api_mid) > 0.02:
                flag = "  <-- book/midpoint mismatch"

            print(
                f"[{now:%H:%M:%S}] {slug} | {row['seconds_left']:6.1f}s left | "
                f"Up bid/ask/mid: {fmt(up['bid'])} / {fmt(up['ask'])} / {fmt(up['mid'])} | "
                f"spread: {fmt(up['spread'])} | api mid: {fmt(up_api_mid)} | "
                f"BTC {btc:,.2f}" if btc else "BTC None"
            )
            if flag:
                print(flag)

        except requests.RequestException as e:
            print(f"[{utc_now():%H:%M:%S}] API/network error: {str(e)[:200]}")
        except Exception as e:
            print(f"[{utc_now():%H:%M:%S}] Unexpected error: {type(e).__name__}: {e}")

        time.sleep(max(0.1, POLL_SECONDS - (time.time() - start)))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nLogger v2 stopped safely.")
