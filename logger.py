import json
import sqlite3
import time
from datetime import datetime, timezone

import requests

DB_FILE = "polymarket_btc15m.db"
EVENT_BASE_URL = "https://gamma-api.polymarket.com/events/slug"
MIDPOINT_URL = "https://clob.polymarket.com/midpoint"
COINBASE_SPOT_URL = "https://api.coinbase.com/v2/prices/BTC-USD/spot"
POLL_SECONDS = 5


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def current_btc_15m_slug():
    ts = int(time.time() // 900) * 900
    return f"btc-updown-15m-{ts}"


def setup_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS markets (
        slug TEXT PRIMARY KEY,
        market_id TEXT,
        question TEXT,
        start_iso TEXT,
        end_iso TEXT,
        up_token_id TEXT,
        down_token_id TEXT,
        created_at TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS ticks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_iso TEXT,
        slug TEXT,
        seconds_left INTEGER,
        up_price REAL,
        down_price REAL,
        btc_spot REAL,
        FOREIGN KEY (slug) REFERENCES markets(slug)
    )
    """)

    conn.commit()
    return conn


def get_event_market_by_slug(slug):
    resp = requests.get(f"{EVENT_BASE_URL}/{slug}", timeout=15)

    if resp.status_code == 404:
        return None

    resp.raise_for_status()
    event = resp.json()

    markets = event.get("markets", [])
    if not markets:
        return None

    return markets[0]


def extract_token_ids(market):
    outcomes_raw = market.get("outcomes", "[]")
    token_ids_raw = market.get("clobTokenIds", "[]")

    outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
    token_ids = json.loads(token_ids_raw) if isinstance(token_ids_raw, str) else token_ids_raw

    if len(outcomes) != 2 or len(token_ids) != 2:
        raise ValueError(f"Expected 2 outcomes and 2 token IDs, got outcomes={outcomes}, token_ids={token_ids}")

    up_token = None
    down_token = None

    for outcome, token_id in zip(outcomes, token_ids):
        name = str(outcome).strip().lower()
        if name == "up":
            up_token = str(token_id)
        elif name == "down":
            down_token = str(token_id)

    if not up_token or not down_token:
        raise ValueError(f"Could not map outcomes to Up/Down: {outcomes}")

    return up_token, down_token


def get_midpoint(token_id):
    resp = requests.get(
        MIDPOINT_URL,
        params={"token_id": str(token_id)},
        timeout=15
    )

    if resp.status_code == 404:
        return None

    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, dict):
        if "mid" in data:
            return data["mid"]
        if "midpoint" in data:
            return data["midpoint"]

    return data


def get_btc_spot():
    resp = requests.get(COINBASE_SPOT_URL, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return float(data["data"]["amount"])


def upsert_market(conn, market, up_token_id, down_token_id):
    cur = conn.cursor()
    cur.execute("""
    INSERT OR IGNORE INTO markets (
        slug, market_id, question, start_iso, end_iso, up_token_id, down_token_id, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        market.get("slug"),
        str(market.get("id")),
        market.get("question"),
        market.get("startDate"),
        market.get("endDate"),
        up_token_id,
        down_token_id,
        utc_now_iso()
    ))
    conn.commit()


def insert_tick(conn, slug, seconds_left, up_price, down_price, btc_spot):
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO ticks (
        ts_iso, slug, seconds_left, up_price, down_price, btc_spot
    ) VALUES (?, ?, ?, ?, ?, ?)
    """, (
        utc_now_iso(),
        slug,
        seconds_left,
        up_price,
        down_price,
        btc_spot
    ))
    conn.commit()


def main():
    print("Starting BTC 15m logger...")
    conn = setup_db()

    while True:
        try:
            slug = current_btc_15m_slug()
            market = get_event_market_by_slug(slug)

            if not market:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] No market found for slug: {slug}")
                time.sleep(POLL_SECONDS)
                continue

            end_iso = market.get("endDate")
            if not end_iso:
                raise ValueError("Market missing endDate")

            end_dt = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
            seconds_left = int((end_dt - datetime.now(timezone.utc)).total_seconds())

            up_token_id, down_token_id = extract_token_ids(market)
            upsert_market(conn, market, up_token_id, down_token_id)

            up_raw = get_midpoint(up_token_id)
            down_raw = get_midpoint(down_token_id)

            up_price = float(up_raw) if up_raw not in (None, "") else None
            down_price = float(down_raw) if down_raw not in (None, "") else None
            btc_spot = get_btc_spot()

            insert_tick(conn, slug, seconds_left, up_price, down_price, btc_spot)

            up_text = f"{up_price:.3f}" if up_price is not None else "None"
            down_text = f"{down_price:.3f}" if down_price is not None else "None"

            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"{slug} | {seconds_left:>4}s left | "
                f"Up {up_text} | Down {down_text} | BTC {btc_spot:,.2f}"
            )

        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error: {e}")

        time.sleep(POLL_SECONDS)

    conn.close()


if __name__ == "__main__":
    main()