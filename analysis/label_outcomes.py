from pathlib import Path
import json
import sqlite3
import time

import pandas as pd
import requests


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
GAMMA_URL = "https://gamma-api.polymarket.com"

OUTPUT_DIR.mkdir(exist_ok=True)


def parse_array(value):
    if isinstance(value, list):
        return value

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []

    return []


database_files = list(DATA_DIR.glob("*.db"))

if not database_files:
    raise FileNotFoundError(f"No .db file found in {DATA_DIR}")

db_path = max(database_files, key=lambda path: path.stat().st_mtime)

print(f"Database: {db_path.name}")

with sqlite3.connect(db_path) as connection:
    markets = pd.read_sql_query(
        """
        SELECT
            m.slug,
            m.market_id,
            m.question,
            m.start_iso,
            m.end_iso,
            m.up_token_id,
            m.down_token_id,
            COUNT(t.id) AS tick_count,
            MAX(t.seconds_left) AS max_seconds,
            MIN(t.seconds_left) AS min_seconds
        FROM markets AS m
        LEFT JOIN ticks AS t
            ON m.slug = t.slug
        GROUP BY
            m.slug,
            m.market_id,
            m.question,
            m.start_iso,
            m.end_iso,
            m.up_token_id,
            m.down_token_id
        ORDER BY m.end_iso
        """,
        connection,
    )

markets["coverage_complete"] = (
    (markets["max_seconds"] >= 890)
    & (markets["min_seconds"] <= 10)
)

session = requests.Session()
session.headers.update(
    {"User-Agent": "polymarket-calibration-project/1.0"}
)

results = []
errors = []

total = len(markets)

for number, row in enumerate(markets.itertuples(index=False), start=1):
    print(f"[{number}/{total}] {row.slug}")

    try:
        response = session.get(
            f"{GAMMA_URL}/markets/{row.market_id}",
            timeout=20,
        )
        response.raise_for_status()
        market = response.json()

        labels = parse_array(market.get("outcomes"))
        raw_prices = parse_array(market.get("outcomePrices"))
        token_ids = parse_array(market.get("clobTokenIds"))

        prices = []
        for value in raw_prices:
            try:
                prices.append(float(value))
            except (TypeError, ValueError):
                prices.append(None)

        outcome_data = []

        for index, label in enumerate(labels):
            outcome_data.append(
                {
                    "label": str(label),
                    "price": prices[index] if index < len(prices) else None,
                    "token_id": (
                        str(token_ids[index])
                        if index < len(token_ids)
                        else None
                    ),
                }
            )

        closed = market.get("closed") is True
        winner = None
        resolution_status = "unresolved"

        valid_prices = [
            outcome
            for outcome in outcome_data
            if outcome["price"] is not None
        ]

        terminal_winners = [
            outcome
            for outcome in valid_prices
            if outcome["price"] >= 0.99
        ]

        terminal_losers = [
            outcome
            for outcome in valid_prices
            if outcome["price"] <= 0.01
        ]

        if (
            closed
            and len(terminal_winners) == 1
            and len(terminal_losers) >= 1
        ):
            winner = terminal_winners[0]["label"]
            resolution_status = "resolved"
        elif closed:
            resolution_status = "closed_nonterminal"

        label_lookup = {
            outcome["label"].strip().lower(): outcome
            for outcome in outcome_data
        }

        up_outcome = label_lookup.get("up")
        down_outcome = label_lookup.get("down")

        up_api_token = (
            up_outcome["token_id"] if up_outcome else None
        )
        down_api_token = (
            down_outcome["token_id"] if down_outcome else None
        )

        token_ids_match = (
            up_api_token == str(row.up_token_id)
            and down_api_token == str(row.down_token_id)
        )

        results.append(
            {
                "slug": row.slug,
                "market_id": row.market_id,
                "question": row.question,
                "start_iso": row.start_iso,
                "end_iso": row.end_iso,
                "tick_count": row.tick_count,
                "max_seconds": row.max_seconds,
                "min_seconds": row.min_seconds,
                "coverage_complete": row.coverage_complete,
                "closed": closed,
                "resolution_status": resolution_status,
                "winner": winner,
                "outcome_labels": json.dumps(labels),
                "outcome_prices": json.dumps(raw_prices),
                "up_final_price": (
                    up_outcome["price"] if up_outcome else None
                ),
                "down_final_price": (
                    down_outcome["price"] if down_outcome else None
                ),
                "token_ids_match": token_ids_match,
                "uma_resolution_status": market.get(
                    "umaResolutionStatus"
                ),
                "closed_time": market.get("closedTime"),
                "automatically_resolved": market.get(
                    "automaticallyResolved"
                ),
            }
        )

    except Exception as error:
        print(f"  ERROR: {error}")

        errors.append(
            {
                "slug": row.slug,
                "market_id": row.market_id,
                "error": str(error),
            }
        )

    time.sleep(0.1)

outcomes = pd.DataFrame(results)
error_frame = pd.DataFrame(errors)

outcomes_path = OUTPUT_DIR / "market_outcomes.csv"
errors_path = OUTPUT_DIR / "outcome_fetch_errors.csv"

outcomes.to_csv(outcomes_path, index=False)
error_frame.to_csv(errors_path, index=False)

resolved = (
    int((outcomes["resolution_status"] == "resolved").sum())
    if not outcomes.empty
    else 0
)

closed_nonterminal = (
    int(
        (
            outcomes["resolution_status"]
            == "closed_nonterminal"
        ).sum()
    )
    if not outcomes.empty
    else 0
)

unresolved = (
    int((outcomes["resolution_status"] == "unresolved").sum())
    if not outcomes.empty
    else 0
)

up_wins = (
    int(outcomes["winner"].str.lower().eq("up").sum())
    if not outcomes.empty
    else 0
)

down_wins = (
    int(outcomes["winner"].str.lower().eq("down").sum())
    if not outcomes.empty
    else 0
)

token_mismatches = (
    int((~outcomes["token_ids_match"]).sum())
    if not outcomes.empty
    else 0
)

complete_and_resolved = (
    int(
        (
            outcomes["coverage_complete"]
            & outcomes["resolution_status"].eq("resolved")
        ).sum()
    )
    if not outcomes.empty
    else 0
)

print("\nOutcome-labeling results:")
print(f"Markets requested: {total}")
print(f"Successful API responses: {len(outcomes)}")
print(f"API errors: {len(error_frame)}")
print(f"Resolved markets: {resolved}")
print(f"Closed but nonterminal: {closed_nonterminal}")
print(f"Unresolved markets: {unresolved}")
print(f"Up wins: {up_wins}")
print(f"Down wins: {down_wins}")
print(f"Token-ID mismatches: {token_mismatches}")
print(f"Complete and resolved markets: {complete_and_resolved}")

print("\nSaved:")
print(outcomes_path)
print(errors_path)

if not outcomes.empty:
    print("\nLatest five labels:")
    print(
        outcomes[
            [
                "slug",
                "coverage_complete",
                "closed",
                "winner",
                "up_final_price",
                "down_final_price",
            ]
        ]
        .tail(5)
        .to_string(index=False)
    )