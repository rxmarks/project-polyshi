from pathlib import Path
from urllib.parse import quote
import json
import time

import pandas as pd
import requests


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

PAIRED_PATH = OUTPUT_DIR / "paired_cross_market_checkpoints.csv"
OUTCOMES_PATH = OUTPUT_DIR / "cross_market_outcomes.csv"
ERRORS_PATH = OUTPUT_DIR / "cross_market_outcome_errors.csv"

KALSHI_BASE = "https://external-api.kalshi.com/trade-api/v2"
POLYMARKET_BASE = "https://gamma-api.polymarket.com"

if not PAIRED_PATH.exists():
    raise FileNotFoundError(
        f"Missing {PAIRED_PATH}. "
        "Run build_cross_market_checkpoints.py first."
    )


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


def fetch_kalshi_market(session, ticker):
    encoded_ticker = quote(str(ticker), safe="")

    live_url = f"{KALSHI_BASE}/markets/{encoded_ticker}"
    response = session.get(live_url, timeout=30)

    if response.status_code == 200:
        return response.json()["market"], "live"

    historical_url = (
        f"{KALSHI_BASE}/historical/markets/{encoded_ticker}"
    )
    response = session.get(historical_url, timeout=30)
    response.raise_for_status()

    return response.json()["market"], "historical"


def fetch_polymarket_market(session, slug):
    encoded_slug = quote(str(slug), safe="")
    url = f"{POLYMARKET_BASE}/markets/slug/{encoded_slug}"

    response = session.get(url, timeout=30)
    response.raise_for_status()

    return response.json()


def determine_polymarket_winner(market):
    labels = parse_array(market.get("outcomes"))
    prices_raw = parse_array(market.get("outcomePrices"))

    outcomes = []

    for index, label in enumerate(labels):
        try:
            price = float(prices_raw[index])
        except (IndexError, TypeError, ValueError):
            price = None

        outcomes.append(
            {
                "label": str(label).strip(),
                "price": price,
            }
        )

    terminal_winners = [
        outcome
        for outcome in outcomes
        if outcome["price"] is not None
        and outcome["price"] >= 0.99
    ]

    terminal_losers = [
        outcome
        for outcome in outcomes
        if outcome["price"] is not None
        and outcome["price"] <= 0.01
    ]

    closed = market.get("closed") is True
    winner = None
    status = "unresolved"

    if (
        closed
        and len(terminal_winners) == 1
        and len(terminal_losers) >= 1
    ):
        winner = terminal_winners[0]["label"].lower()
        status = "resolved"
    elif closed:
        status = "closed_nonterminal"

    return {
        "winner": winner,
        "status": status,
        "closed": closed,
        "outcomes": labels,
        "prices": prices_raw,
    }


paired = pd.read_csv(PAIRED_PATH)

required_columns = {
    "window_key",
    "kalshi_market",
    "poly_market",
}

missing_columns = required_columns - set(paired.columns)

if missing_columns:
    raise ValueError(
        "Missing required columns from paired checkpoint file: "
        f"{sorted(missing_columns)}"
    )

pairs = (
    paired[
        [
            "window_key",
            "kalshi_market",
            "poly_market",
        ]
    ]
    .drop_duplicates()
    .sort_values("window_key")
    .reset_index(drop=True)
)

mapping_check = (
    pairs.groupby("window_key")
    .agg(
        kalshi_markets=("kalshi_market", "nunique"),
        polymarket_markets=("poly_market", "nunique"),
    )
)

invalid_mappings = mapping_check[
    (mapping_check["kalshi_markets"] != 1)
    | (mapping_check["polymarket_markets"] != 1)
]

if not invalid_mappings.empty:
    raise ValueError(
        "Some windows do not map one-to-one between platforms:\n"
        f"{invalid_mappings}"
    )

session = requests.Session()
session.headers.update(
    {
        "User-Agent": (
            "polyshi-cross-market-research/1.0"
        )
    }
)

records = []
errors = []

total = len(pairs)

for number, row in enumerate(
    pairs.itertuples(index=False),
    start=1,
):
    print(
        f"[{number}/{total}] "
        f"{row.window_key} | "
        f"{row.kalshi_market} | "
        f"{row.poly_market}"
    )

    record = {
        "window_key": row.window_key,
        "kalshi_ticker": row.kalshi_market,
        "polymarket_slug": row.poly_market,
    }

    try:
        kalshi_market, kalshi_api_source = (
            fetch_kalshi_market(
                session,
                row.kalshi_market,
            )
        )

        kalshi_result = str(
            kalshi_market.get("result") or ""
        ).strip().lower()

        if kalshi_result not in {"yes", "no"}:
            kalshi_result = None

        record.update(
            {
                "kalshi_api_source": kalshi_api_source,
                "kalshi_status": kalshi_market.get("status"),
                "kalshi_result": kalshi_result,
                "kalshi_winner": (
                    "up"
                    if kalshi_result == "yes"
                    else "down"
                    if kalshi_result == "no"
                    else None
                ),
                "kalshi_settlement_value": (
                    kalshi_market.get(
                        "settlement_value_dollars"
                    )
                    or kalshi_market.get(
                        "settlement_value"
                    )
                ),
                "kalshi_settlement_ts": (
                    kalshi_market.get("settlement_ts")
                ),
                "kalshi_resolved": (
                    kalshi_result in {"yes", "no"}
                ),
            }
        )

    except Exception as error:
        print(f"  KALSHI ERROR: {error}")

        errors.append(
            {
                "window_key": row.window_key,
                "platform": "Kalshi",
                "market": row.kalshi_market,
                "error": str(error),
            }
        )

        record.update(
            {
                "kalshi_api_source": None,
                "kalshi_status": None,
                "kalshi_result": None,
                "kalshi_winner": None,
                "kalshi_settlement_value": None,
                "kalshi_settlement_ts": None,
                "kalshi_resolved": False,
            }
        )

    try:
        polymarket_market = fetch_polymarket_market(
            session,
            row.poly_market,
        )

        polymarket_result = (
            determine_polymarket_winner(
                polymarket_market
            )
        )

        record.update(
            {
                "polymarket_closed": (
                    polymarket_result["closed"]
                ),
                "polymarket_status": (
                    polymarket_result["status"]
                ),
                "polymarket_winner": (
                    polymarket_result["winner"]
                ),
                "polymarket_resolved": (
                    polymarket_result["status"]
                    == "resolved"
                ),
                "polymarket_outcomes": json.dumps(
                    polymarket_result["outcomes"]
                ),
                "polymarket_final_prices": json.dumps(
                    polymarket_result["prices"]
                ),
                "polymarket_closed_time": (
                    polymarket_market.get("closedTime")
                ),
            }
        )

    except Exception as error:
        print(f"  POLYMARKET ERROR: {error}")

        errors.append(
            {
                "window_key": row.window_key,
                "platform": "Polymarket",
                "market": row.poly_market,
                "error": str(error),
            }
        )

        record.update(
            {
                "polymarket_closed": False,
                "polymarket_status": None,
                "polymarket_winner": None,
                "polymarket_resolved": False,
                "polymarket_outcomes": None,
                "polymarket_final_prices": None,
                "polymarket_closed_time": None,
            }
        )

    record["both_resolved"] = (
        record["kalshi_resolved"]
        and record["polymarket_resolved"]
    )

    if record["both_resolved"]:
        record["outcomes_match"] = (
            record["kalshi_winner"]
            == record["polymarket_winner"]
        )
    else:
        record["outcomes_match"] = None

    records.append(record)
    time.sleep(0.10)


outcomes = pd.DataFrame(records)

error_columns = [
    "window_key",
    "platform",
    "market",
    "error",
]

error_frame = pd.DataFrame(
    errors,
    columns=error_columns,
)

outcomes.to_csv(OUTCOMES_PATH, index=False)
error_frame.to_csv(ERRORS_PATH, index=False)

both_resolved = int(
    outcomes["both_resolved"].fillna(False).sum()
)

matching_outcomes = int(
    outcomes["outcomes_match"].fillna(False).sum()
)

disagreements = outcomes[
    outcomes["both_resolved"]
    & ~outcomes["outcomes_match"].fillna(False)
].copy()

kalshi_up = int(
    outcomes["kalshi_winner"].eq("up").sum()
)

kalshi_down = int(
    outcomes["kalshi_winner"].eq("down").sum()
)

polymarket_up = int(
    outcomes["polymarket_winner"].eq("up").sum()
)

polymarket_down = int(
    outcomes["polymarket_winner"].eq("down").sum()
)

print("\nCROSS-MARKET OUTCOME RESULTS")
print(f"Unique paired windows: {len(outcomes)}")
print(f"Both platforms resolved: {both_resolved}")
print(f"Matching outcomes: {matching_outcomes}")
print(f"Outcome disagreements: {len(disagreements)}")
print(f"API errors: {len(error_frame)}")

print("\nOUTCOME COUNTS")
print(f"Kalshi Up: {kalshi_up}")
print(f"Kalshi Down: {kalshi_down}")
print(f"Polymarket Up: {polymarket_up}")
print(f"Polymarket Down: {polymarket_down}")

if not disagreements.empty:
    print("\nOUTCOME DISAGREEMENTS")
    print(
        disagreements[
            [
                "window_key",
                "kalshi_ticker",
                "polymarket_slug",
                "kalshi_winner",
                "polymarket_winner",
            ]
        ].to_string(index=False)
    )
else:
    print("\nOUTCOME DISAGREEMENTS")
    print("None")

unresolved = outcomes[
    ~outcomes["both_resolved"]
]

if not unresolved.empty:
    print("\nUNRESOLVED WINDOWS")
    print(
        unresolved[
            [
                "window_key",
                "kalshi_ticker",
                "kalshi_status",
                "kalshi_result",
                "polymarket_status",
            ]
        ].to_string(index=False)
    )

print("\nSaved:")
print(OUTCOMES_PATH)
print(ERRORS_PATH)