from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = (
    PROJECT_DIR
    / "data"
    / "cross_market"
    / "raw"
    / "2026-09-29"
)
OUTPUT_DIR = PROJECT_DIR / "outputs"

KALSHI_PATH = (
    RAW_DIR / "kalshi_btc_15m_snapshots_2026-09-29.csv"
)
POLY_PATH = (
    RAW_DIR / "polymarket_btc15m_v2_2026-09-29.db"
)

OUTPUT_DIR.mkdir(exist_ok=True)


def count_invalid_range(frame, columns):
    counts = {}

    for column in columns:
        if column not in frame:
            continue

        values = pd.to_numeric(frame[column], errors="coerce")
        counts[column] = int(((values < 0) | (values > 1)).sum())

    return counts


def midpoint_errors(frame, bid, ask, midpoint):
    usable = frame[[bid, ask, midpoint]].dropna()

    if usable.empty:
        return 0

    calculated = (usable[bid] + usable[ask]) / 2

    return int(
        (~np.isclose(
            usable[midpoint],
            calculated,
            atol=0.000001,
        )).sum()
    )


print("Loading Kalshi CSV...")
kalshi = pd.read_csv(KALSHI_PATH)

print("Loading Polymarket database...")
with sqlite3.connect(POLY_PATH) as connection:
    integrity = connection.execute(
        "PRAGMA integrity_check"
    ).fetchone()[0]

    poly_markets = pd.read_sql_query(
        "SELECT * FROM markets",
        connection,
    )

    poly_ticks = pd.read_sql_query(
        "SELECT * FROM ticks",
        connection,
    )


# Parse timestamps

kalshi["timestamp_utc"] = pd.to_datetime(
    kalshi["timestamp_utc"],
    utc=True,
    errors="coerce",
)
kalshi["window_start_utc"] = pd.to_datetime(
    kalshi["window_start_utc"],
    utc=True,
    errors="coerce",
)
kalshi["window_end_utc"] = pd.to_datetime(
    kalshi["window_end_utc"],
    utc=True,
    errors="coerce",
)

poly_ticks["ts_iso"] = pd.to_datetime(
    poly_ticks["ts_iso"],
    utc=True,
    errors="coerce",
)
poly_markets["start_iso"] = pd.to_datetime(
    poly_markets["start_iso"],
    utc=True,
    errors="coerce",
)
poly_markets["end_iso"] = pd.to_datetime(
    poly_markets["end_iso"],
    utc=True,
    errors="coerce",
)


# Kalshi quality checks

kalshi_price_columns = [
    "yes_bid",
    "yes_ask",
    "yes_midpoint",
    "no_bid",
    "no_ask",
    "no_midpoint",
    "last_price",
]

kalshi_invalid_prices = count_invalid_range(
    kalshi,
    kalshi_price_columns,
)

kalshi_duplicate_rows = int(
    kalshi.duplicated(
        subset=["timestamp_utc", "market_ticker"]
    ).sum()
)

kalshi_crossed_yes = int(
    (
        kalshi["yes_bid"].notna()
        & kalshi["yes_ask"].notna()
        & (kalshi["yes_bid"] > kalshi["yes_ask"])
    ).sum()
)

kalshi_crossed_no = int(
    (
        kalshi["no_bid"].notna()
        & kalshi["no_ask"].notna()
        & (kalshi["no_bid"] > kalshi["no_ask"])
    ).sum()
)

kalshi_yes_midpoint_errors = midpoint_errors(
    kalshi,
    "yes_bid",
    "yes_ask",
    "yes_midpoint",
)

kalshi_no_midpoint_errors = midpoint_errors(
    kalshi,
    "no_bid",
    "no_ask",
    "no_midpoint",
)


# Polymarket quality checks

poly_price_columns = [
    "up_bid",
    "up_ask",
    "up_midpoint",
    "down_bid",
    "down_ask",
    "down_midpoint",
    "up_api_midpoint",
    "up_last_trade",
    "down_last_trade",
]

poly_invalid_prices = count_invalid_range(
    poly_ticks,
    poly_price_columns,
)

poly_duplicate_rows = int(
    poly_ticks.duplicated(
        subset=["ts_iso", "slug"]
    ).sum()
)

poly_crossed_up = int(
    (
        poly_ticks["up_bid"].notna()
        & poly_ticks["up_ask"].notna()
        & (poly_ticks["up_bid"] > poly_ticks["up_ask"])
    ).sum()
)

poly_crossed_down = int(
    (
        poly_ticks["down_bid"].notna()
        & poly_ticks["down_ask"].notna()
        & (poly_ticks["down_bid"] > poly_ticks["down_ask"])
    ).sum()
)

poly_up_midpoint_errors = midpoint_errors(
    poly_ticks,
    "up_bid",
    "up_ask",
    "up_midpoint",
)

poly_down_midpoint_errors = midpoint_errors(
    poly_ticks,
    "down_bid",
    "down_ask",
    "down_midpoint",
)


# Window coverage

kalshi_coverage = (
    kalshi.groupby(
        ["market_ticker", "window_start_utc", "window_end_utc"],
        dropna=False,
    )
    .agg(
        kalshi_ticks=("timestamp_utc", "size"),
        kalshi_first_observation=("timestamp_utc", "min"),
        kalshi_last_observation=("timestamp_utc", "max"),
        kalshi_max_seconds=("seconds_left", "max"),
        kalshi_min_seconds=("seconds_left", "min"),
        kalshi_quoted_rows=("yes_midpoint", "count"),
    )
    .reset_index()
)

kalshi_coverage["kalshi_complete"] = (
    (kalshi_coverage["kalshi_max_seconds"] >= 890)
    & (kalshi_coverage["kalshi_min_seconds"] <= 10)
)

poly_coverage = (
    poly_ticks.groupby("slug")
    .agg(
        poly_ticks=("ts_iso", "size"),
        poly_first_observation=("ts_iso", "min"),
        poly_last_observation=("ts_iso", "max"),
        poly_max_seconds=("seconds_left", "max"),
        poly_min_seconds=("seconds_left", "min"),
        poly_quoted_rows=("up_midpoint", "count"),
    )
    .reset_index()
)

poly_coverage = poly_coverage.merge(
    poly_markets[["slug", "start_iso", "end_iso"]],
    on="slug",
    how="left",
)

poly_coverage["poly_complete"] = (
    (poly_coverage["poly_max_seconds"] >= 890)
    & (poly_coverage["poly_min_seconds"] <= 10)
)


# Match platforms by UTC expiration window

kalshi_coverage["window_key"] = (
    kalshi_coverage["window_end_utc"].dt.floor("min")
)

poly_coverage["window_key"] = (
    poly_coverage["end_iso"].dt.floor("min")
)

matched = kalshi_coverage.merge(
    poly_coverage,
    on="window_key",
    how="outer",
    indicator=True,
)

conditions = [
    matched["_merge"].eq("both")
    & matched["kalshi_complete"].fillna(False)
    & matched["poly_complete"].fillna(False),

    matched["_merge"].eq("both"),

    matched["_merge"].eq("left_only"),

    matched["_merge"].eq("right_only"),
]

labels = [
    "matched_complete",
    "matched_partial",
    "kalshi_only",
    "polymarket_only",
]

matched["match_status"] = np.select(
    conditions,
    labels,
    default="unknown",
)


# Save reports

kalshi_coverage.to_csv(
    OUTPUT_DIR / "kalshi_cross_market_coverage.csv",
    index=False,
)

poly_coverage.to_csv(
    OUTPUT_DIR / "polymarket_cross_market_coverage.csv",
    index=False,
)

matched.to_csv(
    OUTPUT_DIR / "cross_market_windows.csv",
    index=False,
)

quality_rows = [
    {
        "platform": "Kalshi",
        "rows": len(kalshi),
        "markets": kalshi["market_ticker"].nunique(),
        "duplicate_timestamp_market_rows": kalshi_duplicate_rows,
        "crossed_primary_books": kalshi_crossed_yes,
        "crossed_secondary_books": kalshi_crossed_no,
        "primary_midpoint_errors": kalshi_yes_midpoint_errors,
        "secondary_midpoint_errors": kalshi_no_midpoint_errors,
        "invalid_price_values": sum(kalshi_invalid_prices.values()),
        "invalid_timestamps": int(
            kalshi["timestamp_utc"].isna().sum()
        ),
    },
    {
        "platform": "Polymarket",
        "rows": len(poly_ticks),
        "markets": poly_ticks["slug"].nunique(),
        "duplicate_timestamp_market_rows": poly_duplicate_rows,
        "crossed_primary_books": poly_crossed_up,
        "crossed_secondary_books": poly_crossed_down,
        "primary_midpoint_errors": poly_up_midpoint_errors,
        "secondary_midpoint_errors": poly_down_midpoint_errors,
        "invalid_price_values": sum(poly_invalid_prices.values()),
        "invalid_timestamps": int(
            poly_ticks["ts_iso"].isna().sum()
        ),
    },
]

quality = pd.DataFrame(quality_rows)

quality.to_csv(
    OUTPUT_DIR / "cross_market_quality.csv",
    index=False,
)


# Print summary

print("\nDATASET SUMMARY")
print(f"Polymarket integrity: {integrity}")
print(f"Kalshi rows: {len(kalshi):,}")
print(f"Kalshi markets: {kalshi['market_ticker'].nunique():,}")
print(f"Polymarket rows: {len(poly_ticks):,}")
print(f"Polymarket markets: {poly_ticks['slug'].nunique():,}")

print("\nQUALITY CHECKS")
print(quality.to_string(index=False))

print("\nMISSING QUOTES")
print(
    kalshi[
        [
            "yes_bid",
            "yes_ask",
            "yes_midpoint",
            "no_bid",
            "no_ask",
            "no_midpoint",
        ]
    ]
    .isna()
    .sum()
    .to_string()
)

print("\nWINDOW COVERAGE")
print(
    "Kalshi complete:",
    int(kalshi_coverage["kalshi_complete"].sum()),
    "of",
    len(kalshi_coverage),
)
print(
    "Polymarket complete:",
    int(poly_coverage["poly_complete"].sum()),
    "of",
    len(poly_coverage),
)

print("\nMATCH STATUS")
print(
    matched["match_status"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nSHARED PERIOD")
shared = matched[matched["_merge"].eq("both")]

if shared.empty:
    print("No matching UTC windows found.")
else:
    print("First:", shared["window_key"].min())
    print("Last:", shared["window_key"].max())
    print("Shared windows:", len(shared))

print("\nSaved validation reports in:")
print(OUTPUT_DIR)