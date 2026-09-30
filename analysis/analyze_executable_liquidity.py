from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "cross_market" / "raw"
OUTPUT_DIR = PROJECT_DIR / "outputs"

DETAIL_PATH = OUTPUT_DIR / "cross_market_disagreement_detail.csv"
OUTPUT_PATH = OUTPUT_DIR / "executable_liquidity_candidates.csv"

csv_files = list(RAW_DIR.rglob("*.csv"))
database_files = list(RAW_DIR.rglob("*.db"))

if not DETAIL_PATH.exists():
    raise FileNotFoundError(f"Missing {DETAIL_PATH}")

if not csv_files:
    raise FileNotFoundError("No raw Kalshi CSV found.")

if not database_files:
    raise FileNotFoundError("No raw Polymarket database found.")

kalshi_path = max(csv_files, key=lambda path: path.stat().st_mtime)
polymarket_path = max(
    database_files,
    key=lambda path: path.stat().st_mtime,
)

print(f"Kalshi file: {kalshi_path.name}")
print(f"Polymarket file: {polymarket_path.name}")

detail = pd.read_csv(DETAIL_PATH)

detail["kalshi_timestamp"] = pd.to_datetime(
    detail["kalshi_timestamp"],
    utc=True,
)

detail["poly_timestamp"] = pd.to_datetime(
    detail["poly_timestamp"],
    utc=True,
)

detail["outcomes_match"] = (
    detail["kalshi_winner"]
    .astype(str)
    .str.lower()
    .eq(
        detail["polymarket_winner"]
        .astype(str)
        .str.lower()
    )
)

candidates = detail[
    detail["outcomes_match"]
    & detail["timestamp_gap_seconds"].le(1)
    & detail["best_raw_edge"].gt(0.02)
].copy()

kalshi_columns = [
    "timestamp_utc",
    "market_ticker",
    "yes_ask",
    "yes_ask_size",
    "no_ask",
    "no_ask_size",
    "volume",
    "liquidity",
    "open_interest",
]

kalshi = pd.read_csv(
    kalshi_path,
    usecols=kalshi_columns,
)

kalshi["timestamp_utc"] = pd.to_datetime(
    kalshi["timestamp_utc"],
    utc=True,
)

kalshi = kalshi.rename(
    columns={
        "timestamp_utc": "kalshi_timestamp",
        "market_ticker": "kalshi_market",
        "yes_ask": "raw_kalshi_yes_ask",
        "yes_ask_size": "kalshi_yes_ask_size",
        "no_ask": "raw_kalshi_no_ask",
        "no_ask_size": "kalshi_no_ask_size",
        "volume": "kalshi_volume",
        "liquidity": "kalshi_liquidity",
        "open_interest": "kalshi_open_interest",
    }
)

with sqlite3.connect(polymarket_path) as connection:
    polymarket = pd.read_sql_query(
        """
        SELECT
            ts_iso,
            slug,
            up_ask,
            up_ask_size,
            down_ask,
            down_ask_size
        FROM ticks
        """,
        connection,
    )

polymarket["ts_iso"] = pd.to_datetime(
    polymarket["ts_iso"],
    utc=True,
)

polymarket = polymarket.rename(
    columns={
        "ts_iso": "poly_timestamp",
        "slug": "poly_market",
        "up_ask": "raw_poly_up_ask",
        "up_ask_size": "poly_up_ask_size",
        "down_ask": "raw_poly_down_ask",
        "down_ask_size": "poly_down_ask_size",
    }
)

result = candidates.merge(
    kalshi,
    on=["kalshi_market", "kalshi_timestamp"],
    how="left",
    validate="many_to_one",
)

result = result.merge(
    polymarket,
    on=["poly_market", "poly_timestamp"],
    how="left",
    validate="many_to_one",
)

kalshi_up_direction = result["best_raw_direction"].eq(
    "Kalshi Up + Polymarket Down"
)

poly_up_direction = result["best_raw_direction"].eq(
    "Polymarket Up + Kalshi Down"
)

result["leg_one_ask"] = np.where(
    kalshi_up_direction,
    result["raw_kalshi_yes_ask"],
    result["raw_poly_up_ask"],
)

result["leg_two_ask"] = np.where(
    kalshi_up_direction,
    result["raw_poly_down_ask"],
    result["raw_kalshi_no_ask"],
)

result["leg_one_size"] = np.where(
    kalshi_up_direction,
    result["kalshi_yes_ask_size"],
    result["poly_up_ask_size"],
)

result["leg_two_size"] = np.where(
    kalshi_up_direction,
    result["poly_down_ask_size"],
    result["kalshi_no_ask_size"],
)

result["recalculated_cost"] = (
    result["leg_one_ask"] + result["leg_two_ask"]
)

result["recalculated_raw_edge"] = (
    1 - result["recalculated_cost"]
)

result["quoted_executable_contracts"] = result[
    ["leg_one_size", "leg_two_size"]
].min(axis=1)

result["limiting_leg"] = np.select(
    [
        result["leg_one_size"] < result["leg_two_size"],
        result["leg_two_size"] < result["leg_one_size"],
    ],
    [
        "leg_one",
        "leg_two",
    ],
    default="equal",
)

result["gross_profit_at_top_size"] = (
    result["quoted_executable_contracts"]
    * result["recalculated_raw_edge"]
)

result["capital_required_at_top_size"] = (
    result["quoted_executable_contracts"]
    * result["recalculated_cost"]
)

result["gross_return_on_cost"] = (
    result["recalculated_raw_edge"]
    / result["recalculated_cost"]
)

result["cost_matches_previous"] = np.isclose(
    result["recalculated_cost"],
    result["best_complementary_cost"],
    atol=0.000001,
    equal_nan=False,
)

result["size_data_complete"] = (
    result["leg_one_size"].notna()
    & result["leg_two_size"].notna()
)

output_columns = [
    "window_key",
    "horizon",
    "timestamp_gap_seconds",
    "best_raw_direction",
    "leg_one_ask",
    "leg_two_ask",
    "recalculated_cost",
    "recalculated_raw_edge",
    "leg_one_size",
    "leg_two_size",
    "quoted_executable_contracts",
    "limiting_leg",
    "gross_profit_at_top_size",
    "capital_required_at_top_size",
    "gross_return_on_cost",
    "cost_matches_previous",
    "size_data_complete",
    "kalshi_volume",
    "kalshi_liquidity",
    "kalshi_open_interest",
    "kalshi_winner",
    "polymarket_winner",
]

result = result[output_columns].sort_values(
    "recalculated_raw_edge",
    ascending=False,
)

result.to_csv(OUTPUT_PATH, index=False)

print("\nTOP-OF-BOOK LIQUIDITY RESULTS")
print(f"Strict candidate rows: {len(candidates)}")
print(f"Rows matched to both raw datasets: {len(result)}")
print(
    "Rows with complete size data: "
    f"{int(result['size_data_complete'].sum())}"
)
print(
    "Rows matching previous cost: "
    f"{int(result['cost_matches_previous'].sum())}"
)

if result.empty:
    print("\nNo candidates met the strict criteria.")
else:
    display_columns = [
        "window_key",
        "horizon",
        "timestamp_gap_seconds",
        "best_raw_direction",
        "recalculated_raw_edge",
        "leg_one_size",
        "leg_two_size",
        "quoted_executable_contracts",
        "gross_profit_at_top_size",
    ]

    print("\nCandidate liquidity:")
    print(
        result[display_columns].to_string(
            index=False,
            formatters={
                "timestamp_gap_seconds": "{:.3f}".format,
                "recalculated_raw_edge": "{:.1%}".format,
                "gross_profit_at_top_size": "{:.2f}".format,
            },
        )
    )

print(f"\nSaved: {OUTPUT_PATH}")

print(
    "\nQuoted size does not guarantee execution. "
    "Results still exclude official fees, latency, "
    "partial fills, and settlement-source risk."
)