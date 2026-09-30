from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "cross_market" / "raw"
OUTPUT_DIR = PROJECT_DIR / "outputs"

DETAIL_PATH = OUTPUT_DIR / "cross_market_disagreement_detail.csv"
ALL_OUTPUT_PATH = OUTPUT_DIR / "corrected_executable_edges.csv"
SUMMARY_PATH = OUTPUT_DIR / "corrected_cost_sensitivity.csv"
STRICT_PATH = OUTPUT_DIR / "corrected_strict_candidates.csv"

kalshi_files = list(RAW_DIR.rglob("*.csv"))
polymarket_files = list(RAW_DIR.rglob("*.db"))

if not DETAIL_PATH.exists():
    raise FileNotFoundError(f"Missing {DETAIL_PATH}")

if not kalshi_files:
    raise FileNotFoundError("No raw Kalshi CSV found.")

if not polymarket_files:
    raise FileNotFoundError("No raw Polymarket database found.")

kalshi_path = max(
    kalshi_files,
    key=lambda path: path.stat().st_mtime,
)

polymarket_path = max(
    polymarket_files,
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

detail["timestamp_gap_seconds"] = pd.to_numeric(
    detail["timestamp_gap_seconds"],
    errors="coerce",
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

kalshi = pd.read_csv(
    kalshi_path,
    usecols=[
        "timestamp_utc",
        "market_ticker",
        "yes_ask",
        "yes_ask_size",
        "no_ask",
        "no_ask_size",
        "volume",
        "open_interest",
    ],
)

kalshi["timestamp_utc"] = pd.to_datetime(
    kalshi["timestamp_utc"],
    utc=True,
)

kalshi = kalshi.rename(
    columns={
        "timestamp_utc": "kalshi_timestamp",
        "market_ticker": "kalshi_market",
        "yes_ask": "kalshi_yes_ask",
        "yes_ask_size": "kalshi_yes_ask_size",
        "no_ask": "kalshi_no_ask",
        "no_ask_size": "kalshi_no_ask_size",
        "volume": "kalshi_volume",
        "open_interest": "kalshi_open_interest",
    }
)

kalshi = kalshi.drop_duplicates(
    subset=["kalshi_market", "kalshi_timestamp"],
    keep="last",
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
        "up_ask": "poly_yes_ask",
        "up_ask_size": "poly_yes_ask_size",
        "down_ask": "poly_no_ask",
        "down_ask_size": "poly_no_ask_size",
    }
)

polymarket = polymarket.drop_duplicates(
    subset=["poly_market", "poly_timestamp"],
    keep="last",
)

result = detail.merge(
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

result["kalshi_up_poly_down_cost"] = (
    result["kalshi_yes_ask"] + result["poly_no_ask"]
)

result["poly_up_kalshi_down_cost"] = (
    result["poly_yes_ask"] + result["kalshi_no_ask"]
)

choose_kalshi_up = (
    result["kalshi_up_poly_down_cost"]
    <= result["poly_up_kalshi_down_cost"]
)

result["explicit_best_direction"] = np.where(
    choose_kalshi_up,
    "Kalshi Up + Polymarket Down",
    "Polymarket Up + Kalshi Down",
)

result["explicit_complementary_cost"] = np.where(
    choose_kalshi_up,
    result["kalshi_up_poly_down_cost"],
    result["poly_up_kalshi_down_cost"],
)

result["explicit_raw_edge"] = (
    1 - result["explicit_complementary_cost"]
)

result["leg_one_size"] = np.where(
    choose_kalshi_up,
    result["kalshi_yes_ask_size"],
    result["poly_yes_ask_size"],
)

result["leg_two_size"] = np.where(
    choose_kalshi_up,
    result["poly_no_ask_size"],
    result["kalshi_no_ask_size"],
)

required_columns = [
    "kalshi_yes_ask",
    "kalshi_no_ask",
    "kalshi_yes_ask_size",
    "kalshi_no_ask_size",
    "poly_yes_ask",
    "poly_no_ask",
    "poly_yes_ask_size",
    "poly_no_ask_size",
]

missing = [col for col in required_columns if col not in result.columns]
if missing:
    raise KeyError(f"Missing merged columns: {missing}")

result["quoted_executable_contracts"] = result[
    ["leg_one_size", "leg_two_size"]
].min(axis=1)

result["size_data_complete"] = (
    result["leg_one_size"].notna()
    & result["leg_two_size"].notna()
)

result["gross_profit_at_top_size"] = (
    result["explicit_raw_edge"]
    * result["quoted_executable_contracts"]
)

result["capital_required_at_top_size"] = (
    result["explicit_complementary_cost"]
    * result["quoted_executable_contracts"]
)

result["old_cost_difference"] = (
    result["explicit_complementary_cost"]
    - result["best_complementary_cost"]
)

result.to_csv(ALL_OUTPUT_PATH, index=False)

eligible = result[
    result["outcomes_match"]
    & result["timestamp_gap_seconds"].notna()
    & result["explicit_raw_edge"].notna()
    & result["explicit_raw_edge"].gt(0)
].copy()

time_limits = [5, 2, 1]
cost_hurdles = [0.000, 0.005, 0.010, 0.020, 0.030, 0.050]

summary_rows = []

for time_limit in time_limits:
    timed = eligible[
        eligible["timestamp_gap_seconds"].le(time_limit)
    ].copy()

    for cost_hurdle in cost_hurdles:
        survivors = timed[
            timed["explicit_raw_edge"].gt(cost_hurdle)
        ].copy()

        net_edges = (
            survivors["explicit_raw_edge"] - cost_hurdle
        )

        summary_rows.append(
            {
                "maximum_timestamp_gap_seconds": time_limit,
                "combined_cost_hurdle": cost_hurdle,
                "surviving_checkpoint_rows": len(survivors),
                "surviving_unique_windows": (
                    survivors["window_key"].nunique()
                ),
                "mean_gross_edge": (
                    survivors["explicit_raw_edge"].mean()
                    if not survivors.empty
                    else None
                ),
                "mean_net_edge_after_hurdle": (
                    net_edges.mean()
                    if not survivors.empty
                    else None
                ),
                "maximum_net_edge_after_hurdle": (
                    net_edges.max()
                    if not survivors.empty
                    else None
                ),
                "rows_with_complete_size": (
                    int(survivors["size_data_complete"].sum())
                ),
            }
        )

summary = pd.DataFrame(summary_rows)
summary.to_csv(SUMMARY_PATH, index=False)

strict = eligible[
    eligible["timestamp_gap_seconds"].le(1)
    & eligible["explicit_raw_edge"].gt(0.02)
].copy()

strict["net_edge_after_2pct_hurdle"] = (
    strict["explicit_raw_edge"] - 0.02
)

strict["net_profit_at_top_size_after_hurdle"] = (
    strict["net_edge_after_2pct_hurdle"]
    * strict["quoted_executable_contracts"]
)

strict = strict.sort_values(
    [
        "net_edge_after_2pct_hurdle",
        "timestamp_gap_seconds",
    ],
    ascending=[False, True],
)

strict_columns = [
    "window_key",
    "horizon",
    "timestamp_gap_seconds",
    "explicit_best_direction",
    "explicit_complementary_cost",
    "explicit_raw_edge",
    "net_edge_after_2pct_hurdle",
    "leg_one_size",
    "leg_two_size",
    "quoted_executable_contracts",
    "gross_profit_at_top_size",
    "net_profit_at_top_size_after_hurdle",
    "kalshi_winner",
    "polymarket_winner",
]

strict = strict[strict_columns]
strict.to_csv(STRICT_PATH, index=False)

print("\nCORRECTED EXECUTABLE-EDGE ANALYSIS")
print(f"All checkpoint rows: {len(result)}")
print(f"Positive-edge matching-outcome rows: {len(eligible)}")
print(
    "Rows with explicit asks available: "
    f"{int(result['explicit_complementary_cost'].notna().sum())}"
)

print("\nCorrected cost sensitivity:")
print(
    summary.to_string(
        index=False,
        formatters={
            "combined_cost_hurdle": "{:.1%}".format,
            "mean_gross_edge": lambda value: (
                "" if pd.isna(value) else f"{value:.1%}"
            ),
            "mean_net_edge_after_hurdle": lambda value: (
                "" if pd.isna(value) else f"{value:.1%}"
            ),
            "maximum_net_edge_after_hurdle": lambda value: (
                "" if pd.isna(value) else f"{value:.1%}"
            ),
        },
    )
)

print("\nCorrected strict candidates:")
print("Timestamp gap <= 1 second")
print("Explicit raw edge > 2%")
print(f"Checkpoint rows: {len(strict)}")
print(f"Unique windows: {strict['window_key'].nunique()}")

if strict.empty:
    print("No corrected strict candidates.")
else:
    print(
        strict.to_string(
            index=False,
            formatters={
                "timestamp_gap_seconds": "{:.3f}".format,
                "explicit_complementary_cost": "{:.3f}".format,
                "explicit_raw_edge": "{:.1%}".format,
                "net_edge_after_2pct_hurdle": "{:.1%}".format,
                "gross_profit_at_top_size": "{:.2f}".format,
                "net_profit_at_top_size_after_hurdle":
                    "{:.2f}".format,
            },
        )
    )

print("\nSaved:")
print(ALL_OUTPUT_PATH)
print(SUMMARY_PATH)
print(STRICT_PATH)

print(
    "\nThese use explicit recorded asks, but still do not "
    "include official fees or guarantee simultaneous fills."
)