from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

INPUT_PATH = OUTPUT_DIR / "cross_market_disagreement_detail.csv"
SUMMARY_PATH = OUTPUT_DIR / "cost_sensitivity_summary.csv"
SURVIVORS_PATH = OUTPUT_DIR / "one_second_cost_survivors.csv"

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"{INPUT_PATH.name} was not found. "
        "Run the disagreement analysis first."
    )

data = pd.read_csv(INPUT_PATH)

required_columns = {
    "window_key",
    "horizon",
    "timestamp_gap_seconds",
    "best_raw_edge",
    "best_raw_direction",
    "best_complementary_cost",
    "kalshi_winner",
    "polymarket_winner",
}

missing = required_columns - set(data.columns)

if missing:
    raise ValueError(
        "Missing required columns: "
        + ", ".join(sorted(missing))
    )

numeric_columns = [
    "timestamp_gap_seconds",
    "best_raw_edge",
    "best_complementary_cost",
]

for column in numeric_columns:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce",
    )

data["outcomes_match"] = (
    data["kalshi_winner"]
    .astype(str)
    .str.lower()
    .eq(
        data["polymarket_winner"]
        .astype(str)
        .str.lower()
    )
)

eligible = data[
    data["outcomes_match"]
    & data["timestamp_gap_seconds"].notna()
    & data["best_raw_edge"].notna()
    & data["best_raw_edge"].gt(0)
].copy()

time_limits = [5, 2, 1]
cost_hurdles = [0.000, 0.005, 0.010, 0.020, 0.030, 0.050]

summary_rows = []

for time_limit in time_limits:
    timed = eligible[
        eligible["timestamp_gap_seconds"].le(time_limit)
    ].copy()

    for cost_hurdle in cost_hurdles:
        timed["net_edge_after_hurdle"] = (
            timed["best_raw_edge"] - cost_hurdle
        )

        survivors = timed[
            timed["net_edge_after_hurdle"].gt(0)
        ].copy()

        summary_rows.append(
            {
                "maximum_timestamp_gap_seconds": time_limit,
                "combined_cost_hurdle": cost_hurdle,
                "surviving_checkpoint_rows": len(survivors),
                "surviving_unique_windows": (
                    survivors["window_key"].nunique()
                ),
                "mean_gross_edge": (
                    survivors["best_raw_edge"].mean()
                    if not survivors.empty
                    else None
                ),
                "mean_net_edge_after_hurdle": (
                    survivors["net_edge_after_hurdle"].mean()
                    if not survivors.empty
                    else None
                ),
                "maximum_net_edge_after_hurdle": (
                    survivors["net_edge_after_hurdle"].max()
                    if not survivors.empty
                    else None
                ),
                "kalshi_up_poly_down_rows": (
                    survivors["best_raw_direction"]
                    .eq("Kalshi Up + Polymarket Down")
                    .sum()
                ),
                "poly_up_kalshi_down_rows": (
                    survivors["best_raw_direction"]
                    .eq("Polymarket Up + Kalshi Down")
                    .sum()
                ),
            }
        )

summary = pd.DataFrame(summary_rows)

one_second = eligible[
    eligible["timestamp_gap_seconds"].le(1)
].copy()

one_second["net_edge_after_2pct_hurdle"] = (
    one_second["best_raw_edge"] - 0.02
)

one_second_survivors = one_second[
    one_second["net_edge_after_2pct_hurdle"].gt(0)
].copy()

one_second_survivors = one_second_survivors.sort_values(
    [
        "net_edge_after_2pct_hurdle",
        "timestamp_gap_seconds",
    ],
    ascending=[False, True],
)

survivor_columns = [
    "window_key",
    "horizon",
    "timestamp_gap_seconds",
    "best_raw_direction",
    "best_complementary_cost",
    "best_raw_edge",
    "net_edge_after_2pct_hurdle",
    "kalshi_winner",
    "polymarket_winner",
]

one_second_survivors = one_second_survivors[
    survivor_columns
]

summary.to_csv(SUMMARY_PATH, index=False)
one_second_survivors.to_csv(
    SURVIVORS_PATH,
    index=False,
)

print("TRANSACTION-COST SENSITIVITY")
print(f"Eligible positive-edge rows: {len(eligible)}")

print("\nSensitivity results:")
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

print("\nStrict one-second results:")
print("Combined cost hurdle: 2%")
print(f"Checkpoint rows: {len(one_second_survivors)}")
print(
    "Unique windows: "
    f"{one_second_survivors['window_key'].nunique()}"
)

if one_second_survivors.empty:
    print("No rows survived.")
else:
    print(
        one_second_survivors.to_string(
            index=False,
            formatters={
                "timestamp_gap_seconds": "{:.3f}".format,
                "best_complementary_cost": "{:.3f}".format,
                "best_raw_edge": "{:.1%}".format,
                "net_edge_after_2pct_hurdle": "{:.1%}".format,
            },
        )
    )

print("\nSaved:")
print(SUMMARY_PATH)
print(SURVIVORS_PATH)

print(
    "\nThese are cost-hurdle scenarios, not official "
    "fee-adjusted profits or verified arbitrage."
)
