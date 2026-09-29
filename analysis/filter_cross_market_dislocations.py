from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

INPUT_PATH = OUTPUT_DIR / "cross_market_disagreement_detail.csv"
SUMMARY_PATH = OUTPUT_DIR / "strict_dislocation_summary.csv"
CANDIDATES_PATH = OUTPUT_DIR / "strict_dislocation_candidates.csv"

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"{INPUT_PATH.name} was not found. "
        "Run the cross-market disagreement analysis first."
    )

data = pd.read_csv(INPUT_PATH)

required_columns = {
    "window_key",
    "horizon",
    "timestamp_gap_seconds",
    "kalshi_winner",
    "polymarket_winner",
    "best_raw_edge",
    "best_raw_direction",
    "best_complementary_cost",
    "kalshi_up_bid",
    "kalshi_up_ask",
    "poly_up_bid",
    "poly_up_ask",
}

missing_columns = required_columns - set(data.columns)

if missing_columns:
    raise ValueError(
        "Missing required columns: "
        + ", ".join(sorted(missing_columns))
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

data["best_raw_edge"] = pd.to_numeric(
    data["best_raw_edge"],
    errors="coerce",
)

data["timestamp_gap_seconds"] = pd.to_numeric(
    data["timestamp_gap_seconds"],
    errors="coerce",
)

eligible = data[
    data["outcomes_match"]
    & data["best_raw_edge"].notna()
    & data["timestamp_gap_seconds"].notna()
    & data["best_raw_edge"].gt(0)
].copy()

time_limits = [5, 2, 1]
edge_thresholds = [0.01, 0.02, 0.05, 0.10]

summary_rows = []

for time_limit in time_limits:
    for edge_threshold in edge_thresholds:
        filtered = eligible[
            eligible["timestamp_gap_seconds"].le(time_limit)
            & eligible["best_raw_edge"].ge(edge_threshold)
        ]

        summary_rows.append(
            {
                "maximum_timestamp_gap_seconds": time_limit,
                "minimum_raw_edge": edge_threshold,
                "checkpoint_rows": len(filtered),
                "unique_windows": filtered["window_key"].nunique(),
                "mean_raw_edge": (
                    filtered["best_raw_edge"].mean()
                    if not filtered.empty
                    else None
                ),
                "median_raw_edge": (
                    filtered["best_raw_edge"].median()
                    if not filtered.empty
                    else None
                ),
                "maximum_raw_edge": (
                    filtered["best_raw_edge"].max()
                    if not filtered.empty
                    else None
                ),
                "kalshi_up_poly_down_rows": (
                    filtered["best_raw_direction"]
                    .eq("Kalshi Up + Polymarket Down")
                    .sum()
                ),
                "poly_up_kalshi_down_rows": (
                    filtered["best_raw_direction"]
                    .eq("Polymarket Up + Kalshi Down")
                    .sum()
                ),
            }
        )

summary = pd.DataFrame(summary_rows)

strict_candidates = eligible[
    eligible["timestamp_gap_seconds"].le(2)
    & eligible["best_raw_edge"].ge(0.02)
].copy()

strict_candidates = strict_candidates.sort_values(
    ["best_raw_edge", "timestamp_gap_seconds"],
    ascending=[False, True],
)

candidate_columns = [
    "window_key",
    "horizon",
    "timestamp_gap_seconds",
    "kalshi_timestamp",
    "poly_timestamp",
    "kalshi_up_bid",
    "kalshi_up_ask",
    "poly_up_bid",
    "poly_up_ask",
    "best_raw_direction",
    "best_complementary_cost",
    "best_raw_edge",
    "kalshi_winner",
    "polymarket_winner",
]

strict_candidates = strict_candidates[candidate_columns]

summary.to_csv(SUMMARY_PATH, index=False)
strict_candidates.to_csv(CANDIDATES_PATH, index=False)

print("STRICT CROSS-MARKET DISLOCATION FILTER")
print(f"Input rows: {len(data)}")
print(f"Matching-outcome positive-edge rows: {len(eligible)}")

print("\nSensitivity analysis:")
print(
    summary.to_string(
        index=False,
        formatters={
            "minimum_raw_edge": "{:.1%}".format,
            "mean_raw_edge": lambda value: (
                "" if pd.isna(value) else f"{value:.1%}"
            ),
            "median_raw_edge": lambda value: (
                "" if pd.isna(value) else f"{value:.1%}"
            ),
            "maximum_raw_edge": lambda value: (
                "" if pd.isna(value) else f"{value:.1%}"
            ),
        },
    )
)

print("\nStrict candidates:")
print("Timestamp gap <= 2 seconds and raw edge >= 2%")
print(f"Checkpoint rows: {len(strict_candidates)}")
print(
    f"Unique windows: "
    f"{strict_candidates['window_key'].nunique()}"
)

if strict_candidates.empty:
    print("No strict candidates found.")
else:
    print(
        strict_candidates[
            [
                "window_key",
                "horizon",
                "timestamp_gap_seconds",
                "best_raw_direction",
                "best_raw_edge",
            ]
        ].to_string(
            index=False,
            formatters={
                "timestamp_gap_seconds": "{:.3f}".format,
                "best_raw_edge": "{:.1%}".format,
            },
        )
    )

print("\nSaved:")
print(SUMMARY_PATH)
print(CANDIDATES_PATH)

print(
    "\nImportant: these remain gross historical "
    "dislocations, not verified arbitrage."
)