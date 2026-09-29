from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

PAIRED_PATH = OUTPUT_DIR / "paired_cross_market_checkpoints.csv"
OUTCOMES_PATH = OUTPUT_DIR / "cross_market_outcomes.csv"

for path in [PAIRED_PATH, OUTCOMES_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

paired = pd.read_csv(PAIRED_PATH)
outcomes = pd.read_csv(OUTCOMES_PATH)


def to_bool(series):
    return series.astype(str).str.strip().str.lower().eq("true")


outcomes["both_resolved_bool"] = to_bool(outcomes["both_resolved"])
outcomes["outcomes_match_bool"] = to_bool(outcomes["outcomes_match"])

common_outcomes = outcomes[
    outcomes["both_resolved_bool"]
    & outcomes["outcomes_match_bool"]
].copy()

common_outcomes["actual_up"] = (
    common_outcomes["polymarket_winner"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("up")
    .astype(int)
)

data = paired.merge(
    common_outcomes[
        [
            "window_key",
            "kalshi_winner",
            "polymarket_winner",
            "actual_up",
        ]
    ],
    on="window_key",
    how="inner",
)

numeric_columns = [
    "target_seconds",
    "kalshi_up_bid",
    "kalshi_up_ask",
    "kalshi_up_midpoint",
    "kalshi_up_spread",
    "poly_up_bid",
    "poly_up_ask",
    "poly_up_midpoint",
    "poly_up_spread",
    "timestamp_gap_seconds",
]

for column in numeric_columns:
    data[column] = pd.to_numeric(data[column], errors="coerce")

required_price_columns = [
    "kalshi_up_bid",
    "kalshi_up_ask",
    "kalshi_up_midpoint",
    "kalshi_up_spread",
    "poly_up_bid",
    "poly_up_ask",
    "poly_up_midpoint",
    "poly_up_spread",
]

missing_required = data[required_price_columns].isna().any(axis=1)
excluded_missing_prices = int(missing_required.sum())
data = data.loc[~missing_required].copy()

data["midpoint_difference"] = (
    data["kalshi_up_midpoint"] - data["poly_up_midpoint"]
)

data["absolute_midpoint_difference"] = (
    data["midpoint_difference"].abs()
)

data["kalshi_absolute_error"] = (
    data["kalshi_up_midpoint"] - data["actual_up"]
).abs()

data["poly_absolute_error"] = (
    data["poly_up_midpoint"] - data["actual_up"]
).abs()

data["kalshi_squared_error"] = (
    data["kalshi_up_midpoint"] - data["actual_up"]
) ** 2

data["poly_squared_error"] = (
    data["poly_up_midpoint"] - data["actual_up"]
) ** 2

tolerance = 1e-12

data["closer_platform"] = np.select(
    [
        data["kalshi_absolute_error"]
        < data["poly_absolute_error"] - tolerance,
        data["poly_absolute_error"]
        < data["kalshi_absolute_error"] - tolerance,
    ],
    ["kalshi", "polymarket"],
    default="tie",
)

data["tighter_platform"] = np.select(
    [
        data["kalshi_up_spread"]
        < data["poly_up_spread"] - tolerance,
        data["poly_up_spread"]
        < data["kalshi_up_spread"] - tolerance,
    ],
    ["kalshi", "polymarket"],
    default="tie",
)

data["kalshi_direction"] = (
    data["kalshi_up_midpoint"] >= 0.5
).astype(int)

data["poly_direction"] = (
    data["poly_up_midpoint"] >= 0.5
).astype(int)

data["directional_disagreement"] = (
    data["kalshi_direction"] != data["poly_direction"]
)

data["kalshi_direction_correct"] = (
    data["kalshi_direction"] == data["actual_up"]
)

data["poly_direction_correct"] = (
    data["poly_direction"] == data["actual_up"]
)

data["kalshi_extremity"] = (
    data["kalshi_up_midpoint"] - 0.5
).abs()

data["poly_extremity"] = (
    data["poly_up_midpoint"] - 0.5
).abs()

data["more_extreme_platform"] = np.select(
    [
        data["kalshi_extremity"]
        > data["poly_extremity"] + tolerance,
        data["poly_extremity"]
        > data["kalshi_extremity"] + tolerance,
    ],
    ["kalshi", "polymarket"],
    default="tie",
)

data["more_extreme_was_closer"] = (
    (data["more_extreme_platform"] == "kalshi")
    & (data["closer_platform"] == "kalshi")
) | (
    (data["more_extreme_platform"] == "polymarket")
    & (data["closer_platform"] == "polymarket")
)

# Infer Down asks using binary-contract complementarity.
data["kalshi_down_ask"] = 1 - data["kalshi_up_bid"]
data["poly_down_ask"] = 1 - data["poly_up_bid"]

# Buy Kalshi Up and Polymarket Down.
data["kalshi_up_poly_down_cost"] = (
    data["kalshi_up_ask"] + data["poly_down_ask"]
)

data["kalshi_up_poly_down_raw_edge"] = (
    1 - data["kalshi_up_poly_down_cost"]
)

# Buy Polymarket Up and Kalshi Down.
data["poly_up_kalshi_down_cost"] = (
    data["poly_up_ask"] + data["kalshi_down_ask"]
)

data["poly_up_kalshi_down_raw_edge"] = (
    1 - data["poly_up_kalshi_down_cost"]
)

data["best_complementary_cost"] = data[
    [
        "kalshi_up_poly_down_cost",
        "poly_up_kalshi_down_cost",
    ]
].min(axis=1)

data["best_raw_edge"] = 1 - data["best_complementary_cost"]

data["best_raw_direction"] = np.where(
    data["kalshi_up_poly_down_cost"]
    <= data["poly_up_kalshi_down_cost"],
    "Kalshi Up + Polymarket Down",
    "Polymarket Up + Kalshi Down",
)

data["raw_opportunity"] = data["best_raw_edge"] > tolerance

horizon_order = [
    "15 minutes",
    "10 minutes",
    "5 minutes",
    "2 minutes",
    "1 minute",
]

summary_rows = []

for horizon in horizon_order:
    group = data[data["horizon"] == horizon].copy()

    if group.empty:
        continue

    non_tied_extreme = group[
        group["more_extreme_platform"] != "tie"
    ]

    summary_rows.append(
        {
            "horizon": horizon,
            "paired_rows": len(group),
            "unique_windows": group["window_key"].nunique(),
            "mean_signed_difference": (
                group["midpoint_difference"].mean()
            ),
            "mean_absolute_difference": (
                group["absolute_midpoint_difference"].mean()
            ),
            "median_absolute_difference": (
                group["absolute_midpoint_difference"].median()
            ),
            "p90_absolute_difference": (
                group["absolute_midpoint_difference"].quantile(0.90)
            ),
            "kalshi_mean_spread": group["kalshi_up_spread"].mean(),
            "polymarket_mean_spread": (
                group["poly_up_spread"].mean()
            ),
            "kalshi_median_spread": (
                group["kalshi_up_spread"].median()
            ),
            "polymarket_median_spread": (
                group["poly_up_spread"].median()
            ),
            "kalshi_tighter_rows": (
                group["tighter_platform"].eq("kalshi").sum()
            ),
            "polymarket_tighter_rows": (
                group["tighter_platform"].eq("polymarket").sum()
            ),
            "tied_spread_rows": (
                group["tighter_platform"].eq("tie").sum()
            ),
            "kalshi_closer_rows": (
                group["closer_platform"].eq("kalshi").sum()
            ),
            "polymarket_closer_rows": (
                group["closer_platform"].eq("polymarket").sum()
            ),
            "tied_accuracy_rows": (
                group["closer_platform"].eq("tie").sum()
            ),
            "directional_disagreements": (
                group["directional_disagreement"].sum()
            ),
            "difference_at_least_5pp": (
                group["absolute_midpoint_difference"] >= 0.05
            ).sum(),
            "difference_at_least_10pp": (
                group["absolute_midpoint_difference"] >= 0.10
            ).sum(),
            "difference_at_least_15pp": (
                group["absolute_midpoint_difference"] >= 0.15
            ).sum(),
            "raw_opportunity_rows": (
                group["raw_opportunity"].sum()
            ),
            "maximum_raw_edge": group["best_raw_edge"].max(),
            "mean_timestamp_gap_seconds": (
                group["timestamp_gap_seconds"].mean()
            ),
            "more_extreme_was_closer_rate": (
                non_tied_extreme["more_extreme_was_closer"].mean()
                if not non_tied_extreme.empty
                else np.nan
            ),
        }
    )

horizon_summary = pd.DataFrame(summary_rows)

threshold_rows = []

for threshold in [0.05, 0.10, 0.15]:
    subset = data[
        data["absolute_midpoint_difference"] >= threshold
    ].copy()

    threshold_rows.append(
        {
            "threshold": threshold,
            "checkpoint_rows": len(subset),
            "unique_windows": subset["window_key"].nunique(),
            "kalshi_closer_rows": (
                subset["closer_platform"].eq("kalshi").sum()
            ),
            "polymarket_closer_rows": (
                subset["closer_platform"].eq("polymarket").sum()
            ),
            "tied_rows": (
                subset["closer_platform"].eq("tie").sum()
            ),
            "directional_disagreements": (
                subset["directional_disagreement"].sum()
            ),
            "mean_absolute_difference": (
                subset["absolute_midpoint_difference"].mean()
                if not subset.empty
                else np.nan
            ),
        }
    )

threshold_summary = pd.DataFrame(threshold_rows)

raw_opportunities = data[data["raw_opportunity"]].copy()

data_path = OUTPUT_DIR / "cross_market_disagreement_detail.csv"
summary_path = OUTPUT_DIR / "cross_market_disagreement_metrics.csv"
threshold_path = OUTPUT_DIR / "cross_market_disagreement_thresholds.csv"
opportunity_path = OUTPUT_DIR / "cross_market_raw_opportunities.csv"

data.to_csv(data_path, index=False)
horizon_summary.to_csv(summary_path, index=False)
threshold_summary.to_csv(threshold_path, index=False)
raw_opportunities.to_csv(opportunity_path, index=False)

print("CROSS-MARKET DISAGREEMENT ANALYSIS")
print(f"Common-outcome checkpoint rows: {len(data)}")
print(f"Unique common-outcome windows: {data['window_key'].nunique()}")
print(f"Rows excluded for missing prices: {excluded_missing_prices}")
print(
    "Mean absolute midpoint difference: "
    f"{data['absolute_midpoint_difference'].mean():.4f}"
)
print(
    "Median absolute midpoint difference: "
    f"{data['absolute_midpoint_difference'].median():.4f}"
)
print(
    "Directional disagreement rows: "
    f"{int(data['directional_disagreement'].sum())}"
)
print(
    "Raw complementary-price opportunities: "
    f"{int(data['raw_opportunity'].sum())}"
)

print("\nRESULTS BY HORIZON")
print(
    horizon_summary.to_string(
        index=False,
        formatters={
            "mean_signed_difference": "{:+.4f}".format,
            "mean_absolute_difference": "{:.4f}".format,
            "median_absolute_difference": "{:.4f}".format,
            "p90_absolute_difference": "{:.4f}".format,
            "kalshi_mean_spread": "{:.4f}".format,
            "polymarket_mean_spread": "{:.4f}".format,
            "kalshi_median_spread": "{:.4f}".format,
            "polymarket_median_spread": "{:.4f}".format,
            "maximum_raw_edge": "{:.4f}".format,
            "mean_timestamp_gap_seconds": "{:.2f}".format,
            "more_extreme_was_closer_rate": "{:.1%}".format,
        },
    )
)

print("\nLARGE-DIFFERENCE RESULTS")
print(
    threshold_summary.to_string(
        index=False,
        formatters={
            "threshold": "{:.0%}".format,
            "mean_absolute_difference": "{:.4f}".format,
        },
    )
)

if raw_opportunities.empty:
    print("\nNo raw complementary ask sums below 1 were found.")
else:
    print("\nTOP RAW COMPLEMENTARY-PRICE OPPORTUNITIES")
    print(
        raw_opportunities[
            [
                "window_key",
                "horizon",
                "best_raw_direction",
                "best_complementary_cost",
                "best_raw_edge",
                "timestamp_gap_seconds",
            ]
        ]
        .sort_values("best_raw_edge", ascending=False)
        .head(15)
        .to_string(
            index=False,
            formatters={
                "best_complementary_cost": "{:.4f}".format,
                "best_raw_edge": "{:.4f}".format,
                "timestamp_gap_seconds": "{:.2f}".format,
            },
        )
    )

print("\nIMPORTANT:")
print(
    "Raw opportunities are not guaranteed arbitrage. "
    "They exclude fees, depth, latency, and settlement-source risk."
)

print("\nSaved:")
print(data_path)
print(summary_path)
print(threshold_path)
print(opportunity_path)