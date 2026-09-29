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

HORIZONS = {
    "15 minutes": 900,
    "10 minutes": 600,
    "5 minutes": 300,
    "2 minutes": 120,
    "1 minute": 60,
}

MAX_DISTANCE = 20


def select_checkpoints(
    frame,
    market_column,
    timestamp_column,
    seconds_column,
    probability_column,
    bid_column,
    ask_column,
    spread_column,
    prefix,
):
    selected = []

    usable = frame.dropna(
        subset=[
            "window_key",
            timestamp_column,
            seconds_column,
            probability_column,
        ]
    ).copy()

    for (market, window_key), group in usable.groupby(
        [market_column, "window_key"]
    ):
        for horizon, target_seconds in HORIZONS.items():
            distances = (
                group[seconds_column] - target_seconds
            ).abs()

            nearest_index = distances.idxmin()
            nearest = group.loc[nearest_index]
            distance = float(distances.loc[nearest_index])

            if distance > MAX_DISTANCE:
                continue

            selected.append(
                {
                    "window_key": window_key,
                    "horizon": horizon,
                    "target_seconds": target_seconds,
                    f"{prefix}_market": market,
                    f"{prefix}_timestamp": nearest[timestamp_column],
                    f"{prefix}_seconds_left": nearest[seconds_column],
                    f"{prefix}_distance": distance,
                    f"{prefix}_up_bid": nearest.get(bid_column),
                    f"{prefix}_up_ask": nearest.get(ask_column),
                    f"{prefix}_up_midpoint": nearest[
                        probability_column
                    ],
                    f"{prefix}_up_spread": nearest.get(
                        spread_column
                    ),
                }
            )

    return pd.DataFrame(selected)


print("Loading Kalshi...")
kalshi = pd.read_csv(KALSHI_PATH)

print("Loading Polymarket...")
with sqlite3.connect(POLY_PATH) as connection:
    poly_ticks = pd.read_sql_query(
        "SELECT * FROM ticks",
        connection,
    )

    poly_markets = pd.read_sql_query(
        """
        SELECT slug, start_iso, end_iso
        FROM markets
        """,
        connection,
    )


# Parse Kalshi times

kalshi["timestamp_utc"] = pd.to_datetime(
    kalshi["timestamp_utc"],
    utc=True,
    errors="coerce",
)

kalshi["window_end_utc"] = pd.to_datetime(
    kalshi["window_end_utc"],
    utc=True,
    errors="coerce",
)

kalshi["window_key"] = (
    kalshi["window_end_utc"].dt.floor("min")
)


# Parse Polymarket times

poly_ticks["ts_iso"] = pd.to_datetime(
    poly_ticks["ts_iso"],
    utc=True,
    errors="coerce",
)

poly_markets["end_iso"] = pd.to_datetime(
    poly_markets["end_iso"],
    utc=True,
    errors="coerce",
)

poly_ticks = poly_ticks.merge(
    poly_markets[["slug", "end_iso"]],
    on="slug",
    how="left",
)

poly_ticks["window_key"] = (
    poly_ticks["end_iso"].dt.floor("min")
)


# Select nearest valid checkpoint on each platform

kalshi_points = select_checkpoints(
    frame=kalshi,
    market_column="market_ticker",
    timestamp_column="timestamp_utc",
    seconds_column="seconds_left",
    probability_column="yes_midpoint",
    bid_column="yes_bid",
    ask_column="yes_ask",
    spread_column="yes_spread",
    prefix="kalshi",
)

poly_points = select_checkpoints(
    frame=poly_ticks,
    market_column="slug",
    timestamp_column="ts_iso",
    seconds_column="seconds_left",
    probability_column="up_midpoint",
    bid_column="up_bid",
    ask_column="up_ask",
    spread_column="up_spread",
    prefix="poly",
)


# Keep checkpoints available on both platforms

paired = kalshi_points.merge(
    poly_points,
    on=["window_key", "horizon", "target_seconds"],
    how="inner",
)

paired["timestamp_gap_seconds"] = (
    paired["kalshi_timestamp"]
    - paired["poly_timestamp"]
).abs().dt.total_seconds()

paired["seconds_left_gap"] = (
    paired["kalshi_seconds_left"]
    - paired["poly_seconds_left"]
).abs()

paired["midpoint_difference"] = (
    paired["kalshi_up_midpoint"]
    - paired["poly_up_midpoint"]
)

paired["absolute_midpoint_difference"] = (
    paired["midpoint_difference"].abs()
)

paired["kalshi_tighter_spread"] = (
    paired["kalshi_up_spread"]
    < paired["poly_up_spread"]
)

paired["poly_tighter_spread"] = (
    paired["poly_up_spread"]
    < paired["kalshi_up_spread"]
)


# Summarize each horizon

summary_rows = []

for horizon, target_seconds in HORIZONS.items():
    group = paired[
        paired["target_seconds"] == target_seconds
    ]

    if group.empty:
        summary_rows.append(
            {
                "horizon": horizon,
                "paired_windows": 0,
            }
        )
        continue

    correlation = np.nan

    if (
        len(group) >= 2
        and group["kalshi_up_midpoint"].nunique() > 1
        and group["poly_up_midpoint"].nunique() > 1
    ):
        correlation = group[
            ["kalshi_up_midpoint", "poly_up_midpoint"]
        ].corr().iloc[0, 1]

    summary_rows.append(
        {
            "horizon": horizon,
            "paired_windows": len(group),
            "mean_kalshi_up": group[
                "kalshi_up_midpoint"
            ].mean(),
            "mean_poly_up": group[
                "poly_up_midpoint"
            ].mean(),
            "mean_difference": group[
                "midpoint_difference"
            ].mean(),
            "median_absolute_difference": group[
                "absolute_midpoint_difference"
            ].median(),
            "mean_absolute_difference": group[
                "absolute_midpoint_difference"
            ].mean(),
            "maximum_absolute_difference": group[
                "absolute_midpoint_difference"
            ].max(),
            "price_correlation": correlation,
            "mean_timestamp_gap_seconds": group[
                "timestamp_gap_seconds"
            ].mean(),
            "mean_kalshi_spread": group[
                "kalshi_up_spread"
            ].mean(),
            "mean_poly_spread": group[
                "poly_up_spread"
            ].mean(),
        }
    )

summary = pd.DataFrame(summary_rows)


# Save generated reports

kalshi_points.to_csv(
    OUTPUT_DIR / "kalshi_checkpoints.csv",
    index=False,
)

poly_points.to_csv(
    OUTPUT_DIR / "polymarket_checkpoints.csv",
    index=False,
)

paired.to_csv(
    OUTPUT_DIR / "paired_cross_market_checkpoints.csv",
    index=False,
)

summary.to_csv(
    OUTPUT_DIR / "cross_market_checkpoint_summary.csv",
    index=False,
)


# Print results

print("\nCHECKPOINT AVAILABILITY")

for horizon in HORIZONS:
    kalshi_count = int(
        kalshi_points["horizon"].eq(horizon).sum()
    )
    poly_count = int(
        poly_points["horizon"].eq(horizon).sum()
    )
    paired_count = int(
        paired["horizon"].eq(horizon).sum()
    )

    print(
        f"{horizon}: "
        f"Kalshi={kalshi_count}, "
        f"Polymarket={poly_count}, "
        f"Paired={paired_count}"
    )

print("\nCROSS-MARKET SUMMARY")
print(
    summary.to_string(
        index=False,
        formatters={
            "mean_kalshi_up": "{:.3f}".format,
            "mean_poly_up": "{:.3f}".format,
            "mean_difference": "{:.3f}".format,
            "median_absolute_difference": "{:.3f}".format,
            "mean_absolute_difference": "{:.3f}".format,
            "maximum_absolute_difference": "{:.3f}".format,
            "price_correlation": "{:.3f}".format,
            "mean_timestamp_gap_seconds": "{:.2f}".format,
            "mean_kalshi_spread": "{:.3f}".format,
            "mean_poly_spread": "{:.3f}".format,
        },
    )
)

print("\nLARGEST MIDPOINT DISAGREEMENTS")
print(
    paired.nlargest(
        10,
        "absolute_midpoint_difference",
    )[
        [
            "window_key",
            "horizon",
            "kalshi_up_midpoint",
            "poly_up_midpoint",
            "midpoint_difference",
            "timestamp_gap_seconds",
            "kalshi_up_spread",
            "poly_up_spread",
        ]
    ].to_string(index=False)
)

print("\nSaved checkpoint reports in:")
print(OUTPUT_DIR)