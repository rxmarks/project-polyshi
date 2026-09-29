from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

CHECKPOINTS_PATH = (
    OUTPUT_DIR / "paired_cross_market_checkpoints.csv"
)
OUTCOMES_PATH = OUTPUT_DIR / "cross_market_outcomes.csv"

DETAIL_PATH = (
    OUTPUT_DIR / "cross_market_accuracy_detail.csv"
)
METRICS_PATH = (
    OUTPUT_DIR / "cross_market_accuracy_metrics.csv"
)
PAIRED_PATH = (
    OUTPUT_DIR / "cross_market_accuracy_paired.csv"
)

if not CHECKPOINTS_PATH.exists():
    raise FileNotFoundError(
        f"Missing {CHECKPOINTS_PATH}"
    )

if not OUTCOMES_PATH.exists():
    raise FileNotFoundError(
        f"Missing {OUTCOMES_PATH}"
    )


def as_bool(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )


def add_scores(frame, probability_column, outcome_column, prefix):
    probability = pd.to_numeric(
        frame[probability_column],
        errors="coerce",
    )

    actual = pd.to_numeric(
        frame[outcome_column],
        errors="coerce",
    )

    clipped = probability.clip(0.001, 0.999)

    frame[f"{prefix}_brier"] = (
        probability - actual
    ) ** 2

    frame[f"{prefix}_log_loss"] = -(
        actual * np.log(clipped)
        + (1 - actual) * np.log(1 - clipped)
    )

    frame[f"{prefix}_correct"] = (
        (probability >= 0.5)
        == actual.astype(bool)
    )

    return frame


checkpoints = pd.read_csv(CHECKPOINTS_PATH)
outcomes = pd.read_csv(OUTCOMES_PATH)

required_checkpoint_columns = {
    "window_key",
    "horizon",
    "target_seconds",
    "kalshi_up_midpoint",
    "poly_up_midpoint",
    "kalshi_up_spread",
    "poly_up_spread",
    "timestamp_gap_seconds",
}

required_outcome_columns = {
    "window_key",
    "kalshi_winner",
    "polymarket_winner",
    "both_resolved",
    "outcomes_match",
}

missing_checkpoints = (
    required_checkpoint_columns
    - set(checkpoints.columns)
)

missing_outcomes = (
    required_outcome_columns
    - set(outcomes.columns)
)

if missing_checkpoints:
    raise ValueError(
        "Missing checkpoint columns: "
        f"{sorted(missing_checkpoints)}"
    )

if missing_outcomes:
    raise ValueError(
        "Missing outcome columns: "
        f"{sorted(missing_outcomes)}"
    )

checkpoints["window_key"] = pd.to_datetime(
    checkpoints["window_key"],
    utc=True,
)

outcomes["window_key"] = pd.to_datetime(
    outcomes["window_key"],
    utc=True,
)

outcomes["both_resolved"] = as_bool(
    outcomes["both_resolved"]
)

outcomes["outcomes_match"] = as_bool(
    outcomes["outcomes_match"]
)

merged = checkpoints.merge(
    outcomes[
        [
            "window_key",
            "kalshi_winner",
            "polymarket_winner",
            "both_resolved",
            "outcomes_match",
        ]
    ],
    on="window_key",
    how="left",
    validate="many_to_one",
)

merged = merged[
    merged["both_resolved"].fillna(False)
].copy()

merged["kalshi_actual_up"] = (
    merged["kalshi_winner"]
    .str.lower()
    .eq("up")
    .astype(int)
)

merged["poly_actual_up"] = (
    merged["polymarket_winner"]
    .str.lower()
    .eq("up")
    .astype(int)
)

merged = add_scores(
    merged,
    "kalshi_up_midpoint",
    "kalshi_actual_up",
    "kalshi_contract",
)

merged = add_scores(
    merged,
    "poly_up_midpoint",
    "poly_actual_up",
    "poly_contract",
)

merged["common_outcome_eligible"] = (
    merged["outcomes_match"]
)

merged["common_actual_up"] = np.where(
    merged["common_outcome_eligible"],
    merged["kalshi_actual_up"],
    np.nan,
)

merged = add_scores(
    merged,
    "kalshi_up_midpoint",
    "common_actual_up",
    "kalshi_common",
)

merged = add_scores(
    merged,
    "poly_up_midpoint",
    "common_actual_up",
    "poly_common",
)

horizon_order = [
    "15 minutes",
    "10 minutes",
    "5 minutes",
    "2 minutes",
    "1 minute",
]

records = []

metric_definitions = [
    (
        "contract_specific",
        "Kalshi",
        "kalshi_contract",
        merged,
    ),
    (
        "contract_specific",
        "Polymarket",
        "poly_contract",
        merged,
    ),
    (
        "common_outcome",
        "Kalshi",
        "kalshi_common",
        merged[merged["common_outcome_eligible"]],
    ),
    (
        "common_outcome",
        "Polymarket",
        "poly_common",
        merged[merged["common_outcome_eligible"]],
    ),
]

for sample_type, platform, prefix, frame in metric_definitions:
    for horizon in horizon_order:
        group = frame[
            frame["horizon"].eq(horizon)
        ].copy()

        if group.empty:
            continue

        records.append(
            {
                "sample_type": sample_type,
                "platform": platform,
                "horizon": horizon,
                "checkpoint_rows": len(group),
                "unique_windows": (
                    group["window_key"].nunique()
                ),
                "brier_score": (
                    group[f"{prefix}_brier"].mean()
                ),
                "log_loss": (
                    group[f"{prefix}_log_loss"].mean()
                ),
                "directional_accuracy": (
                    group[f"{prefix}_correct"].mean()
                ),
                "mean_up_probability": (
                    group[
                        "kalshi_up_midpoint"
                        if platform == "Kalshi"
                        else "poly_up_midpoint"
                    ].mean()
                ),
            }
        )

metrics = pd.DataFrame(records)

common = merged[
    merged["common_outcome_eligible"]
].copy()

common["brier_difference"] = (
    common["kalshi_common_brier"]
    - common["poly_common_brier"]
)

common["log_loss_difference"] = (
    common["kalshi_common_log_loss"]
    - common["poly_common_log_loss"]
)

common["kalshi_better_brier"] = (
    common["brier_difference"] < 0
)

common["poly_better_brier"] = (
    common["brier_difference"] > 0
)

common["brier_tie"] = np.isclose(
    common["brier_difference"],
    0,
    atol=1e-12,
)

paired_records = []

for horizon in horizon_order:
    group = common[
        common["horizon"].eq(horizon)
    ].copy()

    if group.empty:
        continue

    paired_records.append(
        {
            "horizon": horizon,
            "paired_rows": len(group),
            "unique_windows": (
                group["window_key"].nunique()
            ),
            "kalshi_brier": (
                group["kalshi_common_brier"].mean()
            ),
            "polymarket_brier": (
                group["poly_common_brier"].mean()
            ),
            "kalshi_minus_polymarket_brier": (
                group["brier_difference"].mean()
            ),
            "kalshi_log_loss": (
                group["kalshi_common_log_loss"].mean()
            ),
            "polymarket_log_loss": (
                group["poly_common_log_loss"].mean()
            ),
            "kalshi_accuracy": (
                group["kalshi_common_correct"].mean()
            ),
            "polymarket_accuracy": (
                group["poly_common_correct"].mean()
            ),
            "kalshi_better_rows": int(
                group["kalshi_better_brier"].sum()
            ),
            "polymarket_better_rows": int(
                group["poly_better_brier"].sum()
            ),
            "tied_rows": int(
                group["brier_tie"].sum()
            ),
            "mean_timestamp_gap_seconds": (
                group["timestamp_gap_seconds"].mean()
            ),
        }
    )

paired_metrics = pd.DataFrame(paired_records)

merged.to_csv(DETAIL_PATH, index=False)
metrics.to_csv(METRICS_PATH, index=False)
paired_metrics.to_csv(PAIRED_PATH, index=False)

disagreement_rows = merged[
    ~merged["outcomes_match"]
].copy()

print("CROSS-MARKET ACCURACY ANALYSIS")
print(f"Resolved checkpoint rows: {len(merged)}")
print(
    "Resolved unique windows: "
    f"{merged['window_key'].nunique()}"
)
print(
    "Common-outcome checkpoint rows: "
    f"{len(common)}"
)
print(
    "Common-outcome unique windows: "
    f"{common['window_key'].nunique()}"
)
print(
    "Disagreement checkpoint rows excluded "
    f"from common comparison: {len(disagreement_rows)}"
)

print("\nCOMMON-OUTCOME PAIRED RESULTS")
print(
    paired_metrics.to_string(
        index=False,
        formatters={
            "kalshi_brier": "{:.4f}".format,
            "polymarket_brier": "{:.4f}".format,
            "kalshi_minus_polymarket_brier":
                "{:+.4f}".format,
            "kalshi_log_loss": "{:.4f}".format,
            "polymarket_log_loss": "{:.4f}".format,
            "kalshi_accuracy": "{:.1%}".format,
            "polymarket_accuracy": "{:.1%}".format,
            "mean_timestamp_gap_seconds":
                "{:.2f}".format,
        },
    )
)

overall_kalshi_brier = (
    common["kalshi_common_brier"].mean()
)

overall_poly_brier = (
    common["poly_common_brier"].mean()
)

overall_kalshi_log_loss = (
    common["kalshi_common_log_loss"].mean()
)

overall_poly_log_loss = (
    common["poly_common_log_loss"].mean()
)

overall_kalshi_accuracy = (
    common["kalshi_common_correct"].mean()
)

overall_poly_accuracy = (
    common["poly_common_correct"].mean()
)

print("\nOVERALL COMMON-OUTCOME RESULTS")
print(
    f"Kalshi Brier score: "
    f"{overall_kalshi_brier:.4f}"
)
print(
    f"Polymarket Brier score: "
    f"{overall_poly_brier:.4f}"
)
print(
    "Brier difference, Kalshi minus Polymarket: "
    f"{overall_kalshi_brier - overall_poly_brier:+.4f}"
)
print(
    f"Kalshi log loss: "
    f"{overall_kalshi_log_loss:.4f}"
)
print(
    f"Polymarket log loss: "
    f"{overall_poly_log_loss:.4f}"
)
print(
    f"Kalshi directional accuracy: "
    f"{overall_kalshi_accuracy:.1%}"
)
print(
    f"Polymarket directional accuracy: "
    f"{overall_poly_accuracy:.1%}"
)

print("\nINTERPRETATION")
print(
    "Negative Brier differences favor Kalshi; "
    "positive differences favor Polymarket."
)

print("\nSaved:")
print(DETAIL_PATH)
print(METRICS_PATH)
print(PAIRED_PATH)