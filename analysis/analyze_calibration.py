from pathlib import Path
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

database_files = list(DATA_DIR.glob("*.db"))

if not database_files:
    raise FileNotFoundError(f"No .db file found in {DATA_DIR}")

db_path = max(database_files, key=lambda path: path.stat().st_mtime)
outcomes_path = OUTPUT_DIR / "market_outcomes.csv"

if not outcomes_path.exists():
    raise FileNotFoundError(
        "market_outcomes.csv was not found. "
        "Run analysis\\label_outcomes.py first."
    )

outcomes = pd.read_csv(outcomes_path)

coverage_complete = (
    outcomes["coverage_complete"]
    .astype(str)
    .str.lower()
    .eq("true")
)

eligible = outcomes[
    coverage_complete
    & outcomes["resolution_status"].eq("resolved")
].copy()

eligible["actual_up"] = (
    eligible["winner"].str.lower().eq("up").astype(int)
)

with sqlite3.connect(db_path) as connection:
    ticks = pd.read_sql_query(
        """
        SELECT
            slug,
            ts_iso,
            seconds_left,
            up_price,
            down_price,
            btc_spot
        FROM ticks
        """,
        connection,
    )

ticks = ticks.merge(
    eligible[["slug", "winner", "actual_up"]],
    on="slug",
    how="inner",
)

targets = {
    "15 minutes": 900,
    "10 minutes": 600,
    "5 minutes": 300,
    "2 minutes": 120,
    "1 minute": 60,
}

selected_rows = []

for slug, market_ticks in ticks.groupby("slug"):
    market_ticks = market_ticks.copy()

    for horizon, target_seconds in targets.items():
        distances = (
            market_ticks["seconds_left"] - target_seconds
        ).abs()

        nearest_index = distances.idxmin()
        nearest = market_ticks.loc[nearest_index]
        distance = abs(nearest["seconds_left"] - target_seconds)

        if distance <= 20:
            selected_rows.append(
                {
                    "slug": slug,
                    "horizon": horizon,
                    "target_seconds": target_seconds,
                    "actual_seconds": nearest["seconds_left"],
                    "up_probability": nearest["up_price"],
                    "down_probability": nearest["down_price"],
                    "actual_up": nearest["actual_up"],
                    "winner": nearest["winner"],
                    "btc_spot": nearest["btc_spot"],
                    "ts_iso": nearest["ts_iso"],
                }
            )

sample = pd.DataFrame(selected_rows)

sample["squared_error"] = (
    sample["up_probability"] - sample["actual_up"]
) ** 2

sample["correct_prediction"] = (
    (sample["up_probability"] >= 0.5)
    == sample["actual_up"].astype(bool)
)

clipped_probability = sample["up_probability"].clip(
    lower=0.001,
    upper=0.999,
)

sample["log_loss"] = -(
    sample["actual_up"] * np.log(clipped_probability)
    + (1 - sample["actual_up"])
    * np.log(1 - clipped_probability)
)

horizon_order = list(targets.keys())

horizon_metrics = (
    sample.groupby("horizon")
    .agg(
        forecasts=("slug", "count"),
        markets=("slug", "nunique"),
        mean_up_probability=("up_probability", "mean"),
        actual_up_rate=("actual_up", "mean"),
        brier_score=("squared_error", "mean"),
        log_loss=("log_loss", "mean"),
        accuracy=("correct_prediction", "mean"),
    )
    .reindex(horizon_order)
    .reset_index()
)

bin_edges = np.linspace(0, 1, 11)

sample["probability_bin"] = pd.cut(
    sample["up_probability"],
    bins=bin_edges,
    include_lowest=True,
    labels=False,
)

calibration = (
    sample.groupby("probability_bin", observed=True)
    .agg(
        forecast_count=("slug", "count"),
        market_count=("slug", "nunique"),
        mean_predicted_probability=("up_probability", "mean"),
        observed_up_rate=("actual_up", "mean"),
        brier_score=("squared_error", "mean"),
    )
    .reset_index()
)

calibration["bin_lower"] = calibration["probability_bin"] / 10
calibration["bin_upper"] = (
    calibration["probability_bin"] + 1
) / 10

overall_brier = sample["squared_error"].mean()
overall_accuracy = sample["correct_prediction"].mean()
overall_log_loss = sample["log_loss"].mean()

sample.to_csv(
    OUTPUT_DIR / "calibration_sample.csv",
    index=False,
)

horizon_metrics.to_csv(
    OUTPUT_DIR / "horizon_metrics.csv",
    index=False,
)

calibration.to_csv(
    OUTPUT_DIR / "calibration_bins.csv",
    index=False,
)

fig, ax = plt.subplots(figsize=(8, 7))

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    label="Perfect calibration",
)

ax.plot(
    calibration["mean_predicted_probability"],
    calibration["observed_up_rate"],
    marker="o",
    linewidth=2,
    color="#2563eb",
    label="Polymarket observations",
)

for row in calibration.itertuples(index=False):
    ax.annotate(
        f"n={row.forecast_count}",
        (
            row.mean_predicted_probability,
            row.observed_up_rate,
        ),
        xytext=(5, 6),
        textcoords="offset points",
        fontsize=8,
    )

ax.set(
    title="Polymarket BTC 15-Minute Calibration",
    xlabel="Mean predicted probability of Up",
    ylabel="Observed frequency of Up",
    xlim=(0, 1),
    ylim=(0, 1),
)

ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()

chart_path = OUTPUT_DIR / "calibration_curve.png"
fig.savefig(chart_path, dpi=200)
plt.close(fig)

print(f"Database: {db_path.name}")
print(f"Eligible complete markets: {eligible['slug'].nunique()}")
print(f"Forecast observations: {len(sample)}")
print(f"Overall Brier score: {overall_brier:.4f}")
print(f"Overall log loss: {overall_log_loss:.4f}")
print(f"Directional accuracy: {overall_accuracy:.1%}")

print("\nMetrics by time remaining:")
print(
    horizon_metrics.to_string(
        index=False,
        formatters={
            "mean_up_probability": "{:.3f}".format,
            "actual_up_rate": "{:.3f}".format,
            "brier_score": "{:.4f}".format,
            "log_loss": "{:.4f}".format,
            "accuracy": "{:.1%}".format,
        },
    )
)

print("\nCalibration bins:")
print(
    calibration[
        [
            "bin_lower",
            "bin_upper",
            "forecast_count",
            "mean_predicted_probability",
            "observed_up_rate",
        ]
    ].to_string(
        index=False,
        formatters={
            "mean_predicted_probability": "{:.3f}".format,
            "observed_up_rate": "{:.3f}".format,
        },
    )
)

print("\nSaved:")
print(OUTPUT_DIR / "calibration_sample.csv")
print(OUTPUT_DIR / "horizon_metrics.csv")
print(OUTPUT_DIR / "calibration_bins.csv")
print(chart_path)