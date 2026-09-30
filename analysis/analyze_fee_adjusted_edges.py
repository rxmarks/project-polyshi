from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

INPUT_PATH = OUTPUT_DIR / "corrected_executable_edges.csv"
OUTPUT_PATH = OUTPUT_DIR / "fee_adjusted_executable_edges.csv"
STRICT_PATH = OUTPUT_DIR / "fee_adjusted_strict_candidates.csv"

EPS = 1e-9

# Confirm these rates against the applicable platform schedules
# before treating results as a live-trading estimate.
POLYMARKET_CRYPTO_TAKER_RATE = 0.07
KALSHI_TAKER_RATE = 0.07

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Missing {INPUT_PATH}")

data = pd.read_csv(INPUT_PATH)

required_columns = [
    "timestamp_gap_seconds",
    "outcomes_match",
    "explicit_best_direction",
    "explicit_complementary_cost",
    "explicit_raw_edge",
    "quoted_executable_contracts",
    "kalshi_yes_ask",
    "kalshi_no_ask",
    "poly_yes_ask",
    "poly_no_ask",
]

missing = [column for column in required_columns if column not in data.columns]

if missing:
    raise KeyError(f"Missing required columns: {missing}")

data["outcomes_match"] = (
    data["outcomes_match"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("true")
)

kalshi_up_poly_down = data["explicit_best_direction"].eq(
    "Kalshi Up + Polymarket Down"
)

data["kalshi_fill_price"] = np.where(
    kalshi_up_poly_down,
    data["kalshi_yes_ask"],
    data["kalshi_no_ask"],
)

data["polymarket_fill_price"] = np.where(
    kalshi_up_poly_down,
    data["poly_no_ask"],
    data["poly_yes_ask"],
)

data["kalshi_fee_per_contract"] = (
    KALSHI_TAKER_RATE
    * data["kalshi_fill_price"]
    * (1 - data["kalshi_fill_price"])
)

data["polymarket_fee_per_contract"] = (
    POLYMARKET_CRYPTO_TAKER_RATE
    * data["polymarket_fill_price"]
    * (1 - data["polymarket_fill_price"])
)

data["combined_fee_per_contract"] = (
    data["kalshi_fee_per_contract"]
    + data["polymarket_fee_per_contract"]
)

data["all_in_cost_per_contract"] = (
    data["explicit_complementary_cost"]
    + data["combined_fee_per_contract"]
)

data["fee_adjusted_edge_per_contract"] = (
    1 - data["all_in_cost_per_contract"]
)

data["gross_profit_at_top_size"] = (
    data["explicit_raw_edge"]
    * data["quoted_executable_contracts"]
)

data["kalshi_fee_at_top_size"] = (
    data["kalshi_fee_per_contract"]
    * data["quoted_executable_contracts"]
)

data["polymarket_fee_at_top_size"] = (
    data["polymarket_fee_per_contract"]
    * data["quoted_executable_contracts"]
)

data["combined_fee_at_top_size"] = (
    data["combined_fee_per_contract"]
    * data["quoted_executable_contracts"]
)

data["fee_adjusted_profit_at_top_size"] = (
    data["fee_adjusted_edge_per_contract"]
    * data["quoted_executable_contracts"]
)

data["fee_adjusted_roi_on_all_in_cost"] = np.where(
    data["all_in_cost_per_contract"].gt(0),
    (
        data["fee_adjusted_edge_per_contract"]
        / data["all_in_cost_per_contract"]
    ),
    np.nan,
)

data.to_csv(OUTPUT_PATH, index=False)

strict = data[
    data["outcomes_match"]
    & data["timestamp_gap_seconds"].le(1)
    & (
        data["fee_adjusted_edge_per_contract"]
        > EPS
    )
].copy()

strict = strict.sort_values(
    [
        "fee_adjusted_edge_per_contract",
        "timestamp_gap_seconds",
    ],
    ascending=[False, True],
)

strict_columns = [
    "window_key",
    "horizon",
    "timestamp_gap_seconds",
    "explicit_best_direction",
    "kalshi_fill_price",
    "polymarket_fill_price",
    "explicit_complementary_cost",
    "explicit_raw_edge",
    "kalshi_fee_per_contract",
    "polymarket_fee_per_contract",
    "combined_fee_per_contract",
    "all_in_cost_per_contract",
    "fee_adjusted_edge_per_contract",
    "quoted_executable_contracts",
    "gross_profit_at_top_size",
    "combined_fee_at_top_size",
    "fee_adjusted_profit_at_top_size",
    "fee_adjusted_roi_on_all_in_cost",
    "kalshi_winner",
    "polymarket_winner",
]

strict = strict[strict_columns]
strict.to_csv(STRICT_PATH, index=False)

print("\nFEE-ADJUSTED EXECUTION ANALYSIS")
print(f"All rows analyzed: {len(data)}")
print(
    "Assumed Kalshi taker rate: "
    f"{KALSHI_TAKER_RATE:.4f}"
)
print(
    "Assumed Polymarket crypto taker rate: "
    f"{POLYMARKET_CRYPTO_TAKER_RATE:.4f}"
)

print("\nStrict filter:")
print("Matching outcomes")
print("Timestamp gap <= 1 second")
print("Fee-adjusted edge > 0")
print(f"Surviving rows: {len(strict)}")
print(f"Unique windows: {strict['window_key'].nunique()}")

if strict.empty:
    print("\nNo fee-adjusted strict candidates.")
else:
    print("\nFee-adjusted strict candidates:")
    print(
        strict.to_string(
            index=False,
            formatters={
                "timestamp_gap_seconds": "{:.3f}".format,
                "kalshi_fill_price": "{:.3f}".format,
                "polymarket_fill_price": "{:.3f}".format,
                "explicit_complementary_cost": "{:.3f}".format,
                "explicit_raw_edge": "{:.2%}".format,
                "kalshi_fee_per_contract": "{:.5f}".format,
                "polymarket_fee_per_contract": "{:.5f}".format,
                "combined_fee_per_contract": "{:.5f}".format,
                "all_in_cost_per_contract": "{:.5f}".format,
                "fee_adjusted_edge_per_contract": "{:.2%}".format,
                "gross_profit_at_top_size": "${:,.2f}".format,
                "combined_fee_at_top_size": "${:,.2f}".format,
                "fee_adjusted_profit_at_top_size": "${:,.2f}".format,
                "fee_adjusted_roi_on_all_in_cost": "{:.2%}".format,
            },
        )
    )

print("\nSaved:")
print(OUTPUT_PATH)
print(STRICT_PATH)

print(
    "\nImportant: results assume immediate taker fills at the "
    "recorded asks. They exclude stale-quote risk, latency, "
    "partial fills, cancellation, withdrawal/funding costs, "
    "and any settlement-rule mismatch."
)