from pathlib import Path
import sqlite3
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

database_files = list(DATA_DIR.glob("*.db"))

if not database_files:
    raise FileNotFoundError(
        f"No .db file found in {DATA_DIR}. "
        "Move the copied database into the data folder."
    )

db_path = max(database_files, key=lambda path: path.stat().st_mtime)

print(f"Database: {db_path.name}")

with sqlite3.connect(db_path) as connection:
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]

    market_count = connection.execute(
        "SELECT COUNT(*) FROM markets"
    ).fetchone()[0]

    tick_count = connection.execute(
        "SELECT COUNT(*) FROM ticks"
    ).fetchone()[0]

    coverage = pd.read_sql_query(
        """
        SELECT
            slug,
            COUNT(*) AS tick_count,
            MAX(seconds_left) AS max_seconds,
            MIN(seconds_left) AS min_seconds,
            MIN(ts_iso) AS first_observation,
            MAX(ts_iso) AS last_observation
        FROM ticks
        GROUP BY slug
        ORDER BY first_observation
        """,
        connection,
    )

    invalid_values = pd.read_sql_query(
        """
        SELECT
            SUM(CASE WHEN ts_iso IS NULL THEN 1 ELSE 0 END) AS missing_timestamp,
            SUM(CASE WHEN slug IS NULL THEN 1 ELSE 0 END) AS missing_slug,
            SUM(CASE WHEN seconds_left IS NULL THEN 1 ELSE 0 END) AS missing_seconds,
            SUM(CASE WHEN up_price IS NULL THEN 1 ELSE 0 END) AS missing_up_price,
            SUM(CASE WHEN down_price IS NULL THEN 1 ELSE 0 END) AS missing_down_price,
            SUM(CASE WHEN btc_spot IS NULL THEN 1 ELSE 0 END) AS missing_btc_spot,
            SUM(CASE WHEN up_price < 0 OR up_price > 1 THEN 1 ELSE 0 END)
                AS invalid_up_price,
            SUM(CASE WHEN down_price < 0 OR down_price > 1 THEN 1 ELSE 0 END)
                AS invalid_down_price
        FROM ticks
        """,
        connection,
    )

coverage["complete"] = (
    (coverage["max_seconds"] >= 890)
    & (coverage["min_seconds"] <= 10)
)

complete_count = int(coverage["complete"].sum())
partial_count = int((~coverage["complete"]).sum())
completion_rate = complete_count / len(coverage) * 100

coverage.to_csv(OUTPUT_DIR / "market_coverage.csv", index=False)
invalid_values.to_csv(OUTPUT_DIR / "data_quality.csv", index=False)

print(f"Integrity check: {integrity}")
print(f"Markets table rows: {market_count:,}")
print(f"Tick rows: {tick_count:,}")
print(f"Observed markets: {len(coverage):,}")
print(f"Complete markets: {complete_count:,}")
print(f"Partial markets: {partial_count:,}")
print(f"Completion rate: {completion_rate:.1f}%")
print(f"Average ticks per market: {coverage['tick_count'].mean():.1f}")

print("\nPartial markets:")
if partial_count == 0:
    print("None")
else:
    print(
        coverage.loc[
            ~coverage["complete"],
            ["slug", "tick_count", "max_seconds", "min_seconds"],
        ].to_string(index=False)
    )

print("\nData-quality checks:")
print(invalid_values.to_string(index=False))

print("\nSaved:")
print(OUTPUT_DIR / "market_coverage.csv")
print(OUTPUT_DIR / "data_quality.csv")