from pathlib import Path
import sqlite3

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = (
    PROJECT_DIR
    / "data"
    / "cross_market"
    / "raw"
    / "2026-09-29"
)

KALSHI_PATH = (
    RAW_DIR
    / "kalshi_btc_15m_snapshots_2026-09-29.csv"
)

POLYMARKET_PATH = (
    RAW_DIR
    / "polymarket_btc15m_v2_2026-09-29.db"
)


def inspect_kalshi():
    print("=" * 70)
    print("KALSHI CSV")
    print("=" * 70)
    print(f"Path: {KALSHI_PATH}")

    if not KALSHI_PATH.exists():
        raise FileNotFoundError(
            f"Kalshi file not found:\n{KALSHI_PATH}"
        )

    data = pd.read_csv(KALSHI_PATH)

    print(f"Rows: {len(data):,}")
    print(f"Columns: {len(data.columns):,}")
    print("\nColumn names:")

    for column in data.columns:
        print(f"- {column}: {data[column].dtype}")

    print("\nFirst three rows:")
    print(data.head(3).to_string(index=False))

    print("\nLast three rows:")
    print(data.tail(3).to_string(index=False))

    print("\nMissing values:")
    print(data.isna().sum().to_string())


def inspect_polymarket():
    print("\n" + "=" * 70)
    print("POLYMARKET V2 DATABASE")
    print("=" * 70)
    print(f"Path: {POLYMARKET_PATH}")

    if not POLYMARKET_PATH.exists():
        raise FileNotFoundError(
            f"Polymarket file not found:\n{POLYMARKET_PATH}"
        )

    with sqlite3.connect(POLYMARKET_PATH) as connection:
        integrity = connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        tables = [
            row[0]
            for row in connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                  AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            )
        ]

        print(f"Integrity: {integrity}")
        print(f"Tables: {tables}")

        for table in tables:
            row_count = connection.execute(
                f'SELECT COUNT(*) FROM "{table}"'
            ).fetchone()[0]

            columns = [
                (row[1], row[2])
                for row in connection.execute(
                    f'PRAGMA table_info("{table}")'
                )
            ]

            print(f"\nTable: {table}")
            print(f"Rows: {row_count:,}")
            print("Columns:")

            for name, data_type in columns:
                print(f"- {name}: {data_type}")

            preview = pd.read_sql_query(
                f'SELECT * FROM "{table}" LIMIT 3',
                connection,
            )

            print("\nFirst three rows:")
            print(preview.to_string(index=False))


inspect_kalshi()
inspect_polymarket()