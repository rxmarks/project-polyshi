from pathlib import Path
import sqlite3

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "cross_market" / "raw"

csv_files = list(RAW_DIR.rglob("*.csv"))
database_files = list(RAW_DIR.rglob("*.db"))

print("RAW SCHEMA INSPECTION")

print("\nCSV FILES")

if not csv_files:
    print("No CSV files found.")

for csv_path in csv_files:
    print(f"\nFILE: {csv_path}")
    print(f"Size: {csv_path.stat().st_size:,} bytes")

    if csv_path.stat().st_size == 0:
        print("Empty file")
        continue

    try:
        columns = pd.read_csv(
            csv_path,
            nrows=0,
        ).columns.tolist()

        print("Columns:")

        for column in columns:
            print(f"  - {column}")

    except Exception as error:
        print(f"ERROR: {error}")

print("\nSQLITE DATABASES")

if not database_files:
    print("No .db files found.")

for database_path in database_files:
    print(f"\nDATABASE: {database_path}")
    print(f"Size: {database_path.stat().st_size:,} bytes")

    try:
        with sqlite3.connect(database_path) as connection:
            integrity = connection.execute(
                "PRAGMA integrity_check"
            ).fetchone()[0]

            print(f"Integrity: {integrity}")

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

            if not tables:
                print("No user tables found.")
                continue

            for table in tables:
                escaped_table = table.replace('"', '""')

                row_count = connection.execute(
                    f'SELECT COUNT(*) FROM "{escaped_table}"'
                ).fetchone()[0]

                table_columns = connection.execute(
                    f'PRAGMA table_info("{escaped_table}")'
                ).fetchall()

                print(f"\nTABLE: {table}")
                print(f"Rows: {row_count:,}")
                print("Columns:")

                for column in table_columns:
                    column_name = column[1]
                    column_type = column[2]
                    print(
                        f"  - {column_name}"
                        f" ({column_type or 'untyped'})"
                    )

    except Exception as error:
        print(f"ERROR: {error}")