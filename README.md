# Project Polyshi

A data science project comparing Polymarket and Kalshi on the same 15-minute Bitcoin prediction markets.

## What this project studies

This project asks:

- How do Polymarket and Kalshi differ when pricing the same short-horizon BTC up/down contracts?
- Which platform appears better calibrated?
- Which platform shows tighter executable prices and more useful liquidity?
- Do apparent cross-market price dislocations survive fees and execution constraints?

The goal is to study market efficiency, calibration, liquidity, and short-horizon price discovery — not to build an automated trading bot.

## Current workflow

The project uses two live data collectors running side by side:

- **Polymarket logger** writes market snapshots to SQLite
- **Kalshi logger** writes market snapshots to CSV

The analysis pipeline then:

1. Matches equivalent BTC 15-minute windows across both platforms
2. Aligns near-simultaneous observations by timestamp
3. Compares executable ask prices rather than inferred prices
4. Applies official taker-fee formulas
5. Tests whether apparent dislocations remain after costs and top-of-book size limits

## Main tools

- Python
- pandas
- SQLite
- CSV pipelines
- Time-series matching
- Fee-adjusted execution analysis

## Repository structure

```text
analysis/   # data processing and comparison scripts
README.md   # project overview
```

Raw daily data and generated outputs are kept out of the public repo because they are large and updated continuously.

## Current status

This repository is an active research project.

The codebase currently supports:

- cross-market window matching
- executable-price reconstruction from explicit asks
- top-of-book liquidity checks
- fee-adjusted edge analysis

## Why it matters

This project is meant to show practical skills in:

- data collection
- working with APIs and market data
- SQLite and CSV handling
- timestamp alignment
- market microstructure reasoning
- reproducible analysis