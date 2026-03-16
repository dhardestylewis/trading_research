# Trading Research

## Overview
This repository contains the ongoing trading research, experiments, and automated strategy pipelines. It leverages machine learning models (TabPFN, Chronos) alongside classical statistical methods for forecasting, backtesting, and paper trading deployment.

## Repository Structure
- **`src/`**: Core source code modules.
- **`scripts/`**: Various scripts for processing data, running simulations, and generating reports.
- **`reports/`**: Experiment results and daily/quarterly reports.
- **`data/`**: Processed and intermediate data files.
- **`configs/`**: Configuration files for experiments.
- **`logs/`**: System logs.

## Current Focus
The current research branch focuses on:
- Exploring new trading lanes and assets using TabPFN walk-forward simulations.
- Validating the XLE strategy and evaluating edge significance after costs (e.g., measuring gross bps before execution costs).
- Paper trading deployments via Alpaca for testing signals and strategies.
- Density checks and fast evaluation loops to optimize pipeline efficiency.

## Setup & Dependencies
This project uses Python. Core dependencies include:
- `pandas`, `numpy`, `scikit-learn`
- ML frameworks like `TabPFN` and time-series models like `Chronos`.
- See `requirements.txt` for the full list of packages.

## Note on Experiments
All experiments (exp001 - exp029) are continually evaluated for economic viability over 30+ bps round-trip hurdles. Check individual `reports/` for ongoing experiment statuses and validation verdicts (such as `PROGRAM_DECISION.md`).
