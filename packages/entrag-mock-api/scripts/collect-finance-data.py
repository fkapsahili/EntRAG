"""
This script collects structured financial data from Yahoo Finance.
"""

import datetime
import json
import os
from typing import Any

import click
import pandas as pd
import yfinance as yf
from loguru import logger


DEFAULT_TICKERS = ["MSFT", "AAPL", "GOOGL", "AMZN", "BRK.B", "CVS", "XOM", "UNH", "VWAGY", "WMT", "TSLA", "META"]
DATA_DIR = "datasets"
SCHEMA_VERSION = "1.0"


def collect_company_data(ticker: str, include_history: bool = False) -> dict[str, Any] | None:
    """
    Collect financial metrics for a given company ticker.
    Returns a dictionary with structured data or None if fetch fails.
    """
    try:
        company = yf.Ticker(ticker)
        info = company.info

        metrics = {
            "eps": float(info["trailingEps"]) if "trailingEps" in info else None,
            "pe_ratio": float(info["trailingPE"]) if "trailingPE" in info else None,
            "dividend": float(info["dividendRate"]) if "dividendRate" in info else None,
            "market_cap": float(info["marketCap"]) if "marketCap" in info else None,
        }
        data = {
            "ticker": ticker,
            "company_name": info.get("shortName", ""),
            "metrics": metrics,
        }

        if include_history:
            income_statement = company.income_stmt
            balance_sheet = company.balance_sheet

            if not income_statement.empty and "Net Income" in income_statement.index:
                historical_eps = {}

                for date in income_statement.columns:
                    net_income = income_statement.loc["Net Income", date]

                    # Find the closest matching date in balance sheet
                    matching_dates = [d for d in balance_sheet.columns if d.year == date.year]
                    if matching_dates:
                        closest_date = matching_dates[0]
                        shares_outstanding = None
                        for key in ["Ordinary Shares Number", "Common Stock"]:
                            if key in balance_sheet.index:
                                shares_outstanding = balance_sheet.loc[key, closest_date]
                                break

                        if pd.notna(net_income) and shares_outstanding and pd.notna(shares_outstanding):
                            date_str = date.strftime("%Y-%m-%d")
                            historical_eps[date_str] = float(net_income / shares_outstanding)

                data["historical_eps"] = historical_eps

        return data

    except Exception as e:
        logger.warning(f"Failed to fetch data for {ticker}: {e}")
        return None


def collect_all_data(tickers: tuple[str], include_history: bool = False) -> dict[str, Any]:
    """
    Collect data for a tuple of tickers and wrap it with metadata.
    """
    logger.info("Starting data collection...")
    data = {}

    for ticker in tickers:
        logger.info(f"Collecting data for ticker: {ticker}")
        company_data = collect_company_data(ticker, include_history=include_history)
        if company_data:
            data[ticker] = company_data

    return {
        "_meta": {
            "schema_version": SCHEMA_VERSION,
            "collected_at": datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",
            "include_history": include_history,
        },
        "data": data,
    }


@click.command()
@click.option("--output", default=f"{DATA_DIR}/finance_data.json", help="Output path for the collected data")
@click.option("--tickers", multiple=True, default=DEFAULT_TICKERS, help="List of tickers to collect data for")
@click.option("--include-history", is_flag=True, help="Include historical EPS data")
def collect_finance_data_json(output: str, tickers: tuple[str], include_history: bool) -> None:
    """
    CLI entry point using Click to collect and save financial data.
    """
    logger.info(f"Data directory: {os.path.dirname(output)}")
    collected_data = collect_all_data(set(tickers), include_history=include_history)

    if not len(collected_data["data"]) == len(tickers):
        logger.warning(
            f"Not all tickers were successfully collected. Expected {len(tickers)}, got {len(collected_data["data"])}."
        )
        return

    # Write data to JSON
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump(collected_data, f, indent=2)
    logger.info(f"Data saved to {output}")


if __name__ == "__main__":
    collect_finance_data_json()
