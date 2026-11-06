"""Download FRED economic indicators into one monthly CSV dataset.

Install: pip install pandas requests
Set FRED_API_KEY in your environment, then run: python fred_recession.py

This downloads currently available historical observations (which may have
been revised), not historical real-time data vintages.
"""

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("FRED_API_KEY")

import os
import time
from pathlib import Path

import pandas as pd
import requests


# FRED ID: (economic category, approximate reporting cadence).
# Codes are individually checked by FRED's API at runtime; unavailable series
# are reported and skipped rather than crashing the entire download.
SERIES = {
    # Yield curve and monetary policy
    "T10Y3M": ("rates", "daily"), "T10Y2Y": ("rates", "daily"),
    "DGS10": ("rates", "daily"), "DGS2": ("rates", "daily"),
    "DGS3MO": ("rates", "daily"), "FEDFUNDS": ("rates", "monthly"),
    "DFEDTARU": ("rates", "daily"), "MORTGAGE30US": ("rates", "weekly"),
    "T10YIE": ("rates", "daily"), "DFII10": ("rates", "daily"),
    # Credit conditions, financial stress, and bank lending
    "BAMLH0A0HYM2": ("credit", "daily"), "BAA10Y": ("credit", "daily"),
    "AAA10Y": ("credit", "daily"), "STLFSI4": ("credit", "weekly"),
    "NFCI": ("credit", "weekly"), "ANFCI": ("credit", "weekly"),
    "BUSLOANS": ("credit", "monthly"), "TOTALSL": ("credit", "monthly"),
    "DRCCLACBS": ("credit", "quarterly"),
    "DRCLACBS": ("credit", "quarterly"),
    "TOTRESNS": ("credit", "monthly"),
    # Labor market
    "UNRATE": ("labor", "monthly"), "U6RATE": ("labor", "monthly"),
    "PAYEMS": ("labor", "monthly"), "MANEMP": ("labor", "monthly"),
    "ICSA": ("labor", "weekly"), "CCSA": ("labor", "weekly"),
    "JTSJOL": ("labor", "monthly"), "JTSQUR": ("labor", "monthly"),
    "JTSLDR": ("labor", "monthly"), "AWHMAN": ("labor", "monthly"),
    "CIVPART": ("labor", "monthly"), "EMRATIO": ("labor", "monthly"),
    "CES0500000003": ("labor", "monthly"),
    # Output, orders, inventories, and trade
    "INDPRO": ("production", "monthly"), "IPMAN": ("production", "monthly"),
    "TCU": ("production", "monthly"), "DGORDER": ("production", "monthly"),
    "NEWORDER": ("production", "monthly"), "AMTMNO": ("production", "monthly"),
    "ISRATIO": ("production", "monthly"), "RSAFS": ("production", "monthly"),
    "RSXFS": ("production", "monthly"),
    # Household incomes, spending, inflation and expectations
    "PCE": ("consumer", "monthly"), "PCEC96": ("consumer", "monthly"),
    "DSPIC96": ("consumer", "monthly"), "PSAVERT": ("consumer", "monthly"),
    "UMCSENT": ("consumer", "monthly"), "TOTALSA": ("consumer", "monthly"),
    "CPIAUCSL": ("prices", "monthly"), "CPILFESL": ("prices", "monthly"),
    "PCEPI": ("prices", "monthly"), "PCEPILFE": ("prices", "monthly"),
    "PPIACO": ("prices", "monthly"), "DCOILWTICO": ("prices", "daily"),
    "DHHNGSP": ("prices", "daily"),
    # Housing and construction
    "PERMIT": ("housing", "monthly"), "PERMIT1": ("housing", "monthly"),
    "HOUST": ("housing", "monthly"), "HOUST1F": ("housing", "monthly"),
    "HSN1F": ("housing", "monthly"), "EXHOSLUSM495S": ("housing", "monthly"),
    "MSPUS": ("housing", "quarterly"),
    "CSUSHPINSA": ("housing", "monthly"),
    # Markets, money and international
    "SP500": ("markets", "daily"), "VIXCLS": ("markets", "daily"),
    "DTWEXBGS": ("markets", "daily"), "M2SL": ("markets", "monthly"),
    "M1SL": ("markets", "monthly"), "BOPGSTB": ("markets", "monthly"),
    "USSLIND": ("leading", "monthly"),
}

TARGET_ID = "USREC"  # Historical NBER recession indicator: 0 or 1.
API_URL = "https://api.stlouisfed.org/fred/series/observations"


def get_series(series_id, api_key, start="1959-01-01", session=None):
    """Fetch raw observations directly from the official FRED JSON API."""
    http = session or requests.Session()
    response = http.get(
        API_URL,
        params={"series_id": series_id, "api_key": api_key,
                "file_type": "json", "observation_start": start},
        timeout=45,
    )
    response.raise_for_status()
    payload = response.json()
    if "observations" not in payload:
        raise ValueError(f"Unexpected FRED response for {series_id}: {payload}")
    values = pd.Series(
        {pd.Timestamp(x["date"]): pd.to_numeric(x["value"], errors="coerce")
         for x in payload["observations"]},
        name=series_id, dtype=float,
    ).sort_index()
    return values


def to_monthly(series, cadence):
    """Aggregate observations by their recorded month; do not impute missing values.

    Quarterly observations stay in their original quarter's recorded month.
    For daily/weekly series, use the last available observation in the month.
    """
    return series.resample("MS").last()


def download_data(api_key):
    columns = {}
    with requests.Session() as session:
        for i, (series_id, (group, cadence)) in enumerate(SERIES.items(), 1):
            try:
                raw = get_series(series_id, api_key, session=session)
                columns[series_id] = to_monthly(raw, cadence)
                print(f"[{i:02d}/{len(SERIES)}] OK   {series_id:17} {group}")
            except (requests.RequestException, ValueError, KeyError) as exc:
                print(f"[{i:02d}/{len(SERIES)}] SKIP {series_id:17} {exc}")
            time.sleep(0.1)
        recession = get_series(TARGET_ID, api_key, session=session)

    if not columns:
        raise RuntimeError("No FRED series loaded. Check the key and connection.")
    data = pd.concat([*columns.values(), to_monthly(recession, "monthly")], axis=1)
    data.index.name = "month"
    return data.sort_index()


if __name__ == "__main__":
    key = os.environ.get("FRED_API_KEY")
    df = download_data(key)
    output = Path("fred_economic_data.csv")
    df.to_csv(output)
    print(f"Saved {len(df):,} monthly rows and {len(df.columns):,} columns to {output}")
