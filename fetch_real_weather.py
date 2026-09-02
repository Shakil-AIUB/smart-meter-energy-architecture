"""
Fetch real historical outdoor temperature for Bangkok (where the CU-BEMS
building is located) using the free Open-Meteo Historical Weather API.
No API key required.

This replaces the synthetic outdoor_temp_c column with real weather data
that matches the CU-BEMS dataset's date range.

Usage:
    python fetch_real_weather.py --start 2019-01-01 --end 2019-01-31 --out bangkok_weather.csv

Then merge bangkok_weather.csv with the CU-BEMS data on matching timestamps
(see preprocess_cubems.py).
"""

import argparse
import requests  # pip install requests --break-system-packages
import pandas as pd

BANGKOK_LAT = 13.7563
BANGKOK_LON = 100.5018


def fetch_weather(start_date: str, end_date: str) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": BANGKOK_LAT,
        "longitude": BANGKOK_LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m",
        "timezone": "Asia/Bangkok",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(data["hourly"]["time"]),
        "outdoor_temp_c": data["hourly"]["temperature_2m"],
    })
    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch real historical outdoor weather for Bangkok.")
    parser.add_argument("--start", type=str, required=True, help="Start date, e.g. 2019-01-01")
    parser.add_argument("--end", type=str, required=True, help="End date, e.g. 2019-01-31")
    parser.add_argument("--out", type=str, default="bangkok_weather.csv")
    args = parser.parse_args()

    df = fetch_weather(args.start, args.end)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} hourly readings to {args.out}")
    print(df.head())


if __name__ == "__main__":
    main()
