"""
Preprocess the real CU-BEMS dataset (2018Floor4.csv, or any similar
floor/year file from the same release) into the project's standard schema:

    timestamp, occupancy, hvac_kw, lighting_kw, plug_kw, total_kw,
    outdoor_temp_c, indoor_temp_c

This is REAL sensor data from a 7-storey office building in Bangkok
(CU-BEMS, Pipattanasomporn et al. 2020, figshare DOI 10.6084/m9.figshare.11726517),
not synthetic. It replaces generate_dataset.py's output once you want to
run the evaluation on real data instead of the simulation used for early
prototyping.

What this script does, step by step:
1. Parses the raw per-zone CU-BEMS columns (z1_AC1(kW), z2_Light(kW), ...)
2. Aggregates all AC columns -> hvac_kw, all Light columns -> lighting_kw,
   all Plug columns -> plug_kw (summed across the 5 zones on the floor)
3. Averages the available zone temperature sensors -> indoor_temp_c
4. Builds an OCCUPANCY PROXY from normalised lighting+plug usage, since
   CU-BEMS does not include an occupancy sensor. This is a documented
   assumption -- state it explicitly in the paper's dataset section.
5. Resamples from 1-minute to a coarser interval (default 15 min) to
   reduce noise and file size, using the mean of each window.
6. Optionally merges in REAL outdoor temperature from fetch_real_weather.py
   (matched by nearest hourly reading) instead of a synthetic value.
7. Drops rows where indoor temperature is still missing after resampling
   (the CU-BEMS sensors have real gaps; this is expected of real data --
   report the dropped fraction in the paper as a data-quality note).

Usage:
    python preprocess_cubems.py --raw 2018Floor4.csv --weather bangkok_weather.csv \
        --interval-min 15 --out real_building_energy.csv

    # or without real weather (keeps outdoor_temp_c empty / NaN):
    python preprocess_cubems.py --raw 2018Floor4.csv --interval-min 15 --out real_building_energy.csv
"""

import argparse
import numpy as np
import pandas as pd


def load_and_aggregate(raw_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    df["timestamp"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %H:%M")
    df = df.drop(columns=["Date"])

    ac_cols = [c for c in df.columns if "_AC" in c]
    light_cols = [c for c in df.columns if "_Light" in c]
    plug_cols = [c for c in df.columns if "_Plug" in c]
    temp_cols = [c for c in df.columns if "(degC)" in c]

    # Sum across zones for each load category. Missing meter readings (NaN)
    # are treated as "no reading" and excluded from the sum for that row
    # rather than assumed zero, so a fully-offline meter doesn't silently
    # drag the total down -- min_count=1 keeps a row NaN if *all* inputs
    # are missing, but sums whatever real readings exist otherwise.
    hvac_kw = df[ac_cols].sum(axis=1, min_count=1)
    lighting_kw = df[light_cols].sum(axis=1, min_count=1)
    plug_kw = df[plug_cols].sum(axis=1, min_count=1)
    indoor_temp_c = df[temp_cols].mean(axis=1, skipna=True)

    out = pd.DataFrame({
        "timestamp": df["timestamp"],
        "hvac_kw": hvac_kw,
        "lighting_kw": lighting_kw,
        "plug_kw": plug_kw,
        "indoor_temp_c": indoor_temp_c,
    })
    out["total_kw"] = out[["hvac_kw", "lighting_kw", "plug_kw"]].sum(axis=1, min_count=1)
    return out


def add_occupancy_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    CU-BEMS has no occupancy sensor. We build a proxy from combined
    lighting + plug load (both correlate strongly with people being
    present, unlike AC which can run on a schedule independent of
    occupancy). Min-max normalised to [0, 1] using robust percentiles
    to avoid a handful of spikes compressing the whole scale.
    """
    signal = df["lighting_kw"].fillna(0) + df["plug_kw"].fillna(0)
    low, high = np.nanpercentile(signal, 2), np.nanpercentile(signal, 98)
    occupancy = (signal - low) / max(high - low, 1e-6)
    df["occupancy"] = occupancy.clip(0, 1).round(3)
    return df


def resample(df: pd.DataFrame, interval_min: int) -> pd.DataFrame:
    df = df.set_index("timestamp")
    resampled = df.resample(f"{interval_min}min").mean(numeric_only=True)
    resampled = resampled.reset_index()
    return resampled


def merge_weather(df: pd.DataFrame, weather_path: str) -> pd.DataFrame:
    weather = pd.read_csv(weather_path, parse_dates=["timestamp"])
    weather = weather.sort_values("timestamp")
    df = df.sort_values("timestamp")
    merged = pd.merge_asof(df, weather, on="timestamp", direction="nearest",
                            tolerance=pd.Timedelta("2h"))
    return merged


def main():
    parser = argparse.ArgumentParser(description="Preprocess real CU-BEMS data into project schema.")
    parser.add_argument("--raw", type=str, required=True, help="Path to raw CU-BEMS floor CSV, e.g. 2018Floor4.csv")
    parser.add_argument("--weather", type=str, default=None, help="Path to real weather CSV from fetch_real_weather.py")
    parser.add_argument("--interval-min", type=int, default=15, help="Resample interval in minutes (default: 15)")
    parser.add_argument("--out", type=str, default="real_building_energy.csv")
    args = parser.parse_args()

    df = load_and_aggregate(args.raw)
    df = add_occupancy_proxy(df)
    df = resample(df, args.interval_min)

    if args.weather:
        df = merge_weather(df, args.weather)
    else:
        df["outdoor_temp_c"] = np.nan
        print("No --weather file given: outdoor_temp_c left empty. "
              "Run fetch_real_weather.py and re-run with --weather to fill it in.")

    n_before = len(df)
    df = df.dropna(subset=["indoor_temp_c"]).reset_index(drop=True)
    n_after = len(df)
    dropped_pct = 100 * (n_before - n_after) / n_before if n_before else 0

    # Reorder to match the schema used by generate_dataset.py / control_loop.py
    df = df[["timestamp", "occupancy", "hvac_kw", "lighting_kw", "plug_kw",
             "total_kw", "outdoor_temp_c", "indoor_temp_c"]]

    df.to_csv(args.out, index=False)
    print(f"Wrote {n_after} rows to {args.out} "
          f"(dropped {n_before - n_after} rows / {dropped_pct:.1f}% with no indoor temp reading)")
    print(df.describe())


if __name__ == "__main__":
    main()
