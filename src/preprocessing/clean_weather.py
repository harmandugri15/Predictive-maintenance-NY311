"""
Cleaning Script 2 — NOAA Weather Data
=======================================
Input  : data/raw_weather.csv
Output : data/cleaned/cleaned_weather.csv
Fixes  : Parse DATE, convert tenths to real units, fill missing days,
         cap outliers, rename columns to lowercase
"""

import pandas as pd
import numpy as np
import os

INPUT  = "data/raw_weather.csv"
OUTPUT = "data/cleaned/cleaned_weather.csv"
os.makedirs("data/cleaned", exist_ok=True)

print("Loading raw weather data...")
df = pd.read_csv(INPUT, low_memory=False)
print(f"Raw shape: {df.shape}")
print(f"Columns  : {list(df.columns)}")

# 1. Parse DATE column
df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
before = len(df)
df = df.dropna(subset=["DATE"])
print(f"Dropped {before - len(df)} rows with unparseable dates")

# 2. Filter to 2015–2024
df = df[(df["DATE"] >= "2015-01-01") & (df["DATE"] <= "2024-12-31")]
print(f"Rows after date filter: {len(df)}")

# 3. NOAA stores values in tenths — divide by 10 to get real units
#    TMAX, TMIN → tenths of °C   → divide by 10 → °C
#    PRCP       → tenths of mm   → divide by 10 → mm
#    SNOW, SNWD → already in mm  → no change
#    AWND       → tenths of m/s  → divide by 10 → m/s
tenths_cols = ["TMAX", "TMIN", "PRCP", "AWND"]
for col in tenths_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce") / 10.0

for col in ["SNOW", "SNWD"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 4. Cap obvious outliers (sanity bounds for NYC)
if "TMAX" in df.columns:
    df["TMAX"] = df["TMAX"].clip(-40, 50)   # °C
if "TMIN" in df.columns:
    df["TMIN"] = df["TMIN"].clip(-40, 50)
if "PRCP" in df.columns:
    df["PRCP"] = df["PRCP"].clip(0, 300)    # mm per day
if "SNOW" in df.columns:
    df["SNOW"] = df["SNOW"].clip(0, 600)
if "SNWD" in df.columns:
    df["SNWD"] = df["SNWD"].clip(0, 1500)
if "AWND" in df.columns:
    df["AWND"] = df["AWND"].clip(0, 50)

# 5. Fill missing weather days with forward fill then backward fill
df = df.sort_values("DATE")
weather_cols = [c for c in ["TMAX","TMIN","PRCP","SNOW","SNWD","AWND"] if c in df.columns]
df[weather_cols] = df[weather_cols].ffill().bfill()

# 6. Build a complete date index (no gaps) and merge
full_dates = pd.DataFrame({"DATE": pd.date_range("2015-01-01", "2024-12-31")})
df = full_dates.merge(df, on="DATE", how="left")
df[weather_cols] = df[weather_cols].ffill().bfill()

# 7. Rename to lowercase
rename = {"DATE": "date", "TMAX": "tmax", "TMIN": "tmin",
          "PRCP": "prcp", "SNOW": "snow", "SNWD": "snwd", "AWND": "awnd"}
df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

# 8. Keep only useful columns
keep = [c for c in ["date","tmax","tmin","prcp","snow","snwd","awnd"] if c in df.columns]
df = df[keep]

print(f"\nCleaned shape : {df.shape}")
print(f"Date range    : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"Missing values:\n{df.isnull().sum()}")

df.to_csv(OUTPUT, index=False)
print(f"\nSaved → {OUTPUT}")
print(df.head(5))
