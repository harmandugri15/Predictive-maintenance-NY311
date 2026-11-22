"""
Cleaning Script 4 — NYC Water Main Breaks
==========================================
Input  : data/raw_water_main_breaks.csv
Output : data/cleaned/cleaned_water_main_breaks.csv
Fixes  : Parse dates, extract ZIP from address, drop irrelevant
         columns, drop duplicates, filter to 2015–2024
"""

import pandas as pd
import re
import os

INPUT  = "data/raw_water_main_breaks.csv"
OUTPUT = "data/cleaned/cleaned_water_main_breaks.csv"
os.makedirs("data/cleaned", exist_ok=True)

print("Loading raw water main breaks data...")
df = pd.read_csv(INPUT, low_memory=False)
print(f"Raw shape : {df.shape}")
print(f"Columns   : {list(df.columns)}")

# 1. Drop exact duplicates
before = len(df)
df = df.drop_duplicates()
print(f"Dropped {before - len(df)} duplicate rows")

# 2. Find and parse date column
date_col = None
for candidate in ["created_date", "opendate", "open_date", "date", "closedate"]:
    if candidate in df.columns:
        date_col = candidate
        break

if date_col:
    df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    before = len(df)
    df = df.dropna(subset=["date"])
    print(f"Dropped {before - len(df)} rows with unparseable dates")

    # Filter to 2015–2024
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= "2015-01-01") & (df["date"] <= "2024-12-31")]
    df["date"] = df["date"].dt.date
    print(f"Rows after date filter (2015–2024): {len(df)}")
else:
    print("No date column found — skipping date filter")

# 3. Extract ZIP from existing ZIP column or from address
def extract_zip(val):
    val = str(val).strip()
    match = re.search(r"\b(\d{5})\b", val)
    return match.group(1) if match else None

zip_col = None
for candidate in ["incident_zip", "zipcode", "zip_code", "zip"]:
    if candidate in df.columns:
        zip_col = candidate
        break

if zip_col:
    df["incident_zip"] = df[zip_col].astype(str).str.strip().str[:5].str.zfill(5)
else:
    # Try extracting from address columns
    for addr_col in ["incident_address", "address", "location"]:
        if addr_col in df.columns:
            df["incident_zip"] = df[addr_col].apply(extract_zip)
            print(f"ZIP extracted from '{addr_col}' column")
            break

# 4. Drop rows with no valid ZIP
if "incident_zip" in df.columns:
    before = len(df)
    df = df[df["incident_zip"].str.match(r"^\d{5}$", na=False)]
    print(f"Dropped {before - len(df)} rows missing valid ZIP")

# 5. Keep only relevant columns
keep = [c for c in [
    "unique_key", "date", "incident_zip",
    "complaint_type", "descriptor", "status",
    "borough", "latitude", "longitude"
] if c in df.columns]
df = df[keep]

# 6. Cast numeric columns
for col in ["latitude", "longitude"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

print(f"\nCleaned shape : {df.shape}")
if "incident_zip" in df.columns:
    print(f"Unique ZIPs   : {df['incident_zip'].nunique()}")

df.to_csv(OUTPUT, index=False)
print(f"\nSaved → {OUTPUT}")
print(df.head(3))
