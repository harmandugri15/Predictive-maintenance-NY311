"""
Cleaning Script 1 — NYC 311 Service Requests
=============================================
Input  : data/raw_311.csv
Output : data/cleaned/cleaned_311.csv
Fixes  : Parse dates, drop missing ZIP, standardise ZIP to 5-digit,
         remove non-NYC ZIPs, drop duplicates, uppercase borough
"""

import pandas as pd
import os

INPUT  = "data/raw_311.csv"
OUTPUT = "data/cleaned/cleaned_311.csv"
os.makedirs("data/cleaned", exist_ok=True)

# NYC valid ZIP code range
NYC_ZIPS = set(
    [str(z) for z in range(10001, 10283)] +  # Manhattan
    [str(z) for z in range(10301, 10315)] +  # Staten Island
    [str(z) for z in range(10451, 10476)] +  # Bronx
    [str(z) for z in range(11004, 11110)] +  # Queens
    [str(z) for z in range(11201, 11257)]    # Brooklyn
)

print("Loading raw 311 data...")
df = pd.read_csv(INPUT, low_memory=False)
print(f"Raw shape: {df.shape}")

# 1. Drop exact duplicate rows
before = len(df)
df = df.drop_duplicates()
print(f"Dropped {before - len(df)} duplicate rows")

# 2. Parse created_date
df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")

# 3. Drop rows where date could not be parsed
before = len(df)
df = df.dropna(subset=["created_date"])
print(f"Dropped {before - len(df)} rows with unparseable dates")

# 4. Extract date only (no time)
df["date"] = df["created_date"].dt.date

# 5. Standardise incident_zip
df["incident_zip"] = (
    df["incident_zip"]
    .astype(str)
    .str.strip()
    .str[:5]
    .str.zfill(5)
)

# 6. Drop rows missing ZIP
before = len(df)
df = df[df["incident_zip"].str.match(r"^\d{5}$")]
print(f"Dropped {before - len(df)} rows with invalid ZIP format")

# 7. Keep only NYC ZIPs
before = len(df)
df = df[df["incident_zip"].isin(NYC_ZIPS)]
print(f"Dropped {before - len(df)} rows with non-NYC ZIPs")

# 8. Uppercase borough
if "borough" in df.columns:
    df["borough"] = df["borough"].str.upper().str.strip()

# 9. Keep only relevant columns
keep = [c for c in [
    "unique_key", "date", "created_date",
    "complaint_type", "descriptor",
    "incident_zip", "borough",
    "latitude", "longitude",
    "status", "agency"
] if c in df.columns]
df = df[keep]

# 10. Cast lat/lon to float
for col in ["latitude", "longitude"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

print(f"\nCleaned shape : {df.shape}")
print(f"Unique ZIPs   : {df['incident_zip'].nunique()}")
print(f"Date range    : {df['date'].min()} → {df['date'].max()}")

df.to_csv(OUTPUT, index=False)
print(f"\nSaved → {OUTPUT}")
print(df.head(3))
