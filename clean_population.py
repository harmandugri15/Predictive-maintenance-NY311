"""
Cleaning Script 8 — US Census Population by ZIP
=================================================
Input  : data/raw_population.csv
Output : data/cleaned/cleaned_population.csv
Fixes  : Rename zcta → zip, cast population to int,
         filter NYC ZIPs only, drop margin of error column
"""

import pandas as pd
import os

INPUT  = "data/raw_population.csv"
OUTPUT = "data/cleaned/cleaned_population.csv"
os.makedirs("data/cleaned", exist_ok=True)

# NYC valid ZIP code range
NYC_ZIPS = set(
    [str(z) for z in range(10001, 10283)] +
    [str(z) for z in range(10301, 10315)] +
    [str(z) for z in range(10451, 10476)] +
    [str(z) for z in range(11004, 11110)] +
    [str(z) for z in range(11201, 11257)]
)

print("Loading raw population data...")
df = pd.read_csv(INPUT, low_memory=False)
print(f"Raw shape : {df.shape}")
print(f"Columns   : {list(df.columns)}")

# 1. Drop exact duplicates
before = len(df)
df = df.drop_duplicates()
print(f"Dropped {before - len(df)} duplicate rows")

# 2. Identify ZIP column
zip_col = None
for candidate in ["zcta", "zip code tabulation area", "zip", "zipcode",
                  "zip_code", "incident_zip", "geoid"]:
    if candidate in df.columns:
        zip_col = candidate
        break

if zip_col is None:
    zip_col = df.columns[-1]  # Census API puts ZCTA as last column
    print(f"Fallback: using last column as ZIP: '{zip_col}'")

print(f"ZIP column: '{zip_col}'")

# 3. Standardise to 5-digit ZIP
df["incident_zip"] = (
    df[zip_col]
    .astype(str)
    .str.strip()
    .str[:5]
    .str.zfill(5)
)

# 4. Filter to NYC ZIPs only
before = len(df)
df = df[df["incident_zip"].isin(NYC_ZIPS)]
print(f"Kept {len(df)} NYC ZIP rows (dropped {before - len(df)} non-NYC)")

# 5. Find and cast population column
pop_col = None
for candidate in ["population", "b01003_001e", "total_population", "pop"]:
    if candidate in df.columns:
        pop_col = candidate
        break

if pop_col:
    df["population"] = pd.to_numeric(df[pop_col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["population"])
    df["population"] = df["population"].astype(int)
    print(f"Dropped {before - len(df)} rows with invalid population")
    print(f"Population range: {df['population'].min()} – {df['population'].max()}")

# 6. Drop margin of error and raw zip columns
drop_cols = [c for c in df.columns if "moe" in c.lower() or "margin" in c.lower()
             or c == zip_col or (pop_col and c == pop_col and c != "population")]
df = df.drop(columns=drop_cols, errors="ignore")

# 7. Keep only clean columns
keep = ["incident_zip", "population"]
if "name" in df.columns:
    keep.append("name")
df = df[[c for c in keep if c in df.columns]]

# 8. Drop duplicate ZIPs — keep highest population entry
df = df.sort_values("population", ascending=False).drop_duplicates(subset=["incident_zip"])

print(f"\nCleaned shape  : {df.shape}")
print(f"Unique ZIPs    : {df['incident_zip'].nunique()}")
print(f"Total NYC pop  : {df['population'].sum():,}")

df.to_csv(OUTPUT, index=False)
print(f"\nSaved → {OUTPUT}")
print(df.head(5))
