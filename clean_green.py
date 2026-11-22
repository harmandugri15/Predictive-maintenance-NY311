"""
Cleaning Script 5 — NYC DEP Green Infrastructure
=================================================
Input  : data/raw_green_infrastructure.csv
Output : data/cleaned/cleaned_green_infrastructure.csv
Fixes  : Standardise ZIP, drop nulls, keep relevant columns,
         drop duplicates
"""

import pandas as pd
import os

INPUT  = "data/raw_green_infrastructure.csv"
OUTPUT = "data/cleaned/cleaned_green_infrastructure.csv"
os.makedirs("data/cleaned", exist_ok=True)

print("Loading raw green infrastructure data...")
df = pd.read_csv(INPUT, low_memory=False)
print(f"Raw shape : {df.shape}")
print(f"Columns   : {list(df.columns)}")

# 1. Drop exact duplicates
before = len(df)
df = df.drop_duplicates()
print(f"Dropped {before - len(df)} duplicate rows")

# 2. Find ZIP column
zip_col = None
for candidate in ["zipcode", "zip_code", "zip", "postal_code", "incident_zip"]:
    if candidate in df.columns:
        zip_col = candidate
        break

if zip_col:
    df["incident_zip"] = (
        df[zip_col]
        .astype(str)
        .str.strip()
        .str[:5]
        .str.zfill(5)
    )
    before = len(df)
    df = df[df["incident_zip"].str.match(r"^\d{5}$")]
    print(f"Dropped {before - len(df)} rows with invalid ZIP")
else:
    print("No ZIP column found — proceeding without ZIP filter")

# 3. Identify and keep useful columns
#    Common green infra columns: asset type, status, community board, borough, zip
useful_keywords = ["zip", "borough", "boro", "asset", "type", "status",
                   "community", "district", "latitude", "longitude",
                   "install", "date", "location", "id"]

keep = ["incident_zip"] if "incident_zip" in df.columns else []
for col in df.columns:
    if col == zip_col:
        continue
    if any(kw in col.lower() for kw in useful_keywords):
        keep.append(col)

# Always include geometry-free columns only
drop_keywords = ["the_geom", "geometry", "geom", "shape", "wkt"]
keep = [c for c in keep if not any(kw in c.lower() for kw in drop_keywords)]
keep = list(dict.fromkeys(keep))  # deduplicate while preserving order

df = df[keep] if keep else df

# 4. Drop rows where all key fields are null
df = df.dropna(how="all")

# 5. Standardise borough if present
for col in df.columns:
    if "borough" in col.lower() or "boro" in col.lower():
        df[col] = df[col].astype(str).str.upper().str.strip()

print(f"\nCleaned shape  : {df.shape}")
print(f"Columns kept   : {list(df.columns)}")
if "incident_zip" in df.columns:
    print(f"Unique ZIPs    : {df['incident_zip'].nunique()}")

df.to_csv(OUTPUT, index=False)
print(f"\nSaved → {OUTPUT}")
print(df.head(3))
