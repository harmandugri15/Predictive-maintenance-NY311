"""
Cleaning Script 3 (Fixed) — NYC ZIP Code Boundaries
=====================================================
Input  : data/raw_zip_boundaries.geojson  (GeoJSON format)
Output : data/cleaned/cleaned_zip.csv
Fixes  : Reads GeoJSON, extracts ZIP + borough from properties,
         drops geometry, standardises ZIP, drops duplicates
"""

import json
import pandas as pd
import os

INPUT  = "data/raw_zip_boundaries.geojson"
OUTPUT = "data/cleaned/cleaned_zip.csv"
os.makedirs("data/cleaned", exist_ok=True)

print("Loading raw ZIP boundaries (GeoJSON)...")

with open(INPUT, "r") as f:
    raw = json.load(f)

# GeoJSON can be a FeatureCollection or a plain list of records
if isinstance(raw, dict) and "features" in raw:
    # Standard GeoJSON FeatureCollection
    rows = [f["properties"] for f in raw["features"]]
elif isinstance(raw, list):
    # NYC Open Data sometimes returns a plain JSON array
    rows = raw
else:
    rows = [raw]

df = pd.DataFrame(rows)
print(f"Raw shape : {df.shape}")
print(f"Columns   : {list(df.columns)}")

# 1. Find ZIP column
zip_col = None
for candidate in ["zipcode", "zip_code", "postalcode", "modzcta",
                  "zcta", "zip", "ZIPCODE", "ZIP_CODE"]:
    if candidate in df.columns:
        zip_col = candidate
        break

if zip_col is None:
    matches = [c for c in df.columns if "zip" in c.lower()]
    zip_col = matches[0] if matches else df.columns[0]

print(f"Using ZIP column: '{zip_col}'")

# 2. Standardise ZIP to 5-digit string
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

# 3. Extract borough
borough_col = None
for candidate in ["borough", "boro_name", "borough_name", "boro", "BOROUGH"]:
    if candidate in df.columns:
        borough_col = candidate
        break

if borough_col:
    df["borough"] = df[borough_col].str.upper().str.strip()
    print(f"Borough column: '{borough_col}'")
else:
    def zip_to_borough(z):
        z = str(z)
        if z.startswith("100") or z.startswith("101") or z.startswith("102"):
            return "MANHATTAN"
        elif z.startswith("104"):
            return "BRONX"
        elif z.startswith("112") or z.startswith("113") or z.startswith("114"):
            return "BROOKLYN"
        elif z.startswith("110") or z.startswith("111") or z.startswith("116"):
            return "QUEENS"
        elif z.startswith("103"):
            return "STATEN ISLAND"
        return "UNKNOWN"
    df["borough"] = df["incident_zip"].apply(zip_to_borough)
    print("Borough derived from ZIP prefix")

# 4. Keep only ZIP + borough
df = df[["incident_zip", "borough"]].drop_duplicates(subset=["incident_zip"])

print(f"\nCleaned shape : {df.shape}")
print(f"Boroughs      : {df['borough'].value_counts().to_dict()}")

df.to_csv(OUTPUT, index=False)
print(f"\nSaved → {OUTPUT}")
print(df.head(5))