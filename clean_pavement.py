"""
Cleaning Script 6 — NYC DOT Pavement Condition
===============================================
Input  : data/raw_pavement_current.csv
Output : data/cleaned/cleaned_pavement.csv
Fixes  : Cast rating to numeric, drop nulls, keep rating +
         street columns, drop duplicates
"""

import pandas as pd
import os

INPUT  = "data/raw_pavement_current.csv"
OUTPUT = "data/cleaned/cleaned_pavement.csv"
os.makedirs("data/cleaned", exist_ok=True)

print("Loading raw pavement data...")
df = pd.read_csv(INPUT, low_memory=False)
print(f"Raw shape : {df.shape}")
print(f"Columns   : {list(df.columns)}")

# 1. Drop exact duplicates
before = len(df)
df = df.drop_duplicates()
print(f"Dropped {before - len(df)} duplicate rows")

# 2. Find rating column
rating_col = None
for candidate in ["rating", "pavement_condition", "condition", "rating_word",
                  "pvmnt_type", "overall_rating", "cr_label"]:
    if candidate in df.columns:
        rating_col = candidate
        break

if rating_col:
    print(f"Rating column found: '{rating_col}'")
    # Try casting to numeric; if not numeric keep as category
    numeric_rating = pd.to_numeric(df[rating_col], errors="coerce")
    if numeric_rating.notna().sum() > len(df) * 0.5:
        df[rating_col] = numeric_rating
        df[rating_col] = df[rating_col].clip(1, 10)
        print(f"Rating range: {df[rating_col].min()} – {df[rating_col].max()}")
    else:
        df[rating_col] = df[rating_col].str.upper().str.strip()
        print(f"Rating categories: {df[rating_col].unique()}")

# 3. Find and standardise ZIP column
zip_col = None
for candidate in ["zipcode", "zip_code", "zip", "from_zip", "postcode"]:
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

# 4. Keep relevant columns
keep_keywords = ["street", "segment", "rating", "condition", "borough",
                 "boro", "zip", "from", "to", "block", "length", "type"]
keep = ["incident_zip"] if "incident_zip" in df.columns else []
for col in df.columns:
    if col == zip_col:
        continue
    if any(kw in col.lower() for kw in keep_keywords):
        keep.append(col)

# Remove geometry columns
drop_keywords = ["the_geom", "geometry", "geom", "shape"]
keep = [c for c in keep if not any(kw in c.lower() for kw in drop_keywords)]
keep = list(dict.fromkeys(keep))

df = df[keep] if keep else df

# 5. Drop rows where rating is missing
if rating_col and rating_col in df.columns:
    before = len(df)
    df = df.dropna(subset=[rating_col])
    print(f"Dropped {before - len(df)} rows with missing rating")

print(f"\nCleaned shape  : {df.shape}")
print(f"Columns kept   : {list(df.columns)}")

df.to_csv(OUTPUT, index=False)
print(f"\nSaved → {OUTPUT}")
print(df.head(3))
