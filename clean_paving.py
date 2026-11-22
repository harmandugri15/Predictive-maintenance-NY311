"""
Cleaning Script 7 — NYC DOT Paving Schedule
=============================================
Input  : data/raw_paving_schedule.csv
Output : data/cleaned/cleaned_paving_schedule.csv
Fixes  : Parse dates, drop cancelled jobs, keep ZIP + date +
         status columns, drop duplicates, filter 2015–2024
"""

import pandas as pd
import os

INPUT  = "data/raw_paving_schedule.csv"
OUTPUT = "data/cleaned/cleaned_paving_schedule.csv"
os.makedirs("data/cleaned", exist_ok=True)

print("Loading raw paving schedule data...")
df = pd.read_csv(INPUT, low_memory=False)
print(f"Raw shape : {df.shape}")
print(f"Columns   : {list(df.columns)}")

# 1. Drop exact duplicates
before = len(df)
df = df.drop_duplicates()
print(f"Dropped {before - len(df)} duplicate rows")

# 2. Find and parse date column
date_col = None
for candidate in ["start_date", "startdate", "work_date", "date",
                  "schedule_date", "planned_date", "from_date"]:
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
    print(f"Rows after date filter: {len(df)}")
else:
    print("No date column found — skipping date filter")

# 3. Drop cancelled jobs
status_col = None
for candidate in ["status", "job_status", "work_status"]:
    if candidate in df.columns:
        status_col = candidate
        break

if status_col:
    before = len(df)
    cancelled = df[status_col].str.upper().str.contains("CANCEL", na=False)
    df = df[~cancelled]
    print(f"Dropped {before - len(df)} cancelled jobs")

# 4. Standardise ZIP column
zip_col = None
for candidate in ["zipcode", "zip_code", "zip", "postcode"]:
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

# 5. Keep relevant columns
keep_keywords = ["date", "zip", "status", "street", "borough",
                 "boro", "block", "type", "segment", "work", "district"]
keep = ["incident_zip", "date"] if "incident_zip" in df.columns else ["date"]
keep = [c for c in keep if c in df.columns]

for col in df.columns:
    if col in ["date", zip_col]:
        continue
    if any(kw in col.lower() for kw in keep_keywords):
        keep.append(col)

drop_keywords = ["the_geom", "geometry", "geom", "shape"]
keep = [c for c in keep if not any(kw in c.lower() for kw in drop_keywords)]
keep = list(dict.fromkeys(keep))

df = df[keep] if keep else df

print(f"\nCleaned shape  : {df.shape}")
print(f"Columns kept   : {list(df.columns)}")

df.to_csv(OUTPUT, index=False)
print(f"\nSaved → {OUTPUT}")
print(df.head(3))
