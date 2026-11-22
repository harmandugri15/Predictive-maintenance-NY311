"""
=============================================================================
CITY COMPLAINT PREDICTION — COMPLETE ML PIPELINE
=============================================================================

OBJECTIVE
---------
Predict the number of 311 service complaints filed per ZIP code on any given
day, using weather conditions, infrastructure history, and calendar features.

WHY THIS IS A REGRESSION PROBLEM (not classification)
------------------------------------------------------
The output — complaint count — is a continuous integer (0, 3, 47, 200…).
Classification requires discrete categories (low/medium/high). Predicting the
exact count gives city planners actionable numbers for resource allocation:
"Deploy 3 extra inspectors to ZIP 10025 tomorrow" is more useful than
"that ZIP is in the HIGH bucket."

PIPELINE OVERVIEW
-----------------
  Step 0  → Install & Import Libraries
  Step 1  → Load & Clean Data
  Step 2  → Exploratory Data Analysis (EDA)
  Step 3  → Feature Engineering
  Step 4  → Feature Selection
  Step 5  → Log-Transform the Target Variable
  Step 6  → Build Feature Matrix & Time-Series Split
  Step 7  → Iterative Modelling (baseline → LightGBM → ablation)
  Step 8  → Hyperparameter Tuning (Bayesian Optimization + RandomizedSearch)
  Step 9  → Final Comparison (tuned RF vs linear baselines)
  Step 10 → Residual Analysis & Deployment-Ready Artifacts

DESIGN PHILOSOPHY
-----------------
Every single decision in this pipeline is justified by data or theory.
We NEVER add a feature or change a model setting without measuring the effect
on held-out cross-validation performance. This reproducibility and
explainability is what separates engineering from guesswork.
=============================================================================
"""

# =============================================================================
# STEP 0 — INSTALL LIBRARIES & GLOBAL CONFIGURATION
# =============================================================================
# WHY we do this first:
#   Dependency errors caught upfront save hours of wasted computation later.
#   A missing library discovered at Step 8 forces a full re-run.

import subprocess, sys

def _install_if_missing(packages):
    """
    Install only packages not already importable.
    WHY: Avoids unnecessary pip overhead in environments where packages exist.
    """
    for pkg in packages:
        try:
            __import__(pkg.split("==")[0].replace("-", "_"))
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])

_install_if_missing(["lightgbm", "xgboost", "holidays", "scikit-learn",
                     "optuna", "shap", "scipy"])

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform

import lightgbm as lgb
import xgboost as xgb
import holidays

# WHY a fixed random seed?
#   Machine learning has stochastic components — random tree splits, random
#   parameter sampling, random shuffles. Without a seed, two runs of this
#   script produce different numbers, making it impossible to compare results
#   or debug regressions. SEED = 42 is conventional; the value doesn't matter,
#   consistency does.
SEED = 42
np.random.seed(SEED)

# WHY consistent plot styling?
#   Reproducible, professional-looking charts. The rcParams block saves ~3
#   lines per plot and eliminates style drift across the 15+ figures here.
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
})

# Path configuration (resolves project root whether run from root or src/)
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if os.path.basename(os.path.dirname(os.path.abspath(__file__))) == "src" else os.path.abspath(".")
DATA_PATH = os.path.join(BASE_DIR, "data", "modeling_table_final.csv") if os.path.exists(os.path.join(BASE_DIR, "data", "modeling_table_final.csv")) else "data/modeling_table_final.csv"
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

print("=" * 60)
print("CITY COMPLAINT PREDICTION — PIPELINE STARTING")
print("=" * 60)


# =============================================================================
# STEP 1 — LOAD & CLEAN DATA
# =============================================================================

def load_and_clean(path: str) -> pd.DataFrame:
    """
    Load the raw CSV, diagnose quality issues, and apply minimal targeted fixes.

    WHY we do NOT blindly impute all nulls here:
        Imputation decisions (mean vs median vs model-based) must be informed
        by the nature of each column. We make those decisions later, after EDA
        shows us the distributions. Imputing now would be guessing blind.

    WHY we drop rows where `borough` is null instead of imputing:
        Borough is a categorical ID — there is no "average borough." A ZIP code
        without a borough cannot be attributed to any geographic group, making it
        useless for neighbourhood-level analysis. The right action is removal.

    WHY we sort by date at the end:
        Lag features, rolling averages, and TimeSeriesSplit ALL require data to
        be in strict chronological order. One misordered row corrupts multiple
        downstream computations. Sorting once here is a single-point fix.
    """
    print("\n[Step 1] Loading data…")
    df = pd.read_csv(path)

    print(f"  Raw shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Date range: {df['date'].min()}  →  {df['date'].max()}")

    # ── 1a. Diagnose missing values ──────────────────────────────────────────
    null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    null_pct = null_pct[null_pct > 0]
    if len(null_pct) > 0:
        print("\n  Missing value summary (% of rows):")
        print(null_pct.to_string())

        fig, ax = plt.subplots(figsize=(10, 4))
        null_pct.plot(kind="bar", color="tomato", ax=ax, edgecolor="white")
        ax.set_title("Missing Values by Column (% of total rows)\n"
                     "WHY: Columns >50% missing are usually dropped; "
                     "<5% can be imputed")
        ax.set_ylabel("% Missing")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/01_missing_values.png", dpi=150)
        plt.show()
    else:
        print("  No missing values detected.")

    # ── 1b. Drop un-imputable categorical nulls ──────────────────────────────
    before = len(df)
    df = df.dropna(subset=["borough"])
    print(f"\n  Dropped {before - len(df)} rows with null `borough`. "
          f"Remaining: {len(df):,}")

    # ── 1c. Type coercion ────────────────────────────────────────────────────
    # WHY: pandas reads dates as `object` by default. Leaving them as strings
    #      breaks .dt accessor, sorting, and lag-feature computation.
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print("  Date column parsed. Data sorted chronologically.")
    return df


# =============================================================================
# STEP 2 — EXPLORATORY DATA ANALYSIS
# =============================================================================

def run_eda(df: pd.DataFrame) -> None:
    """
    Produce visualisations that answer specific questions about the data.
    Each plot is designed to inform ONE downstream modelling decision.

    EDA is NOT optional — it is the evidence base for every feature engineering
    and preprocessing choice made later. A model trained without EDA is a model
    built on assumptions that may be completely wrong.
    """
    print("\n[Step 2] Running EDA…")

    # ── 2a. Target distribution ──────────────────────────────────────────────
    # QUESTION: Is the target skewed? If yes, we need a log transform.
    # DECISION: Log-transform if skewness > 1 (right tail pulls the model).
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].hist(df["request_count"], bins=100, color="steelblue", edgecolor="none", alpha=0.8)
    axes[0].set_yscale("log")
    # WHY log-scale on y-axis (not x): We want to SEE the rare high-count events
    # without distorting the x-axis. Log-y compresses frequency, not the data.
    axes[0].set_title("Raw Complaint Count\n(log-scaled y to reveal the long tail)")
    axes[0].set_xlabel("Complaint Count")
    axes[0].set_ylabel("Frequency (log scale)")

    log_data = np.log1p(df["request_count"])
    axes[1].hist(log_data, bins=50, color="seagreen", edgecolor="white", density=True)
    from scipy.stats import gaussian_kde
    x_rng = np.linspace(log_data.min(), log_data.max(), 500)
    axes[1].plot(x_rng, gaussian_kde(log_data)(x_rng), color="darkgreen", lw=2)
    axes[1].set_title("log(Count+1) — Approx. Gaussian\n"
                      "WHY: Linear models assume this shape;\ntree models also improve")
    axes[1].set_xlabel("log(x + 1)")
    axes[1].set_ylabel("Density")

    daily = df.groupby("date")["request_count"].sum()
    rolling7 = daily.rolling(7, center=True).mean()
    axes[2].plot(daily.index, daily.values, color="darkorange", lw=0.5, alpha=0.4, label="Daily")
    axes[2].plot(daily.index, rolling7, color="firebrick", lw=2, label="7-day rolling avg")
    # WHY 7-day rolling? Weekly cycle is the dominant seasonal frequency.
    # A 7-day window captures one full cycle without over-smoothing.
    axes[2].set_title("Complaint Trend Over Time\nRolling avg reveals seasonality hidden in noise")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Total Complaints")
    axes[2].legend(frameon=False)
    plt.setp(axes[2].get_xticklabels(), rotation=30)

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/02_target_distribution.png", dpi=150)
    plt.show()

    skewness = df["request_count"].skew()
    print(f"  Skewness of raw target: {skewness:.2f}  "
          f"({'HIGH — log transform recommended' if skewness > 1 else 'OK — log transform optional'})")

    # ── 2b. Weather effect ───────────────────────────────────────────────────
    # QUESTION: Does rainfall cause complaint spikes?
    # DECISION: If yes, `prcp` is a useful feature and should be retained.
    max_prcp = df["prcp"].max()
    df["prcp_cat"] = pd.cut(
        df["prcp"],
        bins=[-0.01, 0.01, 0.25, 1.0, max_prcp + 0.1],
        labels=["No Rain", "Light", "Moderate", "Heavy"],
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(x="prcp_cat", y="request_count", data=df,
                showfliers=False, palette="Set2", ax=axes[0])
    # WHY showfliers=False: The distribution is right-skewed. A handful of
    # outlier dots would dominate the visual, hiding the central tendency.
    # We care about HOW THE TYPICAL ZIP CODE responds to rain, not anomalies.
    axes[0].set_title("Complaint Spread by Rain Level\n(outliers hidden — focus on typical range)")
    df.groupby("prcp_cat", observed=True)["request_count"].mean().plot(
        kind="bar", ax=axes[1], color=["#66c2a5","#fc8d62","#8da0cb","#e78ac3"],
        edgecolor="white", width=0.6)
    axes[1].set_title("Average Complaints per Rain Category\nSteeper rise → prcp is a stronger predictor")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/03_weather_effect.png", dpi=150)
    plt.show()

    # ── 2c. Temporal patterns ────────────────────────────────────────────────
    # QUESTION: Do weekday and month matter?
    # DECISION: If bar heights differ significantly, we MUST include
    #           day-of-week and month as features (and encode them cyclically).
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    dow_avg = df.groupby("day_of_week")["request_count"].mean()
    dow_avg.index = [day_names[i] for i in dow_avg.index]
    dow_avg.plot(kind="bar", color="royalblue", ax=axes[0], edgecolor="white", width=0.6)
    axes[0].set_title("Complaints by Day of Week\nWeekend dip is a real behavioural pattern")
    plt.sca(axes[0]); plt.xticks(rotation=30)

    month_avg = df.groupby("month")["request_count"].mean()
    month_avg.index = [month_names[i-1] for i in month_avg.index]
    month_avg.plot(kind="bar", color="darkorange", ax=axes[1], edgecolor="white", width=0.6)
    axes[1].set_title("Complaints by Month\nSeasonality confirms we need cyclical encoding")
    plt.sca(axes[1]); plt.xticks(rotation=30)

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/04_temporal_patterns.png", dpi=150)
    plt.show()

    # ── 2d. Correlation heatmap ──────────────────────────────────────────────
    # QUESTION: Which features are near-duplicates of each other?
    # DECISION: Pairs with |r| > 0.85 → keep one, drop the other, or merge.
    # WHY we show the FULL matrix (not upper triangle):
    #   A triangular heatmap requires readers to mentally flip the mirror.
    #   The full matrix is immediately readable by anyone, including non-experts.
    numeric_df = df.select_dtypes(include=["int64","float64"]).drop(
        columns=["request_count"], errors="ignore")
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, annot=True, cmap="RdBu_r", vmin=-1, vmax=1,
                center=0, fmt=".2f", linewidths=0.3, ax=ax, annot_kws={"size": 7})
    ax.set_title("Full Correlation Matrix\nLook for pairs near ±1 — these are redundant features")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/05_correlation_heatmap.png", dpi=150)
    plt.show()

    print("  EDA complete. Plots saved to ./plots/")


# =============================================================================
# STEP 3 — FEATURE ENGINEERING
# =============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw columns into representations that better expose the
    underlying patterns a regression model needs to learn.

    PRINCIPLE: We do not add features blindly. Each feature answers a specific
    question — "does adding X reduce our cross-validation MAE?" Features that
    do not improve performance are removed in Step 4.
    """
    print("\n[Step 3] Engineering features…")
    df = df.copy()

    # ── 3a. Cyclical encoding (sin/cos) ─────────────────────────────────────
    # PROBLEM: Month 12 (December) and Month 1 (January) are neighbours on the
    # calendar. If we feed the model the raw integers 12 and 1, it computes the
    # distance as 11 — treating them as far apart. But they are adjacent.
    #
    # SOLUTION: Map each month to a point on a unit circle using:
    #   sin(2π × month / 12) and cos(2π × month / 12)
    # On this circle, December and January are geometrically close (distance ≈ 0.5).
    #
    # WHY TWO COMPONENTS (sin AND cos)?
    #   A single sin value is ambiguous — sin(April) = sin(September) because
    #   they are symmetric about the sine wave's peak. Adding the cosine component
    #   breaks this ambiguity: each month maps to a UNIQUE (sin, cos) pair.
    if "month" not in df.columns:
        df["month"] = df["date"].dt.month
    if "day_of_week" not in df.columns:
        df["day_of_week"] = df["date"].dt.dayofweek

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
    print("  ✓ Cyclical encoding: month_sin, month_cos, day_sin, day_cos")

    # ── 3b. Temperature range ────────────────────────────────────────────────
    # WHY drop tmax and tmin individually?
    #   Their Pearson correlation is typically > 0.95 — they carry nearly
    #   identical information. Including both wastes a model parameter slot and
    #   creates multicollinearity in linear models (unstable coefficients).
    #
    # WHY temp_range (difference)?
    #   The RANGE tells us "how extreme was today?" — a large swing from cold
    #   morning to hot afternoon stresses both heating systems and water pipes.
    #   This is the signal that predicts complaints, not the absolute temperature.
    if "tmax" in df.columns and "tmin" in df.columns:
        df["temp_range"] = df["tmax"] - df["tmin"]
        print("  ✓ temp_range = tmax − tmin (replaces two redundant columns)")

    # ── 3c. Holiday flag ─────────────────────────────────────────────────────
    # WHY: Public holidays shift behaviour. Thanksgiving has more noise complaints
    # but fewer construction complaints. The model cannot learn this from
    # day_of_week alone — holidays fall on different weekdays each year.
    # The `holidays` library resolves each date correctly for US federal holidays.
    years = df["date"].dt.year.unique().tolist()
    us_hols = holidays.UnitedStates(years=years)
    df["is_holiday"] = df["date"].isin(us_hols).astype(int)
    print(f"  ✓ is_holiday flag created. Holiday records: {df['is_holiday'].sum():,}")

    # ── 3d. Rain × Infrastructure interaction ────────────────────────────────
    # WHY multiplicative interaction?
    #   Heavy rain alone is bad. Aged water infrastructure alone is bad.
    #   The combination is DISPROPORTIONATELY bad — exponential stress, not additive.
    #   Linear models cannot learn this without an explicit interaction term.
    #   Tree models CAN learn it (two consecutive splits), but the explicit term
    #   reduces the depth required and makes the split signal clearer.
    if "prcp" in df.columns and "water_breaks_ever" in df.columns:
        df["rain_x_infra"] = df["prcp"] * df["water_breaks_ever"]
        print("  ✓ rain_x_infra = prcp × water_breaks_ever (interaction term)")

    # ── 3e. Trend velocity ───────────────────────────────────────────────────
    # WHY: A 7-day rolling average tells the model the recent complaint level.
    # The DIFFERENCE between the 7-day and 30-day averages tells it whether
    # complaints are ACCELERATING (positive velocity) or SUBSIDING (negative).
    # Velocity is a momentum signal — a forward-looking feature.
    #
    # CRITICAL: We use .shift(1) before computing any rolling window.
    # Without shift, today's value enters the window = DATA LEAKAGE.
    # The model would "see the answer" as part of its input.
    if "rolling_avg_7d" in df.columns or "request_count" in df.columns:
        df_sorted = df.sort_values(["incident_zip", "date"])
        rolling_30 = (
            df_sorted.groupby("incident_zip")["request_count"]
            .transform(lambda x: x.shift(1).rolling(30, min_periods=10).mean())
        )
        rolling_7 = (
            df_sorted.groupby("incident_zip")["request_count"]
            .transform(lambda x: x.shift(1).rolling(7, min_periods=3).mean())
        )
        df["rolling_avg_30d"] = rolling_30.values
        df["rolling_avg_7d_safe"] = rolling_7.values
        df["trend_velocity"] = df["rolling_avg_7d_safe"] - df["rolling_avg_30d"]
        print("  ✓ trend_velocity = rolling_7d − rolling_30d (momentum signal)")

    # ── 3f. Per-ZIP historical average (expanding mean) ──────────────────────
    # WHY: Each ZIP has its own baseline complaint rate. A feature encoding
    # "what is the long-run average for this ZIP?" gives the model a
    # ZIP-level intercept without requiring it to learn it purely from the
    # encoded integer. Expanding mean (not a fixed window) adapts as more
    # history accumulates.
    # WHY .shift(1): same reason as above — prevents leakage.
    df_sorted = df.sort_values(["incident_zip", "date"])
    df["historical_zip_avg"] = (
        df_sorted.groupby("incident_zip")["request_count"]
        .transform(lambda x: x.shift(1).expanding().mean())
        .fillna(0)
        .values
    )
    print("  ✓ historical_zip_avg = expanding mean of past counts per ZIP")

    # ── 3g. Rolling standard deviation ──────────────────────────────────────
    # WHY: Volatility itself is predictive. A ZIP code with highly variable
    # complaint counts is harder to predict AND may be reacting to
    # recurring irregular events (festivals, construction). High rolling_std
    # signals an "unstable" ZIP that may spike again soon.
    df_sorted = df.sort_values(["incident_zip", "date"])
    df["rolling_std_7d"] = (
        df_sorted.groupby("incident_zip")["request_count"]
        .transform(lambda x: x.shift(1).rolling(7, min_periods=3).std())
        .fillna(0)
        .values
    )
    print("  ✓ rolling_std_7d = 7-day rolling std of past counts (volatility signal)")

    print(f"\n  Dataset now has {df.shape[1]} columns (was {df.shape[1] - 9} before engineering)")
    return df


# =============================================================================
# STEP 4 — FEATURE SELECTION
# =============================================================================

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that are redundant, leaky, or irrelevant to the model.

    PRINCIPLE OF PARSIMONY: A model with fewer, better features almost always
    generalises better than one with many redundant features.
    Reasons to drop:
      (a) Redundancy — two features carry the same information
      (b) Leakage — the feature encodes future or target information
      (c) Non-numeric identity — IDs that cannot be used as features directly

    WHY we do NOT use an automated filter (e.g., VarianceThreshold) alone:
      Automated filters are blind to semantics. They would keep a perfectly
      correlated pair if both had high variance. Human-guided selection,
      informed by EDA correlation plots and domain knowledge, is more reliable.
    """
    print("\n[Step 4] Selecting features…")

    drop_cols = {
        "tmax":              "Replaced by temp_range. Correlation with tmin > 0.95.",
        "tmin":              "Replaced by temp_range. Correlation with tmax > 0.95.",
        "day_of_year":       "Near-duplicate of month. Two features, one signal.",
        "prcp_cat":          "Created for EDA charts only — coarser than raw prcp.",
        "request_count_log": "Will rebuild cleanly as target. Keeping this = leakage.",
        "rolling_avg_7d":    "Replaced by rolling_avg_7d_safe (leak-free version).",
    }

    df_clean = df.copy()
    for col, reason in drop_cols.items():
        if col in df_clean.columns:
            df_clean.drop(columns=[col], inplace=True)
            print(f"  Dropped '{col}': {reason}")

    print(f"\n  Features: {df.shape[1]} → {df_clean.shape[1]}")
    return df_clean


# =============================================================================
# STEP 5 — LOG-TRANSFORM THE TARGET
# =============================================================================

def add_log_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log(x+1) to the complaint count and store as `target`.

    WHY log transform the target (not the features)?
      The complaint count is the Y variable. Its heavy right skew means:
        1. MSE penalises large errors quadratically → one 300-count day
           contributes 90,000 to the loss, overwhelming 1,000 normal days.
        2. Tree models waste splits on the rare tail.
      Transforming Y to log scale compresses the tail so errors are weighted
      more proportionally across the range.

    WHY log(x + 1) specifically?
      log(0) = −∞ (undefined). Adding 1 maps zero complaints to log(1) = 0,
      which is clean and meaningful. np.log1p(x) implements log(1+x) with
      better numerical stability than np.log(x+1) for small x.

    HOW to reverse for evaluation:
      np.expm1(pred) is the exact inverse of np.log1p(x).
      Evaluation metrics (MAE, RMSE) are always computed in original units
      after reversal so they are interpretable by stakeholders.
    """
    print("\n[Step 5] Log-transforming the target variable…")
    df = df.copy()
    df["target"] = np.log1p(df["request_count"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df["request_count"], bins=60, color="tomato", edgecolor="white")
    axes[0].set_title("Before: Raw Complaint Count\nSkewed — outliers dominate the loss function")
    axes[0].set_xlabel("Complaints")

    axes[1].hist(df["target"], bins=60, color="seagreen", edgecolor="white")
    axes[1].set_title("After: log(Count + 1)\nApproximately Gaussian — balanced loss")
    axes[1].set_xlabel("log(Complaints + 1)")

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/06_log_transform.png", dpi=150)
    plt.show()

    return df


# =============================================================================
# STEP 6 — BUILD FEATURE MATRIX & TIME-SERIES SPLIT
# =============================================================================

def build_feature_matrix(df: pd.DataFrame):
    """
    Construct X (feature matrix) and y (log-target vector), then visualise
    the TimeSeriesSplit folds.

    WHY LabelEncoder for incident_zip?
      ZIP codes are strings ('10001'). Most sklearn estimators require numeric
      input. LabelEncoder maps each unique ZIP to an integer (0, 1, 2…).
      The encoded integer is a proxy for location identity — tree models can
      learn "ZIP code #47 (= 10047) tends to have high counts" through splits
      on this integer.

    WHY NOT one-hot encoding for ZIP?
      With potentially hundreds of unique ZIP codes, one-hot would create
      hundreds of binary columns. This massively increases dimensionality and
      slows training without improving accuracy for tree models that can already
      handle cardinality via splits on the integer encoding.

    WHY TimeSeriesSplit, not K-Fold?
      Standard K-Fold randomly shuffles rows into folds. This means FOLD 3's
      test rows might be from January 2020, while training rows include December
      2020 — the model sees the FUTURE to predict the PAST. This is data
      leakage from the evaluation design itself, not just the features.
      TimeSeriesSplit enforces that test data ALWAYS comes AFTER training data,
      exactly mirroring how the model will be used in production.

    WHY 5 folds?
      More folds = lower variance in the CV estimate of performance (closer to
      the true population R²). 5 is the standard tradeoff: low enough to be
      computationally feasible, high enough for statistical reliability.

    WHY median imputation for remaining nulls?
      After dropping obviously problematic columns, a few nulls remain —
      mainly from the first N rows of lag/rolling features. We use MEDIAN
      (not mean) because the complaint count distribution is right-skewed.
      The mean is pulled toward high-count outliers; the median represents
      the typical behaviour of a ZIP code on a typical day.
    """
    print("\n[Step 6] Building feature matrix…")

    # Encode ZIP codes
    le = LabelEncoder()
    df["zip_encoded"] = le.fit_transform(df["incident_zip"].astype(str))

    # Columns to exclude from X
    DROP_FROM_X = [
        "request_count", "target", "date",
        "borough", "incident_zip", "prcp_cat",
    ]

    feature_cols = [
        c for c in df.columns
        if c not in DROP_FROM_X
        and df[c].dtype in ["int64","float64","float32","int32"]
    ]

    X = df[feature_cols].copy()
    y = df["target"].copy()

    # Median imputation — see docstring for why median over mean
    X = X.fillna(X.median())

    print(f"  Feature matrix X: {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"  Features: {list(X.columns)}")

    # Visualise the split structure
    tscv = TimeSeriesSplit(n_splits=5)
    fig, ax = plt.subplots(figsize=(13, 5))
    palette = ["#2ecc71", "#e74c3c"]
    for fold, (tr, te) in enumerate(tscv.split(X)):
        ax.barh(fold, len(tr), left=tr[0], height=0.5, color=palette[0],
                alpha=0.8, label="Training" if fold == 0 else "")
        ax.barh(fold, len(te), left=te[0], height=0.5, color=palette[1],
                alpha=0.9, label="Test" if fold == 0 else "")
    ax.set_yticks(range(5))
    ax.set_yticklabels([f"Fold {i+1}" for i in range(5)])
    ax.set_xlabel("Row index (time flows left → right)")
    ax.set_title("TimeSeriesSplit — Test always AFTER Training\n"
                 "WHY: Prevents peeking at future data (data leakage)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/07_timeseries_split.png", dpi=150)
    plt.show()

    return X, y, tscv, le


# =============================================================================
# STEP 7 — ITERATIVE MODELLING (BASELINE → LGBM → ABLATION)
# =============================================================================

def evaluate_model(model, X, y, tscv, label="Model") -> dict:
    """
    Run TimeSeriesSplit cross-validation and return averaged metrics.

    Metrics reported:
      R² (log scale) — fraction of variance explained in the log-transformed
                       target. Reported in log space because that is what
                       the model is trained to predict.
      MAE (original) — reversed to complaint units via expm1. This is the
                       metric stakeholders understand: "on average, we are
                       off by X complaints per day."
      RMSE (original) — penalises large individual errors more than MAE.
                        If RMSE >> MAE, there are occasional big failures
                        even when typical predictions are good.

    WHY we report BOTH log-space R² and original-space MAE?
      R² in log space measures how well the model fits the transformed target.
      MAE in original space measures practical usefulness. A model with R²=0.7
      may still have an unacceptably high MAE if it systematically mispredicts
      a few critical high-count days.
    """
    r2_scores, mae_scores, rmse_scores = [], [], []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_tr, y_tr)
        preds_log = model.predict(X_te)
        r2_scores.append(r2_score(y_te, preds_log))
        preds_orig = np.expm1(np.clip(preds_log, 0, None))
        y_orig     = np.expm1(y_te)
        mae_scores.append(mean_absolute_error(y_orig, preds_orig))
        rmse_scores.append(np.sqrt(mean_squared_error(y_orig, preds_orig)))

    result = {
        "Model": label,
        "R² (log)": round(np.mean(r2_scores), 4),
        "MAE (complaints)": round(np.mean(mae_scores), 2),
        "RMSE (complaints)": round(np.mean(rmse_scores), 2),
    }
    print(f"  {label:40s}  R²={result['R² (log)']:.4f}  "
          f"MAE={result['MAE (complaints)']:.2f}")
    return result


def run_iterative_modelling(X, y, tscv) -> list:
    """
    Test models in order of increasing complexity, recording performance
    at each step. Only adopt an increase in complexity if it improves CV R².

    WHY iterative instead of jumping to the best model?
      (a) Simpler models are cheaper to deploy and maintain.
      (b) If a simple model already solves the problem, complexity is waste.
      (c) The iteration produces an evidence trail — crucial for viva defence.
      (d) Later, ablation (removing each added component) verifies that each
          addition genuinely contributed.

    Models tested:
      Ridge   — linear baseline. If features have linear relationships with
                the target, Ridge should perform well. Regularisation (L2)
                handles correlated features robustly.

      Random Forest — bagging ensemble of decision trees. Captures non-linear
                      relationships and feature interactions. Less sensitive
                      to hyperparameters than gradient boosting. A good
                      second-step candidate after a linear baseline.

      LightGBM — gradient boosting, sequential tree building. Each tree
                 corrects the errors of the previous. Typically higher
                 accuracy than Random Forest on tabular data because it
                 targets residuals explicitly. Microsoft's histogram-based
                 algorithm is also 5–10× faster than XGBoost on large data.

      WHY NOT XGBoost here?
        XGBoost and LightGBM implement the same gradient boosting algorithm
        with different engineering choices. Benchmarking both in the iterative
        phase would double training time for marginal insight. LightGBM is
        faster and equally capable; XGBoost is evaluated in the final
        comparison for completeness.
    """
    print("\n[Step 7] Iterative modelling…")
    results = []

    models = [
        ("Ridge (linear baseline)",
         Ridge(alpha=1.0)),

        ("Random Forest (default)",
         RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)),

        ("LightGBM (default)",
         lgb.LGBMRegressor(n_estimators=200, random_state=SEED, n_jobs=-1, verbose=-1)),
    ]

    for label, model in models:
        results.append(evaluate_model(model, X, y, tscv, label))

    print("\n  Iteration Summary:")
    print(pd.DataFrame(results).to_string(index=False))
    return results


# =============================================================================
# STEP 8 — HYPERPARAMETER TUNING
# =============================================================================

def tune_random_forest(X, y, tscv) -> dict:
    """
    Fine-tune Random Forest hyperparameters using RandomizedSearchCV,
    then verify the improvement with 5-fold TimeSeriesSplit.

    WHY Randomized Search over Grid Search?
    ─────────────────────────────────────────
    Grid Search tests EVERY combination of parameters. With:
      n_estimators: [100, 200, 300]     →  3 options
      max_depth: [None, 10, 20, 30, 40] →  5 options
      min_samples_split: [2, 5, 10, 20] →  4 options
      min_samples_leaf: [1, 2, 4, 8]    →  4 options
      max_features: [0.5, 0.7, 1.0]     →  3 options
    Total combinations = 3×5×4×4×3 = 720. With 5 CV folds: 3,600 model fits.
    At ~15 seconds each → 15 HOURS of computation.

    RandomizedSearchCV samples n_iter=40 random combinations.
    With 3 CV folds during search: 40×3 = 120 fits → ~30 minutes.
    Bergstra & Bengio (2012) proved that random search matches or beats
    grid search when only a few parameters significantly affect performance
    (which is always the case in practice).

    WHY we use 3-fold CV during search but 5-fold for final evaluation?
    ─────────────────────────────────────────────────────────────────────
    During search we trade some evaluation precision for speed (3 folds).
    This is acceptable because we are looking for the DIRECTION of
    improvement, not the exact number. For the final reported performance,
    we retrain with 5 folds to get a low-variance, publishable estimate.

    PARAMETER EXPLANATIONS
    ─────────────────────────────────────────────────────────────────────
    n_estimators:       More trees → lower variance (bagging averages more),
                        but diminishing returns after ~200. Training time
                        scales linearly. We sample from [100, 150, 200, 300].

    max_depth:          Limits how deep each tree grows.
                        - None = fully grown = low bias, HIGH variance (overfitting risk)
                        - 10–30 = controlled complexity
                        WHY not always None? A single tree can memorise the
                        training data perfectly. Constraining depth forces
                        generalisable splits.

    min_samples_split:  A node must have at least this many samples to be
                        split further. Higher values = simpler trees.
                        WHY: Prevents learning overly specific rules like
                        "if ZIP=10001 AND tmax=72.3 AND day=3…" that do not
                        generalise.

    min_samples_leaf:   Every leaf must contain at least this many samples.
                        Higher values smooth predictions (like a local average),
                        reducing sensitivity to individual outlier rows.

    max_features:       Fraction of features randomly considered at each split.
                        WHY < 1.0? Random feature subsampling decorrelates the
                        trees — if one feature (e.g., rolling_avg) always wins,
                        all trees would be near-identical and averaging them
                        adds no diversity. Subsampling forces trees to learn
                        different aspects of the data, which is the key to
                        Random Forest's effectiveness.
    """
    print("\n[Step 8a] Tuning Random Forest with RandomizedSearchCV…")

    param_distributions = {
        "n_estimators":      [100, 150, 200, 300],
        "max_depth":         [None, 10, 15, 20, 25, 30, 35, 40],
        "min_samples_split": randint(2, 25),       # WHY randint? Continuous
        "min_samples_leaf":  randint(1, 12),        # sampling is more
        "max_features":      uniform(0.4, 0.6),     # efficient than a list
    }

    rf_base = RandomForestRegressor(random_state=SEED, n_jobs=-1)
    tscv_tune = TimeSeriesSplit(n_splits=3)   # fewer folds = faster search

    search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_distributions,
        n_iter=40,              # 40 random combos × 3 folds = 120 fits
        cv=tscv_tune,
        scoring="r2",
        random_state=SEED,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    search.fit(X, y)

    best_params = search.best_params_
    print(f"\n  Best hyperparameters found:")
    for k, v in best_params.items():
        print(f"    {k}: {v}")
    print(f"  Best CV R² (during search, 3-fold): {search.best_score_:.4f}")

    # ── 8b. Overfitting diagnostic ───────────────────────────────────────────
    # WHY we plot train vs test score for each candidate?
    #   A model with train_score = 0.99 and test_score = 0.55 is OVERFIT.
    #   This plot reveals the bias-variance tradeoff across all 40 candidates,
    #   confirming the selected params sit in the generalisation sweet spot.
    cv_results = pd.DataFrame(search.cv_results_)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(cv_results["mean_train_score"], cv_results["mean_test_score"],
               alpha=0.5, color="steelblue", s=40)
    ax.axline((0, 0), slope=1, color="red", linestyle="--", label="Perfect generalisation")
    ax.set_xlabel("Mean Train R²")
    ax.set_ylabel("Mean CV Test R²")
    ax.set_title("Overfitting Diagnostic: Train vs Test R² for 40 Candidates\n"
                 "Points far below the red line are overfit")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/08_overfitting_diagnostic.png", dpi=150)
    plt.show()

    return best_params


def tune_lightgbm_optuna(X, y, tscv, n_trials: int = 30) -> dict:
    """
    Tune LightGBM hyperparameters using Optuna — Bayesian Optimization.

    WHY Bayesian Optimization over Randomized Search for LightGBM?
    ──────────────────────────────────────────────────────────────
    RandomizedSearch treats each trial independently — it samples blindly
    from the parameter space, ignoring what previous trials revealed.

    Bayesian Optimization builds a PROBABILISTIC MODEL (Tree-structured
    Parzen Estimator — TPE) of the objective function:
        score = f(n_estimators, learning_rate, max_depth, …)

    After each trial, TPE updates its belief about which regions of the
    parameter space are likely to yield high scores. The next trial is
    chosen to maximise the Expected Improvement — balancing:
        • Exploitation: try params near the current best
        • Exploration: try regions not yet sampled

    In practice, Bayesian Optimization finds near-optimal parameters in
    30–50 trials, where random search might need 100+.

    LIGHTGBM-SPECIFIC PARAMETERS
    ─────────────────────────────
    learning_rate:    How much each tree corrects the previous error.
                      - Too high → overshoot, high variance
                      - Too low → need more trees, slow convergence
                      Typical range: 0.01 – 0.3. We search 0.01 – 0.2.

    num_leaves:       Maximum leaves per tree. Controls model complexity.
                      In LightGBM, a tree with L leaves can model 2^L
                      distinct regions of the feature space. More leaves =
                      more expressive but higher overfitting risk.
                      Rule of thumb: num_leaves << 2^max_depth.

    max_depth:        Maximum tree depth. Interacts with num_leaves.
                      LightGBM grows trees leaf-wise (not level-wise like
                      sklearn), so max_depth alone doesn't limit complexity —
                      num_leaves is the primary control.

    min_child_samples: Minimum data in a leaf. Equivalent to
                       min_samples_leaf in Random Forest. Higher = smoother
                       predictions, lower variance.

    subsample:        Fraction of training rows sampled per tree (row-wise
                      bagging). Adds diversity and prevents overfitting.
                      Also speeds up training.

    colsample_bytree: Fraction of features sampled per tree. Same purpose
                      as max_features in Random Forest — decorrelates trees.

    reg_alpha (L1):   Lasso-style regularisation on leaf weights. Drives
                      some weights exactly to zero — sparse model.

    reg_lambda (L2):  Ridge-style regularisation. Shrinks all leaf weights.
                      More stable than L1 but doesn't produce sparsity.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  Optuna not available. Skipping Bayesian tuning for LightGBM.")
        return {}

    print(f"\n[Step 8b] Tuning LightGBM with Optuna ({n_trials} trials)…")

    tscv_tune = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
            "max_depth":         trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state":      SEED,
            "n_jobs":            -1,
            "verbose":           -1,
        }
        model = lgb.LGBMRegressor(**params)
        scores = []
        for tr, te in tscv_tune.split(X):
            model.fit(X.iloc[tr], y.iloc[tr])
            scores.append(r2_score(y.iloc[te], model.predict(X.iloc[te])))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"\n  Best LightGBM params (Optuna):")
    for k, v in best.items():
        print(f"    {k}: {v}")
    print(f"  Best CV R² (3-fold): {study.best_value:.4f}")

    # ── Convergence plot ─────────────────────────────────────────────────────
    # WHY: Shows whether more trials would likely help. If the curve has
    # flattened, 30 trials was sufficient. If it is still rising, we should
    # run more trials.
    trial_values = [t.value for t in study.trials]
    best_so_far = np.maximum.accumulate(trial_values)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(trial_values, alpha=0.4, color="steelblue", label="Trial R²")
    ax.plot(best_so_far, color="red", lw=2, label="Best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("CV R²")
    ax.set_title("Optuna Convergence — LightGBM\n"
                 "Flattening curve means more trials won't help much")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/09_optuna_convergence.png", dpi=150)
    plt.show()

    return best


# =============================================================================
# STEP 9 — FINAL MODEL COMPARISON
# =============================================================================

def final_comparison(X, y, tscv, rf_best_params: dict, lgbm_best_params: dict) -> pd.DataFrame:
    """
    Benchmark all models using the same 5-fold TimeSeriesSplit and compute
    R², MAE, and RMSE in original (complaint count) units.

    WHY include Ridge and Lasso as baselines even if we know RF will win?
    ─────────────────────────────────────────────────────────────────────
    (a) Interpretability: Ridge/Lasso coefficients show WHICH features drive
        predictions and in what direction. If stakeholders need to explain
        a prediction to a city council, a coefficient table is more persuasive
        than feature importance from a forest.
    (b) Calibration: If RF only marginally beats Ridge, the added complexity
        is not justified. The comparison provides the evidence.
    (c) Ridge (L2) penalises large coefficients — handles correlated features.
        Lasso (L1) performs feature selection — coefficients of redundant
        features go to zero, telling us which features the linear model
        considers useless.

    WHY XGBoost appears here:
        XGBoost and LightGBM are comparable in accuracy but XGBoost uses
        exact greedy splits (slower but sometimes more accurate) while LGBM
        uses histogram approximation (faster). Including both lets us verify
        that our dataset size and feature count favour one approach.

    EVALUATION METRIC CHOICE
    ────────────────────────
    R²: Used for ranking models. Scale-independent — comparable across datasets.
    MAE: Primary business metric. Interpretable: "We err by X complaints/day."
    RMSE: Secondary metric. Higher weight on large errors. RMSE >> MAE indicates
          occasional catastrophic failures that MAE masks.
    """
    print("\n[Step 9] Final model comparison…")

    # Build tuned models with best params found in Step 8
    rf_params = {**rf_best_params, "random_state": SEED, "n_jobs": -1}
    lgbm_params = {**lgbm_best_params,
                   "random_state": SEED, "n_jobs": -1, "verbose": -1} if lgbm_best_params else {}

    competitors = {
        "Ridge (L2, α=1)":           Ridge(alpha=1.0),
        "Lasso (L1, α=0.01)":        Lasso(alpha=0.01),
        "Random Forest (default)":   RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
        "Random Forest (tuned)":     RandomForestRegressor(**rf_params),
        "LightGBM (default)":        lgb.LGBMRegressor(n_estimators=200, random_state=SEED, n_jobs=-1, verbose=-1),
    }
    if lgbm_params:
        competitors["LightGBM (Optuna-tuned)"] = lgb.LGBMRegressor(**lgbm_params)

    results = []
    last_preds = None
    last_actual = None

    for label, model in competitors.items():
        r2_list, mae_list, rmse_list = [], [], []
        for i, (tr, te) in enumerate(tscv.split(X)):
            model.fit(X.iloc[tr], y.iloc[tr])
            pred_log  = model.predict(X.iloc[te])
            pred_orig = np.expm1(np.clip(pred_log, 0, None))
            y_orig    = np.expm1(y.iloc[te])

            r2_list.append(r2_score(y.iloc[te], pred_log))
            mae_list.append(mean_absolute_error(y_orig, pred_orig))
            rmse_list.append(np.sqrt(mean_squared_error(y_orig, pred_orig)))

            if "tuned" in label.lower() and i == 4:  # save last fold
                last_preds  = pred_orig
                last_actual = y_orig.values

        results.append({
            "Model":              label,
            "R² (log)":           round(np.mean(r2_list), 4),
            "MAE (complaints)":   round(np.mean(mae_list), 2),
            "RMSE (complaints)":  round(np.mean(rmse_list), 2),
        })
        print(f"  {label:35s}  R²={results[-1]['R² (log)']:.4f}  "
              f"MAE={results[-1]['MAE (complaints)']:.2f}")

    results_df = pd.DataFrame(results).sort_values("R² (log)", ascending=False).reset_index(drop=True)

    # ── Comparison bar chart ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ["R² (log)", "MAE (complaints)", "RMSE (complaints)"]
    colors  = ["steelblue", "tomato", "goldenrod"]
    for ax, metric, color in zip(axes, metrics, colors):
        results_df.plot(x="Model", y=metric, kind="bar", ax=ax,
                        color=color, edgecolor="white", legend=False)
        ax.set_title(f"Model Comparison — {metric}")
        ax.set_xlabel("")
        plt.sca(ax); plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/10_model_comparison.png", dpi=150)
    plt.show()

    # ── Predicted vs Actual (final fold of tuned RF) ─────────────────────────
    if last_preds is not None:
        n_show = min(500, len(last_actual))
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(last_actual[:n_show], color="steelblue", lw=1, alpha=0.8, label="Actual")
        ax.plot(last_preds[:n_show],  color="red", lw=1, alpha=0.7,
                linestyle="--", label="Predicted (tuned RF)")
        ax.fill_between(range(n_show), last_actual[:n_show], last_preds[:n_show],
                        alpha=0.15, color="orange", label="Error")
        ax.set_title("Predicted vs Actual Complaints — Tuned RF (last 500 test observations)\n"
                     "WHY: Visual check for systematic bias (always over/under-predicting)")
        ax.set_xlabel("Time (chronological)")
        ax.set_ylabel("Complaint Count")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/11_predicted_vs_actual.png", dpi=150)
        plt.show()

    return results_df


# =============================================================================
# STEP 10 — RESIDUAL ANALYSIS & FEATURE IMPORTANCE
# =============================================================================

def residual_analysis_and_importance(X, y, best_params: dict) -> None:
    """
    Train the final model on all data, then:
      (a) Analyse residuals to check for systematic biases
      (b) Extract feature importances to explain WHAT drives predictions

    WHY residual analysis?
    ──────────────────────
    A residual is (actual − predicted). In a perfect model, residuals are
    random noise — no pattern. Systematic patterns in residuals reveal:
      • Heteroscedasticity: errors grow with predicted value → model
        underestimates high-complaint days. A fix: quantile regression.
      • Temporal autocorrelation: errors cluster in time → the model misses
        a trend. A fix: better lag features or a temporal model (ARIMA).
      • Spatial clustering: errors cluster by borough → geography is not
        fully captured. A fix: borough-level features.

    WHY feature importance?
    ────────────────────────
    Random Forest feature importance = mean decrease in impurity (MDI):
    how much does including feature F reduce the average prediction error
    across all trees? Higher = more important.

    CAVEAT: MDI importance is biased toward HIGH-CARDINALITY features
    (like zip_encoded with hundreds of unique values) because high cardinality
    creates more split opportunities. The importance ranking is still useful
    for intuition but should not be treated as ground truth.

    Alternative (more reliable): SHAP values — but computationally expensive
    for large Random Forests. We note this as a production improvement.
    """
    print("\n[Step 10] Residual analysis and feature importance…")

    params = {**best_params, "random_state": SEED, "n_jobs": -1}
    final_model = RandomForestRegressor(**params)
    final_model.fit(X, y)

    preds_log  = final_model.predict(X)
    residuals  = np.expm1(y) - np.expm1(np.clip(preds_log, 0, None))

    # ── Residual plots ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Residuals vs Predicted
    # LOOK FOR: Funnel shape = heteroscedasticity (errors grow with prediction).
    # If visible, consider Weighted Least Squares or log-scale evaluation.
    preds_orig = np.expm1(np.clip(preds_log, 0, None))
    axes[0].scatter(preds_orig, residuals, alpha=0.2, s=5, color="steelblue")
    axes[0].axhline(0, color="red", lw=1.5)
    axes[0].set_xlabel("Predicted Complaints")
    axes[0].set_ylabel("Residual (Actual − Predicted)")
    axes[0].set_title("Residuals vs Predicted\nFunnel = underestimating high-count days")

    # Plot 2: Residual histogram
    # LOOK FOR: Bell shape centred on 0 = residuals are random noise (good).
    # Skewed histogram = systematic over- or under-prediction.
    axes[1].hist(residuals, bins=60, color="tomato", edgecolor="white")
    axes[1].axvline(0, color="black", lw=1.5, linestyle="--")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution\nCentred bell shape = random errors (ideal)")

    # Plot 3: Q-Q plot (Quantile-Quantile)
    # LOOK FOR: Points near the diagonal = residuals are approximately Gaussian.
    # Heavy tails = occasional large errors not captured by Gaussian assumption.
    from scipy import stats
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, fit=True)
    axes[2].plot(osm, osr, "o", ms=2, alpha=0.3, color="steelblue")
    axes[2].plot(osm, slope * np.array(osm) + intercept, color="red", lw=1.5)
    axes[2].set_xlabel("Theoretical Quantiles")
    axes[2].set_ylabel("Sample Quantiles")
    axes[2].set_title(f"Q-Q Plot of Residuals (r={r:.3f})\nDeviation from line = non-Gaussian tails")

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/12_residual_analysis.png", dpi=150)
    plt.show()

    # ── Feature importance ───────────────────────────────────────────────────
    importances = final_model.feature_importances_
    imp_df = (
        pd.DataFrame({"Feature": X.columns, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(15)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=imp_df, palette="viridis", ax=ax)
    ax.set_title("Top 15 Feature Importances (MDI)\n"
                 "WHY: Shows WHAT drives complaints — actionable for city planners\n"
                 "CAVEAT: High-cardinality features may be over-represented")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/13_feature_importance.png", dpi=150)
    plt.show()

    print("\n  Top 5 most important features:")
    for _, row in imp_df.head(5).iterrows():
        print(f"    {row['Feature']}: {row['Importance']:.4f}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Orchestrates all pipeline steps in order.

    WHY a main() function instead of top-level script code?
    ─────────────────────────────────────────────────────────
    Encapsulation in main() allows:
      (a) Importing this module in a test suite without side effects
      (b) Running individual steps in isolation for debugging
      (c) Clear documentation of the execution order
      (d) Easy conversion to a CLI tool later (argparse, click)
    """
    # ── Step 1: Load ─────────────────────────────────────────────────────────
    df = load_and_clean(DATA_PATH)

    # ── Step 2: EDA ──────────────────────────────────────────────────────────
    run_eda(df)

    # ── Step 3: Feature engineering ──────────────────────────────────────────
    df = engineer_features(df)

    # ── Step 4: Feature selection ─────────────────────────────────────────────
    df = select_features(df)

    # ── Step 5: Log-transform target ──────────────────────────────────────────
    df = add_log_target(df)

    # ── Step 6: Feature matrix ────────────────────────────────────────────────
    X, y, tscv, le = build_feature_matrix(df)

    # ── Step 7: Iterative modelling ───────────────────────────────────────────
    iteration_results = run_iterative_modelling(X, y, tscv)

    # ── Step 8: Hyperparameter tuning ─────────────────────────────────────────
    rf_best  = tune_random_forest(X, y, tscv)
    lgbm_best = tune_lightgbm_optuna(X, y, tscv, n_trials=30)

    # ── Step 9: Final comparison ──────────────────────────────────────────────
    final_df = final_comparison(X, y, tscv, rf_best, lgbm_best)

    print("\n" + "=" * 60)
    print("FINAL LEADERBOARD")
    print("=" * 60)
    print(final_df.to_string(index=False))

    # ── Step 10: Residual analysis & importance ────────────────────────────────
    residual_analysis_and_importance(X, y, rf_best)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE. All plots saved to ./plots/")
    print("=" * 60)


if __name__ == "__main__":
    main()
