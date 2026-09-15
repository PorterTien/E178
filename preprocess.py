"""
Preprocessing pipeline for NASA Li-ion battery capacity prediction.

Builds a per-cycle feature table from the 4 canonical batteries
(B0005, B0006, B0007, B0018), drops invalid rows, does a chronological
70/15/15 train/val/test split (no shuffle), and fits a StandardScaler on the
training set only.

Outputs (under data/):
    X_train.csv, X_val.csv, X_test.csv
    y_train.csv, y_val.csv, y_test.csv
    scaler_params.json       -> {"mean": [...], "scale": [...], "features": [...]}
    feature_names.txt        -> one feature name per line, in order

Run:
    python preprocess.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

SCRIPT_DIR = Path(__file__).resolve().parent
# Prefer raw_data/ (as per the prompt) but fall back to this repo's layout.
CANDIDATE_DIRS = [SCRIPT_DIR / "raw_data", SCRIPT_DIR / "cleaned_dataset"]
DATA_ROOT = next((p for p in CANDIDATE_DIRS if p.exists()), CANDIDATE_DIRS[0])
METADATA_CSV = DATA_ROOT / "metadata.csv"
CYCLES_DIR = DATA_ROOT / "data"

OUT_DIR = SCRIPT_DIR / "data"

FEATURE_COLS = [
    "cycle_number",
    "voltage_mean",
    "voltage_min",
    "current_mean",
    "temp_mean",
    "temp_max",
    "ambient_temp",
]
TARGET_COL = "capacity"

# Capacity sanity range for Li-ion 18650 cells in this dataset (Ah).
CAPACITY_MIN = 1.0
CAPACITY_MAX = 2.1

# Chronological split fractions.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# Remainder (0.15) goes to test.


# --------------------------------------------------------------------------- #
# Loading + aggregation (mirrors eda.py so the two scripts are independent)
# --------------------------------------------------------------------------- #

def load_discharge_metadata(battery_ids: list[str]) -> pd.DataFrame:
    meta = pd.read_csv(METADATA_CSV)
    mask = (meta["type"] == "discharge") & (meta["battery_id"].isin(battery_ids))
    meta = meta.loc[mask].copy()
    meta["Capacity"] = pd.to_numeric(meta["Capacity"], errors="coerce")
    meta["ambient_temperature"] = pd.to_numeric(
        meta["ambient_temperature"], errors="coerce"
    )
    return meta.reset_index(drop=True)


def get_available_battery_ids() -> list[str]:
    """Discover all batteries that have discharge cycles in metadata.csv."""
    meta = pd.read_csv(METADATA_CSV)
    discharge = meta.loc[meta["type"] == "discharge", "battery_id"].dropna()
    return sorted(discharge.unique().tolist())


def aggregate_cycle(cycle_csv: Path) -> dict[str, float] | None:
    try:
        df = pd.read_csv(cycle_csv)
    except Exception as exc:  # pragma: no cover
        print(f"  skipping {cycle_csv.name}: {exc}")
        return None
    if df.empty:
        return None
    return {
        "voltage_mean": df["Voltage_measured"].mean(),
        "voltage_min": df["Voltage_measured"].min(),
        "current_mean": df["Current_measured"].mean(),
        "temp_mean": df["Temperature_measured"].mean(),
        "temp_max": df["Temperature_measured"].max(),
    }


def build_cycle_dataframe(battery_ids: list[str]) -> pd.DataFrame:
    meta = load_discharge_metadata(battery_ids)
    rows: list[dict] = []
    for battery_id, battery_meta in meta.groupby("battery_id", sort=False):
        battery_meta = battery_meta.reset_index(drop=True)
        for cycle_idx, row in battery_meta.iterrows():
            cycle_csv = CYCLES_DIR / row["filename"]
            stats = aggregate_cycle(cycle_csv)
            if stats is None:
                continue
            rows.append(
                {
                    "battery_id": battery_id,
                    "cycle_number": cycle_idx + 1,
                    **stats,
                    "ambient_temp": row["ambient_temperature"],
                    TARGET_COL: row["Capacity"],
                }
            )
    df = pd.DataFrame(rows)
    return df[["battery_id", *FEATURE_COLS, TARGET_COL]]


# --------------------------------------------------------------------------- #
# Cleaning + splitting
# --------------------------------------------------------------------------- #

def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop rows with null/out-of-range capacity. Return the cleaned df and a
    summary of what was removed so we can print it at the end."""
    initial = len(df)

    null_mask = df[TARGET_COL].isna()
    n_null = int(null_mask.sum())

    # After removing nulls, flag out-of-range capacity values.
    non_null = df.loc[~null_mask].copy()
    out_of_range_mask = (non_null[TARGET_COL] < CAPACITY_MIN) | (
        non_null[TARGET_COL] > CAPACITY_MAX
    )
    n_out_of_range = int(out_of_range_mask.sum())
    cleaned = non_null.loc[~out_of_range_mask].reset_index(drop=True)

    summary = {
        "initial_rows": initial,
        "dropped_null_capacity": n_null,
        "dropped_out_of_range": n_out_of_range,
        "final_rows": len(cleaned),
    }
    return cleaned, summary


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sort by cycle_number and take the first 70% / next 15% / last 15%.

    No shuffling — this is the correct protocol for time-series degradation
    forecasting so the model never sees the future.
    """
    df_sorted = df.sort_values("cycle_number", kind="mergesort").reset_index(drop=True)
    n = len(df_sorted)
    n_train = int(np.floor(TRAIN_FRAC * n))
    n_val = int(np.floor(VAL_FRAC * n))

    train = df_sorted.iloc[:n_train]
    val = df_sorted.iloc[n_train : n_train + n_val]
    test = df_sorted.iloc[n_train + n_val :]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    battery_ids = get_available_battery_ids()

    print(f"Data root:   {DATA_ROOT}")
    print(f"Batteries:   {len(battery_ids)} total")
    print(f"Battery IDs: {', '.join(battery_ids)}")
    print("Building per-cycle dataframe ...")
    df = build_cycle_dataframe(battery_ids)

    # --- Clean ------------------------------------------------------------- #
    df_clean, drop_summary = clean(df)

    # Also drop rows with any NaN in the feature columns (rare but possible if
    # a cycle CSV has only one timestep, etc.). We track these separately.
    feature_null_mask = df_clean[FEATURE_COLS].isna().any(axis=1)
    n_feature_nulls = int(feature_null_mask.sum())
    df_clean = df_clean.loc[~feature_null_mask].reset_index(drop=True)
    drop_summary["dropped_feature_nulls"] = n_feature_nulls
    drop_summary["final_rows"] = len(df_clean)

    # --- Split ------------------------------------------------------------- #
    train, val, test = chronological_split(df_clean)

    X_train = train[FEATURE_COLS]
    X_val = val[FEATURE_COLS]
    X_test = test[FEATURE_COLS]
    y_train = train[[TARGET_COL]]
    y_val = val[[TARGET_COL]]
    y_test = test[[TARGET_COL]]

    # --- Scale (fit on train only) ---------------------------------------- #
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=FEATURE_COLS, index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val), columns=FEATURE_COLS, index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=FEATURE_COLS, index=X_test.index
    )

    # --- Export ------------------------------------------------------------ #
    X_train_scaled.to_csv(OUT_DIR / "X_train.csv", index=False)
    X_val_scaled.to_csv(OUT_DIR / "X_val.csv", index=False)
    X_test_scaled.to_csv(OUT_DIR / "X_test.csv", index=False)
    y_train.to_csv(OUT_DIR / "y_train.csv", index=False)
    y_val.to_csv(OUT_DIR / "y_val.csv", index=False)
    y_test.to_csv(OUT_DIR / "y_test.csv", index=False)

    scaler_params = {
        "features": FEATURE_COLS,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    with open(OUT_DIR / "scaler_params.json", "w") as f:
        json.dump(scaler_params, f, indent=2)

    with open(OUT_DIR / "feature_names.txt", "w") as f:
        for name in FEATURE_COLS:
            f.write(name + "\n")

    # --- Summary ----------------------------------------------------------- #
    print("\n=== Preprocessing summary ===")
    print(f"  initial rows (all discharge cycles): {drop_summary['initial_rows']}")
    print(
        f"  dropped: {drop_summary['dropped_null_capacity']} rows with null capacity"
    )
    print(
        f"  dropped: {drop_summary['dropped_out_of_range']} rows with capacity "
        f"outside [{CAPACITY_MIN}, {CAPACITY_MAX}] Ah"
    )
    print(
        f"  dropped: {drop_summary['dropped_feature_nulls']} rows with null feature values"
    )
    print(f"  final clean rows: {drop_summary['final_rows']}")
    print()
    print(f"  train: {len(X_train)} rows")
    print(f"  val:   {len(X_val)} rows")
    print(f"  test:  {len(X_test)} rows")
    print()
    print(f"  wrote {OUT_DIR}/X_{{train,val,test}}.csv")
    print(f"  wrote {OUT_DIR}/y_{{train,val,test}}.csv")
    print(f"  wrote {OUT_DIR}/scaler_params.json")
    print(f"  wrote {OUT_DIR}/feature_names.txt")


if __name__ == "__main__":
    main()
