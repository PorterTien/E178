"""
EDA for NASA Li-ion battery capacity degradation dataset.

Loads the 4 canonical batteries (B0005, B0006, B0007, B0018) from the NASA
Battery Dataset, aggregates each discharge cycle into a single row of features,
and saves exploratory plots (degradation curves + feature histograms) to
figures/.

Run:
    python eda.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# The prompt assumes a ``raw_data/`` folder. This repo ships the NASA data in
# ``cleaned_dataset/`` (metadata.csv + one CSV per cycle in data/). We fall back
# to whichever exists so the script works either way.
SCRIPT_DIR = Path(__file__).resolve().parent
CANDIDATE_DIRS = [SCRIPT_DIR / "raw_data", SCRIPT_DIR / "cleaned_dataset"]
DATA_ROOT = next((p for p in CANDIDATE_DIRS if p.exists()), CANDIDATE_DIRS[0])
METADATA_CSV = DATA_ROOT / "metadata.csv"
CYCLES_DIR = DATA_ROOT / "data"

FIGURES_DIR = SCRIPT_DIR / "figures"

# The 7 per-cycle features we care about.
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


# --------------------------------------------------------------------------- #
# Loading + aggregation
# --------------------------------------------------------------------------- #

def load_discharge_metadata(battery_ids: list[str]) -> pd.DataFrame:
    """Return the metadata rows for discharge cycles of the chosen batteries."""
    meta = pd.read_csv(METADATA_CSV)
    mask = (meta["type"] == "discharge") & (meta["battery_id"].isin(battery_ids))
    meta = meta.loc[mask].copy()

    # Numeric capacity; anything unparseable becomes NaN (handled later).
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
    """Collapse one cycle's timestep CSV into mean/min/max summary statistics."""
    try:
        df = pd.read_csv(cycle_csv)
    except Exception as exc:  # pragma: no cover - defensive
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
    """Build one row per discharge cycle with the 7 features + capacity target.

    Cycle numbering is assigned per battery, in the order the discharge cycles
    appear in metadata.csv (which is chronological).
    """
    meta = load_discharge_metadata(battery_ids)

    rows: list[dict] = []
    for battery_id, battery_meta in meta.groupby("battery_id", sort=False):
        # Per-battery chronological cycle counter starting at 1.
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
    # Consistent column order: id, features, target.
    return df[["battery_id", *FEATURE_COLS, TARGET_COL]]


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_degradation_curves(df: pd.DataFrame, out_path: Path) -> None:
    """All selected batteries on one chart: capacity vs cycle number."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for battery_id, sub in df.groupby("battery_id"):
        sub = sub.sort_values("cycle_number")
        ax.plot(
            sub["cycle_number"],
            sub[TARGET_COL],
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=battery_id,
        )
    ax.set_xlabel("Cycle number")
    ax.set_ylabel("Capacity (Ah)")
    ax.set_title("Capacity degradation across cycles (all available batteries)")
    ax.grid(alpha=0.3)
    ax.legend(title="Battery")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path.relative_to(SCRIPT_DIR)}")


def plot_feature_histograms(df: pd.DataFrame, out_path: Path) -> None:
    """One histogram per feature, arranged in a grid."""
    n = len(FEATURE_COLS)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, FEATURE_COLS):
        values = df[col].dropna()
        ax.hist(values, bins=40, color="steelblue", edgecolor="white")
        ax.set_title(col)
        ax.grid(alpha=0.3)

    # Hide any unused subplots (if feature count isn't a multiple of ncols).
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Feature distributions (per-cycle aggregates)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.relative_to(SCRIPT_DIR)}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    battery_ids = get_available_battery_ids()

    print(f"Data root:   {DATA_ROOT}")
    print(f"Batteries:   {len(battery_ids)} total")
    print(f"Battery IDs: {', '.join(battery_ids)}")
    print("Building per-cycle dataframe ...")
    df = build_cycle_dataframe(battery_ids)

    # --- Basic stats ------------------------------------------------------- #
    print("\n=== Shape ===")
    print(df.shape)

    print("\n=== Rows per battery ===")
    print(df["battery_id"].value_counts().sort_index().to_string())

    print("\n=== Null counts ===")
    print(df.isna().sum().to_string())

    print("\n=== Describe (features + target) ===")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(df[FEATURE_COLS + [TARGET_COL]].describe())

    # --- Plots ------------------------------------------------------------- #
    print("\nGenerating plots ...")
    plot_degradation_curves(df, FIGURES_DIR / "capacity_vs_cycle.png")
    plot_feature_histograms(df, FIGURES_DIR / "feature_histograms.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
