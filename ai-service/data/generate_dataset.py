import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_dataset(n_rows: int = 4000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    days_until_expiry = rng.integers(1, 366, size=n_rows)
    skew_mask = rng.random(n_rows) < 0.6
    days_until_expiry[skew_mask] = rng.integers(1, 91, size=skew_mask.sum())

    avg_monthly_sales = rng.poisson(lam=6, size=n_rows)
    avg_monthly_sales = np.clip(avg_monthly_sales, 0, 50)

    recent_sales = np.round(avg_monthly_sales * rng.normal(1.0, 0.3, size=n_rows)).astype(int)
    recent_sales = np.clip(recent_sales, 0, 60)

    sales_rate = 0.7 * avg_monthly_sales + 0.3 * recent_sales
    expected_consumption = sales_rate * (days_until_expiry / 30.0)

    stock_multiplier = rng.lognormal(mean=0.3, sigma=0.7, size=n_rows)
    current_stock = np.round(expected_consumption * stock_multiplier).astype(int)
    current_stock = np.clip(current_stock, 0, 500)

    risk = np.full(n_rows, "LOW", dtype=object)
    high_mask = current_stock > 2 * expected_consumption
    medium_mask = (current_stock > expected_consumption) & (~high_mask)
    risk[high_mask] = "HIGH"
    risk[medium_mask] = "MEDIUM"

    boundary_high = np.abs(current_stock - 2 * expected_consumption) / (expected_consumption + 1e-6)
    boundary_low = np.abs(current_stock - expected_consumption) / (expected_consumption + 1e-6)

    near_high = (boundary_high < 0.25) & (risk != "LOW")
    near_low = (boundary_low < 0.25) & (risk != "HIGH")

    flip_prob_high = 0.35
    flip_prob_low = 0.30

    flip_high = near_high & (rng.random(n_rows) < flip_prob_high)
    flip_low = near_low & (rng.random(n_rows) < flip_prob_low)

    risk[flip_high & (risk == "HIGH")] = "MEDIUM"
    risk[flip_high & (risk == "MEDIUM")] = "HIGH"
    risk[flip_low & (risk == "LOW")] = "MEDIUM"
    risk[flip_low & (risk == "MEDIUM")] = "LOW"

    df = pd.DataFrame({
        "currentStock": current_stock,
        "averageMonthlySales": avg_monthly_sales,
        "recentSales": recent_sales,
        "daysUntilExpiry": days_until_expiry,
        "risk": risk
    })

    assert df.notnull().all().all(), "Null values found"
    assert (df.select_dtypes(include=[np.number]) >= 0).all().all(), "Negative values found"
    assert set(df["risk"].unique()).issubset({"LOW", "MEDIUM", "HIGH"}), "Invalid risk labels"
    assert list(df.columns) == [
        "currentStock", "averageMonthlySales", "recentSales", "daysUntilExpiry", "risk"
    ], "Column order mismatch"

    return df


def print_report(df: pd.DataFrame):
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    print("\nClass counts and percentages:")
    counts = df["risk"].value_counts()
    for cls in ["LOW", "MEDIUM", "HIGH"]:
        cnt = counts.get(cls, 0)
        pct = 100 * cnt / len(df)
        print(f"  {cls}: {cnt} ({pct:.1f}%)")

    print("\nMin/max for numeric columns:")
    for col in ["currentStock", "averageMonthlySales", "recentSales", "daysUntilExpiry"]:
        print(f"  {col}: min={df[col].min()}, max={df[col].max()}")

    sales_rate = 0.7 * df["averageMonthlySales"] + 0.3 * df["recentSales"]
    expected = sales_rate * (df["daysUntilExpiry"] / 30.0)
    rule_risk = np.full(len(df), "LOW", dtype=object)
    rule_risk[df["currentStock"] > 2 * expected] = "HIGH"
    rule_risk[(df["currentStock"] > expected) & (df["currentStock"] <= 2 * expected)] = "MEDIUM"
    flipped = (rule_risk != df["risk"]).sum()
    print(f"\nRows with label flipped by noise: {flipped} ({100 * flipped / len(df):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic inventory risk dataset")
    parser.add_argument("--rows", type=int, default=4000, help="Number of rows to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="synthetic_inventory_risk.csv", help="Output CSV path")
    args = parser.parse_args()

    df = generate_dataset(args.rows, args.seed)
    output_path = Path(__file__).resolve().parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    print_report(df)


if __name__ == "__main__":
    main()