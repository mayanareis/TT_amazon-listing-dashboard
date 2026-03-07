"""Validate cleaned Amazon product data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEANED_FILE = PROJECT_ROOT / "data" / "cleaned" / "amazon_products_clean.csv"


def main() -> None:
    if not CLEANED_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned file not found at {CLEANED_FILE}. Run scripts/clean_data.py first."
        )

    df = pd.read_csv(CLEANED_FILE)

    print(f"Row count: {len(df):,}")
    if "asin" in df.columns:
        print(f"Unique ASIN count: {df['asin'].nunique(dropna=True):,}")
    else:
        print("Unique ASIN count: column 'asin' not present")

    print("\nDescriptive summaries:")
    for col in ["stars", "reviewsCount", "price_value"]:
        if col in df.columns:
            print(f"\n{col}:")
            print(df[col].describe().to_string())
        else:
            print(f"\n{col}: column not present")

    print("\nMissing values:")
    print(df.isna().sum().sort_values(ascending=False).to_string())

    suspicious_mask = pd.Series(False, index=df.index)

    if "stars" in df.columns:
        suspicious_mask |= df["stars"].notna() & ((df["stars"] < 0) | (df["stars"] > 5))
    if "price_value" in df.columns:
        suspicious_mask |= df["price_value"].notna() & (df["price_value"] <= 0)
    if "reviewsCount" in df.columns:
        suspicious_mask |= df["reviewsCount"].notna() & (df["reviewsCount"] < 0)

    suspicious_df = df[suspicious_mask]
    print(f"\nSuspicious rows count: {len(suspicious_df):,}")
    if not suspicious_df.empty:
        sample_cols = [
            col
            for col in ["asin", "title", "brand", "stars", "reviewsCount", "price_value", "url"]
            if col in suspicious_df.columns
        ]
        print("\nSuspicious rows sample:")
        print(suspicious_df[sample_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
