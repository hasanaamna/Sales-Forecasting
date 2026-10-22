
from pathlib import Path
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = RAW_DATA_DIR / "train.csv"


def main():
    if not TRAIN_FILE.exists():
        print(f"ERROR: Dataset not found at:\n{TRAIN_FILE}")
        print("Download and extract the Kaggle dataset into data/raw/.")
        return

    print("=" * 65)
    print("SALES FORECASTING - DATASET EXPLORATION")
    print("=" * 65)

    # Load historical sales
    df = pd.read_csv(TRAIN_FILE, parse_dates=["date"])

    print("\n1. DATASET SHAPE")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    print("\n2. COLUMN NAMES")
    print(df.columns.tolist())

    print("\n3. FIRST FIVE ROWS")
    print(df.head().to_string(index=False))

    print("\n4. DATA TYPES")
    print(df.dtypes.to_string())

    print("\n5. MISSING VALUES")
    missing = df.isnull().sum().sort_values(ascending=False)
    print(missing[missing > 0] if missing.any() else "No missing values.")

    print("\n6. DUPLICATE ROWS")
    print(f"Duplicate rows: {df.duplicated().sum():,}")

    print("\n7. DATE RANGE")
    print(f"First date: {df['date'].min()}")
    print(f"Last date:  {df['date'].max()}")

    print("\n8. SALES SUMMARY")
    print(df["sales"].describe().round(2).to_string())

    print("\n9. UNIQUE STORES AND PRODUCT FAMILIES")
    print(f"Stores: {df['store_nbr'].nunique():,}")
    print(f"Product families: {df['family'].nunique():,}")

    # Aggregate sales by date
    daily_sales = (
        df.groupby("date", as_index=False)["sales"]
        .sum()
        .sort_values("date")
    )

    daily_sales.to_csv(
        REPORT_DIR / "daily_sales.csv",
        index=False
    )

    # Aggregate sales by month
    monthly_sales = (
        df.set_index("date")["sales"]
        .resample("MS")
        .sum()
        .reset_index()
    )

    monthly_sales.to_csv(
        REPORT_DIR / "monthly_sales.csv",
        index=False
    )

    # Sales by product family
    family_sales = (
        df.groupby("family", as_index=False)["sales"]
        .sum()
        .sort_values("sales", ascending=False)
    )

    family_sales.to_csv(
        REPORT_DIR / "sales_by_family.csv",
        index=False
    )

    print("\n10. TOP 10 PRODUCT FAMILIES BY TOTAL SALES")
    print(family_sales.head(10).to_string(index=False))

    print("\n11. REPORT FILES CREATED")
    print(f"Daily sales:   {REPORT_DIR / 'daily_sales.csv'}")
    print(f"Monthly sales: {REPORT_DIR / 'monthly_sales.csv'}")
    print(f"Family sales:  {REPORT_DIR / 'sales_by_family.csv'}")

    print("\nDataset inspection completed!")


if __name__ == "__main__":
    main()
