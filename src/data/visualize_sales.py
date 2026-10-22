
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PROJECT PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = RAW_DIR / "train.csv"


def main():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {TRAIN_FILE}")

    print("Loading sales dataset...")
    df = pd.read_csv(TRAIN_FILE, parse_dates=["date"])

    # ========================================================
    # 1. DAILY AND MONTHLY SALES
    # ========================================================
    daily_sales = (
        df.groupby("date")["sales"]
        .sum()
        .sort_index()
    )

    monthly_sales = daily_sales.resample("MS").sum()

    # Daily sales plot
    plt.figure(figsize=(14, 6))
    plt.plot(daily_sales.index, daily_sales.values, linewidth=0.8)
    plt.title("Daily Total Sales Over Time")
    plt.xlabel("Date")
    plt.ylabel("Total Sales")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "daily_sales_trend.png", dpi=150)
    plt.close()

    # Monthly sales plot
    plt.figure(figsize=(14, 6))
    plt.plot(
        monthly_sales.index,
        monthly_sales.values,
        marker="o",
        markersize=3,
        linewidth=1.5,
    )
    plt.title("Monthly Total Sales Over Time")
    plt.xlabel("Month")
    plt.ylabel("Total Sales")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "monthly_sales_trend.png", dpi=150)
    plt.close()

    # ========================================================
    # 2. AVERAGE SALES BY DAY OF WEEK
    # ========================================================
    df["day_of_week"] = df["date"].dt.dayofweek

    weekday_names = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
    ]

    weekday_sales = (
        df.groupby("day_of_week")["sales"]
        .mean()
        .reindex(range(7))
    )

    plt.figure(figsize=(10, 5))
    plt.bar(weekday_names, weekday_sales.values)
    plt.title("Average Sales by Day of Week")
    plt.xlabel("Day")
    plt.ylabel("Average Sales per Store and Product Family")
    plt.xticks(rotation=25)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "weekday_sales.png", dpi=150)
    plt.close()

    # ========================================================
    # 3. TOP 10 PRODUCT FAMILIES
    # ========================================================
    family_sales = (
        df.groupby("family")["sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .sort_values()
    )

    plt.figure(figsize=(11, 6))
    plt.barh(family_sales.index, family_sales.values)
    plt.title("Top 10 Product Families by Total Sales")
    plt.xlabel("Total Sales")
    plt.ylabel("Product Family")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "top_product_families.png", dpi=150)
    plt.close()

    # ========================================================
    # 4. SALES QUALITY CHECKS
    # ========================================================
    zero_sales_count = (df["sales"] == 0).sum()
    zero_sales_percentage = (df["sales"] == 0).mean() * 100

    print("\nSALES SUMMARY")
    print(f"Total records: {len(df):,}")
    print(f"Total daily dates: {daily_sales.size:,}")
    print(f"Total monthly periods: {monthly_sales.size:,}")
    print(f"Total sales across dataset: {df['sales'].sum():,.2f}")

    print("\nZERO-SALES CHECK")
    print(f"Zero-sales records: {zero_sales_count:,}")
    print(f"Zero-sales percentage: {zero_sales_percentage:.2f}%")

    print("\nAVERAGE SALES BY DAY OF WEEK")
    print(weekday_sales.rename(index=dict(enumerate(weekday_names))).round(2))

    print("\nMONTHLY SALES SUMMARY")
    print(monthly_sales.describe().round(2))

    print("\nCharts saved to:")
    print(REPORT_DIR)

    print("\nVisualization completed successfully!")


if __name__ == "__main__":
    main()
