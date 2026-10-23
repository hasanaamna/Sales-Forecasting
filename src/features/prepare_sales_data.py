
from pathlib import Path

import pandas as pd

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = RAW_DIR / "train.csv"
HOLIDAYS_FILE = RAW_DIR / "holidays_events.csv"
TRANSACTIONS_FILE = RAW_DIR / "transactions.csv"


def main():
    # --------------------------------------------------------
    # 1. LOAD TRAINING DATA
    # --------------------------------------------------------
    for file in [TRAIN_FILE, HOLIDAYS_FILE, TRANSACTIONS_FILE]:
        if not file.exists():
            raise FileNotFoundError(f"Required file not found: {file}")

    print("Loading training data...")
    train = pd.read_csv(TRAIN_FILE, parse_dates=["date"])

    # Daily total sales and promotion counts
    daily = (
        train.groupby("date")
        .agg(
            sales=("sales", "sum"),
            promotion_count=("onpromotion", "sum"),
        )
        .sort_index()
    )

    daily.index = pd.DatetimeIndex(daily.index, name="date")
    daily = daily.asfreq("D")

    # Explicit assumption used in the earlier analysis:
    # the four missing Christmas dates represent zero recorded sales.
    missing_dates = daily.index[daily["sales"].isna()]

    if len(missing_dates):
        holidays_check = pd.read_csv(
            HOLIDAYS_FILE, parse_dates=["date"]
        )
        holiday_dates = set(holidays_check["date"].dt.normalize())

        for date in missing_dates:
            if date.normalize() not in holiday_dates:
                raise ValueError(
                    f"Missing sales on {date.date()} without "
                    "a matching holiday record. Investigate first."
                )
            if not (date.month == 12 and date.day == 25):
                raise ValueError(
                    f"Unexpected missing date: {date.date()}"
                )

        daily.loc[missing_dates, "sales"] = 0.0
        # Promotion counts remain unknown for missing dates.
        # They are filled below only after checking source coverage.

    if daily["sales"].isna().any():
        raise ValueError("Daily sales still contain missing values.")

    # Missing promotion totals are not assumed to be zero
    # without checking source coverage.
    # Inspect missing promotion totals
    missing_promotions = daily.index[
        daily["promotion_count"].isna()
    ]

    print(
        f"\nDates with missing promotion totals: "
        f"{len(missing_promotions)}"
    )

    if len(missing_promotions):
        print(
            "Missing promotion dates:",
            [d.strftime("%Y-%m-%d") for d in missing_promotions]
        )

        # These are dates without source training records.
        # Keep promotion counts unknown rather than assuming zero.
        daily["promotion_missing"] = (
            daily["promotion_count"].isna().astype(int)
        )

        # Use zero as a placeholder only alongside a missingness flag.
        daily["promotion_count"] = daily["promotion_count"].fillna(0)
    else:
        daily["promotion_missing"] = 0

    # --------------------------------------------------------
    # 2. HOLIDAY FEATURES
    # --------------------------------------------------------
    holidays = pd.read_csv(
        HOLIDAYS_FILE,
        parse_dates=["date"],
    )
    holidays["date"] = holidays["date"].dt.normalize()

    # Include actual holidays/events; exclude transferred entries
    # and entries marked as additional days off.
    valid_holidays = holidays[
        (~holidays["type"].isin(["Transfer", "Additional"]))
        & (holidays["transferred"].fillna(False) != True)
    ].copy()

    # Conservative date-level flag: a holiday/event appears
    # somewhere in the supplied holiday calendar on this date.
    holiday_flag = (
        valid_holidays.groupby("date")
        .size()
        .gt(0)
        .astype(int)
        .rename("is_holiday")
    )

    daily = daily.join(holiday_flag, how="left")
    daily["is_holiday"] = daily["is_holiday"].fillna(0).astype(int)

    # --------------------------------------------------------
    # 3. TRANSACTION FEATURES
    # --------------------------------------------------------
    transactions = pd.read_csv(
        TRANSACTIONS_FILE,
        parse_dates=["date"],
    )

    daily_transactions = (
        transactions.groupby("date")["transactions"]
        .sum()
        .sort_index()
        .rename("transactions")
    )

    # Use only past transaction counts as model inputs.
    # Missing transaction dates remain missing until handled.
    daily = daily.join(daily_transactions, how="left")
    daily["transactions_lag_1"] = daily["transactions"].shift(1)
    daily["transactions_lag_7"] = daily["transactions"].shift(7)

    # Do not use contemporaneous or future transactions as predictors.
    daily = daily.drop(columns=["transactions"])

    # --------------------------------------------------------
    # 4. CALENDAR FEATURES
    # --------------------------------------------------------
    daily["day_of_week"] = daily.index.dayofweek
    daily["day_of_month"] = daily.index.day
    daily["month"] = daily.index.month
    daily["year"] = daily.index.year
    daily["day_of_year"] = daily.index.dayofyear
    daily["is_weekend"] = (
        daily["day_of_week"] >= 5
    ).astype(int)

    # --------------------------------------------------------
    # 5. SALES LAGS AND ROLLING FEATURES
    # --------------------------------------------------------
    for lag in [1, 7, 14, 28]:
        daily[f"sales_lag_{lag}"] = daily["sales"].shift(lag)

    daily["sales_rolling_mean_7"] = (
        daily["sales"].shift(1).rolling(7).mean()
    )
    daily["sales_rolling_mean_28"] = (
        daily["sales"].shift(1).rolling(28).mean()
    )

    # --------------------------------------------------------
    # 6. SAVE
    # --------------------------------------------------------
    daily.to_csv(
        PROCESSED_DIR / "daily_sales_features.csv",
        index_label="date",
    )

    daily[["sales"]].to_csv(
        PROCESSED_DIR / "daily_sales_target.csv",
        index_label="date",
    )

    print("\nFeature preparation completed.")
    print(f"Observations: {len(daily):,}")
    print(f"Date range: {daily.index.min().date()} to "
          f"{daily.index.max().date()}")
    print(f"Holiday dates flagged: {daily['is_holiday'].sum():,}")
    print(f"Promotion feature total: {daily['promotion_count'].sum():,}")
    print(
        "Missing lagged transaction values:",
        int(daily["transactions_lag_1"].isna().sum()),
    )
    print(f"Saved features: {PROCESSED_DIR / 'daily_sales_features.csv'}")


if __name__ == "__main__":
    main()
