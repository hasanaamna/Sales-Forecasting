# Sales Forecasting Using Machine Learning

A machine learning project to analyze historical retail sales, identify trends and seasonality, and forecast future sales using time-series analysis and predictive modeling.

## Project Overview

Accurate sales forecasting helps businesses make better decisions about inventory management, demand planning, staffing, and revenue targets.

This project uses historical retail sales data to explore sales patterns and develop forecasting models. It will also include an interactive dashboard to visualize historical performance and predicted sales.

## Objectives

- Analyze historical sales trends and patterns.
- Identify seasonal effects and weekly sales variations.
- Examine sales performance across stores and product families.
- Investigate promotional effects on sales.
- Develop and compare sales forecasting models.
- Evaluate forecasting accuracy using appropriate metrics.
- Build an interactive dashboard for exploring predictions.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn, XGBoost |
| Time-Series Forecasting | Statistical baselines and forecasting models |
| Backend (Planned) | FastAPI |
| Frontend (Planned) | React, TypeScript |
| Development Tools | VS Code, Jupyter Notebook |
| Version Control | Git, GitHub |

## Dataset

**Dataset:** Store Sales — Time Series Forecasting

**Source:** [Kaggle Competition Dataset](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)

The dataset contains historical retail sales records from multiple stores and product families. Additional files provide information about stores, promotions, oil prices, holidays, and transactions.

### Dataset Files

| File | Description |
|---|---|
| `train.csv` | Historical sales observations |
| `test.csv` | Records for which sales must be forecast |
| `stores.csv` | Store information |
| `oil.csv` | Daily oil prices |
| `holidays_events.csv` | Holiday and event information |
| `transactions.csv` | Daily transaction counts by store |
| `sample_submission.csv` | Example competition submission format |

The dataset is not included in this repository because of its size. Download it from Kaggle and place the extracted CSV files in `data/raw/`.

## Project Structure

```text
Sales-Forecasting/
│
├── app/
│   ├── backend/             # Planned FastAPI application
│   └── frontend/            # Planned React application
│
├── data/
│   ├── raw/                 # Original dataset (not tracked)
│   └── processed/           # Processed data (not tracked)
│
├── models/                  # Saved trained models (not tracked)
│
├── notebooks/               # EDA and experiments
│
├── reports/                 # Analysis outputs and charts
│
├── src/
│   ├── data/
│   │   └── inspect_dataset.py
│   ├── features/             # Planned feature engineering
│   ├── training/             # Planned model training
│   └── evaluation/           # Planned model evaluation
│
├── tests/                    # Planned automated tests
├── .gitignore
├── requirements.txt
└── README.md
```

## Current Progress

### Completed

- [x] Project directory structure created.
- [x] Python virtual environment configured.
- [x] Required initial Python libraries installed.
- [x] Git ignore rules configured.
- [x] Kaggle dataset downloaded and extracted.
- [x] Dataset inspection script implemented.
- [x] Initial data quality checks completed.
- [x] Daily, monthly, and product-family sales summary CSV files generated.

### In Progress

- [ ] Exploratory data analysis and visualization.
- [ ] Sales trend and seasonality analysis.
- [ ] Feature engineering.
- [ ] Baseline forecasting model.
- [ ] Advanced model training and comparison.
- [ ] Model evaluation and error analysis.
- [ ] FastAPI backend.
- [ ] React dashboard.
- [ ] Automated tests and deployment.

## Initial Data Exploration

The initial inspection of `train.csv` produced the following results:

| Metric | Result |
|---|---:|
| Number of records | 3,000,888 |
| Number of columns | 6 |
| Unique stores | 54 |
| Product families | 33 |
| First recorded date | January 1, 2013 |
| Last recorded date | August 15, 2017 |
| Missing values | 0 |
| Exact duplicate rows | 0 |
| Mean sales per record | 357.78 |
| Median sales per record | 11.00 |
| Maximum sales per record | 124,717.00 |

The dataset contains substantial variation in sales. Further analysis is required to understand zero-sales records, outliers, promotional effects, and seasonal patterns before selecting the final forecasting model.

### Highest-Selling Product Families

The initial aggregation identified the following leading categories by total recorded sales:

1. Grocery I
2. Beverages
3. Produce
4. Cleaning
5. Dairy
6. Bread/Bakery
7. Poultry
8. Meats
9. Personal Care
10. Deli

These rankings are based on total sales across the available historical records, not profitability.

## Planned Forecasting Workflow

1. **Data validation:** Verify date coverage, record completeness, and unusual observations.
2. **Exploratory analysis:** Visualize daily and monthly sales and identify recurring patterns.
3. **Feature engineering:** Generate calendar features, lagged sales, rolling statistics, and relevant external features.
4. **Baseline modeling:** Establish a benchmark using simple historical or seasonal forecasts.
5. **Model development:** Train and compare suitable statistical and machine learning models.
6. **Evaluation:** Use chronological validation and metrics such as MAE, RMSE, and WAPE.
7. **Forecast generation:** Produce predictions for a defined future horizon.
8. **Application development:** Build a FastAPI service and React-based dashboard.
9. **Testing and documentation:** Validate the pipeline and document reproducible instructions.

## Evaluation Metrics

| Metric | Purpose |
|---|---|
| MAE | Measures average absolute prediction error. |
| RMSE | Penalizes larger prediction errors more heavily. |
| WAPE | Measures total absolute error relative to total actual sales. |

Model selection will be based on validation results rather than assumptions about which algorithm is best. Time-series data will be split chronologically to avoid using future information during training.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/hasanaamna/Sales-Forecasting
cd Sales-Forecasting
```



### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

If you use Command Prompt instead:

```cmd
venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download the dataset

Download the files from the Kaggle competition page and extract them into:

```text
data/raw/
```

Confirm that `data/raw/train.csv` exists before running the inspection script.

### 5. Inspect the dataset

```bash
python src/data/inspect_dataset.py
```

The script generates the following summaries in `reports/`:

- `daily_sales.csv`
- `monthly_sales.csv`
- `sales_by_family.csv`

## Future Enhancements

- Interactive sales forecasting dashboard.
- Store-level and product-family-level forecasting.
- Promotion and holiday impact analysis.
- Forecast horizon selection.
- Actual-versus-predicted sales comparisons.
- Downloadable forecasts and evaluation reports.
- Model persistence and API integration.

## Limitations

- Forecasting accuracy has not yet been established.
- The final model and forecast horizon are still to be determined.
- Future performance will depend on data quality, feature availability, and changing sales patterns.
- The initial results describe the training dataset and do not demonstrate predictive performance.

## License and Data Attribution

This project is intended for educational and portfolio purposes. Refer to the Kaggle competition page and its applicable terms for dataset access, attribution, and usage conditions.

## Author

**Aamna Hasan**

GitHub: [hasanaamna](https://github.com/hasanaamna)

---

*This project is under active development. Model results, dashboard screenshots, and deployment instructions will be added as implementation progresses.*
