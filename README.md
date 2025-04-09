# Cafe sales forecasting dashboard

**Machine learning-powered dashboard for forecasting category-level product sales using real-world cafe data, external signals (ex. macroeconomic indicators, weather data), and Streamlit interface.**

---

## Live Demo

Interact with the full dashboard here: [Streamlit Cloud Link]()

Interact with:
- 10-day sales forecasts
- Product category trends
- Main metrics for each category and total predictions
- Simple visuals for Coffee, Non-Coffee Drinks, and Food items

---

## ⚠️ Forecast Limitations & Usage Notes

This dashboard was built using in a small set of real historical data from my time operating a café, with machine learning models trained on data from `October 1, 2023` up to `November 30, 2024`. Unfortunately, after transitioning to a less involved role in the business the dashboard hasn't been used with live data yet.

You can explore forecasts for future 10 day periods that fall within this **limited testing window**:
> **December 1, 2024 to February 1, 2025**

These dates were chosen because:
- They simulate a realistic future forecasting period
- The models have *not seen this future data*, making it a fair demonstration of performance

Please note: **Entering dates beyond February 1, 2025 will result in no predictions**, as there is no valid feature data beyond that point.

This version is designed as a showcase of modeling + deployment with real world data—not a live production tool.

---

## Demo Notebook

Open `forecast.ipynb` to test model predictions on a cleaned, pre-engineered dataset.

It will:
- Load the trained models (`.pkl`)
- Load a future date feature set (Dec 1, 2024 – Feb 1, 2025)
- Output forecasted sales per category
- Display graphs + metrics

No training needed.

---

## Project Summary

This project started during my time operating a café-restaurant in Montréal with the goal of predicting sales in key product categories (Coffee, Non-coffee Beverages, Food) to support smarter invetory and prep decisions. It is also a self-improvement challenge.

---

## Features

- **3 seperate models** (one per product category)
- **10-day forecasts** based on historical trends + external signals
- **API usage** for feature engineering
- **Streamlit dashboard** for non-technical users
- **Modular pipeline** with scripts and reusable functions

---

## Model performance summary

Each product category has its own ML model, trained and evaluated on historical café data.

### 📏 Model Evaluation Summary

XGBoost

| Category      | Train RMSE | Train MAPE | Train R² | Train Acc@20 | Train Acc@50 | Train Acc@100  | Test RMSE | Test MAPE | Test R² | CV RMSE | Test Acc@20 | Test Acc@50| Test Acc@100 |
|---------------|------------|------------|----------|--------------|--------------|----------------|-----------|-----------|---------|---------|-------------|------------|--------------|
| Coffee        | 60.86      | 9.68       | 0.89     | 29.9%        | 62.1%        | 90.5%          | 86.98     | 13.47     | 0.63    | 100.68  | 21.4%       | 46.4%      | 71.4%        |
| Without Coffee| 30.35      | 17.97      | 0.75     | 54.2%        | 90.5%        | 99.2%          | 39.48     | 22.95     | 0.60    | 48.36   | 35.7%       | 82.1%      | 100.0%       |
| Food          | 70.60      | 16.06      | 0.76     | 28.4%        | 57.8%        | 84.7%          | 73.64     | 19.62     | 0.75    | 82.92   | 10.7%       | 46.4%      | 89.3%        |

MAPE (mean absolute percentage error)
Acc@20 (percentage of predictions falling within 20$ range)
Acc@50 (percentage of predictions falling within 50$ range)
Acc@100 (percentage of predictions falling within 100$ range)


These metrics reflect out-of-sample performance, using a train-test split from the original dataset (up to Nov 30, 2024).

📓 **Full model training process and evaluation** available in:  
[`cafe_sales_analysis.ipynb`](cafe_sales_analysis.ipynb)

## Tech Stack

- `Python 3.13`, `pandas`, `scikit-learn`, `XGBoost`, `Ridge`, `GridSearchCV`, `TimeSeriesSplit`
- `Matplotlib`, `Seaborn`, `Plotly` for visuals
- `Streamlit` for interactive UI
- `meteostat`, `FRED`, `calendarific` APIs for weather, macroeconomic, holiday data fetching
- `Pickle`, `dill` for model storage

---

## Potential Improvements

- Add inventory optimization logic based on predicted sales.
- Auto-retraining on new weekly data (current sample size is very small).
- Add LLM-generated report summaries.
- Transition to agentic nature potentially giving alerts and suggestion for invetory management.

---



Vasilis Christopoulos