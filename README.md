# Cafe sales forecasting dashboard

**Machine learning-powered dashboard for forecasting category-level product sales using real-world cafe data, external signals (ex. macroeconomic indicators, weather data), and Streamlit interface.**

---

## Live Demo

Interact with the full dashboard here: [Streamlit Cloud Link](https://cafe-sales-forecasting-rwxbrpaed944zibn4zdtfc.streamlit.app/)

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

Here is a detailed step by step on how to do this:
**Step 0: Make sure that you have python 3.13 installed on your computer**

```bash
# Step 1: Clone the repository to your local machine's desktop
cd ~/Desktop
git clone repo_link

# Step 2: Change directory into the cloned repository
cd repo_name

# Step 3: create the .env file to load your api keys
cp .env.example .env
```
**Step 4:**<br>
On macOS:

Open Finder
Press Cmd + Shift + . to reveal hidden files

On Windows:

Open File Explorer
Press Ctrl + H or go to View > Show > Hidden items

Open the newly created .env file and input your [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html) and your [Calendarific API](https://calendarific.com/api-documentation) key.
```bash
CALENDARIFIC_API_KEY=your_real_key_here
WEATHER_API_KEY=your_real_key_here
```

**Back in the terminal:**
```bash
# Step 5: Create a virtual environment using python 3.13 and activate it 
python3.13 -m venv testenv
source testenv/bin/activate # Windows: source testenv\Scripts\activate

# Step 6: Install all required dependencies
pip install -r requirements.txt

# Step 7: Open jupyter notebook
jupyter notebook
```

**Step 8: Navigate to the jupyter notebook localhost tab and open the forecast.ipynb notebook.**

**Step 9: You are all set you can run the cells.**

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

| Category        | Train RMSE | Train MAPE | Train R2  | Train Accuracy 20 | Train Accuracy 50 | Train Accuracy 100 | Test RMSE | Test MAPE | Test R2   | CV RMSE   | Test Accuracy 20 | Test Accuracy 50 | Test Accuracy 100 |
|-----------------|------------|------------|-----------|-------------------|-------------------|--------------------|-----------|-----------|-----------|-----------|------------------|------------------|-------------------|
| Coffee          | 75.762664  | 12.169980  | 0.832161  | 26.342711         | 54.219949         | 80.818414          | 66.751111 | 10.267258 | 0.783905  | 87.792579 | 10.714286        | 50.000000        | 85.714286         |
| Without Coffee  | 31.499437  | 18.898585  | 0.725800  | 53.708440         | 88.491049         | 99.232737          | 39.052506 | 22.003901 | 0.604147  | 46.283391 | 42.857143        | 75.000000        | 100.000000        |
| Food            | 53.874728  | 12.250084  | 0.860050  | 37.340153         | 69.309463         | 92.583120          | 65.837330 | 16.490270 | 0.798241  | 91.087567 | 32.142857        | 53.571429        | 85.714286         |

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

## License

This project is licensed under the **MIT License** — feel free to use, modify, and share it.  
See the [LICENSE](LICENSE) file for details.

Vasilis Christopoulos
