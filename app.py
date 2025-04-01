import streamlit as st
import pandas as pd
from datetime import datetime
from scripts import forecast_pipeline as fp
import plotly.express as px

st.title("Cafe Sales Dashboard")

# Sidebar for user inputs
st.sidebar.header("User Inputs")
date = st.sidebar.date_input("Select Forecast Start Date", datetime(2024, 12, 1))
ped_start = st.sidebar.date_input("Pedestrianization Start Date", datetime(2025, 6, 1))
ped_end = st.sidebar.date_input("Pedestrianization End Date", datetime(2025, 6, 10))
closed_dates = st.sidebar.text_input("Enter Closed Dates (comma separated, YYYY-MM-DD)", "")

# Process closed dates input
if closed_dates:
    closed_dates = [date.strip() for date in closed_dates.split(",")]
else:
    closed_dates = None

run_forecast = st.button("Run Forecast")

if run_forecast:
    with st.spinner("Running forecast, please wait..."):
        try:
            # Create input DataFrame using your pipeline
            data = fp.forecast_pipe(date.strftime("%Y-%m-%d"), 
                                ped_start.strftime("%Y-%m-%d"), 
                                ped_end.strftime("%Y-%m-%d"), 
                                closed_dates=closed_dates)

            # Load models and get predictions for 3 categories
            predictions_cat1 = fp.load_sales_model_and_forecast("sales_models/xgb_model_Coffee.pkl", data, date)
            predictions_cat2 = fp.load_sales_model_and_forecast("sales_models/xgb_model_Without_Coffee.pkl", data, date)
            predictions_cat3 = fp.load_sales_model_and_forecast("sales_models/xgb_model_Food.pkl", data, date)

            # Display the predictions
            st.subheader("Coffee Sales Predictions")
            col1, col2 = st.columns([2,1])
            with col1:
                fig1 = px.line(pd.DataFrame(predictions_cat1, columns=["sales"]), markers=True)
                fig1.update_layout(xaxis_title="Date", yaxis_title="Coffee Sales")
                st.plotly_chart(fig1)
            with col2:
               st.dataframe(pd.DataFrame(predictions_cat1, columns=["sales"]), width = 200)

            st.subheader("Without Coffee Beverages Sales Predictions")
            col1, col2 = st.columns([2,1])
            with col1:
                fig2 = px.line(pd.DataFrame(predictions_cat2, columns=["sales"]), markers=True)
                fig2.update_layout(xaxis_title="Date", yaxis_title="Without Coffee Beverages Sales")
                st.plotly_chart(fig2)
            with col2:
                st.dataframe(pd.DataFrame(predictions_cat2, columns=["sales"]), width = 200)

            st.subheader("Food Sales Predictions")
            col1, col2 = st.columns([2,1])
            with col1:
                fig3 = px.line(pd.DataFrame(predictions_cat3, columns=["sales"]), markers=True)
                fig3.update_layout(xaxis_title="Date", yaxis_title="Food Sales")
                st.plotly_chart(fig3)
            with col2:
                st.dataframe(pd.DataFrame(predictions_cat3, columns=["sales"]), width = 200)
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("Click the 'Run Forecast' button to generate predictions.")
