import streamlit as st
import pandas as pd
from datetime import datetime
from scripts import forecast_pipeline as fp
import plotly.express as px


# Cafe Sales Dashboard Application

# This Streamlit app forecasts cafe sales and displays predictions across different
# sales categories (Coffee Sales, Without Coffee Beverages Sales, and Food Sales).
# It also displays the total sales predictions and the contribution of each category
# to the total sales over a 10-day period using interactive Plotly charts.


# Set the title of the dashboard
st.title("Cafe Sales Dashboard")


# Sidebar Configuration for User Inputs
st.sidebar.header("User Inputs")

# User selects the forecast start date with a default and a maximum value
date = st.sidebar.date_input("Select Forecast Start Date", datetime(2024, 12, 1), max_value=datetime(2025, 1, 21))
print(type(date))

# User selects pedestrianization start and end dates (may impact the forecast model)
ped_start = st.sidebar.date_input("Pedestrianization Start Date", datetime(2025, 6, 1))
ped_end = st.sidebar.date_input("Pedestrianization End Date", datetime(2025, 6, 10))

# Input for dates when the store is closed (comma separated, format: YYYY-MM-DD)
closed_dates = st.sidebar.text_input("Enter Closed Dates (comma separated, YYYY-MM-DD)", "")

# Process the closed dates input: split by comma and strip whitespace, if provided
if closed_dates:
    closed_dates = [date.strip() for date in closed_dates.split(",")]
else:
    closed_dates = None

# Button to trigger running the forecast
run_forecast = st.button("Run Forecast")

# Forecast Execution and Visualization
if run_forecast:
    with st.spinner("Running forecast, please wait..."):
        try:
            # Data Preparation using the Forecast Pipeline
            # The pipeline prepares the input data based on the selected dates and closed dates
            data = fp.forecast_pipe(date.strftime("%Y-%m-%d"), 
                                      ped_start.strftime("%Y-%m-%d"), 
                                      ped_end.strftime("%Y-%m-%d"), 
                                      closed_dates=closed_dates)

            # Generate Predictions for Each Sales Category
            predictions_cat1 = fp.load_sales_model_and_forecast("sales_models/xgb_model_Coffee.pkl", data, date)
            predictions_cat2 = fp.load_sales_model_and_forecast("sales_models/xgb_model_Without_Coffee.pkl", data, date)
            predictions_cat3 = fp.load_sales_model_and_forecast("sales_models/xgb_model_Food.pkl", data, date)

            # Calculate total predictions by summing forecasts from all categories
            total_predictions = predictions_cat1 + predictions_cat2 + predictions_cat3
            
            # Display Total Sales Predictions
            st.subheader("Total Sales Predictions")
            # Create two columns: one for the chart and one for the data table
            col1, col2 = st.columns([2,1])
            with col1:
                # Create a line chart for total predictions with markers
                fig_total = px.line(total_predictions, markers=True)
                mean_sales_total = total_predictions["sales"].mean()
                # Add a dashed horizontal line showing the mean sales
                fig_total.add_hline(y=mean_sales_total, 
                                    line_dash="dash", 
                                    annotation_position="top right", 
                                    line_color='dodgerblue', 
                                    annotation_text=f'Mean Sales: {mean_sales_total:.2f}')

                # Compute the indices and values of the minimum and maximum sales
                min_index_total = total_predictions["sales"].idxmin()
                max_index_total = total_predictions["sales"].idxmax()
                min_value_total = total_predictions["sales"].min()
                max_value_total = total_predictions["sales"].max()
                
                # Add scatter markers to highlight the min and max sales points
                fig_total.add_scatter(
                    x=[min_index_total],
                    y=[min_value_total],
                    mode='markers',
                    marker=dict(color='dodgerblue', size=8),
                    name='Min Sales'
                )
                fig_total.add_scatter(
                    x=[max_index_total],
                    y=[max_value_total],
                    mode='markers',
                    marker=dict(color='dodgerblue', size=8),
                    name='Max Sales'
                )
                
                # Set axis titles for clarity
                fig_total.update_layout(xaxis_title="Date", yaxis_title="Total Sales")
                st.plotly_chart(fig_total)
            with col2:
                # Show the total predictions in a small data table
                st.dataframe(total_predictions, width=200)

            # Display Sales Contribution by Category (Histogram)
            st.subheader("Sales Contribution by Category")
            
            # Calculate the percentage contribution of each category to total sales
            coffee_contribution = predictions_cat1["sales"].sum() / total_predictions["sales"].sum() * 100
            without_coffee_contribution = predictions_cat2["sales"].sum() / total_predictions["sales"].sum() * 100
            food_contribution = predictions_cat3["sales"].sum() / total_predictions["sales"].sum() * 100
            
            # Create a DataFrame summarizing the contributions
            contributions = pd.DataFrame({
                "Category": ["Coffee Sales", "Without Coffee Beverages Sales", "Food Sales"],
                "Contribution (%)": [coffee_contribution, without_coffee_contribution, food_contribution]
            })
            
            # Create a bar chart to display the contribution percentages
            fig_contrib = px.bar(contributions, x="Category", y="Contribution (%)", text="Contribution (%)")
            # Format the labels to show percentages with two decimal places and add a percent sign
            fig_contrib.update_traces(texttemplate='%{text:.2f}%', textposition='outside', width=0.5)
            fig_contrib.update_layout(title="Category Contribution to Total Sales", 
                                      xaxis_title="Category", 
                                      yaxis_title="Contribution (%)")
            st.plotly_chart(fig_contrib)

            # Display Coffee Sales Predictions
            st.subheader("Coffee Sales Predictions")
            col1, col2 = st.columns([2,1])
            with col1:
                # Prepare a DataFrame for coffee sales
                df_cat1 = pd.DataFrame(predictions_cat1, columns=["sales"])
                # Create the line chart for coffee sales
                fig1 = px.line(df_cat1, markers=True)
                mean_sales_cat1 = df_cat1["sales"].mean()
                # Add a horizontal line for the mean coffee sales
                fig1.add_hline(y=mean_sales_cat1, 
                               line_dash="dash", 
                               annotation_position="top right", 
                               line_color='dodgerblue', 
                               annotation_text=f'Mean Sales: {mean_sales_cat1:.2f}')

                # Compute min and max for coffee sales
                min_index_cat1 = df_cat1["sales"].idxmin()
                max_index_cat1 = df_cat1["sales"].idxmax()
                min_value_cat1 = df_cat1["sales"].min()
                max_value_cat1 = df_cat1["sales"].max()
                
                # Highlight the min and max points
                fig1.add_scatter(
                    x=[min_index_cat1],
                    y=[min_value_cat1],
                    mode='markers',
                    marker=dict(color='dodgerblue', size=8),
                    name='Min Sales'
                )
                fig1.add_scatter(
                    x=[max_index_cat1],
                    y=[max_value_cat1],
                    mode='markers',
                    marker=dict(color='dodgerblue', size=8),
                    name='Max Sales'
                )

                fig1.update_layout(xaxis_title="Date", yaxis_title="Coffee Sales")
                st.plotly_chart(fig1)
            with col2:
                # Show the coffee sales data in a table
                st.dataframe(pd.DataFrame(predictions_cat1, columns=["sales"]), width=200)

            # Display Without Coffee Beverages Sales Predictions
            st.subheader("Without Coffee Beverages Sales Predictions")
            col1, col2 = st.columns([2,1])
            with col1:
                # Prepare DataFrame for without coffee beverages sales
                df_cat2 = pd.DataFrame(predictions_cat2, columns=["sales"])
                # Create the line chart
                fig2 = px.line(df_cat2, markers=True)
                mean_sales_cat2 = df_cat2["sales"].mean()
                # Add a horizontal line for the mean
                fig2.add_hline(y=mean_sales_cat2, 
                               line_dash="dash", 
                               annotation_position="top right", 
                               line_color='dodgerblue', 
                               annotation_text=f'Mean Sales: {mean_sales_cat2:.2f}')

                # Compute min and max values
                min_index_cat2 = df_cat2["sales"].idxmin()
                max_index_cat2 = df_cat2["sales"].idxmax()
                min_value_cat2 = df_cat2["sales"].min()
                max_value_cat2 = df_cat2["sales"].max()
                
                # Highlight the min and max sales
                fig2.add_scatter(
                    x=[min_index_cat2],
                    y=[min_value_cat2],
                    mode='markers',
                    marker=dict(color='dodgerblue', size=8),
                    name='Min Sales'
                )
                fig2.add_scatter(
                    x=[max_index_cat2],
                    y=[max_value_cat2],
                    mode='markers',
                    marker=dict(color='dodgerblue', size=8),
                    name='Max Sales'
                )

                fig2.update_layout(xaxis_title="Date", yaxis_title="Without Coffee Beverages Sales")
                st.plotly_chart(fig2)
            with col2:
                # Display without coffee beverages sales data in a table
                st.dataframe(pd.DataFrame(predictions_cat2, columns=["sales"]), width=200)

            # Display Food Sales Predictions
            st.subheader("Food Sales Predictions")
            col1, col2 = st.columns([2,1])
            with col1:
                # Create a DataFrame for food sales predictions
                df_cat3 = pd.DataFrame(predictions_cat3, columns=["sales"])
                # Generate a line chart for food sales
                fig3 = px.line(df_cat3, markers=True)
                mean_sales_cat3 = df_cat3["sales"].mean()
                # Add a horizontal line for the mean
                fig3.add_hline(y=mean_sales_cat3, 
                               line_dash="dash", 
                               annotation_position="top right", 
                               line_color='dodgerblue', 
                               annotation_text=f'Mean Sales: {mean_sales_cat3:.2f}')

                # Compute and annotate min and max values
                min_index_cat3 = df_cat3["sales"].idxmin()
                max_index_cat3 = df_cat3["sales"].idxmax()
                min_value_cat3 = df_cat3["sales"].min()
                max_value_cat3 = df_cat3["sales"].max()
                
                # Highlight min and max points
                fig3.add_scatter(
                    x=[min_index_cat3],
                    y=[min_value_cat3],
                    mode='markers',
                    marker=dict(color='dodgerblue', size=8),
                    name='Min Sales'
                )
                fig3.add_scatter(
                    x=[max_index_cat3],
                    y=[max_value_cat3],
                    mode='markers',
                    marker=dict(color='dodgerblue', size=8),
                    name='Max Sales'
                )

                fig3.update_layout(xaxis_title="Date", yaxis_title="Food Sales")
                st.plotly_chart(fig3)
            with col2:
                # Display food sales predictions data
                st.dataframe(pd.DataFrame(predictions_cat3, columns=["sales"]), width=200)

        except Exception as e:
            # If any error occurs, display it in the app
            st.error(f"An error occurred: {e}")
    st.success("Forecast completed successfully!")
else:
    # Instructions to the user before running the forecast
    st.info("Click the 'Run Forecast' button to generate predictions.")
