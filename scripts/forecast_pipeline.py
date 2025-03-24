import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scripts.data_fetching import macroeconomic_fetch_fred, make_request
from scripts.data_preprocessing import daily_resample
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from scripts.model_training import calculate_mape
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from scripts.feature_engineering import create_pedestrianization, create_time_features, create_lag_features
import pickle
import dill
import os
from dotenv import load_dotenv
load_dotenv()
from fredapi import Fred
import warnings

# Macroeconomic data
# This function would be adjusted with the API for fetching the macroeconomic foreasts
def macro_forecast(date) -> pd.DataFrame:
    """
    Fetches macroeconomic data for 30 days before and 10 days after the current date.
    Args:
        date (str): The current date
    Returns:
        pandas.DataFrame: A DataFrame with macroeconomic data
    """
    
    try:
        # Fetch macroeconomic data
        start_date = datetime.strptime(date, '%Y-%m-%d') - timedelta(days=29)
        end_date = datetime.strptime(date, '%Y-%m-%d') + timedelta(days=10)
        cpi, unemployment, bond_yields = macroeconomic_fetch_fred(start_date=start_date)
    except Exception as e:
        print(f"An error occurred while fetching macroeconomic data: {e}")


    cpi_daily = daily_resample(cpi, start_date=start_date, end_date=end_date)
    unemployment_daily = daily_resample(unemployment, start_date=start_date, end_date=end_date)
    bond_yields_daily = daily_resample(bond_yields, start_date=start_date, end_date=end_date)

    # Concatenate the data
    macroeconomic = pd.concat([cpi_daily, unemployment_daily, bond_yields_daily], axis=1)

    # Create lag features
    macroeconomic_lags = create_lag_features(macroeconomic, cols=['CPI', 'Unemployment Rate', 'Bond Yields'], lags=[7, 10, 14, 30])

    # Keep only the required columns
    macroeconomic_final = macroeconomic_lags[['CPI', 'Unemployment Rate', 'CPI_lag_10', 'Unemployment Rate_lag_7', 'Unemployment Rate_lag_10', 'Unemployment Rate_lag_14', 'Unemployment Rate_lag_30', 'Bond Yields_lag_7', 'Bond Yields_lag_10', 'Bond Yields_lag_14', 'Bond Yields_lag_30']]
    
    return macroeconomic_final.dropna()

# Weather Forecast

import meteostat
from datetime import datetime, timedelta

def weather_forecast(date):
    """
    Fetches the daily forecast for 10 days from the current date.
    Args:
        date (str): The current date
    Returns:
        pandas.DataFrame: A forecast of weather data
    """
    try:
        # Set the location
        location = meteostat.Point(lat=45.47, lon=-73.74, alt = 32)
        # Get the forecast data
        forecast = meteostat.Daily(location, start=datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1), end=datetime.strptime(date, '%Y-%m-%d') + timedelta(days=10))
        forecast = forecast.fetch()
    except Exception as e:
        print(f"An error occurred while fetching weather forecast data: {e}")
    
    filtered_forecast = forecast[['tavg', 'wspd']]

    return filtered_forecast

# Holiday Feature

def holidays(date):
    """
    Fetches the holiday data for 10 days from the current date.
    Args:
        date (str): The current date
    Returns:
        pandas.DataFrame: A forecast of holiday data
    """

    start_date = datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)
    end_date = datetime.strptime(date, '%Y-%m-%d') + timedelta(days=10)
    start_year = start_date.year
    end_year = end_date.year
    try:
        if start_year != end_year:
            start_year_holidays = make_request(start_year)
            end_year_holidays = make_request(end_year)
            df = pd.concat([start_year_holidays, end_year_holidays])
        else:
            df = make_request(start_year)
    except Exception as e:
        print(f"An error occurred while fetching holiday data: {e}")
   
    df['Date'] = df['Date'].apply(lambda x: datetime.fromisoformat(x).date())
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    df.set_index('Date', inplace=True)
    df.index = pd.to_datetime(df.index)

    df = df[(df.index >= start_date) & (df.index <= end_date)]

    return df

def major_holiday_feature(df, date):
    """
    Create a binary feature indicating whether a date is a major holiday.
    Args:
        df (pandas.DataFrame): The holiday DataFrame
        date (str): The start date
    Returns:
        pandas.DataFrame: The holiday DataFrame with the major holiday feature
    """

    dates = pd.date_range(datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1), datetime.strptime(date, '%Y-%m-%d') + timedelta(days=10))
    major_holidays = [
        'Thanksgiving Day', 
        'Halloween', 
        'Christmas Eve', 
        'Christmas Day', 
        'New Year\'s Eve', 
        'New Year\'s Day', 
        'National Patriots\' Day', 
        'St. Jean Baptiste Day', 
        'Canada Day', 
        'Labour Day', 
        'Valentine\'s Day', 
        'Mother\'s Day', 
        'Father\'s Day', 
        'Good Friday', 
        'Easter Sunday'
    ]
    df['holiday_type_2'] = df['Name'].apply(lambda x: 1 if x in major_holidays else 0)
    # Identify dates missing from the current df index
    missing_dates = dates.difference(df.index)
    # Create a DataFrame for these missing dates with the same columns as df
    missing_df = pd.DataFrame(index=missing_dates, columns=df.columns)
    missing_df['holiday_type_2'] = 0
    # Concatenate the missing dates with the original df and sort by the index
    df = pd.concat([df, missing_df]).sort_index()

    return df.drop('Name', axis=1)

def holiday_feature(date):
    holiday_data = holidays(date)
    holiday_feature = major_holiday_feature(holiday_data, date)
    holiday_feature = holiday_feature.groupby(holiday_feature.index).max()
    holiday_feature = holiday_feature.sort_index()
    holiday_feature['before_holiday'] = 0
    for holiday_date in holiday_feature.index[holiday_feature['holiday_type_2'] == 1]:
        for offset in range(1, 4):
            prior_date = holiday_date - pd.Timedelta(days=offset)
            if prior_date in holiday_feature.index:
                holiday_feature.loc[prior_date, 'before_holiday'] = 1

    return holiday_feature

# Pedestrianization

def pedestrianization(date, ped_start, ped_end):
    start_date = datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)
    end_date = datetime.strptime(date, '%Y-%m-%d') + timedelta(days=10)
    pedestrian_feature = create_pedestrianization('2025-06-01', '2025-09-30', start_date, end_date)

    return pedestrian_feature

# Time features

def time_features(date):
    start_date = datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)
    end_date = datetime.strptime(date, '%Y-%m-%d') + timedelta(days=10)
    time_features = create_time_features(start_date, end_date)
    time_features = time_features.drop(columns = ['year', 'month', 'day_of_month'])

    time_features = pd.get_dummies(time_features, columns=['day_of_week']).astype(int)
    time_features['quarter_1'] = time_features['quarter'].apply(lambda x: 1 if x == 1 else 0)
    time_features['quarter_3'] = time_features['quarter'].apply(lambda x: 1 if x == 3 else 0)
    time_features = time_features[['is_weekend', 'day_of_week_1', 'day_of_week_2', 'day_of_week_4', 'quarter_1',  'quarter_3']]

    return time_features


# Forecast Pipeline

def load_sales_model_and_forecast(model_path, df, date):
    """
    Loads a pickled dilled model and produces forecasts.

    Args:
        model_path (str): Path to the pickled dilled model.
        df (pandas.DataFrame): The DataFrame containing the features for forecasting.

    Returns:
        numpy.ndarray: The forecasted values.
    """
    with open(model_path, 'rb') as f:
        model = dill.load(f)
    predictions = model.predict(df)
    dates = pd.date_range(datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1), datetime.strptime(date, '%Y-%m-%d') + timedelta(days=10))
    predictions = pd.DataFrame(predictions, index=dates, columns=['sales'])
    
    return predictions

def reorder_columns(df):
    """
    Reorders the columns of the input DataFrame to the desired order.
    
    Args:
        df (pd.DataFrame): The DataFrame with the original columns.
        
    Returns:
        pd.DataFrame: A new DataFrame with the columns in the desired order.
        
    Raises:
        ValueError: If any required column is missing in the DataFrame.
    """
    desired_order = ['closed', 'holiday_type_2',
       'is_pedestrian', 'is_weekend', 'CPI', 'Unemployment Rate', 'tavg',
       'wspd', 'quarter_1', 'quarter_3', 'day_of_week_1', 'day_of_week_2',
       'day_of_week_4', 'CPI_lag_10', 'Unemployment Rate_lag_7',
       'Unemployment Rate_lag_10', 'Unemployment Rate_lag_14',
       'Unemployment Rate_lag_30', 'Bond Yields_lag_7', 'Bond Yields_lag_10',
       'Bond Yields_lag_14', 'Bond Yields_lag_30', 'before_holiday',
       'tavg_weekend'
    ]
    
    # Check if all required columns are present
    missing = [col for col in desired_order if col not in df.columns]
    if missing:
        raise ValueError(f"The following required columns are missing from the DataFrame: {missing}")
    
    # Return a new DataFrame with the columns in the desired order
    return df[desired_order]

## 

def forecast_pipe(date, ped_start, ped_end, closed_dates = None):
   
   # Macroeconomic indicators fetch
   macroeconomic = macro_forecast(date)

   # Weather forecast fetch
   weather = weather_forecast(date)

   # Holiday feature
   holidays = holiday_feature(date)

   # Pedestrianization feature
   pedestrian = pedestrianization(date, ped_start, ped_end)

   # Time features
   time_fs = time_features(date)

   # Merge the features & extra feature engineering
   data = pd.concat([macroeconomic, weather, holidays, pedestrian, time_fs], axis=1)
   data['tavg_weekend'] = data['tavg'] * data['is_weekend']
   data['closed'] = 0
   if closed_dates is not None:
        data.loc[closed_dates, 'closed'] = 1
   data = reorder_columns(data)

   return data