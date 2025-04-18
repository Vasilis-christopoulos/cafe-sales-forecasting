# Description: This file contains functions that are used to fetch data from csv files as well as functions that use APIs to fetch weather data, macroeconmic data and local holiday data.


# CSV file fetching
# ------------------------------------------------------------------------------

import importlib
import pandas as pd
import os
import glob
import scripts.data_preprocessing as dp

importlib.reload(dp)

def item_sales_csv_fetch(file_path) -> pd.DataFrame:
    """
    Reads daily item sales data from CSV file.
    Parameters:
        file_path (str): Path to the CSV file
    Returns:
        pandas.DataFrame: Preprocessed sales data with:
        - Single date column
        - Selected columns (Name, Gross Sales, Net Sales, Sold)
        - Preprocessed values (thousands separated, blank values handled)
    """
    try:
        # Extract date
        with open(file_path, 'r', encoding='utf-8') as file:
            next(file)  
            date_str = next(file).strip().strip('"')
            
        # Parse single date
        date = pd.to_datetime(date_str.split(' - ')[0].split(' 12:00')[0])
        
        # Find header with verification
        with open(file_path, 'r', encoding='utf-8') as file:
            header_idx = next((idx for idx, line in enumerate(file) 
                              if 'Category Name' in line), 0)
        
        cols = ['Category Name', 'Name', 'Net Sales', 'Sold']
        
        # Read CSV with explicit dtype
        df = pd.read_csv(file_path,
                         skiprows=header_idx,
                         encoding='utf-8',
                         quotechar='"',
                         na_values=[' ', ''],
                         skip_blank_lines=True,
                         usecols=cols)
        
        # Add single date column
        df['date'] = pd.to_datetime(date)

        # Initial preprocessing
        return dp.item_sales_preprocess(df)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def sales_csv_fetch(file_path: str) -> pd.DataFrame:
    """
    Reads daily sales data from CSV files with varying metadata.
    Parameters:
        file_path (str): Path to CSV file
    Returns:
        pandas.DataFrame: Sales data with:
        - Extracted date (index)
        - Selected columns (Gross Sales, Net Sales)
        - Preprocessed values (dollar sign removed, thousands separated, blank values handled)
    """
    try:
        # Find start row efficiently
        with open(file_path, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                if any(day in line for day in ['Mon,', 'Tue,', 'Wed,']):
                    start_idx = i
                    break
        
        # Read only required rows
        df = pd.read_csv(file_path,
                        skiprows=start_idx,
                        nrows=5,
                        encoding='utf-8',
                        thousands=',',    # Handle thousands separator
                        na_values=[''])   # Minimize na_values list
        
        sales_data = dp.sales_preprocess(df)
        
        return sales_data
    
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return pd.DataFrame()


def merge_all_sales(directory_path) -> pd.DataFrame:
    """
    Merges all CSV files in a directory into a single DataFrame.
    Parameters:
        directory_path (str): Path to the directory containing CSV files
    Returns:
        pandas.DataFrame: Merged DataFrame
    """
        
    all_files = glob.glob(os.path.join(directory_path, "*.csv"))
    dfs = []
    
    for file in all_files:
        try:
            if directory_path.startswith('data/Item Sales'):
                df = item_sales_csv_fetch(file)
            elif directory_path == 'data/Sales':
                df = sales_csv_fetch(file)
            else:
                print(f"Unknown directory: {directory_path}")
                return None
            if df is not None:
                dfs.append(df)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            
    if not dfs:
        return None
        
    df = pd.concat(dfs)
    df.sort_index(inplace=True)
    return df

# ------------------------------------------------------------------------------
# API data fetching
# ------------------------------------------------------------------------------
from meteostat import Point, Daily
def fetch_the_weather(start_date, end_date, lat=45.47, lon=-73.74) -> pd.DataFrame:
    """
    Fetches the weather data for Montreal from the Meteostat API.
    Parameters:
        start_date (datetime): Start date of the weather data
        end_date (datetime): End date of the weather data
        lat (float): Latitude of the location
        lon (float): Longitude of the location
        alt (float): Altitude of the location
    Returns:
        pandas.DataFrame: Weather data for Montreal or empty DataFrame on error
    """
    try:
        # Ensure the dates are in datetime format
        if not isinstance(start_date, datetime):
            start_date = pd.to_datetime(start_date)
        if not isinstance(end_date, datetime):
            end_date = pd.to_datetime(end_date)
            
        # Create point for Montreal
        montreal = Point(lat, lon)

        # Fetch the weather data for Montreal
        data = Daily(montreal, start_date, end_date)
        if data is None:
            raise ValueError("Could not initialize Daily weather data")

        data = data.normalize()
        data = data.fetch()

        if data.empty:
            print("Warning: No weather data found for specified period")
        else:
            data.index = pd.to_datetime(data.index)
            
        return data

    except Exception as e:
        print(f"Error fetching weather data: {str(e)}")
        return pd.DataFrame()

import os
from dotenv import load_dotenv
load_dotenv()
from fredapi import Fred
import streamlit as st

def macroeconomic_fetch_fred(start_date = '2023-10-01', end_date = None) -> pd.DataFrame:
    """
    Fetches macroeconomic data from the FRED API.
    Returns:
        pandas.Series: CPI data
        pandas.Series: Unemployment data
        pandas.Series: Bond yields
    """
    # Attempt to get the API key from Streamlit secrets.
    # In your deployed environment, this will be provided through st.secrets.
    try:
        api_key = st.secrets["fred"]["api_key"]
    except Exception as e:
        # Fallback: Load from the .env file for local development.
        from dotenv import load_dotenv
        import os
        load_dotenv()
        api_key = os.getenv('FRED_API_KEY')

    try:
        # Connect to the FRED API
        fred = Fred(api_key=api_key)

        # Fetch the macroeconomic data
        cpi = fred.get_series('CORESTICKM159SFRBATL', observation_start = start_date, observation_end = end_date)
        unemployment = fred.get_series('LRUNTTTTCAM156S', observation_start=start_date, observation_end = end_date)
        bond_yields = fred.get_series('IRLTLT01CAM156N', observation_start=start_date, observation_end = end_date)
    except Exception as e:
        print(f"Error fetching macroeconomic data: {str(e)}")
        return None, None, None, None

    return cpi.to_frame(name = 'CPI'), unemployment.to_frame(name = 'Unemployment Rate'), bond_yields.to_frame(name = 'Bond Yields')

import requests
from datetime import datetime
def make_request(year) -> pd.DataFrame:
    """
    Makes a request to the Calendarific API for Quebec holidays.
    Parameters:
        year (int): Year to fetch holidays for
    Returns:
        pandas.DataFrame: DataFrame with holiday names and dates
    """
     # Attempt to get the API key from Streamlit secrets.
    try:
        api_key = st.secrets["calendarific"]["api_key"]
    except Exception as e:
        # Fallback to reading the key from environment variables (loaded via a .env file).
        api_key = os.getenv('CALENDARIFIC_API_KEY')
    
    url = "https://calendarific.com/api/v2/holidays"

    params = {
        'api_key': api_key,
        'country': 'ca',
        'year': year,
        'location': 'ca-qc'
    }
    try:
        # Make the API request for 2023
        response = requests.get(url, params=params)

        holidays = response.json().get('response').get('holidays')

        holiday_data = [{'Name': h['name'], 
                        'Date': h['date']['iso']} for h in holidays]
    except Exception as e:
        print(f"Error fetching holidays: {str(e)}")
        return pd.DataFrame()
    
    return pd.DataFrame(holiday_data)
    

def local_holidays_fetch(start_date = '2023-10-01', end_date = '2024-10-31') -> pd.DataFrame:
    """
    Fetches Quebec local holidays for 2023-2024.
    Returns DataFrame with dates and holiday names.
    """
    # Fetch data for both years
    holiday_data_23 = make_request(2023)
    holiday_data_24 = make_request(2024)
    
    # Combine data
    holiday_data = pd.concat([holiday_data_23, holiday_data_24])
    
    holiday_data = dp.local_holidays_preprocess(holiday_data)

    return holiday_data
