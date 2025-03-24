import pandas as pd

#CSV files
#------------------------------------------------------------------------------
def item_sales_preprocess(df):
    """
    Preprocesses item sales data by cleaning columns, filtering categories, handling missing values, and setting the index.
    Parameters:
        df (pd.DataFrame): Input DataFrame.
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Remove dollar sign and convert to float
    df['Net Sales'] = df['Net Sales'].apply(remove_dollar_sign_and_convert)
    # Remove rows with 'Total' in 'Category Name' column
    df = df[~df['Category Name'].str.contains('Total', case=False, na=False)]
    # Fill the 'Category Name' column with the previous non
    df.loc[:, 'Category Name'] = df['Category Name'].ffill()
    # Drop rows that have empty columns and reset the index
    df = df.dropna().reset_index(drop=True)
    # Convert Sold column to integer
    df['Sold'] = df['Sold'].astype(int)
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    # Sort the DataFrame by date
    df.sort_values(by='date', inplace=True)

    return df


def sales_preprocess(df):
    """
    Preprocesses daily sales data by cleaning columns, handling missing values, and setting the index.
    Parameters:
        df (pd.DataFrame): Input DataFrame.
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Extract year once
    year = df.columns[0].split(',')[1].strip().split(' - ')[0]
    
    # Create date string by adding the year and converting to datetime
    dates = pd.to_datetime(
        df.columns[1:-1].str.strip('"') + ' ' + year, 
        format='%a, %b %d %Y'
    )
    
    # Create sales DataFrame
    sales_data = pd.DataFrame({
        'Net Sales': df.iloc[3, 1:-1].values     # Row 4 is Net Sales
    }, index=dates)
    sales_data.index = pd.to_datetime(sales_data.index)
    sales_data['Net Sales'] = sales_data['Net Sales'].apply(remove_dollar_sign_and_convert)

    return sales_data

# Local Holidays
#------------------------------------------------------------------------------
from datetime import datetime
def local_holidays_preprocess(df, start_date='2023-10-01', end_date='2024-10-01'):
    """
    Preprocesses local holidays data by parsing dates properly and setting the index.
    Parameters:
        df (pd.DataFrame): Input DataFrame.
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Parse dates properly handling timezone info
    df['Date'] = df['Date'].apply(lambda x: datetime.fromisoformat(x).date())
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Filter records by date range
    df = df[start_date:end_date]

    return df

# Weather
#------------------------------------------------------------------------------
def weather_preprocess(weather):
    weather_clean = weather.drop(columns=['tsun', 'wpgt', 'snow'], axis=1)
    weather_clean = weather_clean.fillna(0)
    return weather_clean

#Utility functions
#------------------------------------------------------------------------------
def daily_resample(data, start_date='2023-10-01', end_date='2024-10-31'):
    """
    Resamples data to daily frequency and fills missing values.
    Parameters:
        data (pd.DataFrame): Input DataFrame.
        start_date (str): Start date.
        end_date (str): End date.
    Returns:
        pd.DataFrame: Resampled DataFrame.
    """
    data_daily = data.resample('D').interpolate(method='time')
    data_daily = data_daily.reindex(pd.date_range(start=start_date, end=end_date, freq='D')).interpolate()
    return data_daily

def remove_dollar_sign_and_convert(x):
    """
    Removes dollar sign and converts to float.
    Parameters:
        x (str): Dollar amount
    Returns:
        float: Dollar amount as float
    """
    if isinstance(x, str):
        return float(x.replace('$', '').replace(',', '').strip())
    return x

def remove_outliers(data, categories, threshold=3):
    """
    Removes outliers from the data for each category.
    Parameters:
        data (pd.DataFrame): Input data.
        categories (list): List of target variables.
        threshold (int): Z-score threshold.
    Returns:
        pd.DataFrame: Data without outliers.
    """
    for category in categories:
        z_scores = (data[category] - data[category].mean()) / data[category].std()
        data = data[abs(z_scores) < threshold]
    return data