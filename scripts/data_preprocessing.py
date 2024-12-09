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
    sales_data['Net Sales'] = sales_data['Net Sales'].apply(remove_dollar_sign_and_convert)

    return sales_data

# Local Holidays
#------------------------------------------------------------------------------
from datetime import datetime
def local_holidays_preprocess(df):
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
    df = df.loc['2023-10-01':'2024-10-01']

    return df

# Weather
#------------------------------------------------------------------------------
def weather_preprocess(weather):
    weather_clean = weather.drop(columns=['tsun', 'wpgt'], axis=1)
    weather_clean = weather_clean.fillna(0)
    return weather_clean

#Utility functions
#------------------------------------------------------------------------------
def daily_resample(data):
    """
    Resamples daily data to fill missing dates and forward fill values.
    Parameters:
        data (pd.DataFrame): Input DataFrame.
    Returns:
        pd.DataFrame: Resampled DataFrame.
    """
    data_daily = data.resample('D').ffill()
    data_daily = data_daily.reindex(pd.date_range(start='2023-10-01', end='2024-10-31', freq='D')).ffill()
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