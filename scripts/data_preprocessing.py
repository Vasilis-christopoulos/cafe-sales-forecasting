import pandas as pd


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

def item_sales_preprocess(df):
    """
    Preprocesses item sales data by cleaning columns, filtering categories, handling missing values, and setting the index.
    Parameters:
        df (pd.DataFrame): Input DataFrame.
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Remove dollar sign and convert to float
    df['Gross Sales'] = df['Gross Sales'].apply(remove_dollar_sign_and_convert)
    df['Net Sales'] = df['Net Sales'].apply(remove_dollar_sign_and_convert)
    # Remove rows with 'Total' in 'Category Name' column
    df = df[~df['Category Name'].str.contains('Total', case=False, na=False)]
    # Fill the 'Category Name' column with the previous non
    df.loc[:, 'Category Name'] = df['Category Name'].ffill()
    # Drop rows that have empty columns and reset the index
    df = df.dropna().reset_index(drop=True)
    # Convert Sold column to integer
    df['Sold'] = df['Sold'].astype(int)
    # Set the 'date' column as the index
    df = df.set_index('date')

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
        'Gross Sales': df.iloc[0, 1:-1].values,  # Row 1 is Gross Sales
        'Net Sales': df.iloc[3, 1:-1].values     # Row 4 is Net Sales
    }, index=dates)

    sales_data['Gross Sales'] = sales_data['Gross Sales'].apply(remove_dollar_sign_and_convert)
    sales_data['Net Sales'] = sales_data['Net Sales'].apply(remove_dollar_sign_and_convert)

    return sales_data

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
