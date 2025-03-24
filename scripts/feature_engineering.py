import pandas as pd 


# Local holidays
# ------------------------------------------------------------------------------
def create_holiday_features(holidays_df, start_date='2023-10-01', end_date='2024-10-31'):
    """
    Create holiday features DataFrame with numerical holiday classification
    0 = no holiday
    1 = minor holiday
    2 = major holiday
    """
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
    holidays_df.index = pd.to_datetime(holidays_df.index)
    holidays_df = holidays_df[(holidays_df.index >= start_date) & (holidays_df.index <= end_date)]

    holidays_df['holiday_type_2'] = holidays_df['Name'].apply(lambda x: 1 if x in major_holidays else 0)

    # Remove duplicate dates if any
    holidays_df = holidays_df[~holidays_df.index.duplicated(keep='first')]
    # Resample to daily frequency by reindexing and filling missing dates with 0 (no holiday)
    daily_range = pd.date_range(start=start_date, end=end_date, freq='D')
    holiday_features = holidays_df.reindex(daily_range, fill_value=0)
    holiday_features.index.name = 'Date'
    return holiday_features.drop(columns='Name')

# Pedestrinization
# ------------------------------------------------------------------------------
def create_pedestrianization(ped_start, ped_end, start_date='2023-10-01', end_date='2024-10-31'):
    """
    Create pedestrianization DataFrame with binary classification
    0 = no pedestrianization
    1 = pedestrianization
    """
    pedestrianization_dates = pd.date_range(start = ped_start, end = ped_end, freq='D')
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    pedestrianization_features = pd.DataFrame(index=date_range)
    
    # Initialize with 0 (no pedestrianization)
    pedestrianization_features['is_pedestrian'] = 0

    pedestrianization_features.loc[pedestrianization_features.index.isin(pedestrianization_dates), 'is_pedestrian'] = 1

    return pedestrianization_features

# Time based features
# ------------------------------------------------------------------------------
def create_time_features(start_date = '2023-10-01', end_date = '2024-10-31'):
    """
    Create time-based features DataFrame with numerical classification
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    time_features = pd.DataFrame(index=date_range)
    
    # Extract time-based features
    time_features['day_of_week'] = time_features.index.dayofweek
    time_features['day_of_month'] = time_features.index.day
    time_features['month'] = time_features.index.month
    time_features['year'] = time_features.index.year
    time_features['quarter'] = time_features.index.quarter
    time_features['is_weekend'] = time_features['day_of_week'].isin([5, 6]).astype(int)
    
    return time_features

def create_lag_features(df, cols, lags):
    """
    Creates lag features for specified columns and lag periods.

    Args:
    df (DataFrame): Original DataFrame
    cols (list): Columns to create lag features for
    lags (list): List of lag periods

    Returns:
    DataFrame: Updated DataFrame with lag features
    """
    lag_features = []  # Store lag features here
    for col in cols:
        for lag in lags:
            lag_col = df[col].shift(lag)  # Create the lagged feature
            lag_col.name = f'{col}_lag_{lag}'  # Rename the column to match the naming convention
            lag_features.append(lag_col)
    # Concatenate the lag features with the original DataFrame
    df = pd.concat([df] + lag_features, axis=1)
    
    return df