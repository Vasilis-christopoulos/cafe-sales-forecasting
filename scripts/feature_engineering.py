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
        'New Year\’s Eve', 
        'New Year\’s Day', 
        'National Patriots\’ Day, St.', 
        'Jean Baptiste Day', 
        'Canada Day', 
        'Labour Day', 
        'Valentine\’s Day', 
        'Mother\’s Day', 
        'Father\’s Day', 
        'Good Friday', 
        'Easter Sunday'
    ]

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    holiday_features = pd.DataFrame(index=date_range)
    
    # Initialize with 0 (no holiday)
    holiday_features['is_holiday'] = 0
    holiday_features['holiday_type'] = 0
    
    # Update holiday classifications (1 for minor, 2 for major)
    for date in holidays_df.index:
        if date in holiday_features.index:
            # Get holiday name as string, not Series
            holiday_name = holidays_df.loc[date, 'Name']
            if isinstance(holiday_name, pd.Series):
                holiday_name = holiday_name.iloc[0]
            
            holiday_features.loc[date, 'is_holiday'] = 1
            holiday_features.loc[date, 'holiday_type'] = 2 if holiday_name in major_holidays else 1
    
    # Drop 'is_holiday' column since the info is now in 'holiday_type'
    return holiday_features.drop(columns='is_holiday')

# Pedestrinization
# ------------------------------------------------------------------------------
def create_pedestrianization(start_date='2023-10-01', end_date='2024-10-31'):
    """
    Create pedestrianization DataFrame with binary classification
    0 = no pedestrianization
    1 = pedestrianization
    """
    pedestrianization_dates = pd.date_range(start = '2024-06-01', end = '2024-08-30', freq='D')
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    pedestrianization_features = pd.DataFrame(index=date_range)
    
    # Initialize with 0 (no pedestrianization)
    pedestrianization_features['is_pedestrian'] = 0

    # If date is in pedestrianization_dates, set is_pedestrian to 1
    pedestrianization_features.loc[pedestrianization_dates, 'is_pedestrian'] = 1

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

