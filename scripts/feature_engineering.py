import pandas as pd 

# Item Sales
# ------------------------------------------------------------------------------
def identify_and_filter_top_seasonal_items(item_sales_df, top_percentage=0.3):
    """
    Identify the best-performing items per season and filter the original DataFrame to keep only these items.
    
    Args:
        item_sales_df (pd.DataFrame): Sales data with columns [start_date, Name, Category Name, Net Sales, Sold]
        top_percentage (float): Percentage of top items to select per season (default: 0.3)
    
    Returns:
        pd.DataFrame: Filtered DataFrame with only the top-performing items per season
    """
    # Add season column
    item_sales_df['season'] = pd.to_datetime(item_sales_df['start_date']).dt.month.map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    # Initialize an empty list to store top-performing items per season
    top_items_list = []
    
    # Process each season
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        # Filter data for current season
        season_data = item_sales_df[item_sales_df['season'] == season].copy()
        
        if not season_data.empty:
            # Calculate metrics per item
            season_metrics = season_data.groupby(['Category Name', 'Name']).agg({
                'Net Sales': ['sum', 'mean', 'count'],
                'Sold': ['sum', 'mean']
            }).round(2)
            
            # Flatten column names
            season_metrics.columns = [
                'total_net_sales', 'avg_monthly_sales', 'months_active',
                'total_quantity', 'avg_monthly_quantity'
            ]
            
            # Normalize metrics within season
            for col in ['total_net_sales', 'avg_monthly_sales', 'total_quantity', 'avg_monthly_quantity']:
                max_val = season_metrics[col].max()
                if max_val > 0:
                    season_metrics[f'{col}_normalized'] = (season_metrics[col] / max_val).round(3)
                else:
                    season_metrics[f'{col}_normalized'] = 0
            
            # Calculate seasonal importance score
            weights = {
                'total_net_sales_normalized': 0.3,
                'avg_monthly_sales_normalized': 0.3,
                'total_quantity_normalized': 0.2,
                'avg_monthly_quantity_normalized': 0.2
            }
            
            season_metrics['importance_score'] = sum(
                season_metrics[col] * weight 
                for col, weight in weights.items()
            )
            
            # Select top items
            n_items = int(len(season_metrics) * top_percentage)
            top_items = season_metrics.nlargest(n_items, 'importance_score')
            
            # Get the item names for the top items
            top_items_names = top_items.reset_index()[['Category Name', 'Name']]
            
            # Add to the list of top items
            top_items_list.append(top_items_names)
    
    # Concatenate the list of top items for each season into a single DataFrame
    top_items_df = pd.concat(top_items_list).drop_duplicates()

    # Filter the original item_sales_df to include only the top-performing items
    filtered_item_sales_df = item_sales_df.merge(
        top_items_df,
        on=['Category Name', 'Name'],
        how='inner'
    )
    
    # Drop the 'Category Name'
    return filtered_item_sales_df.drop(columns='Category Name')


def item_sales_transform(item_sales):
    """
    Transforms item sales data by aggregating, calculating average net sales and sold per day, and pivoting the data.
    Parameters:
        item_sales (pd.DataFrame): Input DataFrame.
    Returns
        pd.DataFrame: Transformed DataFrame.
    """

    # Calculate average net sales and average sold per day
    item_sales['Avg Net Sales'] = item_sales['Net Sales'] / (item_sales['end_date'] - item_sales['start_date']).dt.days
    item_sales['Avg Sold'] = item_sales['Sold'] / (item_sales['end_date'] - item_sales['start_date']).dt.days

    # Drop unnecessary columns
    avg_item_sales = item_sales.drop(columns=['Net Sales', 'Sold'])

    # Pivot the aggregated data
    pivoted_df = avg_item_sales.pivot_table(
        index=['start_date', 'end_date'], 
        columns='Name', 
        values=['Avg Net Sales', 'Avg Sold'],
        aggfunc='sum'
    ).reset_index()

    # Flatten multi-level columns
    pivoted_df.columns = ['start_date', 'end_date'] + [f'{metric}_{item}' for metric, item in pivoted_df.columns[2:]]

    # Ensure 'start_date' and 'end_date' are datetime objects
    pivoted_df['start_date'] = pd.to_datetime(pivoted_df['start_date'])
    pivoted_df['end_date'] = pd.to_datetime(pivoted_df['end_date'])

    # Create date range from the earliest start_date to the latest end_date
    date_range = pd.date_range(start=pivoted_df['start_date'].min(), end=pivoted_df['end_date'].max(), freq='D')

    # Create a DataFrame with the Date Index
    dates_df = pd.DataFrame({'date': date_range})

    # Create intervals representing each period in 'pivoted_df'
    intervals = pd.IntervalIndex.from_arrays(pivoted_df['start_date'], pivoted_df['end_date'], closed='both')

    # Assign each date to the interval it falls into
    dates_df['interval'] = pd.cut(dates_df['date'], intervals)

    # Map intervals to the indices of 'pivoted_df' for merging
    interval_to_index = pd.Series(range(len(intervals)), index=intervals)
    dates_df['pivot_index'] = dates_df['interval'].map(interval_to_index)

    # Merge 'dates_df' with 'pivoted_df' on the interval indices
    daily_pivoted_df = dates_df.merge(pivoted_df.drop(['start_date', 'end_date'], axis=1), left_on='pivot_index', right_index=True, how='left')

    # Set 'date' as the index and drop unnecessary columns
    daily_pivoted_df.set_index('date', inplace=True)
    daily_pivoted_df.drop(['interval', 'pivot_index'], axis=1, inplace=True)

    # Fill missing values with 0 showing that no sales were made
    daily_pivoted_df.fillna(0, inplace=True)
    
    return daily_pivoted_df

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
    
    return holiday_features

# Pedestrinization
# ------------------------------------------------------------------------------
def create_pedestrianization():
    """
    Create pedestrianization DataFrame with binary classification
    0 = no pedestrianization
    1 = pedestrianization
    """
    pedestrianization_dates = pd.date_range(start = '2024-06-01', end = '2024-08-30', freq='D')
    
    date_range = pd.date_range(start='2023-10-01', end='2024-10-31', freq='D')
    pedestrianization_features = pd.DataFrame(index=date_range)
    
    # Initialize with 0 (no pedestrianization)
    pedestrianization_features['is_pedestrian'] = 0

    # If date is in pedestrianization_dates, set is_pedestrian to 1
    pedestrianization_features.loc[pedestrianization_dates, 'is_pedestrian'] = 1

    return pedestrianization_features
