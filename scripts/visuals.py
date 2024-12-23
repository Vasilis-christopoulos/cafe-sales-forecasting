import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_heatmap(df, figsize=(30, 12), cmap='viridis'):
    """
    Plots a correlation heatmap for the given dataframe.

    Parameters:
    df (pd.DataFrame): The dataframe for which to plot the correlation heatmap.
    figsize (tuple): The size of the figure.
    cmap (str): The colormap to use for the heatmap.
    """
    # Calculate the correlation matrix
    corr_matrix = df.corr()

    # Set up the matplotlib figure
    plt.figure(figsize=figsize)

    # Draw the heatmap with the correlation matrix
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=cmap, linewidths=0.5)

    # Set the title
    plt.title('Correlation Heatmap')

    # Show the plot
    plt.show()


def plot_monthly_sales_vs_macro(data: pd.DataFrame):
    """
    Plot monthly sales data for each category against macro indicators.

    Parameters:
    data (pd.DataFrame): The dataframe containing the monthly sales data.
    """
    try:
        # Ensure we have a datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            data = data.copy()
            data.index = pd.to_datetime(data.index)
        
        # Resample to monthly frequency
        monthly_data = data.resample('ME').mean()
        
        # Define categories and macro indicators
        categories = ['Cold Coffee', 'Hot Coffee', 'Without Coffee', 'Food', 'Desserts', 'Merch']
        macro_indicators = ['GDP', 'CPI', 'Unemployment Rate', 'Bond Yields']
        
        # Set seaborn style
        sns.set_style("whitegrid")
        colors = sns.color_palette("pastel", n_colors=len(categories))
        
        # Create figure and subplots
        fig, axes = plt.subplots(len(macro_indicators), 1, figsize=(15, 5*len(macro_indicators)))
        
        # Plot each macro indicator
        for idx, indicator in enumerate(macro_indicators):
            ax1 = axes[idx]
            ax2 = ax1.twinx()
            
            # Plot each category with seaborn
            for cat_idx, category in enumerate(categories):
                sns.lineplot(data=monthly_data, x=monthly_data.index, y=category, ax=ax1, color=colors[cat_idx], legend = False)

                # Add category name at the end of each line
                last_value = monthly_data[category].iloc[-1]
                ax1.annotate(category, 
                            xy=(monthly_data.index[-1], last_value),
                            xytext = (10, 0),
                            textcoords = 'offset points',
                            va='center',
                            color=colors[cat_idx])
            
            # Plot macro indicator
            ax2.plot(monthly_data.index, monthly_data[indicator], 
                    color='black', linestyle='--')
            
            # Formatting
            xlims = ax1.get_xlim()
            ax1.set_xlim(xlims[0], xlims[1] + (xlims[1] - xlims[0]) * 0.05)
            ax2.set_xlim(ax1.get_xlim())

            ax1.set_xlabel('Date')
            ax1.set_ylabel('Average Monthly Sales')
            ax2.set_ylabel(indicator)
            ax1.set_title(f'Monthly Sales Categories vs {indicator}')
            
        plt.show()
        
    except Exception as e:
        print(f"Error in plotting: {str(e)}")
        raise

def plot_sales_vs_macro_lags(data: pd.DataFrame):
    categories = ['Hot Coffee', 'Cold Coffee', 'Without Coffee', 'Food', 'Desserts', 'Merch', 'Coffee Beans']
    base_indicators = ['GDP', 'CPI', 'Unemployment Rate', 'Bond Yields']
    lags = [7, 14, 30, 45]
    
    sns.set_style("whitegrid")
    colors = sns.color_palette("viridis", n_colors=len(lags))
    
    fig, axes = plt.subplots(len(base_indicators), len(categories), 
                            figsize=(20, 6*len(base_indicators)))
    
    for idx, indicator in enumerate(base_indicators):
        for cat_idx, category in enumerate(categories):
            ax = axes[idx, cat_idx]
            
            for lag_idx, lag in enumerate(lags):
                lag_col = f'{indicator}_lag_{lag}'
                
                # Sort values for clean visualization
                sorted_data = data.sort_values(lag_col).copy()
                
                # Plot trend line using sns.lineplot
                sns.lineplot(data=sorted_data, x=lag_col, y=category, 
                           color=colors[lag_idx], label=f'Lag {lag}',
                           ax=ax)
            
            # Formatting
            if idx == 0:
                ax.set_title(category)
            if cat_idx == 0:
                ax.set_ylabel(f'{indicator}\nSales')
            else:
                ax.set_ylabel('')
            if idx == len(base_indicators)-1:
                ax.set_xlabel('Macro Value')
            else:
                ax.set_xlabel('')
                
            if cat_idx == len(categories)-1:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                ax.get_legend().remove() if ax.get_legend() else None
    
    plt.tight_layout()
    plt.show()

def plot_lag_features(data, category_sales, features, title):
    data = data.resample('M').mean().reset_index()
    data_melted = data.melt(id_vars='index', value_vars=features, var_name='Feature', value_name='Value')
    
    g = sns.FacetGrid(data_melted, col='Feature', col_wrap=2, sharex=True, sharey=False, height=4, aspect=2)
    g.map_dataframe(sns.lineplot, x='index', y='Value')
    g.set_axis_labels('Date', 'Value')
    g.set_titles(col_template='{col_name}')
    g.fig.suptitle(title, y=1.02)

    # Plot all categories in a single larger chart
    fig, ax = plt.subplots(figsize=(20, 8))

    # Resample the data to monthly frequency and plot each category
    category_sales_monthly = category_sales.resample('M').sum()

    # Plot each category
    for category in ['Hot Coffee', 'Cold Coffee', 'Without Coffee', 'Food', 'Desserts', 'Coffee Beans', 'Merch']:
        sns.lineplot(data=category_sales_monthly, x=category_sales_monthly.index, y=category, ax=ax, label=category)

    # Set titles and labels
    ax.set_title('Monthly Sales for All Categories')
    ax.set_xlabel('Date')
    ax.set_ylabel('Net Sales')
    ax.legend(title='Category')

    plt.tight_layout()
    plt.show()