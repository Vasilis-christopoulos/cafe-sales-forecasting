import xgboost as xgb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple

def analyze_category_feature_importance(
    data: pd.DataFrame, 
    target_categories: List[str]
) -> Tuple[Dict[str, xgb.XGBRegressor], pd.DataFrame]:
    """
    Analyze and visualize feature importance across categories using XGBoost.
    Features are sorted by importance in each category.
    
    Args:
        data: DataFrame containing features and target categories
        target_categories: List of category names to analyze
    
    Returns:
        Tuple containing:
        - Dictionary of trained XGBoost models for each category
        - DataFrame with feature importance data
    """
    # Prepare features
    feature_cols = [col for col in data.columns if col not in target_categories]
    X = data[feature_cols]
    models = {}
    importance_data = []
    
    try:
        # Train models and collect importance data
        for category in target_categories:
            # Train model
            y = data[category]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            model = xgb.XGBRegressor(random_state=42)
            model.fit(X_train, y_train)
            
            # Get feature importance and sort
            features_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_,
                'category': category
            }).sort_values('importance', ascending=False)
            
            importance_data.append(features_importance)
        
        # Combine importance data
        all_importance = pd.concat(importance_data, ignore_index=True)
        
        # Calculate and sort overall importance through average
        overall_importance = (all_importance.groupby('feature')['importance']
                            .mean()
                            .sort_values(ascending=False)
                            .reset_index())
        overall_importance['category'] = 'Overall'
        
        final_data = pd.concat([all_importance, overall_importance], ignore_index=True)
        
        # Create visualization with sorted features
        for cat in target_categories + ['Overall']:
            # Sort features based on importance for each category
            cat_order = final_data[final_data['category'] == cat].sort_values('importance', ascending=True)['feature']
            final_data.loc[final_data['category'] == cat, 'feature'] = pd.Categorical(
                final_data.loc[final_data['category'] == cat, 'feature'],
                categories=cat_order,
                ordered=True
            )
        
        g = sns.catplot(
            data=final_data,
            y='feature',
            x='importance',
            col='category',
            kind='bar',
            col_wrap=3,
            height=7,
            aspect=1,
            sharex=False,
            sharey=False,
            palette='viridis',
            hue = 'feature'
        )
        g.fig.suptitle('Feature Importance by Category (including Overall)', y=1.02)
        plt.tight_layout()
        plt.show()
        
        return final_data
    
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return None, None
