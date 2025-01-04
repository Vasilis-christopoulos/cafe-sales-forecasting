import xgboost as xgb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
    
def calculate_mape(y_test, y_pred):
    """Calculate MAPE excluding zero values"""
    mask = y_test != 0
    return np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
    

def train_category_models(
    df, 
    model_type, 
    target_cols, 
    date_split, 
    xgb_params=None,
    lgb_params=None,
    random_state=42
):
    """
    Train models with customizable parameters
    
    Args:
        df: Input dataframe with datetime index
        model_type: 'xgboost', 'lightgbm', or 'randomforest'
        target_cols: List of sales columns to predict
        date_split: Datetime string for train/test split
        xgb_params: Dict of XGBoost parameters to override defaults
        lgb_params: Dict of LightGBM parameters to override defaults
        rf_params: Dict of RandomForest parameters to override defaults
        random_state: Random seed
    """
    
    # Default parameter grids
    default_xgb_params = {
        'model__max_depth': [3, 7],
        'model__learning_rate': [0.01, 0.1],
        'model__n_estimators': [100, 200],
        'model__min_child_weight': [1, 5],
        'model__subsample': [0.8, 1.0],
        'model__colsample_bytree': [0.8, 1.0],
        'model__gamma': [0, 0.1],
        'model__nthread': [-1]
    }
    
    default_lgb_params = {
    'model__num_leaves': [3, 7],            
    'model__learning_rate': [0.01, 0.1],    
    'model__n_estimators': [100, 200],      
    'model__min_child_samples': [3, 5],     
    'model__subsample': [0.8, 1.0],          
    'model__colsample_bytree': [0.8, 1.0],  
    'model__reg_alpha': [0.01, 0.1],         
    'model__reg_lambda': [0.01, 0.1],         
    'model__max_depth': [3, 7],
    'model__nthread': [-1]  
    }

    
    # Update with custom parameters if provided
    if xgb_params:
        default_xgb_params.update(xgb_params)
    if lgb_params:
        default_lgb_params.update(lgb_params)
    
    tscv = TimeSeriesSplit(n_splits=5)
    results = {}


    
    for target_col in target_cols:
        print(f"\nTraining models for category sales: {target_col}")
        
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        X_train = X[X.index < date_split]
        X_test = X[X.index >= date_split]
        y_train = y[y.index < date_split]
        y_test = y[y.index >= date_split]
        
        print(f"Train period: {X_train.index.min()} to {X_train.index.max()}")
        print(f"Test period: {X_test.index.min()} to {X_test.index.max()}")
        
        if model_type.lower() == 'xgboost':
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', xgb.XGBRegressor(random_state=random_state))
            ])
            param_grid = default_xgb_params
        elif model_type.lower() == 'lightgbm':
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', lgb.LGBMRegressor(random_state=random_state, verbose=-1))
            ])
            param_grid = default_lgb_params
        else:
            raise ValueError("model_type must be 'xgboost', 'lightgbm', or 'randomforest'")
        
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring='neg_root_mean_squared_error',
            cv=tscv,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        y_pred = grid_search.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = calculate_mape(y_test, y_pred)
        
        results[target_col] = {
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_,
            'test_rmse': rmse,
            'test_mape': mape,
            'test_r2': r2,
            'test_mae': mae,
            'test_mse': mse
        }
        
        print(f"Best RMSE for {target_col}: {rmse:.4f}")
        print(f"Best MAPE for {target_col}: {mape:.4f}")
        print(f"Best R2 for {target_col}: {r2:.4f}")
        print(f"Best MAE for {target_col}: {mae:.4f}")
        print(f"Best MSE for {target_col}: {mse:.4f}")
    
    summary_df = pd.DataFrame({
        'Category': list(results.keys()),
        'Test RMSE': [results[cat]['test_rmse'] for cat in results.keys()],
        'Test MAPE': [results[cat]['test_mape'] for cat in results.keys()],
        'Test R2': [results[cat]['test_r2'] for cat in results.keys()],
        'Test MAE': [results[cat]['test_mae'] for cat in results.keys()],
        'Test MSE': [results[cat]['test_mse'] for cat in results.keys()],
        'CV RMSE': [results[cat]['best_score'] for cat in results.keys()]
    })
    
    return results, summary_df


