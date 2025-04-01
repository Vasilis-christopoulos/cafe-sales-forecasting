import xgboost as xgb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
from typing import List, Dict, Tuple
from sklearn.linear_model import Ridge
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer

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
    
# Custom metrics
    
def calculate_mape(y_test, y_pred):
    """Calculate MAPE excluding zero values"""

    mask = y_test != 0
    return np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100

def accuracy_range(range, y_test, y_pred):
    """
    Calculate accuracy within a certain range
    """

    within = (abs(y_test - y_pred) <= range)
    accuracy = within.sum() / len(y_test) * 100
    
    return accuracy

def log_transform(x):
    return np.log1p(x)

def inverse_log_transform(x):
    return np.expm1(x)

def xgb_train_log(df, categories, n_splits, xgb_params, random_state=42, date_split='2024-11-16'):
    """
    Similar to ridge_train, but uses a TransformedTargetRegressor with log1p/expm1
    so that the target is log-transformed during training.
    """
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler, FunctionTransformer
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    results = {}
    residuals = {}
    best_models = {}
    
    transformer = FunctionTransformer(func=log_transform, inverse_func=inverse_log_transform)

    for category in categories:
        # Prepare data
        X = df.drop(categories, axis=1)
        y = df[category]

        X_train = X[X.index <= date_split]
        y_train = y[y.index <= date_split]
        X_test = X[X.index > date_split]
        y_test = y[y.index > date_split]

        # Time series CV
        tcsv = TimeSeriesSplit(n_splits=n_splits, test_size=30, gap=0)

        scaler = StandardScaler()
        pipeline = Pipeline([
            ('scaler', scaler),
            ('model', TransformedTargetRegressor(
                regressor=xgb.XGBRegressor(random_state=random_state),
                transformer=transformer
            ))
        ])

        
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=xgb_params,
            scoring='neg_root_mean_squared_error',
            cv=tcsv,
            refit=True,
            n_jobs=-1,
            verbose=False
        )
        
        grid_search.fit(X_train, y_train)

        y_pred_train = grid_search.predict(X_train)
        y_pred_test = grid_search.predict(X_test)

        # Evaluate metrics on original scale
        train_metrics = {
            'mse': mean_squared_error(y_train, y_pred_train),
            'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'mae': mean_absolute_error(y_train, y_pred_train),
            'r2': r2_score(y_train, y_pred_train),
            'mape': calculate_mape(y_train, y_pred_train),
            'accuracy_20': accuracy_range(20, y_train, y_pred_train),
            'accuracy_50': accuracy_range(50, y_train, y_pred_train),
            'accuracy_100': accuracy_range(100, y_train, y_pred_train)
        }

        test_metrics = {
            'mse': mean_squared_error(y_test, y_pred_test),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'mae': mean_absolute_error(y_test, y_pred_test),
            'r2': r2_score(y_test, y_pred_test),
            'mape': calculate_mape(y_test, y_pred_test),
            'accuracy_20': accuracy_range(20, y_test, y_pred_test),
            'accuracy_50': accuracy_range(50, y_test, y_pred_test),
            'accuracy_100': accuracy_range(100, y_test, y_pred_test)
        }

        results[category] = {
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_, 
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }

        # Residuals in original scale
        residuals[category] = {
            'train_residuals': y_train - y_pred_train,
            'test_residuals': y_test - y_pred_test
        }

        best_models[category] = grid_search.best_estimator_

        print(f"\nMetrics for {category}:")
        print("Training Metrics:")
        print(f"RMSE: {train_metrics['rmse']:.4f}")
        print(f"MAPE: {train_metrics['mape']:.4f}")
        print(f"R2: {train_metrics['r2']:.4f}")
        print("\nTest Metrics:")
        print(f"RMSE: {test_metrics['rmse']:.4f}")
        print(f"MAPE: {test_metrics['mape']:.4f}")
        print(f"R2: {test_metrics['r2']:.4f}")

    # Build summary DataFrame
    summary_df = pd.DataFrame({
        'Category': list(results.keys()),
        'Train RMSE': [results[cat]['train_metrics']['rmse'] for cat in results.keys()],
        'Train MAPE': [results[cat]['train_metrics']['mape'] for cat in results.keys()],
        'Train R2': [results[cat]['train_metrics']['r2'] for cat in results.keys()],
        'Train Accuracy 20': [results[cat]['train_metrics']['accuracy_20'] for cat in results.keys()],
        'Train Accuracy 50': [results[cat]['train_metrics']['accuracy_50'] for cat in results.keys()],
        'Train Accuracy 100': [results[cat]['train_metrics']['accuracy_100'] for cat in results.keys()],
        'Test RMSE': [results[cat]['test_metrics']['rmse'] for cat in results.keys()],
        'Test MAPE': [results[cat]['test_metrics']['mape'] for cat in results.keys()],
        'Test R2': [results[cat]['test_metrics']['r2'] for cat in results.keys()],
        'CV RMSE': [results[cat]['best_score'] for cat in results.keys()],
        'Test Accuracy 20': [results[cat]['test_metrics']['accuracy_20'] for cat in results.keys()],
        'Test Accuracy 50': [results[cat]['test_metrics']['accuracy_50'] for cat in results.keys()],
        'Test Accuracy 100': [results[cat]['test_metrics']['accuracy_100'] for cat in results.keys()]
    })

    return results, summary_df, residuals, best_models

def ridge_train_log(df, categories, n_splits, ridge_params, random_state=42, date_split='2024-11-16'):
    """
    Similar to ridge_train, but uses a TransformedTargetRegressor with log1p/expm1
    so that the target is log-transformed during training.
    """
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler, FunctionTransformer
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import warnings

    warnings.filterwarnings('ignore')
    
    results = {}
    residuals = {}
    best_models  = {}

    transformer = FunctionTransformer(func=log_transform, inverse_func=inverse_log_transform)

    for category in categories:
        # Prepare data
        X = df.drop(categories, axis=1)
        y = df[category]

        X_train = X[X.index <= date_split]
        y_train = y[y.index <= date_split]
        X_test = X[X.index > date_split]
        y_test = y[y.index > date_split]


        # Time series CV
        tcsv = TimeSeriesSplit(n_splits=n_splits, test_size=30, gap=0)

        scaler = StandardScaler()
        pipeline = Pipeline([
            ('scaler', scaler),
            ('model', TransformedTargetRegressor(
                regressor=Ridge(random_state=random_state),
                transformer=transformer
            ))
        ])

        
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=ridge_params,
            scoring='neg_root_mean_squared_error',
            cv=tcsv,
            refit=True,
            n_jobs=-1,
            verbose=False
        )
        
        grid_search.fit(X_train, y_train)

        y_pred_train = grid_search.predict(X_train)
        y_pred_test = grid_search.predict(X_test)

        # Evaluate metrics on original scale
        train_metrics = {
            'mse': mean_squared_error(y_train, y_pred_train),
            'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'mae': mean_absolute_error(y_train, y_pred_train),
            'r2': r2_score(y_train, y_pred_train),
            'mape': calculate_mape(y_train, y_pred_train),
            'accuracy_20': accuracy_range(20, y_train, y_pred_train),
            'accuracy_50': accuracy_range(50, y_train, y_pred_train),
            'accuracy_100': accuracy_range(100, y_train, y_pred_train)
        }

        test_metrics = {
            'mse': mean_squared_error(y_test, y_pred_test),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'mae': mean_absolute_error(y_test, y_pred_test),
            'r2': r2_score(y_test, y_pred_test),
            'mape': calculate_mape(y_test, y_pred_test),
            'accuracy_20': accuracy_range(20, y_test, y_pred_test),
            'accuracy_50': accuracy_range(50, y_test, y_pred_test),
            'accuracy_100': accuracy_range(100, y_test, y_pred_test)
        }

        results[category] = {
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_, 
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }

        # Residuals in original scale
        residuals[category] = {
            'train_residuals': y_train - y_pred_train,
            'test_residuals': y_test - y_pred_test
        }
        
        best_models[category] = grid_search.best_estimator_

        print(f"\nMetrics for {category}:")
        print("Training Metrics:")
        print(f"RMSE: {train_metrics['rmse']:.4f}")
        print(f"MAPE: {train_metrics['mape']:.4f}")
        print(f"R2: {train_metrics['r2']:.4f}")
        print("\nTest Metrics:")
        print(f"RMSE: {test_metrics['rmse']:.4f}")
        print(f"MAPE: {test_metrics['mape']:.4f}")
        print(f"R2: {test_metrics['r2']:.4f}")

    # Build summary DataFrame
    summary_df = pd.DataFrame({
        'Category': list(results.keys()),
        'Train RMSE': [results[cat]['train_metrics']['rmse'] for cat in results.keys()],
        'Train MAPE': [results[cat]['train_metrics']['mape'] for cat in results.keys()],
        'Train R2': [results[cat]['train_metrics']['r2'] for cat in results.keys()],
        'Train Accuracy 20': [results[cat]['train_metrics']['accuracy_20'] for cat in results.keys()],
        'Train Accuracy 50': [results[cat]['train_metrics']['accuracy_50'] for cat in results.keys()],
        'Train Accuracy 100': [results[cat]['train_metrics']['accuracy_100'] for cat in results.keys()],
        'Test RMSE': [results[cat]['test_metrics']['rmse'] for cat in results.keys()],
        'Test MAPE': [results[cat]['test_metrics']['mape'] for cat in results.keys()],
        'Test R2': [results[cat]['test_metrics']['r2'] for cat in results.keys()],
        'CV RMSE': [results[cat]['best_score'] for cat in results.keys()],
        'Test Accuracy 20': [results[cat]['test_metrics']['accuracy_20'] for cat in results.keys()],
        'Test Accuracy 50': [results[cat]['test_metrics']['accuracy_50'] for cat in results.keys()],
        'Test Accuracy 100': [results[cat]['test_metrics']['accuracy_100'] for cat in results.keys()]
    })

    return results, summary_df, residuals, best_models