import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def load_data(X_path, y_path):
    """Load featured data from filepath."""
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).values.ravel()
    return X, y

def train_and_evaluate_model(model, X, y, cv=5):
    """Train a model and evaluate its performance using cross-validation."""
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-cv_scores)
    return model.fit(X, y), rmse_scores.mean(), rmse_scores.std()

def main():
    # Load data
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        trained_model, rmse_mean, rmse_std = train_and_evaluate_model(model, X_train, y_train)
        results[name] = {'model': trained_model, 'rmse_mean': rmse_mean, 'rmse_std': rmse_std}
        print(f"{name} - RMSE: {rmse_mean:.2f} (+/- {rmse_std:.2f})")
    
    # Find best model
    best_model_name = min(results, key=lambda x: results[x]['rmse_mean'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBest model: {best_model_name}")
    print(f"Best RMSE: {results[best_model_name]['rmse_mean']:.2f}")
    
    # Evaluate best model on test set
    y_pred = best_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 = r2_score(y_test, y_pred)
    
    print(f"\nTest set performance:")
    print(f"RMSE: {test_rmse:.2f}")
    print(f"R2 Score: {test_r2:.2f}")
    
    # Save best model and feature names
    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(X_train.columns.tolist(), 'models/feature_names.pkl')

if __name__ == '__main__':
    main()