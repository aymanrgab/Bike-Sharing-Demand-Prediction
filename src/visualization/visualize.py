import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.inspection import PartialDependenceDisplay

def load_data(filepath):
    """Load data from filepath."""
    return pd.read_csv(filepath)

def load_model(model_path):
    """Load the trained model."""
    return joblib.load(model_path)

def plot_feature_importance(model, feature_names, top_n=20):
    """Plot feature importance."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        raise ValueError("Model doesn't have feature_importances_ or coef_ attribute")
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(top_n))
    plt.title(f'Top {top_n} Feature Importance')
    plt.tight_layout()
    plt.savefig('reports/figures/feature_importance.png')
    plt.close()

def plot_predictions_vs_actual(y_true, y_pred):
    """Plot predictions vs actual values."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predictions vs Actual')
    plt.tight_layout()
    plt.savefig('reports/figures/predictions_vs_actual.png')
    plt.close()

def plot_residuals(y_true, y_pred):
    """Plot residuals."""
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.tight_layout()
    plt.savefig('reports/figures/residuals.png')
    plt.close()

def plot_partial_dependence(model, X, feature_names):
    """Plot partial dependence for top features."""
    top_features = feature_names[:5]  # Adjust the number of features as needed
    fig, ax = plt.subplots(figsize=(12, 8))
    PartialDependenceDisplay.from_estimator(model, X, top_features, ax=ax)
    plt.tight_layout()
    plt.savefig('reports/figures/partial_dependence.png')
    plt.close()

def plot_correlation_heatmap(df):
    """Plot correlation heatmap."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('reports/figures/correlation_heatmap.png')
    plt.close()

def plot_target_distribution(y):
    """Plot target variable distribution."""
    plt.figure(figsize=(10, 6))
    sns.histplot(y, kde=True)
    plt.title('Distribution of Target Variable')
    plt.xlabel('Bike Rentals Count')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('reports/figures/target_distribution.png')
    plt.close()

def main():
    # Load data
    X_train = load_data('data/processed/X_train.csv')
    y_train = load_data('data/processed/y_train.csv').values.ravel()
    X_test = load_data('data/processed/X_test.csv')
    y_test = load_data('data/processed/y_test.csv').values.ravel()
    
    # Load model and make predictions
    model = load_model('models/best_model.pkl')
    y_pred = model.predict(X_test)
    
    # Generate visualizations
    plot_feature_importance(model, X_train.columns)
    plot_predictions_vs_actual(y_test, y_pred)
    plot_residuals(y_test, y_pred)
    plot_partial_dependence(model, X_test, X_test.columns)
    plot_correlation_heatmap(X_train)
    plot_target_distribution(y_train)
    
    print("Visualizations have been saved in the 'reports/figures/' directory.")

if __name__ == '__main__':
    main()