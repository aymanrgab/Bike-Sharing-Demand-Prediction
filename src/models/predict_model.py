import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score

def load_data(filepath):
    """Load data from filepath."""
    return pd.read_csv(filepath)

def load_model(model_path):
    """Load the trained model."""
    return joblib.load(model_path)

def load_feature_names(feature_names_path):
    """Load the feature names used during training."""
    return joblib.load(feature_names_path)

def preprocess_data(df, feature_names):
    """Preprocess the data for prediction."""
    # Ensure all expected features are present
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0  # or some other appropriate default value

    # Reorder columns to match the order used during training
    return df[feature_names]

def make_predictions(model, X):
    """Make predictions using the trained model."""
    return model.predict(X)

def evaluate_predictions(y_true, y_pred):
    """Evaluate the predictions."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return rmse, r2

def main():
    # Load data
    X_test = load_data('data/processed/X_test.csv')
    y_test = load_data('data/processed/y_test.csv').values.ravel()
    
    # Load model and feature names
    model = load_model('models/best_model.pkl')
    feature_names = load_feature_names('models/feature_names.pkl')
    
    # Preprocess data
    X_test = preprocess_data(X_test, feature_names)
    
    # Make predictions
    predictions = make_predictions(model, X_test)
    
    # Evaluate predictions
    rmse, r2 = evaluate_predictions(y_test, predictions)
    
    print(f"Model performance on test set:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.2f}")
    
    # Add predictions to test dataframe
    X_test['predictions'] = predictions
    
    # Save results
    X_test.to_csv('data/processed/test_results.csv', index=False)
    
    print("Predictions saved to 'data/processed/test_results.csv'")

if __name__ == '__main__':
    main()