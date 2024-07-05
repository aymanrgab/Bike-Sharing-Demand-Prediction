from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def fetch_data():
    """Fetch Bike Sharing dataset from UCI ML Repository."""
    bike_sharing = fetch_ucirepo(id=275)
    X = bike_sharing.data.features
    y = bike_sharing.data.targets
    return X, y

def preprocess_data(X, y):
    """Preprocess the data with advanced feature engineering."""
    # Combine features and target
    df = pd.concat([X, y], axis=1)
    
    print("Columns in the dataset:", df.columns.tolist())
    
    # Convert date to datetime
    df['dteday'] = pd.to_datetime(df['dteday'])
    
    # Create new features
    df['year'] = df['dteday'].dt.year
    df['month'] = df['dteday'].dt.month
    df['day_of_week'] = df['dteday'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Create cyclic features for month and hour
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['hour_sin'] = np.sin(2 * np.pi * df['hr']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hr']/24)
    
    # Interaction features
    df['temp_atemp_interaction'] = df['temp'] * df['atemp']
    df['hum_windspeed_interaction'] = df['hum'] * df['windspeed']
    
    # Log transform for skewed continuous variables
    df['windspeed_log'] = np.log1p(df['windspeed'])
    
    # Time-based features
    df['rush_hour'] = ((df['hr'] >= 7) & (df['hr'] <= 9) | (df['hr'] >= 16) & (df['hr'] <= 18)).astype(int)
    
    # Weather-based features
    df['extreme_weather'] = (df['weathersit'] >= 3).astype(int)
    
    # Drop unnecessary columns
    df = df.drop(['dteday'], axis=1)
    
    # Convert categorical variables to dummy variables
    categorical_columns = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
    df = pd.get_dummies(df, columns=categorical_columns)
    
    return df

def split_and_scale_data(df, target_column, test_size=0.2, random_state=42):
    """Split the data into training and testing sets and scale features."""
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns

def main():
    # Fetch data
    X, y = fetch_data()
    
    # Preprocess data
    df_processed = preprocess_data(X, y)
    
    # Split and scale data
    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_scale_data(df_processed, 'cnt')
    
    # Save processed data
    pd.DataFrame(X_train, columns=feature_names).to_csv('data/processed/X_train.csv', index=False)
    pd.DataFrame(X_test, columns=feature_names).to_csv('data/processed/X_test.csv', index=False)
    pd.Series(y_train, name='cnt').to_csv('data/processed/y_train.csv', index=False)
    pd.Series(y_test, name='cnt').to_csv('data/processed/y_test.csv', index=False)
    
    # Save scaler and feature names
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')

    print("Data preprocessing completed successfully.")
    print(f"Number of features: {len(feature_names)}")
    print(f"Number of training samples: {len(y_train)}")
    print(f"Number of test samples: {len(y_test)}")

if __name__ == '__main__':
    main()