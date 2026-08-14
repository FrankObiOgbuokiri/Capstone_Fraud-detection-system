import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Loads a CSV dataset from a given file path.
def load_data(file_path):
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not file_path.endswith(".csv"):
        raise ValueError("Only CSV files are supported.")
    
    df = pd.read_csv(file_path)
    print(f"CSV data loaded successfully with shape: {df.shape}")
    return df

def inspect_data(df):
    """Prints basic summary information about the dataset."""
    print("--- FIRST 5 ROWS ---")
    print(df.head())
    
    print("\n--- DATA INFO ---")
    print(df.info())
    
    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum())
    
    print("\n--- SUMMARY STATISTICS ---")
    print(df.describe())

def feature_engineering(df):
    """
    Creates new features if needed.
    Example: Log-transforms the 'Amount' column.
    """
    df_engineered = df.copy()
    
    # Log-transform Amount to handle skewed values
    if "Amount" in df_engineered.columns:
        import numpy as np
        df_engineered["Amount_log"] = np.log1p(df_engineered["Amount"])
        print("Created feature: 'Amount_log'")
        
    return df_engineered

def split_data(df, target_column='Class', test_size=0.2, random_state=42):
    """Splits the dataset into features (X) and target (y), then into train and test sets."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Train set: {X_train.shape} | Test set: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test, columns_to_scale):
    """Scales specified numerical columns using StandardScaler."""
    scaler = StandardScaler()
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Scale training data and apply the same scale to test data
    X_train_scaled[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
    X_test_scaled[columns_to_scale] = scaler.transform(X_test[columns_to_scale])
    
    print(f"Scaled columns: {columns_to_scale}")
    return X_train_scaled, X_test_scaled, scaler

