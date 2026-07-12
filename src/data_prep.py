import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def load_and_inspect(filepath):
    df = pd.read_csv(filepath)
    print(f"Shape: {df.shape}")
    print(f"Class Distribution:\n{df['Class'].value_counts(normalize=True)}")
    return df

def scale_features(df, fit=True, scaler_path="scaler.pkl"):
    # Assumes target column is 'Class'
    features = df.drop(columns=['Class'], errors='ignore')
    
    if fit:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features[['Time', 'Amount']])
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)
        scaled_features = scaler.transform(features[['Time', 'Amount']])
        
    df_scaled = df.copy()
    df_scaled[['Time', 'Amount']] = scaled_features
    return df_scaled

def split_data(df, target_col='Class', test_size=0.2):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)