from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    # scale_pos_weight helps with imbalance
    ratio = float(y_train.value_counts()[0]) / y_train.value_counts()[1]
    model = XGBClassifier(scale_pos_weight=ratio, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, filepath="fraud_model.pkl"):
    joblib.dump(model, filepath)