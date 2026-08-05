from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def train_logistic_regression(X_train, y_train, class_weight=None, random_state=42):
    """Trains a Logistic Regression model."""
    model = LogisticRegression(class_weight=class_weight, random_state=random_state, max_iter=1000)
    model.fit(X_train, y_train)
    print("[INFO] Trained Logistic Regression model.")
    return model


def train_random_forest(X_train, y_train, class_weight=None, random_state=42):
    """Trains a Random Forest classifier."""
    model = RandomForestClassifier(class_weight=class_weight, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    print("[INFO] Trained Random Forest model.")
    return model


def train_xgboost(X_train, y_train, scale_pos_weight=1, random_state=42):
    """
    Trains an XGBoost classifier.
    scale_pos_weight handles class imbalance in XGBoost instead of class_weight.
    """
    model = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=random_state, eval_metric='logloss')
    model.fit(X_train, y_train)
    print("[INFO] Trained XGBoost model.")
    return model


def train_lightgbm(X_train, y_train, class_weight=None, random_state=42):
    """Trains a LightGBM classifier."""
    model = LGBMClassifier(class_weight=class_weight, random_state=random_state, verbose=-1)
    model.fit(X_train, y_train)
    print("[INFO] Trained LightGBM model.")
    return model


def train_catboost(X_train, y_train, auto_class_weights='Balanced', random_state=42):
    """
    Trains a CatBoost classifier.
    Supports auto_class_weights='Balanced' for handling imbalanced target distributions.
    """
    model = CatBoostClassifier(
        auto_class_weights=auto_class_weights, 
        random_seed=random_state, 
        verbose=0
    )
    model.fit(X_train, y_train)
    print("[INFO] Trained CatBoost model.")
    return model