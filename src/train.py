from xgboost import XGBClassifier


def train_xgboost(X_train, y_train, scale_pos_weight=1, random_state=42):
    """
    Trains an XGBoost classifier.
    scale_pos_weight handles class imbalance in XGBoost instead of class_weight.
    """
    model = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=random_state, eval_metric='logloss')
    model.fit(X_train, y_train)
    print("[INFO] Trained XGBoost model.")
    return model
