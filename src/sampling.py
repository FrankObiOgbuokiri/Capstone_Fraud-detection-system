import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# add to src/sampling.py

def compute_scale_pos_weight(y_train):
    """
    Computes the scale_pos_weight ratio for imbalanced binary classification,
    as expected by XGBoost's and CatBoost's `scale_pos_weight` parameter.

    Ratio = (# negative class) / (# positive class)
    
    Parameters
    ----------
    y_train : array-like of binary labels (0 = negative/majority, 1 = positive/minority)

    Returns
    -------
    float — the scale_pos_weight ratio
    """
    ratio = (len(y_train) - sum(y_train)) / sum(y_train)
    print(f"[INFO] scale_pos_weight computed: {ratio:.4f} "
          f"(negative: {len(y_train) - sum(y_train)}, positive: {sum(y_train)})")
    return ratio



def apply_smote(X_train, y_train, random_state=42):
    """
    Applies SMOTE oversampling to the training data.
    Should only ever be applied to training data, never to test data.

    Returns
    -------
    X_resampled, y_resampled
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"[INFO] SMOTE applied: {len(y_train)} -> {len(y_resampled)} rows "
          f"(fraud cases: {sum(y_train)} -> {sum(y_resampled)})")
    return X_resampled, y_resampled


def apply_rus(X_train, y_train, random_state=42):
    """
    Applies Random Under-Sampling to the training data.
    Should only ever be applied to training data, never to test data.

    Returns
    -------
    X_resampled, y_resampled
    """
    rus = RandomUnderSampler(random_state=random_state)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    print(f"[INFO] RUS applied: {len(y_train)} -> {len(y_resampled)} rows "
          f"(fraud cases: {sum(y_train)} -> {sum(y_resampled)})")
    return X_resampled, y_resampled