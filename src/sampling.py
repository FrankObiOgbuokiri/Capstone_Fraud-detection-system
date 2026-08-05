import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def get_class_weights(y_train):
    """
    Computes balanced class weights for cost-sensitive learning.
    
    Returns:
        dict: Class weight mapping (e.g., {0: weight_0, 1: weight_1})
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    
    class_weights_dict = dict(zip(classes, weights))
    print(f"[INFO] Computed Class Weights: {class_weights_dict}")
    return class_weights_dict


def apply_smote(X_train, y_train, random_state=42):
    """
    Applies Synthetic Minority Over-sampling Technique (SMOTE) 
    only to training data to balance class counts.
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print("[INFO] Applied SMOTE Resampling:")
    print(f"       Original distribution: {np.bincount(y_train)}")
    print(f"       Resampled distribution: {np.bincount(y_resampled)}")
    
    return X_resampled, y_resampled


def apply_random_undersampling(X_train, y_train, sampling_strategy=1.0, random_state=42):
    """
    Applies Random Under-Sampling (RUS) only to training data 
    to reduce majority class instances.
    """
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    
    print("[INFO] Applied Random Under-Sampling:")
    print(f"       Original distribution: {np.bincount(y_train)}")
    print(f"       Resampled distribution: {np.bincount(y_resampled)}")
    
    return X_resampled, y_resampled