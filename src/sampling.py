import numpy as np
from sklearn.utils.class_weight import compute_class_weight


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

