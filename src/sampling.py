from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X_train, y_train)

def apply_undersampling(X_train, y_train):
    rus = RandomUnderSampler(random_state=42)
    return rus.fit_resample(X_train, y_train)