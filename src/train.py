# src/train.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def train_xgboost(X_train, y_train, scale_pos_weight=1, random_state=42):
    """
    Trains an XGBoost classifier.
    scale_pos_weight handles class imbalance in XGBoost instead of class_weight.
    """
    model = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=random_state, eval_metric='logloss')
    model.fit(X_train, y_train)
    print("[INFO] Trained XGBoost model.")
    return model


def tune_xgboost_class_weighted(X_train, y_train, scale_pos_weight, param_distributions=None,
                                 n_iter=20, cv=5, random_state=42, scoring='average_precision'):
    """
    Tunes XGBoost hyperparameters with class-weighted imbalance handling.
    scale_pos_weight is fixed (not searched) since it's determined by class ratio, not a tunable choice.
    """
    if param_distributions is None:
        param_distributions = {
            'max_depth': [3, 5, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': [100, 300, 500],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0]
        }

    base_model = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=random_state, eval_metric='logloss')
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        base_model, param_distributions, n_iter=n_iter, cv=cv_strategy,
        scoring=scoring, random_state=random_state, n_jobs=-1
    )
    search.fit(X_train, y_train)

    print(f"[INFO] XGBoost (class-weighted) best params: {search.best_params_}")
    print(f"[INFO] XGBoost (class-weighted) best CV {scoring}: {search.best_score_:.4f}")
    return search.best_estimator_, search.best_params_


def tune_random_forest_smote(X_train, y_train, param_distributions=None,
                              n_iter=20, cv=5, random_state=42, scoring='average_precision'):
    """
    Tunes Random Forest hyperparameters with SMOTE applied inside each CV fold.
    Uses imblearn's Pipeline so SMOTE is refit on each fold's training portion only,
    preventing synthetic-sample leakage into validation folds.

    NOTE: pass the ORIGINAL (unresampled) X_train, y_train here, not pre-resampled data.
    SMOTE happens inside the pipeline per-fold.
    """
    if param_distributions is None:
        param_distributions = {
            'rf__n_estimators': [100, 300, 500],
            'rf__max_depth': [5, 10, 20, None],
            'rf__min_samples_split': [2, 5, 10],
            'rf__min_samples_leaf': [1, 2, 4]
        }

    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=random_state)),
        ('rf', RandomForestClassifier(random_state=random_state))
    ])
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        pipeline, param_distributions, n_iter=n_iter, cv=cv_strategy,
        scoring=scoring, random_state=random_state, n_jobs=-1, verbose=3
    )
    
    search.fit(X_train, y_train)

    print(f"[INFO] Random Forest (SMOTE) best params: {search.best_params_}")
    print(f"[INFO] Random Forest (SMOTE) best CV {scoring}: {search.best_score_:.4f}")
    return search.best_estimator_, search.best_params_




def build_experiments(X_train_scaled, y_train, X_train_smote, y_train_smote,
                       X_train_rus, y_train_rus, scale_pos_w, random_state=42):
    """
    Builds the experiments dict used by run_model_comparison: 3 resampling
    strategies (RUS, SMOTE, Class Weighted) x 5 models each.
    """
    return {
        'RUS': {
            'X_tr': X_train_rus,
            'y_tr': y_train_rus,
            'models': {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state),
                'Random Forest': RandomForestClassifier(random_state=random_state, n_jobs=-1),
                'XGBoost': XGBClassifier(random_state=random_state, eval_metric='logloss', n_jobs=-1),
                'LightGBM': LGBMClassifier(random_state=random_state, verbose=-1, n_jobs=-1),
                'CatBoost': CatBoostClassifier(random_state=random_state, verbose=0)
            }
        },
        'SMOTE': {
            'X_tr': X_train_smote,
            'y_tr': y_train_smote,
            'models': {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state),
                'Random Forest': RandomForestClassifier(random_state=random_state, n_jobs=-1),
                'XGBoost': XGBClassifier(random_state=random_state, eval_metric='logloss', n_jobs=-1),
                'LightGBM': LGBMClassifier(random_state=random_state, verbose=-1, n_jobs=-1),
                'CatBoost': CatBoostClassifier(random_state=random_state, verbose=0)
            }
        },
        'Class Weighted': {
            'X_tr': X_train_scaled,
            'y_tr': y_train,
            'models': {
                'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=random_state),
                'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=random_state, n_jobs=-1),
                'XGBoost': XGBClassifier(scale_pos_weight=scale_pos_w, eval_metric='logloss', random_state=random_state, n_jobs=-1),
                'LightGBM': LGBMClassifier(class_weight='balanced', random_state=random_state, verbose=-1, n_jobs=-1),
                'CatBoost': CatBoostClassifier(scale_pos_weight=scale_pos_w, random_state=random_state, verbose=0)
            }
        }
    }


def run_model_comparison(experiments, X_test_scaled, y_test, plot=True):
    """
    Trains and evaluates every model/strategy combination in `experiments`,
    returning a comparison DataFrame sorted by PR-AUC. Optionally plots all
    precision-recall curves on one chart for visual comparison.

    Parameters
    ----------
    experiments : dict, as built by build_experiments()
    X_test_scaled, y_test : held-out test data
    plot : bool, whether to render the combined PR-curve plot

    Returns
    -------
    pd.DataFrame — comparison table, sorted by PR-AUC descending
    """
    all_results = []

    if plot:
        plt.figure(figsize=(10, 6))

    for strategy_name, exp in experiments.items():
        X_tr, y_tr = exp['X_tr'], exp['y_tr']

        for model_name, model in exp['models'].items():
            print(f"[INFO] Training {model_name} ({strategy_name})...")
            model.fit(X_tr, y_tr)

            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

            if y_proba is not None:
                pr_auc_val = average_precision_score(y_test, y_proba)
                if plot:
                    precision_pts, recall_pts, _ = precision_recall_curve(y_test, y_proba)
                    full_model_label = f"{model_name} ({strategy_name})"
                    plt.plot(recall_pts, precision_pts, label=f"{full_model_label} (PR-AUC = {pr_auc_val:.3f})")
            else:
                pr_auc_val = np.nan

            all_results.append({
                'Strategy': strategy_name,
                'Model': model_name,
                'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
                'Precision': prec,
                'Recall': rec,
                'F1-Score': f1,
                'ROC-AUC': roc_auc,
                'PR-AUC': pr_auc_val
            })

    comparison_df = pd.DataFrame(all_results).sort_values(by='PR-AUC', ascending=False).reset_index(drop=True)

    print("=" * 80)
    print(" Comprehensive Model Evaluation Summary (Sorted by PR-AUC)")
    print("=" * 80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if plot:
        plt.title('Precision-Recall Curves Across Models & Strategies', fontsize=14)
        plt.xlabel('Recall (Sensitivity)', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.tight_layout()
        plt.show()

    return comparison_df