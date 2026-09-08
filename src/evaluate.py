import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np


from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score, f1_score,
    precision_score,
    recall_score
)

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluates a trained classifier on test data by printing classification 
    metrics and plotting the confusion matrix.
    """
    # 1. Generate predictions and probability scores
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    print("=" * 60)
    print(f" EVALUATION REPORT: {model_name.upper()}")
    print("=" * 60)

    # 2. Print Detailed Classification Report
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraudulent"]))

    # 3. Calculate AUC Scores
    if y_proba is not None:
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)

        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print(f"PR-AUC Score (Average Precision): {pr_auc:.4f}")

    # 4. Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 4.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legit (0)', 'Fraud (1)'], 
                yticklabels=['Legit (0)', 'Fraud (1)'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

    return {
        "model_name": model_name,
        "y_pred": y_pred,
        "y_proba": y_proba
    }



def cross_validate_model(estimator, X_train, y_train, cv=5, scoring='average_precision',
                          random_state=42, model_name="Model"):
    """
    Runs stratified k-fold cross-validation and reports mean/std of the scoring metric.
    Pass an unfitted estimator (or an imblearn Pipeline) — it will be refit per fold.
    """
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = cross_val_score(estimator, X_train, y_train, cv=cv_strategy, scoring=scoring, n_jobs=-1)

    print(f"[INFO] {model_name} CV {scoring}: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"[INFO] Fold scores: {np.round(scores, 4)}")

    return {"model_name": model_name, "scores": scores, "mean": scores.mean(), "std": scores.std()}




def tune_threshold(y_test, y_proba, target_recall=None, model_name="Model"):
    """
    Analyzes precision/recall tradeoffs across thresholds and plots the curve.

    If target_recall is provided, finds the lowest threshold that achieves at least
    that recall. Otherwise, reports the threshold that maximizes F1 score.

    Returns:
    dict with 'best_threshold', 'precision', 'recall', 'f1' at that threshold
    """
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    """
    precision_recall_curve returns len(thresholds) = len(precision) - 1
    trim precision/recall to align with thresholds for indexing
    """
    precision_t = precision[:-1]
    recall_t = recall[:-1]

    if target_recall is not None:
        """find lowest threshold that achieves at least target_recall"""
        valid_idx = np.where(recall_t >= target_recall)[0]
        if len(valid_idx) == 0:
            print(f"[WARNING] No threshold achieves recall >= {target_recall}. Using max available recall instead.")
            best_idx = np.argmax(recall_t)
        else:
            """among thresholds achieving target recall, pick the one with highest precision"""
            best_idx = valid_idx[np.argmax(precision_t[valid_idx])]
    else:
        """maximize F1"""
        f1_scores = 2 * (precision_t * recall_t) / (precision_t + recall_t + 1e-10)
        best_idx = np.argmax(f1_scores)

    best_threshold = thresholds[best_idx]
    best_precision = precision_t[best_idx]
    best_recall = recall_t[best_idx]
    best_f1 = 2 * (best_precision * best_recall) / (best_precision + best_recall + 1e-10)

    print(f"[INFO] {model_name} — Selected threshold: {best_threshold:.4f}")
    print(f"[INFO] At this threshold — Precision: {best_precision:.4f}, Recall: {best_recall:.4f}, F1: {best_f1:.4f}")
    print(f"[INFO] Default threshold (0.5) comparison — see classification report above")

    """plot precision & recall vs threshold"""
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precision_t, label="Precision", color="steelblue")
    plt.plot(thresholds, recall_t, label="Recall", color="darkorange")
    plt.axvline(best_threshold, color="green", linestyle="--", label=f"Selected threshold = {best_threshold:.3f}")
    plt.axvline(0.5, color="gray", linestyle=":", label="Default threshold = 0.5")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Precision-Recall vs. Threshold — {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "best_threshold": best_threshold,
        "precision": best_precision,
        "recall": best_recall,
        "f1": best_f1
    }


def apply_threshold(y_proba, threshold):
    """
    Converts predicted probabilities into class predictions using a custom threshold.
    """
    return (y_proba >= threshold).astype(int)


def threshold_sweep_table(y_test, y_proba, thresholds=None):
    """
    Prints precision, recall, and F1 at a handful of candidate thresholds
    for manual inspection and selection.
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    print(f"{'Threshold':<12}{'Precision':<12}{'Recall':<12}{'F1':<12}")
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        p = precision_score(y_test, y_pred_t, zero_division=0)
        r = recall_score(y_test, y_pred_t, zero_division=0)
        f1 = 2 * (p * r) / (p + r + 1e-10)
        print(f"{t:<12}{p:<12.4f}{r:<12.4f}{f1:<12.4f}")


       


def compute_metrics(model, X_test, y_test, model_name="Model", resampling_name="None"):
    """
    Computes precision, recall, F1, PR-AUC, and ROC-AUC without printing or plotting.
    Used for building model comparison tables.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    roc_auc, pr_auc = None, None
    if y_proba is not None:
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)

    return {
        "Model": model_name,
        "Resampling": resampling_name,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "PR-AUC": round(pr_auc, 4) if pr_auc is not None else None,
        "ROC-AUC": round(roc_auc, 4) if roc_auc is not None else None,
    }