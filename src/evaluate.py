import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc
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
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)

        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print(f"PR-AUC Score (Precision-Recall Area): {pr_auc:.4f}")

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