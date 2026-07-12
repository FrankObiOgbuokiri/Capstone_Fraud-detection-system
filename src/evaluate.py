from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC (Important for Imbalance): {pr_auc:.4f}")
    return {"roc_auc": roc_auc, "pr_auc": pr_auc}
