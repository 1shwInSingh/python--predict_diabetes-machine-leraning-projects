from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

def evaluate_model(model, X_test_scaled, y_test):
    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)
    
    print(f"\n{'='*50}")
    print(f"  MODEL EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"{'='*50}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=["No Diabetes", "Diabetes/Pre-diabetes"]))
    
    cm = confusion_matrix(y_test, predictions)
    print("Confusion Matrix:")
    print(f"  TN={cm[0][0]:>6}  FP={cm[0][1]:>6}")
    print(f"  FN={cm[1][0]:>6}  TP={cm[1][1]:>6}")
    print()
    
    return accuracy, roc_auc