import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    brier_score_loss,
    f1_score,
)


def find_optimal_threshold(y_true, probabilities, target_recall=0.88):
    """
    Find the probability threshold that achieves the target recall for the 
    positive class (diabetes), prioritizing sensitivity (catching cases).
    Searches 0.05 – 0.50 in steps of 0.01.
    """
    from sklearn.metrics import recall_score
    best_thresh = 0.5
    for t in np.arange(0.05, 0.51, 0.01):
        preds = (probabilities >= t).astype(int)
        recall = recall_score(y_true, preds, pos_label=1, zero_division=0)
        if recall >= target_recall:
            best_thresh = t
    return round(float(best_thresh), 2)


def evaluate_model(model, X_test_scaled, y_test, threshold=None):
    """
    Full evaluation: accuracy, ROC-AUC, Brier score, per-class metrics.
    If threshold is None the optimal one is derived from the data.
    Returns (accuracy, roc_auc, optimal_threshold).
    """
    probabilities = model.predict_proba(X_test_scaled)[:, 1]

    if threshold is None:
        threshold = find_optimal_threshold(y_test, probabilities)

    predictions = (probabilities >= threshold).astype(int)

    accuracy  = accuracy_score(y_test, predictions)
    roc_auc   = roc_auc_score(y_test, probabilities)
    brier     = brier_score_loss(y_test, probabilities)

    print(f"\n{'='*55}")
    print(f"  MODEL EVALUATION  (threshold = {threshold:.2f})")
    print(f"{'='*55}")
    print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print(f"  Brier Scr : {brier:.4f}  (lower = better calibrated)")
    print(f"{'='*55}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test, predictions,
            target_names=["No Diabetes", "Diabetes/Pre-diabetes"],
        )
    )

    cm = confusion_matrix(y_test, predictions)
    print("Confusion Matrix:")
    print(f"  TN={cm[0][0]:>6}  FP={cm[0][1]:>6}")
    print(f"  FN={cm[1][0]:>6}  TP={cm[1][1]:>6}")

    tn, fp, fn, tp = cm.ravel()
    recall    = tp / (tp + fn) if (tp + fn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    print(f"\n  Diabetic Recall    : {recall:.3f}  (of all diabetics, % caught)")
    print(f"  Diabetic Precision : {precision:.3f}  (of flagged cases, % correct)")
    print()

    return accuracy, roc_auc, threshold