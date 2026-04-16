import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    brier_score_loss,
    f1_score,
)


def find_optimal_threshold(y_true, probabilities):
    """
    Find the probability threshold that maximises F1 for the positive (diabetic) class.
    Searches 0.20 – 0.65 in steps of 0.01.
    """
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.20, 0.65, 0.01):
        preds = (probabilities >= t).astype(int)
        f1 = f1_score(y_true, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
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