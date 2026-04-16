from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.data_preprocessing import add_features
from src.evaluation import find_optimal_threshold


def train_model(df):
    """
    Full training pipeline:
      1. Feature engineering
      2. Stratified train/test split
      3. SMOTE oversampling (inside pipeline — applied to train only)
      4. StandardScaler
      5. Soft-voting ensemble: RF + GBM + LR
      6. Isotonic calibration for reliable probabilities
      7. Auto-detect optimal F1 threshold
    Returns (fitted_pipeline, X_test, X_test_raw, y_test, optimal_threshold)
    """
    # --- Feature engineering ---
    df = add_features(df)

    # Binary target: 0 = no diabetes, 1 = pre-diabetes or diabetes
    df["Diabetes_binary"] = (df["Diabetes_012"] > 0).astype(int)
    y = df["Diabetes_binary"]
    X = df.drop(["Diabetes_012", "Diabetes_binary"], axis=1)

    # --- Stratified split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")
    print(f"Class distribution (train) — No diabetes: {(y_train==0).sum():,}  Diabetes: {(y_train==1).sum():,}")

    # --- Base classifiers ---
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    gbm = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=5,
        min_samples_leaf=20,
        subsample=0.8,
        max_features="sqrt",
        random_state=42,
        verbose=0,
    )

    lr = LogisticRegression(
        C=1.0,
        max_iter=500,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
        n_jobs=-1,
    )

    # --- Soft-voting ensemble ---
    voting = VotingClassifier(
        estimators=[("rf", rf), ("gbm", gbm), ("lr", lr)],
        voting="soft",
        weights=[2, 2, 1],   # RF and GBM slightly preferred
        n_jobs=-1,
    )

    # --- Full imbalanced-learn pipeline: SMOTE → Scaler → Calibrated Ensemble ---
    pipeline = ImbPipeline(steps=[
        ("smote",  SMOTE(random_state=42, sampling_strategy=0.7)),
        ("scaler", StandardScaler()),
        ("model",  CalibratedClassifierCV(voting, method="isotonic", cv=3)),
    ])

    print("\nFitting ensemble pipeline (this may take a few minutes)…")
    pipeline.fit(X_train, y_train)
    print("Done training.")

    # --- Find optimal threshold on held-out test set ---
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    optimal_threshold = find_optimal_threshold(y_test, probabilities)
    print(f"Optimal classification threshold (F1-maximised): {optimal_threshold:.2f}")

    return pipeline, X_test, X_test, y_test, optimal_threshold