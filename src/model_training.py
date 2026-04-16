from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from src.data_preprocessing import add_features

def train_model(df):
    # Engineer features
    df = add_features(df)
    
    # Create binary target (0 = no diabetes, 1 = prediabetes or diabetes)
    df["Diabetes_binary"] = df["Diabetes_012"].apply(lambda x: 1 if x > 0 else 0)

    y = df["Diabetes_binary"]
    X = df.drop(["Diabetes_012", "Diabetes_binary"], axis=1)

    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handle class imbalance with SMOTE — oversample to 70% of majority class
    smote = SMOTE(random_state=42, sampling_strategy=0.7)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    print(f"Training samples after SMOTE: {len(X_train_resampled)}")

    # Gradient Boosting Classifier with tuned hyperparameters
    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=6,
        min_samples_split=40,
        min_samples_leaf=15,
        subsample=0.85,
        max_features='sqrt',
        random_state=42,
        verbose=1
    )

    model.fit(X_train_resampled, y_train_resampled)
    
    return model, X_test, X_test_scaled, y_test, scaler