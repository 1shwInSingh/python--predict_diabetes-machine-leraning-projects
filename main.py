from src.utils import load_config, save_model
from src.data_preprocessing import load_data, clean_data
from src.model_training import train_model
from src.evaluation import evaluate_model
import joblib
import os


def main():
    config = load_config()

    print("=" * 55)
    print("  DIABETES PREDICTION - MODEL TRAINING")
    print("=" * 55)

    # Load and clean data
    print("\n[1/4] Loading and cleaning data...")
    df = load_data(config["data_path"])
    df = clean_data(df)
    print(f"      Dataset shape after cleaning: {df.shape}")

    # Train ensemble pipeline (returns threshold instead of scaler)
    print("\n[2/4] Training ensemble pipeline...")
    pipeline, X_test, _, y_test, optimal_threshold = train_model(df)

    # Evaluate
    print("\n[3/4] Evaluating model...")
    accuracy, roc_auc, _ = evaluate_model(pipeline, X_test, y_test, threshold=optimal_threshold)

    # Save pipeline + threshold
    print("\n[4/4] Saving pipeline and threshold...")
    os.makedirs("models", exist_ok=True)

    pipeline_path  = config.get("pipeline_path", "models/pipeline.pkl")
    threshold_path = "models/threshold.pkl"

    joblib.dump(pipeline,          pipeline_path)
    joblib.dump(optimal_threshold, threshold_path)

    # Also save under legacy model_path so the app can still find it
    legacy_path = config.get("model_path", "models/model.pkl")
    if legacy_path != pipeline_path:
        joblib.dump(pipeline, legacy_path)

    print(f"      Pipeline  -> {pipeline_path}")
    print(f"      Threshold -> {threshold_path}  (value: {optimal_threshold:.2f})")
    print(f"      Legacy    -> {legacy_path}")
    print("\nAll done!")


if __name__ == "__main__":
    main()