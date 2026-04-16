from src.utils import load_config, save_model
from src.data_preprocessing import load_data, clean_data
from src.model_training import train_model
from src.evaluation import evaluate_model
from src.predict import load_model, predict
import joblib

def main():
    config = load_config()

    # Load and clean data
    df = load_data(config["data_path"])
    df = clean_data(df)

    # Train model (now returns scaler too)
    model, X_test, X_test_scaled, y_test, scaler = train_model(df)

    # Evaluate with comprehensive metrics
    accuracy, roc_auc = evaluate_model(model, X_test_scaled, y_test)

    # Save model and scaler
    save_model(model, config["model_path"])
    joblib.dump(scaler, "models/scaler.pkl")

    print("Model and scaler saved successfully!")

    # Prediction demo
    model, scaler = load_model(config["model_path"])
    
    # Sample input (use raw features from test set)
    sample_input = X_test.iloc[0].tolist()
    result = predict(model, scaler, sample_input)

    print("\nPrediction for sample input:")
    if result['prediction'] == 1:
        print("  Result: [WARNING] Diabetes Risk")
    else:
        print("  Result: [OK] No Diabetes")
    print(f"  Confidence: {max(result['probability_diabetes'], result['probability_no_diabetes'])}%")

if __name__ == "__main__":
    main()