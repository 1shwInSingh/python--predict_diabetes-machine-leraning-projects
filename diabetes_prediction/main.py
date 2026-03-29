from src.utils import load_config, save_model
from src.data_preprocessing import load_data, clean_data
from src.model_training import train_model
from src.evaluation import evaluate_model
from src.predict import load_model, predict

def main():
    config = load_config()

    df = load_data(config["data_path"])
    df = clean_data(df)

    # train model
    model, X_test, y_test = train_model(df)

    # evaluate
    evaluate_model(model, X_test, y_test)

    # save model
    save_model(model, config["model_path"])

    print("Model trained and saved successfully!")

    # 🔮 Prediction part
    model = load_model(config["model_path"])

    # sample input 
    sample_input = X_test.iloc[0].tolist()

    result = predict(model, sample_input)

    print("\nPrediction for sample input:", result)

if __name__ == "__main__":
    main()