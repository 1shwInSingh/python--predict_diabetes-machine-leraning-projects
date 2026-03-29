from src.utils import load_config, save_model
from src.data_preprocessing import load_data, clean_data
from src.model_training import train_model
from src.evaluation import evaluate_model

def main():
    config = load_config()

    df = load_data(config["data_path"])
    df = clean_data(df)

    print("Data loaded successfully!\n")

    model, X_test, y_test = train_model(df)

    evaluate_model(model, X_test, y_test)

    save_model(model, config["model_path"])

    print("Model trained and saved successfully!")

if __name__ == "__main__":
    main()