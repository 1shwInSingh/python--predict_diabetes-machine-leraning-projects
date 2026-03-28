from src.utils import load_config
from src.data_preprocessing import load_data, clean_data

def main():
    config = load_config()

    df = load_data(config["data_path"])
    df = clean_data(df)

    print("Data loaded successfully!\n")
    
    print("Columns in dataset:")
    print(df.columns)

    print("\nFirst 5 rows:")
    print(df.head())

if __name__ == "__main__":
    main()

#calling for  model training

from src.utils import load_config, save_model
from src.data_preprocessing import load_data, clean_data
from src.model_training import train_model

def main():
    config = load_config()

    df = load_data(config["data_path"])
    df = clean_data(df)

    model = train_model(df)

    save_model(model, config["model_path"])

    print("Model trained and saved successfully!")

if __name__ == "__main__":
    main()