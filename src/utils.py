import yaml
import os
import joblib
def load_config():
    base_path = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(base_path, "config", "config.yaml")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def save_model(model, path):
    joblib.dump(model, path)