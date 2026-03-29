import joblib

def load_model(path):
    return joblib.load(path)

def predict(model, input_data):
    prediction = model.predict([input_data])
    return prediction[0]