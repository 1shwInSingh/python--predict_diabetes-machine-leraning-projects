from flask import Flask, render_template, request, jsonify
from predict_diabetes import predict_diabetes

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    values = [
        float(data['preg']),
        float(data['gluc']),
        float(data['bp']),
        float(data['skin']),
        float(data['ins']),
        float(data['bmi']),
        float(data['dpf']),
        float(data['age'])
    ]

    result = predict_diabetes(values)

    return jsonify({
        "result": "Diabetic" if result == 1 else "Not Diabetic"
    })

if __name__ == '__main__':
    app.run(debug=True)