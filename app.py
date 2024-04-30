from flask import Flask, request, jsonify
import json
import pickle

app = Flask(__name__)


with open('crop_prediction.pkl', 'rb') as model_file:
    crop_model = pickle.load(model_file)

with open('fertilizer_prediction.pkl', 'rb') as model_file:
    fertilizer_model = pickle.load(model_file)

with open('encoded_categories.json', 'r') as f:
    encoded_categories = json.load(f)

@app.route('/fertilizer_predict', methods=['POST'])
def fertilizer_predict():
    data = request.get_json()

    temperature = data['Temperature']
    humidity = data['Humidity']
    moisture = data['Moisture']
    soil_type = data['Soil Type']
    crop_type = data['Crop Type']
    nitrogen = data['Nitrogen']
    potassium = data['Potassium']
    phosphorous = data['Phosphorous']

    encoded_soil_type = encoded_categories['Soil Type'].get(soil_type, None)
    encoded_crop_type = encoded_categories['Crop Type'].get(crop_type, None)

    if encoded_soil_type is None:
        return jsonify({'error': f"'{soil_type}' not found in encoded categories for Soil Type."}), 400
    if encoded_crop_type is None:
        return jsonify({'error': f"'{crop_type}' not found in encoded categories for Crop Type."}), 400

    prediction = fertilizer_model.predict([[temperature, humidity, moisture, encoded_soil_type, encoded_crop_type, nitrogen, potassium, phosphorous]])

    predicted_fertilizer_name = prediction[0]

    return jsonify({'predicted_fertilizer_name': predicted_fertilizer_name})


@app.route('/crop_predict', methods=['POST'])
def crop_predict():
    try:
        data = request.json
        N = float(data['N'])
        P = float(data['P'])
        K = float(data['K'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        pH = float(data['ph'])
        rainfall = float(data['rainfall'])

        prediction = crop_model.predict([[N, P, K, temperature, humidity, pH, rainfall]])

        response = {'prediction': prediction[0]}
        return jsonify(response), 200

    except Exception as e:
        error_message = {'error': str(e)}
        return jsonify(error_message), 400

if __name__ == '__main__':
    app.run(debug=True)
