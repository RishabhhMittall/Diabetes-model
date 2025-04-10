from flask import Flask, request, jsonify
# from flask_cors import CORS
import pickle
import numpy as np

# Load the trained diabetes model and scaler
with open("diabetes_model.pkl", "rb") as model_file:
    diabetes_model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Initialize Flask app
app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

@app.route("/")
def home():
    return "Welcome to the Diabetes Prediction API! Use the /predict endpoint with POST to get results."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.get_json()

        # Required features
        features = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]

        # Check if all features are provided
        if not all(feature in input_data for feature in features):
            return jsonify({"error": "Missing required features in input data"}), 400

        # Convert to numpy array
        input_values = [float(input_data[feature]) for feature in features]
        input_array = np.array([input_values])

        # Standardize input
        std_data = scaler.transform(input_array)

        # Predict
        prediction = diabetes_model.predict(std_data)

        # Return prediction result
        result = "The person is diabetic" if prediction[0] == 1 else "The person is not diabetic"
        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)
