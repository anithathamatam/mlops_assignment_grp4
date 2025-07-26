from flask import Flask, request, jsonify
import mlflow
import mlflow.pyfunc
import pandas as pd

app = Flask(__name__)

# Set MLflow tracking URI explicitly
mlflow.set_tracking_uri("file:///C:/Users/ANITHA/Documents/GitHub/mlops_assignment_grp4/mlruns")

# Try loading the model
try:
    model = mlflow.pyfunc.load_model(model_uri="models:/IrisClassifier/latest")
    print("✅ Model loaded successfully from MLflow registry")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None  # Avoid crashing the app on load failure

@app.route("/", methods=["GET"])
def home():
    return "🌼 IrisClassifier API is up and running!"

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        # Get JSON input
        input_json = request.get_json()

        # Validate input
        if not input_json:
            return jsonify({"error": "No input data provided"}), 400

        # Convert input to DataFrame
        input_df = pd.DataFrame([input_json])
        prediction = model.predict(input_df)

        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

if __name__ == "__main__":
    print("🚀 Starting Flask server at http://127.0.0.1:5000")
    app.run(debug=True)
