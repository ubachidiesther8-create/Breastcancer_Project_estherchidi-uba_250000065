import os
from flask import Flask, render_template, request
import numpy as np
import joblib
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
MODEL_PATH = "model/breast_cancer_nn_model.keras"
SCALER_PATH = "scaler.save"
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
FEATURES = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "concavity_mean"]
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probability = None
    if request.method == "POST":
        try:
            input_values = [float(request.form[f]) for f in FEATURES]
            input_scaled = scaler.transform([input_values])
            pred_prob = model.predict(input_scaled, verbose=0)[0][0]
            pred_class = 1 if pred_prob > 0.5 else 0
            result = "Malignant" if pred_class == 1 else "Benign"
            probability = f"{pred_prob*100:.2f}%"
        except Exception as e:
            result = f"Error: {e}"
    return render_template("index.html", result=result, probability=probability)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)