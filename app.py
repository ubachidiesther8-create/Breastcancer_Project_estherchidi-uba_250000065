from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model
app = Flask(__name__)
model = load_model("model/breast_cancer_nn_model.keras")
scaler = joblib.load("scaler.save")
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probability = None

    if request.method == "POST":
        try:
            input_features = [
                float(request.form["radius_mean"]),
                float(request.form["texture_mean"]),
                float(request.form["perimeter_mean"]),
                float(request.form["area_mean"]),
                float(request.form["concavity_mean"])
            ]
            input_array = np.array(input_features).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            pred_prob = model.predict(input_scaled)[0][0]
            pred_class = 1 if pred_prob > 0.5 else 0
            result = "Malignant" if pred_class == 1 else "Benign"
            probability = f"{pred_prob * 100:.2f}%"
        except Exception as e:
            result = f"Error: {str(e)}"
    return render_template(
        "index.html",
        result=result,
        probability=probability
    )
if __name__ == "__main__":
    app.run(debug=True)
