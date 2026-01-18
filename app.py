from flask import Flask, render_template, request
import numpy as np
import joblib
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# Load saved model and scaler
model = tf.keras.models.load_model("breast_cancer_model.keras")
scaler = joblib.load("scaler.save")

# Home route
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probability = None

    if request.method == "POST":
        try:
            # Get input values from form
            input_features = [
                float(request.form["clump_thickness"]),
                float(request.form["cell_size"]),
                float(request.form["cell_shape"]),
                float(request.form["marginal_adhesion"]),
                float(request.form["epithelial_cell_size"]),
                float(request.form["bare_nuclei"]),
                float(request.form["bland_chromatin"]),
                float(request.form["normal_nucleoli"]),
                float(request.form["mitoses"])
            ]
            
            # Scale inputs
            input_scaled = scaler.transform([input_features])
            
            # Predict probability
            pred_prob = model.predict(input_scaled)[0][0]
            pred_class = 1 if pred_prob > 0.5 else 0
            
            # Prepare results
            result = "Malignant" if pred_class == 1 else "Benign"
            probability = f"{pred_prob*100:.2f}%"

        except Exception as e:
            result = f"Error: {e}"

    return render_template("index.html", result=result, probability=probability)


if __name__ == "__main__":
    app.run(debug=True)
