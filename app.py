from flask import Flask, render_template, request
import joblib
import pandas as pd
app = Flask(__name__)
MODEL_PATH = "model/breast_cancer_lr.joblib"
FEATURES_PATH = "model/breast_cancer_features.joblib"
model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probability = None
    if request.method == "POST":
        try:
            input_dict = {f: [float(request.form[f])] for f in feature_names}
            input_df = pd.DataFrame(input_dict)
            pred_prob = model.predict_proba(input_df)[0][1]  # probability of class 1 (Benign)
            pred_class = model.predict(input_df)[0]

            result = "Benign" if pred_class == 1 else "Malignant"
            probability = f"{pred_prob*100:.2f}%"
        except Exception as e:
            result = f"Error: {e}"
    return render_template("index.html", feature_names=feature_names, result=result, probability=probability)
if __name__ == "__main__":
    app.run(debug=True)