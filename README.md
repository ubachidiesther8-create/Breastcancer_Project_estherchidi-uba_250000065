# Breast Cancer Prediction System

## Project Overview
This project implements a **Breast Cancer Prediction System** using machine learning to classify breast tumors as **Benign** or **Malignant**.  
The model is trained on the **Breast Cancer Wisconsin (Diagnostic) Dataset** and is strictly developed for **educational purposes**, not as a medical diagnostic tool.

---

## Dataset
The dataset is sourced from **scikit-learn**:
- Breast Cancer Wisconsin (Diagnostic) Dataset

Each sample contains numeric features computed from digitized images of fine needle aspirate (FNA) of breast masses.

### Selected Features
The model uses **five (5)** predictive features:
- `radius_mean`
- `texture_mean`
- `perimeter_mean`
- `area_mean`
- `concavity_mean`

### Target Variable
- `diagnosis`
  - `0` → Benign  
  - `1` → Malignant  

---

## Machine Learning Algorithm
- **Logistic Regression**

Logistic Regression was selected due to:
- High interpretability
- Strong performance on linearly separable medical datasets
- Lower computational cost compared to neural networks

---

## Data Preprocessing
The following preprocessing steps were applied:
- Handling missing values
- Feature selection
- Feature scaling using **StandardScaler**
- Train-test split with stratification to preserve class balance

---

## Model Evaluation
The model is evaluated using standard classification metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve

These metrics ensure balanced performance evaluation, especially in the presence of class imbalance.

---

## Model Persistence
The trained model is saved using **Joblib**, allowing reuse without retraining.

Saved files:
- `model/model.joblib` – trained logistic regression model
- `model/scaler.pkl` – fitted feature scaler

---

## Web Application
A Flask-based web application allows users to:
- Input tumor feature values
- Receive real-time prediction (Benign or Malignant)
- View prediction probability

### Technologies Used
- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- HTML/CSS

---

## Project Structure

