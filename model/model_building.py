import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
data = load_breast_cancer(as_frame=True)
df = data.frame
df['diagnosis'] = df['target'].map({0: 'Malignant', 1: 'Benign'})
df.drop(columns=['target'], inplace=True)
selected_features = [
    'mean radius',
    'mean texture',
    'mean perimeter',
    'mean area',
    'mean concavity'
]
X = df[selected_features]
y = df['diagnosis']
X = X.fillna(X.mean())
y = y.map({'Benign': 0, 'Malignant': 1})
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.1,
    verbose=1
)
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
roc_display = RocCurveDisplay(
    fpr=fpr,
    tpr=tpr,
    roc_auc=roc_auc,
    estimator_name="Neural Network Model"
)
roc_display.plot()
plt.title("ROC Curve")
plt.savefig("roc_curve.png")
plt.show()
print(f"AUC-ROC Score: {roc_auc:.4f}")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("\n--- Model Performance ---")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Benign", "Malignant"]
)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
pr_display = PrecisionRecallDisplay(
    precision=precision,
    recall=recall
)
pr_display.plot()
plt.title("Precision-Recall Curve")
plt.savefig("precision_recall_curve.png")
plt.show()
model.save("breast_cancer_nn_model.keras")
joblib.dump(scaler, "scaler.save")
loaded_model = load_model("breast_cancer_nn_model.keras")
loaded_scaler = joblib.load("scaler.save")
sample = X_test[0].reshape(1, -1)
prediction = loaded_model.predict(sample)
diagnosis = "Malignant" if prediction > 0.5 else "Benign"
print("\nReloaded Model Prediction:", diagnosis)
