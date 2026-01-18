import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Load CSV
df = pd.read_csv("data/data.csv")
print("First 5 rows:")
print(df.head())
print("\nShape:", df.shape)
print("\nColumns:", df.columns)

# Preprocessing
df = df.drop('Sample code number', axis=1)
df['Class'] = df['Class'].map({2: 0, 4: 1})  # 0 = benign, 1 = malignant
print("\nMissing values:\n", df.isnull().sum())

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build Neural Network
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))  # Input layer
model.add(Dense(16, activation='relu'))       # Hidden layer
model.add(Dense(1, activation='sigmoid'))    # Output layer

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.1,
    verbose=1
)

# valuate model
loss, accuracy = model.evaluate(X_test, y_test)
print("\nTest Loss:", loss)
print("Test Accuracy:", accuracy)

# Predictions
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_prob)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC-ROC: {auc_roc:.4f}")

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign","Malignant"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show() 
plt.savefig("conf_matrix.png")  # <-- this opens the plot window

# --- ROC Curve ---
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
roc_disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="Breast Cancer Model")
roc_disp.plot()
plt.title("ROC Curve")
plt.show()
plt.savefig("roc_curve.png")
# --- Precision-Recall Curve ---
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
pr_disp = PrecisionRecallDisplay(precision=precision, recall=recall)
pr_disp.plot()
plt.title("Precision-Recall Curve")
plt.show()
plt.savefig("precision_recall.png")


# Model Summary
model.summary()
import joblib

# Save trained model (Keras format)
model.save("breast_cancer_model.keras")
joblib.dump(scaler, "scaler.save")
print("Model and scaler saved successfully!")

