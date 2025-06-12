# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Load Dataset
df = pd.read_csv('diabetes.csv')  # Replace with the path if needed
print("First 5 rows:\n", df.head())

# Step 3: Data Exploration
print("\nDataset Info:\n")
print(df.info())

print("\nSummary Statistics:\n")
print(df.describe())

# Check missing values
print("\nMissing values:\n", df.isnull().sum())

# Step 4: Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Step 5: Data Preprocessing
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Step 7: Train a Model (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 8: Make Predictions
y_pred = model.predict(X_test)

# Step 9: Evaluate the Model
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 10 (Optional): Save Model for Later Use
import joblib
joblib.dump(model, 'diabetes_rf_model.pkl')
print("\nModel saved as 'diabetes_rf_model.pkl'")
