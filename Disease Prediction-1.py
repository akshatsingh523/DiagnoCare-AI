import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load dataset (assuming it's in CSV format)
df = pd.read_csv("symbipredict_2022.csv")

# Preprocess the data
# Assuming the last column is 'prognosis' and the rest are symptoms
X = df.iloc[:, :-1]  # Symptoms
y = df.iloc[:, -1]   # Prognosis

# Encode the target variable (prognosis)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Handle categorical features (if any)
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()

# Model Pipelines
pipelines = {
    'dt': Pipeline([('scaler', scaler), ('classifier', DecisionTreeClassifier())]),
    'rf': Pipeline([('scaler', scaler), ('classifier', RandomForestClassifier())]),
    'svc': Pipeline([('scaler', scaler), ('classifier', SVC())]),
    'lr': Pipeline([('scaler', scaler), ('classifier', LogisticRegression())])
}

# Train and evaluate each model
for model_name, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}\n")

# Example of saving the best model (let's assume RandomForest performed the best)
import joblib
best_model = pipelines['rf']
joblib.dump(best_model, 'best_model1.pkl')  # Save using a different extension to avoid confusion


# Load the model for future predictions
# loaded_model = joblib.load('best_model.pkl')
