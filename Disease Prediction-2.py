import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Load the dataset
file_path = "E:/MV Project/Disease_symptom_and_patient_profile_dataset_expanded.csv"  # Replace with the path to your dataset
df = pd.read_csv(file_path)

# 1. Preprocessing
# Handle missing values (if any)
df.interpolate(method='linear', inplace=True)

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Split the dataset into features (X) and target (y)
X = df.drop('Outcome Variable', axis=1)
y = df['Outcome Variable']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. Model Training
# Initialize models
models = {
    'RandomForest': RandomForestClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'LogisticRegression': LogisticRegression(max_iter=1000)
}

# Train models and evaluate using cross-validation
for name, model in models.items():
    pipeline = Pipeline([('model', model)])
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    print(f'{name} Cross-Validation Accuracy: {scores.mean():.4f}')

# 3. Model Optimization (example with RandomForest)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f'Best Model Parameters: {grid_search.best_params_}')


# 4. Model Evaluation
# Predict on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
print(f'Test Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))


import joblib
joblib.dump(best_model, 'best_model2.pkl')  # Save using a different extension to avoid confusion

