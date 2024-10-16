import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('heart_disease_uci.csv')

# One-hot encoding for categorical features
dataset = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

# Standardizing continuous features
scaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = scaler.fit_transform(dataset[columns_to_scale])

# Define dependent and independent features
X = dataset.drop('target', axis=1)
y = dataset['target']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model with k=12
knn_classifier = KNeighborsClassifier(n_neighbors=12)
knn_classifier.fit(X_train, y_train)

# Evaluate using cross-validation
cv_scores = cross_val_score(knn_classifier, X_train, y_train, cv=10)
print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Mean cross-validation accuracy: {cv_scores.mean()}")

def predict_heart_disease(input_dict):
    # Create a zero array for 30 features
    input_features = np.zeros(X.shape[1])

    # Manually map the input values to the correct indices in the one-hot encoded features
    input_features[0] = input_dict['age']
    input_features[1] = input_dict['trestbps']
    input_features[2] = input_dict['chol']
    input_features[3] = input_dict['thalach']
    input_features[4] = input_dict['oldpeak']
    
    # Map categorical values to one-hot encoded positions
    if input_dict['sex'] == 1:
        input_features[5] = 1  # Male
    else:
        input_features[6] = 1  # Female
    
    input_features[7 + input_dict['cp']] = 1  # Chest pain type
    input_features[11 + input_dict['fbs']] = 1  # Fasting blood sugar
    input_features[13 + input_dict['restecg']] = 1  # Resting electrocardiographic results
    input_features[15 + input_dict['exang']] = 1  # Exercise induced angina
    input_features[17 + input_dict['slope']] = 1  # Slope of the peak exercise ST segment
    input_features[20 + input_dict['ca']] = 1  # Number of major vessels colored by fluoroscopy
    input_features[25 + input_dict['thal']] = 1  # Thalassemia

    input_scaled = scaler.transform([input_features[:5]])  # Only scale the first 5 features
    
    input_features[:5] = input_scaled[0]        # Merge scaled and non-scaled features

    prediction = knn_classifier.predict([input_features])
    
    return prediction[0]  # Return(0 or 1)

# Dummy input values
input_data = {
    'age': 63,
    'sex': 1,
    'cp': 3,
    'trestbps': 145,
    'chol': 233,
    'fbs': 1,
    'restecg': 0,
    'thalach': 150,
    'exang': 0,
    'oldpeak': 2.3,
    'slope': 0,
    'ca': 0,
    'thal': 1
}

prediction = predict_heart_disease(input_data)

print("Predicted class:", prediction)
