import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample data (replace with your actual data)
data = {'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'outcome': np.random.randint(0, 2, 100)}
df = pd.DataFrame(data)

# Prepare data for the model
X = df[['feature1', 'feature2']] # Features
y = df['outcome'] # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)

# Example of prediction for new data
new_data = pd.DataFrame({'feature1': [0.6, 0.2], 'feature2': [0.3, 0.8]})
predicted_probabilities = model.predict_proba(new_data)
predicted_classes = model.predict(new_data)

print("Predicted Probabilities:\n", predicted_probabilities)
print("Predicted Classes:", predicted_classes)