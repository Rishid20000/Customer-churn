# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv('churn_data.csv')

# Example: selecting features and target
X = df[['tenure', 'monthly_charges', 'total_charges', 'contract_type', 'internet_service']]
y = df['churn']  # 1 for churn, 0 for not

# Convert categorical to numeric (simplified)
X = pd.get_dummies(X, drop_first=True)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# Save model
pickle.dump(model, open('model.pkl', 'wb'))
