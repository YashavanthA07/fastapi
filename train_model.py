import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv('dataset.csv')
X = df[['Age', 'Salary']]
y = df['Approved']

model = RandomForestClassifier(random_state=42)

model.fit(X,y)

joblib.dump(model,"loan_model.joblib")
print("Model saved successfully")
