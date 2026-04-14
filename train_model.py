import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load and Clean
df = pd.read_csv('scholarship_results.csv')
# Keep only rows with valid data
df = df.dropna(subset=['score', 'income', 'Extracurricular', 'Status'])

# Features and Target (1 for Eligible, 0 for not Eligible)
X = df[['score', 'income', 'Extracurricular']]
y = df['Status'].apply(lambda x: 1 if 'not' not in str(x).lower() else 0)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model
with open('scholarship_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as scholarship_model.pkl")