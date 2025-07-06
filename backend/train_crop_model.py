import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
df = pd.read_csv("datasets/crop.csv")

X = df.drop('label', axis=1)
y = df['label']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
with open('crop_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ crop_model.pkl saved.")
