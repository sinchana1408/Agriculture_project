import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv("datasets/pesticide.csv")

le_crop = LabelEncoder()
le_pest = LabelEncoder()
le_intensity = LabelEncoder()
le_target = LabelEncoder()

X = pd.DataFrame({
    'crop': le_crop.fit_transform(df['crop']),
    'pest_type': le_pest.fit_transform(df['pest_type']),
    'intensity': le_intensity.fit_transform(df['intensity'])
})
y = le_target.fit_transform(df['pesticide'])

model = RandomForestClassifier()
model.fit(X, y)

# Save model and encoders
pickle.dump(model, open("pest_model.pkl", "wb"))
pickle.dump([le_crop, le_pest, le_intensity, le_target], open("pest_encoders.pkl", "wb"))

print("✅ pest_model.pkl and pest_encoders.pkl saved.")
