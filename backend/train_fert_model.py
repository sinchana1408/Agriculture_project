import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv("datasets/fertilizer.csv")

le_crop = LabelEncoder()
le_soil = LabelEncoder()
le_def = LabelEncoder()
le_target = LabelEncoder()

X = pd.DataFrame({
    'crop_type': le_crop.fit_transform(df['crop_type']),
    'soil_type': le_soil.fit_transform(df['soil_type']),
    'deficiency_level': le_def.fit_transform(df['deficiency_level'])
})
y = le_target.fit_transform(df['fertilizer'])

model = RandomForestClassifier()
model.fit(X, y)

# ✅ Save model and all encoders correctly
pickle.dump(model, open("fert_model.pkl", "wb"))
pickle.dump([le_crop, le_soil, le_def, le_target], open("fert_encoders.pkl", "wb"))

print("✅ fert_model.pkl and fert_encoders.pkl saved.")
