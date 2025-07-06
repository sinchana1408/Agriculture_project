import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv("datasets/crop_price.csv")

le_crop = LabelEncoder()
le_season = LabelEncoder()

X = pd.DataFrame({
    'crop': le_crop.fit_transform(df['crop']),
    'area': df['area'],
    'season': le_season.fit_transform(df['season'])
})
y = df['price']

model = RandomForestRegressor()
model.fit(X, y)

# Save model and encoders
pickle.dump(model, open("price_model.pkl", "wb"))
pickle.dump([le_crop, le_season], open("price_encoders.pkl", "wb"))

print("✅ price_model.pkl and price_encoders.pkl saved.")
