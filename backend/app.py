from flask import Flask, render_template, request
import pickle

app = Flask(__name__, template_folder='../templates')

# Load models
crop_model = pickle.load(open('crop_model.pkl', 'rb'))
fert_model = pickle.load(open('fert_model.pkl', 'rb'))
fert_enc = pickle.load(open('fert_encoders.pkl', 'rb'))
pest_model = pickle.load(open('pest_model.pkl', 'rb'))
pest_enc = pickle.load(open('pest_encoders.pkl', 'rb'))
price_model = pickle.load(open('price_model.pkl', 'rb'))
price_enc = pickle.load(open('price_encoders.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/crop')
def crop():
    return render_template('crop.html')

@app.route('/fertilizer')
def fertilizer():
    return render_template('fertilizer.html')

@app.route('/pesticide')
def pesticide():
    return render_template('pesticide.html')

@app.route('/price')
def price():
    return render_template('price.html')

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    data = [float(request.form[i]) for i in ['N','P','K','temperature','humidity','ph','rainfall']]
    result = crop_model.predict([data])[0]
    return render_template('crop_result.html', crop=result)

@app.route('/predict_fertilizer', methods=['POST'])
def predict_fertilizer():
    crop = request.form['crop_type']
    soil = request.form['soil_type']
    deficiency = request.form['deficiency']
    
    le1, le2, le3, le_target = fert_enc
    
    # Validate inputs
    if crop not in le1.classes_:
        return f"Error: Unknown crop type '{crop}'. Allowed: {list(le1.classes_)}"
    if soil not in le2.classes_:
        return f"Error: Unknown soil type '{soil}'. Allowed: {list(le2.classes_)}"
    if deficiency not in le3.classes_:
        return f"Error: Unknown deficiency '{deficiency}'. Allowed: {list(le3.classes_)}"
    
    # Predict
    X = [[le1.transform([crop])[0], le2.transform([soil])[0], le3.transform([deficiency])[0]]]
    y = fert_model.predict(X)[0]
    fertilizer = le_target.inverse_transform([y])[0]
    return render_template('fertilizer_result.html', fertilizer=fertilizer)

@app.route('/predict_pesticide', methods=['POST'])
def predict_pesticide():
    crop = request.form['crop']
    pest = request.form['pest']
    intensity = request.form['intensity']
    
    le1, le2, le3, le_target = pest_enc
    
    # Validate inputs
    if crop not in le1.classes_:
        return f"Error: Unknown crop '{crop}'. Allowed: {list(le1.classes_)}"
    if pest not in le2.classes_:
        return f"Error: Unknown pest '{pest}'. Allowed: {list(le2.classes_)}"
    if intensity not in le3.classes_:
        return f"Error: Unknown intensity '{intensity}'. Allowed: {list(le3.classes_)}"
    
    # Predict
    X = [[le1.transform([crop])[0], le2.transform([pest])[0], le3.transform([intensity])[0]]]
    y = pest_model.predict(X)[0]
    pesticide = le_target.inverse_transform([y])[0]
    return render_template('pesticide_result.html', pesticide=pesticide)

@app.route('/predict_price', methods=['POST'])
def predict_price():
    crop = request.form['crop']
    area = float(request.form['area'])
    season = request.form['season']
    
    le_crop, le_season = price_enc
    
    # Validate inputs
    if crop not in le_crop.classes_:
        return f"Error: Unknown crop '{crop}'. Allowed: {list(le_crop.classes_)}"
    if season not in le_season.classes_:
        return f"Error: Unknown season '{season}'. Allowed: {list(le_season.classes_)}"
    
    X = [[le_crop.transform([crop])[0], area, le_season.transform([season])[0]]]
    price = price_model.predict(X)[0]
    return render_template('price_result.html', crop=crop, price=round(price, 2))

if __name__ == '__main__':
    app.run(debug=True)
