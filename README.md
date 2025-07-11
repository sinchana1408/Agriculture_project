**Agri Recommendation System**
Empowering Smart Farming with ML— Predict Crops, Suggest Fertilizers and Pesticides, Estimate Market Prices
📌 Overview
The Agri Recommendation System is a smart farming web app built using machine learning. It takes soil nutrients and weather conditions as input and recommends:

- The best crop to grow 🌱
- Fertilizer suggestions for soil enrichment 💊
- Pesticide suggestions based on pest type 🐛
- Estimated crop market price 💰

This system helps improve agricultural decision-making for farmers, students, and researchers.
**🧠 Core Features**
- Crop Recommendation: Uses N, P, K, temperature, humidity, pH, and rainfall to recommend the most suitable crop.
- Fertilizer Recommendation: Based on soil and crop type, recommends the correct balance of nutrients.
- Pesticide Suggestion: Provides pest treatment suggestions using encoded crop-pest data.
- Crop Price Estimation: Predicts the price per quintal using historical crop and region data.
- Tips Section: Each result page includes expert tips for successful implementation.
  
**📁 Folder Structure**

![Folder Structure](https://github.com/sinchana1408/Agriculture_project/blob/39394e0f97dbf11fd74ebe10ec29716d313d4054/Screenshot%202025-07-11%20085710.png)

**🔧 How to Set Up and Run**

Step 1: Clone the Repository
    git clone https://github.com/your-username/agri-recommendation.git
    cd agri-recommendation/backend

Step 2: Install Required Packages
    pip install flask pandas numpy scikit-learn

Step 3: Train All Models
    python train_crop_model.py
    python train_fert_model.py
    python train_pest_model.py
    python train_price_model.py

Step 4: Run the Application
    python app.py

Open in your browser:
    http://127.0.0.1:5000/

**🏠 Home Page**
![Home Page](https://github.com/sinchana1408/Agriculture_project/blob/f1f14645414e64852a0d7d180e940aa926f02468/Screenshot%202025-07-07%20184022.png )

**🌱 Crop Recommendation Form**
![Crop Form](https://github.com/sinchana1408/Agriculture_project/blob/f1f14645414e64852a0d7d180e940aa926f02468/Screenshot%202025-07-07%20183203.png)

**✅ Crop Prediction Result**
![Crop Result](https://github.com/sinchana1408/Agriculture_project/blob/f1f14645414e64852a0d7d180e940aa926f02468/Screenshot%202025-07-07%20183225.png)

**💊 Fertilizer Suggestion Form**
![Fertilizer Form](https://github.com/sinchana1408/Agriculture_project/blob/f1f14645414e64852a0d7d180e940aa926f02468/Screenshot%202025-07-07%20183445.png)

**✅ Fertilizer Suggestion Result**
![Fertilizer Result](https://github.com/sinchana1408/Agriculture_project/blob/f1f14645414e64852a0d7d180e940aa926f02468/Screenshot%202025-07-07%20183500.png)

**🐛 Pesticide Suggestion Form**
![Pesticide Form](https://github.com/sinchana1408/Agriculture_project/blob/f1f14645414e64852a0d7d180e940aa926f02468/Screenshot%202025-07-07%20183810.png)

**✅ Pesticide Suggestion Result**
![Pesticide Result](https://github.com/sinchana1408/Agriculture_project/blob/f1f14645414e64852a0d7d180e940aa926f02468/Screenshot%202025-07-07%20183734.png)

**💰 Crop Price Estimation Form**
![Price Form](https://github.com/sinchana1408/Agriculture_project/blob/f1f14645414e64852a0d7d180e940aa926f02468/Screenshot%202025-07-07%20183913.png)

**✅ Price Estimation Result**
![Price Result](https://github.com/sinchana1408/Agriculture_project/blob/f1f14645414e64852a0d7d180e940aa926f02468/Screenshot%202025-07-07%20183845.png)                                                                                                                                                                                  
**✅ Smart Farming Tips**

- Ensure timely irrigation based on water requirements
- Monitor pests and use pesticides only as needed
- Use compost or organic fertilizers when possible
- Test soil every season to adjust treatment
- Rotate crops to improve yield and preserve soil

**🌐 Technologies Used**

- Python + Flask (Backend)
- HTML5 + CSS3 + Bootstrap (Frontend)
- scikit-learn (Machine Learning)
- Pandas, NumPy (Data Processing)
- CSV datasets + Pickled models


**📃 License**
MIT License — Free to use, modify, and distribute with proper attribution.
