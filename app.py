import os
import time
import datetime
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Constants
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODERS_PATH = os.path.join(MODEL_DIR, "label_encoders.pkl")

# Lat/Long center coordinates for states to calculate distance_km
STATE_COORDS = {
    "NY": (40.7128, -74.0060),
    "CA": (34.0522, -118.2437),
    "TX": (29.7604, -95.3698),
    "FL": (25.7617, -80.1918),
    "IL": (41.8781, -87.6298),
    "PA": (39.9526, -75.1652),
    "OH": (39.9612, -82.9988),
    "MI": (42.3314, -83.0458),
    "GA": (33.7490, -84.3880),
    "NC": (35.2271, -80.8431),
    "NJ": (40.7357, -74.1724),
    "VA": (37.5407, -77.4360),
    "WA": (47.6062, -122.3321),
    "AZ": (33.4484, -112.0740),
    "CO": (39.7392, -104.9903)
}

# Load ML artifacts globally
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(ENCODERS_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        label_encoders = joblib.load(ENCODERS_PATH)
        print("ML models loaded successfully.")
    else:
        model = None
        scaler = None
        label_encoders = None
        print("Warning: ML models not found. Please run the model training script first.")
except Exception as e:
    model = None
    scaler = None
    label_encoders = None
    print(f"Error loading models: {e}")

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

def safe_encode(encoder, value):
    """
    Encodes categorical features safely. If the category is not found, 
    falls back to the first available category.
    """
    try:
        if value in encoder.classes_:
            return encoder.transform([value])[0]
        else:
            # Fallback to default class in label encoder
            return encoder.transform([encoder.classes_[0]])[0]
    except Exception:
        return 0

@app.route('/')
def home():
    # If encoders are loaded, get categories, states, and merchants to populate dropdowns
    categories = []
    states = []
    merchants = []
    
    if label_encoders:
        categories = sorted(list(label_encoders['category'].classes_))
        states = sorted(list(label_encoders['state'].classes_))
        merchants = sorted([m.replace("fraud_", "") for m in label_encoders['merchant'].classes_])
        
    return render_template(
        'index.html',
        categories=categories,
        states=states,
        merchants=merchants
    )

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None or label_encoders is None:
        return jsonify({
            'status': 'error',
            'message': 'Machine learning models are not loaded on the backend. Please check system logs.'
        }), 500
        
    try:
        # 1. Parse and validate input data
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'No input data provided.'}), 400
            
        try:
            amt = float(data.get('amt'))
            city_pop = int(data.get('city_pop'))
            age = int(data.get('age'))
            gender = str(data.get('gender')).upper()
            state = str(data.get('state')).upper()
            category = str(data.get('category')).lower()
            merchant = data.get('merchant', '')
        except (ValueError, TypeError) as e:
            return jsonify({'status': 'error', 'message': f'Invalid value format: {str(e)}'}), 400
            
        # 2. Input validation bounds
        if amt <= 0:
            return jsonify({'status': 'error', 'message': 'Transaction amount must be greater than 0.'}), 400
        if city_pop < 0:
            return jsonify({'status': 'error', 'message': 'City population cannot be negative.'}), 400
        if age < 18 or age > 120:
            return jsonify({'status': 'error', 'message': 'Age must be between 18 and 120.'}), 400
        if gender not in ['M', 'F']:
            return jsonify({'status': 'error', 'message': 'Gender must be M or F.'}), 400
            
        # 3. Resolve Merchant
        # If merchant is empty, assign a default sample merchant
        if not merchant:
            # We select one typical merchant class
            merchant = "Kozey-McGlynn"
            
        # Ensure merchant name conforms to training format
        if not merchant.startswith("fraud_"):
            merchant_full = f"fraud_{merchant}"
        else:
            merchant_full = merchant
            
        # 4. Generate derived attributes automatically
        current_time = datetime.datetime.now()
        hour = current_time.hour
        day_of_week = current_time.weekday()
        month = current_time.month
        
        # Align year to model training range (2019-2020) to avoid out-of-bounds scaling
        aligned_year = 2019 if current_time.year % 2 == 0 else 2020
        aligned_time = current_time.replace(year=aligned_year)
        unix_time = int(aligned_time.timestamp())
        
        # 5. Geolocation Simulation (Calculate distance_km)
        # Fetch base coordinate for the state
        base_lat, base_long = STATE_COORDS.get(state, (37.0902, -95.7129)) # US Centroid fallback
        
        # Determine distance offset based on transaction amount and time to simulate fraud anomalies
        # If amount is very large (> $250) or transaction occurs in late night hours, simulate a large distance
        is_suspicious_amount = amt > 250
        is_suspicious_time = hour in [23, 0, 1, 2, 3, 4]
        
        if is_suspicious_amount or is_suspicious_time:
            # Shift coordinate by 0.5 to 3.0 degrees (~50 to 300 km)
            lat_offset = np.random.uniform(0.5, 3.0)
            long_offset = np.random.uniform(0.5, 3.0)
        else:
            # Standard local transaction offset (~1 to 20 km)
            lat_offset = np.random.uniform(0.01, 0.15)
            long_offset = np.random.uniform(0.01, 0.15)
            
        merch_lat = base_lat + lat_offset
        merch_long = base_long + long_offset
        
        distance_km = haversine_distance(base_long, base_lat, merch_long, merch_lat)
        
        # 6. Encode Categorical variables using label encoders
        category_encoded = safe_encode(label_encoders['category'], category)
        gender_encoded = safe_encode(label_encoders['gender'], gender)
        state_encoded = safe_encode(label_encoders['state'], state)
        merchant_encoded = safe_encode(label_encoders['merchant'], merchant_full)
        
        # 7. Construct complete feature vector matching model training order
        features_list = [
            "amt", "category", "gender", "state", "city_pop", 
            "age", "hour", "day_of_week", "month", "distance_km", 
            "unix_time", "merchant"
        ]
        
        input_dict = {
            "amt": amt,
            "category": category_encoded,
            "gender": gender_encoded,
            "state": state_encoded,
            "city_pop": city_pop,
            "age": age,
            "hour": hour,
            "day_of_week": day_of_week,
            "month": month,
            "distance_km": distance_km,
            "unix_time": unix_time,
            "merchant": merchant_encoded
        }
        
        input_df = pd.DataFrame([input_dict])
        
        # 8. Scale numerical features
        numerical_cols = ["amt", "city_pop", "age", "hour", "day_of_week", "month", "distance_km", "unix_time"]
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
        
        # Rearrange columns to match model training exactly
        input_final = input_df[features_list]
        
        # 9. Perform prediction
        pred = int(model.predict(input_final)[0])
        prob = float(model.predict_proba(input_final)[0][1])
        
        # Formatting result
        confidence = prob if pred == 1 else (1.0 - prob)
        result_label = "Fraudulent Transaction" if pred == 1 else "Legitimate Transaction"
        
        # Recommendations
        if pred == 1:
            recommendation = "This transaction is highly suspicious. Manual verification is recommended."
            verdict = "fraud"
        else:
            recommendation = "This transaction appears safe."
            verdict = "legitimate"
            
        # Return complete details
        return jsonify({
            'status': 'success',
            'prediction': verdict,
            'label': result_label,
            'probability': prob,
            'confidence': confidence,
            'recommendation': recommendation,
            'details': {
                'distance_km': round(distance_km, 2),
                'hour': hour,
                'day_of_week': day_of_week,
                'month': month,
                'unix_time': unix_time,
                'merchant_used': merchant.replace("fraud_", "")
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Server error processing prediction: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
