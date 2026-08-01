import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lon1_rad, lat1_rad, lon2_rad, lat2_rad = map(np.radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat/2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

def engineer_features(df):
    """
    Perform feature engineering on the credit card transactions dataframe.
    """
    df = df.copy()
    
    # Convert datetime columns
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])
    
    # Extract temporal features
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df['month'] = df['trans_date_trans_time'].dt.month
    
    # Calculate age at the time of transaction
    df['age'] = df['trans_date_trans_time'].dt.year - df['dob'].dt.year
    
    # Calculate geographical distance between customer and merchant
    df['distance_km'] = haversine_np(df['long'], df['lat'], df['merch_long'], df['merch_lat'])
    
    return df

def main():
    print("Starting Machine Learning Pipeline...")
    
    # 1. Load the dataset
    train_path = os.path.join("dataset", "fraudTrain.csv")
    test_path = os.path.join("dataset", "fraudTest.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Dataset files not found. Please run generate_data.py first.")
        
    print("Loading training data...")
    train_df = pd.read_csv(train_path)
    print("Loading test data...")
    test_df = pd.read_csv(test_path)
    
    # 2. Data Cleaning & Preprocessing
    print("Cleaning dataset (removing duplicates and handling missing values)...")
    train_df = train_df.drop_duplicates()
    test_df = test_df.drop_duplicates()
    
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    
    # 3. Feature Engineering
    print("Engineering features...")
    train_processed = engineer_features(train_df)
    test_processed = engineer_features(test_df)
    
    # 4. Feature Selection
    # Drop pure identifiers and irrelevant columns
    # We keep: amt, category, gender, state, city_pop, age, hour, day_of_week, month, distance_km, unix_time, merchant
    features = [
        "amt", "category", "gender", "state", "city_pop", 
        "age", "hour", "day_of_week", "month", "distance_km", 
        "unix_time", "merchant"
    ]
    target = "is_fraud"
    
    X_train_raw = train_processed[features].copy()
    y_train = train_processed[target]
    
    X_test_raw = test_processed[features].copy()
    y_test = test_processed[target]
    
    # 5. Label Encoding for categorical features
    print("Encoding categorical features...")
    categorical_cols = ["category", "gender", "state", "merchant"]
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        # Fit on both train and test to ensure all categories are covered
        combined_series = pd.concat([X_train_raw[col], X_test_raw[col]], axis=0).astype(str)
        le.fit(combined_series)
        
        X_train_raw[col] = le.transform(X_train_raw[col].astype(str))
        X_test_raw[col] = le.transform(X_test_raw[col].astype(str))
        
        label_encoders[col] = le
        
    # 6. Feature Scaling
    print("Scaling numerical features...")
    numerical_cols = ["amt", "city_pop", "age", "hour", "day_of_week", "month", "distance_km", "unix_time"]
    
    scaler = StandardScaler()
    # Fit only on training numerical data
    scaler.fit(X_train_raw[numerical_cols])
    
    # Transform both datasets
    X_train_scaled = X_train_raw.copy()
    X_test_scaled = X_test_raw.copy()
    
    X_train_scaled[numerical_cols] = scaler.transform(X_train_raw[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test_raw[numerical_cols])
    
    # Save the files for model ingestion (ensures original column order is preserved)
    # The models will expect columns in the exact order of features list
    X_train_final = X_train_scaled[features]
    X_test_final = X_test_scaled[features]
    
    # 7. Model Training and Comparison
    print("\nTraining models...")
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=10),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_final, y_train)
        trained_models[name] = model
        
        # Predictions
        y_pred = model.predict(X_test_final)
        
        # Probability for ROC-AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_final)[:, 1]
        else:
            y_prob = model.decision_function(X_test_final)
            
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        
        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC-AUC": auc,
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred, zero_division=0)
        }
        
        print(f"\n--- {name} Performance ---")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC-AUC:   {auc:.4f}")
        print("Confusion Matrix:")
        print(results[name]["confusion_matrix"])
        print("Classification Report:")
        print(results[name]["classification_report"])
        print("="*40)
        
    # 8. Auto-select best performing model based on F1 Score
    # F1 Score is standard for highly imbalanced tabular fraud detection
    best_model_name = max(results, key=lambda k: results[k]["F1 Score"])
    print(f"\nBest Performing Model (based on F1 Score): {best_model_name}")
    print(f"F1 Score: {results[best_model_name]['F1 Score']:.4f}")
    
    # 9. Save artifacts
    print("\nSaving artifacts...")
    os.makedirs("models", exist_ok=True)
    
    # Save the scaler and encoders
    joblib.dump(scaler, os.path.join("models", "scaler.pkl"))
    joblib.dump(label_encoders, os.path.join("models", "label_encoders.pkl"))
    print("Saved scaler.pkl and label_encoders.pkl successfully.")
    
    # Save Random Forest model specifically (as requested)
    rf_model = trained_models["Random Forest"]
    joblib.dump(rf_model, os.path.join("models", "random_forest_model.pkl"))
    print("Saved random_forest_model.pkl (Random Forest Classifier) successfully.")
    
    # If the best model was not Random Forest, we save the best model under its own name too
    if best_model_name != "Random Forest":
        best_model_clean_name = best_model_name.lower().replace(" ", "_")
        joblib.dump(trained_models[best_model_name], os.path.join("models", f"{best_model_clean_name}_model.pkl"))
        print(f"Saved {best_model_clean_name}_model.pkl as the auto-selected best model.")
        
    print("\nMachine Learning Pipeline execution finished successfully!")

if __name__ == "__main__":
    main()
