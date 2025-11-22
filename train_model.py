import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import joblib
import json

print("=" * 60)
print("SUPPLY CHAIN ORACLE - MODEL TRAINING")
print("=" * 60)

# Step 1: Load your CSV data
print("\n[1/6] Loading CSV data...")
csv_path = r"C:\Users\ADMIN\Downloads\shipment_delay_100_minimal_clean_indian_cities.csv"
df = pd.read_csv(csv_path)

# Normalize column names
df.columns = df.columns.str.lower().str.strip()
print(f"✓ Loaded {len(df)} shipments")
print(f"✓ Columns: {list(df.columns)}")

# Step 2: Data Preprocessing
print("\n[2/6] Preprocessing data...")

# Handle missing values
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if col not in ['shipment_id']:
        df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')

# Create target variable
if 'delayed' in df.columns:
    df['target'] = (df['delayed'].str.lower() == 'yes').astype(int)
elif 'actual_days' in df.columns and 'scheduled_days' in df.columns:
    df['target'] = (df['actual_days'] > df['scheduled_days']).astype(int)
else:
    raise ValueError("No target variable found! Need 'delayed' or 'actual_days' column")

# Feature engineering
df['speed_ratio'] = df['distance_km'] / (df['scheduled_days'] + 1)
df['delay_days'] = df.get('actual_days', df['scheduled_days']) - df['scheduled_days']

print(f"✓ Target distribution: {df['target'].value_counts().to_dict()}")

# Step 3: Encode categorical variables
print("\n[3/6] Encoding categorical features...")

le_carrier = LabelEncoder()
le_weather = LabelEncoder()
le_traffic = LabelEncoder()

df['carrier_encoded'] = le_carrier.fit_transform(df['carrier'])
df['weather_encoded'] = le_weather.fit_transform(df['weather'])
df['traffic_encoded'] = le_traffic.fit_transform(df['traffic'])

print(f"✓ Carriers: {list(le_carrier.classes_)}")
print(f"✓ Weather conditions: {list(le_weather.classes_)}")
print(f"✓ Traffic levels: {list(le_traffic.classes_)}")

# Step 4: Prepare training data
print("\n[4/6] Preparing training data...")

feature_cols = ['distance_km', 'scheduled_days', 'carrier_encoded', 
                'weather_encoded', 'traffic_encoded', 'speed_ratio']

X = df[feature_cols]
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Training samples: {len(X_train)}")
print(f"✓ Testing samples: {len(X_test)}")

# Step 5: Train Models
print("\n[5/6] Training machine learning models...")

# XGBoost Model
print("\n  Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)
xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
xgb_accuracy = accuracy_score(y_test, xgb_pred)
print(f"  ✓ XGBoost Accuracy: {xgb_accuracy*100:.2f}%")

# LightGBM Model
print("\n  Training LightGBM...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)
lgb_model.fit(X_train, y_train)

lgb_pred = lgb_model.predict(X_test)
lgb_pred_proba = lgb_model.predict_proba(X_test)[:, 1]
lgb_accuracy = accuracy_score(y_test, lgb_pred)
print(f"  ✓ LightGBM Accuracy: {lgb_accuracy*100:.2f}%")

# Ensemble Model
print("\n  Creating Ensemble Model...")
ensemble_pred_proba = (xgb_pred_proba + lgb_pred_proba) / 2
ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)
ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
print(f"  ✓ Ensemble Accuracy: {ensemble_accuracy*100:.2f}%")

# Step 6: Save Models
print("\n[6/6] Saving models and encoders...")

# Create models directory
import os
os.makedirs('models', exist_ok=True)

# Save models
joblib.dump(xgb_model, 'models/xgb_model.pkl')
joblib.dump(lgb_model, 'models/lgb_model.pkl')
joblib.dump(le_carrier, 'models/le_carrier.pkl')
joblib.dump(le_weather, 'models/le_weather.pkl')
joblib.dump(le_traffic, 'models/le_traffic.pkl')

# Save metadata
metadata = {
    'feature_cols': feature_cols,
    'xgb_accuracy': float(xgb_accuracy),
    'lgb_accuracy': float(lgb_accuracy),
    'ensemble_accuracy': float(ensemble_accuracy),
    'carriers': list(le_carrier.classes_),
    'weather_conditions': list(le_weather.classes_),
    'traffic_levels': list(le_traffic.classes_),
    'total_samples': len(df),
    'training_samples': len(X_train),
    'testing_samples': len(X_test)
}

with open('models/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

print("✓ Models saved to 'models/' directory")

# Display detailed metrics
print("\n" + "=" * 60)
print("MODEL EVALUATION REPORT")
print("=" * 60)

print("\n📊 XGBoost Model:")
print(classification_report(y_test, xgb_pred, target_names=['On-Time', 'Delayed']))

print("\n📊 LightGBM Model:")
print(classification_report(y_test, lgb_pred, target_names=['On-Time', 'Delayed']))

print("\n📊 Ensemble Model:")
print(classification_report(y_test, ensemble_pred, target_names=['On-Time', 'Delayed']))

print("\n" + "=" * 60)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 60)
print("\nNext step: Run 'streamlit run app.py' to launch the web application")
