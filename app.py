# app.py - Main Flask Application

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

# Configuration
MODEL_PATH = 'models'
DATA_PATH = 'data'
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)

# Sample data generation (would be replaced by actual data in production)
def generate_sample_data():
    """Generate synthetic weather data for demonstration"""
    # Create date range for the past year
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate synthetic weather data
    np.random.seed(42)  # For reproducibility
    temp_min = np.sin(np.linspace(0, 2*np.pi, len(date_range))) * 10 + 5 + np.random.normal(0, 3, len(date_range))
    temp_max = temp_min + 5 + np.random.normal(0, 2, len(date_range))
    humidity = np.cos(np.linspace(0, 2*np.pi, len(date_range))) * 20 + 60 + np.random.normal(0, 5, len(date_range))
    pressure = np.random.normal(1013, 5, len(date_range))
    wind_speed = np.abs(np.random.normal(10, 3, len(date_range)))
    precipitation = np.abs(np.random.exponential(0.5, len(date_range)))
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': date_range,
        'temp_min': temp_min,
        'temp_max': temp_max,
        'humidity': humidity,
        'pressure': pressure,
        'wind_speed': wind_speed,
        'precipitation': precipitation
    })
    
    # Add some seasonal features
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    
    return df

# Data preprocessing
def preprocess_data(df):
    """Prepare weather data for modeling"""
    # Create features from date
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year']/365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year']/365)
    
    # Create lag features (previous days' weather)
    for i in range(1, 4):
        df[f'temp_min_lag_{i}'] = df['temp_min'].shift(i)
        df[f'temp_max_lag_{i}'] = df['temp_max'].shift(i)
        df[f'humidity_lag_{i}'] = df['humidity'].shift(i)
        df[f'precipitation_lag_{i}'] = df['precipitation'].shift(i)
    
    # Drop rows with NaN values (created by lag features)
    df = df.dropna()
    
    return df

# Model training
def train_model(df, target_col, feature_cols):
    """Train a Random Forest model to predict weather metrics"""
    X = df[feature_cols]
    y = df[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Save model and scaler
    os.makedirs(f'{MODEL_PATH}/{target_col}', exist_ok=True)
    joblib.dump(model, f'{MODEL_PATH}/{target_col}/model.pkl')
    joblib.dump(scaler, f'{MODEL_PATH}/{target_col}/scaler.pkl')
    joblib.dump(feature_cols, f'{MODEL_PATH}/{target_col}/features.pkl')
    
    return {
        'model': model,
        'scaler': scaler,
        'features': feature_cols,
        'metrics': {
            'mse': mse,
            'r2': r2
        }
    }

def predict_weather(df, target_col, days_to_predict=7):
    """Generate weather predictions for upcoming days"""
    # Load model and scaler
    model = joblib.load(f'{MODEL_PATH}/{target_col}/model.pkl')
    scaler = joblib.load(f'{MODEL_PATH}/{target_col}/scaler.pkl')
    feature_cols = joblib.load(f'{MODEL_PATH}/{target_col}/features.pkl')
    
    # Get the most recent data for prediction
    last_date = df['date'].max()
    
    # Create prediction dataframe
    pred_dates = [last_date + timedelta(days=i+1) for i in range(days_to_predict)]
    pred_df = pd.DataFrame({'date': pred_dates})
    pred_df['day_of_year'] = pred_df['date'].dt.dayofyear
    pred_df['month'] = pred_df['date'].dt.month
    pred_df['day'] = pred_df['date'].dt.day
    
    # Add seasonal features
    pred_df['day_of_year_sin'] = np.sin(2 * np.pi * pred_df['day_of_year']/365)
    pred_df['day_of_year_cos'] = np.cos(2 * np.pi * pred_df['day_of_year']/365)
    
    # Initialize all possible lag feature columns that might be needed
    possible_lag_cols = ['temp_min', 'temp_max', 'humidity', 'precipitation']
    
    # First, make sure all target columns exist in the dataframe
    for col in possible_lag_cols:
        if col not in pred_df.columns:
            pred_df[col] = np.nan
    
    # Make predictions iteratively for each day
    for i in range(days_to_predict):
        # Fill in lag features based on previous predictions or known values
        if i == 0:
            # For first prediction, use the last known values from original data
            for j in range(1, 4):
                idx = min(j, len(df))
                for col in possible_lag_cols:
                    if col in df.columns:  # Make sure the column exists in original data
                        pred_df.loc[i, f'{col}_lag_{j}'] = df[col].iloc[-idx]
        else:
            # For subsequent predictions, use earlier predictions as lag features
            for j in range(1, 4):
                if i-j >= 0:
                    # Use already predicted values
                    for col in possible_lag_cols:
                        if col in pred_df.columns:
                            pred_df.loc[i, f'{col}_lag_{j}'] = pred_df.loc[i-j, col]
                else:
                    # Use known values from original data
                    offset = j - i
                    for col in possible_lag_cols:
                        if col in df.columns:
                            pred_df.loc[i, f'{col}_lag_{j}'] = df[col].iloc[-offset]
        
        # Get features for current prediction
        X_pred = pred_df.loc[i:i, feature_cols]
        
        # Scale features
        X_pred_scaled = scaler.transform(X_pred)
        
        # Make prediction
        y_pred = model.predict(X_pred_scaled)[0]
        
        # Store prediction
        pred_df.loc[i, target_col] = y_pred
    
    return pred_df

# Generate visualizations
def create_visualization(df, pred_df, target_col):
    """Create visualization of historical data and predictions"""
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    plt.plot(df['date'][-30:], df[target_col][-30:], 'b-', label='Historical')
    
    # Plot predictions
    plt.plot(pred_df['date'], pred_df[target_col], 'r--', label='Predicted')
    
    # Add labels and title
    plt.xlabel('Date')
    plt.ylabel(target_col.replace('_', ' ').title())
    plt.title(f'{target_col.replace("_", " ").title()} - Historical and Predicted')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot to a base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    graph = base64.b64encode(image_png).decode('utf-8')
    return f'data:image/png;base64,{graph}'

# Generate feature importance visualization
def feature_importance_plot(target_col):
    """Create visualization of feature importance"""
    model = joblib.load(f'{MODEL_PATH}/{target_col}/model.pkl')
    features = joblib.load(f'{MODEL_PATH}/{target_col}/features.pkl')
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Sort feature importances
    indices = np.argsort(importances)[-10:]  # Top 10 features
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.title(f'Feature Importance for {target_col.replace("_", " ").title()}')
    plt.tight_layout()
    
    # Save plot to a base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    graph = base64.b64encode(image_png).decode('utf-8')
    return f'data:image/png;base64,{graph}'

# Initialize models (run once at startup)
@app.before_request
def initialize():
    """Initialize models on application startup"""
    # Generate and save sample data
    df = generate_sample_data()
    df.to_csv(f'{DATA_PATH}/weather_data.csv', index=False)
    
    # Preprocess data
    processed_df = preprocess_data(df)
    
    # Define target columns and features
    targets = ['temp_min', 'temp_max', 'humidity', 'precipitation']
    feature_cols = [
        'day_of_year_sin', 'day_of_year_cos', 'month', 'day',
        'temp_min_lag_1', 'temp_min_lag_2', 'temp_min_lag_3',
        'temp_max_lag_1', 'temp_max_lag_2', 'temp_max_lag_3',
        'humidity_lag_1', 'humidity_lag_2', 'humidity_lag_3',
        'precipitation_lag_1', 'precipitation_lag_2', 'precipitation_lag_3'
    ]
    
    # Train models for each target
    for target in targets:
        train_model(processed_df, target, feature_cols)

# Routes
@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Generate and return weather predictions"""
    days = int(request.form.get('days', 7))
    
    # Load data
    df = pd.read_csv(f'{DATA_PATH}/weather_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Preprocess data
    processed_df = preprocess_data(df)
    
    # Make predictions for each target
    targets = ['temp_min', 'temp_max', 'humidity', 'precipitation']
    predictions = {}
    visualizations = {}
    feature_importances = {}
    
    for target in targets:
        # Generate predictions
        pred_df = predict_weather(processed_df, target, days)
        predictions[target] = pred_df[['date', target]].to_dict('records')
        
        # Create visualizations
        visualizations[target] = create_visualization(df, pred_df, target)
        feature_importances[target] = feature_importance_plot(target)
    
    return jsonify({
        'predictions': predictions,
        'visualizations': visualizations,
        'feature_importances': feature_importances
    })

if __name__ == '__main__':
    app.run(debug=True)
