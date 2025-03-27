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
import joblib
import numpy as np
import pandas as pd
from datetime import timedelta
from tensorflow.python.keras.models import load_model
import tensorflow as tf
app = Flask(__name__)

# Configuration
MODEL_PATH = 'models'
DATA_PATH = 'data'
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)

# Initialize models (run once at startup)
@app.before_request
def initialize():
    """Initialize models on application startup"""
    # Load data from Guwahati_weather.csv
    try:
        df = pd.read_csv(f'{DATA_PATH}/Guwahati_weather.csv')
        df['date'] = pd.to_datetime(df['date'])  # Ensure the 'date' column is in datetime format
    except FileNotFoundError:
        raise FileNotFoundError(f"Guwahati_weather.csv not found in {DATA_PATH}. Please provide the file.")
    

# Data preprocessing
def preprocess_data(df, lag=7):
    """Prepare weather data for modeling"""
    # Ensure the 'date' column is in datetime format
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_year'] = df['date'].dt.dayofyear
    # Create seasonal features
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df = df.drop(columns=['day_of_year'])
    # Create lag features (previous days' weather)
    for i in range(1, lag):
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



def predict_weather_lstm(df, days_to_predict=7):
    """Generate multi-target weather predictions for upcoming days using LSTM, appending directly to pred_df"""

    # Load model and scalers from MODEL_PATH
    model = tf.keras.models.load_model('models/weather_lstm_model.keras')    
    scaler_X = joblib.load(f'models/scaler_X.pkl')
    scaler_y = joblib.load(f'models/scaler_y.pkl')

    target_cols = ['temp_min', 'temp_max', 'humidity', 'precipitation']
    feature_cols = [col for col in df.columns if col not in target_cols]
    print(f"Columns in lstm function: {feature_cols}")
    # Get the most recent date
    last_date = df['date'].max()

    # Create prediction dates and base pred_df
    pred_dates = [last_date + timedelta(days=i+1) for i in range(days_to_predict)]
    pred_df = pd.DataFrame({'date': pred_dates})
    pred_df['day_of_year'] = pred_df['date'].dt.dayofyear
    pred_df['day_of_year_sin'] = np.sin(2 * np.pi * pred_df['day_of_year'] / 365)
    pred_df['day_of_year_cos'] = np.cos(2 * np.pi * pred_df['day_of_year'] / 365)

    # Initialize target columns and lag features
    for col in target_cols:
        pred_df[col] = np.nan
        for j in range(1, 7):  # lag_1 to lag_6
            pred_df[f'{col}_lag_{j}'] = np.nan

    # Predict iteratively
    for i in range(days_to_predict):
        # Fill lag features
        for col in target_cols:
            for j in range(1, 7):
                if i - j >= 0:
                    pred_df.loc[i, f'{col}_lag_{j}'] = pred_df.loc[i - j, col]
                else:
                    pred_df.loc[i, f'{col}_lag_{j}'] = df[col].iloc[-j + i]

        # Fill static features from last row of df
        static_cols = ['pressure', 'wind_speed', 'day_of_year_sin', 'day_of_year_cos', 
                        'temp_min_lag_1', 'temp_max_lag_1', 'humidity_lag_1', 'precipitation_lag_1', 
                        'temp_min_lag_2', 'temp_max_lag_2', 'humidity_lag_2', 'precipitation_lag_2', 
                        'temp_min_lag_3', 'temp_max_lag_3', 'humidity_lag_3', 'precipitation_lag_3', 
                        'temp_min_lag_4', 'temp_max_lag_4', 'humidity_lag_4', 'precipitation_lag_4', 
                        'temp_min_lag_5', 'temp_max_lag_5', 'humidity_lag_5', 'precipitation_lag_5', 
                        'temp_min_lag_6', 'temp_max_lag_6', 'humidity_lag_6', 'precipitation_lag_6']
        for col in static_cols:
            pred_df.loc[i, col] = df[col].iloc[-1]
        X_pred = pred_df.drop(columns=['day_of_year', 'temp_min', 'temp_max', 'humidity', 'precipitation', 'date'])
        # Exclude 'date' and other non-numeric columns
        print(X_pred.columns)
        # Prepare input
        X_scaled = scaler_X.transform(X_pred)
        X_scaled = X_scaled.reshape((1, 7, 28))  # LSTM input shape

        # Predict and inverse scale
        y_pred_scaled = model.predict(X_scaled)[0]
        y_pred = scaler_y.inverse_transform([y_pred_scaled])[0]

        # Store in pred_df
        for idx, col in enumerate(target_cols):
            pred_df.loc[i, col] = y_pred[idx]

    return pred_df


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
'''def feature_importance_plot(target_col):
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
    return f'data:image/png;base64,{graph}'''
# Initialize models (run once at startup)

import time
import requests
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import joblib
import base64
import io
from eli5.sklearn import PermutationImportance
import eli5
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, RegressorMixin

from sklearn.base import BaseEstimator, RegressorMixin

class KerasWrapper(BaseEstimator, RegressorMixin):
    """
    A wrapper for Keras LSTM models for permutation importance.
    Expects the input X to be flattened with shape (n_samples, sequence_length * features_per_timestep).
    """
    def __init__(self, model, scaler_X, scaler_y, sequence_length=7, features_per_timestep=28):
        self.model = model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.sequence_length = sequence_length
        self.features_per_timestep = features_per_timestep

    def fit(self, X, y):
        # The model is already trained.
        return self

    def predict(self, X):
        """
        Reshape X into (n_samples, sequence_length, features_per_timestep)
        and return the inverse-scaled predictions.
        """
        # Scale the input features using the pre-fitted scaler.
        X_scaled = self.scaler_X.transform(X)
        # Ensure the input is reshaped to match the LSTM's expectations.
        X_scaled = X_scaled.reshape((X_scaled.shape[0], self.sequence_length, self.features_per_timestep))
        
        # Predict using the LSTM model.
        preds_scaled = self.model.predict(X_scaled)
        # Inverse-transform the predictions.
        preds = self.scaler_y.inverse_transform(preds_scaled)
        # Assuming one target value per sequence.
        return preds[:, 0]



def feature_importance_plot(target_col, X_raw, y_raw):
    """Create a feature importance plot for a given target using permutation importance on LSTM"""

    # Load model and scalers
    model = tf.keras.models.load_model('models/weather_lstm_model.keras')    
    scaler_X = joblib.load(f'models/scaler_X.pkl')
    scaler_y = joblib.load(f'models/scaler_y.pkl')

    # Filter target index
    target_idx_map = {'temp_min': 0, 'temp_max': 1, 'humidity': 2, 'precipitation': 3}
    target_idx = target_idx_map[target_col]

    # Select only numeric columns
    X_numeric = X_raw.select_dtypes(include=[np.number])

    # Replace scaler_X.feature_names_in_ with the expected feature names
    expected_features = ['day_of_year_sin', 'day_of_year_cos', 'temp_min_lag_1',
                        'temp_min_lag_2', 'temp_min_lag_3', 'temp_min_lag_4',
                        'temp_min_lag_5', 'temp_min_lag_6', 'temp_max_lag_1',
                        'temp_max_lag_2', 'temp_max_lag_3', 'temp_max_lag_4',
                        'temp_max_lag_5', 'temp_max_lag_6', 'humidity_lag_1',
                        'humidity_lag_2', 'humidity_lag_3', 'humidity_lag_4',
                        'humidity_lag_5', 'humidity_lag_6', 'precipitation_lag_1',
                        'precipitation_lag_2', 'precipitation_lag_3', 'precipitation_lag_4',
                        'precipitation_lag_5', 'precipitation_lag_6', 'pressure', 'wind_speed']

    # Ensure X_numeric has the correct columns
    missing_features = set(expected_features) - set(X_numeric.columns)
    for feature in missing_features:
        X_numeric[feature] = 0  # Add missing features with default value 0

    # Reorder columns to match the expected order
    X_numeric = X_numeric[expected_features]

    # Scale the input data
    X_scaled = scaler_X.transform(X_numeric)

    # Wrap the model
    wrapped_model = KerasWrapper(model, scaler_X, scaler_y)

    # Compute permutation importance
    result = permutation_importance(
        wrapped_model, X_scaled, y_raw[:, target_idx],
        n_repeats=10, random_state=42, scoring='neg_mean_squared_error'
    )

    importances = result.importances_mean
    feature_cols = expected_features
    indices = np.argsort(importances)[-10:]  # Top 10 features

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
    plt.xlabel('Importance (Permutation MSE decrease)')
    plt.title(f'Feature Importance for {target_col.replace("_", " ").title()}')
    plt.tight_layout()

    # Convert plot to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()

    graph = base64.b64encode(image_png).decode('utf-8')
    return f'data:image/png;base64,{graph}'

def get_historical_weather(latitude, longitude, start_date, end_date):
    """
    Fetches historical daily weather data for a given location and date range.
    Implements retry logic if the API rate limit is exceeded.
    Returns: DataFrame with columns: date, temp_min, temp_max, humidity, pressure, wind_speed, precipitation, day_of_year, month, day
    """
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_min,temperature_2m_max,precipitation_sum,relative_humidity_2m_mean,surface_pressure_mean,windspeed_10m_max",
        "timezone": "auto"
    }

    while True:
        try:
            response = requests.get(base_url, params=params)
            print(f"Requesting URL: {response.url}")  # Debugging: Print the full URL
            response.raise_for_status()  # Raise an error for bad responses (4xx and 5xx)

            data = response.json()
            if "daily" not in data or "time" not in data["daily"]:
                raise ValueError("Invalid API response: Missing 'daily.time' field")

            df = pd.DataFrame({
                "date": pd.to_datetime(data["daily"]["time"]),
                "temp_min": data["daily"].get("temperature_2m_min", []),
                "temp_max": data["daily"].get("temperature_2m_max", []),
                "humidity": data["daily"].get("relative_humidity_2m_mean", []),
                "pressure": data["daily"].get("surface_pressure_mean", []),
                "wind_speed": data["daily"].get("windspeed_10m_max", []),
                "precipitation": data["daily"].get("precipitation_sum", [])
            })

            # Add additional time features
            df["day_of_year"] = df["date"].dt.dayofyear
            df["month"] = df["date"].dt.month
            df["day"] = df["date"].dt.day

            return df

        except requests.exceptions.HTTPError as e:
            if response.status_code == 400:
                print(f"Bad Request: {response.json()}")  # Print the error details from the API
                break
            elif response.status_code == 429:
                print("Rate limit exceeded. Waiting 60 seconds before retrying...")
                time.sleep(60)
            else:
                print(f"HTTP Error: {e}. Retrying in 60 seconds...")
                time.sleep(60)
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}. Retrying in 60 seconds...")
            time.sleep(60)
        except ValueError as e:
            print(f"Value Error: {e}")
            break
# Routes
@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    """Generate and return weather predictions for a selected station"""
    days = int(request.form.get('days', 7))
    selected_station = request.form.get('station')
    print(f"Station is {selected_station}")

    # Load station metadata
    stations_file = f'{DATA_PATH}/Guwahati_stations.csv'
    try:
        stations_df = pd.read_csv(stations_file)
    except FileNotFoundError:
        return jsonify({"error": f"{stations_file} not found. Please provide the file."}), 400

    station = stations_df[stations_df['name'] == selected_station].iloc[0].to_dict()
    name = station["name"]
    latitude = station["latitude"]
    longitude = station["longitude"]
    print(station)
    # Fetch historical weather
    end_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    df = get_historical_weather(latitude, longitude, start_date, end_date)

    # Preprocess and filter up to today
    processed_df = preprocess_data(df)
    current_date = datetime.now().date()
    processed_df = processed_df[processed_df['date'] <= pd.Timestamp(current_date)]
    print(processed_df)
    print("Column Names")
    print(processed_df.columns)
    # Make predictions (all targets at once)
    pred_df = predict_weather_lstm(processed_df, days)
    pred_df = pred_df[pred_df['date'] >= pd.Timestamp(current_date)]

    # Extract target predictions into dictionary format
    targets = ['temp_min', 'temp_max', 'humidity', 'precipitation']
    predictions = {
        target: pred_df[['date', target]].to_dict('records')
        for target in targets
    }

    # Visualizations and feature importances
    visualizations = {
        target: create_visualization(df, pred_df, target)
        for target in targets
    }
    print("Visualization done")
    feature_importances = {
        target: feature_importance_plot(target, processed_df.drop(columns=targets), processed_df[targets].values)
        for target in targets
    }

    # Return full result
    return jsonify({
        'station': name,
        'predictions': predictions,
        'visualizations': visualizations,
        'feature_importances': feature_importances
    })


if __name__ == '__main__':
    app.run(debug=True)
