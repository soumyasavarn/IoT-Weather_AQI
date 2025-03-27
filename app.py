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



import numpy as np
import pandas as pd
from datetime import timedelta
import joblib
import tensorflow as tf

def predict_weather_lstm(df, days_to_predict=7, sequence_length=7):
    """
    Predicts future weather using a sliding window approach for an LSTM model.
    
    Parameters:
        df (DataFrame): Historical data containing at least the columns 'date',
                        target columns (e.g. 'temp_min', 'temp_max', 'humidity', 'precipitation'),
                        and additional feature columns.
        days_to_predict (int): How many future days to predict.
        sequence_length (int): Number of time steps (rows) expected by the LSTM.
        
    Returns:
        DataFrame: A DataFrame with columns for date and predicted target values.
    """
    # Load the trained LSTM model and the scalers
    model = tf.keras.models.load_model('models/weather_lstm_model.keras')
    scaler_X = joblib.load('models/scaler_X.pkl')
    scaler_y = joblib.load('models/scaler_y.pkl')
    
    # Define target columns (these must match what your model predicts)
    target_cols = ['temp_min', 'temp_max', 'humidity', 'precipitation']
    
    # Ensure the data is sorted by date
    df = df.sort_values(by='date').reset_index(drop=True)
    
    # Use the last `sequence_length` rows as the initial sequence for prediction.
    current_sequence = df.tail(sequence_length).copy().reset_index(drop=True)
    predictions = []
    
    # Get the most recent date from the historical data.
    last_date = df['date'].max()
    
    for i in range(days_to_predict):
        # Prepare input features: drop columns that are not features.
        # Here, we drop 'date' and the target columns. Adjust if your scaler was fit on a different set.
        X_pred = current_sequence.drop(columns=['date', 'month', 'day'] + target_cols, errors='ignore')
        # X_pred shape should be (sequence_length, n_features)
        
        # Scale the features using the pre-fitted scaler.
        X_scaled = scaler_X.transform(X_pred)
        # Reshape to match LSTM expected input: (1, sequence_length, n_features)
        X_scaled = X_scaled.reshape((1, sequence_length, X_pred.shape[1]))
        
        # Get prediction (the model outputs a scaled prediction for the targets)
        y_pred_scaled = model.predict(X_scaled)[0]
        # Inverse transform to get the actual predicted values.
        y_pred = scaler_y.inverse_transform([y_pred_scaled])[0]
        
        # Determine the new date for this prediction.
        new_date = last_date + timedelta(days=i+1)
        
        # Build a new row for the predicted day.
        # For non-target features, copy the last row of the current sequence.
        new_row = current_sequence.iloc[-1].copy()
        new_row['date'] = new_date
        
        # Update seasonal features if they are part of your model.
        # (Assuming you use day_of_year, and its sine and cosine transforms)
        day_of_year = new_date.timetuple().tm_yday
        if 'day_of_year' in new_row:
            new_row['day_of_year'] = day_of_year
        if 'day_of_year_sin' in new_row:
            new_row['day_of_year_sin'] = np.sin(2 * np.pi * day_of_year / 365)
        if 'day_of_year_cos' in new_row:
            new_row['day_of_year_cos'] = np.cos(2 * np.pi * day_of_year / 365)
        
        # Update target columns with the predicted values.
        for j, col in enumerate(target_cols):
            new_row[col] = y_pred[j]
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(1, -1))[0]
        # Clip negative precipitation predictions to 0
        y_pred[3] = np.clip(y_pred[3], 0, None)

        # Save the prediction in a list.
        predictions.append({
            'date': new_date.strftime('%Y-%m-%d'),
            'temp_min': y_pred[0],
            'temp_max': y_pred[1],
            'humidity': y_pred[2],
            'precipitation': y_pred[3]
        })
        
        # Update the current sequence by dropping the oldest row and appending the new prediction.
        new_row_df = pd.DataFrame([new_row])
        current_sequence = pd.concat([current_sequence.iloc[1:], new_row_df], ignore_index=True)
    
    # Convert predictions list to a DataFrame before returning.
    return pd.DataFrame(predictions)


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
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
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
    pred_df['date'] = pd.to_datetime(pred_df['date'])
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

    # Return full result
    try:
        # Your existing prediction logic here
        return jsonify({
            'station': name,
            'predictions': predictions,
            'visualizations': visualizations
        })
    except Exception as e:
        # Log the error details
        app.logger.error("Error in /predict: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
