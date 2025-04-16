# app.py - Main Flask Application
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import datetime
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.python.keras.models import load_model
import tensorflow as tf
from datetime import timedelta
import time
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt
import base64
import io
from sklearn.inspection import permutation_importance
from utility_functions import * # ADD SMALL UTILITY FUNCTIONS IN THE SCRIPT AND CALL IT HERE
app = Flask(__name__)

# Configuration
MODEL_PATH = 'models'
DATA_PATH = 'data'
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)
API_KEY = "aac405e628f9c30a047d3de13192a7f7"
BASE_URL = "http://api.openweathermap.org/data/2.5/air_pollution/history"

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
    


# Routes
@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/weather', methods=['GET'])
def weather():
    """Render the Weather Prediction page"""
    return render_template('weather.html')

@app.route('/aqi_prediction')
def aqi_prediction():
    """Render the AQI Prediction page"""
    return render_template('aqi_prediction.html')






@app.route('/predict-aqi', methods=['POST'])
def predict_aqi():
    days = int(request.form.get('days', 7))
    station = request.form.get('station')
    print(f"Station is {station}")

    stations = {
        'Railway Colony, Guwahati - APCB_6941': {'latitude': 26.1445, 'longitude': 91.7362},
        'Railway Colony, Guwahati - APCB_10903': {'latitude': 26.181742, 'longitude': 91.78063},
        'Pan Bazaar, Guwahati - APCB_42240': {'latitude': 26.1875, 'longitude': 91.744194},
        'IITG, Guwahati - PCBA_361411': {'latitude': 26.2028636, 'longitude': 91.70046436},
        'IITG, Guwahati - PCBA_3409360': {'latitude': 26.2028636, 'longitude': 91.70046436},
        'LGBI Airport, Guwahati - PCBA_3409390': {'latitude': 26.10887, 'longitude': 91.589544}
    }

    # Set up for fetching the latest 7 days with valid data
    end_date = datetime.date.today()
    window_size = 7  # We need a full 7-day window for predictions
    records = []
    current_date = end_date

    print(f"\nFetching air quality data for {station} (lat: {stations[station]['latitude']}, lon: {stations[station]['longitude']})")
    # Keep going back day-by-day until we have 7 valid records
    while len(records) < window_size:
        avg_aqi, avg_components = fetch_daily_air_quality(stations[station], current_date)
        
        # If the API returns valid data (adjust the condition as needed)
        if avg_aqi is not None:
            # Build a record (dictionary) with the station name, date, and pollutant averages
            record = {
                "station": station,
                "date": current_date,
                "avg_aqi": avg_aqi
            }
            for comp, value in avg_components.items():
                record[comp] = value

            records.append(record)
            print(f"  {current_date}: AQI={avg_aqi}, Pollutants={avg_components}")
        else:
            print(f"  No data for {current_date}. Skipping.")

        # Go one day back and pause briefly (to respect API rate limits)
        current_date -= datetime.timedelta(days=1)
        time.sleep(1)

    # Sort the records in ascending order by date to form a proper time series
    records = sorted(records, key=lambda r: r["date"])
    df = pd.DataFrame(records)

    # Prepare for prediction using the last 7 days of data
    pollutants = ['avg_aqi', 'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
    window_size = 7

    # Ensure the data is sorted by date and drop any rows with missing values in required columns
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=pollutants)

    # Create the initial window from the latest 7 days
    window = df[pollutants].values[-window_size:]  # shape: (7, 9)

    # Load scaler and trained model
    scaler = joblib.load("models/aqi_scaler.pkl")
    model = tf.keras.models.load_model("models/aqi_best_model.keras")

    predictions = []
    prediction_dates = []

    # Predict AQI for the specified number of future days
    for step in range(days):
        window_flat = window.flatten().reshape(1, -1)   # shape: (1, 63)
        X_scaled = scaler.transform(window_flat)
        next_day_pred = model.predict(X_scaled)[0]  # shape: (9,)
        predictions.append(next_day_pred)

        # Update the window by dropping the oldest day and appending the new prediction
        window = np.vstack([window[1:], next_day_pred])
        prediction_dates.append(end_date + datetime.timedelta(days=step + 1))

    # Convert predictions to a DataFrame for easier handling
    pred_df = pd.DataFrame(predictions, columns=pollutants)
    pred_df['date'] = prediction_dates

    # Generate a plot of the predicted AQI over time
    plt.figure(figsize=(10, 6))
    plt.plot(pred_df['date'], pred_df['avg_aqi'], label='Predicted AQI', marker='o')
    plt.xlabel('Date')
    plt.ylabel('AQI')
    plt.title('Predicted AQI Over Time')
    plt.legend()
    plt.grid(True)

    # Save the plot to a base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    graph = base64.b64encode(image_png).decode('utf-8')

    return jsonify({
        'predictions': pred_df.to_dict('records'),
        'plot': f'data:image/png;base64,{graph}'
    })


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


    print(f"Station is {station}")
    # Fetch historical weather
    end_date = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
    print(start_date, end_date)
    df = get_historical_weather(latitude, longitude, start_date, end_date)
    print(df)
    # Preprocess and filter up to today
    processed_df = preprocess_data(df)
    current_date = datetime.datetime.now().date()
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

#####################################################
#######################################################
######################################################
# REAL TIME DATA RELATED CODE BELOW
#----------------------------------------

from collections import deque
pollutants = ['avg_aqi', 'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
latest_data = deque(maxlen=7)  # Store the latest 7 days of data
# Initialize with empty readings
for _ in range(7):
    latest_data.append({pollutant: None for pollutant in pollutants})
    latest_data[-1]["alert"] = None  # Include alert field

@app.route("/get_latest_data")
def get_latest_data():
    return jsonify(list(latest_data))

@app.route('/iot_prediction')
def iot_prediction():
    """Render the Prediction Using IoT Device page"""
    return render_template('iot_prediction.html', data=latest_data)

@app.route('/about')
def about():
    return render_template('about.html')


@app.route("/post_data", methods=["POST"])
def post_data():
    if request.is_json:
        data = request.get_json()
        print("HTTP POST received:", data)

        # Construct a new entry with pollutants and alert status
        new_entry = {pollutant: data.get(pollutant) for pollutant in pollutants}
        new_entry["alert"] = data.get("alert")

        # Append the latest data, automatically maintaining the sliding window
        latest_data.append(new_entry)

        # Trigger alert if needed
        if data.get("alert") == 1:
            send_weather_alert("Pollution Alert", "High pollution levels detected!")
            play_alert_sound()

        return jsonify({"status": "success"}), 200
    else:
        return jsonify({"error": "Invalid or missing JSON payload"}), 400

###########################################################################
###########################################################################
##########################################################################

    

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
