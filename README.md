# IoT-Weather_AQI

This project aims to monitor and predict Air Quality Index (AQI) and weather conditions using IoT-driven data and machine learning models. It integrates environmental datasets, predictive models, and a web-based dashboard for visualization and interaction.

---

##  Directory Structure

```
IoT-Weather_AQI/
│
├── app.py                        # Main Flask application
├── alert.mp3                    # Audio alert for AQI threshold breach
├── database_credentials.txt     # Stores database credentials (if any)
├── mock_simulation.py           # Simulated data generator for testing
├── README.md                    # Project overview and instructions
├── requirements.txt             # List of Python dependencies
│
├── data/                        # Datasets and model-related files
│   ├── aqi_data_guwahati_1year.csv
│   ├── Guwahati_humidity_rainfall.csv
│   ├── Guwahati_stations.csv
│   ├── Guwahati_weather.csv
│   ├── weather_data.csv
│   ├── models/
│   │   ├── humidity/
│   │   ├── precipitation/
│   │   ├── temp_max/
│   │   ├── temp_min/
│   │   ├── aqi_best_model.keras
│   │   ├── aqi_scaler.pkl
│   │   ├── scaler_X.pkl
│   │   ├── scaler_y.pkl
│   │   └── weather_lstm_model.keras
│
├── templates/                   # HTML templates for the Flask app
│   ├── about.html
│   ├── aqi_prediction.html
│   ├── index.html
│   ├── iot_prediction.html
│   ├── layout.html
│   ├── weather.html
│   └── Under_Construction(in_dev).html
```

---

##  Setup Instructions

1. **Clone the Repository**  
   Download or clone this repository to your local machine.

2. **Install Required Packages**  
   Navigate to the root directory and install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. **Run the Application**

   ```
   python app.py
   ```

4. **View the Dashboard**  
   Open a browser and visit: [http://127.0.0.1:8080/](http://127.0.0.1:8080/)

---

## Features

- Real-time AQI and weather predictions
- Interactive web dashboard with visual insights
- Alerts based on AQI thresholds
- Modular model integration for humidity, temperature, and precipitation

---

## Notes

- Ensure your Python version matches the environment expected by the models.
- You can modify `mock_simulation.py` to simulate incoming sensor data.
- The directory `models/` contains trained `.keras` models and scalers necessary for inference.

---
