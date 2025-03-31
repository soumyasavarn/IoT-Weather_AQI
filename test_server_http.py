from flask import Flask, render_template, request, jsonify
import threading

app = Flask(__name__)

# Global list to store weather data received via HTTP
weather_data_list = []
# Lock to ensure thread-safe access to the global list
weather_data_lock = threading.Lock()

# Flask route to serve the main page
@app.route("/")
def index():
    return render_template("index_test_server.html")

# HTTP GET endpoint to return the collected weather data as JSON
@app.route("/data", methods=["GET"])
def get_data():
    with weather_data_lock:
        data = list(weather_data_list)
        print (jsonify(data))
    return jsonify(data)

# HTTP POST endpoint to receive new weather data from the ESP32 or any client
@app.route("/post_data", methods=["POST"])
def post_data():
    # Expecting a JSON payload, e.g., {"temperature": 25, "humidity": 60}
    if request.is_json:
        data = request.get_json()
        print("HTTP POST received:", data)
        with weather_data_lock:
            weather_data_list.append(data)
            # Keep only the latest 10 records
            if len(weather_data_list) > 10:
                weather_data_list[:] = weather_data_list[-10:]
        return jsonify({"status": "success"}), 200
    else:
        return jsonify({"error": "Invalid or missing JSON payload"}), 400

if __name__ == "__main__":
    # Run the Flask server on all network interfaces on port 5000 with debug mode enabled.
    app.run(host="0.0.0.0", port=5000, debug=True)
