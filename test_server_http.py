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



if __name__ == "__main__":
    # Run the Flask server on all network interfaces on port 5000 with debug mode enabled.
    app.run(host="0.0.0.0", port=5000, debug=True)
