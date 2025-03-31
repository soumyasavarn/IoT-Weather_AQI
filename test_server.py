from flask import Flask, render_template, jsonify
import paho.mqtt.client as mqtt
import threading

app = Flask(__name__)

# Global list to store weather data received from MQTT
weather_data_list = []

# MQTT Callbacks
def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT broker with result code " + str(rc))
    client.subscribe("weather/alert")  # Subscribe to topic

def on_message(client, userdata, msg):
    global weather_data_list
    # try:
    #     # Attempt to decode the message as UTF-8
    #     message = msg.payload.decode("utf-8")
    # except UnicodeDecodeError:
    #     # If decoding fails, handle the error gracefully
    #     print("Failed to decode message payload as UTF-8. Using fallback decoding.")
    message = msg.payload.decode("latin-1", errors="replace")  # Fallback decoding

    print("MQTT Message received:", message)
    weather_data_list.append(message)
    print (type(message))
    # Keep only the latest 10 records
    weather_data_list[:] = weather_data_list[-10:]

# MQTT thread function
def mqtt_thread():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    # Connect to the Mosquitto broker running on your laptop.
    # Use "localhost" if the broker is on the same machine or use your laptop's IP.
    client.connect("localhost", 1883, 60)
    client.loop_forever()

# Start the MQTT client in a separate thread
threading.Thread(target=mqtt_thread, daemon=True).start()

# Flask routes
@app.route("/")
def index():
    return render_template("index_test_server.html")

@app.route("/data")
def get_data():
    # Return the collected weather data as JSON
    return jsonify(weather_data_list)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
