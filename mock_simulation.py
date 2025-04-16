# THIS IS A MOCK SCRIPT SIMILAR TO MICROCONTROLLER'S DATA SENDING FUNCTIONALITY
# This script simulates the behavior of a microcontroller sending data to a server  
# Run it on a computer with Python installed and connected to the same network as the server.
# It generates random air quality data and sends it to the server every 5 seconds.  


import socket
import json
import time
import random

# Configuration
SSID = "SOUMYA 6843"
PASSWORD = "4e%8R843"
SERVER_IP = "10.150.36.127"
SERVER_PORT = 5000

pollutants = ['avg_aqi', 'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']

def get_current_time():
    t = time.localtime()
    return "{:02d}:{:02d}:{:02d} {:02d}/{:02d}/{:04d} IST".format(
        t[3], t[4], t[5], t[2], t[1], t[0])

def get_simulated_date(counter):
    day = 26 + counter
    return f"2024-04-{day:02d}"

def generate_simulated_data(counter):
    return {
        "station": "Railway Colony, Guwahati - APCB",
        "date": get_simulated_date(counter),
        "avg_aqi": round(random.uniform(2.5, 3.5), 2),
        "co": round(random.uniform(700, 900), 2),
        "no": round(random.uniform(5.0, 10.0), 2),
        "no2": round(random.uniform(8.0, 14.0), 2),
        "o3": round(random.uniform(40.0, 60.0), 2),
        "so2": round(random.uniform(3.0, 7.0), 2),
        "pm2_5": round(random.uniform(25.0, 40.0), 2),
        "pm10": round(random.uniform(30.0, 65.0), 2),
        "nh3": round(random.uniform(5.0, 12.0), 2),
        "alert": random.random() < 0.3
    }

def send_http_post(data):
    json_data = json.dumps(data)
    request = (
        f"POST /post_data HTTP/1.1\r\n"
        f"Host: {SERVER_IP}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(json_data)}\r\n\r\n"
        f"{json_data}"
    )
    try:
        sock = socket.socket()
        sock.connect((SERVER_IP, SERVER_PORT))
        sock.send(request.encode())
        response = sock.recv(1024)
        print("Server response:", response)
    except Exception as e:
        print("Connection failed:", e)
    finally:
        sock.close()

def main_loop():
    counter = 0
    while True:
        sim_data = generate_simulated_data(counter)
        send_http_post(sim_data)
        counter += 1
        time.sleep(5)

if __name__ == "__main__":
    main_loop()
