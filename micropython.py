## This code is to run in the microcontroller using micropython like IDE
import network
import socket
import json
import time
import random
import M5
from M5 import Widgets, BtnA, BtnB, BtnC

# Configuration
SSID = "SOUMYA 6843"
PASSWORD = "4e%8R843"
SERVER_IP = "192.168.124.11"
SERVER_PORT = 8080

# Widgets and global variables
title_widget = None
data_labels = {}
current_view = 0  # 0: Overview, 1: Details, 2: Graphs
alert_indicator = None
refresh_button = None

def init_device():
    global title_widget, alert_indicator, refresh_button, data_labels
    M5.begin()
    M5.Display.setBrightness(50)
    
    # Initialize UI components
    Widgets.fillScreen(0x222222)
    title_widget = Widgets.Title("Weather Station", 25, 0xffffff, 0x0000FF, Widgets.FONTS.DejaVu24)
    
    # Alert indicator
    alert_indicator = Widgets.Circle(290, 10, 8, 0xff0000, 0xff0000)
    
    # Refresh button (remains at bottom-right)
    refresh_button = Widgets.Rectangle(270, 190, 40, 40, 0x00ff00, 0x444444)
    Widgets.Label("⟳", 275, 195, 2.0, 0xffffff, 0x444444, Widgets.FONTS.DejaVu18)
    
    # Prepare data labels (split into two columns)
    data_labels = {}
    # Left column x coordinate and y positions for the first 6 items
    left_x = 10
    left_y_positions = [30, 50, 70, 90, 110, 130]
    # Right column for the remaining 6 items
    right_x = 160
    right_y_positions = [30, 50, 70, 90, 110, 130]

    # Left column labels: time, station, date, AQI, CO, NO2
    data_labels['time'] = Widgets.Label("", left_x, left_y_positions[0], 1.0, 0xffffff, 0x222222, Widgets.FONTS.DejaVu12)
    data_labels['station'] = Widgets.Label("", left_x, left_y_positions[1], 1.0, 0xffffff, 0x222222, Widgets.FONTS.DejaVu12)
    data_labels['date'] = Widgets.Label("", left_x, left_y_positions[2], 1.0, 0xffffff, 0x222222, Widgets.FONTS.DejaVu12)
    data_labels['aqi'] = Widgets.Label("", left_x, left_y_positions[3], 1.0, 0xffff00, 0x222222, Widgets.FONTS.DejaVu12)
    data_labels['co'] = Widgets.Label("", left_x, left_y_positions[4], 1.0, 0xffffff, 0x222222, Widgets.FONTS.DejaVu12)
    data_labels['no2'] = Widgets.Label("", left_x, left_y_positions[5], 1.0, 0xffffff, 0x222222, Widgets.FONTS.DejaVu12)

    # Right column labels: O3, SO2, PM2.5, PM10, NH3, Temp
    data_labels['o3'] = Widgets.Label("", right_x, right_y_positions[0], 1.0, 0xffffff, 0x222222, Widgets.FONTS.DejaVu12)
    data_labels['so2'] = Widgets.Label("", right_x, right_y_positions[1], 1.0, 0xffffff, 0x222222, Widgets.FONTS.DejaVu12)
    data_labels['pm2_5'] = Widgets.Label("", right_x, right_y_positions[2], 1.0, 0x00ff00, 0x222222, Widgets.FONTS.DejaVu12)
    data_labels['pm10'] = Widgets.Label("", right_x, right_y_positions[3], 1.0, 0xffffff, 0x222222, Widgets.FONTS.DejaVu12)
    data_labels['nh3'] = Widgets.Label("", right_x, right_y_positions[4], 1.0, 0xffffff, 0x222222, Widgets.FONTS.DejaVu12)
    data_labels['temp'] = Widgets.Label("", right_x, right_y_positions[5], 1.0, 0xff00ff, 0x222222, Widgets.FONTS.DejaVu12)
    
    # Button callbacks
    BtnB.setCallback(BtnB.CB_TYPE.WAS_RELEASED, toggle_view)
    BtnA.setCallback(BtnA.CB_TYPE.WAS_RELEASED, force_refresh)
    
    # Connect to WiFi
    wifi = network.WLAN(network.STA_IF)
    wifi.active(True)
    wifi.connect(SSID, PASSWORD)
    while not wifi.isconnected():
        print("Connecting to Wi-Fi...")
        time.sleep(1)
    return wifi

def update_display(sim_data, current_time):
    # Update left column labels
    data_labels['time'].setText("Time: " + current_time)
    data_labels['station'].setText("Station: " + sim_data['station'])
    data_labels['date'].setText("Date: " + sim_data['date'])
    data_labels['aqi'].setText(f"AQI: {sim_data['avg_aqi']:.2f}")
    data_labels['co'].setText(f"CO: {sim_data['co']:.2f}")
    data_labels['no2'].setText(f"NO2: {sim_data['no2']:.2f}")
    
    # Update right column labels
    data_labels['o3'].setText(f"O3: {sim_data['o3']:.2f}")
    data_labels['so2'].setText(f"SO2: {sim_data['so2']:.2f}")
    data_labels['pm2_5'].setText(f"PM2.5: {sim_data['pm2_5']:.2f}")
    data_labels['pm10'].setText(f"PM10: {sim_data['pm10']:.2f}")
    data_labels['nh3'].setText(f"NH3: {sim_data['nh3']:.2f}")
    data_labels['temp'].setText(f"Temp: {random.uniform(20,30):.1f}°C")
    
    # Update alert indicator based on simulated alert
    alert_indicator.setColor(0xff0000 if sim_data.get('alert', False) else 0x00ff00)

def handle_touch():
    if M5.Touch.getCount() > 0:
        x, y = M5.Touch.getX(), M5.Touch.getY()
        if 270 <= x <= 310 and 190 <= y <= 230:
            force_refresh(None)
            M5.Speaker.tone(3000, 100)

def toggle_view(state):
    global current_view
    current_view = (current_view + 1) % 3
    M5.Speaker.tone(2000, 50)
    # Add view switching logic here

def force_refresh(state):
    M5.Speaker.tone(4000, 50)
    Widgets.fillScreen(0x222222)
    # Ensure all labels are visible after a screen refresh
    for widget in data_labels.values():
        widget.setVisible(True)

# Helper functions from original code
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

def check_and_alert(sim_data):
    if sim_data["avg_aqi"] > 3.3 or sim_data["pm2_5"] > 38:
        print("Anomaly detected! Triggering alert.")
        M5.Speaker.tone(5000, 1000)

def main_loop():
    counter = 0
    while True:
        M5.update()
        handle_touch()
        
        sim_data = generate_simulated_data(counter)
        current_time = get_current_time()
        
        update_display(sim_data, current_time)
        send_http_post(sim_data)
        check_and_alert(sim_data)
        
        counter += 1
        time.sleep(5)

init_device()
main_loop()
