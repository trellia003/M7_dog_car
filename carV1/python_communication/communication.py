import cv2
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import threading
import socket

# esp32_cam_ip = "192.168.1.173"
# arduino_ip = "192.168.1.199"  # Change this to your Arduino's IP address


arduino_ip = "192.168.43.19"
esp32_cam_ip = "192.168.43.128"
arduino_port = 23
speed = 93

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((arduino_ip, arduino_port))

def fetch_frames():
    url = f"http://{esp32_cam_ip}/capture"
    while True:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            yield response.content

def send_to_arduino(command):
    message = f"Com:{command}:{speed}:end"
    sock.sendall(message.encode())

def display_video():
    frame_generator = fetch_frames()
    for frame in frame_generator:
        img = Image.open(BytesIO(frame))
        img = np.array(img)
        cv2.imshow("ESP32-CAM Video Stream", img)
        key = cv2.waitKey(1)
        if key == ord('s'):  # Press 's' to send message to Arduino
            send_to_arduino("stop")  # Send stop command to Arduino
            print("Sent 'stop' command to Arduino")
        elif key == ord('w'):  # Press 'f' to send message to Arduino
            send_to_arduino("forward")  # Send forward command to Arduino
            print("Sent 'forward' command to Arduino")
        elif key == ord('e'):  # Press 's' to send message to Arduino
            send_to_arduino("onspotright")  # Send stop command to Arduino
            print("Sent 'onspotright' command to Arduino")
        elif key == ord('q'):  # Press 'f' to send message to Arduino
            send_to_arduino("onspotleft")  # Send forward command to Arduino
            print("Sent 'onspotleft' command to Arduino")
        elif key == ord('d'):  # Press 's' to send message to Arduino
            send_to_arduino("pvotright")  # Send stop command to Arduino
            print("Sent 'pvotright' command to Arduino")
        elif key == ord('a'):  # Press 'f' to send message to Arduino
            send_to_arduino("pvotleft")  # Send forward command to Arduino
            print("Sent 'pvotleft' command to Arduino")
        elif key == ord('k'):  # Press 'q' to quit
            break

    cv2.destroyAllWindows()
    sock.close()

if __name__ == "__main__":
    display_video_thread = threading.Thread(target=display_video)
    display_video_thread.start()
