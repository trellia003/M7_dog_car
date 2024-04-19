import cv2
import os
import time
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import threading
import socket

arduino_ip = "192.168.43.19"
esp32_cam_ip = "192.168.43.128"
arduino_port = 23


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
    speed = 93
    message = f"{command}:{speed}"
    sock.sendall(message.encode())


def display_video():
    frame_generator = fetch_frames()
    for frame in frame_generator:
        img = Image.open(BytesIO(frame))
        img = np.array(img)
        cv2.imshow("ESP32-CAM Video Stream", img)
        key = cv2.waitKey(1)
        if key == ord('e'):
            send_to_arduino("s")
        elif key == ord('w'):
            send_to_arduino("f")
        elif key == ord('s'):
            send_to_arduino("b")
        elif key == ord('d'):
            send_to_arduino("r")
        elif key == ord('a'):
            send_to_arduino("l")
        elif key == ord('k'):
            save_image(img)
        elif key == ord('p'):
            break

    cv2.destroyAllWindows()
    sock.close()




def save_image(img):
    # Create the smartxp folder if it does not exist
    folder_path = "../getdata/data/backpack"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Generate a unique filename using current timestamp
    timestamp = int(time.time())
    filename = os.path.join(folder_path, f"image_{timestamp}.jpg")

    # Save the image as JPG
    cv2.imwrite(filename, img)


if __name__ == "__main__":
    display_video_thread = threading.Thread(target=display_video)
    display_video_thread.start()
