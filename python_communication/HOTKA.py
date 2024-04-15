import cv2
import requests
import numpy as np
import threading
import time

esp32_cam_ip = "192.168.1.173"

def fetch_frames():
    url = f"http://{esp32_cam_ip}/capture"
    while True:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            yield response.content

def display_video():
    frame_generator = fetch_frames()
    start_time = time.time()
    frame_count = 0
    try:
        for frame in frame_generator:
            nparr = np.frombuffer(frame, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cv2.imshow("ESP32-CAM Video Stream", img)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if frame_count % 10 == 0:  # Print FPS every 10 frames
                end_time = time.time()
                elapsed_time = end_time - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                print(f"Current FPS: {fps:.2f}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    display_thread = threading.Thread(target=display_video)
    display_thread.start()
