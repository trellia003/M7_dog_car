import socket

# Set the IP and Port where your Arduino Uno WiFi Rev2 is listening
arduino_ip = "192.168.1.199"  # Change this to your Arduino's IP address
arduino_port = 23

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect the socket to the server's address and port
server_address = (arduino_ip, arduino_port)
sock.connect(server_address)
try:
    # Send data to Arduino
    # command = "onspotleft"
    # command = "pvotleft"
    # command = "onspotright"
    # command = "pvotright"
    command = "forward"
    # command = "stop"
    speed = 100
    message = f"Com:{command}:{speed}:end"
    sock.sendall(message.encode())
finally:
    sock.close()


# camerawebserver

#
# import cv2
# import requests
# import numpy as np
#
# # Replace this with your ESP32-CAM's IP address
# esp32_cam_ip = "192.168.1.173"
#
#
# def display_video():
#     url = f"http://{esp32_cam_ip}/capture"
#     try:
#         while True:
#             # Get the video stream from ESP32-CAM
#             response = requests.get(url, stream=True)
#             if response.status_code == 200:
#                 # Convert the streamed data into a numpy array
#                 nparr = np.frombuffer(response.content, np.uint8)
#                 # Decode the numpy array into an image
#                 img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#                 # Display the image
#                 cv2.imshow("ESP32-CAM Video Stream", img)
#                 # Wait for 1 millisecond and check for user input to close the window
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break
#             else:
#                 print("Failed to fetch frame")
#     except KeyboardInterrupt:
#         cv2.destroyAllWindows()
#     except Exception as e:
#         print(f"Error: {e}")
#
#
# if __name__ == "__main__":
#     display_video()





#
# import cv2
# import requests
# import numpy as np
# from PIL import Image
# from io import BytesIO
# import threading
# import time
#
# esp32_cam_ip = "192.168.1.173"
#
# def fetch_frames():
#     url = f"http://{esp32_cam_ip}/capture"
#     while True:
#         response = requests.get(url, stream=True)
#         if response.status_code == 200:
#             yield response.content
#
# def display_video():
#     frame_generator = fetch_frames()
#     start_time = time.time()
#     frame_count = 0
#     try:
#         for frame in frame_generator:
#             img = Image.open(BytesIO(frame))
#             img = np.array(img)
#             cv2.imshow("ESP32-CAM Video Stream", img)
#             frame_count += 1
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#             if frame_count % 10 == 0:  # Print FPS every 10 frames
#                 end_time = time.time()
#                 elapsed_time = end_time - start_time
#                 fps = frame_count / elapsed_time if elapsed_time > 0 else 0
#                 print(f"Current FPS: {fps:.2f}")
#     except KeyboardInterrupt:
#         cv2.destroyAllWindows()
#     except Exception as e:
#         print(f"Error: {e}")
#
# if __name__ == "__main__":
#     display_thread = threading.Thread(target=display_video)
#     display_thread.start()


