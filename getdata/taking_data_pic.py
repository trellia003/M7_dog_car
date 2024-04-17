# import socket
# import requests
# import os
# from datetime import datetime
#
# # Get the directory of the Python script
# current_directory = os.path.dirname(__file__)
#
# # Set the IP addresses and ports
# arduino_ip = "192.168.43.19"
# esp_ip = "192.168.43.128"
# arduino_port = 23
#
# # Create a socket object
# client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#
# # Connect to the Arduino server
# client_socket.connect((arduino_ip, arduino_port))
#
# # Create a folder to store images if it doesn't exist
# data_folder = os.path.join(current_directory, "data")
# if not os.path.exists(data_folder):
#     os.makedirs(data_folder)
#
# # Function to send a message to the Arduino server
# def send_message(message):
#     client_socket.sendall(message.encode())
#
# # Function to receive messages from the Arduino server
# def receive_message():
#     data = client_socket.recv(1024).decode()
#     return data
#
# # Continuously listen for messages from the Arduino server
# while True:
#     # Receive a message from the Arduino server
#     message = receive_message()
#     print("Received from Arduino:", message)
#
#     # Check if the message is "take pic"
#     if message.strip() == "take pic":
#         # Send a request to the ESP32-CAM to get the frame
#         try:
#             response = requests.get(f"http://{esp_ip}/capture")
#             if response.status_code == 200:
#                 print("Frame captured successfully.")
#
#                 # Generate a unique filename using timestamp
#                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                 filename = os.path.join(data_folder, f"frame_{timestamp}.jpg")
#
#                 # Save the frame to a file
#                 with open(filename, "wb") as f:
#                     f.write(response.content)
#                     print(f"Frame saved to '{filename}'")
#             else:
#                 print("Failed to capture frame from ESP32-CAM.")
#         except Exception as e:
#             print(f"Error: {e}")
#
# # Close the socket connection
# client_socket.close()


import socket
import requests
import os
from datetime import datetime

# Get the directory of the Python script
current_directory = os.path.dirname(__file__)

# Set the IP addresses and ports
arduino_ip = "192.168.43.19"
esp_ip = "192.168.43.128"
arduino_port = 23

# Create a socket object
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the Arduino server
client_socket.connect((arduino_ip, arduino_port))

# Create a folder to store images if it doesn't exist
data_folder = os.path.join(current_directory, "data")
if not os.path.exists(data_folder):
    os.makedirs(data_folder)


# Function to send a message to the Arduino server
def send_message(message):
    client_socket.sendall(message.encode())


# Function to receive messages from the Arduino server
def receive_message():
    data = client_socket.recv(1024).decode()
    return data


# Continuously listen for messages from the Arduino server
while True:
    # Receive a message from the Arduino server
    message = receive_message()
    print("Received from Arduino:", message)

    # Check if the message is "take pic"
    if message.strip() == "take pic":
        # Send a request to the ESP32-CAM to get the frame
        try:
            response = requests.get(f"http://{esp_ip}/capture")
            if response.status_code == 200:
                print("Frame captured successfully.")

                # Generate a unique filename using timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(data_folder, f"frame_{timestamp}.jpg")

                # Save the frame to a file
                with open(filename, "wb") as f:
                    f.write(response.content)
                    print(f"Frame saved to '{filename}'")

                # Send success message to Arduino
                send_message("pic")
            else:
                print("Failed to capture frame from ESP32-CAM.")
                # Send failure message to Arduino
                send_message("failed")
        except Exception as e:
            print(f"Error: {e}")
            # Send failure message to Arduino
            send_message("failed")

# Close the socket connection
client_socket.close()


