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
    message = "Your t\n"
    sock.sendall(message.encode())
finally:
    sock.close()
