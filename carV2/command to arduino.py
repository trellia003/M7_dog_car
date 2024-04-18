
import socket

# Set the IP and Port where your Arduino Uno WiFi Rev2 is listening
arduino_ip = "192.168.43.19"
arduino_port = 23


def send_to_arduino(command, speed):
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        # Connect the socket to the server's address and port
        server_address = (arduino_ip, arduino_port)
        sock.connect(server_address)

        # Send data to Arduino
        message = f"{command}:{speed}"
        sock.sendall(message.encode())
    finally:
        # Close the socket connection
        sock.close()


if __name__ == "__main__":
    # Example usage
    command = ["stop","forward","backward","left","right"]
    speed = 100
    send_to_arduino(command[0], speed)