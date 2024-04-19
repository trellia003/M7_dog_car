import cv2
import os
import time
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import threading
import socket
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import Model

arduino_ip = "192.168.43.19"
esp32_cam_ip = "192.168.43.128"
arduino_port = 23

buffer_len = 3
decision_buffer = ["s"] * buffer_len

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((arduino_ip, arduino_port))


# Define signal handler function
def terminate():
    num_iterations = 5
    for _ in range(num_iterations):
        # Reset decision_buffer to contain all "s" values
        for i in range(buffer_len):
            decision_buffer[i] = "s"
        send_to_arduino()
        # Introduce a delay of 200 ms between each signal
        time.sleep(0.2)
    print('Script has been terminated.')
    sock.close()  # Close the socket connection
    cv2.destroyAllWindows()
    quit()


def fetch_frames():
    url = f"http://{esp32_cam_ip}/capture"
    while True:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            yield response.content


def send_to_arduino():
    speed = 100
    # print(decision_buffer)

    if all(item == decision_buffer[0] for item in decision_buffer):
        message = f"{decision_buffer[0]}:{speed}"
        print("message:" + message)
        # else:
        # message = f"{decision_buffer[0]}:{speed}"
        sock.sendall(message.encode())
    else:
        message = f"s:{speed}"
        print("message:" + message)
        # else:
        # message = f"{decision_buffer[0]}:{speed}"
        sock.sendall(message.encode())


def model_load(filepath: str) -> Model:
    loaded_model = load_model(filepath)
    print(f"Model loaded from: {filepath}")
    return loaded_model


def predict(model: Model, image):
    # Add a batch dimension to the input image
    image = np.expand_dims(image, axis=0)

    pred_label, pred_bbox = model.predict(image, verbose=0)
    pred_label = np.argmax(pred_label, axis=1)

    # If no backpacks are detected, the whole image should be the bounding box
    if pred_label == 1:
        pred_bbox = [[0.0, 0.0, input_size - 1, input_size - 1]]

    return np.array(pred_label), np.array(pred_bbox)


def format_image(input_size: int, image: np.ndarray) -> np.ndarray:
    new_image = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    new_image = new_image.astype(np.float32) / 255.0
    return new_image


def display_image_with_box(image: np.ndarray, bounding_box: np.ndarray) -> None:
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Extract bounding box coordinates
    x, y, x2, y2 = bounding_box

    # Draw bounding box on the image
    cv2.rectangle(image, (int(x), int(y)), (int(x2), int(y2)), (0, 0, 255), 1)

    # Display the image with the bounding boxes
    window_name = 'Image with Bounding Box'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 500, 500)
    cv2.imshow(window_name, image)
    key = cv2.waitKey(50)
    if key == ord('q'):
        terminate()


def control_robot(input_size: int, label, bbox):
    x, y, x2, y2 = bbox
    x_mid, y_mid = (x + x2) / 2, (y + y2) / 2

    margin_percentage_orizontal = 0.235

    left_bound = input_size * (0.5 - margin_percentage_orizontal)
    right_bound = input_size * (0.5 + margin_percentage_orizontal)
    top_bound = input_size * (0.5 - 0.1)
    bottom_bound = input_size * (0.5 + 0.1)
    # print(label)
    # print(f"bounds: {left_bound}, {right_bound},{top_bound},{bottom_bound}")
    # print(f"{x_mid}, {y_mid}")

    if label == 1:
        print('No backpack  l')
        decision_buffer.append("l")
        # TODO Drive forwards or smth; seek for bag  Stop moving for like 2 sec and then rotate ish?
    elif label == 0:
        if x_mid < left_bound:
            print('x_mid < left_bound  l')
            decision_buffer.append("l")
            # TODO Move right
        elif x_mid > right_bound:
            print('x_mid > left_bound  r')
            decision_buffer.append("r")
            # TODO Move left
        elif y_mid < top_bound:
            print('y_mid < top_bound  f')
            decision_buffer.append("f")
            # TODO Move forwards
        elif y_mid > bottom_bound:
            print('y_mid > bottom_bound  b')
            decision_buffer.append("b")
            # TODO Move backwards
        else:
            print('Stop     s')
            decision_buffer.append("s")
            # TODO pee
    decision_buffer.pop(0)
    send_to_arduino()


def display_video():
    frame_generator = fetch_frames()
    for frame in frame_generator:
        print("")
        img = Image.open(BytesIO(frame))
        img = np.array(img)
        formatted_image = format_image(input_size, img)
        label, bbox = predict(model, formatted_image)
        label, bbox = label[0], bbox[0]

        print('Prediction class', label)
        control_robot(input_size, label, bbox)

        display_image_with_box(formatted_image, bbox)
    sock.close()

    # Adjust the frequency by changing the time delay
    # time.sleep(1)  # Change the value to set the frequency (in seconds)


if __name__ == "__main__":
    input_size = 96
    model_path = '../models/combined_96.keras'
    model = model_load(model_path)
    display_video_thread = threading.Thread(target=display_video)
    display_video_thread.start()
