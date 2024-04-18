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
        pred_bbox = [[0.0, 0.0, 1.0, 1.0]]

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
    cv2.waitKey(500)
    cv2.destroyAllWindows()

def control_robot(input_size: int, label, bbox):
    x, y, x2, y2 = bbox
    x_mid, y_mid = (x + x2) / 2, (y + y2) / 2

    margin_percentage = 0.1

    left_bound = input_size * (0.5 - margin_percentage)
    right_bound = input_size * (0.5 + margin_percentage)
    top_bound = input_size * (0.5 - margin_percentage)
    bottom_bound = input_size * (0.5 + margin_percentage)

    if label == 1:
        print('No backpack')
        # TODO Drive forwards or smth; seek for bag
    elif label == 0:
        if x_mid < left_bound:
            print('x_mid < left_bound')
            # TODO Move right
        elif x_mid > right_bound:
            print('x_mid > left_bound')
            # TODO Move left
        elif y_mid < top_bound:
            print('y_mid < top_bound')
            # TODO Move forwards
        elif y_mid > bottom_bound:
            print('y_mid > bottom_bound')
            # TODO Move backwards
        else:
            print('Stop')
            # TODO Stop moving for like 2 sec and then rotate ish?

def display_video():
    frame_generator = fetch_frames()
    for frame in frame_generator:
        img = Image.open(BytesIO(frame))
        img = np.array(img)
        formatted_image = format_image(input_size, img)
        label, bbox = predict(model, formatted_image)
        label, bbox = label[0], bbox[0]

        # control_robot(input_size, label, bbox)

        print('Prediction class', label)
        display_image_with_box(formatted_image,bbox)

        # key = cv2.waitKey(1)
        # if key == ord('e'):
        #     send_to_arduino("s")
        # elif key == ord('w'):
        #     send_to_arduino("f")
        # elif key == ord('s'):
        #     send_to_arduino("b")
        # elif key == ord('d'):
        #     send_to_arduino("r")
        # elif key == ord('a'):
        #     send_to_arduino("l")
        # elif key == ord('k'):
        #     save_image(img)
        # elif key == ord('p'):
        #     break

    cv2.destroyAllWindows()
    sock.close()

    # Adjust the frequency by changing the time delay
    # time.sleep(1)  # Change the value to set the frequency (in seconds)


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
    input_size = 96
    model_path = 'train_backpacks.keras'
    model = model_load(model_path)
    display_video_thread = threading.Thread(target=display_video)
    display_video_thread.start()


















