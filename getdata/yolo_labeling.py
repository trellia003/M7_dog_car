import os
import cv2
from ultralytics import YOLO

model = YOLO("yolov9c.pt")


def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results


def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    boxes = []
    results = predict(chosen_model, img, classes, conf=conf)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            boxes.append(box.xyxy[0][0])
            boxes.append(box.xyxy[0][1])
            boxes.append(box.xyxy[0][2])
            boxes.append(box.xyxy[0][3])

            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
    return img, results, boxes


def main():
    # Path to the folder containing images
    folder_path = "data/backpack"

    # Path to the folder where images will be saved with YOLO annotations
    yolo_folder = "data/yolo"

    # Create the YOLO folder if it doesn't exist
    os.makedirs(yolo_folder, exist_ok=True)

    # Get a list of image files in the folder
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    total_images = len(image_files)
    current_index = 0

    # Main loop
    while current_index < total_images:
        # Load the current image
        image_path = os.path.join(folder_path, image_files[current_index])
        image = cv2.imread(image_path)

        display_size = 400


        img_dup = image.copy()
        img_dup = cv2.resize(img_dup, (display_size, display_size))
        # Display the image
        # cv2.imshow("Image", image)
        result_img, _, boxes = predict_and_detect(model, img_dup, classes=[24], conf=0.05) # 24
        cv2.imshow("Image", result_img)

        desired_size = 96

        image = cv2.resize(image, (desired_size, desired_size))
        # Wait for key press
        key = cv2.waitKey(0)

        # Check if 's' key is pressed
        if key == ord('s'):
            # Save the image in YOLO folder
            cv2.imwrite(os.path.join(yolo_folder, image_files[current_index]), image)

            # Create a corresponding text file
            with open(os.path.join(yolo_folder, os.path.splitext(image_files[current_index])[0] + ".txt"), "w") as f:
                # Write some dummy content (you can adjust this according to your YOLO format)
                f.write(
                    f"1 {boxes[0] / display_size} {boxes[1] / display_size} {boxes[2] / display_size} {boxes[3] / display_size}")

        if key == ord('k'):
            # Save the image in YOLO folder
            cv2.imwrite(os.path.join(yolo_folder, image_files[current_index]), image)
            h, w, _ = image.shape
            # Create a corresponding text file
            with open(os.path.join(yolo_folder, os.path.splitext(image_files[current_index])[0] + ".txt"), "w") as f:
                # Write some dummy content (you can adjust this according to your YOLO format)
                f.write(f"0 0 0 {w / desired_size} {h / desired_size}")

        if key == ord('q'):
            cv2.destroyAllWindows()
            quit()

        # Move to the next image
        current_index += 1

    # Close OpenCV windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()