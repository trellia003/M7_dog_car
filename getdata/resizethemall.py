import cv2
import os


def resize_images_in_directory(input_dir, target_size=(240, 240)):
    # Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # Add more image formats if necessary
            # Read the image
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)

            # Resize the image
            resized_image = cv2.resize(image, target_size)

            # Overwrite the original image with the resized one
            cv2.imwrite(image_path, resized_image)

            print(f"Resized and saved: {image_path}")


# Example usage:
input_directory = "data/empty"
resize_images_in_directory(input_directory)
