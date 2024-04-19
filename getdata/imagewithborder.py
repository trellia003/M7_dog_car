import os
import cv2
import numpy as np

def draw_rectangles_on_images(directory):
    # Traverse through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            # Load image
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Error: Unable to load image {filename}")
                continue

            # Load corresponding txt file
            txt_path = os.path.join(directory, os.path.splitext(filename)[0] + ".txt")
            if not os.path.exists(txt_path):
                print(f"Error: No corresponding txt file found for {filename}")
                continue

            # Get image dimensions
            img_height, img_width, _ = image.shape

            with open(txt_path, 'r') as txt_file:
                for line in txt_file:
                    try:
                        # Extract normalized coordinates from the txt file
                        values = line.split()
                        # Extract values and convert them to float
                        label = values[0]
                        x1_norm, y1_norm, width, height = map(float, values[1:])
                    except ValueError:
                        print(f"Error: Invalid format in txt file for {filename}: {line.strip()}")
                        continue

                    x1 = int((x1_norm-width/2)*img_width)
                    x2 = int((x1_norm+width/2)*img_height)
                    y1 = int((y1_norm-height/2)*img_width)
                    y2 = int((y1_norm+height/2)*img_height)



                    # Calculate pixel coordinates
                    # x1 = int(x1_norm * img_width)
                    # y1 = int(y1_norm * img_height)
                    # x2 = int(x2_norm * img_width)
                    # y2 = int(y2_norm * img_height)

                    # Draw rectangle on image
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display image with rectangle
            cv2.imshow("Image", image)
            cv2.waitKey(0)

    cv2.destroyAllWindows()

# Example usage
draw_rectangles_on_images("data/backpack")
