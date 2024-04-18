import cv2
import os

# Global variables for mouse event handling
mouseX, mouseY = -1, -1
drawing = False


# Function to draw rectangle
def draw_rectangle(event, x, y, flags, params):
    global mouseX, mouseY, drawing, img, img_copy, coordinates

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        mouseX, mouseY = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img = img_copy.copy()
            cv2.rectangle(img, (mouseX, mouseY), (x, y), (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (mouseX, mouseY), (x, y), (0, 255, 0), 2)
        coordinates = ((mouseX, mouseY), (x, y))


# Folder containing images
folder_path = 'data/empty'

# Get a list of image files in the folder
image_files = [f for f in os.listdir(folder_path) if
               os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    # Read image
    img = cv2.imread(os.path.join(folder_path, image_file))
    img_copy = img.copy()

    # Create a window
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_rectangle)

    coordinates = None

    while True:
        cv2.imshow('image', img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and coordinates is not None:
            # Save coordinates to a text file with the same name as the image
            with open(os.path.join(folder_path, os.path.splitext(image_file)[0] + '.txt'), 'w') as f:
                f.write(f"0 {coordinates[0][0]} {coordinates[0][1]} {coordinates[1][0]} {coordinates[1][1]}")
            break
        else:
            # Save default coordinates to a text file with the same name as the image
            with open(os.path.join(folder_path, os.path.splitext(image_file)[0] + '.txt'), 'w') as f:
                f.write("1 0 0 1 1")  # Default coordinates
            break

    cv2.destroyAllWindows()
