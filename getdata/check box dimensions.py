import cv2
import os


def check_max_txt_images_in_directory(input_dir, target_size = 240):
    # Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # Add more image formats if necessary
            # Read the image
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            height_img, width_img, channels_img = image.shape
            image_path_without_extension = os.path.splitext(image_path)[0]

            #this should have the image name
            with open(image_path_without_extension+".txt", 'r') as f:
                dimensions_str = f.readline().strip()

            # Split the string into individual values
            dimensions_list = dimensions_str.split()
            values_list = [float(value) for value in dimensions_list[1:5]]
            print(values_list)

            # if values_list[2]>target_size[0] or values_list[3]>target_size[1]:
            #     print(image_path_without_extension)


            # width = abs(values_list[2]-values_list[0])
            # height = abs(values_list[1]-values_list[3])
            #
            # mid_x = values_list[0] + width/2
            # mid_y = values_list[1] + height/2
            #
            #
            #
            # new_values_list = [mid_x,mid_y,width,height]
            #
            # new_values_list =[(values/target_size) for values in new_values_list]
            # print(new_values_list)

            # quit()


            #for creating a txt for empty images
            # with open(image_path_without_extension+".txt", 'w') as f:
            #     f.write(f"0 {new_values_list[0]} {new_values_list[1]} {new_values_list[2]} {new_values_list[3]}")  # Default coordinates
            #




# Example usage:
input_directory = "data/backpack"
check_max_txt_images_in_directory(input_directory)