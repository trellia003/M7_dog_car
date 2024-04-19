import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, average_precision_score

from tensorflow.keras import Input, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import keras

keras.utils.set_random_seed(42)


def files_counter(data_path: str) -> None:
    image_counter = 0
    textfile_counter = 0

    # Loop through all files
    for files in os.listdir(data_path):
        if files.endswith('.jpg'):
            # Increment the image counter if it's an image file
            image_counter += 1
        if files.endswith('.txt'):
            # Increment the text file counter if it's a text file
            textfile_counter += 1

    # Print the total number of images and text files found
    print('Number of images:', image_counter)
    print('Number of annotated files:', textfile_counter)


def find_paired_files(data_path: str) -> list[str]:
    files_stripped = []

    for file in os.listdir(data_path):
        if file.endswith('.txt'):
            # Get the filename without the extension
            stripped_name = os.path.splitext(file)[0]

            # Check if the paired image exist
            if os.path.join(data_path, stripped_name + '.jpg'):
                files_stripped.append(stripped_name)

    return files_stripped


def format_image(input_size: int, image: np.ndarray, bounding_box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Extracting dimensions of the input image
    image_height, image_width, image_depth = image.shape

    # Determining the maximum dimension of the image
    # max_size = max(image_height, image_width)
    #
    # # Calculating the scale factor for resizing
    # scale = max_size / input_size
    #
    # # Calculating the new dimensions after resizing
    # new_width, new_height = image_width / scale, image_height / scale
    #
    # new_width_int, new_height_int = int(new_width), int(new_height)
    # new_size = (new_width_int, new_height_int)
    #
    # # Creating an empty canvas with the specified input size
    # new_image = np.zeros((input_size, input_size, image_depth), dtype=float)
    #
    # # Resizing the original image to fit the new dimensions
    # resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    #
    # # Filling the empty canvas with the resized image
    # new_image[0:new_height_int, 0:new_width_int, 0:image_depth] = resized_image

    new_image = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    new_width, new_height = input_size, input_size

    # Normalise pixel values to the range [0, 1]
    new_image = new_image.astype(np.float32) / 255.0

    x_mid, y_mid, box_width, box_height = bounding_box

    # Calculating new bounding box coordinates and dimensions based on the resized image
    new_box = [
        (x_mid - box_width / 2) * new_width,
        (y_mid - box_height / 2) * new_height,
        (x_mid + box_width / 2) * new_width,
        (y_mid + box_height / 2) * new_height
    ]

    new_box = np.asarray(new_box, dtype=np.float32)

    return new_image, new_box


def load_data(data_path: str, input_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    images = []
    labels = []
    bboxes = []

    files = find_paired_files(data_path)
    for file in files:
        # Read the image
        image = cv2.imread(os.path.join(data_path, file + '.jpg'))

        # Open the corresponding annotation file and extract class label and bounding box information
        with open(os.path.join(data_path, file + '.txt'), 'r') as fp:
            entries = fp.readlines()[0].split()
            # Split the class label from the bounding box coordinates
            label = float(entries[0])
            bounding_box = np.array(entries[1:], dtype=float)

        # Resize the image and bounding box to fit the input size
        image, bounding_box = format_image(input_size, image, bounding_box)

        # Append the resized image, label, and bounding box
        images.append(image)
        labels.append(label)
        bboxes.append(bounding_box)

    images = np.array(images)
    labels = np.array(labels)
    bboxes = np.array(bboxes)

    return images, labels, bboxes


def display_image_with_box(image: np.ndarray, bounding_box: np.ndarray,
                           pred_box: np.ndarray = None) -> None:
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Extract bounding box coordinates
    x, y, x2, y2 = bounding_box

    # Calculating the scale factor for resizing

    # Draw bounding box on the image
    cv2.rectangle(image, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 1)  # Green rectangle

    if pred_box is not None:
        # Extract predicted box coordinates
        x_pred, y_pred, w_pred, h_pred = pred_box

        # Calculate the coordinates of the top-left and bottom-right corners of the predicted box
        x2_pred = x_pred + w_pred
        y2_pred = y_pred + h_pred

        # Draw predicted box on the image
        cv2.rectangle(image, (int(x_pred), int(y_pred)), (int(x2_pred), int(y2_pred)), (255, 0, 0),
                      1)  # Blue rectangle for predicted box

    # Display the image with the bounding boxes
    window_name = 'Image with Bounding Box'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 500, 500)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def split_dataset(images: np.ndarray, labels: np.ndarray, bboxes: np.ndarray) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray], tuple[
        np.ndarray, np.ndarray, np.ndarray]]:
    # Splitting the dataset into training and testing sets
    X_train_images, X_test_images, y_train_class, y_test_class, y_train_bbox, y_test_bbox = train_test_split(
        images, labels, bboxes, test_size=0.2, random_state=42, shuffle=True)

    # Further splitting the training set into training and validation sets
    X_train_images, X_val_images, y_train_class, y_val_class, y_train_bbox, y_val_bbox = train_test_split(
        X_train_images, y_train_class, y_train_bbox, test_size=0.25, random_state=42, shuffle=True)

    train_data = (X_train_images, y_train_class, y_train_bbox)
    val_data = (X_val_images, y_val_class, y_val_bbox)
    test_data = (X_test_images, y_test_class, y_test_bbox)

    return train_data, val_data, test_data


def augment_dataset(dataset: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    augmented_images = []
    augmented_labels = []
    augmented_bboxes = []

    images, labels, bboxes = dataset

    # Loop through each image in the dataset
    for i, image in enumerate(images):
        # Add original image and its corresponding label and bounding box
        augmented_images.append(image)
        augmented_labels.append(labels[i])
        augmented_bboxes.append(bboxes[i])

    # Perform data augmentation by iterating through each image in the dataset
    for i in range(len(images)):
        image = images[i]
        bbox = bboxes[i]

        x, y, x2, y2 = bbox
        height, width, _ = image.shape

        flip_horizontal = np.random.default_rng().choice([True, False])
        flip_vertical = np.random.default_rng().choice([True, False])

        if flip_horizontal:
            image = cv2.flip(image, 1)
            bbox = [width - x2, y, width - x, y2]

        if flip_vertical:
            image = cv2.flip(image, 0)
            if not flip_horizontal:
                bbox = [x, height - y2, x2, height - y]
            else:
                bbox = [width - x2, height - y2, width - x, height - y]

        new_x, new_y, new_x2, new_y2 = bbox

        if not flip_horizontal and not flip_vertical:
            rotation_angle = np.random.default_rng().choice([90, 180, 270])
        elif flip_horizontal and flip_vertical:
            rotation_angle = np.random.default_rng().choice([0, 90, 180])
        else:
            rotation_angle = np.random.default_rng().choice([0, 90, 180, 270])

        if rotation_angle == 90:
            image = np.rot90(image, 1)
            bbox = [new_y, width - new_x2, new_y2, width - new_x]
        elif rotation_angle == 180:
            image = np.rot90(image, 2)
            bbox = [width - new_x2, height - new_y2, width - new_x, height - new_y]
        elif rotation_angle == 270:
            image = np.rot90(image, 3)
            bbox = [height - new_y2, new_x, width - new_y, new_x2]

        # print("flip_horizontal", flip_horizontal)
        # print("flip_vertical", flip_vertical)
        # print("rotation angle", rotation_angle)
        # display_image_with_box(image, bbox)

        augmented_images.append(image)
        augmented_labels.append(labels[i])
        augmented_bboxes.append(bbox)

    augmented_dataset = (np.array(augmented_images), np.array(augmented_labels), np.array(augmented_bboxes))

    return augmented_dataset


def build_model(input_shape: tuple[int, int, int]):
    input_layer = Input(shape=input_shape)

    base_layers = Conv2D(32, (3, 3), activation='relu')(input_layer)
    base_layers = MaxPooling2D((2, 2))(base_layers)

    base_layers = Conv2D(64, (3, 3), activation='relu')(base_layers)
    base_layers = MaxPooling2D((2, 2))(base_layers)

    base_layers = Conv2D(128, (3, 3), activation='relu')(base_layers)
    base_layers = MaxPooling2D((2, 2))(base_layers)

    base_layers = Conv2D(256, (3, 3), activation='relu')(base_layers)
    base_layers = MaxPooling2D((2, 2))(base_layers)
    base_layers = Dropout(0.25)(base_layers)

    # Flatten layer to convert the 2D feature maps to a 1D feature vector
    base_layers = Flatten()(base_layers)

    # create the classifier branch
    classifier_branch = Dense(64, kernel_regularizer=l2(0.001), activation='relu')(base_layers)
    classifier_branch = BatchNormalization()(classifier_branch)
    classifier_branch = Dense(32, kernel_regularizer=l2(0.001), activation='relu')(classifier_branch)
    classifier_branch = BatchNormalization()(classifier_branch)
    classifier_head = Dense(2, activation='softmax', name='cl_head')(classifier_branch)

    # Create the regression branch
    regressor_branch = Dense(128, kernel_regularizer=l2(0.001), activation='relu')(base_layers)
    regressor_branch = Dropout(0.1)(regressor_branch)
    regressor_branch = BatchNormalization()(regressor_branch)
    regressor_branch = Dense(64, kernel_regularizer=l2(0.001), activation='relu')(regressor_branch)
    regressor_branch = Dense(32, kernel_regularizer=l2(0.001), activation='relu')(regressor_branch)
    regressor_branch = BatchNormalization()(regressor_branch)
    regressor_head = Dense(4, name='rg_head')(regressor_branch)

    # Create the final model
    model = Model(inputs=input_layer, outputs=[classifier_head, regressor_head])

    return model


def compile_model(model: Model) -> Model:
    # Define Adam optimizer
    optimizer = Adam(learning_rate=1e-2)
    # Define the loss functions for each output head
    losses = {
        'cl_head': 'sparse_categorical_crossentropy',
        'rg_head': 'mse'
    }
    # Define the evaluation metrics for each output head
    metrics = {
        'cl_head': 'accuracy',
        'rg_head': 'mse'
    }

    model.compile(optimizer=optimizer, loss=losses, metrics=metrics)

    return model


def train_model(model: Model, train_data: tuple[np.ndarray, np.ndarray, np.ndarray],
                val_data: tuple[np.ndarray, np.ndarray, np.ndarray], epochs: int, batch_size: int) -> Model:
    train_targets = {
        'cl_head': train_data[1],
        'rg_head': train_data[2]
    }

    val_targets = {
        'cl_head': val_data[1],
        'rg_head': val_data[2]
    }

    # Add learning rate scheduler, so model converges more effectively
    reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=0)

    # Add early stopping to prevent overfitting. The model will stop training once it stop increasing its performance.
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, start_from_epoch=20, restore_best_weights=True,
                                   verbose=1)

    # Train the model
    history = model.fit(train_data[0], train_targets,
                        validation_data=(val_data[0], val_targets),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[reduce_learning_rate, early_stopping],
                        verbose=2)

    return history


def intersection_over_union(box_a, box_b):
    x_box_a, y_box_a, x2_box_a, y2_box_a = box_a[0], box_a[1], box_a[0] + box_a[2], box_a[1] + box_a[3]
    x_box_b, y_box_b, x2_box_b, y2_box_b = box_b[0], box_b[1], box_b[0] + box_b[2], box_b[1] + box_b[3]

    # Calculate areas of bounding boxes
    box_a_area = (x2_box_a - x_box_a) * (y2_box_a - y_box_a)
    box_b_area = (x2_box_b - x_box_b) * (y2_box_b - y_box_b)

    # Calculate coordinates of intersection box
    x_interbox = max(x_box_a, x_box_b)
    y_interbox = max(y_box_a, y_box_b)
    x_2_interbox = min(x2_box_a, x2_box_b)
    y_2_interbox = min(y2_box_a, y2_box_b)

    # Calculate area of intersection
    inter_area = (x_2_interbox - x_interbox) * (y_2_interbox - y_interbox)

    # Calculate Intersection over Union
    iou = inter_area / (box_a_area + box_b_area - inter_area)

    return iou


def evaluate_model(model: Model, test_data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    test_images, test_labels, test_boxes = test_data
    test_targets = {
        'cl_head': test_labels,
        'rg_head': test_boxes
    }

    # Evaluate the model on the test data
    loss, accuracy, mse = model.evaluate(test_images, test_targets, verbose=0)
    pred_labels, pred_boxes = model.predict(test_images, verbose=0)
    pred_labels = np.argmax(pred_labels, axis=1)

    # If no backpacks are detected, the whole image should be the bounding box
    for i in range(len(pred_labels)):
        if pred_labels[i] == 1:
            pred_boxes[i] = [0, 0, test_images[i].shape[1], test_images[1].shape[0]]
    # Calculate F1 score
    f1 = f1_score(test_labels.astype(int), pred_labels, average='weighted')

    # Calculate average precision score
    ap = average_precision_score(test_labels.astype(int), pred_labels)

    # Calculate mean intersection over union
    mean_iou = 0
    denominator = 0
    for i in range(0, test_boxes.shape[0]):
        if pred_labels[i] == 0:
            denominator += 1
            mean_iou += intersection_over_union(test_boxes[i], pred_boxes[i])
    mean_iou /= denominator

    # Print evaluation metrics
    print('--------------------------')
    print('Model Evaluation Results:')
    print('{:<45} {:<10}'.format('Metric', 'Score'))
    print('-' * 53)
    print('{:<45} {:.2f}'.format('Test Loss:', loss))
    print('{:<45} {:.2f}'.format('Test Mean Squared Error Regression:', mse))
    print('{:<45} {:.4f}'.format('Test Intersection Over Union Regression:', mean_iou))
    print('{:<45} {:.4f}'.format('Test Accuracy Classification:', accuracy))
    print('{:<45} {:.4f}'.format('Test F1 Score Classification:', f1))
    print('{:<45} {:.4f}'.format('Test Average Precision Score Classification:', ap))
    print('-' * 53)

    # for i in range(len(test_images)):
    for i in range(10):
        print('{:<12} {:<10}'.format('Image nr.:', i),
              '\n{:<12} {:<10}'.format('Test Label:', int(test_labels[i])),
              '{:<12} {}'.format('Predicted Label:', pred_labels[i]))
        print('')
        display_image_with_box(test_images[i], test_boxes[i], pred_boxes[i])


def analyse_model(history: Model) -> None:
    cl_accuracy = history.history['cl_head_accuracy']
    val_cl_accuracy = history.history['val_cl_head_accuracy']

    rg_mse = history.history['rg_head_mse']
    val_rg_mse = history.history['val_rg_head_mse']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(cl_accuracy) + 1)

    # Plot the training accuracy and validation accuracy over the epochs
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, cl_accuracy, label='Training Accuracy', color='blue')
    plt.plot(epochs, val_cl_accuracy, label='Validation Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.45, 1])
    plt.legend(loc='lower right')
    plt.title('Classification - Accuracy')

    plt.subplot(1, 3, 2)
    plt.plot(epochs, rg_mse, label='Training MSE', color='blue')
    plt.plot(epochs, val_rg_mse, label='Validation MSE', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.ylim([0, 1000])
    plt.legend(loc='upper right')
    plt.title('Regression - Mean Squared Error')

    plt.subplot(1, 3, 3)
    plt.plot(epochs, loss, label='Training Loss', color='blue')
    plt.plot(epochs, val_loss, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0, 1000])
    plt.title('Total Loss')
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()


def save_model(model: Model, filepath: str) -> None:
    model.save(filepath)
    print(f'Model saved to: {filepath}')


def run_model() -> None:
    # Define dataset parameters
    path = 'getdata/data/combined_dataset'
    input_shape = (240, 240, 3)
    epochs = 128
    batch_size = 32

    # Count number of files
    files_counter(path)

    # Load and preprocess dataset
    images, labels, bboxes = load_data(path, input_shape[0])
    train_data, val_data, test_data = split_dataset(images, labels, bboxes)

    augmented_train_data = augment_dataset(train_data)

    print(f'Original Train images: {len(train_data[0])}, Augmented Train images: {len(augmented_train_data[0])}\n'
          f'Original Val images: {len(val_data[0])}\n'
          f'Original Test images: {len(test_data[0])}')

    # Print the shapes of train, validation, and test data
    # print("Train data shapes:")
    # print(train_data[0].shape, train_data[1].shape, train_data[2].shape)
    # print("Validation data shapes:")
    # print(val_data[0].shape, val_data[1].shape, val_data[2].shape)
    # print("Test data shapes:")
    # print(test_data[0].shape, test_data[1].shape, test_data[2].shape)

    # Create model
    model = build_model(input_shape)

    # Compile the model
    model = compile_model(model)

    # Train the model
    history = train_model(model, augmented_train_data, val_data, epochs, batch_size)

    # Evaluate the model
    evaluate_model(model, test_data)

    # Analyse the model
    analyse_model(history)

    # Save the model
    save_data_path = 'models/combined_240.keras'
    save_model(model, save_data_path)


if __name__ == '__main__':
    run_model()
