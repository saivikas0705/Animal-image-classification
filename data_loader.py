import os
import cv2
import numpy as np
def load_images(directory):
    images, labels = [], []
    class_names = os.listdir(directory)  # Get class names (folder names)
    for class_name in class_names:
        class_path = os.path.join(directory, class_name)
        label = class_names.index(class_name)  # Encode class name as a number
        print(f"Loading from: {class_path}")  # Debugging line to show the class path
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            print(f"Loading image: {img_path}")  # Debugging line to show the image path
            # Try to read the image
            img = cv2.imread(img_path)
            if img is not None:  # Check if the image was loaded successfully
                img = cv2.resize(img, (64, 64)) / 255.0  # Resize and normalize
                images.append(img)
                labels.append(label)
            else:
                print(f"Warning: Unable to load image at {img_path}. Skipping.")
    return np.array(images), np.array(labels), class_names
# Load train and test datasets
X_train, y_train, class_names = load_images('dataset/train')
X_test, y_test, _ = load_images('dataset/test')