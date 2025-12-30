import cv2
import numpy as np
import os

def predict_dataset_image(image_path, model, class_names):
    """
    Predict the class of an image from the dataset.
    
    Args:
        image_path (str): Path to the image within the dataset.
        model: Trained model.
        class_names (list): List of class names.
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Unable to load image at {image_path}. Skipping.")
        return  # Exit if image is not loaded
    
    # Resize and preprocess the image
    img = cv2.resize(img, (64, 64)) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img)
    predicted_class = class_names[np.argmax(predictions)]
    confidence_score = predictions[0][np.argmax(predictions)] * 100
    print(f"Predicted class for {image_path}: {predicted_class} with confidence {confidence_score:.2f}%")
    return predicted_class, confidence_score

def predict_external_image(image_path, model, class_names):
    """
    Predict the class of an external image not in the dataset.
    
    Args:
        image_path (str): Path to the external image.
        model: Trained model.
        class_names (list): List of class names.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image at {image_path} not found.")
        return None

    # Load and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read the image at {image_path}.")
        return None
    
    img_resized = cv2.resize(img, (64, 64)) / 255.0  # Resize and normalize
    img_reshaped = np.expand_dims(img_resized, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_reshaped)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]
    confidence_score = predictions[0][predicted_class_index] * 100

    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence Score: {confidence_score:.2f}%")
    return predicted_class, confidence_score
