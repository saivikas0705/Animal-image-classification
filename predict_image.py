import cv2
import numpy as np
from tensorflow.keras.models import load_model

def predict_external_image(image_path, model_path, class_names):
    # Load the trained model
    model = load_model(model_path)
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}.")
        return
    
    # Preprocess the image
    resized_image = cv2.resize(image, (64, 64))  # Resize to model input size
    normalized_image = resized_image / 255.0  # Normalize pixel values
    reshaped_image = np.expand_dims(normalized_image, axis=0)  # Add batch dimension
    
    # Predict the class
    predictions = model.predict(reshaped_image)
    predicted_class_index = np.argmax(predictions)  # Get the class index with the highest probability
    predicted_class_name = class_names[predicted_class_index]
    
    # Display the result
    print(f"Prediction: {predicted_class_name}")
    return predicted_class_name
