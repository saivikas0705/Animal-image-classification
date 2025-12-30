import os
import matplotlib.pyplot as plt
from data_loader import load_images
from model import build_model
from train import train_model
from predict import predict_dataset_image, predict_external_image

# Load datasets
X_train, y_train, class_names = load_images('dataset/train')
X_test, y_test, _ = load_images('dataset/test')

# Build and train the model
model = build_model()
history = train_model(model, X_train, y_train)

# Plot the training history
def plot_training_history(history):
    # Extract accuracy and loss from history
    acc = history.history['accuracy']
    val_acc = history.history.get('val_accuracy')
    loss = history.history['loss']
    val_loss = history.history.get('val_loss')
    epochs = range(1, len(acc) + 1)

    # Plot training and validation accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
    if val_acc:
        plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    if val_loss:
        plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Call the plot function
plot_training_history(history)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Predict an image from the dataset
dataset_image_path = 'C:/Users/VIKAS.R/Desktop/cv_project/dataset/test/Bird/Bird_1.jpeg'
if os.path.exists(dataset_image_path):
    print(f"Dataset image exists: {dataset_image_path}")
    predicted_class, confidence = predict_dataset_image(dataset_image_path, model, class_names)
    if predicted_class:
        print(f"Dataset image predicted as: {predicted_class} with confidence {confidence:.2f}%.")
else:
    print(f"Dataset image does not exist: {dataset_image_path}")

# Predict an external image
external_image_path = 'C:/Users/VIKAS.R/Desktop/cv_project/external_image.jpeg'
if os.path.exists(external_image_path):
    print(f"External image exists: {external_image_path}")
    predicted_class, confidence = predict_external_image(external_image_path, model, class_names)
    if predicted_class:
        print(f"External image predicted as: {predicted_class} with confidence {confidence:.2f}%.")
else:
    print(f"External image does not exist: {external_image_path}")
