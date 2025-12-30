import matplotlib.pyplot as plt
def train_model(model, X_train, y_train, epochs=100, batch_size=32, validation_split=0.2):
    # Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1  # Displays progress during training
    )
    # Extract loss values
    train_loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    # Plot the loss curve
    plot_loss_curve(train_loss, val_loss)
    return history
def plot_loss_curve(train_loss, val_loss):
    epochs_range = range(1, len(train_loss) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, train_loss, label='Training Loss', color='blue', linewidth=2)
    plt.plot(epochs_range, val_loss, label='Validation Loss', color='red', linestyle='--', linewidth=2)
    plt.title('Training and Validation Loss', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()