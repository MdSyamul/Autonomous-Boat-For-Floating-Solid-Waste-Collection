import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Load training history
with open('training_history.npy', 'rb') as f:
    history = np.load(f, allow_pickle=True).item()

# Visualize training accuracy and loss
def plot_training_history(history):
    # Plot Accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Load trained model
model = tf.keras.models.load_model('cnn_model.h5')

# Visualize example predictions
def show_predictions(model, data_dir, class_names):
    val_data = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        image_size=(128, 128),
        batch_size=1,
        label_mode='categorical'
    )
    
    for images, labels in val_data.take(5):
        predictions = model.predict(images)
        predicted_label = np.argmax(predictions[0])
        actual_label = np.argmax(labels[0])

        plt.imshow(images[0].numpy().astype('uint8'))
        plt.title(f"Predicted: {class_names[predicted_label]}, Actual: {class_names[actual_label]}")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    # Plot training history
    plot_training_history(history)
    
    # Show predictions
    data_dir = "data/val"
    class_names = ['battery','biological','brown-glass','cardboard','clothes','green-glass','metal','paper', 'plastic','shoes','trash','white-glass']
    show_predictions(model, data_dir, class_names)

if __name__ == "__main__":
    # Plot training history
    plot_training_history(history)
    
    # Show predictions
    data_dir = "data/val"
    show_predictions(model, data_dir)
