import tensorflow as tf
import numpy as np
from keras.utils import plot_model

def create_cnn_model(input_shape=(128, 128, 3), num_classes=12):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')  
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',  
                  metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    # Load preprocessed datasets
    train_data = tf.data.Dataset.load('train_data.tfds')
    val_data = tf.data.Dataset.load('val_data.tfds')
    
    # Create the model
    model = create_cnn_model(num_classes=12)

    # Plot the model
    plot_model(model, to_file='cnn_model.png', show_shapes=True)

    
    # Train the model
    history = model.fit(train_data, validation_data=val_data, epochs=10)
    
    # Save the model for future predictions
    model.save('cnn_model.h5')
    
    # Save training history for visualization
    with open('training_history.npy', 'wb') as f:
        np.save(f, history.history)
