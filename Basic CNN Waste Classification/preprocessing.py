import tensorflow as tf
import os
import matplotlib.pyplot as plt

def load_and_preprocess_data(data_dir, image_size=(128, 128), batch_size=32):
    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir, 'train'),
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical'  
    )
    
    val_data = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir, 'val'),
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical'
    )
    
    return train_data, val_data

def visualize_data_distribution(data, class_names):
    labels = []
    for x, y in data:
        labels.extend(tf.argmax(y, axis=1).numpy())
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    label_counts = tf.math.bincount(labels)
    plt.bar(class_names, label_counts.numpy())
    plt.xlabel('Class Label')
    plt.ylabel('Count')
    plt.title('Data Distribution')
    plt.show()

# Data Augmentation Function
def augment_data(data):
    data = data.map(lambda x, y: (tf.image.random_flip_left_right(x), y))
    return data

if __name__ == "__main__":
    data_dir = "data"
    train_data, val_data = load_and_preprocess_data(data_dir)
    
    # Optionally augment the training data
    train_data = augment_data(train_data)
    
    # Define the class names
    class_names = ['battery','biological','brown-glass','cardboard','clothes','green-glass','metal','paper', 'plastic','shoes','trash','white-glass']


    # Visualize the data distribution
    visualize_data_distribution(train_data,class_names)
    
    # Save the datasets for the next stage
    tf.data.Dataset.save(train_data, 'train_data.tfds')
    tf.data.Dataset.save(val_data, 'val_data.tfds')
