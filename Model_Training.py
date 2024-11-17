# model.py
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from Data_processing import generate_landmarks

def train_model(dataset_path):
    # Generate data and labels from images
    data, labels = generate_landmarks(dataset_path)

    # Convert string labels to integers (if needed)
    label_to_int = {label: index for index, label in enumerate(sorted(set(labels)))}
    labels = np.array([label_to_int[label] for label in labels])

    # Prepare data for training
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=29)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=29)

    # Define the model structure
    num_classes = 29  # Assuming 26 letters in ASL plus special characters
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(63,)),  # Adjust input_shape if necessary
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')  # Use the number of classes from the generator
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model using the data
    model.fit(
        X_train,
        y_train,
        epochs=20,
        validation_data=(X_test, y_test)
    )

    # Save the trained model
    model.save('asl_model.h5')

def load_model():
    return tf.keras.models.load_model('asl_model.h5')


if __name__ == "__main__":


    # Check if GPU is available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs found: {gpus}")
    else:
        print("No GPU found.")


    train_model("C:\\Users\\Brandon\\Downloads\\ASL dataset\\asl_alphabet_train")
    print("Complete")