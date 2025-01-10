from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def load_mnist():
    # Load the MNIST dataset from Keras
    (training_inputs, training_labels), (test_inputs, test_labels) = keras.datasets.mnist.load_data()
    
    # Normalize the input values to be between 0 and 1
    training_inputs = training_inputs.astype('float32') / 255.0
    test_inputs = test_inputs.astype('float32') / 255.0
    
    # Reshape the inputs to add a channel dimension
    training_inputs = training_inputs.reshape((training_inputs.shape[0], 28, 28, 1))
    test_inputs = test_inputs.reshape((test_inputs.shape[0], 28, 28, 1))
    
    # Reshape labels to be a column vector
    training_labels = training_labels.reshape(-1, 1)
    test_labels = test_labels.reshape(-1, 1)
    
    return training_inputs, training_labels, test_inputs, test_labels

def create_and_train_model(training_inputs, training_labels, blocks, filter_size, filter_number, region_size, epochs, cnn_activation):
    # Create a sequential model
    model = keras.Sequential()
    model.add(layers.Input(shape=(28, 28, 1)))
    
    # Add convolutional and max pooling layers
    for _ in range(blocks):
        model.add(layers.Conv2D(filter_number, (filter_size, filter_size), activation=cnn_activation))
        model.add(layers.MaxPooling2D(pool_size=(region_size, region_size)))
    
    # Flatten the output from the convolutional layers
    model.add(layers.Flatten())
    
    # Add the output layer
    model.add(layers.Dense(10, activation='softmax'))  # 10 classes for MNIST
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(training_inputs, training_labels, epochs=epochs, verbose=0)
    
    return model 