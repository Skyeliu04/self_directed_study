from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf

def train_model(model, cifar_tr_inputs, cifar_tr_labels, batch_size, epochs):
    # Compile the model with Sparse Categorical Crossentropy and Adam optimizer
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # Train the model
    model.fit(cifar_tr_inputs, cifar_tr_labels, 
              batch_size=batch_size, 
              epochs=epochs, 
              verbose=1)

def load_and_refine(filename, training_inputs, training_labels, batch_size, epochs):
    # Load the pre-trained model
    pre_trained_model = load_model(filename)
    
    # Reshape and pad MNIST images to match CIFAR dimensions
    padded_inputs = np.zeros((training_inputs.shape[0], 32, 32, 3))
    for i in range(len(training_inputs)):
        padded_inputs[i, 2:30, 2:30, 0] = training_inputs[i]
        padded_inputs[i, 2:30, 2:30, 1] = training_inputs[i]
        padded_inputs[i, 2:30, 2:30, 2] = training_inputs[i]
    
    # Preprocess inputs
    padded_inputs = padded_inputs.astype('float32')  # No need to divide by 255 as it's already normalized
    
    # Create a new model by cloning the architecture
    new_model = tf.keras.models.Sequential([
        layer for layer in pre_trained_model.layers[:-1]  # Copy all layers except the last one
    ])
    new_model.add(Dense(10, activation='softmax'))  # Add new output layer
    
    # Copy weights from pre-trained model
    for i in range(len(new_model.layers) - 1):
        new_model.layers[i].set_weights(pre_trained_model.layers[i].get_weights())
        new_model.layers[i].trainable = False  # Freeze the pre-trained layers
    
    # Compile the new model
    new_model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
    
    # Train the new model
    new_model.fit(padded_inputs, training_labels,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=1)
    
    return new_model

def evaluate_my_model(model, test_inputs, test_labels):
    # Reshape and pad MNIST test images to match CIFAR dimensions
    padded_test_inputs = np.zeros((test_inputs.shape[0], 32, 32, 3))
    for i in range(len(test_inputs)):
        padded_test_inputs[i, 2:30, 2:30, 0] = test_inputs[i]
        padded_test_inputs[i, 2:30, 2:30, 1] = test_inputs[i]
        padded_test_inputs[i, 2:30, 2:30, 2] = test_inputs[i]
    
    # Preprocess test inputs (normalization is already done in the base file)
    loss, accuracy = model.evaluate(padded_test_inputs, test_labels, verbose=1)
    
    return accuracy