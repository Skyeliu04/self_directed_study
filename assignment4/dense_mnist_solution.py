from tensorflow import keras

def load_mnist():
    # Load the MNIST dataset from Keras
    (training_inputs, training_labels), (test_inputs, test_labels) = keras.datasets.mnist.load_data()
    
    # Normalize the input values to be between 0 and 1
    training_inputs = training_inputs.astype('float32') / 255.0
    test_inputs = test_inputs.astype('float32') / 255.0
    
    # Flatten the input images to 2D arrays
    training_inputs = training_inputs.reshape((training_inputs.shape[0], -1))
    test_inputs = test_inputs.reshape((test_inputs.shape[0], -1))
    
    return training_inputs, training_labels, test_inputs, test_labels

def create_and_train_model(training_inputs, training_labels, layers, units_per_layer, epochs, hidden_activations):
    # Create a sequential model
    model = keras.Sequential()
    
    # Add the input layer
    model.add(keras.layers.InputLayer(shape=(training_inputs.shape[1],)))
    
    # Add hidden layers
    for i in range(layers - 2):
        model.add(keras.layers.Dense(units_per_layer[i], activation=hidden_activations[i]))
    
    # Add the output layer
    model.add(keras.layers.Dense(10, activation='softmax'))  # 10 classes for MNIST
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(training_inputs, training_labels, epochs=epochs, verbose=0)
    
    return model 