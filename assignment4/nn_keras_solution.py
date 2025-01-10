from tensorflow import keras
import numpy as np

def create_and_train_model(training_inputs, training_labels, layers, units_per_layer, epochs, hidden_activations):
    input_dim = training_inputs.shape[1]
    num_classes = len(np.unique(training_labels))
    
    model = keras.Sequential()
    
    # Input layer
    model.add(keras.layers.Input(shape=(input_dim,)))
    
    # Hidden layers
    for i in range(layers - 2):
        model.add(keras.layers.Dense(units_per_layer[i], activation=hidden_activations[i]))
    
    # Output layer - always use softmax for classification
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train the model
    model.fit(training_inputs, training_labels, epochs=epochs, verbose=0)
    
    return model

def test_model(model, test_inputs, test_labels, ints_to_labels):
    # Evaluate the model
    predictions = model.predict(test_inputs, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Calculate accuracy
    correct = 0
    total = len(test_labels)
    accuracies = []

    for i in range(total):
        predicted_class = ints_to_labels[predicted_classes[i]]
        actual_class = ints_to_labels[test_labels[i, 0]]
        
        # Check for ties
        max_prob = np.max(predictions[i])
        tied_classes = np.where(predictions[i] == max_prob)[0]
        
        if len(tied_classes) > 1:
            if test_labels[i, 0] in tied_classes:
                accuracy = 1 / len(tied_classes)
            else:
                accuracy = 0
        else:
            accuracy = 1 if predicted_classes[i] == test_labels[i, 0] else 0
        
        accuracies.append(accuracy)
        
        print('ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f' % 
              (i, predicted_class, actual_class, accuracy))
    
    # Overall classification accuracy
    test_accuracy = np.mean(accuracies)
    print('Classification accuracy on test set: %.2f%%' % (test_accuracy * 100))
    return test_accuracy
