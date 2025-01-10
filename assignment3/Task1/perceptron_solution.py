import numpy as np

def perceptron_train_and_test(tr_data, tr_labels, test_data, test_labels, training_rounds):
    """
    Trains and tests a perceptron for binary classification
    """
    # Normalize data using maximum absolute value from training data
    max_abs_val = np.max(np.abs(tr_data))
    tr_data_normalized = tr_data / max_abs_val
    test_data_normalized = test_data / max_abs_val
    
    # Initialize weights randomly between -0.05 and 0.05
    num_features = tr_data.shape[1]
    weights = np.random.uniform(-0.05, 0.05, num_features)
    bias = np.random.uniform(-0.05, 0.05)
    
    # Training phase
    for round in range(training_rounds):
        # Update learning rate for each round
        learning_rate = 0.98 ** (round) 
        
        # Process each training example
        for i in range(tr_data_normalized.shape[0]):
            # Forward pass
            x = tr_data_normalized[i]
            weighted_sum = np.dot(weights, x) + bias
            output = 1.0 / (1.0 + np.exp(-weighted_sum))  # sigmoid activation
            
            # Compute error and update weights
            error = tr_labels[i,0] - output
            weights += learning_rate * error * x
            bias += learning_rate * error
    
    # Testing phase
    correct_predictions = 0
    
    for i in range(test_data_normalized.shape[0]):
        # Forward pass
        x = test_data_normalized[i]
        weighted_sum = np.dot(weights, x) + bias
        output = 1.0 / (1.0 + np.exp(-weighted_sum))
        
        # Determine predicted class
        predicted_class = 1 if output >= 0.5 else 0
        true_class = test_labels[i,0]
        
        # Calculate accuracy for this example
        accuracy = 1.0 if predicted_class == true_class else 0.0
        correct_predictions += accuracy
        
        # Print results for this test object
        print('ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f' % 
              (i, str(predicted_class), str(true_class), accuracy))
    
    # Calculate and print overall classification accuracy
    classification_accuracy = correct_predictions / test_data_normalized.shape[0]
    print('\nclassification accuracy=%6.4f' % (classification_accuracy))