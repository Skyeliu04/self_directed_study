import numpy as np

def nn_2l_train_and_test(tr_data, tr_labels, test_data, test_labels,
                         labels_to_ints, ints_to_labels, training_rounds):
    # Get dimensions
    num_features = tr_data.shape[1]
    num_classes = len(labels_to_ints)
    
    # Normalize data
    max_abs_val = np.max(np.abs(tr_data))
    tr_data_normalized = tr_data / max_abs_val
    test_data_normalized = test_data / max_abs_val
    
    # Initialize weights and biases for each output unit
    weights = np.random.uniform(-0.05, 0.05, (num_classes, num_features))
    bias = np.random.uniform(-0.05, 0.05, num_classes)
    
    # Training phase
    for round in range(training_rounds):
        # Update learning rate
        learning_rate = 0.98 ** round
        
        for i in range(len(tr_data_normalized)):
            # Forward pass
            x = tr_data_normalized[i]
            a = np.dot(weights, x) + bias  # weighted sums
            z = 1.0 / (1.0 + np.exp(-a))   # sigmoid activation
            
            # Create target vector (1 for correct class, 0 for others)
            target = np.zeros(num_classes)
            target[labels_to_ints[int(tr_labels[i,0])]] = 1
            
            # Compute deltas for output layer
            deltas = z * (1 - z) * (z - target)
            
            # Update weights and biases
            for j in range(num_classes):
                weights[j] -= learning_rate * deltas[j] * x
                bias[j] -= learning_rate * deltas[j]
    
    # Testing phase
    correct_predictions = 0
    
    for i in range(len(test_data_normalized)):
        # Forward pass
        x = test_data_normalized[i]
        a = np.dot(weights, x) + bias
        z = 1.0 / (1.0 + np.exp(-a))
        
        # Find classes that tied for highest output
        max_output = np.max(z)
        tied_indices = np.where(z == max_output)[0]
        tied_classes = [ints_to_labels[idx] for idx in tied_indices]
        
        # Calculate accuracy
        true_class = test_labels[i,0]
        if true_class in tied_classes:
            accuracy = 1.0 / len(tied_classes)
        else:
            accuracy = 0.0
        
        correct_predictions += accuracy
        
        # Print results for this test object
        predicted_str = str(tied_classes[0]) if len(tied_classes) == 1 else str(tied_classes)
        print('ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f' % 
              (i, predicted_str, str(true_class), accuracy))
    
    # Print overall classification accuracy
    classification_accuracy = correct_predictions / len(test_data_normalized)
    print('\nclassification accuracy=%6.4f' % (classification_accuracy))
