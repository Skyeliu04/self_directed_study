import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def nn_train_and_test(tr_data, tr_labels, test_data, test_labels,
                      labels_to_ints, ints_to_labels, parameters):
    num_layers = parameters.num_layers
    units_per_layer = parameters.units_per_layer
    training_rounds = parameters.training_rounds

    print(f"Training and testing on the pendigits dataset, with {num_layers} layers, "
          f"{training_rounds} training rounds, {units_per_layer[0]} units for the first hidden layer, "
          f"{units_per_layer[1]} units for the second hidden layer.")
    # Normalize the data
    max_abs_value = np.max(np.abs(tr_data))
    tr_data = tr_data / max_abs_value
    test_data = test_data / max_abs_value

    # Initialize weights and biases with uniform distribution between -0.05 and 0.05
    weights = []
    biases = []

    # Input layer to first hidden layer
    weights.append(np.random.uniform(-0.05, 0.05, (tr_data.shape[1], units_per_layer[0])))
    biases.append(np.random.uniform(-0.05, 0.05, units_per_layer[0]))

    # Hidden layers
    for i in range(1, num_layers - 2):
        weights.append(np.random.uniform(-0.05, 0.05, (units_per_layer[i-1], units_per_layer[i])))
        biases.append(np.random.uniform(-0.05, 0.05, units_per_layer[i]))

    # Last hidden layer to output layer
    weights.append(np.random.uniform(-0.05, 0.05, (units_per_layer[-1], len(labels_to_ints))))
    biases.append(np.random.uniform(-0.05, 0.05, len(labels_to_ints)))

    # Training
    for round in range(training_rounds):
        learning_rate = 1 * (0.98 ** round)  # Adjust learning rate

        for x, t in zip(tr_data, tr_labels):
            # Forward pass
            activations = [x]
            for l in range(num_layers - 1):
                z = np.dot(activations[-1], weights[l]) + biases[l]
                a = sigmoid(z)
                activations.append(a)

            # Backward pass
            deltas = [activations[-1] - t]
            for l in range(num_layers - 2, 0, -1):
                delta = np.dot(deltas[0], weights[l].T) * sigmoid_derivative(activations[l])
                deltas.insert(0, delta)

            # Update weights and biases
            for l in range(num_layers - 1):
                weights[l] -= learning_rate * np.outer(activations[l], deltas[l])
                biases[l] -= learning_rate * deltas[l]

    # Evaluation
    correct_predictions = 0
    for x, t in zip(test_data, test_labels):
        # Forward pass
        activation = x
        for l in range(num_layers - 1):
            z = np.dot(activation, weights[l]) + biases[l]
            activation = sigmoid(z)

        # Determine predicted class
        predicted_label = np.argmax(activation)
        if predicted_label == t:
            correct_predictions += 1

    accuracy = correct_predictions / len(test_labels)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")