import numpy as np

def step_function(x):
    """Step activation function"""
    return 1.0 if x >= 0 else 0.0

def sigmoid_function(x):
    """Sigmoid activation function"""
    return 1.0 / (1.0 + np.exp(-x))

def apply_activation(x, activation):
    """Apply activation function element-wise to input"""
    if activation == "step":
        return np.array([[step_function(float(xi))] for xi in x])
    elif activation == "sigmoid":
        return np.array([[sigmoid_function(float(xi))] for xi in x])
    else:
        raise ValueError("Activation must be 'step' or 'sigmoid'")

def nn_inference(layers, units, biases, weights, activation, input_vector):
    """
    Compute neural network output for given parameters and input.
    
    Args:
        layers (int): Number of layers in the network
        units (list): Number of units in each layer
        biases (list): List of bias vectors for each layer
        weights (list): List of weight matrices for each layer
        activation (str): Either "step" or "sigmoid"
        input_vector (numpy.ndarray): Input vector (2D array with single column)
    
    Returns:
        tuple: (a_values, z_values) where:
            a_values (list): Pre-activation values for each layer
            z_values (list): Post-activation values for each layer
    """
    # Initialize lists to store a and z values
    a_values = [None, None]  # No layer 0, and no a values for input layer
    z_values = [None, input_vector]  # No layer 0, input_vector for layer 1
    
    # Forward propagation through each layer after input layer
    for layer in range(2, layers + 1):
        # Compute pre-activation values (a = Wx + b)
        a = np.dot(weights[layer], z_values[layer-1]) + biases[layer]
        a_values.append(a)
        
        # Apply activation function to get post-activation values
        z = apply_activation(a, activation)
        z_values.append(z)
    return (a_values, z_values)
