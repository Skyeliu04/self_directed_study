import numpy as np

def step_function(x):
    """Step activation function"""
    return 1.0 if x >= 0 else 0.0

def sigmoid_function(x):
    """Sigmoid activation function"""
    return 1.0 / (1.0 + np.exp(-x))

def perceptron_inference(b, w, activation, input_vector):
    """
    Compute perceptron output given weights, bias, and input.
    
    Args:
        b (float): Bias weight
        w (numpy.ndarray): Weight vector (2D array with single column)
        activation (str): Either "step" or "sigmoid"
        input_vector (numpy.ndarray): Input vector (2D array with single column)
    
    Returns:
        tuple: (a, z) where:
            a (float): dot product of weights and input plus bias
            z (float): result of applying activation function to a
    """
    # Step 1: Compute dot product and add bias
    a = float(np.dot(w.T, input_vector) + b)
    
    # Step 2: Apply activation function
    if activation == "step":
        z = step_function(a)
    elif activation == "sigmoid":
        z = sigmoid_function(a)
    else:
        raise ValueError("Activation must be 'step' or 'sigmoid'")
        
    return (a, z)
    
