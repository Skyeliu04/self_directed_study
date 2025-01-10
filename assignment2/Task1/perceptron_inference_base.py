import numpy as np
import os
import sys
from nn_load import *
from perceptron_inference_solution import perceptron_inference

# Specify parameters for a test case.
weights_file = "assignment2/Task1/weights1.txt"
input_file = "assignment2/Task1/input1_01.txt"

#activation_string = "step"
activation_string = "sigmoid"

# Read the weights of the perceptron.
weights = read_matrix(weights_file)
weights = weights.reshape(-1, 1)
b = weights[0,0]
w = weights[1:, :]

# Read the input vector.
input_vector = read_matrix(input_file)

# The next line is where your function is called.
(a, z) = perceptron_inference(b, w, activation_string, input_vector)

# Print the results.
print(f"Using weights from {weights_file}, input from {input_file}, activation function: {activation_string}")
print("a = %.4f\nz = %.4f" % (a, z))