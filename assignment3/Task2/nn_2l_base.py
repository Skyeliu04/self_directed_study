import numpy as np
from uci_load import read_uci_file
from nn_2l_solution import nn_2l_train_and_test


# Feel free to change the default directory to point to where you have
# stored the datasets.
directory = "assignment3/uci_datasets"

# Feel free to change the dataset name, so that you can experiment
# with different datasets.
#dataset = "pendigits"
#dataset = "satellite"
dataset = "yeast"

# Feel free to change the value in the next line, so that you can experiment
# with different parameters.
training_rounds = 10

training_file = directory + "/" + dataset + "_training.txt"
test_file = directory + "/" + dataset + "_test.txt"

labels_to_ints = {}
ints_to_labels = {}

# These lines read the dataset
(tr_data, tr_labels) = read_uci_file(training_file, labels_to_ints, ints_to_labels)
if tr_data is None or tr_labels is None:
    raise FileNotFoundError(f"Training data could not be loaded from {training_file}")

(test_data, test_labels) = read_uci_file(test_file, labels_to_ints, ints_to_labels)
if test_data is None or test_labels is None:
    raise FileNotFoundError(f"Test data could not be loaded from {test_file}")

# Populate labels_to_ints and ints_to_labels
unique_labels = np.unique(tr_labels)
for idx, label in enumerate(unique_labels):
    labels_to_ints[label] = idx
    ints_to_labels[idx] = label

# This is where your code is called.
nn_2l_train_and_test(tr_data, tr_labels, test_data, test_labels,
                     labels_to_ints, ints_to_labels, training_rounds)
