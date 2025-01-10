import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

def data_normalization(raw_data, train_start, train_end):
    # Calculate mean and std on training data
    train_data = raw_data[train_start:train_end]
    means = np.mean(train_data, axis=0)
    stds = np.std(train_data, axis=0)
    
    # Normalize all data using training statistics
    normalized_data = (raw_data - means) / stds
    return normalized_data

def make_inputs_and_targets(data, months, size, sampling):
    # Calculate dimensions
    observations_per_hour = 6  # Dataset has 1 observation per 10 minutes
    hours_in_two_weeks = 24 * 14
    steps = hours_in_two_weeks // (sampling // observations_per_hour)  # 336 for sampling=6
    
    # Initialize outputs
    inputs = np.zeros((size, steps, data.shape[1]))
    targets = np.zeros(size)
    
    # Generate random starting points
    max_start = len(data) - steps
    start_indices = np.random.randint(0, max_start, size=size)
    
    # Create input sequences and targets
    for i in range(size):
        start_idx = start_indices[i]
        inputs[i] = data[start_idx:start_idx + steps:1]
        mid_point = start_idx + steps//2
        targets[i] = months[mid_point]
    
    return inputs, targets

def build_and_train_dense(train_inputs, train_targets, val_inputs, val_targets, filename):
    input_shape = train_inputs.shape[1:]
    
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="tanh"),
        keras.layers.Dense(32, activation="tanh"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(12, activation="softmax")
    ])
    
    model.compile(optimizer="adam",
                 loss="sparse_categorical_crossentropy",
                 metrics=["accuracy"])
    
    # Save best model during training
    checkpoint = keras.callbacks.ModelCheckpoint(
        filename, monitor='val_accuracy', save_best_only=True, mode='max'
    )
    
    history = model.fit(
        train_inputs, train_targets,
        epochs=10,
        validation_data=(val_inputs, val_targets),
        callbacks=[checkpoint]
    )
    
    return history

def build_and_train_rnn(train_inputs, train_targets, val_inputs, val_targets, filename):
    input_shape = train_inputs.shape[1:]
    
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        keras.layers.Bidirectional(keras.layers.LSTM(32)),
        keras.layers.Dense(32, activation="tanh"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(12, activation="softmax")
    ])
    
    model.compile(optimizer="adam",
                 loss="sparse_categorical_crossentropy",
                 metrics=["accuracy"])
    
    # Save best model during training
    checkpoint = keras.callbacks.ModelCheckpoint(
        filename, monitor='val_accuracy', save_best_only=True, mode='max'
    )
    
    history = model.fit(
        train_inputs, train_targets,
        epochs=10,
        validation_data=(val_inputs, val_targets),
        callbacks=[checkpoint]
    )
    
    return history

def test_model(filename, test_inputs, test_targets):
    model = keras.models.load_model(filename)
    _, accuracy = model.evaluate(test_inputs, test_targets, verbose=0)
    return accuracy

def confusion_matrix(filename, test_inputs, test_targets):
    model = keras.models.load_model(filename)
    predictions = model.predict(test_inputs)
    pred_classes = np.argmax(predictions, axis=1)
    
    # Initialize 12x12 confusion matrix
    conf_matrix = np.zeros((12, 12), dtype=int)
    
    # Fill confusion matrix
    for true_class, pred_class in zip(test_targets, pred_classes):
        conf_matrix[int(true_class)][int(pred_class)] += 1
    
    return conf_matrix
