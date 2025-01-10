import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
import random

def create_training_examples(filename, num_examples=1000, min_words=40):
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(filename, 'r', encoding=encoding) as f:
                text = f.read()
            break  # If successful, break the loop
        except UnicodeDecodeError:
            if encoding == encodings[-1]:  # If this was the last encoding to try
                raise  # Re-raise the exception if none of the encodings worked
            continue
    
    # Split into sentences (roughly)
    sentences = text.split('.')
    
    # Initialize storage for examples
    examples = []
    current_piece = ""
    word_count = 0
    
    # Create examples of sufficient length
    for sentence in sentences:
        if len(examples) >= num_examples:
            break
            
        # Clean the sentence
        sentence = sentence.strip()
        if not sentence:
            continue
            
        words = sentence.split()
        word_count += len(words)
        
        if current_piece:
            current_piece += ". " + sentence
        else:
            current_piece = sentence
            
        if word_count >= min_words:
            examples.append(current_piece)
            current_piece = ""
            word_count = 0
            
    # Ensure we have exactly num_examples
    examples = examples[:num_examples]
    return examples

def learn_model(train_files):
    # Parameters
    vocab_size = 20000
    sequence_length = 500
    num_examples_per_author = 1000
    
    # Create training examples for each author
    all_texts = []
    all_labels = []
    
    for author_idx, author_files in enumerate(train_files):
        author_texts = []
        for file in author_files:
            texts = create_training_examples(file, 
                                          num_examples=num_examples_per_author // len(author_files))
            author_texts.extend(texts)
        all_texts.extend(author_texts)
        all_labels.extend([author_idx] * len(author_texts))
    
    # Create text vectorization layer
    vectorize_layer = TextVectorization(
        max_tokens=vocab_size,
        output_mode='tf-idf',
        ngrams=2
    )
    
    # Adapt the layer to the training text
    text_ds = tf.data.Dataset.from_tensor_slices(all_texts)
    vectorize_layer.adapt(text_ds)
    
    # Create the model
    model = keras.Sequential([
        keras.Input(shape=(1,), dtype=tf.string),
        vectorize_layer,
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(3, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Convert inputs to numpy arrays
    x_train = np.array(all_texts, dtype=object)
    y_train = np.array(all_labels)
    
    # Create tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.batch(32)
    
    # Train the model
    model.fit(
        dataset,
        epochs=10,
        shuffle=True
    )
    
    return model 