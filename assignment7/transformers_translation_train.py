#%%

"""
Credits: This code is adapted from the textbook "Deep Learning with Python", 
2nd Edition, by François Chollet. 
"""

#%%

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
import string
import re

from transformers_common import *
from tv_to_file import *

#%% We are using a fixed split of the dataset into training, test and 
#   validation, to make sure that we can later duplicate the exact same
#   text vectorization layers that we used at training.

train_pairs = load_pairs("model_backup/eng_spa_train.txt")
val_pairs = load_pairs("model_backup/eng_spa_val.txt")
test_pairs = load_pairs("model_backup/eng_spa_test.txt")

#%% Create text vectorization layers for English text and for Spanish text.

strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(
        lowercase, f"[{re.escape(strip_chars)}]", "")


vocab_size = 15000
sequence_length = 20
source_vectorization = layers.TextVectorization(max_tokens=vocab_size,
                                                output_mode="int",
                                                output_sequence_length=sequence_length,)

target_vectorization = layers.TextVectorization(max_tokens=vocab_size,
                                                output_mode="int",
                                                output_sequence_length=sequence_length + 1,
                                                standardize=custom_standardization,)
train_english_texts = [pair[0] for pair in train_pairs]
train_spanish_texts = [pair[1] for pair in train_pairs]
source_vectorization.adapt(train_english_texts)
target_vectorization.adapt(train_spanish_texts)

#%% Create Tensorflow datasets

batch_size = 64

def format_dataset(eng, spa):
    eng = source_vectorization(eng)
    spa = target_vectorization(spa)
    return ({"english": eng,
             "spanish": spa[:, :-1]}, 
            spa[:, 1:])


def make_dataset(pairs):
    eng_texts, spa_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset)
    return dataset.shuffle(2048).prefetch(16).cache()


train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

#%% Define the Encoder-Decoder Transformer model.

embed_dim = 256
dense_dim = 2048
num_heads = 8

encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="english")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
encoder_outputs = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="spanish")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
x = TransformerDecoder(embed_dim, dense_dim, num_heads)(x, encoder_outputs)
x = layers.Dropout(0.5)(x)

decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)
transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

transformer.compile(optimizer="rmsprop",
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"])

filename = "eng_spa_transformer_best.keras"
callbacks = [keras.callbacks.ModelCheckpoint(filename,
                                             save_best_only=True)]


transformer.summary()

#%% Train the model.

transformer.fit(train_ds, epochs=30, validation_data=val_ds)
transformer.save("eng_spa_transformer_final.keras")
