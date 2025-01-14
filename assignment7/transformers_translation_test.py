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
import random
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

#%% Load the pre-trained transformer model.

#filename = "model_backup/eng_spa_transformer_best.keras"
filename = "model_backup/eng_spa_transformer_final1.keras"
transformer = keras.models.load_model(
    filename,
    custom_objects={"TransformerEncoder": TransformerEncoder,
                    "PositionalEmbedding": PositionalEmbedding,
                    "TransformerDecoder": TransformerDecoder,})

spa_vocab = target_vectorization.get_vocabulary()
spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))

test_eng_texts = [pair[0] for pair in test_pairs]
test_spa_texts = [pair[1] for pair in test_pairs]

#%% Use the model to translate a random sentence from the test set.

index = random.randint(0, len(test_eng_texts))
input_sentence = test_eng_texts[index]
target= test_spa_texts[index]
result = decode_sequence(transformer, input_sentence, 
                         source_vectorization, target_vectorization,
                         spa_index_lookup)

print("Input:  \"%s\"" % (input_sentence))
print("Result: \"%s\"" % (result))
print("Target: \"%s\"" % (target))

#%% Use the model to translate some input text that we specify.

input_text = "I did not like this movie at all"
#input_text = "if it rains I won't go shopping"
print(input_text)
print(decode_sequence(transformer, input_text, 
                      source_vectorization, target_vectorization,
                      spa_index_lookup))

