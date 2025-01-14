import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
import string
import re

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential([
            layers.Dense(dense_dim, activation="relu"),
            layers.Dense(embed_dim),])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            attention_mask = tf.cast(mask[:, :, tf.newaxis], dtype="int32")
            attention_mask = tf.matmul(attention_mask, padding_mask)
        else:
            attention_mask = None
            
        attention_output = self.attention(
            query=inputs, 
            key=inputs, 
            value=inputs,
            attention_mask=attention_mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential([
            layers.Dense(dense_dim, activation="relu"),
            layers.Dense(embed_dim),])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)
        else:
            padding_mask = causal_mask

        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, sequence_length, sequence_length))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1),
             tf.constant([1, 1], dtype=tf.int32)], axis=0)
        return tf.tile(mask, mult)

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return keras.ops.not_equal(inputs, 0)

def decode_sequence(model, input_sentence, source_vectorization, target_vectorization, target_index_lookup):
    tokenized_input_sentence = source_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(20):
        tokenized_target_sentence = target_vectorization([decoded_sentence])[:, :-1]
        predictions = model([tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = target_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break
    return decoded_sentence

def train_enc_dec(train_sentences, validation_sentences, epochs):
    # Create and configure vectorization layers
    sequence_length = 20
    vocab_size = 5000
    
    source_vectorization = TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=sequence_length)
    
    target_vectorization = TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=sequence_length + 1)  # +1 for the start token
    
    # Adapt to the data
    source_vectorization.adapt(train_sentences)
    target_vectorization.adapt([s[::-1] for s in train_sentences])  # Reversed sentences
    
    # Create the dataset
    batch_size = 64
    
    def format_dataset(source, target):
        source = source_vectorization(source)
        target = target_vectorization(target)
        return ({"encoder_inputs": source,
                "decoder_inputs": target[:, :-1]},
               target[:, 1:])
    
    def make_dataset(sentences):
        reversed_sentences = [s[::-1] for s in sentences]
        ds = tf.data.Dataset.from_tensor_slices((sentences, reversed_sentences))
        ds = ds.batch(batch_size)
        ds = ds.map(format_dataset)
        return ds.shuffle(2048).prefetch(16).cache()
    
    train_ds = make_dataset(train_sentences)
    val_ds = make_dataset(validation_sentences)
    
    # Create the model
    embed_dim = 256
    dense_dim = 2048
    num_heads = 8
    
    encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
    x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
    encoder_outputs = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
    
    decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
    x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
    x = TransformerDecoder(embed_dim, dense_dim, num_heads)(x, encoder_outputs)
    x = layers.Dropout(0.5)(x)
    decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)
    
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    # Compile and train
    model.compile(
        optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
    
    model.fit(train_ds, epochs=epochs, validation_data=val_ds)
    
    return model, source_vectorization, target_vectorization

def get_enc_dec_results(model, test_sentences, source_vec_layer, target_vec_layer):
    target_vocab = target_vec_layer.get_vocabulary()
    target_index_lookup = dict(zip(range(len(target_vocab)), target_vocab))
    results = []
    
    for sentence in test_sentences:
        result = decode_sequence(model, sentence, 
                               source_vec_layer, target_vec_layer,
                               target_index_lookup)
        # Clean up the result
        result = result.replace("[start]", "").replace("[end]", "").strip()
        results.append(result)
    
    return results

# Alias for compatibility with the base code
get_best_model_results = get_enc_dec_results
train_best_model = train_enc_dec

