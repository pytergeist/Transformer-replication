# positional_encoding.py
import tensorflow as tf

from src.layers.embedding import EmbeddingLayer


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model, length, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.length = length
        self.embedding = EmbeddingLayer(vocab_size, d_model)
        self.positional_encoding = self.positional_encoder(length, d_model)

    def positional_encoder(self, length, depth):
        position = tf.range(length, dtype=tf.float32)[:, tf.newaxis]
        dimension = tf.range(depth, dtype=tf.float32)[tf.newaxis, :]

        frequency_scale = 10000 ** ((2 * dimension) / tf.cast(depth, tf.float32))
        angle = position / frequency_scale
        return tf.where(dimension % 2 == 0, tf.math.sin(angle), tf.math.cos(angle))

    def call(self, inputs):  # TODO: broadcast to allow for flexibility in tensor shape
        embedded_inputs = self.embedding(inputs)
        embedded_inputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        position_encoded_inputs = self.positional_encoder(self.length, self.d_model)
        embedded_inputs += position_encoded_inputs
        return embedded_inputs


if __name__ == "__main__":
    pos_encoder = PositionalEncoding(vocab_size=10, d_model=512, length=2048)
    test_input = tf.random.uniform((2, 2048), minval=0, maxval=10, dtype=tf.int32)

    output = pos_encoder(test_input)

    print("Output shape:", output.shape)
    print("Example output:", output[0])
