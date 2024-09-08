# embedding.py

import tensorflow as tf


class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model, **kwargs):
        super(EmbeddingLayer, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding_matrix = self.add_weight(
            shape=(vocab_size, d_model),
            initializer="random_normal",
            trainable=True,
            name="embedding_matrix",
        )

    def call(self, inputs):
        return tf.gather(self.embedding_matrix, inputs)


if __name__ == "__main__":
    vocab_size = 1000
    d_model = 64

    embedding_layer = EmbeddingLayer(vocab_size=vocab_size, d_model=d_model)

    input_sequence = tf.constant([[1, 23, 45, 67], [89, 123, 456, 789]])

    output = embedding_layer(input_sequence)

    print("Input Shape:", input_sequence.shape)
    print("Embedding Output Shape:", output.shape)
    print("Embedding Output:\n", output.numpy())
