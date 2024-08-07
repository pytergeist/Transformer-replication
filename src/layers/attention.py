import tensorflow as tf


class Attention(tf.keras.layers.Layer):
    def __init__(self, d_embedding, d_k, num_heads, **kwargs):
        self.d_embedding = d_embedding
        self.d_k = d_k
        self.num_heads = num_heads
        self.keys = tf.keras.layers.Dense(self.d_embedding, use_bias=False)
        self.queries = tf.keras.layers.Dense(self.d_embedding, use_bias=False)
        super().__init__()

    def call(self, inputs):
        queries = self.queries(inputs)
        keys = self.keys(inputs)
        scores = tf.matmul(queries, keys, transpose_b=True)
        normalised_scores = tf.divide(scores, tf.math.sqrt(tf.cast(self.d_k, tf.float32)))
        attention_weights = tf.nn.softmax(normalised_scores, axis=-1)
        return tf.matmul(attention_weights, inputs)


if __name__ == "__main__":
    import numpy as np
    batch_size = 2
    seq_length = 3
    d_embedding = 4
    d_k = 4
    num_heads = 1

    np.random.seed(0)
    dummy_input = np.random.rand(batch_size, seq_length, d_embedding).astype(np.float32)

    attention_layer = Attention(d_embedding, d_k, num_heads)

    output = attention_layer(dummy_input)

    print("Input:")
    print(dummy_input)
    print("\nOutput:")
    print(output)
