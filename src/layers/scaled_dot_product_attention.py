import tensorflow as tf


class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self, d_embedding: int, d_k: int, *args: list, **kwargs: dict) -> None:
        super().__init__(*args, **kwargs)
        self.d_embedding = d_embedding
        self.d_k = d_k
        self.keys = tf.keras.layers.Dense(self.d_k, use_bias=False)
        self.queries = tf.keras.layers.Dense(self.d_k, use_bias=False)
        self.values = tf.keras.layers.Dense(self.d_k, use_bias=False)
        self.output_dense = tf.keras.layers.Dense(d_embedding, use_bias=False)

    def call(self, inputs):
        queries = self.queries(inputs)
        keys = self.keys(inputs)
        values = self.values(inputs)

        scores = tf.matmul(queries, keys, transpose_b=True)
        normalised_scores = tf.divide(
            scores, tf.math.sqrt(tf.cast(self.d_k, tf.float32))
        )

        attention_weights = tf.nn.softmax(normalised_scores, axis=-1)
        weighted_input = tf.matmul(attention_weights, values)

        return self.output_dense(weighted_input)


if __name__ == "__main__":
    import numpy as np

    batch_size = 2
    seq_length = 3
    d_embedding = 4
    d_k = 4

    np.random.seed(0)
    dummy_input = np.random.rand(batch_size, seq_length, d_embedding).astype(np.float32)

    attention_layer = ScaledDotProductAttention(
        d_embedding,
        d_k,
    )

    output = attention_layer(dummy_input)

    print("Input:")
    print(dummy_input)
    print("\nOutput:")
    print(output)
