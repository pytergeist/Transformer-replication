import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):  # TODO: implament masking
    def __init__(
        self, d_embedding: int, num_heads: int, *args: list, **kwargs: dict
    ) -> None:
        super().__init__(*args, **kwargs)
        self.d_embedding = d_embedding
        self.num_heads = num_heads
        self.d_k = self.d_embedding // self.num_heads

        self.keys = tf.keras.layers.Dense(self.d_embedding, use_bias=False)
        self.queries = tf.keras.layers.Dense(self.d_embedding, use_bias=False)
        self.values = tf.keras.layers.Dense(self.d_embedding, use_bias=False)

        self.output_dense = tf.keras.layers.Dense(self.d_embedding, use_bias=False)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.d_k))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        queries = self.queries(inputs)
        keys = self.keys(inputs)
        values = self.values(inputs)

        queries = self.split_heads(queries, batch_size)
        keys = self.split_heads(keys, batch_size)
        values = self.split_heads(values, batch_size)

        scores = tf.matmul(queries, keys, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(self.d_k, tf.float32))
        attention_weights = tf.nn.softmax(scores, axis=-1)

        output = tf.matmul(attention_weights, values)

        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_output = tf.reshape(output, (batch_size, -1, self.d_embedding))

        return self.output_dense(concat_output)


if __name__ == "__main__":
    import numpy as np

    batch_size = 2
    seq_length = 3
    d_embedding = 8  # Must be divisible by num_heads
    num_heads = 2

    np.random.seed(0)
    dummy_input = np.random.rand(batch_size, seq_length, d_embedding).astype(np.float32)

    attention_layer = MultiHeadAttention(
        d_embedding,
        num_heads,
    )

    output = attention_layer(dummy_input)

    print("Input:")
    print(dummy_input)
    print("\nOutput:")
    print(output)
