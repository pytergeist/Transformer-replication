import tensorflow as tf

from src.logging.log_config import logger
from src.logging.log_utils import log_tensor_shape


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(
        self, d_embedding: int, num_heads: int, *args: list, **kwargs: dict
    ) -> None:
        super().__init__(*args, **kwargs)
        assert (
            d_embedding % num_heads == 0
        ), "d_embedding must be divisible by num_heads"
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

    def apply_attention_mask(self, attention_scores, mask=None):
        if mask is not None:
            # mask = tf.cast(mask[:, tf.newaxis, :, :], tf.float32)
            log_tensor_shape(mask, "mask")
            attention_scores += mask * -1e9
        return attention_scores

    def call(self, query_input, key_input=None, value_input=None, mask=None):
        logger.debug("MultiHeadAttention call")
        batch_size = tf.shape(query_input)[0]
        logger.info(f"{batch_size}")

        if key_input is None:  # TODO: make more elegant
            key_input = query_input
        if value_input is None:
            value_input = query_input

        queries, keys, values = (
            self.queries(query_input),
            self.keys(key_input),
            self.values(value_input),
        )

        log_tensor_shape(queries, "queries")
        log_tensor_shape(keys, "keys")
        log_tensor_shape(values, "values")

        queries = self.split_heads(queries, batch_size)
        keys = self.split_heads(keys, batch_size)
        values = self.split_heads(values, batch_size)

        log_tensor_shape(queries, "split queries")
        log_tensor_shape(keys, "split keys")
        log_tensor_shape(values, "split values")

        scores = tf.matmul(queries, keys, transpose_b=True)
        log_tensor_shape(scores, "queries keys matmul")
        scores = scores / tf.math.sqrt(tf.cast(self.d_k, tf.float32))
        log_tensor_shape(scores, "scaled scores")
        scores = self.apply_attention_mask(scores, mask)
        log_tensor_shape(scores, "masked scores")

        attention_weights = tf.nn.softmax(scores, axis=-1)
        log_tensor_shape(attention_weights, "attention_weights")
        log_tensor_shape(values, "values")

        output = tf.matmul(attention_weights, values)

        log_tensor_shape(output, "output")

        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_output = tf.reshape(output, (batch_size, -1, self.d_embedding))

        return self.output_dense(concat_output)


if __name__ == "__main__":
    import numpy as np

    batch_size = 2
    seq_length = 3
    d_embedding = 8
    num_heads = 2

    np.random.seed(0)
    dummy_input = np.random.rand(batch_size, seq_length, d_embedding).astype(np.float32)

    mask = np.array([[1, 1, 0], [1, 1, 1]])

    attention_layer = MultiHeadAttention(
        d_embedding,
        num_heads,
    )

    output = attention_layer(dummy_input, mask=mask)

    print("Input:")
    print(dummy_input)
    print("\nMask:")
    print(mask)
    print("\nOutput:")
    print(output)
