# feed_forward.py

import tensorflow as tf


class FeedForwardLayer(tf.keras.layers.Layer):

    def __init__(self, units, *args, **kwargs):
        super(FeedForwardLayer, self).__init__(*args, **kwargs)
        self.units = units

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.bias = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel) + self.bias


if __name__ == "__main__":
    tf.random.set_seed(42)

    sample_input = tf.random.uniform((64, 50, 512))

    ffn = FeedForwardLayer(units=512)

    output = ffn(sample_input)

    print("Output shape:", output.shape)
    print("Sample output:", output[0][0])
