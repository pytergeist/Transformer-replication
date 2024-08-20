# layer_norm.py

import tensorflow as tf


class LayerNormalisation(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name="gamma",
            shape=(input_shape[-1:]),
            initializer="ones",
            trainable=True,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(input_shape[-1:]),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[-1], keepdims=True)
        numerator = inputs - mean
        denominator = tf.sqrt(variance + self.epsilon)
        normalised_inputs = numerator / denominator
        return self.gamma + normalised_inputs + self.beta


if __name__ == "__main__":
    tf.random.set_seed(42)

    sample_input = tf.random.uniform(
        (64, 50, 512)
    )  # Example: batch_size=64, sequence_length=50, d_model=512

    ffn = LayerNormalisation()

    output = ffn(sample_input)

    print("Output shape:", output.shape)
    print("Sample output:", output[0][0])
