# feed_forward.py

import tensorflow as tf


class FeedForwardLayer(tf.keras.Model):

    def __init__(self, units, input_d, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.units = units

    def build(self, input_shape):
        self.weights = self.add_weights(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True
        )
        self.bias = self.add_weight(shape=(self.units,), initializer='random_normal', trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.weights) + self.bias


class FeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, d_model=512, d_ff=2048, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense_1 = tf.keras.layers.Dense(units=d_ff, activation='relu')
        self.relu_1 = tf.keras.layers.ReLU()
        self.dense_2 = tf.keras.layers.Dense(units=d_model)

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.relu_1(x)
        x = self.dense_2(x)
        return x


if __name__ == "__main__":
    tf.random.set_seed(42)

    sample_input = tf.random.uniform((64, 50, 512))  # Example: batch_size=64, sequence_length=50, d_model=512

    ffn = FeedForwardNetwork(d_model=512, d_ff=2048)

    output = ffn(sample_input)

    print("Output shape:", output.shape)
    print("Sample output:", output[0][0])