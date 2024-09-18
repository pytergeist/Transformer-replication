# feed_forward_network
import os

import tensorflow as tf

from src.layers.feed_forward import FeedForwardLayer
from src.activations.relu import ReLUActivation


class FeedForwardNetwork(tf.keras.Model):
    def __init__(self, d_model=512, d_ff=2048, *args, **kwargs):
        super(FeedForwardNetwork, self).__init__(*args, **kwargs)
        self.dense_1 = FeedForwardLayer(units=d_ff)
        self.relu_1 = ReLUActivation()
        self.dense_2 = FeedForwardLayer(units=d_model)

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.relu_1(x)
        x = self.dense_2(x)
        return x


if __name__ == "__main__":
    tf.random.set_seed(42)

    sample_input = tf.random.uniform((64, 50, 512))

    ffn = FeedForwardNetwork(d_model=512, d_ff=2048)

    output = ffn(sample_input)

    print("Output shape:", output.shape)
    print("Sample output:", output[0][0])
