# feed_forward_network
import tensorflow as tf

from src.layers.feed_forward import FeedForwardLayer

class FeedForwardNetwork(tf.keras.Model): # TODO add feed forward layer?
    def __init__(self, d_model=512, d_ff=2048, *args, **kwargs):
        super(FeedForwardNetwork, self).__init__(*args, **kwargs)
        self.dense_1 = tf.keras.layers.Dense(units=d_ff, activation="relu")
        self.relu_1 = tf.keras.layers.ReLU()
        self.dense_2 = tf.keras.layers.Dense(units=d_model)

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.relu_1(x)
        x = self.dense_2(x)
        return x
