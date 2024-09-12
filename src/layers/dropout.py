# dropout.py

import tensorflow as tf


class DropoutLayer(tf.keras.layers.Layer):  # TODO: add k.learning_phase
    def __init__(self, rate):
        super(DropoutLayer, self).__init__()
        self.rate = rate

    def call(self, inputs, training=None):
        if training:
            uniform_dist = tf.random.uniform(shape=tf.shape(inputs), maxval=1)
            dropout_mask = tf.cast(uniform_dist >= self.rate, dtype=tf.float32)
            return inputs * dropout_mask / (1 - self.rate)
        return inputs
