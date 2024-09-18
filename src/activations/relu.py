# relu.py

import tensorflow as tf


class ReLUActivation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ReLUActivation, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.math.maximum(inputs, 0)
