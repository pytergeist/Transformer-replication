# relu.py

import tensorflow as tf


class ReLUActivation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ReLUActivation, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.math.maximum(inputs, 0)


if __name__ == '__main__':
    inputs = tf.random.uniform(shape=[1, 10, 10], minval=-0.5, maxval=0.9)
    relu = ReLUActivation()
    outputs = relu(inputs)
    print(inputs)
    print(outputs)
