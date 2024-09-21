# encoder.py

import tensorflow as tf

from src.layers.multi_head_attention import MultiHeadAttention
from src.layers.dropout import DropoutLayer
from src.blocks.feed_forward_block import FeedForwardNetwork
from src.layers.layer_norm import LayerNormalisation


class EncoderBlock(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1, *args, **kwargs):
        super(EncoderBlock, self).__init__(*args, **kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff)
        self.layer_norm1 = LayerNormalisation(epsilon=1e-8)
        self.layer_norm2 = LayerNormalisation(epsilon=1e-8)

    def call(self, inputs, mask=None):
        attention_output = self.attention(inputs)
        attention_output = DropoutLayer(self.dropout_rate)(attention_output)
        output_skip1 = self.layer_norm1(inputs + attention_output)
        ffn_output = self.feed_forward(output_skip1)
        ffn_output = DropoutLayer(self.dropout_rate)(ffn_output)
        return self.layer_norm2(output_skip1 + ffn_output)
