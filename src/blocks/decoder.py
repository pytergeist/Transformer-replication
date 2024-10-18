# decoder.py

import tensorflow as tf
from src.layers.multi_head_attention import MultiHeadAttention
from src.layers.dropout import DropoutLayer
from src.blocks.feed_forward_block import FeedForwardNetwork
from src.layers.layer_norm import LayerNormalisation
from src.logging.log_config import logger
from src.logging.log_utils import log_tensor_shape


class DecoderBlock(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1, *args, **kwargs):
        super(DecoderBlock, self).__init__(*args, **kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.attention1 = MultiHeadAttention(d_model, num_heads)
        self.attention2 = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff)
        self.layer_norm1 = LayerNormalisation(epsilon=1e-8)
        self.layer_norm2 = LayerNormalisation(epsilon=1e-8)
        self.layer_norm3 = LayerNormalisation(epsilon=1e-8)

    def call(self, inputs, encoder_output, look_ahead_mask=None, padding_mask=None):
        try:

            logger.info("Starting MH1")
            attn1 = self.attention1(inputs, mask=look_ahead_mask)

            attn1 = DropoutLayer(self.dropout_rate)(attn1)
            out1 = self.layer_norm1(inputs + attn1)
            logger.info("Starting MH2")
            log_tensor_shape(padding_mask, "padding_mask for MH2")
            attn2 = self.attention2(
                out1,
                key_input=encoder_output,
                value_input=encoder_output,
                mask=padding_mask,
            )

            attn2 = DropoutLayer(self.dropout_rate)(attn2)
            out2 = self.layer_norm2(out1 + attn2)

            ffn_output = self.feed_forward(out2)

            ffn_output = DropoutLayer(self.dropout_rate)(ffn_output)
            output = self.layer_norm3(out2 + ffn_output)

            return output

        except Exception as e:
            logger.error(f"Error in DecoderBlock: {e}")
            raise


class DecoderStack(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers=6,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        dropout_rate=0.1,
        *args,
        **kwargs,
    ):
        super(DecoderStack, self).__init__(*args, **kwargs)
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.decoder_blocks = [
            DecoderBlock(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(self.num_layers)
        ]

    def call(self, inputs, encoder_output, look_ahead_mask=None, padding_mask=None):
        logger.info("EncoderStack called")
        x = inputs
        for i, block in enumerate(self.decoder_blocks):
            x = block(x, encoder_output, look_ahead_mask, padding_mask)
        return x


if __name__ == "__main__":
    tf.random.set_seed(42)

    batch_size = 64
    seq_length = 50
    d_model = 512
    num_heads = 8
    d_ff = 2048

    sample_input = tf.random.uniform((batch_size, seq_length, d_model))
    encoder_output = tf.random.uniform((batch_size, seq_length, d_model))

    sample_mask = None

    transformer_stack = DecoderStack(d_model=d_model, num_heads=num_heads, d_ff=d_ff)

    output = transformer_stack(
        sample_input,
        encoder_output,
        look_ahead_mask=sample_mask,
        padding_mask=sample_mask,
    )

    print(f"Output shape: {output.shape}")
    print(f"Sample output (first element of the first batch): {output[0][0]}")
