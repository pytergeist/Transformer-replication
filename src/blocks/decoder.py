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
            logger.info("DecoderBlock call")
            log_tensor_shape(inputs, "inputs")
            log_tensor_shape(encoder_output, "encoder_output")

            attn1 = self.attention1(inputs, mask=look_ahead_mask)
            logger.info("After first attention (attn1)")
            log_tensor_shape(attn1, "attn1")

            attn1 = DropoutLayer(self.dropout_rate)(attn1)
            out1 = self.layer_norm1(inputs + attn1)
            logger.info("After first LayerNorm")
            log_tensor_shape(out1, "out1")

            attn2 = self.attention2(out1, encoder_output, mask=padding_mask)
            logger.info("After second attention (attn2)")
            log_tensor_shape(attn2, "attn2")

            attn2 = DropoutLayer(self.dropout_rate)(attn2)
            out2 = self.layer_norm2(out1 + attn2)
            logger.info("After second LayerNorm")
            log_tensor_shape(out2, "out2")

            ffn_output = self.feed_forward(out2)
            logger.info("After feed forward network (ffn_output)")
            log_tensor_shape(ffn_output, "ffn_output")

            ffn_output = DropoutLayer(self.dropout_rate)(ffn_output)
            output = self.layer_norm3(out2 + ffn_output)
            logger.info("After third LayerNorm (final output)")
            log_tensor_shape(output, "output")

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
        logger.info("DecoderStack call")
        log_tensor_shape(inputs, "inputs")
        log_tensor_shape(encoder_output, "encoder_output")

        x = inputs
        for i, block in enumerate(self.decoder_blocks):
            logger.info(f"Passing through DecoderBlock {i + 1}")
            x = block(x, encoder_output, look_ahead_mask, padding_mask)
            log_tensor_shape(x, f"output from DecoderBlock {i + 1}")
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
