import tensorflow as tf
from src.blocks.encoder import EncoderStack
from src.blocks.decoder import DecoderStack
from src.layers.feed_forward import FeedForwardLayer
from src.layers.positional_encoding import PositionalEncoding
from src.layers.dropout import DropoutLayer
from src.logging.log_config import logger
from src.logging.log_utils import log_tensor_shape


class Transformer(tf.keras.Model):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        d_ff,
        input_vocab_size,
        target_vocab_size,
        max_seq_len_input,
        max_seq_len_target,
        dropout_rate=0.1,
        *args,
        **kwargs,
    ):
        super(Transformer, self).__init__(*args, **kwargs)

        self.num_heads = num_heads  # Initialize num_heads
        self.encoder = EncoderStack(num_layers, d_model, num_heads, d_ff, dropout_rate)
        self.decoder = DecoderStack(num_layers, d_model, num_heads, d_ff, dropout_rate)
        self.final_layer = FeedForwardLayer(target_vocab_size)
        self.positional_encoding_inputs = PositionalEncoding(
            input_vocab_size, d_model, max_seq_len_input
        )
        self.positional_encoding_targets = PositionalEncoding(
            target_vocab_size, d_model, max_seq_len_target
        )
        self.dropout = DropoutLayer(dropout_rate)

    def create_padding_mask(self, seq):
        return tf.cast(tf.math.equal(seq, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)

    def call(
        self,
        inputs,
        targets,
        enc_padding_mask=None,
        look_ahead_mask=None,
        dec_padding_mask=None,
    ):
        logger.info("Starting Transformer call")

        # Log input shapes
        log_tensor_shape(inputs, "inputs")
        log_tensor_shape(targets, "targets")

        inputs = self.positional_encoding_inputs(inputs)
        targets = self.positional_encoding_targets(targets)

        logger.info("After positional encoding")

        inputs = self.dropout(inputs)
        targets = self.dropout(targets)

        logger.info("After dropout")

        enc_padding_mask = self.create_padding_mask(inputs)
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(targets)[1])
        look_ahead_mask = look_ahead_mask[
            tf.newaxis, tf.newaxis, :, :
        ]  # (1, 1, seq_len, seq_len)
        look_ahead_mask = tf.tile(
            look_ahead_mask, [tf.shape(inputs)[0], self.num_heads, 1, 1]
        )  # (batch_size, num_heads, seq_len, seq_len)
        dec_padding_mask = self.create_padding_mask(targets)

        logger.info("Before encoder call")
        enc_output = self.encoder(inputs, mask=enc_padding_mask)
        log_tensor_shape(enc_output, "enc_output")

        logger.info("Before decoder call")
        dec_output = self.decoder(
            targets, enc_output, look_ahead_mask, dec_padding_mask
        )
        log_tensor_shape(dec_output, "dec_output")

        final_output = self.final_layer(dec_output)
        log_tensor_shape(final_output, "final_output")

        logger.info("Completed Transformer call")

        return final_output


if __name__ == "__main__":
    tf.random.set_seed(42)

    batch_size = 64
    seq_length_input = 50
    seq_length_target = 60
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6
    input_vocab_size = 8500
    target_vocab_size = 8000
    max_seq_len_input = 100
    max_seq_len_target = 100

    sample_input = tf.random.uniform(
        (batch_size, seq_length_input), maxval=input_vocab_size, dtype=tf.int32
    )
    sample_target = tf.random.uniform(
        (batch_size, seq_length_target), maxval=target_vocab_size, dtype=tf.int32
    )

    transformer = Transformer(
        num_layers,
        d_model,
        num_heads,
        d_ff,
        input_vocab_size,
        target_vocab_size,
        max_seq_len_input,
        max_seq_len_target,
    )

    output = transformer(sample_input, sample_target)

    logger.info(f"Output shape: {output.shape}")
    logger.info(f"Sample output (first element of the first batch): {output[0][0]}")
