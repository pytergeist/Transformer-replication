# transformer.py

import tensorflow as tf

from src.blocks.decoder import DecoderStack
from src.blocks.encoder import EncoderStack
from src.layers.dropout import DropoutLayer
from src.layers.feed_forward import FeedForwardLayer
from src.layers.positional_encoding import PositionalEncoding
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

        self.num_heads = num_heads
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
        mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
        mask = tf.squeeze(mask[:, tf.newaxis, tf.newaxis, :, :1], axis=-1)
        log_tensor_shape(mask, "initial padding mask shape")
        return mask

    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)

    def call(
        self,
        inputs,
    ):
        inputs, targets = inputs
        inputs = self.positional_encoding_inputs(inputs)
        targets = self.positional_encoding_targets(targets)

        inputs = self.dropout(inputs)
        targets = self.dropout(targets)

        enc_padding_mask = self.create_padding_mask(inputs)
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(targets)[1])
        look_ahead_mask = look_ahead_mask[
            tf.newaxis, tf.newaxis, :, :
        ]  # (1, 1, seq_len, seq_len)
        look_ahead_mask = tf.tile(
            look_ahead_mask, [tf.shape(inputs)[0], self.num_heads, 1, 1]
        )  # (batch_size, num_heads, seq_len, seq_len)
        dec_padding_mask = self.create_padding_mask(targets)

        enc_output = self.encoder(inputs, mask=enc_padding_mask)

        dec_output = self.decoder(
            targets, enc_output, look_ahead_mask, dec_padding_mask
        )

        final_output = self.final_layer(dec_output)

        return final_output


if __name__ == "__main__":
    tf.random.set_seed(42)

    batch_size = 64
    seq_length_input = 256
    seq_length_target = 256
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6
    input_vocab_size = 85000
    target_vocab_size = 80000
    max_seq_len_input = 256
    max_seq_len_target = 256
    num_epochs = 20

    # Create sample data
    sample_input = tf.random.uniform(
        (batch_size, seq_length_input), maxval=input_vocab_size, dtype=tf.int32
    )
    sample_target = tf.random.uniform(
        (batch_size, seq_length_target), maxval=target_vocab_size, dtype=tf.int32
    )

    # Prepare inputs and targets
    inputs = sample_input
    targets = sample_target[:, :-1]
    targets_shifted = sample_target[:, 1:]

    padded_targets = tf.pad(targets, [[0, 0], [0, 1]], "CONSTANT")
    padded_targets_shifted = tf.pad(targets_shifted, [[0, 0], [0, 1]], "CONSTANT")

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

    # Compile the model
    transformer.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    # Train the model
    transformer.fit(
        x=(inputs, padded_targets),
        y=padded_targets_shifted,
        batch_size=batch_size,
        epochs=num_epochs,
    )
