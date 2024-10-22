import tensorflow as tf

from experiments.english_to_german.prepare_data import load_and_preprocess_data
from src.blocks.transformer import Transformer


def run_experiment():
    # Set GPU memory growth (optional, depending on your hardware setup)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    tf.random.set_seed(42)

    batch_size = 64
    seq_length_input = 256
    seq_length_target = 256
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6
    num_epochs = 20

    # Load and preprocess data
    train_dataset, val_dataset, input_vocab_size, target_vocab_size = (
        load_and_preprocess_data(
            batch_size=batch_size,
            max_length=seq_length_input,
            train_size=100,  # Limit to 100,000 training examples
            val_size=50,  # Limit to 5,000 validation examples
        )
    )

    # Prepare (inputs, targets) tuples
    def prepare_inputs_targets(inputs, targets):
        targets_input = targets[:, :-1]
        targets_shifted = targets[:, 1:]

        # Pad targets to match the expected output shape
        padded_targets_input = tf.pad(targets_input, [[0, 0], [0, 1]], "CONSTANT")
        padded_targets_shifted = tf.pad(targets_shifted, [[0, 0], [0, 1]], "CONSTANT")

        return (inputs, padded_targets_input), padded_targets_shifted

    train_dataset = train_dataset.map(prepare_inputs_targets)
    val_dataset = val_dataset.map(prepare_inputs_targets)

    # Pad the dataset to ensure consistent sequence lengths
    train_dataset = train_dataset.padded_batch(
        batch_size, padded_shapes=(([None], [None]), [None])
    )
    val_dataset = val_dataset.padded_batch(
        batch_size, padded_shapes=(([None], [None]), [None])
    )

    # Initialize the Transformer model
    transformer = Transformer(
        num_layers,
        d_model,
        num_heads,
        d_ff,
        input_vocab_size,
        target_vocab_size,
        max_seq_len_input=seq_length_input,
        max_seq_len_target=seq_length_target,
    )

    # Compile the model
    transformer.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    # Train the model
    transformer.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=val_dataset,
    )

    # Evaluate the model
    loss, accuracy = transformer.evaluate(val_dataset)
    print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")


if __name__ == "__main__":
    run_experiment()
