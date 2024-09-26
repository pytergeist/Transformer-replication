import tensorflow as tf
from experiments.english_to_german.prepare_data import load_and_preprocess_data
from src.blocks.transformer import Transformer


def run_experiment():
    tf.random.set_seed(42)

    batch_size = 64
    seq_length_input = 40
    seq_length_target = 40
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6
    num_epochs = 20

    # Load and preprocess data
    train_dataset, val_dataset, input_vocab_size, target_vocab_size = (
        load_and_preprocess_data(batch_size=batch_size, max_length=seq_length_input)
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
