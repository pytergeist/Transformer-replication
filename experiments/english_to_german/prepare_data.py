# prepare_data.py

import tensorflow as tf
import tensorflow_datasets as tfds


def load_and_preprocess_data(
    batch_size=64, max_length=40, train_size=None, val_size=None
):
    examples, _ = tfds.load("wmt14_translate/de-en", with_info=True, as_supervised=True)

    # Limit the dataset size if train_size or val_size is specified
    train_examples = (
        examples["train"].take(train_size) if train_size else examples["train"]
    )
    val_examples = (
        examples["validation"].take(val_size) if val_size else examples["validation"]
    )

    tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        (en.numpy() for de, en in train_examples), target_vocab_size=2**13
    )
    tokenizer_de = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        (de.numpy() for de, en in train_examples), target_vocab_size=2**13
    )

    def encode(lang1, lang2):
        lang1 = (
            [tokenizer_de.vocab_size]
            + tokenizer_de.encode(lang1.numpy())
            + [tokenizer_de.vocab_size + 1]
        )
        lang2 = (
            [tokenizer_en.vocab_size]
            + tokenizer_en.encode(lang2.numpy())
            + [tokenizer_en.vocab_size + 1]
        )
        return lang1, lang2

    def tf_encode(de, en):
        result_de, result_en = tf.py_function(encode, [de, en], [tf.int64, tf.int64])
        result_de.set_shape([None])
        result_en.set_shape([None])
        return result_de, result_en

    train_dataset = train_examples.map(tf_encode)
    val_dataset = val_examples.map(tf_encode)

    def filter_max_length(x, y, max_length=max_length):
        return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)

    train_dataset = train_dataset.filter(filter_max_length)
    val_dataset = val_dataset.filter(filter_max_length)

    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(1024)
    train_dataset = train_dataset.padded_batch(
        batch_size, padded_shapes=([None], [None])
    )
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = val_dataset.padded_batch(batch_size, padded_shapes=([None], [None]))

    input_vocab_size = tokenizer_de.vocab_size + 2
    target_vocab_size = tokenizer_en.vocab_size + 2

    return train_dataset, val_dataset, input_vocab_size, target_vocab_size
