import numpy
from collections import Counter, defaultdict


class BytePairEncoding:
    def __init__(self):
        pass

    def get_vocab_frequencies(self, corpus):
        word_frequencies = Counter(corpus.split())
        return {word: frequency for word, frequency in word_frequencies.items()}

    def split_vocab_into_chars(self, vocab):
        return {" ".join(key): value for key, value in vocab.items()}

    def calculate_pair_frequency(self, vocab):
        pair_frequencies = defaultdict(int)
        for word, frequency in vocab.items():
            chars = word.split()
            for idx in range(len(chars) - 1):
                pair_frequencies[(chars[idx], chars[idx + 1])] += frequency
        return pair_frequencies


if __name__ == "__main__":
    corpus = "Tom is working hard to build a tokeniser because a tokeniser is working for tom"
    encoder = BytePairEncoding()
    vocab = encoder.get_vocab_frequencies(corpus)
    vocab = encoder.split_vocab_into_chars(vocab)
    freqs = encoder.calculate_pair_frequency(vocab)
    print(freqs)
