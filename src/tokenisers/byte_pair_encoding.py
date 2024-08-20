import numpy as np
from collections import Counter, defaultdict


class BytePairEncoding:
    def __init__(self):
        pass

    def get_vocab_frequencies(self, corpus):
        word_frequencies = Counter(corpus.split())
        return {word: frequency for word, frequency in word_frequencies.items()}

    def split_vocab_into_chars(self, vocab):
        return {" ".join(list(key)): value for key, value in vocab.items()}

    def calculate_pair_frequency(self, vocab):
        pair_frequencies = defaultdict(int)
        for word, frequency in vocab.items():
            chars = word.split()
            for idx in range(len(chars) - 1):
                pair_frequencies[(chars[idx], chars[idx + 1])] += frequency
        return pair_frequencies

    def merge_pairs(self, pair_frequencies, vocab):
        most_frequent_pair = max(pair_frequencies, key=pair_frequencies.get)
        most_frequent_pair_str = ' '.join(most_frequent_pair)
        replacement = ''.join(most_frequent_pair)

        merged_vocab = {}
        for word, frequency in vocab.items():
            new_word = word.replace(most_frequent_pair_str, replacement)
            merged_vocab[new_word] = frequency

        return merged_vocab

    def encode(self, corpus):
        vocab = self.get_vocab_frequencies(corpus)
        vocab = self.split_vocab_into_chars(vocab)
        step = 0
        while True:
            pair_frequencies = self.calculate_pair_frequency(vocab)
            if not pair_frequencies:
                break
            most_frequent_pair = max(pair_frequencies, key=pair_frequencies.get)
            if pair_frequencies[most_frequent_pair] == 0:
                break
            vocab = self.merge_pairs(pair_frequencies, vocab)
            step += 1
        return vocab


if __name__ == "__main__":
    import nltk

    nltk.download('brown')
    from nltk.corpus import brown

    corpus = ' '.join(brown.words())

    encoder = BytePairEncoding()
    final_vocab = encoder.encode(corpus)
    print("Final Vocab:", final_vocab)


