import numpy as np
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

    def merge_pairs(self, pair_frequencies, vocab):
        most_frequent_pair_idx = np.argmax(list(pair_frequencies.values()))
        most_frequent_pair = list(pair_frequencies.keys())[most_frequent_pair_idx]
        most_frequent_pair_str = ' '.join(most_frequent_pair)
        replacement = ''.join(most_frequent_pair)

        merged_vocab = {}
        for word, frequency in vocab.items():
            new_word = word.replace(most_frequent_pair_str, replacement)
            merged_vocab[new_word] = frequency

        return merged_vocab

    def encode(self, corpus, num_merges):
        vocab = self.get_vocab_frequencies(corpus)
        vocab = self.split_vocab_into_chars(vocab)

        for i in range(num_merges):
            pair_frequencies = self.calculate_pair_frequency(vocab)
            if not pair_frequencies:
                break
            vocab = self.merge_pairs(pair_frequencies, vocab)
            print(f"Step {i + 1}, updated vocab: {vocab}")

        return vocab


if __name__ == "__main__":
    corpus = "Tom is working hard to build a tokeniser because a tokeniser is working for tom"
    encoder = BytePairEncoding()
    final_vocab = encoder.encode(corpus, num_merges=5)
    print("Final Vocab:", final_vocab)
