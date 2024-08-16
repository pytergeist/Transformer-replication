from collections import Counter


class BytePairEncoding:
    def __init__(self):
        pass

    def get_word_frequencies(self, corpus):
        word_frequencies = Counter(corpus.split())
        return {word: frequency for word, frequency in word_frequencies.items()}


if __name__ == "__main__":
    corpus = "Tom is working hard to build a tokeniser because a tokeniser is working for tom"
    encoder = BytePairEncoding()
    print(encoder.bin_vocab(corpus))
