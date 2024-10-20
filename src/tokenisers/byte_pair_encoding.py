# byte_pair_encoder.py

import re
from collections import Counter, defaultdict


class BytePairEncoding:
    def __init__(self, corpus: str, num_merges: int):
        self.corpus = corpus
        self.num_merges = num_merges

    def set_normaliser(self, normalise: list[object]):
        if not isinstance(normalise, list):
            normalise = [normalise]
        setattr(self, "normalise", normalise)

    def preprocess_corpus(self, text):
        return re.sub(r"(\W)", r" \1 ", text)

    @staticmethod
    def get_vocab_frequencies(corpus):
        word_frequencies = Counter(corpus.split())
        return {word: frequency for word, frequency in word_frequencies.items()}

    @staticmethod
    def split_vocab_into_chars(vocab):
        return {" ".join(list(key)): value for key, value in vocab.items()}

    @staticmethod
    def calculate_pair_frequency(vocab):
        pair_frequencies = defaultdict(int)
        for word, frequency in vocab.items():
            chars = word.split()
            for idx in range(len(chars) - 1):
                pair_frequencies[(chars[idx], chars[idx + 1])] += frequency
        return pair_frequencies

    @staticmethod
    def merge_pairs(pair_frequencies, vocab):
        most_frequent_pair = max(pair_frequencies, key=pair_frequencies.get)
        most_frequent_pair_str = " ".join(most_frequent_pair)
        replacement = "".join(most_frequent_pair)

        merged_vocab = {}
        for word, frequency in vocab.items():
            new_word = word.replace(most_frequent_pair_str, replacement)
            merged_vocab[new_word] = frequency

        return merged_vocab

    def encode(self):
        if hasattr(self, "normalise"):
            for normalise in self.normalise:
                self.corpus = normalise.apply(self.corpus)
        self.corpus = self.preprocess_corpus(self.corpus)
        vocab = self.get_vocab_frequencies(self.corpus)
        vocab = self.split_vocab_into_chars(vocab)
        step = 0
        while step < self.num_merges:
            pair_frequencies = self.calculate_pair_frequency(vocab)
            if not pair_frequencies:
                break
            most_frequent_pair = max(pair_frequencies, key=pair_frequencies.get)
            if pair_frequencies[most_frequent_pair] == 0:
                break
            vocab = self.merge_pairs(pair_frequencies, vocab)
            step += 1
        return vocab

    def tokenise_sentence(self, sentence):
        if hasattr(self, "normalise"):
            for normalise in self.normalise:
                sentence = normalise.apply(sentence)
        sentence = self.preprocess_corpus(sentence)
        words = sentence.split()
        tokens = []
        for word in words:
            chars = " ".join(list(word))
            for _ in range(self.num_merges):
                pair_frequencies = self.calculate_pair_frequency({chars: 1})
                if not pair_frequencies:
                    break
                chars = self.merge_pairs(pair_frequencies, {chars: 1})
                chars = list(chars.keys())[0]
            tokens.append(chars.replace(" ", ""))
        return tokens


if __name__ == "__main__":
    import time

    import nltk

    start_time = time.time()

    nltk.download("brown")
    from nltk.corpus import brown

    corpus = " ".join(brown.words())

    encoder = BytePairEncoding(corpus, 200)
    final_vocab = encoder.encode()

    sample_sentence = "The Fulton County Grand Jury said Friday an investigation of Atlanta's recent primary election produced no evidence that any irregularities took place."
    custom_tokens = encoder.tokenise_sentence(sample_sentence)

    end_time = time.time()
    print("Final Vocab:", final_vocab)
    print("Custom Tokens:", custom_tokens)
    print(f"Execution time: {end_time - start_time} seconds")
