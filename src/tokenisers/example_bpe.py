import os

import nltk
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from nltk.corpus import brown
import sys

file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(file_path)

from src.tokenisers.byte_pair_encoding import BytePairEncoding
from src.tokenisers.normalisers import LowerCaseNormaliser

nltk.download("brown")

corpus = [" ".join(sent) for sent in brown.sents()]
corpus_str = " ".join(brown.words())

tokenizer = Tokenizer(BPE())

tokenizer.pre_tokenizer = Whitespace()

tokenizer.normalizer = normalizers.Sequence([Lowercase()])  # NFD(), , StripAccents()

tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=10000, min_frequency=2, special_tokens=["<pad>", "<s>", "</s>", "<unk>"]
)
tokenizer.train_from_iterator(corpus, trainer=trainer)

sample_sentence = "The Fulton County Grand Jury said Friday an investigation of Atlanta's recent primary election produced no evidence that any irregularities took place."
encoded = tokenizer.encode(sample_sentence)

encoder = BytePairEncoding(corpus_str, 500)
encoder.set_normaliser([LowerCaseNormaliser()])
final_vocab = encoder.encode()
custom_tokens = encoder.tokenise_sentence(sample_sentence)

print("Original Sentence:", sample_sentence)
print("Hugging Face Tokens:", encoded.tokens)
print("Custom Tokens:", custom_tokens)
print("Token IDs:", encoded.ids)
print("Decoded Back:", tokenizer.decode(encoded.ids))
