# normalisers.py

class LowerCaseNormaliser:
    def __init__(self):
        pass

    @staticmethod
    def apply(corpus):
        return corpus.lower()
