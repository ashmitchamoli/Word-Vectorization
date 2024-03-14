import nltk
import pandas as pd

class Preprocessor:
    def __init__(self) -> None:
        pass

    def getTokens(self, fileName) -> tuple[list[list[str]], dict]:
        """
        returns:
            tokens: a list of list of tokens.
            wordIndices: a dictionary containing all unique words and their unique index.
        """
        tokens = []
        vocabuary = set()

        data = self.readData(fileName)

        def tokenizeSentence(row):
            sentence = row['Description']
            sentenceTokens = nltk.word_tokenize(sentence)
            tokens.append(sentenceTokens)
            vocabuary.update(sentenceTokens)

        data.apply(tokenizeSentence, axis=1)

        wordIndices = {word: i for i, word in enumerate(vocabuary)}

        return tokens, wordIndices

    def readData(self, fileName) -> pd.DataFrame:
        return pd.read_csv(fileName)