import nltk
import pandas as pd

class Preprocessor:
    def __init__(self) -> None:
        pass

    def getTokens(self, fileName) -> list[list[str]]:
        tokens = []

        data = self.readData(fileName)

        def tokenizeSentence(row):
            sentence = row['Description']
            sentenceTokens = nltk.word_tokenize(sentence)
            tokens.append(sentenceTokens)

        data.apply(tokenizeSentence, axis=1)

        return tokens

    def readData(self, fileName) -> pd.DataFrame:
        return pd.read_csv(fileName)