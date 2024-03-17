import nltk
import pandas as pd
import re

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
        stemmer = nltk.stem.PorterStemmer()

        data = self.readData(fileName)

        def tokenizeSentence(row):
            sentence = row['Description']
            sentenceTokens = nltk.tokenize.word_tokenize(sentence)
            
            newSentenceTokens = []
            for token in sentenceTokens:
                tokenList = re.split('[^a-zA-Z0-9]', token)
                tokenList = [ word.lower() for word in tokenList ]
                newSentenceTokens += tokenList

            stemmedTokens = [stemmer.stem(token) for token in newSentenceTokens]
            tokens.append(stemmedTokens)
            vocabuary.update(stemmedTokens)

        data.apply(tokenizeSentence, axis=1)

        wordIndices = {word: i for i, word in enumerate(vocabuary)}

        return tokens, wordIndices

    def readData(self, fileName) -> pd.DataFrame:
        return pd.read_csv(fileName)