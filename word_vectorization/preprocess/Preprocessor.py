import nltk
import pandas as pd
import re

class Preprocessor:
    def __init__(self) -> None:
        self.stemmer = nltk.stem.PorterStemmer()
        pass

    def getTokens(self, fileName : str) -> tuple[list[list[str]], dict]:
        """
        returns:
            tokens: a list of list of tokens.
            wordIndices: a dictionary containing all unique words and their unique index.
        """
        tokens = []
        vocabuary = set()

        data = self.readData(fileName)
        self.labels = list(data['Class Index'])

        def tokenizeSentence(row):
            sentence = row['Description']

            # get tokenized sentence
            stemmedTokens = self.tokenizeSentence(sentence)       
            tokens.append(stemmedTokens)

            # update vocabulary
            vocabuary.update(stemmedTokens)

        data.apply(tokenizeSentence, axis=1)

        wordIndices : dict[str, int] = {word: i for i, word in enumerate(vocabuary)}
        wordIndices['<UNK>'] = len(wordIndices)
        wordIndices['<PAD>'] = len(wordIndices)

        return tokens, wordIndices
    
    def tokenizeSentence(self, sentence : str) -> list[str]:
        """
        return a list of tokenized, stemmed, lowered words of the given sentence.
        """
        # basic tokenization
        sentenceTokens = nltk.tokenize.word_tokenize(sentence)

        # extract alphanumeric phrases
        newSentenceTokens = []
        for token in sentenceTokens:
            tokenList = re.split('[^a-zA-Z0-9]', token)
            tokenList = [ word.lower() for word in tokenList ]
            # tokenList = [ word.strip().lower() for word in tokenList if word.strip() != '' ]
            newSentenceTokens += tokenList

        # stemming
        stemmedTokens = self.stemWords(newSentenceTokens)
        # stemmedTokens = newSentenceTokens
        
        return stemmedTokens
    
    def stemWords(self, tokenList : list[str]) -> list[str]:
        return [self.stemmer.stem(token) for token in tokenList]
    
    def processWord(self, word : str) -> str:
        return self.stemmer.stem(word.lower())

    def readData(self, fileName) -> pd.DataFrame:
        return pd.read_csv(fileName)