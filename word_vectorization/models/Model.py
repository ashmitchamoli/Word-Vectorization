import os
import numpy as np
import pickle as pkl

from word_vectorization.preprocess.Preprocessor import Preprocessor

class WordVectorizationModel:
    def __init__(self, contextSize : int, trainFilePath : str) -> None:
        self.contextSize = contextSize
        self.trainFilePath = trainFilePath

        self.preprocessor = Preprocessor()
        self.tokens, self.wordIndices = self.preprocessor.getTokens(trainFilePath)
        self.indexWords = {i: word for word, i in self.wordIndices.items()}
        self.embeddings = None
        self.MODEL_CACHE_PATH = None
        self._modelFileSuffix = None

        print(f'Loaded training data. Vocabulary size: {len(self.wordIndices)}')
    
    def getEmbedding(self, word : str):
        word = self.preprocessor.processWord(word)
        if(word not in self.wordIndices):
            return None
        wordIndex = self.wordIndices[word]
        return self.embeddings[wordIndex]
    
    def _saveEmbeddings(self):
        if not os.path.exists(self.MODEL_CACHE_PATH):
            os.makedirs(self.MODEL_CACHE_PATH)

        embeddings = {}
        for i in self.indexWords:
            embeddings[self.indexWords[i]] = self.embeddings[i]
        
        pkl.dump(embeddings, open(os.path.join(self.MODEL_CACHE_PATH, f'embeddings{self._modelFileSuffix}.pkl'), 'wb'))
    
    def _loadEmbeddings(self) -> bool:
        if not os.path.exists(os.path.join(self.MODEL_CACHE_PATH, f'embeddings{self._modelFileSuffix}.pkl')):
            return False

        embeddings = np.zeros(shape=(len(self.wordIndices), self.embeddingSize))
        embeddingDict = pkl.load(open(os.path.join(self.MODEL_CACHE_PATH, f'embeddings{self._modelFileSuffix}.pkl'), 'rb'))
        for word in embeddingDict:
            embeddings[self.wordIndices[word], :] = embeddingDict[word]

        self.embeddings = embeddings

        return True
