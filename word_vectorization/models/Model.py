from word_vectorization.preprocess.Preprocessor import Preprocessor

class WordVectorizationModel:
    def __init__(self, contextSize : int, trainFilePath : str) -> None:
        self.contextSize = contextSize

        self.preprocessor = Preprocessor()
        self.tokens, self.wordIndices = self.preprocessor.getTokens(trainFilePath)
        self.indexWords = {i: word for word, i in self.wordIndices.items()}

        # print(f'Loaded vocabulary of size {len(self.wordIndices)}')
    
    def getEmbedding(self, word : str):
        word = self.preprocessor.processWord(word)
        if(word not in self.wordIndices):
            return None
        wordIndex = self.wordIndices[word]
        return self.embeddings[wordIndex]
