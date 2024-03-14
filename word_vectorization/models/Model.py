from word_vectorization.preprocess.Preprocessor import Preprocessor

class WordVectorizationModel:
    def __init__(self, contextSize : int, trainFilePath : str) -> None:
        self.contextSize = contextSize

        self.preprocessor = Preprocessor()
        self.tokens, self.wordIndices = self.preprocessor.getTokens(trainFilePath)
        self.indexWords = {i: word for word, i in self.wordIndices.items()}