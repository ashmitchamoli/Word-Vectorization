from word_vectorization.preprocess.Preprocessor import Preprocessor

class WordVectorizationModel:
    def __init__(self, contextSize : int) -> None:
        self.contextSize = contextSize
