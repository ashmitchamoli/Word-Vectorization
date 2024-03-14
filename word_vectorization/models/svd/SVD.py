from word_vectorization.models.Model import WordVectorizationModel

class SvdWordVectorizationModel(WordVectorizationModel):
    def __init__(self, *args) -> None:
        super().__init__(args)

    def train(self, trainTokens):
        pass

    def computeCoOccurrenceMatrix(self, tokens):
        coOccurrenceMatrix = {}
        for sentence in tokens:
            currWindowSize = 0
        pass
