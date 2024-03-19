from word_vectorization.models.Model import WordVectorizationModel

class Word2Vec(WordVectorizationModel):
    def __init__(self, *parentClassArgs, embeddingSize :  int):
        """
        """
        super().__init__(*parentClassArgs)

    