import numpy as np

from word_vectorization.models.Model import WordVectorizationModel

class SvdWordVectorizationModel(WordVectorizationModel):
    def __init__(self, *parentClassArgs, embeddingSize : int = 16) -> None:
        """
        Args:
            parentClassArgs: Positional arguments for the WordVectorizationModel parent class.
            embeddingSize: Size of the embedding. Should be less than the vocabulary size.
        """
        super().__init__(*parentClassArgs)

        if embeddingSize > len(self.wordIndices):
            raise ValueError("Embedding size should not be greater than the vocabulary size.")

        self.embeddingSize = embeddingSize

    def train(self):
        coOccurrenceMatrix = self.computeCoOccurrenceMatrix()

        self.U, self.S, self.V = np.linalg.svd(coOccurrenceMatrix)

        # selecting first embeddingSize
        self.embeddings = self.U[:, :self.embeddingSize]

        return self.embeddings

    def computeCoOccurrenceMatrix(self):
        coOccurrenceMatrix = self.initializeCoOccurrenceMatrix()
        for sentence in self.tokens:
            i = 0
            j = 0
            while j < len(sentence)-1:
                if j-i < self.contextSize:
                    j += 1

                elif j-i == self.contextSize:
                    j += 1
                    i += 1

                for k in range(i, j):
                    coOccurrenceMatrix[self.wordIndices[sentence[k]], self.wordIndices[sentence[j]]] += 1
                    coOccurrenceMatrix[self.wordIndices[sentence[j]], self.wordIndices[sentence[k]]] += 1
    
    def initializeCoOccurrenceMatrix(self):
        coOccuranceMatrix = np.zeros(shape=(len(self.wordIndices), len(self.wordIndices)))
        
        return coOccuranceMatrix
    
if __name__ == "__main__":
    model = SvdWordVectorizationModel()