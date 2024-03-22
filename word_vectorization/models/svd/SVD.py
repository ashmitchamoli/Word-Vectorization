import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import svds
import pickle as pkl
import os

from word_vectorization.models.Model import WordVectorizationModel

MODEL_CACHE_PATH = './model_cache/svd'

class SvdWordVectorizationModel(WordVectorizationModel):
    def __init__(self, *parentClassArgs, embeddingSize : int = 16) -> None:
        """
        Args:
            parentClassArgs: Positional arguments for the WordVectorizationModel parent class.
            embeddingSize: Size of the embedding. Should be less than the vocabulary size.
        """
        super().__init__(*parentClassArgs)

        self.embeddingSize = embeddingSize
        self.__trained = False
        self.__modelFileSuffix = f"_{self.contextSize}_{self.embeddingSize}_{len(self.indexWords)}"

    def train(self, verbose : bool = True, retrain : bool = False):
        if self.__loadEmbeddings() and not retrain:
            if verbose:
                print("Model already trained. Embeddings loaded.")
            return self.embeddings
        else:
            if verbose:
                print("Embeddings not found. Starting training from scratch.")

        coOccurrenceMatrix = self.computeCoOccurrenceMatrix()
        if verbose:
            print("Computed co-occurence matrix.")
        
        self.embeddings, self.S, self.Vh = svds(coOccurrenceMatrix, k=self.embeddingSize, which='LM')
        if verbose:
            print("Computed Partial Singular Value Decomposition of the co-occurence matrix.")

        self.__saveEmbeddings() 

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
        
        return coOccurrenceMatrix
    
    def initializeCoOccurrenceMatrix(self):
        coOccuranceMatrix = lil_matrix((len(self.wordIndices), len(self.wordIndices)), dtype=np.float64)
        
        return coOccuranceMatrix
    
    def __saveEmbeddings(self):
        # check if directory exists
        if not os.path.exists(MODEL_CACHE_PATH):
            os.mkdir(MODEL_CACHE_PATH)
        
        pkl.dump(self.embeddings, open(os.path.join(MODEL_CACHE_PATH, f'embeddings{self.__modelFileSuffix}.pkl'), 'wb'))

    def __loadEmbeddings(self):
        if not os.path.exists(f'./model_cache/svd/embeddings{self.__modelFileSuffix}.pkl'):
            # raise ValueError("Embeddings not found. Train the model first.")
            return False

        self.embeddings = pkl.load(open(f'./model_cache/svd/embeddings{self.__modelFileSuffix}.pkl', 'rb'))

        return True

if __name__ == "__main__":
    model = SvdWordVectorizationModel()