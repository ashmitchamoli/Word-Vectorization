import torch
import numpy as np
import tqdm
import os
from itertools import chain
import pickle as pkl

from word_vectorization.models.Model import WordVectorizationModel
from word_vectorization.datasets.Word2VecDataset import Word2VecDataset


class Word2Vec(WordVectorizationModel):
    def __init__(self, *parentClassArgs, embeddingSize :  int, k : int):
        """
        """
        super().__init__(*parentClassArgs)
        
        self.embeddingSize = embeddingSize
        self.k = k
        self.word2VecModel = Word2VecClassifier(self.embeddingSize, self.wordIndices)
        self._modelFileSuffix = f"_{self.contextSize}_{self.k}_{self.embeddingSize}_{len(self.indexWords)}"
        self.MODEL_CACHE_PATH = './model_cache/word2vec/'

    def train(self, epochs : int = 10, lr : float = 0.001, batchSize : int = 32, verbose : bool = True, retrain : bool = False) -> None:
        if(not retrain and self._loadEmbeddings()):
            if verbose:
                print("Found and loaded trained embeddings.")
            return self.embeddings
        else:
            if verbose:
                print("Saved embeddings not found. Starting training from scratch.")    
        
        trainDataset = Word2VecDataset(list(chain.from_iterable(self.tokens)), self.contextSize, self.k, self.wordIndices)
        trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
        
        optimizer = torch.optim.Adam(self.word2VecModel.parameters(), lr=lr)
        criterion = torch.nn.BCELoss()

        trainLoss = []
        for epoch in range(epochs):
            runningLoss = 0
            for x, y in tqdm.tqdm(trainLoader, desc='Training', leave=True):
                optimizer.zero_grad()
                output = self.word2VecModel(x)
                
                loss = criterion(output.reshape(-1, self.contextSize*2*(self.k+1)), y[:, 1:].float())
                loss.backward()
                optimizer.step()

                runningLoss += loss.item()

            runningLoss /= len(trainLoader)
            trainLoss.append(runningLoss)

            if verbose:
                print(f'Epoch {epoch+1}/{epochs} Loss: {runningLoss}')
        
        self.embeddings = self.word2VecModel.wordEmbeddings.cpu().weight.detach().numpy() + self.word2VecModel.contextEmbeddings.cpu().weight.detach().numpy()
        # self.embeddings /= np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        self._saveEmbeddings()

        if verbose:
            print("Embeddings saved.")

        return self.embeddings
    
    def getClosestWordEmbeddings(self, wordEmbedding : np.ndarray, k : int = 10) -> dict[str, np.ndarray]:
        """
        Parameters:
            wordEmbedding: a numpy array of size (self.embeddingSize, ). Normalized word embedding.
            k: number of closest word embeddings to return
        """
        # wordEmbedding /= np.linalg.norm(wordEmbedding)
        # distances = 1 - np.dot(self.embeddings, wordEmbedding)
        distances = np.linalg.norm(self.embeddings - wordEmbedding, axis=1)
        kClosestIndices = np.argsort(distances)[:k]

        return {self.indexWords[index]: self.embeddings[index] for index in kClosestIndices}

class Word2VecClassifier(torch.nn.Module):
    def __init__(self, embeddingSize : int, wordIndices : dict[str, int]) -> None:
        super().__init__()

        self.wordEmbeddings = torch.nn.Embedding(len(wordIndices), embeddingSize)
        self.contextEmbeddings = torch.nn.Embedding(len(wordIndices), embeddingSize)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # x contains the indices of 2*contextSize context words and 2*contextSize*k negative samples
        
        # first index of x is the index of the word
        word = self.wordEmbeddings(x[:, 0].reshape(-1, 1)) # size (batch_size, embeddingSize)

        # rest are the context words and negative samples
        embeds = self.contextEmbeddings(x[:, 1:]) # size (batch_size, 2 * contextSize * (k + 1), embeddingSize)

        output = self.sigmoid(torch.matmul(embeds, torch.transpose(word, 1, 2))) # size (2 * contextSize * (k + 1), 1) 
        return output