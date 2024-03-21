import torch
import numpy as np
import tqdm
import os
from itertools import chain
import pickle as pkl

from word_vectorization.models.Model import WordVectorizationModel
from word_vectorization.datasets.Word2VecDataset import Word2VecDataset

MODEL_CACHE_PATH = './model_cache/word2vec/'

class Word2Vec(WordVectorizationModel):
    def __init__(self, *parentClassArgs, embeddingSize :  int, k : int):
        """
        """
        super().__init__(*parentClassArgs)
        
        self.embeddingSize = embeddingSize
        self.k = k
        self.word2VecModel = Word2VecClassifier(self.embeddingSize, self.wordIndices)
        self.__modelFileSuffix = f"_{self.contextSize}_{self.k}_{self.embeddingSize}_{len(self.wordIndices)}"

    def train(self, epochs : int = 10, lr : float = 0.001, batchSize : int = 32, verbose : bool = True, retrain : bool = False) -> None:
        if(not retrain and self.__loadEmbeddings()):
            if verbose:
                print("Found and loaded trained embeddings.")
            return self.embeddings
        else:
            if verbose:
                print("Saved embeddings not found. Starting training from scratch.")    
            
        trainDataset = Word2VecDataset(list(chain.from_iterable(self.tokens)), self.contextSize, self.k, self.wordIndices)
        
        optimizer = torch.optim.Adam(self.word2VecModel.parameters(), lr=lr)
        criterion = torch.nn.BCELoss()

        trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batchSize, shuffle=True)

        trainLoss = []

        for epoch in range(epochs):
            runningLoss = 0
            for x, y in tqdm.tqdm(trainLoader, desc='Training', leave=True):
                optimizer.zero_grad()
                output = self.word2VecModel(x)
                
                loss = criterion(output.reshape(-1, self.contextSize*2*(self.k+1)), y[:, 1:].float())
                runningLoss = loss.item()

                loss.backward()
                optimizer.step()

            runningLoss /= len(trainLoader)
            trainLoss.append(runningLoss)

            if verbose:
                print(f'Epoch {epoch+1}/{epochs} Loss: {runningLoss}')
        
        self.embeddings = self.word2VecModel.wordEmbeddings.cpu().weight.detach().numpy() + self.word2VecModel.contextEmbeddings.cpu().weight.detach().numpy()
        
        self.__saveEmbeddings()

        if verbose:
            print("Embeddings saved.")

        return self.embeddings

    def __saveEmbeddings(self):
        if not os.path.exists(MODEL_CACHE_PATH):
            os.makedirs(MODEL_CACHE_PATH)

        embeddings = {}
        for i in self.indexWords:
            embeddings[self.indexWords[i]] = self.embeddings[i]
        
        pkl.dump(embeddings, open(os.path.join(MODEL_CACHE_PATH, f'embeddings{self.__modelFileSuffix}.pkl'), 'wb'))
    
    def __loadEmbeddings(self) -> bool:
        if not os.path.exists(os.path.join(MODEL_CACHE_PATH, f'embeddings{self.__modelFileSuffix}.pkl')):
            return False

        embeddings = np.zeros(shape=(len(self.wordIndices), self.embeddingSize))
        embeddingDic = pkl.load(open(os.path.join(MODEL_CACHE_PATH, f'embeddings{self.__modelFileSuffix}.pkl'), 'rb'))
        for word in embeddingDic:
            embeddings[self.wordIndices[word], :] = embeddingDic[word]

        self.embeddings = embeddings

        return True

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
    