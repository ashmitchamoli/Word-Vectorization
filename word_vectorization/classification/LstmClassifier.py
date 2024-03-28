import torch
import numpy as np
from typing_extensions import Literal


class LstmClassifier(torch.nn.Module):
    def __init__(self, embeddings : np.ndarray, outChannels : int, hiddenSize : int, numLayers : int, bidirectional : bool, hiddenLayers : list[int] = [], activation : Literal["relu", "tanh", "sigmoid"] | None = None) -> None:
        super().__init__()

        self.embeddings = embeddings
        self.embeddingSize = embeddings.shape[1]
        if activation is None or activation == 'tanh':
            self.activation = torch.nn.Tanh()
        elif activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid()

        self.lstm = torch.nn.LSTM(input_size=self.embeddingSize, 
                                  hidden_size=hiddenSize,
                                  num_layers=numLayers,
                                  batch_first=True,
                                  bidirectional=bidirectional) 
        
        self.classifier = torch.nn.Sequential()

        for i in range(len(hiddenLayers) + 1):
            self.classifier.append(torch.nn.Linear( self.embeddingSize if i == 0 else hiddenLayers[i-1],
                                                    outChannels if i == len(hiddenLayers) else hiddenLayers[i] ))
            if i != len(hiddenLayers):
                self.classifier.append(self.activation)
        
    def forward(self, x):
        pass