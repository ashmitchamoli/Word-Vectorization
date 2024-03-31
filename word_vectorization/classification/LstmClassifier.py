import torch
import numpy as np
import tqdm
import os
from typing_extensions import Literal
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

from word_vectorization.datasets.ClassificationDataset import ClassificationDataset
from word_vectorization.models.Model import WordVectorizationModel

class SentenceClassifier:
    def __init__(self, trainFilePath : str, trainedVectorizationModel : WordVectorizationModel, hiddenSize : int, numLayers : int, bidirectional : bool, hiddenLayers : list[int] = [], activation : Literal["relu", "tanh", "sigmoid"] | None = None) -> None:
        self.trainDataset = ClassificationDataset(filePath=trainFilePath,
                                                  wordIndices=trainedVectorizationModel.wordIndices)
        
        self.classifier = LstmClassifier(embeddings=trainedVectorizationModel.embeddings,
                                         outChannels=self.trainDataset.numClasses,
                                         hiddenSize=hiddenSize,
                                         numLayers=numLayers,
                                         bidirectional=bidirectional,
                                         hiddenLayers=hiddenLayers,
                                         activation=activation)
        
        self.MODEL_CACHE_PATH = './model_cache/lstm_classifier/'
        self._modelFileSuffix = f'_{len(self.trainDataset.tokenizedSentences)}_{self.trainDataset.numClasses}_{hiddenSize}_{numLayers}_{bidirectional}_{hiddenLayers}_{activation}_{trainedVectorizationModel._modelFileSuffix}'

    def train(self, epochs : int = 10, lr : float = 0.001, batchSize : int = 32, verbose : bool = True, retrain : bool = False) -> None:
        if not retrain and self._loadModel():
            if verbose:
                print("Found and loaded trained model.")
            return
        else:
            if verbose:
                print("Saved model not found or retrain flag is set. Starting training from scratch.")
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)

        self.logs = {}

        trainLoader = torch.utils.data.DataLoader(self.trainDataset, batch_size=batchSize, shuffle=True, collate_fn=self.trainDataset._custom_collate)

        for epoch in range(epochs):
            runningLoss = 0
            self.logs[epoch] = {}
            for x, y in tqdm.tqdm(trainLoader, desc='Training', leave=True):
                optimizer.zero_grad()
                output = self.classifier(x)

                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

                runningLoss += loss.item()
            
            runningLoss /= len(trainLoader)
            self.logs[epoch]['Loss'] = runningLoss
            
            # save every 3 epochs
            if (epoch+1) % 3 == 0:
                self._saveModel()

            if verbose:
                print(f'Epoch {epoch+1}/{epochs} | Loss: {runningLoss:.3f}')

        self._saveModel()

    def evaluate(self, dataset : ClassificationDataset) -> dict[str, float]:
        """
        Returns:
            scores : a dictionary with the following fields: 'Accuracy', 'F1', 'Precision', 'Recall', 'Report'
        """
        scores = {}
        preds = []
        groundTruth = []
        for i in range(len(dataset)):
            x, y = dataset[i]
            output = self.classifier(x.unsqueeze(0))
            preds.append(torch.argmax(output, dim=1).item())
            groundTruth.append(y.item())
        
        scores['Accuracy'] = accuracy_score(groundTruth, preds)
        scores['F1'] = f1_score(groundTruth, preds, average='macro', zero_division=0)
        scores['Precision'] = precision_score(groundTruth, preds, average='macro', zero_division=0)
        scores['Recall'] = recall_score(groundTruth, preds, average='macro', zero_division=0)
        scores['Report'] = classification_report(groundTruth, preds, zero_division=0)
        scores['Confusion Matrix'] = confusion_matrix(groundTruth, preds)

        return scores

    def _saveModel(self) -> None:
        """
        saves the classifier weights to MODEL_CACHE_PATH 
        """
        if not os.path.exists(self.MODEL_CACHE_PATH):
            os.makedirs(self.MODEL_CACHE_PATH)

        # torch.save(self.classifier.state_dict(), os.path.join(self.MODEL_CACHE_PATH, 'model.pt'))
        torch.save(self.classifier.state_dict(), os.path.join(self.MODEL_CACHE_PATH, f'model{self._modelFileSuffix}.pt'))

    def _loadModel(self) -> bool:
        """
        loads the classifier weights from MODEL_CACHE_PATH
        """
        modelFilePath = os.path.join(self.MODEL_CACHE_PATH, f'model{self._modelFileSuffix}.pt')
        if not os.path.exists(modelFilePath):
            return False
        
        self.classifier.load_state_dict(torch.load(modelFilePath))
        return True


class LstmClassifier(torch.nn.Module):
    def __init__(self, embeddings : np.ndarray, outChannels : int, hiddenSize : int, numLayers : int, bidirectional : bool, hiddenLayers : list[int] = [], activation : Literal["relu", "tanh", "sigmoid"] | None = None) -> None:
        super().__init__()

        self.embeddings = torch.tensor(embeddings.copy(), dtype=torch.float)
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
            self.classifier.append(torch.nn.Linear( hiddenSize * (bidirectional+1) if i == 0 else hiddenLayers[i-1],
                                                    outChannels if i == len(hiddenLayers) else hiddenLayers[i] ))
            if i != len(hiddenLayers):
                self.classifier.append(self.activation)
        
    def forward(self, x):
        # print(x.shape)

        # get embeddings
        x = self.embeddings[x]
        # print(x.shape)

        # get sentence embedding
        x, _ = self.lstm(x)
        # print(x.shape)

        # classification
        x = self.classifier(x[:, -1, :])
        # print(x.shape)
        
        return x