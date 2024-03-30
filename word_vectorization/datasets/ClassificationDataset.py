import torch

from word_vectorization.preprocess.Preprocessor import Preprocessor

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, filePath : str, wordIndices : dict[str, int]) -> None:
        super().__init__()

        self.preprocessor = Preprocessor()
        self.data = self.preprocessor.readData(filePath)
        self.wordIndices = wordIndices
        self.indexWords = { self.wordIndices[word] : word for word in wordIndices }
        self.classes = self.data['Class Index'].unique()-1
        self.numClasses = len(self.classes)

        def tokenizeSentences(row):
            sentence = row['Description']
            tokenizedSentence = self.preprocessor.tokenizeSentence(sentence)
            row['Description'] = torch.tensor([ self.wordIndices[word] if word in self.wordIndices else -1 for word in tokenizedSentence ])
            return row
        
        self.data = self.data.apply(tokenizeSentences, axis=1)
        self.tokenizedSentences = list(self.data['Description'])
        self.labels = list(self.data['Class Index'])

    def __getitem__(self, index : int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        return a tensor of wordIndices and a tensor, label.
        """
        label = self.labels[index]

        return self.tokenizedSentences[index], torch.tensor(label-1)

    def __len__(self) -> int:
        return len(self.data)
    
    def _custom_collate(self, batch):
        X, y = zip(*batch)

        X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=-1)

        return X, torch.stack(y)
