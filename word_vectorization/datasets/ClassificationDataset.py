import torch

from word_vectorization.preprocess.Preprocessor import Preprocessor

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, filePath : str, wordIndices : dict[str, int]) -> None:
        super().__init__()

        self.preprocessor = Preprocessor()
        self.data = self.preprocessor.readData(filePath)
        self.wordIndices = wordIndices
        self.indexWords = { self.wordIndices[word] : word for word in wordIndices }
    
    def __getitem__(self, index : int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        return 
        """
        sentence = self.data.loc[index, 'Description']
        label = self.data.loc[index, 'Class Index']

        tokenizedSentence = self.preprocessor.tokenizeSentence(sentence)
        
        return torch.tensor([ self.indexWords[word] for word in tokenizedSentence ]), torch.tensor(label)

    def __len__(self) -> int:
        return len(self.data)
    
    def _custom_collate(self):
        pass
