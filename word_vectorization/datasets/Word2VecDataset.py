import torch

class Word2VecDataset(torch.utils.data.Dataset):
    def __init__(self, tokens : list[str], contextSize : int, k : int, wordIndices : dict[str, int]) -> None:
        super().__init__()

        self.tokens = tokens
        self.contextSize = contextSize
        self.k = k
        self.wordIndices = wordIndices
        self.indexWords = {i: word for word, i in wordIndices.items()}

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        # here, index is the index of the word in the dict wordIndices
        context = [self.wordIndices[self.tokens[index]]]
        y = [-1]

        # get all the context words
        for i in range(index-self.contextSize, index+self.contextSize+1):
            if i == index:
                continue
            elif i < 0 or i >= len(self.tokens):
                context.append(self.wordIndices['<PAD>'])
            else:
                context.append(self.wordIndices[self.tokens[i]])
            y.append(1)
        
        # get negative samples
        for i in range(self.k * 2 * self.contextSize):
            negativeSample = torch.randint(len(self.wordIndices), size=(1,)).item()
            while negativeSample in context:
                negativeSample = torch.randint(len(self.wordIndices), size=(1,)).item()
            context.append(negativeSample)
            y.append(0)

        return torch.tensor(context), torch.tensor(y)