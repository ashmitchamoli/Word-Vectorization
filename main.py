from word_vectorization.preprocess.Preprocessor import Preprocessor

fileName = './data/News Classification Dataset/train.csv'
preprocessor = Preprocessor()

# print(preprocessor.readData(fileName))
print(preprocessor.getTokens(fileName))
# print(preprocessor.data)
# print(preprocessor.tokens)