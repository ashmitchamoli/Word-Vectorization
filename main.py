from word_vectorization.preprocess.Preprocessor import Preprocessor
from word_vectorization.models.svd.SVD import SvdWordVectorizationModel

trainFileName = './data/News Classification Dataset/train.csv'

svdModel = SvdWordVectorizationModel(5, trainFileName, embeddingSize=48)
print(svdModel.indexWords)
# print(svdModel.wordIndices)
# embeddings = svdModel.train()

# for i in svdModel.indexWords:
#     print(svdModel.indexWords[i], embeddings[i])

# print(svdModel.indexWords[0])
# print(svdModel.computeCoOccurrenceMatrix()[:1])