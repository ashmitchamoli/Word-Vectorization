from word_vectorization.preprocess.Preprocessor import Preprocessor
from word_vectorization.models.svd.SVD import SvdWordVectorizationModel

trainFileName = './data/SvdData/train.csv'

svdModel = SvdWordVectorizationModel(5, trainFileName, embeddingSize=16)
print(len(svdModel.indexWords))
# print(svdModel.wordIndices)
embeddings = svdModel.train()

for i in svdModel.indexWords:
    print(svdModel.indexWords[i], embeddings[i])

# print(svdModel.indexWords[0])
# print(svdModel.computeCoOccurrenceMatrix()[:1])