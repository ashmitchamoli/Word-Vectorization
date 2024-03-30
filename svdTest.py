import numpy as np

from word_vectorization.models.svd.SVD import SvdWordVectorizationModel

# trainFilePath = "./data/News Classification Dataset/train.csv"
trainFilePath = "./data/SvdData/train.csv"

svdModel = SvdWordVectorizationModel(5, trainFilePath, embeddingSize=200)
svdModel.train(retrain=False)

# print(svdModel.embeddings)
# print(type(svdModel.embeddings))

# print(svdModel.getEmbedding('man'))
# print(svdModel.getEmbedding('woman') + svdModel.getEmbedding('king') - svdModel.getEmbedding('queen'))

# np.linalg.norm(svdModel.getEmbedding('man') - svdModel.getEmbedding('woman') - svdModel.getEmbedding('king') + svdModel.getEmbedding('queen'))

embedding1 = svdModel.getEmbedding('job')
embedding1 /= np.linalg.norm(embedding1)

embedding2 = svdModel.getEmbedding('woman')
embedding2 /= np.linalg.norm(embedding2)

print(np.dot(embedding1, embedding2))