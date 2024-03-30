from word_vectorization.preprocess.Preprocessor import Preprocessor
from word_vectorization.models.svd.SVD import SvdWordVectorizationModel
from word_vectorization.models.word2vec.Word2Vec import Word2Vec

import numpy as np

# trainFileName = './data/W2vData/train.csv'
trainFileName = './data/SvdData/train.csv'
# trainFileName = './data/News Classification Dataset/train.csv'

word2VecModel = Word2Vec(3, trainFileName, embeddingSize=100, k=3)

word2VecModel.train(epochs=1, lr=0.005, batchSize=2**12, verbose=True, retrain=False)

# svdModel = SvdWordVectorizationModel(5, trainFileName, embeddingSize=16)
# print(len(svdModel.indexWords))
# # print(svdModel.wordIndices)
# embeddings = svdModel.train()

# for i in svdModel.indexWords:
#     print(svdModel.indexWords[i], embeddings[i])

# print(svdModel.indexWords[0])
# print(svdModel.computeCoOccurrenceMatrix()[:1])

# for i in word2VecModel.indexWords:
#     print(word2VecModel.indexWords[i], word2VecModel.embeddings[i])

# embedding1 = word2VecModel.getEmbedding('man')
# embedding1 /= np.linalg.norm(embedding1)
# embedding2 = word2VecModel.getEmbedding('woman') + word2VecModel.getEmbedding('king') - word2VecModel.getEmbedding('queen')
# embedding2 /= np.linalg.norm(embedding2)

# print(np.linalg.norm(embedding1-embedding2))

embedding1 = word2VecModel.getEmbedding('job')
embedding1 /= np.linalg.norm(embedding1)

embedding2 = word2VecModel.getEmbedding('woman')
embedding2 /= np.linalg.norm(embedding2)

print(np.dot(embedding1, embedding2))

print([key for key in word2VecModel.getClosestWordEmbeddings(word2VecModel.getEmbedding('job'), 5)])
print([key for key in word2VecModel.getClosestWordEmbeddings(word2VecModel.getEmbedding('man'), 5)])
print([key for key in word2VecModel.getClosestWordEmbeddings(word2VecModel.getEmbedding('woman'), 5)])
print([key for key in word2VecModel.getClosestWordEmbeddings(word2VecModel.getEmbedding('bank'), 5)])
