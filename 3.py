import word_vectorization.models.word2vec.Word2Vec as Word2Vec

trainFileName = './data/News Classification Dataset/train.csv'

word2VecModel = Word2Vec.Word2Vec(4, trainFileName, embeddingSize=300, k=3)
word2VecEmbeddings = word2VecModel.train(epochs=10, lr=0.005, batchSize=2**12, verbose=True)