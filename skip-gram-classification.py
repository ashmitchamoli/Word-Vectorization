from word_vectorization.models.word2vec.Word2Vec import Word2Vec
from word_vectorization.classification.LstmClassifier import SentenceClassifier

trainFileName = './data/News Classification Dataset/train.csv'

word2VecModel = Word2Vec(3, trainFileName, embeddingSize=300, k=3)
word2VecEmbeddings = word2VecModel.train(epochs=3, lr=0.005, batchSize=2**12, verbose=True)

word2VecClassifier = SentenceClassifier(trainFileName,
                                        word2VecModel,
                                        hiddenSize=256,
                                        numLayers=3,
                                        bidirectional=True,
                                        hiddenLayers=[128, 64],
                                        activation='tanh')
word2VecClassifier.train(epochs=15, lr=0.005, batchSize=32)
