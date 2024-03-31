from word_vectorization.classification.LstmClassifier import SentenceClassifier
from word_vectorization.models.svd.SVD import SvdWordVectorizationModel

trainFileName = './data/News Classification Dataset/train.csv'

svdModel = SvdWordVectorizationModel(2, trainFileName, embeddingSize=150)
svdEmbeddings = svdModel.train()

svdClassifier = SentenceClassifier(trainFileName,
                                   svdModel,
                                   hiddenSize=256,
                                   numLayers=3,
                                   bidirectional=True,
                                   hiddenLayers=[128, 64],
                                   activation='tanh')
svdClassifier.train(epochs=15,
                    lr=0.005,
                    batchSize=32)
