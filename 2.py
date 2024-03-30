import word_vectorization.models.svd.SVD as SVD

trainFileName = './data/News Classification Dataset/train.csv'

svdModel = SVD.SvdWordVectorizationModel(3, trainFileName, embeddingSize=300)
svdEmbeddings = svdModel.train(retrain=True)