from word_vectorization.models.svd.SVD import SvdWordVectorizationModel

trainFileName = './data/News Classification Dataset/train.csv'

svdModel = SvdWordVectorizationModel(2, trainFileName, embeddingSize=300)
svdEmbeddings = svdModel.train()