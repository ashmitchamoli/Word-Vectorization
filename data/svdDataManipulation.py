import pandas as pd

trainData = pd.read_csv('./News Classification Dataset/train.csv')
testData = pd.read_csv('./News Classification Dataset/test.csv')

k = min(len(trainData), 500)
trainData = trainData[:k]

trainData.to_csv('./SvdData/train.csv')
testData.to_csv('./SvdData/test.csv')