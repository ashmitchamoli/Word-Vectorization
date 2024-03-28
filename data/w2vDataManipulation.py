import pandas as pd

trainData = pd.read_csv('./News Classification Dataset/train.csv')
testData = pd.read_csv('./News Classification Dataset/test.csv')

k = min(len(trainData), 15000)
trainData = trainData[:k]

trainData.to_csv('./W2vData/train.csv')
testData.to_csv('./W2vData/test.csv')