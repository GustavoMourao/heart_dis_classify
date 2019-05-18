import numpy as np
from sklearn.model_selection import train_test_split

def get_raw_data():
    '''
    Get raw data: Splits on 70% trainind and 30% test.
    '''
    
    # Call data.
    rawData = np.loadtxt('heart.dat')
    
    # Split format: 70% training / 30% test. random_state=42: Always get same results.
    trainData, testData = train_test_split(rawData, random_state=42, test_size=0.3)

    # Separates in features and target.
    xTrain = trainData[0:trainData.shape[0], 0:trainData.shape[1] - 1]
    yTrain = trainData[0:trainData.shape[0], trainData.shape[1] - 1:trainData.shape[1]]
    xTest  = testData[0:testData.shape[0], 0:testData.shape[1] - 1]
    yTest  = testData[0:testData.shape[0], testData.shape[1] - 1:testData.shape[1]]

    # Change target to binary output. (Proposes to use ROC curve).
    yTrain[yTrain == 1] = 0
    yTrain[yTrain == 2] = 1
    yTest[yTest == 1]   = 0
    yTest[yTest == 2]   = 1

    return [yTrain, xTrain, yTest, xTest]