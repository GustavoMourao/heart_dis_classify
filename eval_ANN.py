from sklearn.neural_network import MLPClassifier
from sklearn.metrics import average_precision_score
from sklearn.externals import joblib
import split_set_data
import numpy as np

def train_ANN_models(yTrain, xTrain, yTest, xTest):
    '''
    Obtain best ANN models from comparing the highest precision.
    It is evaluate the modification through change the number
    of neurons on first hidden layer. 
    Select model with precision highrer than 85%.
    Source:
        Graber, Mark L. "The incidence of diagnostic error in medicine." 
        BMJ Qual Saf 22.Suppl 2 (2013): ii21-ii27.
    '''

    # Simulation range parameters (first hidden layer).
    parameters = np.linspace(1,10000,100000)
    
    for i in range(len(parameters)):

        classifier = MLPClassifier(hidden_layer_sizes=(i+1, 500, 500), max_iter=100, alpha=1e-4,
                    solver='lbfgs', verbose=1, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

        classifier.fit(xTrain, yTrain)
        predicted = classifier.predict_proba(xTest)

        # Calculates precision.
        avrPrec = average_precision_score(yTest[:,:], predicted[:,1])

        # Filter condition.
        if (100 * avrPrec > 85):
            joblib.dump(classifier, 'ANN_nlayer_'+ str(i) +'.pkl')


if __name__=='__main__':
    '''
    Main function
    '''
    # Get split data.
    yTrain, xTrain, yTest, xTest = split_set_data.get_raw_data()

    ## Preprocessing data: FAIL!
    #xTrain = preprocessing.scale(xTrain)
    #xTest = preprocessing.scale(xTest)

    # Save best ANN models. 
    train_ANN_models(yTrain, xTrain, yTest, xTest)
